"""Workload simulation and context sweep phases.

Naming convention
-----------------
``phase_workload_sim`` and ``phase_context_sweep`` use different naming
styles on purpose:

* **phase_workload_sim** -- "sim" abbreviates "simulation" because this
  phase *simulates* production workloads (hot-cache TTFT, concurrent users).
* **phase_context_sweep** -- "sweep" is literal: the phase *sweeps* across
  a range of context sizes to find the maximum that fits in VRAM.

Both live in this module because they share the same dependency set
(boot_server, measure_perf) and neither uses Optuna trials.
"""

from __future__ import annotations

import logging

import requests

from ..constants import (
    CONCURRENT_REQUEST_TIMEOUT,
    HAS_AIOHTTP,
    HTTP_OK,
    NIAH_REQUEST_TIMEOUT,
)
from ..engine import (
    boot_server_with_jinja_recovery,
    kill_server,
    start_server,
    wait_for_server,
)
from ..measurement import compute_score, measure_concurrent_load, measure_perf_adaptive
from ..result_types import EngineConfig, PhaseReturnDict
from ..search import save_phase_results
from ..state import AppContext

logger = logging.getLogger(__name__)

# TTFT verdict thresholds (milliseconds)
_TTFT_EXCELLENT = 200
_TTFT_GOOD = 500
_TTFT_SLOW = 1000

__all__ = ["phase_workload_sim", "phase_context_sweep"]


def phase_workload_sim(  # noqa: C901, PLR0912, PLR0915
    ctx: AppContext, base_config: EngineConfig | None = None
) -> PhaseReturnDict | None:
    """Phase 6: Workload simulation -- hot-cache TTFT + concurrent load test.

    No Optuna trials. Measures production-readiness metrics:
    1. Hot-cache TTFT: Time to first token when 95% of prompt is already cached
    2. Concurrent load: System throughput with N simultaneous users
    """
    label = "Workload Simulation"

    logger.info("=" * 60)
    logger.info("  %s", label)
    logger.info("=" * 60)
    logger.info("")

    if base_config is None:
        base_config = dict(ctx.naked_engine)

    # Start server with full optimized config + cache-reuse enabled
    config = {**base_config, "cache_reuse": 256}
    logger.info("Starting server with cache-reuse enabled...")
    kill_server(ctx)
    proc, status = boot_server_with_jinja_recovery(ctx, config)
    if status != "ok":
        logger.warning("Server failed to start for workload simulation")
        return None

    load_ms = proc.load_time_ms
    if load_ms:
        logger.info("Model loaded in %.0fms", load_ms)

    results = {}

    # 1. Hot-cache TTFT test
    logger.info("[HOT CACHE] Measuring TTFT on cached prompt...")
    system_prompt = (
        "You are a helpful AI assistant specialized in software engineering. "
        "You write clean, efficient, well-documented code. You follow best practices "
        "for error handling, testing, and performance optimization. You communicate "
        "clearly and concisely, focusing on practical solutions."
    )
    # First request: prime the cache (cold)
    try:
        r = ctx.http.post(
            f"{ctx.server_url}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Explain the difference between a mutex and a semaphore."
                        ),
                    },
                ],
                "max_tokens": 100,
                "temperature": 0.0,
            },
            timeout=CONCURRENT_REQUEST_TIMEOUT,
        )
        if r.status_code == HTTP_OK:
            timings = r.json().get("timings", {})
            cold_ttft = timings.get("prompt_ms", 0)
            logger.info("Cold TTFT: %.0fms (prompt processing)", cold_ttft)
            results["cold_ttft_ms"] = round(cold_ttft, 1)
    except (requests.RequestException, KeyError, ValueError, OSError) as e:
        logger.warning("Cold request failed: %s", e)

    # Second request: same system prompt, different user query (hot cache)
    hot_ttfts = []
    for i, query in enumerate(
        [
            "How do I implement a binary search tree in Python?",
            "What are the SOLID principles? Give a brief example of each.",
            "Explain the CAP theorem and its practical implications.",
        ]
    ):
        try:
            r = ctx.http.post(
                f"{ctx.server_url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    "max_tokens": 100,
                    "temperature": 0.0,
                },
                timeout=CONCURRENT_REQUEST_TIMEOUT,
            )
            if r.status_code == HTTP_OK:
                timings = r.json().get("timings", {})
                hot_ttft = timings.get("prompt_ms", 0)
                hot_ttfts.append(hot_ttft)
                logger.info("Hot TTFT #%s: %.0fms", i + 1, hot_ttft)
        except (requests.RequestException, KeyError, ValueError, OSError) as e:
            logger.warning("Hot TTFT measurement request failed: %s", e)

    if hot_ttfts:
        avg_hot = sum(hot_ttfts) / len(hot_ttfts)
        results["hot_ttft_avg_ms"] = round(avg_hot, 1)
        results["hot_ttft_min_ms"] = round(min(hot_ttfts), 1)
        results["hot_ttft_max_ms"] = round(max(hot_ttfts), 1)
        logger.info(
            "Hot TTFT avg: %.0fms (min=%.0f, max=%.0f)",
            avg_hot,
            min(hot_ttfts),
            max(hot_ttfts),
        )
        if results.get("cold_ttft_ms"):
            speedup = results["cold_ttft_ms"] / avg_hot if avg_hot > 0 else 0
            logger.info("Cache speedup: %.1fx", speedup)
            results["cache_speedup"] = round(speedup, 2)

    # 2. Concurrent load test
    n_users = ctx.config.get("simulate_users", 0)
    if n_users > 0 and HAS_AIOHTTP:
        logger.info("[LOAD TEST] Simulating %s concurrent users...", n_users)
        load_results = measure_concurrent_load(ctx, n_users=n_users, n_predict=50)
        if load_results:
            results["concurrent"] = load_results
            logger.info("Total throughput: %.1f t/s", load_results.concurrent_total_tps)
            logger.info("Per-user avg:     %.1f t/s", load_results.concurrent_avg_tps)
            logger.info("Avg latency:      %.0fms", load_results.concurrent_avg_wall_ms)
    elif n_users > 0:
        logger.info("[LOAD TEST] Skipped \u2014 aiohttp not available")

    kill_server(ctx)

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Workload Simulation — RESULTS")
    logger.info("=" * 60)
    logger.info("")
    if results.get("hot_ttft_avg_ms"):
        verdict = (
            "EXCELLENT"
            if results["hot_ttft_avg_ms"] < _TTFT_EXCELLENT
            else "GOOD"
            if results["hot_ttft_avg_ms"] < _TTFT_GOOD
            else "SLOW"
            if results["hot_ttft_avg_ms"] < _TTFT_SLOW
            else "POOR"
        )
        logger.info(
            "  Hot-cache TTFT: %.0fms [%s]", results["hot_ttft_avg_ms"], verdict
        )
    if results.get("concurrent"):
        logger.info(
            "  Concurrent (%s users): %.1f t/s total",
            n_users,
            results["concurrent"].concurrent_total_tps,
        )

    save_phase_results(ctx, "workload_sim", results)
    return PhaseReturnDict(best_params=results, phase_name="workload_sim")


def phase_context_sweep(
    ctx: AppContext,
    base_config: dict | None = None,
    contexts: list[int] | None = None,
    n_runs: int = 3,
) -> PhaseReturnDict | None:
    """Test which context sizes the model can handle and measure TPS for each.

    Returns:
        PhaseReturnDict | None: Dict mapping context sizes to results,
        or None if the phase was skipped.
    """
    if contexts is None:
        contexts = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    if base_config is None:
        base_config = dict(ctx.naked_engine)

    logger.info("=" * 60)
    logger.info("CONTEXT SIZE SWEEP")
    logger.info("Testing: %s", contexts)
    logger.info("=" * 60)

    # Build filler sentence for KV cache pressure testing
    filler_sentence = "The quick brown fox jumps over the lazy dog. "  # ~10 tokens

    results = {}
    for ctx_size in contexts:
        # Build a prompt that fills ~75% of the context window
        # so KV cache is under real pressure
        fill_tokens = int(ctx_size * 0.75)
        repeats = max(1, fill_tokens // 10)
        filler = filler_sentence * repeats
        fill_prompt = (
            f"Summarize the following text in one sentence:\n{filler}\nSummary:"
        )
        logger.info(
            "Testing context=%s (filling ~%s tokens)...",
            "{:,}".format(ctx_size),
            "{:,}".format(fill_tokens),
        )

        engine_config = {**base_config, "context": ctx_size}
        # Clamp batch params to context -- llama-server asserts batch_size <= context
        # Also cap batch_size at 8192 to avoid excessive memory usage
        max_batch_size = 8192
        if engine_config.get("batch_size", 512) > ctx_size:
            engine_config["batch_size"] = min(ctx_size, max_batch_size)
        if engine_config.get("ubatch_size", 128) > engine_config.get(
            "batch_size", ctx_size
        ):
            engine_config["ubatch_size"] = engine_config["batch_size"]
        # Clamp speculation draft tokens -- draft >= context crashes llama.cpp
        if engine_config.get("draft_max", 0) >= ctx_size:
            engine_config["draft_max"] = max(1, ctx_size // 4)
            # Safety: clamp draft_min so it never exceeds draft_max (C++ assert crash)
            if engine_config.get("draft_min", 0) >= engine_config["draft_max"]:
                engine_config["draft_min"] = max(0, engine_config["draft_max"] - 1)
        kill_server(ctx)
        proc = start_server(ctx, engine_config)
        if wait_for_server(ctx, proc=proc, timeout=NIAH_REQUEST_TIMEOUT) != "ok":
            logger.info("context=%s: DOES NOT FIT", "{:,}".format(ctx_size))
            kill_server(ctx)
            results[ctx_size] = {
                "fits": False,
                "tps": 0.0,
                "score": 0.0,
                "prompt_tps": 0.0,
            }
            continue
        perf, _ = measure_perf_adaptive(ctx, runs=n_runs, prompt=fill_prompt)
        score = compute_score(perf)
        pp_tps = perf.prompt_tps or 0.0
        results[ctx_size] = {
            "fits": True,
            "tps": round(perf.tps, 2),
            "score": round(score, 2),
            "prompt_tps": round(pp_tps, 2),
        }
        logger.info(
            "context=%s: %.1f t/s gen | %.1f t/s PP (score: %.1f)",
            "{:,}".format(ctx_size),
            perf.tps,
            pp_tps,
            score,
        )
        kill_server(ctx)

    logger.info("Context Sweep Results:")
    logger.info("%s  %s  %s  %s  %s", "Context", "Fits", "Gen t/s", "PP t/s", "Score")
    logger.info("" + "-" * 50)
    for ctx_size in contexts:
        r = results[ctx_size]
        fits_str = "yes" if r["fits"] else "NO"
        tps_str = f"{r['tps']:.1f}" if r["fits"] else "-"
        pp_str = f"{r.get('prompt_tps', 0):.1f}" if r["fits"] else "-"
        score_str = f"{r['score']:.1f}" if r["fits"] else "-"
        logger.info(
            "%10s  %5s  %8s  %8s  %8s",
            f"{ctx_size:,}",
            fits_str,
            tps_str,
            pp_str,
            score_str,
        )

    save_phase_results(
        ctx,
        "context_sweep",
        {"contexts": {str(k): v for k, v in results.items()}, "n_runs": n_runs},
    )
    return PhaseReturnDict(best_params=results, phase_name="context_sweep")
