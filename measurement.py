"""Performance measurement, scoring, and concurrent load testing."""
import asyncio
import logging
import math
import time

import requests

from .state import ctx, _config

logger = logging.getLogger(__name__)
from .constants import (
    SCORE_VERSION, SCORE_TTFT_BASELINE, SCORE_PP_BASELINE,
    ADAPTIVE_THRESHOLD, CV_TARGET, CV_MIN_RUNS, CV_MAX_RUNS,
    TPS_TEST_PROMPT,
)

try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

import optuna


# ============================================================
# Large-prompt benchmarking — stress-test TPS under high context load
# ============================================================

def _get_actual_ctx():
    """Probe the server's actual n_ctx via /props endpoint.

    Falls back to sending an oversized prompt and parsing the error to extract
    the real context limit. Returns int or None if detection fails.
    """
    # Try /props first (llama-server >= b3000)
    try:
        r = ctx.http.get(f"{ctx.server_url}/props", timeout=5)
        if r.status_code == 200:
            data = r.json()
            n_ctx = data.get("n_ctx") or data.get("default_generation_settings", {}).get("n_ctx")
            if n_ctx and int(n_ctx) > 0:
                return int(n_ctx)
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.debug("Context detection via /props failed: %s", e)

    # Fallback: send a huge prompt, parse error for context limit
    try:
        huge_prompt = "x " * 200000  # ~200k tokens, guaranteed to exceed any context
        r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": huge_prompt}],
            "max_tokens": 1,
        }, timeout=10)
        if r.status_code != 200:
            err = r.text
            import re
            m = re.search(r'(?:context.*?|n_ctx\s*[=:]\s*)(\d{3,})', err)
            if m:
                return int(m.group(1))
    except (requests.RequestException, ValueError) as e:
        logger.debug("Context detection via error parsing failed: %s", e)

    return None


def _build_large_prompt(max_ctx):
    """Build a prompt that fills ~90% of the server's context window.

    Uses repeated coding questions to create realistic token density.
    Returns the prompt string.
    """
    # Rough estimate: 1 token ≈ 4 chars for English text
    target_chars = int(max_ctx * 0.90 * 4)

    # Base block of coding questions (~500 chars each)
    blocks = [
        "Write a Python function that implements merge sort with detailed comments explaining each step.",
        "Implement a binary tree with insert, delete, and search operations in Python with type hints.",
        "Write a Python class implementing an LRU cache with O(1) get and put operations.",
        "Implement Dijkstra's shortest path algorithm in Python using a priority queue.",
        "Write a Python function to find all permutations of a string using backtracking.",
        "Implement a thread-safe producer-consumer queue in Python using threading primitives.",
        "Write a Python decorator that implements memoization with an LRU eviction policy.",
        "Implement a trie data structure in Python with insert, search, and prefix matching.",
    ]

    prompt_parts = []
    current_chars = 0
    block_idx = 0

    while current_chars < target_chars:
        block = blocks[block_idx % len(blocks)]
        # Add variation so repeated blocks aren't identical (avoids KV cache tricks)
        block_with_var = f"[Question {block_idx + 1}] {block} Use approach variant #{block_idx // len(blocks) + 1}."
        prompt_parts.append(block_with_var)
        current_chars += len(block_with_var)
        block_idx += 1

    return "\n\n".join(prompt_parts)


def _measure_perf_large(n_predict=50):
    """Benchmark TPS with a large prompt filling ~90% of context.

    Returns dict with 'large_tps', 'large_prompt_tps', 'large_ttft' keys,
    or None if the measurement fails (e.g., server doesn't support /props).
    """
    actual_ctx = _get_actual_ctx()
    if not actual_ctx or actual_ctx < 2048:
        return None

    large_prompt = _build_large_prompt(actual_ctx)

    try:
        r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": large_prompt}],
            "max_tokens": n_predict,
            "temperature": 0.4,
        }, timeout=300)  # large prompts can take a while to process

        if r.status_code == 200:
            data = r.json()
            timings = data.get("timings", {})
            tps = timings.get("predicted_per_second", 0)
            if tps > 0:
                return {
                    "large_tps": tps,
                    "large_prompt_tps": timings.get("prompt_per_second", 0),
                    "large_ttft": timings.get("prompt_ms", 0),
                }
    except Exception as e:
        print(f"  [!] Large-prompt measurement failed: {e}")

    return None


def compute_score(perf, vram_used_mb=None, vram_total_mb=None):
    """Dual-mode composite score from performance metrics.

    Full mode (when large-prompt data is present in perf dict):
      score = gen_tps*0.35 + large_tps*0.25 + pp*0.15 + ttft_factor*0.15 + vram_factor*0.10

    Lightweight mode (quick filter pass, no large-prompt data):
      score = gen_tps * (0.60 + 0.25 * pp/pp_baseline + 0.15 * ttft_baseline/ttft)

    The full formula rewards configs that maintain TPS under heavy context load.
    The lightweight formula is used during the initial quick-reject pass in adaptive
    measurement, where running a large-prompt benchmark would be too expensive.
    """
    # Auto-extract VRAM from perf dict if not explicitly passed
    if vram_used_mb is None:
        vram_used_mb = perf.get("vram_used_mb")
    if vram_total_mb is None:
        vram_total_mb = perf.get("vram_total_mb")
    gen_tps = perf["tps"]
    prompt_tps = perf["prompt_tps"]
    ttft = perf["ttft"]

    if gen_tps <= 0:
        return 0.0

    # Sanitize: clamp to sane ranges to prevent inf/NaN from bad server responses.
    prompt_tps = min(50000.0, max(0.0, prompt_tps)) if math.isfinite(prompt_tps) else 50000.0
    ttft = max(1.0, ttft) if math.isfinite(ttft) else 1.0

    # Check if we have large-prompt data — triggers full formula
    large_tps = perf.get("large_tps")

    if large_tps and large_tps > 0:
        # Full mode: weighted sum of all signal dimensions
        pp_norm = min(prompt_tps / SCORE_PP_BASELINE, 3.0)  # cap at 3x baseline
        ttft_norm = min(SCORE_TTFT_BASELINE / ttft, 3.0)    # cap at 3x baseline

        score = (gen_tps * 0.35 +
                 large_tps * 0.25 +
                 pp_norm * gen_tps * 0.15 +       # pp as multiplier on gen_tps scale
                 ttft_norm * gen_tps * 0.15)       # ttft as multiplier on gen_tps scale

        # VRAM efficiency: 10% weight in full mode
        if vram_used_mb is not None and vram_total_mb is not None and vram_total_mb > 0:
            vram_efficiency = 1.0 - (vram_used_mb / vram_total_mb)
            vram_bonus = max(0.0, min(1.0, vram_efficiency))  # 0..1
            score += vram_bonus * gen_tps * 0.10
        else:
            # No VRAM data — redistribute the 10% weight to gen_tps
            score += gen_tps * 0.10
    else:
        # Lightweight mode: quick filter formula (no large-prompt overhead)
        pp_factor = min((prompt_tps / SCORE_PP_BASELINE), 3.0) if prompt_tps > 0 else 0.0
        ttft_factor = min((SCORE_TTFT_BASELINE / ttft), 3.0) if ttft > 0 else 0.0

        score = gen_tps * (0.60 + 0.25 * pp_factor + 0.15 * ttft_factor)

        # VRAM efficiency bonus: up to 5% boost for leaving headroom
        if vram_used_mb is not None and vram_total_mb is not None and vram_total_mb > 0:
            utilization = vram_used_mb / vram_total_mb
            headroom_bonus = min(0.05, max(0.0, (1.0 - utilization) * 0.10))
            score *= (1.0 + headroom_bonus)

    # Concurrent load bonus: if --simulate-users data is present, blend it in
    concurrent_tps = perf.get("concurrent_total_tps")
    if concurrent_tps and concurrent_tps > 0:
        # Concurrent throughput bonus: up to 15% weight
        # Normalized against single-user TPS * n_users (perfect scaling = 1.0)
        n_users = perf.get("concurrent_users", 4)
        scaling_efficiency = concurrent_tps / (gen_tps * n_users) if gen_tps > 0 else 0
        scaling_efficiency = min(1.0, max(0.0, scaling_efficiency))
        # Blend: high scaling efficiency means the config handles concurrency well
        score *= (0.85 + 0.15 * scaling_efficiency)

    return score if math.isfinite(score) else 0.0


def compute_pareto_objectives(perf, quality_factor=1.0):
    """Extract multi-objective values for Pareto optimization.

    Returns tuple of (tps, neg_vram, quality_factor):
      - tps: generation tokens/sec (maximize)
      - neg_vram: negative VRAM usage in MB (maximize = less VRAM)
      - quality_factor: 0.0-1.0 quality gate (maximize)

    The negative VRAM trick converts "minimize VRAM" into "maximize -VRAM"
    so all objectives use the same direction.
    """
    tps = perf.get("tps", 0.0)
    vram_mb = perf.get("vram_used_mb")
    neg_vram = -vram_mb if vram_mb is not None else -99999.0  # penalize unknown VRAM
    return (tps, neg_vram, quality_factor)


def extract_pareto_front(study):
    """Extract Pareto-optimal trials from a multi-objective study.

    Returns list of trials sorted by TPS (first objective, descending).
    """
    try:
        pareto_trials = study.best_trials  # NSGA-II provides this
    except (RuntimeError, ValueError):
        return []
    # Sort by TPS descending
    return sorted(pareto_trials, key=lambda t: t.values[0], reverse=True)


def print_pareto_front(pareto_trials):
    """Print the Pareto front as a table for the user to pick from."""
    if not pareto_trials:
        print("  [!] No Pareto-optimal configs found.")
        return

    print(f"\n  {'':>4}  {'TPS':>8}  {'VRAM MB':>8}  {'Quality':>8}  Key Params")
    print(f"  {'':>4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*30}")
    for i, t in enumerate(pareto_trials):
        tps = t.values[0]
        vram = -t.values[1]  # un-negate
        qf = t.values[2]
        # Extract a few key params for display
        p = t.params
        short = []
        if "threads" in p: short.append(f"t={p['threads']}")
        if "kv_cache_type" in p: short.append(f"kv={p['kv_cache_type']}")
        if "batch_size" in p: short.append(f"b={p['batch_size']}")
        if "flash_attn" in p: short.append(f"fa={p['flash_attn']}")
        if "draft_max" in p: short.append(f"draft={p['draft_max']}")
        params_str = " ".join(short) if short else str(p)[:50]
        print(f"  [{i+1:>2}]  {tps:8.1f}  {vram:8.0f}  {qf:8.2f}  {params_str}")


def get_best_trial(study):
    """Get the best trial from a study, handling both single and multi-objective modes.

    In multi-objective (Pareto) mode, returns the trial with the highest TPS
    from the Pareto front (objective 0 = TPS). Falls back to first trial if needed.
    """
    is_pareto = _config.get("pareto", False)
    if is_pareto:
        pareto = extract_pareto_front(study)
        if pareto:
            return pareto[0]  # highest TPS on the front
        # Fallback: just pick the trial with best first objective
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completed:
            return max(completed, key=lambda t: t.values[0] if t.values else 0)
        return study.trials[0]
    return study.best_trial


def get_best_value(study):
    """Get the best value from a study. In Pareto mode, returns the best TPS."""
    is_pareto = _config.get("pareto", False)
    if is_pareto:
        best = get_best_trial(study)
        return best.values[0] if best.values else 0.0
    try:
        return study.best_value
    except ValueError:
        return None


def measure_perf_adaptive(best_score, n_predict=50, spec_params=None):
    """Adaptive measurement with CV-based stability.

    1. Quick first run as a gate.
    2. If competitive (>= 70% of best), run CV-stabilized measurement.
    3. Runs 3-5 times until CV <= 5% (or max reached). Discards warmup run.
    4. Promoted configs get a large-prompt benchmark too.

    Returns (perf_dict, was_promoted) — was_promoted is True if stable runs were done.
    """
    # Quick first run (warmup — always discarded from final stats)
    first = _measure_perf_once(n_predict=n_predict, spec_params=spec_params)
    if first is None:
        return {"tps": 0.0, "ttft": 0.0, "prompt_tps": 0.0, "total_ms": 0.0}, False

    quick_score = compute_score(first)

    # If clearly bad, don't waste time on more runs
    if best_score > 0 and quick_score < best_score * ADAPTIVE_THRESHOLD:
        return first, False

    # Competitive — CV-stabilized measurement (first run is warmup, start fresh)
    samples = []
    for i in range(CV_MAX_RUNS):
        s = _measure_perf_once(n_predict=n_predict, spec_params=spec_params)
        if s:
            samples.append(s)

        # Check stability after minimum runs
        if len(samples) >= CV_MIN_RUNS:
            tps_values = [x["tps"] for x in samples]
            mean_tps = sum(tps_values) / len(tps_values)
            if mean_tps > 0:
                std_tps = (sum((v - mean_tps) ** 2 for v in tps_values) / len(tps_values)) ** 0.5
                cv = std_tps / mean_tps
                if cv <= CV_TARGET:
                    break  # stable enough

    if not samples:
        return first, False

    result = _aggregate_samples(samples)

    # Store measurement variance for noise-aware GP (reuse CV from loop)
    if len(samples) >= 2:
        tps_values = [x["tps"] for x in samples]
        mean_tps = sum(tps_values) / len(tps_values)
        result["tps_std"] = (sum((v - mean_tps) ** 2 for v in tps_values) / len(tps_values)) ** 0.5
        result["tps_cv"] = result["tps_std"] / mean_tps if mean_tps > 0 else 0.0
        result["n_runs"] = len(samples)

    # Promoted config — run large-prompt benchmark to get full-mode scoring data
    large = _measure_perf_large(n_predict=n_predict)
    if large:
        result.update(large)  # adds large_tps, large_prompt_tps, large_ttft

    # Concurrent load test if --simulate-users is set
    n_users = _config.get("simulate_users", 0)
    if n_users > 0:
        load = measure_concurrent_load(n_users=n_users, n_predict=n_predict)
        if load:
            result.update(load)

    return result, True


def measure_perf_quick_gate(n_predict=5):
    """Ultra-fast 5-token generation for multi-fidelity pruning.

    Measures only TTFT and prompt processing speed with minimal token generation.
    Used as the first step of successive halving — if this is in the bottom 50%
    of known configs, the trial gets pruned before the full benchmark.

    Returns dict with tps, ttft, prompt_tps or None on failure.
    """
    payload = {
        "messages": [{"role": "user", "content": TPS_TEST_PROMPT}],
        "max_tokens": n_predict,
        "temperature": 0.4,
    }
    try:
        r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json=payload, timeout=30)
        if r.status_code == 200:
            data = r.json()
            timings = data.get("timings", {})
            prompt_tps = timings.get("prompt_per_second", 0)
            ttft = timings.get("prompt_ms", 0)
            tps = timings.get("predicted_per_second", 0)
            if prompt_tps > 0 or ttft > 0:
                return {
                    "tps": tps,
                    "ttft": ttft,
                    "prompt_tps": prompt_tps,
                    "total_ms": ttft + timings.get("predicted_ms", 0),
                    "gate_score": prompt_tps * 0.6 + (1000.0 / max(1, ttft)) * 0.4,
                }
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.debug("Quick gate measurement failed: %s", e)
    return None


def _measure_perf_once(n_predict=50, spec_params=None, prompt=None):
    """Single measurement run. Returns dict of raw values or None on failure.

    Uses /v1/chat/completions so the server applies its --chat-template
    automatically — works with any model, not just ChatML.
    """
    from .hardware import _get_vram_used_mb

    payload = {
        "messages": [{"role": "user", "content": prompt or TPS_TEST_PROMPT}],
        "max_tokens": n_predict,
        "temperature": 0.4,
    }
    if spec_params:
        payload["speculative"] = spec_params
    try:
        # Scale timeout for large prompts — 60s base + 1s per 500 prompt tokens
        prompt_len = len((prompt or TPS_TEST_PROMPT))
        est_tokens = prompt_len // 4
        timeout = max(60, 60 + est_tokens // 500)
        r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json=payload, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            timings = data.get("timings", {})
            tps = timings.get("predicted_per_second", 0)
            if tps > 0:
                result = {
                    "tps": tps,
                    "ttft": timings.get("prompt_ms", 0),
                    "prompt_tps": timings.get("prompt_per_second", 0),
                    "total_ms": timings.get("prompt_ms", 0) + timings.get("predicted_ms", 0),
                }
                # Capture VRAM usage (uses cached total from ctx, avoids full GPU re-scan)
                if ctx.vram_total_mb:
                    vram_used = _get_vram_used_mb()
                    if vram_used is not None:
                        result["vram_used_mb"] = vram_used
                        result["vram_total_mb"] = ctx.vram_total_mb
                return result
    except Exception as e:
        print(f"  [!] Request failed: {e}")
    return None


def _aggregate_samples(samples):
    """Given a list of raw measurement dicts, return the median run by composite score.

    Sorting by composite score and returning the middle dict avoids synthesizing
    an impossible combination of fast TTFT from one run and fast TPS from another.
    """
    if not samples:
        return {"tps": 0.0, "ttft": 0.0, "prompt_tps": 0.0, "total_ms": 0.0}
    if len(samples) == 1:
        return samples[0]
    # Sort by composite score, return the middle (median) run as a whole
    ranked = sorted(samples, key=lambda s: compute_score(s))
    return ranked[len(ranked) // 2]


def measure_perf(n_predict=50, spec_params=None, runs=3, prompt=None):
    """Send test prompt and return performance metrics (median of N runs)."""
    samples = []
    for _ in range(runs):
        s = _measure_perf_once(n_predict=n_predict, spec_params=spec_params, prompt=prompt)
        if s:
            samples.append(s)
    return _aggregate_samples(samples)


def measure_concurrent_load(n_users=4, n_predict=50):
    """Concurrent load test: send N simultaneous requests and measure system throughput.

    Scores the config on:
      - Total system throughput (aggregate tokens/sec across all users)
      - Queue latency (time from request sent to first token received per user)
      - Crash resilience (did all requests succeed?)

    Requires aiohttp. Returns dict with aggregate metrics or None on failure.
    """
    if not _HAS_AIOHTTP:
        # Fall back to sequential if aiohttp not available
        return None

    prompts = [
        "Explain the concept of entropy in information theory.",
        "Write a Python function that implements binary search on a sorted array.",
        "What are the key differences between TCP and UDP protocols?",
        "Describe the process of photosynthesis in plants.",
        "Explain how a hash table works and its time complexity.",
        "What is the difference between a stack and a queue?",
        "Describe the CAP theorem in distributed systems.",
        "Explain the concept of polymorphism in OOP.",
    ]

    async def _single_request(session, prompt, user_id):
        payload = {
            "model": "test",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict,
            "temperature": 0.4,
        }
        start_time = time.time()
        try:
            async with session.post(
                f"{ctx.server_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    elapsed = time.time() - start_time
                    timings = data.get("timings", {})
                    return {
                        "user_id": user_id,
                        "tps": timings.get("predicted_per_second", 0),
                        "ttft": timings.get("prompt_ms", 0),
                        "prompt_tps": timings.get("prompt_per_second", 0),
                        "wall_time": elapsed * 1000,
                        "success": True,
                    }
                return {"user_id": user_id, "success": False, "error": f"status {resp.status}"}
        except Exception as e:
            return {"user_id": user_id, "success": False, "error": str(e)}

    async def _run_load_test():
        async with aiohttp.ClientSession() as session:
            tasks = []
            for i in range(n_users):
                prompt = prompts[i % len(prompts)]
                tasks.append(_single_request(session, prompt, i))
            return await asyncio.gather(*tasks)

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                results = pool.submit(asyncio.run, _run_load_test()).result()
        else:
            results = asyncio.run(_run_load_test())
    except (RuntimeError, OSError) as e:
        logger.debug("Concurrent load test failed: %s", e)
        return None

    successful = [r for r in results if r.get("success")]
    if not successful:
        return None

    # Aggregate metrics
    total_tps = sum(r["tps"] for r in successful)
    avg_ttft = sum(r["ttft"] for r in successful) / len(successful)
    avg_wall = sum(r["wall_time"] for r in successful) / len(successful)
    max_wall = max(r["wall_time"] for r in successful)

    return {
        "concurrent_total_tps": total_tps,
        "concurrent_avg_tps": total_tps / len(successful),
        "concurrent_avg_ttft": avg_ttft,
        "concurrent_avg_wall_ms": avg_wall,
        "concurrent_max_wall_ms": max_wall,
        "concurrent_success_rate": len(successful) / n_users,
        "concurrent_users": n_users,
    }


def measure_token_uncertainty():
    """Measure token-level uncertainty on quality gate prompts.

    Returns dict with:
      - uncertain_count: number of tokens with logprob < threshold (-2.0)
      - tail_avg: average logprob of the worst 20% of tokens
      - total_tokens: total tokens measured
    Or None on failure.
    """
    from .constants import (
        QUALITY_GATE_PROMPTS, QUALITY_GATE_N_PREDICT,
        QUALITY_GATE_SEED, QUALITY_GATE_UNCERTAIN_THRESHOLD,
        QUALITY_GATE_TAIL_PCT,
    )

    all_logprobs = []
    for prompt_text in QUALITY_GATE_PROMPTS:
        payload = {
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": QUALITY_GATE_N_PREDICT,
            "temperature": 0.0,
            "seed": QUALITY_GATE_SEED,
            "logprobs": True,
            "top_logprobs": 1,
        }
        try:
            r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json=payload, timeout=300)
            if r.status_code != 200:
                print(f"  [!] Quality request returned status {r.status_code}")
            else:
                data = r.json()
                # OpenAI-format logprobs: choices[0].logprobs.content[].logprob
                logprobs_data = data.get("choices", [{}])[0].get("logprobs", {})
                content_logprobs = logprobs_data.get("content", []) if logprobs_data else []
                if not content_logprobs:
                    print(f"  [!] No logprobs in response. Keys: {list(data.keys())}")
                for token_info in content_logprobs:
                    logprob = token_info.get("logprob")
                    if logprob is not None and logprob < 0:
                        all_logprobs.append(logprob)
        except Exception as e:
            print(f"  [!] Quality measurement failed: {e}")

    if not all_logprobs:
        return None

    # Count uncertain tokens (logprob < -2.0 = less than ~13% confidence)
    uncertain_count = sum(1 for lp in all_logprobs if lp < QUALITY_GATE_UNCERTAIN_THRESHOLD)

    # Tail-20% average: sort logprobs ascending, average the worst 20%
    sorted_lps = sorted(all_logprobs)
    tail_n = max(1, int(len(sorted_lps) * QUALITY_GATE_TAIL_PCT))
    tail_avg = sum(sorted_lps[:tail_n]) / tail_n

    return {
        "uncertain_count": uncertain_count,
        "tail_avg": tail_avg,
        "total_tokens": len(all_logprobs),
    }
