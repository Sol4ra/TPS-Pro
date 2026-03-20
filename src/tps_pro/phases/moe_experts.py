"""Expert count sweep phase: find optimal expert_used_count with quality gate."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..engine import (
    BaselineFailure,
    kill_server,
    start_server,
    wait_for_server,
)
from ..evals import measure_kl_divergence, measure_quality_gate
from ..measurement import compute_score, measure_perf_adaptive
from ..result_types import EngineConfig, PhaseReturnDict
from ..search import load_phase_results, print_trial_result, save_phase_results
from ..state import AppContext
from .moe_shared import (
    _retest_neighbors,
    _run_middle_out_sweep,
)
from .trial_helpers import thermal_gate

logger = logging.getLogger(__name__)

__all__ = ["phase_experts"]


# ---------------------------------------------------------------------------
# Expert test helper
# ---------------------------------------------------------------------------


def _test_expert_value(  # noqa: PLR0913
    ctx, expert_count, base_config, total, results_by_val, state, force_3runs=False
):
    """Test a single expert count value. Returns quality-adjusted score.

    Args:
        expert_count: The expert_used_count value to test.
        base_config: Base server config dict.
        total: Total trial count for display.
        results_by_val: Mutable dict of expert_count -> result dict.
        state: Mutable dict with 'best_score' and 'trial_num'.
        force_3runs: If True, always use 3 measurement runs.

    Returns:
        float: The quality-adjusted score for this configuration.
    """
    state["trial_num"] += 1
    trial_num = state["trial_num"]
    thermal_gate()
    config = {**base_config, "expert_used_count": expert_count}
    params_short = f"experts={expert_count}"

    lbl = "restarting server..." if not force_3runs else "re-testing (3runs)..."
    logger.info("Trial %s: %s | %s", trial_num, lbl, params_short)
    kill_server(ctx)
    proc = start_server(ctx, config)

    if wait_for_server(ctx, proc=proc) != "ok":
        from ..engine import server_start_failed

        server_start_failed(ctx, trial_num, params_short, proc)
        results_by_val[expert_count] = {
            "experts": expert_count,
            "score": 0.0,
            "speed_score": 0.0,
            "perf": None,
            "quality_factor": 0.0,
            "promoted": False,
        }
        return 0.0

    if force_3runs:
        perf, _ = measure_perf_adaptive(ctx, runs=3)
        promoted = True
    else:
        perf, promoted = measure_perf_adaptive(ctx, state["best_score"])
    tps = perf.tps
    speed_score = compute_score(perf)

    # Always measure quality in expert phase -- that's the whole point
    quality_factor = measure_quality_gate(ctx)
    score = speed_score * quality_factor

    results_by_val[expert_count] = {
        "experts": expert_count,
        "score": score,
        "speed_score": speed_score,
        "perf": perf,
        "quality_factor": quality_factor,
        "promoted": promoted,
    }

    qf_label = f" q={quality_factor:.2f}" if quality_factor < 1.0 else ""
    runs_label = "3runs" if promoted else "1run"
    state["best_score"] = print_trial_result(
        ctx,
        trial_num,
        total,
        tps,
        perf,
        f"{params_short} ({runs_label}){qf_label}",
        state["best_score"],
        final_score=score,
    )
    return score


# ---------------------------------------------------------------------------
# Results summary helpers
# ---------------------------------------------------------------------------


def _log_experts_summary(  # noqa: PLR0913
    label,
    baseline,
    default_experts,
    best_experts,
    best_entry,
    best_perf,
    all_results,
    phase_elapsed,
):
    """Log the expert count sweep results summary and histogram."""
    logger.info("%s", "=" * 60)
    logger.info("%s \u2014 RESULTS", label)
    logger.info("%s", "=" * 60)
    logger.info("Baseline:     %.1f t/s (%s experts)", baseline.tps, default_experts)
    logger.info("Best experts: %s", best_experts)
    logger.info("Best Score:   %.1f (speed \u00d7 quality)", best_entry["score"])
    logger.info("Best TPS:     %.1f t/s", best_perf.tps)
    logger.info("Quality:      %.2f", best_entry["quality_factor"])

    # Histogram
    max_score = max(r["score"] for r in all_results) if all_results else 0
    bar_max = 30
    logger.info("Score by expert_used_count (quality-adjusted):")
    logger.info("%6s  %7s  %5s  %4s", "Value", "Score", "QF", "Runs")
    logger.info(
        "%s  %s  %s  %s  %s",
        "\u2500" * 6,
        "\u2500" * 7,
        "\u2500" * 5,
        "\u2500" * 4,
        "\u2500" * bar_max,
    )
    for r in all_results:
        score = r["score"]
        bar_len = int(score / max_score * bar_max) if max_score > 0 else 0
        bar = "\u2588" * bar_len
        marker = " \u25c4 best" if r["experts"] == best_experts else ""
        runs = "3" if r["promoted"] else "1"
        qf = f"{r['quality_factor']:.2f}"
        logger.info(
            "%s  %7.1f  %s  %s  %s%s", r["experts"], score, qf, runs, bar, marker
        )

    logger.info("Duration:     %.1f min", phase_elapsed / 60)


def _build_experts_results(  # noqa: PLR0913
    ctx, baseline, best_entry, best_perf, best_experts, all_results, phase_elapsed
):
    """Build the results dict for the expert count sweep phase."""
    return {
        "phase": "experts",
        "baseline": baseline,
        "baseline_quality": ctx.quality_baseline,
        "best_tps": best_entry["score"],
        "best_metrics": {
            "tps": best_perf.tps,
            "ttft": best_perf.ttft,
            "prompt_tps": best_perf.prompt_tps,
            "total_ms": best_perf.total_ms,
            "quality_factor": best_entry.get("quality_factor", 1.0),
        },
        "best_params": {"expert_used_count": best_experts},
        "duration_seconds": round(phase_elapsed, 1),
        "all_trials": [
            {
                "number": i,
                "tps": r["score"],
                "metrics": r["perf"],
                "params": {"expert_used_count": r["experts"]},
                "quality_factor": r.get("quality_factor", 1.0),
            }
            for i, r in enumerate(all_results)
        ],
    }


# ---------------------------------------------------------------------------
# Phase function
# ---------------------------------------------------------------------------


def phase_experts(  # noqa: PLR0915
    ctx: AppContext,
    locked_moe_threads: int = 18,
    base_config: EngineConfig | None = None,
) -> PhaseReturnDict | None:
    """Expert Count Sweep: Sweep expert_used_count with perplexity quality gate.

    Sequential sweep 1-16 with adaptive measurement + quality gate.
    MoE threads are locked from MoE Thread Sweep.

    Returns PhaseReturnDict with best_params containing
    expert_used_count, or None on failure.
    """
    # Check for existing results
    existing = load_phase_results(ctx, "experts")
    if existing and "best_params" in existing:
        best_exp = existing["best_params"]["expert_used_count"]
        logger.info(
            "Expert sweep already complete \u2014 experts=%s (from previous run)",
            best_exp,
        )
        return PhaseReturnDict(
            best_params={"expert_used_count": best_exp}, phase_name="experts"
        )

    phase_start_time = time.time()
    label = "Expert Count Sweep"
    up_range = list(range(ctx.default_experts, ctx.max_experts + 1))
    down_range = list(range(ctx.default_experts - 1, 0, -1))

    logger.info("=" * 60)
    logger.info("%s", label)
    logger.info("=" * 60)
    logger.info("Locked MoE threads: %s", locked_moe_threads)

    server_config = {**ctx.naked_engine, "n_cpu_moe": locked_moe_threads}
    if base_config:
        server_config.update(base_config)

    # Start with default experts to establish baseline
    logger.info("Starting baseline server (default %s experts)...", ctx.default_experts)
    kill_server(ctx)
    proc = start_server(ctx, server_config)
    if wait_for_server(ctx, proc=proc) != "ok":
        logger.warning("Baseline server failed to start")
        if ctx.fail_fast:
            raise BaselineFailure("Baseline server failed in Expert Count Sweep.")
        return None
    baseline, _ = measure_perf_adaptive(ctx, runs=3)
    load_ms = proc.load_time_ms
    load_str = " | Load: %.0fms" % load_ms if (load_ms and ctx.debug) else ""
    logger.info(
        "Baseline: %.1f t/s | pp: %.0f t/s | TTFT: %.0fms | Score: %.1f%s",
        baseline.tps,
        baseline.prompt_tps,
        baseline.ttft,
        compute_score(baseline),
        load_str,
    )

    # Establish quality baseline with full experts
    logger.info("Measuring baseline quality (token uncertainty calibration)...")
    measure_quality_gate(ctx, is_baseline=True)

    # Populate KL-divergence baseline for expert count penalty
    if ctx.is_moe:
        logger.info("Measuring baseline KL-Divergence distribution...")
        ctx.kl_baseline_cache = measure_kl_divergence(ctx).distributions

    if ctx.quality_baseline is None:
        logger.warning("Could not measure baseline quality!")
        logger.info("The server may not support n_probs / completion_probabilities.")
        logger.info(
            "Falling back to default %s experts (no quality gate available).",
            ctx.default_experts,
        )
        return None

    total = len(up_range) + len(down_range)
    results_by_val: dict[int, dict[str, Any]] = {}
    state = {
        "best_score": compute_score(baseline),
        "best_val": ctx.default_experts,
        "trial_num": 0,
    }

    logger.info(
        "Sweeping 1-%s (middle-out from %s)", ctx.max_experts, ctx.default_experts
    )
    logger.info("Each direction stops when score drops below 50% of best")

    # Pass 1: middle-out sweep
    def test_fn(expert_count, force_3runs=False):
        return _test_expert_value(
            ctx,
            expert_count,
            server_config,
            total,
            results_by_val,
            state,
            force_3runs=force_3runs,
        )

    _run_middle_out_sweep(up_range, down_range, test_fn, state)

    # Pass 2: retest neighbors
    best_entry = max(results_by_val.values(), key=lambda x: x["score"])
    _retest_neighbors(results_by_val, best_entry["experts"], test_fn, retest_range=2)

    # Find final best after all retests
    best_entry = max(results_by_val.values(), key=lambda x: x["score"])
    best_experts = best_entry["experts"]
    all_results = [results_by_val[v] for v in sorted(results_by_val.keys())]
    best_perf = best_entry["perf"] or baseline

    phase_elapsed = time.time() - phase_start_time
    _log_experts_summary(
        label,
        baseline,
        ctx.default_experts,
        best_experts,
        best_entry,
        best_perf,
        all_results,
        phase_elapsed,
    )

    results = _build_experts_results(
        ctx, baseline, best_entry, best_perf, best_experts, all_results, phase_elapsed
    )
    save_phase_results(ctx, "experts", results)

    return PhaseReturnDict(
        best_params={"expert_used_count": best_experts}, phase_name="experts"
    )
