"""MoE sweep phase: dedicated pre-Core-Engine sweep for n_cpu_moe.

Runs BEFORE Core Engine so the optimal MoE thread count is locked and
Core Engine does not waste trials searching the moe dimension.

For non-MoE (dense) models this phase is a no-op and returns None.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..engine import (
    BaselineFailure,
    boot_server_with_jinja_recovery,
    kill_server,
)
from ..measurement import compute_score, measure_perf_adaptive
from ..pipeline_config import PhaseConfig
from ..result_types import PhaseReturnDict
from ..search import load_phase_results, save_phase_results
from ..state import AppContext

logger = logging.getLogger(__name__)

__all__ = ["phase_moe_sweep"]


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------


def _log_trial_line(
    perf: Any, score: float, n_cpu_moe: int, is_best: bool = False
) -> None:
    """Log a single trial result in the standard output format."""
    marker = " ** BEST **" if is_best else ""
    logger.info(
        "  MoE Threads: %5.1f t/s | pp: %4.0f | TTFT: %3.0fms | score: %5.1f%s",
        perf.tps,
        perf.prompt_tps,
        perf.ttft,
        score,
        marker,
    )
    logger.info("    n_cpu_moe=%s", n_cpu_moe)
    logger.info("")


def _log_results_summary(
    baseline: Any,
    baseline_score: float,
    best_perf: Any,
    best_score: float,
    best_moe: int,
) -> None:
    """Log the final results summary block."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  MoE Threads %s RESULTS", "\u2014")
    logger.info("=" * 60)
    logger.info("")
    logger.info(
        "  Baseline:  %5.1f t/s | pp: %4.0f | TTFT: %3.0fms | score: %5.1f",
        baseline.tps,
        baseline.prompt_tps,
        baseline.ttft,
        baseline_score,
    )
    logger.info(
        "  Optimal:   %5.1f t/s | pp: %4.0f | TTFT: %3.0fms | score: %5.1f",
        best_perf.tps,
        best_perf.prompt_tps,
        best_perf.ttft,
        best_score,
    )
    logger.info("")
    logger.info("  Params:    n_cpu_moe=%s", best_moe)


# ---------------------------------------------------------------------------
# Extracted helpers: boot-measure-score, baseline, neighbor recheck
# ---------------------------------------------------------------------------


def _measure_moe_value(
    ctx: AppContext, base_config: dict[str, Any], moe_val: int
) -> tuple[Any, float | None]:
    """Boot server with a given moe value, measure, return (perf, score).

    Returns (perf, score) on success, or (None, None) if the server
    failed to start.
    """
    config = {**base_config, "n_cpu_moe": moe_val}

    kill_server(ctx)
    proc, status = boot_server_with_jinja_recovery(ctx, config)
    if status != "ok":
        logger.warning("  n_cpu_moe=%s: server failed to start", moe_val)
        kill_server(ctx)
        return None, None

    perf, _ = measure_perf_adaptive(ctx, runs=3)
    score = compute_score(perf)
    return perf, score


def _measure_baseline(
    ctx: AppContext, base_config: dict[str, Any]
) -> tuple[Any, float | None]:
    """Boot server with base config and measure baseline score.

    Returns (perf, score) on success, or (None, None) on failure.
    """
    kill_server(ctx)
    proc, status = boot_server_with_jinja_recovery(ctx, base_config)
    if status != "ok":
        return None, None

    perf, _ = measure_perf_adaptive(ctx, runs=3)
    score = compute_score(perf)
    return perf, score


def _recheck_neighbors(  # noqa: PLR0913
    ctx: AppContext,
    base_config: dict[str, Any],
    best_moe: int,
    best_perf: Any,
    best_score: float,
    results: list,
) -> tuple[int, Any, float]:
    """Recheck neighbors (+/-1) of the current winner.

    Mutates *results* in place (appends new trial tuples).
    Returns updated (best_moe, best_perf, best_score).
    """
    tested_values = {r[0] for r in results}
    neighbors = [best_moe - 1, best_moe + 1]
    neighbors = [n for n in neighbors if n >= 1 and n not in tested_values]

    if neighbors:
        logger.info("")
        logger.info("  Rechecking neighbors of winner (moe=%d)...", best_moe)

    for moe_val in neighbors:
        perf, score = _measure_moe_value(ctx, base_config, moe_val)
        if perf is None:
            continue

        results.append((moe_val, perf, score))
        is_best = score > best_score
        if is_best:
            best_moe, best_perf, best_score = moe_val, perf, score
        _log_trial_line(perf, score, moe_val, is_best=is_best)

    kill_server(ctx)
    return best_moe, best_perf, best_score


# ---------------------------------------------------------------------------
# Phase function
# ---------------------------------------------------------------------------


def phase_moe_sweep(  # noqa: C901, PLR0912, PLR0915
    ctx: AppContext,
    force: bool = False,
    phase_config: PhaseConfig | None = None,
) -> PhaseReturnDict | None:
    """Sweep n_cpu_moe to find the optimal MoE thread count.

    This dedicated sweep runs before Core Engine so the winner is locked
    and Core Engine does not waste trials on the moe dimension.

    For dense (non-MoE) models, returns None immediately.

    Args:
        ctx: Application context (must have is_moe, naked_engine, etc.).
        force: If True, re-run even if previous results exist.

    Returns:
        PhaseReturnDict with best_params={"n_cpu_moe": winner}, or None
        if the model is not MoE or baseline fails.
    """
    if not ctx.is_moe:
        logger.debug("Dense model detected -- skipping MoE sweep phase.")
        return None

    # Check for cached results unless force re-run requested
    if not force:
        existing = load_phase_results(ctx, "moe_sweep")
        if existing and "best_params" in existing:
            winner = existing["best_params"].get("n_cpu_moe")
            if winner is not None:
                logger.info(
                    "MoE sweep already complete -- n_cpu_moe=%s (cached)", winner
                )
                return PhaseReturnDict(
                    best_params={"n_cpu_moe": winner}, phase_name="moe_sweep"
                )

    phase_start = time.time()
    sweep_max = getattr(ctx, "moe_sweep_max", 24) or 24

    # Override range/step from phase_config if provided
    sweep_min = 8
    sweep_step = 2
    if phase_config is not None:
        if phase_config.range and len(phase_config.range) >= 2:  # noqa: PLR2004
            sweep_min = phase_config.range[0]
            sweep_max = phase_config.range[1]
        if phase_config.step:
            sweep_step = phase_config.step

    logger.info("")
    logger.info("=" * 60)
    logger.info("  MoE Threads")
    logger.info("=" * 60)
    logger.info("")

    # Build base config from current naked_engine (includes GPU offload results)
    base_config = dict(ctx.naked_engine)

    # --- Baseline measurement ---
    baseline, baseline_score = _measure_baseline(ctx, base_config)
    if baseline is None:
        logger.warning("Baseline server failed to start in MoE sweep")
        if ctx.fail_fast:
            raise BaselineFailure("Baseline server failed in MoE sweep.")
        return None

    logger.info(
        "  Baseline: %5.1f t/s | pp: %4.0f | TTFT: %3.0fms | score: %5.1f",
        baseline.tps,
        baseline.prompt_tps,
        baseline.ttft,
        baseline_score,
    )
    logger.info("")

    # --- Sweep n_cpu_moe from sweep_min to sweep_max ---
    sweep_values = list(range(sweep_min, sweep_max + 1, sweep_step))
    # Always include sweep_max even if it's odd
    if sweep_max not in sweep_values:
        sweep_values.append(sweep_max)

    results = []  # list of (n_cpu_moe, perf, score)
    best_so_far = baseline_score

    for moe_val in sweep_values:
        perf, score = _measure_moe_value(ctx, base_config, moe_val)
        if perf is None:
            continue

        results.append((moe_val, perf, score))
        is_best = score > best_so_far
        if is_best:
            best_so_far = score
        _log_trial_line(perf, score, moe_val, is_best=is_best)

    kill_server(ctx)

    if not results:
        logger.warning("MoE sweep: no successful trials")
        return None

    # --- Pick initial winner and recheck neighbors ---
    best_moe, best_perf, best_score = max(results, key=lambda r: r[2])
    best_moe, best_perf, best_score = _recheck_neighbors(
        ctx,
        base_config,
        best_moe,
        best_perf,
        best_score,
        results,
    )

    _log_results_summary(baseline, baseline_score, best_perf, best_score, best_moe)

    # --- Save results ---
    phase_elapsed = time.time() - phase_start
    results_data = {
        "phase": "moe_sweep",
        "baseline_tps": baseline.tps,
        "baseline_score": baseline_score,
        "best_tps": best_perf.tps,
        "best_score": best_score,
        "best_params": {"n_cpu_moe": best_moe},
        "duration_seconds": round(phase_elapsed, 1),
        "all_trials": [
            {
                "n_cpu_moe": moe_val,
                "tps": perf.tps,
                "prompt_tps": perf.prompt_tps,
                "ttft": perf.ttft,
                "score": score,
            }
            for moe_val, perf, score in results
        ],
    }
    save_phase_results(ctx, "moe_sweep", results_data)

    logger.info("  Duration: %.1f min", phase_elapsed / 60)
    logger.info("")

    return PhaseReturnDict(best_params={"n_cpu_moe": best_moe}, phase_name="moe_sweep")
