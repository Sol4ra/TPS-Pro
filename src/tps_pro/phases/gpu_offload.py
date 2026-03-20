"""GPU Offload phase: find optimal GPU layer offload."""

from __future__ import annotations

import logging

from ..constants import (
    GPU_SWEEP_MAX_POINTS,
    GPU_SWEEP_OOM_DEPTH,
    THERMAL_COOLDOWN_TARGET,
    THERMAL_COOLDOWN_TIMEOUT,
    THERMAL_THROTTLE_THRESHOLD,
)
from ..engine import (
    BenchOOMError,
    kill_server,
    run_bench_trial,
    start_server,
    wait_for_server,
)
from ..hardware import check_thermal_throttle, wait_for_cooldown
from ..measurement import compute_score, measure_perf_adaptive
from ..result_types import PhaseReturnDict
from ..search import load_phase_results, save_phase_results
from ..state import AppContext, update_naked_engine
from ._helpers import bench_score

logger = logging.getLogger(__name__)

__all__ = ["phase_gpu_offload"]


def _test_ngl(ctx: AppContext, ngl: int, use_bench: bool) -> bool:
    """Returns True if *ngl* value works (no OOM), False otherwise."""
    config = {**ctx.naked_engine, "n_gpu_layers": ngl}
    if use_bench:
        try:
            perf = run_bench_trial(ctx, config, repetitions=1)
            return perf is not None
        except BenchOOMError:
            return False
    proc = start_server(ctx, config)
    ok = wait_for_server(ctx, proc=proc) == "ok"
    kill_server(ctx)
    return ok


def _find_oom_boundary(ctx: AppContext, max_ngl: int, use_bench: bool) -> int | None:
    """Binary search for the highest ngl that fits in VRAM.

    Returns the highest working ngl, or None if even 0 fails.
    """
    if _test_ngl(ctx, max_ngl, use_bench):
        logger.debug("Max layers (%s) fits in VRAM — no OOM boundary", max_ngl)
        return max_ngl

    lo, hi = 0, max_ngl
    if not _test_ngl(ctx, 0, use_bench):
        logger.warning(
            "Even 0 GPU layers fails — model may be too large for system RAM"
        )
        return None

    bisect_steps = 0
    while lo < hi - 1:
        mid = (lo + hi) // 2
        bisect_steps += 1
        works = _test_ngl(ctx, mid, use_bench)
        status = "OK" if works else "OOM"
        logger.info(
            "Bisect [%s]: ngl=%s → %s  (range: %s..%s)",
            bisect_steps,
            mid,
            status,
            lo,
            hi,
        )
        if works:
            lo = mid
        else:
            hi = mid

    oom_boundary = lo
    for _retry in range(3):
        if _test_ngl(ctx, oom_boundary, use_bench):
            break
        oom_boundary = max(0, oom_boundary - 1)
        logger.info("Safety margin: dropped to ngl=%s", oom_boundary)
    logger.info(
        "OOM boundary: ngl=%s (found in %s bisections vs %s linear)",
        oom_boundary,
        bisect_steps,
        max_ngl - oom_boundary,
    )
    return oom_boundary


def _score_ngl(ctx: AppContext, ngl: int, use_bench: bool) -> tuple:
    """Measure score for a given ngl. Returns (perf, score) or (None, 0.0)."""
    config = {**ctx.naked_engine, "n_gpu_layers": ngl}
    if use_bench:
        try:
            perf = run_bench_trial(ctx, config, repetitions=3)
        except BenchOOMError:
            return None, 0.0
        if perf is None:
            return None, 0.0
    else:
        kill_server(ctx)
        proc = start_server(ctx, config)
        if wait_for_server(ctx, proc=proc) != "ok":
            kill_server(ctx)
            return None, 0.0
        perf, _ = measure_perf_adaptive(ctx, runs=3)
        kill_server(ctx)
    return perf, bench_score(perf) if use_bench else compute_score(perf)


def _score_sweep(  # noqa: C901
    ctx: AppContext, oom_boundary: int, max_ngl: int, use_bench: bool
) -> tuple[list[dict], float, int]:
    """Run the scoring sweep across NGL values near the OOM boundary.

    Returns (results_list, best_score, best_ngl) tuple.
    """
    results = []
    best_score = 0.0
    best_ngl = oom_boundary

    if oom_boundary == max_ngl:
        checkpoints = _full_range_checkpoints(max_ngl)
        logger.debug("GPU sweep: %s points across %s layers", len(checkpoints), max_ngl)

        for i, ngl in enumerate(checkpoints):
            if check_thermal_throttle(threshold=THERMAL_THROTTLE_THRESHOLD)[0]:
                wait_for_cooldown(
                    target_temp=THERMAL_COOLDOWN_TARGET,
                    timeout=THERMAL_COOLDOWN_TIMEOUT,
                )
            perf, score = _score_ngl(ctx, ngl, use_bench)
            if not perf:
                logger.warning("[%s] ngl=%3d: FAILED", i + 1, ngl)
                continue
            results.append({"ngl": ngl, "perf": perf, "score": score, "promoted": True})
            marker = ""
            if score > best_score:
                best_score = score
                best_ngl = ngl
                marker = " ** BEST **"
            logger.info(
                "  GPU Offload: %.1f t/s | pp: %.0f | TTFT: %.0fms | score: %.1f%s",
                perf.tps,
                perf.prompt_tps,
                perf.ttft,
                score,
                marker,
            )
            logger.info("    ngl=%s", ngl)
            logger.info("")
            if best_score > 0 and score < best_score * 0.50 and len(results) >= 3:  # noqa: PLR2004
                logger.debug("Score below 50%% of best — stopping early")
                break
    else:
        sweep_floor = max(0, oom_boundary - GPU_SWEEP_OOM_DEPTH)
        logger.info("Scoring sweep: ngl=%s→%s", oom_boundary, sweep_floor)

        for ngl in range(oom_boundary, sweep_floor - 1, -1):
            if check_thermal_throttle(threshold=THERMAL_THROTTLE_THRESHOLD)[0]:
                wait_for_cooldown(
                    target_temp=THERMAL_COOLDOWN_TARGET,
                    timeout=THERMAL_COOLDOWN_TIMEOUT,
                )
            perf, score = _score_ngl(ctx, ngl, use_bench)
            if not perf:
                logger.warning("ngl=%3d: FAILED", ngl)
                continue
            results.append({"ngl": ngl, "perf": perf, "score": score, "promoted": True})
            marker = ""
            if score > best_score:
                best_score = score
                best_ngl = ngl
                marker = " ** BEST **"
            logger.info(
                "  GPU Offload: %.1f t/s | pp: %.0f | TTFT: %.0fms | score: %.1f%s",
                perf.tps,
                perf.prompt_tps,
                perf.ttft,
                score,
                marker,
            )
            logger.info("    ngl=%s", ngl)
            logger.info("")
            if best_score > 0 and score < best_score * 0.90 and len(results) > 2:  # noqa: PLR2004
                logger.debug("Score dropped below 90%% of best — stopping sweep")
                break

    return results, best_score, best_ngl


def _full_range_checkpoints(max_ngl: int) -> list[int]:
    """Build evenly-spaced checkpoint list for full-range sweep."""
    n_points = min(GPU_SWEEP_MAX_POINTS, max_ngl + 1)
    step = max(1, max_ngl // (n_points - 1)) if n_points > 1 else 1
    checkpoints = sorted(
        set([max_ngl] + [max(0, max_ngl - i * step) for i in range(n_points)]),
        reverse=True,
    )
    return list(dict.fromkeys(checkpoints))


def phase_gpu_offload(ctx: AppContext) -> PhaseReturnDict | None:
    """GPU Offload: Find optimal GPU layer offload.

    For MoE models: skips entirely, locks to ctx.max_gpu_layers (all on GPU).
    MoE models use n_cpu_moe for smart CPU offloading in the MoE phase instead.

    For dense models: sweeps n_gpu_layers using middle-out approach with
    adaptive measurement and per-direction early stopping.

    Updates ctx.naked_engine and ctx.default_gpu_layers for all subsequent phases.

    Returns:
        PhaseReturnDict | None: Best params dict with 'n_gpu_layers' key,
        or None if the phase was skipped / all levels failed.
    """

    label = "GPU Offload"
    max_ngl = ctx.max_gpu_layers

    # Check for existing results
    existing = load_phase_results(ctx, "gpu")
    if existing and "best_ngl" in existing:
        best_ngl = existing["best_ngl"]
        logger.info(
            "GPU Offload already complete — n_gpu_layers=%s (from previous run)",
            best_ngl,
        )
        ctx.default_gpu_layers = best_ngl
        update_naked_engine(ctx, n_gpu_layers=best_ngl)
        return PhaseReturnDict(best_params={"n_gpu_layers": best_ngl}, phase_name="gpu")

    # MoE models: always full GPU offload — MoE phase handles smart CPU offloading
    if ctx.is_moe:
        logger.info(
            "MoE model — all %s layers on GPU (MoE phase handles CPU offloading)",
            max_ngl,
        )
        ctx.default_gpu_layers = max_ngl
        update_naked_engine(ctx, n_gpu_layers=max_ngl)
        save_phase_results(
            ctx, "gpu", {"phase": "gpu", "best_ngl": max_ngl, "skipped": "moe"}
        )
        return PhaseReturnDict(best_params={"n_gpu_layers": max_ngl}, phase_name="gpu")

    # Skip if max_gpu_layers is 0 or 1 — nothing to sweep
    if max_ngl <= 1:
        logger.info("Model has %s layers — skipping GPU offload sweep.", max_ngl)
        save_phase_results(ctx, "gpu", {"phase": "gpu", "best_ngl": max_ngl})
        return PhaseReturnDict(best_params={"n_gpu_layers": max_ngl}, phase_name="gpu")

    # Binary search for OOM boundary, then score sweep above it.
    # GPU offload is monotonic: more layers on GPU = faster until OOM.
    # Step 1: Bisect to find the exact OOM boundary in O(log N) restarts.
    # Step 2: Score the working range above the boundary.

    logger.info("=" * 60)
    logger.info("  %s", label)
    logger.info("=" * 60)
    logger.info("")

    use_bench = ctx.bench_path is not None
    if use_bench:
        logger.debug("[bench] Using llama-bench for fast GPU sweep")

    oom_boundary = _find_oom_boundary(ctx, max_ngl, use_bench)
    if oom_boundary is None:
        # Even 0 GPU layers fails
        ctx.default_gpu_layers = 0
        update_naked_engine(ctx, n_gpu_layers=0)
        save_phase_results(ctx, "gpu", {"best_ngl": 0, "reason": "all_failed"})
        return PhaseReturnDict(best_params={"n_gpu_layers": 0}, phase_name="gpu")

    results, best_score, best_ngl = _score_sweep(ctx, oom_boundary, max_ngl, use_bench)

    if not results:
        logger.warning("All offload levels failed. Using default.")
        return None

    best_result = sorted(results, key=lambda x: x["score"], reverse=True)[0]
    best_perf = best_result["perf"]

    logger.info("")
    logger.info("=" * 60)
    logger.info("  GPU Offload — RESULTS")
    logger.info("=" * 60)
    logger.info("")
    logger.info("  Winner:   ngl=%s", best_ngl)
    logger.info(
        "  Best TPS: %.1f t/s | pp: %.0f | TTFT: %.0fms | score: %.1f",
        best_perf.tps,
        best_perf.prompt_tps,
        best_perf.ttft,
        best_score,
    )

    # Update globals for all subsequent phases
    ctx.default_gpu_layers = best_ngl
    update_naked_engine(ctx, n_gpu_layers=ctx.default_gpu_layers)

    save_phase_results(
        ctx,
        "gpu",
        {
            "phase": "gpu",
            "best_ngl": best_ngl,
            "best_score": best_score,
            "all_results": [
                {"ngl": r["ngl"], "tps": r["perf"].tps, "score": r["score"]}
                for r in results
            ],
        },
    )

    return PhaseReturnDict(best_params={"n_gpu_layers": best_ngl}, phase_name="gpu")
