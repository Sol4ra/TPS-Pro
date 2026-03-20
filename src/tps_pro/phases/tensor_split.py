"""Tensor Split and Topology Sweep phases (multi-GPU)."""

from __future__ import annotations

import logging
import random

from ..engine import (
    generate_tensor_splits,
    kill_server,
    server_start_failed,
    start_server,
    wait_for_server,
)
from ..measurement import compute_score, measure_perf_adaptive
from ..result_types import EngineConfig, PhaseReturnDict
from ..search import load_phase_results, save_phase_results
from ..state import AppContext, update_naked_engine

logger = logging.getLogger(__name__)

__all__ = ["phase_tensor_split"]


def phase_tensor_split(  # noqa: PLR0915
    ctx: AppContext,
    gpus: list,
    base_config: EngineConfig | None = None,
    n_trials: int = 20,
) -> PhaseReturnDict | None:
    """Tensor Split: Sweep split ratios across multiple GPUs.

    Returns:
        PhaseReturnDict | None: Best params dict with 'tensor_split' key,
        or None if the phase was skipped.
    """
    gpu_count = len(gpus)
    existing = load_phase_results(ctx, "tensor_split")
    if existing and "best_split" in existing:
        best_split = tuple(existing["best_split"])
        best_split_str = ",".join(str(s) for s in best_split)
        logger.info("Tensor Split already complete \u2014 split=%s", best_split)
        return PhaseReturnDict(
            best_params={"tensor_split": best_split_str},
            phase_name="tensor_split",
        )

    if gpu_count < 2:  # noqa: PLR2004
        logger.info("Single GPU \u2014 skipping tensor split sweep")
        save_phase_results(
            ctx,
            "tensor_split",
            {"phase": "tensor_split", "best_split": [1.0], "skipped": "single_gpu"},
        )
        return None

    if base_config is None:
        base_config = dict(ctx.naked_engine)

    logger.info("=" * 60)
    logger.info("Tensor Split")
    logger.info("=" * 60)

    candidates = generate_tensor_splits(gpu_count)
    even_split = tuple([round(1.0 / gpu_count, 2)] * gpu_count)
    if len(candidates) > n_trials:
        random.shuffle(candidates)
        candidates = candidates[: n_trials - 1]
        if even_split not in candidates:
            candidates.insert(0, even_split)

    logger.info("Testing %s split ratios across %s GPUs", len(candidates), gpu_count)

    results = []
    best_score = 0.0
    best_split = even_split

    for trial_num, split in enumerate(candidates, 1):
        split_str = ",".join(str(s) for s in split)
        config = {**base_config, "tensor_split": split_str}
        kill_server(ctx)
        proc = start_server(ctx, config)
        if wait_for_server(ctx, proc=proc) != "ok":
            server_start_failed(ctx, trial_num, f"split={split_str}", proc)
            kill_server(ctx)
            continue
        perf, promoted = measure_perf_adaptive(ctx, best_score)
        score = compute_score(perf)
        results.append(
            {
                "split": list(split),
                "split_str": split_str,
                "perf": perf,
                "score": score,
                "promoted": promoted,
            }
        )
        marker = " *NEW BEST*" if score > best_score else ""
        if score > best_score:
            best_score = score
            best_split = split
        runs_label = "3 runs" if promoted else "1 run"
        load_ms = proc.load_time_ms
        load_str = f" | Load: {load_ms:.0f}ms" if (load_ms and ctx.debug) else ""
        logger.info(
            "[%s] split=%s: %.1f t/s | Score: %.1f (%s)%s%s",
            trial_num,
            split_str,
            perf.tps,
            score,
            runs_label,
            load_str,
            marker,
        )
        kill_server(ctx)

    if not results:
        logger.warning("All tensor split configs failed. Using even split.")
        best_split = even_split

    best_split_str = ",".join(str(s) for s in best_split)
    logger.info(">>> Best tensor split: %s (score: %.1f)", best_split_str, best_score)
    save_phase_results(
        ctx,
        "tensor_split",
        {
            "phase": "tensor_split",
            "best_split": list(best_split),
            "best_split_str": best_split_str,
            "best_score": best_score,
            "gpu_count": gpu_count,
            "all_results": results,
        },
    )
    update_naked_engine(ctx, tensor_split=best_split_str)
    return PhaseReturnDict(
        best_params={"tensor_split": best_split_str},
        phase_name="tensor_split",
    )
