"""KV + Context Sweep phase: find optimal KV type and context size combination.

For each KV cache type (f16, q8_0, q4_0):
  1. Binary-search the maximum bootable context size.
  2. Measure TPS + variable-tracking quality at each practical context level.
  3. Score and pick the best combination.

Quality test: Variable Tracking -- the hardest long-context test.
A variable is assigned a value early in the context (5% depth), then updated
multiple times. The model must recall the FIRST value, forcing it to attend
all the way back to the deepest position. KV cache quantization degrades
recall at deep positions first, making this maximally sensitive.

Test prompts are generated ONCE and reused across all KV types for fair
comparison. Any score difference is purely from cache precision.

Boot scanning lives in kv_sweep_boot.py; measurement and scoring in
kv_sweep_measure.py. This module is the thin orchestration layer.
"""

from __future__ import annotations

import logging

from ..constants import DEFAULT_CONTEXT_SIZE
from ..engine import kill_server
from ..pipeline_config import PhaseConfig, ScoringWeights
from ..result_types import EngineConfig, PhaseReturnDict
from ..search import load_phase_results, save_phase_results
from ..state import AppContext
from .kv_sweep_boot import (
    KV_TYPES,
    discover_bootable_contexts,
    get_model_max_context,
    get_model_metadata,
)
from .kv_sweep_measure import (
    log_sweep_results,
    measure_single_kv_type,
    prepare_test_prompts,
    score_measurements,
)

logger = logging.getLogger(__name__)

_BASELINE_CONTEXT = DEFAULT_CONTEXT_SIZE

__all__ = ["phase_kv_context_sweep"]


def phase_kv_context_sweep(
    ctx: AppContext,
    force: bool = False,
    base_config: EngineConfig | None = None,
    phase_config: PhaseConfig | None = None,
    scoring_weights: ScoringWeights | None = None,
) -> PhaseReturnDict | None:
    """KV + Context Sweep: find the optimal KV cache type and context size.

    Step 1: For each KV type, binary-search the max bootable context.
    Step 2: Measure TPS + NIAH at practical context sizes.
    Step 3: Score and pick the winner.

    Args:
        scoring_weights: Optional scoring weights for the KV sweep formula.
            When ``None``, the default weights (0.5/0.3/0.2) are used.

    Returns:
        PhaseReturnDict with best kv_cache_type and context, or None if
        all types fail.
    """
    existing = load_phase_results(ctx, "kv_context_sweep")
    if not force and existing and "best_params" in existing:
        bp = existing["best_params"]
        logger.info(
            "KV + Context Sweep already complete -- kv=%s, context=%s",
            bp.get("kv_cache_type"),
            bp.get("context"),
        )
        return PhaseReturnDict(best_params=bp, phase_name="kv_context_sweep")

    logger.info("=" * 60)
    logger.info("  KV + Context Sweep")
    logger.info("=" * 60)
    logger.info("")

    effective_config = (
        base_config if base_config is not None else dict(ctx.naked_engine)
    )

    # Override KV types from phase_config if provided
    kv_types_override = None
    if phase_config is not None and phase_config.kv_types:
        kv_types_override = phase_config.kv_types

    try:
        return _run_kv_context_sweep(
            ctx,
            effective_config,
            kv_types_override,
            scoring_weights,
        )
    finally:
        kill_server(ctx)


def _run_kv_context_sweep(
    ctx: AppContext,
    base_config: dict,
    kv_types_override: list[str] | None = None,
    scoring_weights: ScoringWeights | None = None,
) -> PhaseReturnDict | None:
    """Inner implementation -- always called within a kill_server finally block.

    Args:
        kv_types_override: If provided, only test these KV types instead of
            the default KV_TYPES list.
        scoring_weights: Optional scoring weights for the KV sweep formula.
    """
    kv_data: dict[str, dict] = {}
    all_measurements: list[dict] = []

    effective_kv_types = (
        kv_types_override if kv_types_override is not None else KV_TYPES
    )

    meta = get_model_metadata(ctx)
    model_max_ctx = get_model_max_context(meta)
    if model_max_ctx > 0:
        logger.info("  Model max context: %dk tokens", model_max_ctx // 1024)

    # Step 0: Discover bootable contexts and pre-generate prompts
    kv_bootable, all_test_points = discover_bootable_contexts(
        ctx,
        model_max_ctx,
        base_config,
    )
    if not all_test_points:
        logger.warning("  All KV types failed at minimum context -- aborting")
        return None

    test_prompts = prepare_test_prompts(ctx, all_test_points, base_config)

    # Step 1 + 2: Measure each KV type
    for kv_type in effective_kv_types:
        max_bootable = kv_bootable[kv_type]
        if max_bootable is None:
            kv_data[kv_type] = {
                "max_bootable": 0,
                "max_practical": 0,
                "measurements": [],
            }
            continue

        logger.info("")
        logger.info("  Testing %s...", kv_type)
        measurements, max_practical = measure_single_kv_type(
            ctx,
            kv_type,
            max_bootable,
            test_prompts,
            base_config,
        )
        all_measurements.extend(measurements)
        kv_data[kv_type] = {
            "max_bootable": max_bootable,
            "max_practical": max_practical,
            "measurements": measurements,
        }

    # Step 3: Score and pick winner
    weights_tuple = (
        (scoring_weights.tps, scoring_weights.context, scoring_weights.pp_speed)
        if scoring_weights is not None
        else None
    )
    best_score, best_kv, best_ctx, best_measurement = score_measurements(
        all_measurements,
        scoring_weights=weights_tuple,
    )

    if best_kv is None or best_ctx is None:
        logger.warning("KV + Context Sweep: all types failed -- returning None")
        save_phase_results(
            ctx,
            "kv_context_sweep",
            {
                "phase": "kv_context_sweep",
                "kv_types_tested": kv_data,
                "error": "all_failed",
            },
        )
        return None

    baseline_m = next(
        (
            m
            for m in all_measurements
            if m["kv_type"] == "f16" and m["context"] == _BASELINE_CONTEXT
        ),
        None,
    )

    log_sweep_results(baseline_m, best_measurement, best_kv, best_ctx)

    # Save results
    best_params = {"kv_cache_type": best_kv, "context": best_ctx}
    baseline_tps = baseline_m["tps"] if baseline_m else 0
    baseline_qp = baseline_m.get("quality_pass", False) if baseline_m else False

    results_dict = {
        "phase": "kv_context_sweep",
        "kv_types_tested": kv_data,
        "best_kv_type": best_kv,
        "best_context": best_ctx,
        "best_score": best_score,
        "best_params": best_params,
        "baseline": {
            "tps": baseline_tps,
            "quality_pass": baseline_qp,
            "context": DEFAULT_CONTEXT_SIZE,
            "kv_type": "f16",
        },
    }
    save_phase_results(ctx, "kv_context_sweep", results_dict)

    return PhaseReturnDict(best_params=best_params, phase_name="kv_context_sweep")
