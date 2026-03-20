"""Shared helper functions used by multiple phase modules.

Error strategy (see errors.py for full documentation):
    - build_phase_config(): reads completed phase results via
      load_phase_results() which returns None on error -- missing phases
      are silently skipped (their defaults remain in the config).
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from ..constants import (
    BENCH_WEIGHT_GEN_TPS,
    BENCH_WEIGHT_PP_TIEBREAK,
)
from ..result_types import EngineConfig
from ..search import load_phase_results
from ..state import AppContext

logger = logging.getLogger(__name__)

__all__ = [
    "bench_score",
    "get_moe_config",
    "build_phase_config",
]

# Module-level cache for build_phase_config to avoid re-reading
# the same phase results JSON files multiple times in a single pipeline run.
_phase_config_cache: dict[tuple, dict] = {}
_phase_config_cache_lock = threading.Lock()


def _get_val(obj: object, key: str, default: float = 0) -> float:
    """Get a value from a dict or dataclass, supporting both access patterns."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def bench_score(p: object, baseline: object | None = None) -> float:
    """Bench scoring: gen_tps with prompt_tps as a relative tiebreaker.

    Accepts p and baseline as either dicts or dataclass objects
    (PerfResult, BenchResult).

    llama-bench prompt_tps/ttft are not comparable to HTTP in absolute terms,
    but the RELATIVE differences between configs are real (faster bench pp =
    faster server pp). We use prompt_tps normalized against the bench baseline
    as a bounded tiebreaker so two configs with equal gen_tps can be separated
    by their prompt processing speed.

    Score = gen_tps * (BENCH_WEIGHT_GEN_TPS + BENCH_WEIGHT_PP_TIEBREAK * pp_ratio),
    capped at 2x baseline pp.  Without a baseline, falls back to gen_tps only.
    """
    tps = _get_val(p, "tps", 0)
    if tps <= 0:
        return 0.0
    pp = _get_val(p, "prompt_tps", 0)
    baseline_pp = _get_val(baseline, "prompt_tps", 0) if baseline else 0
    if baseline and baseline_pp > 0 and pp > 0:
        pp_ratio = min(pp / baseline_pp, 2.0)
    else:
        pp_ratio = 1.0
    return tps * (BENCH_WEIGHT_GEN_TPS + BENCH_WEIGHT_PP_TIEBREAK * pp_ratio)


def get_moe_config(
    ctx: AppContext,
    p1a_results: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract MoE config dict from MoE phase results, with defaults."""
    if not ctx.is_moe:
        return {}
    if p1a_results and "best_params" in p1a_results:
        bp = p1a_results["best_params"]
        return {
            "n_cpu_moe": bp.get("n_cpu_moe", ctx.moe_sweep_center),
            "expert_used_count": bp.get("expert_used_count", ctx.default_experts),
        }
    moe_data = load_phase_results(ctx, "moe")
    if moe_data and "best_params" in moe_data:
        moe_threads = moe_data["best_params"].get("n_cpu_moe", ctx.moe_sweep_center)
        logger.info(
            "Using MoE threads from MoE sweep: %s (expert sweep was skipped)",
            moe_threads,
        )
        return {"n_cpu_moe": moe_threads, "expert_used_count": ctx.default_experts}
    return {"n_cpu_moe": ctx.moe_sweep_center, "expert_used_count": ctx.default_experts}


def build_phase_config(
    ctx: AppContext, include_phases: list[str] | None = None
) -> EngineConfig:
    """Build server config by merging results from completed phases.

    Starts from ``ctx.naked_engine`` and layers on ``best_params`` from each
    completed phase in dependency order.  Callers can restrict which phases
    are included via *include_phases*; by default all standard phases are
    merged.

    Results are cached per (results_dir, include_phases) tuple to avoid
    re-reading the same JSON files multiple times in a single pipeline run.
    Returns:
        A new config dict (never mutates ``ctx.naked_engine``).
    """
    phase_order = ["gpu", "core_engine", "io_toggles", "speculation", "kv_quality"]
    phases_key = tuple(include_phases) if include_phases else tuple(phase_order)
    cache_key = (str(getattr(ctx, "results_dir", "")), phases_key)

    with _phase_config_cache_lock:
        if cache_key in _phase_config_cache:
            # Return a fresh copy merged with current naked_engine
            # (naked_engine may have been updated by earlier phases like GPU offload)
            cached_params = _phase_config_cache[cache_key]
            return {**ctx.naked_engine, **cached_params}

    merged_params: dict = {}
    for phase_name in phases_key:
        data = load_phase_results(ctx, phase_name)
        if data and "best_params" in data:
            merged_params = {**merged_params, **data["best_params"]}

    with _phase_config_cache_lock:
        _phase_config_cache[cache_key] = merged_params
    return {**ctx.naked_engine, **merged_params}
