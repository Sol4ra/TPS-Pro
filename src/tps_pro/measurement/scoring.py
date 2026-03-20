"""Composite scoring functions for performance evaluation.

Provides dual-mode scoring (full/lite), Pareto objective extraction,
and Pareto front utilities for multi-objective optimization.
"""

from __future__ import annotations

import logging
import math

import optuna

from ..constants import (
    CONCURRENT_BASE_FACTOR,
    CONCURRENT_BONUS_WEIGHT,
    LITE_BASE_MULTIPLIER,
    LITE_MULTIPLIER_CAP,
    LITE_VRAM_BONUS_CAP,
    LITE_VRAM_BONUS_SCALE,
    LITE_WEIGHT_PP,
    LITE_WEIGHT_TTFT,
    NORM_CAP_MULTIPLIER,
    PROMPT_TPS_CLAMP_MAX,
    SCORE_PP_BASELINE,
    TTFT_BASELINE_MS,
    TTFT_FLOOR_MS,
    VRAM_FAILURE_PENALTY,
    WEIGHT_GEN_TPS,
    WEIGHT_LARGE_TPS,
    WEIGHT_PP_COMPONENT,
    WEIGHT_TTFT_COMPONENT,
    WEIGHT_VRAM,
)
from ..result_types import (
    ParetoObjectives,
    PerfResult,
    PerfSample,
)
from ..state import AppContext, get_config

logger = logging.getLogger(__name__)

__all__ = [
    "compute_score",
    "compute_pareto_objectives",
    "extract_pareto_front",
    "print_pareto_front",
    "get_best_trial",
    "_clamp_finite",
]


def _score_full_mode(  # noqa: PLR0913
    gen_tps: float,
    large_tps: float,
    prompt_tps: float,
    ttft: float,
    vram_used_mb: float | None,
    vram_total_mb: float | None,
) -> float:
    """Full scoring mode: weighted sum across all signal dimensions."""
    pp_norm = min(prompt_tps / SCORE_PP_BASELINE, NORM_CAP_MULTIPLIER)
    ttft_norm = min(TTFT_BASELINE_MS / ttft, NORM_CAP_MULTIPLIER)

    score = (
        gen_tps * WEIGHT_GEN_TPS
        + large_tps * WEIGHT_LARGE_TPS
        + pp_norm * gen_tps * WEIGHT_PP_COMPONENT
        + ttft_norm * gen_tps * WEIGHT_TTFT_COMPONENT
    )

    if vram_used_mb is not None and vram_total_mb is not None and vram_total_mb > 0:
        vram_bonus = max(0.0, min(1.0, 1.0 - (vram_used_mb / vram_total_mb)))
        score += vram_bonus * gen_tps * WEIGHT_VRAM
    else:
        score += gen_tps * WEIGHT_VRAM
    return score


def _score_lite_mode(
    gen_tps: float,
    prompt_tps: float,
    ttft: float,
    vram_used_mb: float | None,
    vram_total_mb: float | None,
) -> float:
    """Lightweight scoring mode: quick-reject formula without large-prompt overhead."""
    pp_factor = (
        min(prompt_tps / SCORE_PP_BASELINE, NORM_CAP_MULTIPLIER)
        if prompt_tps > 0
        else 0.0
    )
    ttft_factor = min(TTFT_BASELINE_MS / ttft, NORM_CAP_MULTIPLIER) if ttft > 0 else 0.0

    multiplier = min(
        LITE_BASE_MULTIPLIER
        + LITE_WEIGHT_PP * pp_factor
        + LITE_WEIGHT_TTFT * ttft_factor,
        LITE_MULTIPLIER_CAP,
    )
    score = gen_tps * multiplier

    if vram_used_mb is not None and vram_total_mb is not None and vram_total_mb > 0:
        utilization = vram_used_mb / vram_total_mb
        headroom_bonus = min(
            LITE_VRAM_BONUS_CAP,
            max(0.0, (1.0 - utilization) * LITE_VRAM_BONUS_SCALE),
        )
        score *= 1.0 + headroom_bonus
    return score


def _apply_concurrent_bonus(
    score: float, perf: PerfResult | PerfSample, gen_tps: float
) -> float:
    """Apply concurrent-load scaling bonus if data is present."""
    concurrent_tps = getattr(perf, "concurrent_total_tps", None)
    if not concurrent_tps or concurrent_tps <= 0:
        return score
    n_users = getattr(perf, "concurrent_users", None) or 4
    scaling_efficiency = concurrent_tps / (gen_tps * n_users) if gen_tps > 0 else 0
    scaling_efficiency = min(1.0, max(0.0, scaling_efficiency))
    return score * (
        CONCURRENT_BASE_FACTOR + CONCURRENT_BONUS_WEIGHT * scaling_efficiency
    )


def _clamp_finite(
    value: float, floor: float, cap: float, fallback: float | None = None
) -> float:
    """Clamp a value to [floor, cap], substituting fallback (or floor) if non-finite."""
    if not math.isfinite(value):
        return fallback if fallback is not None else floor
    return min(cap, max(floor, value))


def compute_score(
    perf: PerfResult | PerfSample,
    vram_used_mb: float | None = None,
    vram_total_mb: float | None = None,
) -> float:
    """Dual-mode composite score from performance metrics.

    Accepts a PerfResult or PerfSample dataclass.

    Full mode (when large-prompt data is present):
      score = gen_tps*WEIGHT_GEN_TPS + large_tps*WEIGHT_LARGE_TPS
              + pp*WEIGHT_PP_COMPONENT
              + ttft_factor*WEIGHT_TTFT_COMPONENT
              + vram_factor*WEIGHT_VRAM

    Lightweight mode (quick filter pass, no large-prompt data):
      score = gen_tps * (LITE_BASE_MULTIPLIER + LITE_WEIGHT_PP * pp/pp_baseline
                         + LITE_WEIGHT_TTFT * ttft_baseline/ttft)

    The full formula rewards configs that maintain TPS under heavy context load.
    The lightweight formula is used during the initial quick-reject pass in adaptive
    measurement, where running a large-prompt benchmark would be too expensive.
    """
    if vram_used_mb is None:
        vram_used_mb = getattr(perf, "vram_used_mb", None)
    if vram_total_mb is None:
        vram_total_mb = getattr(perf, "vram_total_mb", None)

    gen_tps = getattr(perf, "tps", 0.0)
    if gen_tps <= 0:
        return 0.0

    prompt_tps = _clamp_finite(
        getattr(perf, "prompt_tps", 0.0),
        0.0,
        PROMPT_TPS_CLAMP_MAX,
        fallback=PROMPT_TPS_CLAMP_MAX,
    )
    ttft = _clamp_finite(
        getattr(perf, "ttft", 0.0),
        TTFT_FLOOR_MS,
        TTFT_FLOOR_MS * 1e6,
        fallback=TTFT_FLOOR_MS,
    )

    large_tps = getattr(perf, "large_tps", None)

    if large_tps and large_tps > 0:
        score = _score_full_mode(
            gen_tps, large_tps, prompt_tps, ttft, vram_used_mb, vram_total_mb
        )
    else:
        score = _score_lite_mode(gen_tps, prompt_tps, ttft, vram_used_mb, vram_total_mb)

    score = _apply_concurrent_bonus(score, perf, gen_tps)
    return score if math.isfinite(score) else 0.0


def compute_pareto_objectives(
    perf: PerfResult | PerfSample, quality_factor: float = 1.0
) -> ParetoObjectives:
    """Extract multi-objective values for Pareto optimization.

    Returns ParetoObjectives(tps, neg_vram, quality_factor):
      - tps: generation tokens/sec (maximize)
      - neg_vram: negative VRAM usage in MB (maximize = less VRAM)
      - quality_factor: 0.0-1.0 quality gate (maximize)

    The negative VRAM trick converts "minimize VRAM" into "maximize -VRAM"
    so all objectives use the same direction.
    """
    tps = getattr(perf, "tps", 0.0)
    vram_mb = getattr(perf, "vram_used_mb", None)
    # Penalize unknown VRAM with sentinel
    neg_vram = -vram_mb if vram_mb is not None else VRAM_FAILURE_PENALTY
    return ParetoObjectives(tps=tps, neg_vram=neg_vram, quality_factor=quality_factor)


def extract_pareto_front(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    """Extract Pareto-optimal trials from a multi-objective study.

    Returns list of trials sorted by TPS (first objective, descending).
    """
    try:
        pareto_trials = study.best_trials  # NSGA-II provides this
    except (RuntimeError, ValueError):
        return []
    # Sort by TPS descending
    return sorted(pareto_trials, key=lambda t: t.values[0], reverse=True)


def print_pareto_front(pareto_trials: list[optuna.trial.FrozenTrial]) -> None:
    """Print the Pareto front as a table for the user to pick from."""
    if not pareto_trials:
        logger.warning("No Pareto-optimal configs found.")
        return

    logger.info("  %4s  %8s  %8s  %8s  Key Params", "", "TPS", "VRAM MB", "Quality")
    logger.info("  %4s  %s  %s  %s  %s", "", "─" * 8, "─" * 8, "─" * 8, "─" * 30)
    for i, t in enumerate(pareto_trials):
        tps = t.values[0]
        vram = -t.values[1]  # un-negate
        qf = t.values[2]
        # Extract a few key params for display
        p = t.params
        short: list[str] = []
        if "threads" in p:
            short.append("t=%s" % p["threads"])
        if "kv_cache_type" in p:
            short.append("kv=%s" % p["kv_cache_type"])
        if "batch_size" in p:
            short.append("b=%s" % p["batch_size"])
        if "flash_attn" in p:
            short.append("fa=%s" % p["flash_attn"])
        if "draft_max" in p:
            short.append("draft=%s" % p["draft_max"])
        params_str = " ".join(short) if short else str(p)[:50]
        logger.info(
            "  [%2d]  %8.1f  %8.0f  %8.2f  %s", i + 1, tps, vram, qf, params_str
        )


def get_best_trial(ctx: AppContext, study: optuna.Study) -> optuna.trial.FrozenTrial:
    """Get the best trial from a study, handling both single and multi-objective modes.

    In multi-objective (Pareto) mode, returns the trial with the highest TPS
    from the Pareto front (objective 0 = TPS). Falls back to first trial if needed.

    Args:
        ctx: Accepted for future extensibility (e.g. user-defined objective
             weights, per-context scoring overrides). Currently only the
             global ``pareto`` config flag is read.
        study: An Optuna study (single- or multi-objective).
    """
    is_pareto = get_config("pareto", False)
    if is_pareto:
        pareto = extract_pareto_front(study)
        if pareto:
            return pareto[0]  # highest TPS on the front
        # Fallback: just pick the trial with best first objective
        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        if completed:
            return max(completed, key=lambda t: t.values[0] if t.values else 0)
        return study.trials[0]
    return study.best_trial
