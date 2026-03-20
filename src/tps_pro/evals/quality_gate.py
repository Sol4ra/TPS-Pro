"""Quality gate using token-level uncertainty comparison against baseline."""

from __future__ import annotations

import logging
from typing import cast

from ..constants import (
    QUALITY_GATE_CEILING,
    QUALITY_GATE_CLIFF,
    QUALITY_GATE_CLIFF_PENALTY,
    QUALITY_GATE_SOFT_PENALTY,
)
from ..measurement import measure_token_uncertainty
from ..result_types import TokenUncertaintyResult
from ..state import AppContext

logger = logging.getLogger(__name__)

__all__ = ["measure_quality_gate"]


def measure_quality_gate(ctx: AppContext, is_baseline: bool = False) -> float:  # noqa: C901, PLR0912
    """Quality gate using token-level uncertainty comparison against baseline.

    Measures two signals:
      1. Uncertain token count increase (tokens with logprob < -2.0)
      2. Tail-20% logprob degradation (worst 20% of tokens)
    Uses the worse of the two signals to determine the quality factor.

    On baseline run (is_baseline=True): measures and stores baseline metrics.
    On subsequent runs: returns quality_factor (0.1-1.0) based on degradation.
    """

    raw_metrics = measure_token_uncertainty(ctx)
    if raw_metrics is None:
        metrics = None
    elif isinstance(raw_metrics, dict):
        metrics = cast(
            TokenUncertaintyResult, TokenUncertaintyResult.from_dict(raw_metrics)
        )
    else:
        metrics = raw_metrics
    if metrics is None:
        if is_baseline:
            return 1.0
        logger.warning("Quality measurement failed/timed out — applying max penalty")
        return QUALITY_GATE_CLIFF_PENALTY

    if is_baseline or ctx.quality_baseline is None:
        ctx.quality_baseline = metrics
        logger.info(
            "Baseline: %d uncertain tokens (of %d), tail-20%% avg: %.3f",
            metrics.uncertain_count,
            metrics.total_tokens,
            metrics.tail_avg,
        )
        return 1.0

    # Convert legacy dict baseline to typed dataclass.
    # This path is hit when test mocks or older serialized state set
    # quality_baseline as a plain dict instead of TokenUncertaintyResult.
    if isinstance(ctx.quality_baseline, dict):
        ctx.quality_baseline = cast(
            TokenUncertaintyResult,
            TokenUncertaintyResult.from_dict(ctx.quality_baseline),
        )

    # Signal 1: uncertain token count increase
    # When baseline has very few uncertain tokens, use a floor
    # based on total token count to avoid extreme sensitivity
    # (e.g., going from 0->3 out of 1698 shouldn't be a cliff)
    assert ctx.quality_baseline is not None
    base_uc = ctx.quality_baseline.uncertain_count
    uc_floor = max(1, int(ctx.quality_baseline.total_tokens * 0.01))  # 1% floor
    base_uc = max(base_uc, uc_floor)
    uc_increase = (metrics.uncertain_count - base_uc) / base_uc

    # Signal 2: tail-20% logprob degradation (more negative = worse)
    base_tail = ctx.quality_baseline.tail_avg
    if base_tail < 0:
        tail_increase = (base_tail - metrics.tail_avg) / abs(
            base_tail
        )  # positive = degraded
    else:
        tail_increase = 0.0

    # Use the worse signal
    degradation = max(uc_increase, tail_increase)

    if degradation <= 0:
        quality_factor = 1.0
    elif degradation <= QUALITY_GATE_CEILING:
        # 0% to 15%: gentle slope from 1.0 → 0.85
        penalty_range = 1.0 - QUALITY_GATE_SOFT_PENALTY
        quality_factor = 1.0 - (degradation / QUALITY_GATE_CEILING) * penalty_range
    elif degradation <= QUALITY_GATE_CLIFF:
        # 15% to 30%: steep cliff from 0.85 → 0.1
        t = (degradation - QUALITY_GATE_CEILING) / (
            QUALITY_GATE_CLIFF - QUALITY_GATE_CEILING
        )
        quality_factor = QUALITY_GATE_SOFT_PENALTY - t * (
            QUALITY_GATE_SOFT_PENALTY - QUALITY_GATE_CLIFF_PENALTY
        )
    else:
        quality_factor = QUALITY_GATE_CLIFF_PENALTY

    logger.info(
        "Uncertain: %d (baseline: %d, %+.0f%%)"
        " | Tail: %.3f (baseline: %.3f, %+.0f%%)"
        " | factor: %.2f",
        metrics.uncertain_count,
        ctx.quality_baseline.uncertain_count,
        uc_increase * 100,
        metrics.tail_avg,
        ctx.quality_baseline.tail_avg,
        tail_increase * 100,
        quality_factor,
    )
    return quality_factor
