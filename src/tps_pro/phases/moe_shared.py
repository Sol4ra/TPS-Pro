"""Shared helpers for MoE sweep phases (middle-out sweep, retesting, histograms)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..constants import EARLY_STOP_RATIO

logger = logging.getLogger(__name__)

__all__ = [
    "_retest_neighbors",
    "_run_middle_out_sweep",
]


def _run_middle_out_sweep(
    up_range: list[int],
    down_range: list[int],
    test_fn: Callable[[int], float],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Execute a middle-out sweep, stopping each direction at 50% of best.

    Args:
        up_range: Values to sweep upward from center.
        down_range: Values to sweep downward from center.
        test_fn: Callable(value) -> score for each value.
        state: Mutable dict with 'best_score' and 'best_val' keys.

    Returns:
        Updated state dict.
    """
    up_stopped = False
    down_stopped = False
    up_idx = 0
    down_idx = 0

    while not (up_stopped and down_stopped):
        if not up_stopped and up_idx < len(up_range):
            score = test_fn(up_range[up_idx])
            up_idx += 1
            if score >= state["best_score"]:
                state["best_val"] = up_range[up_idx - 1]
                state["best_score"] = score
            if (
                state["best_score"] > 0
                and score < state["best_score"] * EARLY_STOP_RATIO
                and up_idx > 2  # noqa: PLR2004
            ):
                logger.info(
                    "\u2191 Upward direction stopped (score dropped below 50% of best)"
                )
                up_stopped = True
        else:
            up_stopped = True

        if not down_stopped and down_idx < len(down_range):
            score = test_fn(down_range[down_idx])
            down_idx += 1
            if score >= state["best_score"]:
                state["best_val"] = down_range[down_idx - 1]
                state["best_score"] = score
            if (
                state["best_score"] > 0
                and score < state["best_score"] * EARLY_STOP_RATIO
                and down_idx > 2  # noqa: PLR2004
            ):
                logger.info(
                    "\u2193 Downward direction stopped"
                    " (score dropped below 50%% of best)"
                )
                down_stopped = True
        else:
            down_stopped = True

    return state


def _retest_neighbors(
    results_by_val: dict[int, Any],
    best_val: int,
    test_fn: Callable[..., float],
    retest_range: int = 2,
) -> None:
    """Re-test best +/-N neighbors with forced 3 runs.

    Args:
        results_by_val: Dict of value -> result dict from the sweep.
        best_val: Current best value to center the retest around.
        test_fn: Callable(value, force_3runs=True) -> score.
        retest_range: How many neighbors on each side to retest.
    """
    retests = [
        best_val + offset
        for offset in range(-retest_range, retest_range + 1)
        if (best_val + offset) in results_by_val
    ]

    logger.info(
        "Re-testing best \u00b1%s neighbors (%s values) with fresh 3 runs...",
        retest_range,
        len(retests),
    )
    for val in retests:
        test_fn(val, force_3runs=True)
