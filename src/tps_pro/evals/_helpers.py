"""Shared helpers for evaluation modules."""

from __future__ import annotations

__all__ = ["quality_factor_curve"]


def quality_factor_curve(
    value: float | None,
    threshold: float,
    hard_fail: float,
    *,
    no_data_value: float = 1.0,
) -> float:
    """Convert a degradation metric to a quality factor (0.0 to 1.0).

    Implements a 3-tier penalty curve used by both PPL and KL-divergence
    quality scoring:

      - value <= 0          -> 1.0  (no degradation)
      - 0 < value <= threshold -> gentle slope 1.0 -> 0.85
      - threshold < value <= hard_fail -> steep cliff 0.85 -> 0.1
      - value > hard_fail   -> 0.1  (floor)

    Args:
        value: The degradation metric (e.g., fractional PPL increase or KL-div).
            None or non-positive values return *no_data_value*.
        threshold: Soft-penalty boundary (maps to quality factor 0.85).
        hard_fail: Hard-fail boundary (maps to quality factor 0.1).
        no_data_value: Value to return when *value* is None or <= 0.

    Returns:
        Quality factor between 0.1 and 1.0.
    """
    if value is None or value <= 0:
        return no_data_value
    if value <= threshold:
        return 1.0 - 0.15 * (value / threshold)
    if value <= hard_fail:
        t = (value - threshold) / (hard_fail - threshold)
        return 0.85 - t * 0.75
    return 0.1
