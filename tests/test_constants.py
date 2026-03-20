"""Tests for constants.py — key constant relationships and sanity checks."""

from __future__ import annotations

import pytest

from tps_pro.constants import (
    BIND_HOST,
    CONCURRENT_BASE_FACTOR,
    CONCURRENT_BONUS_WEIGHT,
    CV_MAX_RUNS,
    CV_MIN_RUNS,
    KL_DIV_HARD_FAIL,
    KL_DIV_THRESHOLD,
    PPL_DEGRADATION_FAIL,
    PPL_DEGRADATION_WARN,
    QUALITY_WEIGHT_CONFIDENCE,
    QUALITY_WEIGHT_CORRECTNESS,
    QUALITY_WEIGHT_EFFICIENCY,
    SCORE_PP_BASELINE,
    TTFT_BASELINE_MS,
    THERMAL_COOLDOWN_TARGET,
    THERMAL_THROTTLE_THRESHOLD,
    TTFT_FLOOR_MS,
    WEIGHT_GEN_TPS,
    WEIGHT_LARGE_TPS,
    WEIGHT_PP_COMPONENT,
    WEIGHT_TTFT_COMPONENT,
    WEIGHT_VRAM,
)


@pytest.mark.unit
class TestConstantRelationships:
    def test_kl_threshold_less_than_hard_fail(self):
        """KL_DIV_THRESHOLD must be strictly less than KL_DIV_HARD_FAIL."""
        assert KL_DIV_THRESHOLD < KL_DIV_HARD_FAIL

    def test_ppl_warn_less_than_fail(self):
        """PPL warning threshold must be below the hard fail threshold."""
        assert PPL_DEGRADATION_WARN < PPL_DEGRADATION_FAIL

    def test_cv_min_less_than_max(self):
        """CV_MIN_RUNS must be less than CV_MAX_RUNS."""
        assert CV_MIN_RUNS < CV_MAX_RUNS

    def test_thermal_cooldown_below_throttle(self):
        """Cooldown target must be below the throttle threshold."""
        assert THERMAL_COOLDOWN_TARGET < THERMAL_THROTTLE_THRESHOLD

    def test_quality_weights_sum_to_one(self):
        """Quality scoring weights should sum to 1.0."""
        total = (
            QUALITY_WEIGHT_CORRECTNESS
            + QUALITY_WEIGHT_CONFIDENCE
            + QUALITY_WEIGHT_EFFICIENCY
        )
        assert abs(total - 1.0) < 1e-9

    def test_full_scoring_weights_sum_to_one(self):
        """Full-mode scoring weights should sum to 1.0."""
        total = (
            WEIGHT_GEN_TPS
            + WEIGHT_LARGE_TPS
            + WEIGHT_PP_COMPONENT
            + WEIGHT_TTFT_COMPONENT
            + WEIGHT_VRAM
        )
        assert abs(total - 1.0) < 1e-9

    def test_concurrent_factors_sum_to_one(self):
        """Concurrent base + bonus should sum to 1.0."""
        assert abs(CONCURRENT_BASE_FACTOR + CONCURRENT_BONUS_WEIGHT - 1.0) < 1e-9

    def test_positive_baselines(self):
        """Score baselines must be positive."""
        assert TTFT_BASELINE_MS > 0
        assert SCORE_PP_BASELINE > 0

    def test_bind_host_is_localhost(self):
        """Bind host should be localhost."""
        assert BIND_HOST == "127.0.0.1"

    def test_ttft_floor_positive(self):
        """TTFT floor must be positive to prevent division by zero."""
        assert TTFT_FLOOR_MS > 0
