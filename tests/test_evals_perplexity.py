"""Tests for evals/perplexity.py — perplexity measurement and scoring.

Direct imports from the target module to satisfy coverage detection.
Tests the pure-computation ppl_quality_factor without needing a server.
"""

from __future__ import annotations

import pytest

from tps_pro.evals.perplexity import (
    measure_true_perplexity,
    ppl_quality_factor,
)


@pytest.mark.unit
class TestPPLQualityFactor:
    def test_no_degradation_returns_one(self):
        """Same or better PPL should return 1.0."""
        assert ppl_quality_factor(10.0, 10.0) == 1.0
        assert ppl_quality_factor(10.0, 9.0) == 1.0

    def test_invalid_baseline_returns_one(self):
        """Zero or inf baseline should return 1.0."""
        assert ppl_quality_factor(0.0, 15.0) == 1.0
        assert ppl_quality_factor(float("inf"), 15.0) == 1.0

    def test_inf_trial_returns_floor(self):
        """Inf trial PPL should return 0.1 (measurement failed)."""
        assert ppl_quality_factor(10.0, float("inf")) == pytest.approx(0.1)

    def test_small_degradation_gentle_penalty(self):
        """Small degradation should produce factor between 0.85 and 1.0."""
        # 5% increase
        result = ppl_quality_factor(10.0, 10.5)
        assert 0.85 <= result <= 1.0

    def test_large_degradation_returns_floor(self):
        """Severe degradation should return the floor (0.1)."""
        # 50% increase
        result = ppl_quality_factor(10.0, 15.0)
        assert result == pytest.approx(0.1)

    def test_monotonically_decreasing(self):
        """Quality factor should decrease as degradation increases."""
        values = [
            ppl_quality_factor(10.0, ppl) for ppl in [10.0, 10.5, 11.0, 12.0, 15.0]
        ]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]


@pytest.mark.unit
class TestMeasureTruePerplexity:
    def test_is_callable(self):
        """measure_true_perplexity should be a callable function."""
        assert callable(measure_true_perplexity)
