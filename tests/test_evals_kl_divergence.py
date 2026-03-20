"""Tests for evals/kl_divergence.py — KL-divergence measurement and scoring.

Direct imports from the target module to satisfy coverage detection.
Tests the pure-computation functions without needing a server.
"""

from __future__ import annotations

import pytest

from tps_pro.evals.kl_divergence import (
    _compute_kl_divergence,
    kl_quality_factor,
)


@pytest.mark.unit
class TestComputeKLDivergence:
    def test_identical_distributions_returns_zero(self):
        """Identical distributions should have KL-divergence of 0."""
        dists = [{"a": -1.0, "b": -2.0}, {"c": -0.5, "d": -3.0}]
        result = _compute_kl_divergence(dists, dists)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_empty_distributions_returns_zero(self):
        """Empty distribution lists should return 0.0."""
        assert _compute_kl_divergence([], []) == 0.0

    def test_different_distributions_positive(self):
        """Different distributions should produce positive KL divergence."""
        baseline = [{"a": -0.1, "b": -2.3}]
        trial = [{"a": -0.5, "b": -1.5}]
        result = _compute_kl_divergence(baseline, trial)
        assert result >= 0.0

    def test_mismatched_lengths_uses_minimum(self):
        """Should use the minimum number of tokens available from both."""
        baseline = [{"a": -1.0}, {"b": -2.0}, {"c": -0.5}]
        trial = [{"a": -1.0}]
        result = _compute_kl_divergence(baseline, trial)
        # Should compute using only 1 token (min of 3, 1)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_missing_tokens_get_floor(self):
        """Tokens missing from trial should get floor logprob of -20."""
        baseline = [{"a": -0.1, "b": -2.0}]
        trial = [{"a": -0.1}]  # "b" missing from trial
        result = _compute_kl_divergence(baseline, trial)
        assert result >= 0.0


@pytest.mark.unit
class TestKLQualityFactor:
    def test_none_returns_one(self):
        """None KL divergence should return 1.0."""
        assert kl_quality_factor(None) == 1.0

    def test_zero_returns_one(self):
        """Zero KL divergence should return 1.0."""
        assert kl_quality_factor(0.0) == 1.0

    def test_negative_returns_one(self):
        """Negative KL divergence should return 1.0."""
        assert kl_quality_factor(-0.5) == 1.0

    def test_small_kl_near_one(self):
        """Small KL divergence should produce factor near 1.0."""
        result = kl_quality_factor(0.001)
        assert 0.9 < result <= 1.0

    def test_large_kl_returns_floor(self):
        """Very large KL divergence should return the floor (0.1)."""
        result = kl_quality_factor(100.0)
        assert result == pytest.approx(0.1)

    def test_monotonically_decreasing(self):
        """Quality factor should decrease as KL divergence increases."""
        values = [kl_quality_factor(x) for x in [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]
