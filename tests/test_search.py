"""Tests for GPSampler encode/decode and expected improvement from search.py.

_encode_param and _expected_improvement are imported directly from search.py.
_decode_param is defined locally because it only exists in this test suite
(search.py does not expose a standalone decode function).
"""

import numpy as np
import optuna
import pytest

from tps_pro.search._callbacks import (
    _encode_param,
    _expected_improvement,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# _decode_param — test-only inverse of _encode_param (not in search.py)
# ---------------------------------------------------------------------------


def _decode_param(encoded, distribution):
    """Decode a [0, 1] value back to the parameter's original type/range."""
    if isinstance(distribution, optuna.distributions.CategoricalDistribution):
        choices = distribution.choices
        idx = int(round(encoded * (len(choices) - 1)))
        idx = max(0, min(idx, len(choices) - 1))
        return choices[idx]
    elif isinstance(distribution, optuna.distributions.IntDistribution):
        low, high = distribution.low, distribution.high
        step = distribution.step
        raw = low + encoded * (high - low)
        return int(round((raw - low) / step) * step + low)
    elif isinstance(distribution, optuna.distributions.FloatDistribution):
        low, high = distribution.low, distribution.high
        return low + encoded * (high - low)
    return encoded


# ===================================================================
# _encode_param / _decode_param round-trips
# ===================================================================


class TestEncodeDecodeRoundTrip:
    """Tests for _encode_param() and _decode_param() round-trips."""

    def test_categorical_roundtrip(self):
        dist = optuna.distributions.CategoricalDistribution(
            choices=["f16", "q8_0", "q4_0"]
        )
        for choice in dist.choices:
            encoded = _encode_param(choice, dist)
            assert 0.0 <= encoded <= 1.0
            decoded = _decode_param(encoded, dist)
            assert decoded == choice

    @pytest.mark.parametrize(
        "choice, expected_encoded",
        [("a", 0.0), ("c", 1.0)],
        ids=["first-choice", "last-choice"],
    )
    def test_categorical_boundary_values(self, choice, expected_encoded):
        dist = optuna.distributions.CategoricalDistribution(choices=["a", "b", "c"])
        assert _encode_param(choice, dist) == pytest.approx(expected_encoded)
        assert _decode_param(expected_encoded, dist) == choice

    def test_int_roundtrip(self):
        dist = optuna.distributions.IntDistribution(low=1, high=16, step=1)
        for val in [1, 4, 8, 12, 16]:
            encoded = _encode_param(val, dist)
            assert 0.0 <= encoded <= 1.0
            decoded = _decode_param(encoded, dist)
            assert decoded == val

    @pytest.mark.parametrize(
        "val, expected_encoded",
        [(1, 0.0), (16, 1.0)],
        ids=["low-bound", "high-bound"],
    )
    def test_int_boundary_values(self, val, expected_encoded):
        dist = optuna.distributions.IntDistribution(low=1, high=16, step=1)
        assert _encode_param(val, dist) == pytest.approx(expected_encoded)
        assert _decode_param(expected_encoded, dist) == val

    def test_int_with_step(self):
        """IntDistribution with step=4: values snap to grid."""
        dist = optuna.distributions.IntDistribution(low=0, high=16, step=4)
        assert _encode_param(8, dist) == pytest.approx(0.5)
        assert _decode_param(0.5, dist) == 8
        # Decode 0.3 -> raw=4.8, snap: round(4.8/4)*4 = 4
        assert _decode_param(0.3, dist) == 4

    def test_float_roundtrip(self):
        dist = optuna.distributions.FloatDistribution(low=0.0, high=1.0)
        for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            encoded = _encode_param(val, dist)
            decoded = _decode_param(encoded, dist)
            assert decoded == pytest.approx(val, abs=1e-6)

    @pytest.mark.parametrize(
        "val, expected_encoded",
        [(0.0, 0.0), (1.0, 1.0)],
        ids=["low-bound", "high-bound"],
    )
    def test_float_boundary_values(self, val, expected_encoded):
        dist = optuna.distributions.FloatDistribution(low=0.0, high=1.0)
        assert _encode_param(val, dist) == pytest.approx(expected_encoded)
        assert _decode_param(expected_encoded, dist) == pytest.approx(val)

    def test_float_wide_range(self):
        dist = optuna.distributions.FloatDistribution(low=-100.0, high=100.0)
        assert _encode_param(0.0, dist) == pytest.approx(0.5)
        assert _decode_param(0.5, dist) == pytest.approx(0.0, abs=1e-4)

    def test_single_choice_categorical(self):
        """Single-choice categorical: encode and decode should both return the only
        choice."""
        dist = optuna.distributions.CategoricalDistribution(choices=["only"])
        assert _encode_param("only", dist) == pytest.approx(0.0)
        assert _decode_param(0.0, dist) == "only"
        assert _decode_param(1.0, dist) == "only"


# ===================================================================
# _expected_improvement
# ===================================================================


class TestExpectedImprovement:
    """Tests for _expected_improvement()."""

    def test_sigma_zero_returns_zero_ei(self):
        """When sigma=0, EI should be 0."""
        mu = np.array([10.0, 20.0, 30.0])
        sigma = np.array([0.0, 0.0, 0.0])
        ei = _expected_improvement(mu, sigma, best_y=15.0)
        np.testing.assert_array_equal(ei, np.zeros(3))

    def test_positive_improvement_positive_ei(self):
        """When mu > best_y + xi and sigma > 0, EI should be positive."""
        ei = _expected_improvement(np.array([20.0]), np.array([5.0]), best_y=10.0)
        assert ei[0] > 0.0

    def test_mu_below_best_small_sigma(self):
        """When mu is well below best_y and sigma is small, EI should be near 0."""
        ei = _expected_improvement(np.array([5.0]), np.array([0.1]), best_y=100.0)
        assert ei[0] == pytest.approx(0.0, abs=1e-3)

    def test_higher_sigma_higher_ei_for_same_mu(self):
        """More uncertainty (higher sigma) should give higher EI for same mu near
        best_y."""
        mu = np.array([10.0, 10.0])
        sigma = np.array([1.0, 5.0])
        ei = _expected_improvement(mu, sigma, best_y=10.0)
        assert ei[1] > ei[0]

    def test_ei_shape_matches_input(self):
        """Output shape should match input shape."""
        mu = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sigma = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        ei = _expected_improvement(mu, sigma, best_y=3.0)
        assert ei.shape == (5,)

    def test_ei_increases_with_mu(self):
        """EI should generally increase as mu increases (with same sigma)."""
        mu = np.array([5.0, 10.0, 15.0, 20.0])
        sigma = np.array([2.0, 2.0, 2.0, 2.0])
        ei = _expected_improvement(mu, sigma, best_y=10.0)
        assert ei[3] > ei[0]

    def test_xi_controls_exploration(self):
        """Higher xi should reduce EI (more conservative acquisition)."""
        mu = np.array([3.0])
        sigma = np.array([1.0])
        ei_low = _expected_improvement(mu, sigma, best_y=2.5, xi=0.0)
        ei_high = _expected_improvement(mu, sigma, best_y=2.5, xi=1.0)
        assert ei_low[0] > ei_high[0]

    def test_ei_nonnegative(self):
        """EI should never be negative for any input."""
        rng = np.random.RandomState(42)
        mu = rng.randn(100) * 5
        sigma = np.abs(rng.randn(100)) + 0.01
        ei = _expected_improvement(mu, sigma, best_y=3.0)
        assert np.all(ei >= -1e-10)

    def test_mu_below_best_large_sigma_still_positive(self):
        """High uncertainty can give positive EI even when mean is below best."""
        ei = _expected_improvement(np.array([1.0]), np.array([5.0]), best_y=2.0)
        assert ei[0] > 0.0


# ===================================================================
# Edge cases
# ===================================================================


class TestEncodeEdgeCases:
    """Additional edge case tests for encode/decode."""

    def test_unknown_distribution_returns_half(self):
        """Unknown distribution type returns 0.5 as a safe default."""
        assert _encode_param(42, "not_a_distribution") == pytest.approx(0.5)

    def test_decode_clamps_categorical_high(self):
        """Encoded values > 1 should be clamped to last choice."""
        dist = optuna.distributions.CategoricalDistribution(["x", "y", "z"])
        assert _decode_param(1.5, dist) == "z"

    def test_decode_clamps_categorical_low(self):
        """Encoded values < 0 should be clamped to first choice."""
        dist = optuna.distributions.CategoricalDistribution(["x", "y", "z"])
        assert _decode_param(-0.5, dist) == "x"

    def test_int_step_snaps_to_grid(self):
        """Decoded int values must always be on the step grid."""
        dist = optuna.distributions.IntDistribution(0, 100, step=10)
        decoded = _decode_param(0.33, dist)
        assert decoded % 10 == 0

    def test_unknown_distribution_decode_passthrough(self):
        """Unknown distribution type passes the encoded value through."""
        assert _decode_param(0.7, "not_a_distribution") == pytest.approx(0.7)
