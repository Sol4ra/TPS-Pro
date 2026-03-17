"""Tests for GPSampler encode/decode and expected improvement from search.py.

Uses optuna distribution types directly (optuna is a required dependency)
and copies the static methods to avoid importing from the package.
"""
import unittest

import numpy as np
import optuna
from scipy.stats import norm


# ---------------------------------------------------------------------------
# GPSampler static methods — copied from search.py (pure functions)
# ---------------------------------------------------------------------------
def _encode_param(value, distribution):
    """Encode a parameter value to [0, 1] range."""
    if isinstance(distribution, optuna.distributions.CategoricalDistribution):
        choices = distribution.choices
        idx = choices.index(value) if value in choices else 0
        return idx / max(1, len(choices) - 1)
    elif isinstance(distribution, optuna.distributions.IntDistribution):
        low, high = distribution.low, distribution.high
        return (value - low) / max(1, high - low)
    elif isinstance(distribution, optuna.distributions.FloatDistribution):
        low, high = distribution.low, distribution.high
        return (value - low) / max(1e-8, high - low)
    return 0.5


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


def _expected_improvement(mu, sigma, best_y, xi=0.01):
    """Compute Expected Improvement."""
    with np.errstate(divide='ignore', invalid='ignore'):
        imp = mu - best_y - xi
        Z = np.where(sigma > 1e-8, imp / sigma, 0.0)
        ei = np.where(sigma > 1e-8, imp * norm.cdf(Z) + sigma * norm.pdf(Z), 0.0)
    return ei


# ===================================================================
# Tests
# ===================================================================

class TestEncodeDecodeRoundTrip(unittest.TestCase):
    """Tests for _encode_param() and _decode_param() round-trips."""

    def test_categorical_roundtrip(self):
        dist = optuna.distributions.CategoricalDistribution(choices=["f16", "q8_0", "q4_0"])
        for choice in dist.choices:
            encoded = _encode_param(choice, dist)
            self.assertGreaterEqual(encoded, 0.0)
            self.assertLessEqual(encoded, 1.0)
            decoded = _decode_param(encoded, dist)
            self.assertEqual(decoded, choice)

    def test_categorical_boundary_values(self):
        dist = optuna.distributions.CategoricalDistribution(choices=["a", "b", "c"])
        # First choice -> 0.0
        self.assertAlmostEqual(_encode_param("a", dist), 0.0)
        # Last choice -> 1.0
        self.assertAlmostEqual(_encode_param("c", dist), 1.0)
        # Decode boundaries
        self.assertEqual(_decode_param(0.0, dist), "a")
        self.assertEqual(_decode_param(1.0, dist), "c")

    def test_int_roundtrip(self):
        dist = optuna.distributions.IntDistribution(low=1, high=16, step=1)
        for val in [1, 4, 8, 12, 16]:
            encoded = _encode_param(val, dist)
            self.assertGreaterEqual(encoded, 0.0)
            self.assertLessEqual(encoded, 1.0)
            decoded = _decode_param(encoded, dist)
            self.assertEqual(decoded, val)

    def test_int_boundary_values(self):
        dist = optuna.distributions.IntDistribution(low=1, high=16, step=1)
        self.assertAlmostEqual(_encode_param(1, dist), 0.0)
        self.assertAlmostEqual(_encode_param(16, dist), 1.0)
        self.assertEqual(_decode_param(0.0, dist), 1)
        self.assertEqual(_decode_param(1.0, dist), 16)

    def test_int_with_step(self):
        """IntDistribution with step=4: values snap to grid."""
        dist = optuna.distributions.IntDistribution(low=0, high=16, step=4)
        # Encode 8 -> (8-0)/(16-0) = 0.5
        self.assertAlmostEqual(_encode_param(8, dist), 0.5)
        # Decode 0.5 -> raw=8, snap to step: round((8-0)/4)*4 + 0 = 8
        self.assertEqual(_decode_param(0.5, dist), 8)
        # Decode 0.3 -> raw=4.8, snap: round(4.8/4)*4 = round(1.2)*4 = 4
        self.assertEqual(_decode_param(0.3, dist), 4)

    def test_float_roundtrip(self):
        dist = optuna.distributions.FloatDistribution(low=0.0, high=1.0)
        for val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            encoded = _encode_param(val, dist)
            decoded = _decode_param(encoded, dist)
            self.assertAlmostEqual(decoded, val, places=6)

    def test_float_boundary_values(self):
        dist = optuna.distributions.FloatDistribution(low=0.0, high=1.0)
        self.assertAlmostEqual(_encode_param(0.0, dist), 0.0)
        self.assertAlmostEqual(_encode_param(1.0, dist), 1.0)
        self.assertAlmostEqual(_decode_param(0.0, dist), 0.0)
        self.assertAlmostEqual(_decode_param(1.0, dist), 1.0)

    def test_float_wide_range(self):
        dist = optuna.distributions.FloatDistribution(low=-100.0, high=100.0)
        encoded = _encode_param(0.0, dist)
        self.assertAlmostEqual(encoded, 0.5)
        decoded = _decode_param(0.5, dist)
        self.assertAlmostEqual(decoded, 0.0, places=4)

    def test_single_choice_categorical(self):
        """Single-choice categorical: encode and decode should both return the only choice."""
        dist = optuna.distributions.CategoricalDistribution(choices=["only"])
        encoded = _encode_param("only", dist)
        self.assertAlmostEqual(encoded, 0.0)
        decoded = _decode_param(0.0, dist)
        self.assertEqual(decoded, "only")
        # Any value should decode to the only choice
        decoded2 = _decode_param(1.0, dist)
        self.assertEqual(decoded2, "only")


class TestExpectedImprovement(unittest.TestCase):
    """Tests for _expected_improvement()."""

    def test_sigma_zero_returns_zero_ei(self):
        """When sigma=0, EI should be 0 (no uncertainty, no exploration value)."""
        mu = np.array([10.0, 20.0, 30.0])
        sigma = np.array([0.0, 0.0, 0.0])
        best_y = 15.0
        ei = _expected_improvement(mu, sigma, best_y)
        np.testing.assert_array_equal(ei, np.zeros(3))

    def test_positive_improvement_positive_ei(self):
        """When mu > best_y + xi and sigma > 0, EI should be positive."""
        mu = np.array([20.0])
        sigma = np.array([5.0])
        best_y = 10.0
        ei = _expected_improvement(mu, sigma, best_y)
        self.assertGreater(ei[0], 0.0)

    def test_mu_below_best_small_sigma(self):
        """When mu is well below best_y and sigma is small, EI should be near 0."""
        mu = np.array([5.0])
        sigma = np.array([0.1])
        best_y = 100.0
        ei = _expected_improvement(mu, sigma, best_y)
        self.assertAlmostEqual(ei[0], 0.0, places=3)

    def test_higher_sigma_higher_ei_for_same_mu(self):
        """More uncertainty (higher sigma) should give higher EI for same mu near best_y."""
        mu = np.array([10.0, 10.0])
        sigma = np.array([1.0, 5.0])
        best_y = 10.0
        ei = _expected_improvement(mu, sigma, best_y)
        self.assertGreater(ei[1], ei[0])

    def test_ei_shape_matches_input(self):
        """Output shape should match input shape."""
        mu = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sigma = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        ei = _expected_improvement(mu, sigma, 3.0)
        self.assertEqual(ei.shape, (5,))

    def test_ei_increases_with_mu(self):
        """EI should generally increase as mu increases (with same sigma)."""
        mu = np.array([5.0, 10.0, 15.0, 20.0])
        sigma = np.array([2.0, 2.0, 2.0, 2.0])
        best_y = 10.0
        ei = _expected_improvement(mu, sigma, best_y)
        # The last two values (mu=15, mu=20) should have higher EI than the first (mu=5)
        self.assertGreater(ei[3], ei[0])

    def test_xi_controls_exploration(self):
        """Higher xi should reduce EI (more conservative acquisition)."""
        mu = np.array([3.0])
        sigma = np.array([1.0])
        best_y = 2.5
        ei_low_xi = _expected_improvement(mu, sigma, best_y, xi=0.0)
        ei_high_xi = _expected_improvement(mu, sigma, best_y, xi=1.0)
        self.assertGreater(ei_low_xi[0], ei_high_xi[0])

    def test_ei_nonnegative(self):
        """EI should never be negative for any input."""
        rng = np.random.RandomState(42)
        mu = rng.randn(100) * 5
        sigma = np.abs(rng.randn(100)) + 0.01
        best_y = 3.0
        ei = _expected_improvement(mu, sigma, best_y)
        self.assertTrue(np.all(ei >= -1e-10))

    def test_mu_below_best_large_sigma_still_positive(self):
        """High uncertainty can give positive EI even when mean is below best."""
        mu = np.array([1.0])
        sigma = np.array([5.0])
        best_y = 2.0
        ei = _expected_improvement(mu, sigma, best_y)
        self.assertGreater(ei[0], 0.0)


class TestEncodeEdgeCases(unittest.TestCase):
    """Additional edge case tests for encode/decode."""

    def test_unknown_distribution_returns_half(self):
        """Unknown distribution type returns 0.5 as a safe default."""
        self.assertAlmostEqual(_encode_param(42, "not_a_distribution"), 0.5)

    def test_decode_clamps_categorical_high(self):
        """Encoded values > 1 should be clamped to last choice."""
        dist = optuna.distributions.CategoricalDistribution(["x", "y", "z"])
        self.assertEqual(_decode_param(1.5, dist), "z")

    def test_decode_clamps_categorical_low(self):
        """Encoded values < 0 should be clamped to first choice."""
        dist = optuna.distributions.CategoricalDistribution(["x", "y", "z"])
        self.assertEqual(_decode_param(-0.5, dist), "x")

    def test_int_step_snaps_to_grid(self):
        """Decoded int values must always be on the step grid."""
        dist = optuna.distributions.IntDistribution(0, 100, step=10)
        decoded = _decode_param(0.33, dist)
        self.assertEqual(decoded % 10, 0)

    def test_unknown_distribution_decode_passthrough(self):
        """Unknown distribution type passes the encoded value through."""
        self.assertAlmostEqual(_decode_param(0.7, "not_a_distribution"), 0.7)


if __name__ == "__main__":
    unittest.main()
