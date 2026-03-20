"""Direct-import unit tests for search/_callbacks.py.

test_search_split.py and test_search_full.py already test callback functions
extensively. This file ensures top-level direct imports for coverage tooling.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import optuna
import pytest

from tps_pro.search._callbacks import (
    GPStoppingCallback,
    ProgressBarUpdateCallback,
    _encode_param,
    _expected_improvement,
    trial_scalar_value,
)


@pytest.mark.unit
class TestTrialScalarValue:
    def test_single_objective_trial(self):
        """trial_scalar_value should return .value for single-objective trials."""
        trial = MagicMock()
        trial.value = 42.5
        assert trial_scalar_value(trial) == 42.5

    def test_multi_objective_trial_fallback(self):
        """trial_scalar_value should fall back to .values[0] for multi-objective."""
        trial = MagicMock()
        trial.value = property(lambda self: (_ for _ in ()).throw(RuntimeError))
        type(trial).value = property(lambda self: (_ for _ in ()).throw(RuntimeError))
        trial.values = [10.0, 20.0]
        result = trial_scalar_value(trial)
        assert result == 10.0


@pytest.mark.unit
class TestEncodeParam:
    def test_categorical_distribution(self):
        """_encode_param should encode categorical as index / (len-1)."""
        dist = optuna.distributions.CategoricalDistribution(["a", "b", "c"])
        assert _encode_param("a", dist) == pytest.approx(0.0)
        assert _encode_param("c", dist) == pytest.approx(1.0)

    def test_int_distribution(self):
        """_encode_param should normalize int to [0, 1] range."""
        dist = optuna.distributions.IntDistribution(0, 10)
        assert _encode_param(0, dist) == pytest.approx(0.0)
        assert _encode_param(10, dist) == pytest.approx(1.0)
        assert _encode_param(5, dist) == pytest.approx(0.5)

    def test_float_distribution(self):
        """_encode_param should normalize float to [0, 1] range."""
        dist = optuna.distributions.FloatDistribution(0.0, 1.0)
        assert _encode_param(0.0, dist) == pytest.approx(0.0)
        assert _encode_param(1.0, dist) == pytest.approx(1.0)


@pytest.mark.unit
class TestExpectedImprovement:
    def test_returns_positive_for_improvement(self):
        """EI should be positive when mu > best_y."""
        mu = np.array([2.0])
        sigma = np.array([1.0])
        ei = _expected_improvement(mu, sigma, best_y=1.0)
        assert np.all(ei > 0)

    def test_zero_sigma_returns_zero(self):
        """EI should be ~0 when sigma is 0."""
        mu = np.array([2.0])
        sigma = np.array([0.0])
        ei = _expected_improvement(mu, sigma, best_y=1.0)
        assert np.all(ei >= 0)


@pytest.mark.unit
class TestCallbackInstantiation:
    def test_gp_stopping_callback_init(self):
        """GPStoppingCallback should be instantiable."""
        cb = GPStoppingCallback(patience_fallback=5, min_trials=10)
        assert cb is not None

    def test_tqdm_update_callback_init(self):
        """ProgressBarUpdateCallback should be instantiable."""
        cb = ProgressBarUpdateCallback()
        assert cb is not None
