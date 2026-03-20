"""Tests for search_callbacks.py and search_display.py — the split search modules.

Uses MagicMock for studies, trials, and progress bars.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import optuna
import pytest

from tps_pro.search._callbacks import (
    GPStoppingCallback,
    ProgressBarUpdateCallback,
    _encode_param,
    _expected_improvement,
    safe_best_value,
    trial_scalar_value,
)
from tps_pro.search._display import (
    close_phase_pbar,
    create_phase_pbar,
    print_param_importance,
)
from tps_pro.search._display import (
    pbar_state as _pbar_state,
)

# ===================================================================
# _encode_param
# ===================================================================


@pytest.mark.unit
def test_encode_param_categorical():
    dist = optuna.distributions.CategoricalDistribution(choices=["a", "b", "c"])
    assert _encode_param("a", dist) == pytest.approx(0.0)
    assert _encode_param("b", dist) == pytest.approx(0.5)
    assert _encode_param("c", dist) == pytest.approx(1.0)


@pytest.mark.unit
def test_encode_param_categorical_single():
    """Single-choice categorical -> 0.0 (no divide by zero)."""
    dist = optuna.distributions.CategoricalDistribution(choices=["only"])
    assert _encode_param("only", dist) == pytest.approx(0.0)


@pytest.mark.unit
def test_encode_param_int():
    dist = optuna.distributions.IntDistribution(low=0, high=10)
    assert _encode_param(0, dist) == pytest.approx(0.0)
    assert _encode_param(5, dist) == pytest.approx(0.5)
    assert _encode_param(10, dist) == pytest.approx(1.0)


@pytest.mark.unit
def test_encode_param_float():
    dist = optuna.distributions.FloatDistribution(low=0.0, high=1.0)
    assert _encode_param(0.0, dist) == pytest.approx(0.0)
    assert _encode_param(0.5, dist) == pytest.approx(0.5)
    assert _encode_param(1.0, dist) == pytest.approx(1.0)


@pytest.mark.unit
def test_encode_param_unknown_distribution():
    """Unknown distribution type returns 0.5 fallback."""
    dist = MagicMock()
    assert _encode_param(42, dist) == pytest.approx(0.5)


# ===================================================================
# _expected_improvement
# ===================================================================


@pytest.mark.unit
def test_expected_improvement_shape():
    """EI output should have same shape as input mu/sigma."""
    mu = np.array([1.0, 2.0, 3.0])
    sigma = np.array([0.5, 0.5, 0.5])
    ei = _expected_improvement(mu, sigma, best_y=2.0)
    assert ei.shape == (3,)


@pytest.mark.unit
def test_expected_improvement_zero_sigma():
    """When sigma=0, EI should be 0 (no uncertainty)."""
    mu = np.array([5.0])
    sigma = np.array([0.0])
    ei = _expected_improvement(mu, sigma, best_y=3.0)
    assert ei[0] == pytest.approx(0.0)


@pytest.mark.unit
def test_expected_improvement_positive_when_mu_above_best():
    """When mu > best_y and sigma > 0, EI should be positive."""
    mu = np.array([10.0])
    sigma = np.array([1.0])
    ei = _expected_improvement(mu, sigma, best_y=5.0)
    assert ei[0] > 0.0


@pytest.mark.unit
def test_expected_improvement_scalar_inputs():
    """Scalar mu/sigma should work (wrapped by atleast_1d)."""
    ei = _expected_improvement(5.0, 1.0, best_y=3.0)
    assert ei.shape == (1,)
    assert ei[0] > 0.0


# ===================================================================
# safe_best_value / trial_scalar_value
# ===================================================================


@pytest.mark.unit
def testsafe_best_value_returns_none_on_empty_study():
    study = MagicMock()
    study.best_value = property(lambda self: (_ for _ in ()).throw(ValueError))
    # Simulate ValueError being raised
    type(study).best_value = property(lambda self: (_ for _ in ()).throw(ValueError))
    assert safe_best_value(study) is None


@pytest.mark.unit
def testtrial_scalar_value_single_objective():
    trial = MagicMock()
    trial.value = 42.0
    assert trial_scalar_value(trial) == 42.0


@pytest.mark.unit
def testtrial_scalar_value_multi_objective():
    """When trial.value raises RuntimeError, fall back to values[0]."""
    trial = MagicMock()
    type(trial).value = property(lambda self: (_ for _ in ()).throw(RuntimeError))
    trial.values = [10.0, 20.0]
    assert trial_scalar_value(trial) == 10.0


# ===================================================================
# ProgressBarUpdateCallback
# ===================================================================


@pytest.mark.unit
def test_tqdm_update_callback_updates_pbar():
    """ProgressBarUpdateCallback increments the active tracker count."""
    from types import SimpleNamespace

    tracker = SimpleNamespace(total=10, desc="test", count=0, current=None)
    original = _pbar_state.current
    try:
        _pbar_state.current = tracker
        cb = ProgressBarUpdateCallback()
        cb(MagicMock(), MagicMock())
        assert tracker.count == 1
    finally:
        _pbar_state.current = original


@pytest.mark.unit
def test_tqdm_update_callback_no_bar():
    """ProgressBarUpdateCallback does nothing when no active bar."""
    original = _pbar_state.current
    try:
        _pbar_state.current = None
        cb = ProgressBarUpdateCallback()
        # Should not raise
        cb(MagicMock(), MagicMock())
    finally:
        _pbar_state.current = original


# ===================================================================
# GPStoppingCallback — patience tracking
# ===================================================================


@pytest.mark.unit
def test_gp_stopping_patience_increments():
    """Patience counter increments when best value doesn't change."""
    cb = GPStoppingCallback(patience_fallback=5, min_trials_before_stop=0)

    study = MagicMock()
    type(study).best_value = property(lambda self: 10.0)
    study.trials = []

    trial = MagicMock()

    # First call sets _best_value to 10.0
    cb(study, trial)
    assert cb._trials_without_improvement == 0

    # Second call: same best_value -> patience increments
    cb(study, trial)
    assert cb._trials_without_improvement == 1


@pytest.mark.unit
def test_gp_stopping_patience_resets_on_improvement():
    """Patience resets when best value improves."""
    cb = GPStoppingCallback(patience_fallback=5, min_trials_before_stop=0)

    study = MagicMock()
    trial = MagicMock()
    study.trials = []

    # First call: best_value = 10
    type(study).best_value = property(lambda self: 10.0)
    cb(study, trial)
    assert cb._trials_without_improvement == 0

    # Same value -> increment
    cb(study, trial)
    assert cb._trials_without_improvement == 1

    # Improvement -> reset
    type(study).best_value = property(lambda self: 15.0)
    cb(study, trial)
    assert cb._trials_without_improvement == 0


@pytest.mark.unit
def test_gp_stopping_stops_after_patience_exhausted():
    """study.stop() is called when patience is exhausted."""
    patience = 3
    cb = GPStoppingCallback(patience_fallback=patience, min_trials_before_stop=0)

    study = MagicMock()
    type(study).best_value = property(lambda self: 10.0)
    # Enough completed trials to pass min_trials_before_stop=0
    study.trials = []
    trial = MagicMock()

    for _ in range(patience + 1):
        cb(study, trial)

    study.stop.assert_called_once()


# ===================================================================
# create_phase_pbar / close_phase_pbar
# ===================================================================


@pytest.mark.unit
def test_create_phase_pbar_returns_bar():
    """create_phase_pbar returns a SimpleNamespace tracker."""
    bar = create_phase_pbar(total=10, desc="test")
    assert bar is not None
    assert bar.total == 10
    assert bar.desc == "test"
    assert bar.count == 0
    _pbar_state.current = None


@pytest.mark.unit
def test_create_phase_pbar_sets_pbar_state():
    """create_phase_pbar sets _pbar_state.current."""
    original = _pbar_state.current
    try:
        bar = create_phase_pbar(total=5, desc="test")
        assert _pbar_state.current is bar
    finally:
        _pbar_state.current = original


@pytest.mark.unit
def test_close_phase_pbar_clears_state():
    """close_phase_pbar sets _pbar_state.current to None."""
    from types import SimpleNamespace

    tracker = SimpleNamespace(total=10, desc="test", count=0, current=None)
    _pbar_state.current = tracker

    close_phase_pbar()

    assert _pbar_state.current is None


@pytest.mark.unit
def test_close_phase_pbar_safe_when_none():
    """close_phase_pbar does nothing when no active bar."""
    _pbar_state.current = None
    # Should not raise
    close_phase_pbar()
    assert _pbar_state.current is None


# ===================================================================
# print_param_importance
# ===================================================================


@pytest.mark.unit
def test_print_param_importance_too_few_trials():
    """Returns empty dict when fewer than 10 completed trials."""
    study = optuna.create_study(direction="maximize")
    for i in range(5):
        study.add_trial(
            optuna.trial.create_trial(
                params={"x": float(i)},
                distributions={"x": optuna.distributions.FloatDistribution(0, 10)},
                values=[float(i + 1)],
            )
        )

    result = print_param_importance(study)
    assert result == {}


@pytest.mark.unit
def test_print_param_importance_with_mock_study():
    """Returns importance dict when optuna.importance succeeds."""
    study = optuna.create_study(direction="maximize")
    for i in range(15):
        study.add_trial(
            optuna.trial.create_trial(
                params={"param_a": float(i), "param_b": float(i * 2)},
                distributions={
                    "param_a": optuna.distributions.FloatDistribution(0, 20),
                    "param_b": optuna.distributions.FloatDistribution(0, 40),
                },
                values=[float(i + 1)],
            )
        )

    mock_importances = {"param_a": 0.6, "param_b": 0.4}
    with patch(
        "optuna.importance.get_param_importances", return_value=mock_importances
    ):
        result = print_param_importance(study)

    assert result == mock_importances
