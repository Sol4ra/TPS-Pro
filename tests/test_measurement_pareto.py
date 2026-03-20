"""Tests for measurement.py Pareto/multi-objective helpers.

Covers: get_best_trial, extract_pareto_front, print_pareto_front.
These were flagged as untested by the reviewer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tps_pro.measurement import (
    extract_pareto_front,
    get_best_trial,
    print_pareto_front,
)
from tps_pro.state import AppContext


def _make_trial(values, params=None, state=None):
    """Build a mock FrozenTrial with given objective values."""
    import optuna

    t = MagicMock()
    t.values = values
    t.params = params or {}
    t.state = state or optuna.trial.TrialState.COMPLETE
    return t


# ===================================================================
# get_best_trial — single-objective mode
# ===================================================================


@pytest.mark.unit
class TestGetBestTrialSingleObjective:
    def test_returns_study_best_trial(self):
        """In single-objective mode, returns study.best_trial directly."""
        ctx = AppContext()
        study = MagicMock()
        expected = _make_trial([50.0])
        study.best_trial = expected

        with patch("tps_pro.state.config", {"pareto": False}):
            result = get_best_trial(ctx, study)

        assert result is expected


# ===================================================================
# get_best_trial — Pareto (multi-objective) mode
# ===================================================================


@pytest.mark.unit
class TestGetBestTrialPareto:
    def test_returns_highest_tps_from_pareto_front(self):
        """In Pareto mode, returns the trial with the highest TPS (objective 0)."""
        ctx = AppContext()
        t_low = _make_trial([30.0, -4000, 0.9])
        t_high = _make_trial([80.0, -6000, 0.95])
        study = MagicMock()
        study.best_trials = [t_low, t_high]

        with patch("tps_pro.state.config", {"pareto": True}):
            result = get_best_trial(ctx, study)

        assert result is t_high

    def test_fallback_when_best_trials_empty(self):
        """Falls back to max of completed trials when Pareto front is empty."""
        import optuna

        ctx = AppContext()
        study = MagicMock()
        study.best_trials = []
        t1 = _make_trial([40.0, -3000, 1.0], state=optuna.trial.TrialState.COMPLETE)
        t2 = _make_trial([60.0, -5000, 0.8], state=optuna.trial.TrialState.COMPLETE)
        study.trials = [t1, t2]

        with patch("tps_pro.state.config", {"pareto": True}):
            result = get_best_trial(ctx, study)

        assert result is t2  # highest TPS

    def test_fallback_when_best_trials_raises(self):
        """Falls back when best_trials raises RuntimeError."""
        import optuna

        ctx = AppContext()
        study = MagicMock()
        type(study).best_trials = property(
            lambda self: (_ for _ in ()).throw(RuntimeError)
        )
        t1 = _make_trial([40.0], state=optuna.trial.TrialState.COMPLETE)
        study.trials = [t1]

        with patch("tps_pro.state.config", {"pareto": True}):
            result = get_best_trial(ctx, study)

        assert result is t1


# ===================================================================
# extract_pareto_front
# ===================================================================


@pytest.mark.unit
class TestExtractParetoFront:
    def test_returns_sorted_by_tps_descending(self):
        """Pareto trials are sorted by TPS (objective 0) descending."""
        t1 = _make_trial([30.0, -4000, 0.9])
        t2 = _make_trial([80.0, -6000, 0.95])
        t3 = _make_trial([50.0, -5000, 0.85])
        study = MagicMock()
        study.best_trials = [t1, t2, t3]

        result = extract_pareto_front(study)

        assert result == [t2, t3, t1]

    def test_empty_when_best_trials_raises(self):
        """Returns empty list when study.best_trials raises."""
        study = MagicMock()
        type(study).best_trials = property(
            lambda self: (_ for _ in ()).throw(RuntimeError)
        )

        result = extract_pareto_front(study)
        assert result == []

    def test_single_trial(self):
        """Single trial is returned as a list of one."""
        t = _make_trial([42.0, -3000, 1.0])
        study = MagicMock()
        study.best_trials = [t]

        result = extract_pareto_front(study)
        assert result == [t]


# ===================================================================
# print_pareto_front
# ===================================================================


@pytest.mark.unit
class TestPrintParetoFront:
    def test_empty_list_warns(self, caplog):
        """Empty Pareto front triggers a warning."""
        import logging

        with caplog.at_level(logging.WARNING):
            print_pareto_front([])
        assert "No Pareto-optimal" in caplog.text

    def test_prints_table_rows(self, caplog):
        """Prints one row per Pareto trial."""
        import logging

        t1 = _make_trial(
            [80.0, -4000, 0.95], params={"threads": 8, "kv_cache_type": "q8_0"}
        )
        t2 = _make_trial([50.0, -3000, 0.90], params={"threads": 4, "batch_size": 512})
        with caplog.at_level(logging.INFO):
            print_pareto_front([t1, t2])
        assert "80.0" in caplog.text
        assert "50.0" in caplog.text

    def test_prints_key_params(self, caplog):
        """Key parameters (threads, kv, batch, fa, draft) appear in output."""
        import logging

        t = _make_trial(
            [60.0, -5000, 0.88],
            params={"threads": 12, "flash_attn": True, "draft_max": 5},
        )
        with caplog.at_level(logging.INFO):
            print_pareto_front([t])
        assert "t=12" in caplog.text
        assert "fa=True" in caplog.text
        assert "draft=5" in caplog.text
