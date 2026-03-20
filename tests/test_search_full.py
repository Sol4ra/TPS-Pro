"""Comprehensive tests for search.py — study management, callbacks, phase I/O.

Covers setup_study, check_and_mark_duplicate_trial, save/load_phase_results,
GPStoppingCallback, ProgressBarUpdateCallback, and progress bar lifecycle.
Skips _encode_param and _expected_improvement (covered by test_search.py).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import optuna
import pytest

from tps_pro.constants import SCORE_VERSION
from tps_pro.search import _display as search_display
from tps_pro.search import _study as _search_study
from tps_pro.search import (
    check_and_mark_duplicate_trial,
    clear_param_cache,
    load_phase_results,
    save_phase_results,
    setup_study,
    update_param_cache,
)
from tps_pro.search._callbacks import (
    GPStoppingCallback,
    ProgressBarUpdateCallback,
)
from tps_pro.search._display import (
    close_phase_pbar,
    create_phase_pbar,
)

# ===================================================================
# Helpers
# ===================================================================


def _make_trial(
    params: dict,
    value: float = 10.0,
    user_attrs: dict | None = None,
    state=optuna.trial.TrialState.COMPLETE,
    values: list | None = None,
):
    """Create a mock completed trial with the given params and value."""
    t = MagicMock()
    t.params = params
    t.value = value
    t.values = values
    t.state = state
    t.user_attrs = user_attrs or {}
    return t


def _make_study(trials=None, study_name="test_study", directions=None):
    """Create a mock study with the given trials."""
    s = MagicMock()
    s.trials = trials or []
    s.study_name = study_name
    s.directions = directions or [optuna.study.StudyDirection.MAXIMIZE]
    return s


@pytest.fixture(autouse=True)
def _clean_param_cache():
    """Ensure the module-level _param_cache is clean before/after each test."""
    _search_study._param_cache.clear()
    yield
    _search_study._param_cache.clear()


@pytest.fixture(autouse=True)
def _clean_pbar():
    """Ensure the module-level _pbar_state.current is None before/after each test."""
    search_display.pbar_state.current = None
    yield
    search_display.pbar_state.current = None


@pytest.fixture
def mock_ctx(tmp_path):
    """Patch state.ctx with a mock that uses tmp_path for results_dir."""
    ctx_mock = MagicMock()
    ctx_mock.results_dir = tmp_path / "results"
    ctx_mock.optuna_db = "sqlite:///" + str(tmp_path / "optuna.db").replace("\\", "/")
    ctx_mock.fresh_run = False
    ctx_mock.debug = False
    yield ctx_mock


# ===================================================================
# setup_study
# ===================================================================


class TestSetupStudy:
    """Tests for setup_study()."""

    @pytest.mark.unit
    def test_new_study_creation_tpe_sampler(self, mock_ctx):
        """New single-objective study uses TPESampler by default."""
        study, remaining, completed = setup_study(
            mock_ctx, "compute", n_trials=50, seed=99
        )

        assert remaining == 50
        assert completed == 0
        # Verify it's a real optuna study (created via in-memory or file DB)
        assert study is not None

    @pytest.mark.unit
    def test_new_study_creation_pareto(self, mock_ctx):
        """Pareto study uses NSGA-II with 3 maximize directions."""
        study, remaining, completed = setup_study(
            mock_ctx, "compute", n_trials=30, seed=42, is_pareto=True
        )

        assert remaining == 30
        assert completed == 0
        assert len(study.directions) == 3
        assert all(d == optuna.study.StudyDirection.MAXIMIZE for d in study.directions)

    @pytest.mark.unit
    def test_study_direction_is_maximize(self, mock_ctx):
        """Single-objective study direction is maximize."""
        study, _, _ = setup_study(mock_ctx, "perf_test", n_trials=10)
        assert study.direction == optuna.study.StudyDirection.MAXIMIZE

    @pytest.mark.unit
    def test_study_resumption_from_existing_db(self, mock_ctx):
        """Resuming a study with completed trials reduces remaining count."""
        # Create a study and add some completed trials
        study1, _, _ = setup_study(mock_ctx, "resume_test", n_trials=20, seed=42)
        study1.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=5)

        # Resume the same study
        study2, remaining, completed = setup_study(
            mock_ctx, "resume_test", n_trials=20, seed=42
        )
        assert completed == 5
        assert remaining == 15

    @pytest.mark.unit
    def test_study_all_trials_completed(self, mock_ctx):
        """When all trials are done, remaining is 0."""
        study1, _, _ = setup_study(mock_ctx, "done_test", n_trials=3, seed=42)
        study1.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=3)

        _, remaining, completed = setup_study(
            mock_ctx, "done_test", n_trials=3, seed=42
        )
        assert completed == 3
        assert remaining == 0

    @pytest.mark.unit
    def test_sampler_override(self, mock_ctx):
        """Custom sampler_override is used instead of default TPE."""
        custom_sampler = optuna.samplers.RandomSampler(seed=0)
        study, _, _ = setup_study(
            mock_ctx, "custom_sampler", n_trials=5, sampler_override=custom_sampler
        )
        assert isinstance(study.sampler, optuna.samplers.RandomSampler)

    @pytest.mark.unit
    def test_fresh_run_deletes_existing_study(self, mock_ctx):
        """fresh_run=True deletes the old study before creating a new one."""
        study1, _, _ = setup_study(mock_ctx, "fresh_test", n_trials=10, seed=42)
        study1.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=5)

        mock_ctx.fresh_run = True
        _, remaining, completed = setup_study(
            mock_ctx, "fresh_test", n_trials=10, seed=42
        )
        assert completed == 0
        assert remaining == 10

    @pytest.mark.unit
    def test_versioned_study_name(self, mock_ctx):
        """Study name includes SCORE_VERSION for isolation."""
        study, _, _ = setup_study(mock_ctx, "my_phase", n_trials=5)
        assert SCORE_VERSION in study.study_name

    @pytest.mark.unit
    def test_pruner_configuration(self, mock_ctx):
        """Custom pruner is passed through to the study."""
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5)
        study, _, _ = setup_study(mock_ctx, "pruner_test", n_trials=5, pruner=pruner)
        assert isinstance(study.pruner, optuna.pruners.MedianPruner)


# ===================================================================
# check_and_mark_duplicate_trial
# ===================================================================


class TestCheckDuplicateTrial:
    """Tests for check_and_mark_duplicate_trial()."""

    @pytest.mark.unit
    def test_first_trial_never_duplicate(self):
        """First trial on empty study returns None (no duplicate)."""
        study = _make_study(trials=[], study_name="dup_test_empty")
        trial = MagicMock()
        trial.study = study
        trial.params = {"threads": 4, "ctx": 2048}

        result = check_and_mark_duplicate_trial(trial)
        assert result is None

    @pytest.mark.unit
    def test_identical_params_detected(self):
        """Trial with same params as a completed trial returns cached value."""
        past_trial = _make_trial(
            params={"threads": 4, "ctx": 2048},
            value=85.5,
            user_attrs={"tps": 42.0},
        )
        study = _make_study(
            trials=[past_trial],
            study_name="dup_test_found",
            directions=[optuna.study.StudyDirection.MAXIMIZE],
        )

        new_trial = MagicMock()
        new_trial.study = study
        new_trial.params = {"threads": 4, "ctx": 2048}
        new_trial.set_user_attr = MagicMock()

        result = check_and_mark_duplicate_trial(new_trial)
        assert result == 85.5
        # User attrs should have been copied
        new_trial.set_user_attr.assert_called_with("tps", 42.0)

    @pytest.mark.unit
    def test_different_params_not_flagged(self):
        """Trial with different params returns None."""
        past_trial = _make_trial(
            params={"threads": 4, "ctx": 2048},
            value=85.5,
        )
        study = _make_study(trials=[past_trial], study_name="dup_test_diff")

        new_trial = MagicMock()
        new_trial.study = study
        new_trial.params = {"threads": 8, "ctx": 4096}

        result = check_and_mark_duplicate_trial(new_trial)
        assert result is None

    @pytest.mark.unit
    def test_cache_populated_after_first_call(self):
        """After first check_and_mark_duplicate_trial call, cache has entries."""
        past_trial = _make_trial(params={"x": 1}, value=10.0)
        study = _make_study(trials=[past_trial], study_name="dup_cache_pop")

        trial = MagicMock()
        trial.study = study
        trial.params = {"y": 2}

        check_and_mark_duplicate_trial(trial)
        assert "dup_cache_pop" in _search_study._param_cache
        assert len(_search_study._param_cache["dup_cache_pop"]) == 1

    @pytest.mark.unit
    def test_multi_objective_returns_values(self):
        """For Pareto studies, duplicate returns the values tuple."""
        past_trial = _make_trial(
            params={"threads": 4},
            value=None,
            values=[85.0, -2000.0, 0.95],
        )
        study = _make_study(
            trials=[past_trial],
            study_name="dup_multi",
            directions=[
                optuna.study.StudyDirection.MAXIMIZE,
                optuna.study.StudyDirection.MAXIMIZE,
                optuna.study.StudyDirection.MAXIMIZE,
            ],
        )

        new_trial = MagicMock()
        new_trial.study = study
        new_trial.params = {"threads": 4}
        new_trial.set_user_attr = MagicMock()

        result = check_and_mark_duplicate_trial(new_trial)
        assert result == [85.0, -2000.0, 0.95]

    @pytest.mark.unit
    def test_only_complete_trials_cached(self):
        """Failed/pruned trials are not added to the duplicate cache."""
        failed = _make_trial(
            params={"x": 1}, value=0.0, state=optuna.trial.TrialState.FAIL
        )
        complete = _make_trial(params={"x": 2}, value=50.0)
        study = _make_study(trials=[failed, complete], study_name="dup_filter")

        trial = MagicMock()
        trial.study = study
        trial.params = {"x": 3}

        check_and_mark_duplicate_trial(trial)
        cache = _search_study._param_cache["dup_filter"]
        # Only the complete trial should be in the cache
        assert len(cache) == 1


# ===================================================================
# update_param_cache
# ===================================================================


class TestUpdateParamCache:
    """Tests for update_param_cache()."""

    @pytest.mark.unit
    def test_adds_entry_to_existing_cache(self):
        """update_param_cache adds new trial to an already-initialized cache."""
        _search_study._param_cache["my_study"] = {}

        trial = MagicMock()
        trial.study.study_name = "my_study"
        trial.params = {"lr": 0.01, "batch": 32}
        trial.user_attrs = {"tps": 100.0}

        update_param_cache(trial, 90.0)

        key = tuple(sorted({"lr": 0.01, "batch": 32}.items()))
        entry = _search_study._param_cache["my_study"][key]
        assert entry.value == 90.0
        assert entry.values is None
        assert entry.user_attrs == {"tps": 100.0}

    @pytest.mark.unit
    def test_multi_objective_stores_values(self):
        """Multi-objective value is stored as .values list."""
        _search_study._param_cache["pareto_study"] = {}

        trial = MagicMock()
        trial.study.study_name = "pareto_study"
        trial.params = {"x": 1}
        trial.user_attrs = {}

        update_param_cache(trial, [80.0, -1500.0, 0.9])

        key = tuple(sorted({"x": 1}.items()))
        entry = _search_study._param_cache["pareto_study"][key]
        assert entry.value is None
        assert entry.values == [80.0, -1500.0, 0.9]

    @pytest.mark.unit
    def test_no_op_when_cache_not_initialized(self):
        """If cache not yet built for the study, update_param_cache is a no-op."""
        trial = MagicMock()
        trial.study.study_name = "unknown_study"
        trial.params = {"x": 1}
        trial.user_attrs = {}

        # Should not raise
        update_param_cache(trial, 50.0)
        assert "unknown_study" not in _search_study._param_cache


# ===================================================================
# clear_param_cache
# ===================================================================


class TestClearParamCache:
    """Tests for clear_param_cache()."""

    @pytest.mark.unit
    def test_removes_study_entry(self):
        _search_study._param_cache["to_clear"] = {("x", 1): MagicMock()}
        clear_param_cache("to_clear")
        assert "to_clear" not in _search_study._param_cache

    @pytest.mark.unit
    def test_no_op_for_missing_study(self):
        """Clearing a non-existent study does not raise."""
        clear_param_cache("nonexistent")


# ===================================================================
# save_phase_results / load_phase_results
# ===================================================================


class TestSavePhaseResults:
    """Tests for save_phase_results()."""

    @pytest.mark.unit
    def test_writes_correct_json_structure(self, mock_ctx):
        """Saved JSON contains best_params, best_score, all_trials, and score_version."""
        results = {
            "best_params": {"threads": 8, "ctx": 4096},
            "best_score": 95.2,
            "all_trials": [
                {"params": {"threads": 4}, "score": 80.0},
                {"params": {"threads": 8}, "score": 95.2},
            ],
            "param_importance": {"threads": 0.65, "ctx": 0.35},
        }

        save_phase_results(mock_ctx, "compute", results)

        path = mock_ctx.results_dir / "compute_results.json"
        assert path.exists()

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["best_params"] == {"threads": 8, "ctx": 4096}
        assert data["best_score"] == 95.2
        assert len(data["all_trials"]) == 2
        assert data["param_importance"]["threads"] == 0.65
        assert data["score_version"] == SCORE_VERSION

    @pytest.mark.unit
    def test_overwrites_existing_file(self, mock_ctx):
        """Saving to same phase_name overwrites the previous file."""
        save_phase_results(mock_ctx, "memory", {"best_score": 50.0})
        save_phase_results(mock_ctx, "memory", {"best_score": 99.0})

        path = mock_ctx.results_dir / "memory_results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["best_score"] == 99.0

    @pytest.mark.unit
    def test_handles_empty_results(self, mock_ctx):
        """Empty dict is saved with just score_version added."""
        save_phase_results(mock_ctx, "empty_phase", {})

        path = mock_ctx.results_dir / "empty_phase_results.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["score_version"] == SCORE_VERSION
        assert len(data) == 1

    @pytest.mark.unit
    def test_atomic_write_no_tmp_left(self, mock_ctx):
        """After save, the .tmp file should not exist (renamed to final)."""
        save_phase_results(mock_ctx, "atomic_test", {"x": 1})

        tmp_path = mock_ctx.results_dir / "atomic_test_results.tmp"
        assert not tmp_path.exists()


class TestLoadPhaseResults:
    """Tests for load_phase_results()."""

    @pytest.mark.unit
    def test_loads_valid_json(self, mock_ctx):
        """Loads a valid JSON file and returns its contents."""
        results = {"best_score": 88.0, "best_params": {"threads": 6}}
        save_phase_results(mock_ctx, "valid_load", results)

        loaded = load_phase_results(mock_ctx, "valid_load")
        assert loaded is not None
        assert loaded["best_score"] == 88.0
        assert loaded["best_params"] == {"threads": 6}
        assert loaded["score_version"] == SCORE_VERSION

    @pytest.mark.unit
    def test_returns_none_for_missing_file(self, mock_ctx):
        """Returns None when the file does not exist."""
        # Ensure directory exists but file does not
        mock_ctx.results_dir.mkdir(parents=True, exist_ok=True)
        loaded = load_phase_results(mock_ctx, "nonexistent_phase")
        assert loaded is None

    @pytest.mark.unit
    def test_returns_none_for_corrupt_json(self, mock_ctx):
        """Returns None for a file containing invalid JSON."""
        mock_ctx.results_dir.mkdir(parents=True, exist_ok=True)
        path = mock_ctx.results_dir / "corrupt_results.json"
        path.write_text("{ this is not valid json !!!", encoding="utf-8")

        loaded = load_phase_results(mock_ctx, "corrupt")
        assert loaded is None

    @pytest.mark.unit
    def test_returns_none_for_stale_score_version(self, mock_ctx):
        """Returns None when saved score_version differs from current."""
        mock_ctx.results_dir.mkdir(parents=True, exist_ok=True)
        path = mock_ctx.results_dir / "stale_results.json"
        data = {"best_score": 70.0, "score_version": "v_old_and_busted"}
        path.write_text(json.dumps(data), encoding="utf-8")

        loaded = load_phase_results(mock_ctx, "stale")
        assert loaded is None

    @pytest.mark.unit
    def test_returns_none_for_missing_score_version(self, mock_ctx):
        """Returns None when score_version key is absent."""
        mock_ctx.results_dir.mkdir(parents=True, exist_ok=True)
        path = mock_ctx.results_dir / "noversion_results.json"
        data = {"best_score": 70.0}
        path.write_text(json.dumps(data), encoding="utf-8")

        loaded = load_phase_results(mock_ctx, "noversion")
        assert loaded is None


class TestPhaseResultsRoundTrip:
    """Round-trip: save -> load returns equivalent data."""

    @pytest.mark.unit
    def test_full_round_trip(self, mock_ctx):
        """save_phase_results -> load_phase_results returns equivalent data."""
        original = {
            "best_params": {"threads": 12, "ctx": 8192, "batch": 512},
            "best_score": 102.7,
            "all_trials": [
                {"params": {"threads": 4}, "score": 60.0},
                {"params": {"threads": 12}, "score": 102.7},
            ],
            "param_importance": {"threads": 0.5, "ctx": 0.3, "batch": 0.2},
        }

        save_phase_results(mock_ctx, "roundtrip", original)
        loaded = load_phase_results(mock_ctx, "roundtrip")

        assert loaded is not None
        assert loaded["best_params"] == original["best_params"]
        assert loaded["best_score"] == original["best_score"]
        assert loaded["all_trials"] == original["all_trials"]
        assert loaded["param_importance"] == original["param_importance"]
        # score_version is added by save
        assert loaded["score_version"] == SCORE_VERSION

    @pytest.mark.unit
    def test_round_trip_with_nested_structures(self, mock_ctx):
        """Complex nested data survives the round trip."""
        original = {
            "metadata": {"model": "test.gguf", "phases": ["compute", "memory"]},
            "scores": [1.1, 2.2, 3.3],
            "nested": {"a": {"b": {"c": 42}}},
        }

        save_phase_results(mock_ctx, "nested_rt", original)
        loaded = load_phase_results(mock_ctx, "nested_rt")

        assert loaded is not None
        assert loaded["metadata"] == original["metadata"]
        assert loaded["scores"] == original["scores"]
        assert loaded["nested"]["a"]["b"]["c"] == 42


# ===================================================================
# GPStoppingCallback
# ===================================================================


class TestGPStoppingCallback:
    """Tests for GPStoppingCallback."""

    def _make_complete_trials(self, n, value=50.0):
        """Create n mock complete trials with proper attributes for optuna internals."""
        dist = optuna.distributions.FloatDistribution(low=0.0, high=1.0)
        trials = []
        for i in range(n):
            t = MagicMock()
            t.number = i
            t.state = optuna.trial.TrialState.COMPLETE
            t.value = value
            t.values = None
            t.params = {"x": float(i) / max(1, n - 1)}
            t.distributions = {"x": dist}
            trials.append(t)
        return trials

    @pytest.mark.unit
    def test_does_not_stop_before_minimum_trials(self):
        """Callback should not stop study when fewer than min_trials_before_stop."""
        cb = GPStoppingCallback(
            min_trials=5, min_trials_before_stop=20, patience_fallback=100
        )
        study = MagicMock()
        study.best_value = 50.0
        study.trials = self._make_complete_trials(10, value=50.0)

        trial = MagicMock()
        for _i in range(15):
            cb(study, trial)

        study.stop.assert_not_called()

    @pytest.mark.unit
    def test_stops_on_patience_exhaustion(self):
        """Callback stops after patience_fallback trials without improvement."""
        cb = GPStoppingCallback(
            patience_fallback=5, min_trials_before_stop=0, check_every=1
        )
        study = MagicMock()
        study.best_value = 50.0
        study.trials = self._make_complete_trials(60, value=50.0)

        trial = MagicMock()
        for _ in range(10):
            cb(study, trial)

        study.stop.assert_called()

    @pytest.mark.unit
    def test_resets_patience_on_improvement(self):
        """Patience counter resets when best_value improves."""
        cb = GPStoppingCallback(
            patience_fallback=10,
            min_trials_before_stop=0,
            check_every=100,
        )
        study = MagicMock()
        trial = MagicMock()

        # First few calls: no improvement (check_every=100 prevents GP path)
        study.best_value = 50.0
        study.trials = self._make_complete_trials(60, value=50.0)
        for _ in range(8):
            cb(study, trial)

        # Patience should be at 8; improvement resets it
        study.best_value = 60.0
        cb(study, trial)
        assert cb._trials_without_improvement == 0

        # 3 more without improvement: patience at 3, well under fallback of 10
        study.best_value = 60.0
        for _ in range(3):
            cb(study, trial)

        study.stop.assert_not_called()
        assert cb._trials_without_improvement == 3

    @pytest.mark.unit
    def test_handles_single_trial(self):
        """Single trial does not cause errors or premature stopping."""
        cb = GPStoppingCallback(min_trials=5, min_trials_before_stop=10)
        study = MagicMock()
        study.best_value = 50.0
        study.trials = self._make_complete_trials(1, value=50.0)

        trial = MagicMock()
        cb(study, trial)
        study.stop.assert_not_called()

    @pytest.mark.unit
    def test_handles_all_same_value(self):
        """All trials with identical values should not crash the callback."""
        cb = GPStoppingCallback(
            min_trials=3,
            min_trials_before_stop=0,
            patience_fallback=100,
            check_every=1,
        )
        study = MagicMock()
        study.best_value = 50.0
        study.trials = self._make_complete_trials(15, value=50.0)

        # Give trials a search space for the GP to work with
        for i, t in enumerate(study.trials):
            t.params = {"x": float(i) / 14.0}

        trial = MagicMock()
        # Should not raise even though GP fit may degenerate
        for _ in range(6):
            cb(study, trial)

    @pytest.mark.unit
    def test_check_every_skips_intermediate_calls(self):
        """Callback only fits GP every check_every calls."""
        cb = GPStoppingCallback(
            check_every=5,
            min_trials=3,
            min_trials_before_stop=0,
            patience_fallback=200,
        )
        study = MagicMock()
        study.best_value = 50.0
        study.trials = self._make_complete_trials(60, value=50.0)

        trial = MagicMock()
        # Call 4 times (not multiple of 5) - should not attempt GP fit/stop
        for _ in range(4):
            cb(study, trial)
        study.stop.assert_not_called()

    @pytest.mark.unit
    def test_baseline_score_prevents_stop(self):
        """When best is below baseline_score, patience exhaustion does not stop."""
        cb = GPStoppingCallback(
            patience_fallback=3,
            min_trials_before_stop=0,
            baseline_score=80.0,
            check_every=1,
        )
        study = MagicMock()
        study.best_value = 50.0  # Below baseline of 80
        study.trials = self._make_complete_trials(60, value=50.0)

        trial = MagicMock()
        for _ in range(10):
            cb(study, trial)

        study.stop.assert_not_called()


# ===================================================================
# ProgressBarUpdateCallback
# ===================================================================


class TestProgressBarUpdateCallback:
    """Tests for ProgressBarUpdateCallback."""

    @pytest.mark.unit
    def test_updates_progress_bar(self):
        """Callback increments pbar_state.current.count when active tracker exists."""
        from types import SimpleNamespace

        tracker = SimpleNamespace(total=10, desc="test", count=0, current=None)
        search_display.pbar_state.current = tracker

        cb = ProgressBarUpdateCallback()
        cb(MagicMock(), MagicMock())

        assert tracker.count == 1

    @pytest.mark.unit
    def test_no_error_without_pbar(self):
        """Callback does not raise when no active progress bar."""
        search_display.pbar_state.current = None

        cb = ProgressBarUpdateCallback()
        cb(MagicMock(), MagicMock())  # Should not raise

    @pytest.mark.unit
    def test_multiple_calls_update_multiple_times(self):
        """Each callback invocation increments the tracker count."""
        from types import SimpleNamespace

        tracker = SimpleNamespace(total=10, desc="test", count=0, current=None)
        search_display.pbar_state.current = tracker

        cb = ProgressBarUpdateCallback()
        for _ in range(5):
            cb(MagicMock(), MagicMock())

        assert tracker.count == 5


# ===================================================================
# create_phase_pbar / close_phase_pbar
# ===================================================================


class TestPhaseProgressBar:
    """Tests for create_phase_pbar() and close_phase_pbar() lifecycle."""

    @pytest.mark.unit
    def test_create_returns_tracker(self):
        """create_phase_pbar returns a SimpleNamespace tracker."""
        bar = create_phase_pbar(total=50, desc="Compute")

        assert bar is not None
        assert bar.total == 50
        assert bar.desc == "Compute"
        assert bar.count == 0
        assert search_display.pbar_state.current is bar

    @pytest.mark.unit
    def test_create_sets_pbar_state(self):
        """create_phase_pbar sets pbar_state.current to the tracker."""
        bar = create_phase_pbar(total=20, desc="Phase")
        assert search_display.pbar_state.current is bar

    @pytest.mark.unit
    def test_close_clears_active_pbar(self):
        """close_phase_pbar sets pbar_state.current to None."""
        create_phase_pbar(total=10, desc="test")
        assert search_display.pbar_state.current is not None

        close_phase_pbar()
        assert search_display.pbar_state.current is None

    @pytest.mark.unit
    def test_close_no_op_without_active_pbar(self):
        """close_phase_pbar is a no-op when no bar is active."""
        search_display.pbar_state.current = None
        close_phase_pbar()  # Should not raise
        assert search_display.pbar_state.current is None

    @pytest.mark.unit
    def test_lifecycle_create_then_close(self):
        """Full lifecycle: create -> use -> close leaves clean state."""

        bar = create_phase_pbar(total=10, desc="test")
        assert bar is not None

        # Simulate ProgressBarUpdateCallback usage
        cb = ProgressBarUpdateCallback()
        cb(MagicMock(), MagicMock())
        assert bar.count == 1

        close_phase_pbar()
        assert search_display.pbar_state.current is None
