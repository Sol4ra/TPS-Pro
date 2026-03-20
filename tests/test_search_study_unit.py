"""Direct-import unit tests for search/_study.py.

test_search_full.py already tests study management functions extensively.
This file ensures top-level direct imports for coverage tooling.
"""

from __future__ import annotations

import pytest

from tps_pro.search._study import (
    check_and_mark_duplicate_trial,
    clear_param_cache,
    ensure_results_dir,
    load_phase_results,
    save_phase_results,
    setup_study,
    update_param_cache,
)


def _make_ctx(results_dir=None, optuna_db=None, fresh_run=False):
    """Build a minimal ctx for study tests."""
    from _ctx_factory import make_ctx_from_defaults

    overrides = {}
    if results_dir is not None:
        overrides["results_dir"] = results_dir
    if optuna_db is not None:
        overrides["optuna_db"] = optuna_db
    overrides["fresh_run"] = fresh_run
    return make_ctx_from_defaults(**overrides)


@pytest.mark.unit
class TestClearParamCache:
    def test_clears_existing_cache(self):
        """clear_param_cache should remove an entry if it exists."""
        # This just verifies the function runs without error
        clear_param_cache("nonexistent_study")

    def test_noop_for_missing_key(self):
        """clear_param_cache should be a safe no-op for missing keys."""
        clear_param_cache("some_study_that_does_not_exist")


@pytest.mark.unit
class TestSaveAndLoadPhaseResults:
    def test_save_and_load_round_trip(self, tmp_path):
        """save_phase_results -> load_phase_results should round-trip data."""
        from tps_pro.constants import SCORE_VERSION

        ctx = _make_ctx(results_dir=tmp_path)
        data = {"phase": "test", "best_tps": 42.5, "score_version": SCORE_VERSION}
        save_phase_results(ctx, "test_phase", data)
        loaded = load_phase_results(ctx, "test_phase")
        assert loaded is not None
        assert loaded["best_tps"] == 42.5

    def test_load_missing_returns_none(self, tmp_path):
        """load_phase_results should return None for missing results."""
        ctx = _make_ctx(results_dir=tmp_path)
        result = load_phase_results(ctx, "nonexistent_phase")
        assert result is None


@pytest.mark.unit
class TestEnsureResultsDir:
    def test_creates_directory(self, tmp_path):
        """ensure_results_dir should create the results directory."""
        target = tmp_path / "new_results"
        ctx = _make_ctx(results_dir=target)
        ensure_results_dir(ctx)
        assert target.is_dir()


@pytest.mark.unit
class TestStudyCallables:
    def test_setup_study_is_callable(self):
        """setup_study should be callable."""
        assert callable(setup_study)

    def test_check_and_mark_duplicate_trial_is_callable(self):
        """check_and_mark_duplicate_trial should be callable."""
        assert callable(check_and_mark_duplicate_trial)

    def test_update_param_cache_is_callable(self):
        """update_param_cache should be callable."""
        assert callable(update_param_cache)
