"""Tests for trial_helpers.py -- shared trial execution helpers.

Mocks all external dependencies (hardware, engine, measurement, search)
so tests run without a real server or GPU.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tps_pro.phases.trial_helpers import (
    finalize_trial,
    print_phase_summary,
    record_trial_attrs,
    recover_best_score,
    run_study_with_callbacks,
    setup_baseline_server,
    thermal_gate,
)
from tps_pro.result_types import ParetoObjectives, PerfResult

# ===================================================================
# Helpers
# ===================================================================


def _make_ctx(**overrides):
    """Build a minimal mock ctx for trial_helpers tests."""
    from _ctx_factory import make_ctx_from_defaults

    return make_ctx_from_defaults(results_dir="/tmp/results", **overrides)


def _make_perf(**overrides):
    """Build a PerfResult with sensible defaults."""
    defaults = dict(tps=50.0, ttft=10.0, prompt_tps=100.0, total_ms=200.0)
    defaults.update(overrides)
    return PerfResult(**defaults)


def _make_trial(number=0):
    """Build a mock Optuna trial."""
    trial = MagicMock()
    trial.number = number
    trial.set_user_attr = MagicMock()
    return trial


# ===================================================================
# thermal_gate
# ===================================================================


@pytest.mark.unit
class TestThermalGate:
    @patch("tps_pro.phases.trial_helpers.wait_for_cooldown")
    @patch("tps_pro.phases.trial_helpers.check_thermal_throttle")
    def test_calls_wait_when_throttled(self, mock_check, mock_wait):
        mock_check.return_value = (True, 85)
        thermal_gate()
        mock_check.assert_called_once()
        mock_wait.assert_called_once()

    @patch("tps_pro.phases.trial_helpers.wait_for_cooldown")
    @patch("tps_pro.phases.trial_helpers.check_thermal_throttle")
    def test_skips_wait_when_not_throttled(self, mock_check, mock_wait):
        mock_check.return_value = (False, 60)
        thermal_gate()
        mock_check.assert_called_once()
        mock_wait.assert_not_called()


# ===================================================================
# record_trial_attrs
# ===================================================================


@pytest.mark.unit
class TestRecordTrialAttrs:
    @patch(
        "tps_pro.phases.trial_helpers.get_vram_used_mb",
        return_value=4096.0,
    )
    def test_perf_result_path_with_vram(self, mock_vram):
        """When perf is a PerfResult and VRAM is available, returns new PerfResult
        with vram_used_mb set."""

        perf = _make_perf()
        trial = _make_trial()

        result = record_trial_attrs(_make_ctx(), trial, perf)
        assert isinstance(result, PerfResult)
        assert result.vram_used_mb == 4096.0

    @patch(
        "tps_pro.phases.trial_helpers.get_vram_used_mb",
        return_value=None,
    )
    def test_no_vram_available(self, mock_vram):
        """When VRAM is not available, perf is returned unchanged."""

        perf = _make_perf()
        trial = _make_trial()

        result = record_trial_attrs(_make_ctx(), trial, perf)
        assert result.vram_used_mb is None


# ===================================================================
# finalize_trial
# ===================================================================


@pytest.mark.unit
class TestFinalizeTrial:
    @patch(
        "tps_pro.phases.trial_helpers.print_trial_result",
        return_value=55.0,
    )
    @patch(
        "tps_pro.phases.trial_helpers.compute_score",
        return_value=50.0,
    )
    def test_non_pareto_calls_print_trial_result(self, mock_score, mock_print):

        perf = _make_perf(tps=40.0)
        trial = _make_trial(number=3)

        return_val, new_best = finalize_trial(
            _make_ctx(),
            trial,
            perf,
            "ngl=99",
            best_score=45.0,
            total_trials=10,
            is_pareto=False,
        )
        mock_print.assert_called_once()
        assert return_val == pytest.approx(50.0)
        assert new_best == pytest.approx(55.0)

    @patch("tps_pro.phases.trial_helpers.compute_pareto_objectives")
    @patch(
        "tps_pro.phases.trial_helpers.compute_score",
        return_value=50.0,
    )
    def test_pareto_mode_returns_objectives(self, mock_score, mock_pareto):

        objectives = ParetoObjectives(tps=50.0, neg_vram=-4096.0, quality_factor=0.9)
        mock_pareto.return_value = objectives
        perf = _make_perf(tps=50.0, quality_factor=0.9)
        trial = _make_trial(number=1)

        return_val, new_best = finalize_trial(
            _make_ctx(),
            trial,
            perf,
            "ngl=99",
            best_score=45.0,
            total_trials=10,
            is_pareto=True,
        )
        assert return_val == objectives
        assert new_best == pytest.approx(45.0)


# ===================================================================
# recover_best_score
# ===================================================================


@pytest.mark.unit
class TestRecoverBestScore:
    def test_recovers_from_completed_trials(self):
        import optuna

        t1 = MagicMock()
        t1.state = optuna.trial.TrialState.COMPLETE
        t1.user_attrs = {
            "tps": 50.0,
            "ttft": 10.0,
            "prompt_tps": 100.0,
            "total_ms": 200.0,
        }

        t2 = MagicMock()
        t2.state = optuna.trial.TrialState.COMPLETE
        t2.user_attrs = {
            "tps": 80.0,
            "ttft": 8.0,
            "prompt_tps": 120.0,
            "total_ms": 150.0,
        }

        t3 = MagicMock()
        t3.state = optuna.trial.TrialState.PRUNED
        t3.user_attrs = {}

        study = MagicMock()
        study.trials = [t1, t2, t3]

        # score_fn returns tps as score for simplicity
        def score_fn(perf):
            return perf.tps

        best = recover_best_score(study, score_fn)
        assert best == pytest.approx(80.0)

    def test_returns_zero_when_no_completed(self):
        import optuna

        t1 = MagicMock()
        t1.state = optuna.trial.TrialState.PRUNED
        t1.user_attrs = {}

        study = MagicMock()
        study.trials = [t1]

        best = recover_best_score(study, lambda p: p.tps)
        assert best == 0.0


# ===================================================================
# run_study_with_callbacks
# ===================================================================


@pytest.mark.unit
class TestRunStudyWithCallbacks:
    @patch("tps_pro.phases.trial_helpers.close_phase_pbar")
    @patch("tps_pro.phases.trial_helpers.create_phase_pbar")
    @patch("tps_pro.phases.trial_helpers.ProgressBarUpdateCallback")
    def test_creates_pbar_and_calls_optimize(self, mock_tqdm, mock_create, mock_close):

        study = MagicMock()
        objective = MagicMock()
        ctx = _make_ctx()

        run_study_with_callbacks(
            ctx, study, objective, remaining=10, label="Phase1", best_score=5.0
        )

        mock_create.assert_called_once_with(10, desc="Phase1")
        study.optimize.assert_called_once()
        mock_close.assert_called_once()

    @patch("tps_pro.phases.trial_helpers.close_phase_pbar")
    @patch("tps_pro.phases.trial_helpers.create_phase_pbar")
    @patch("tps_pro.phases.trial_helpers.ProgressBarUpdateCallback")
    def test_pareto_skips_gp_callback(self, mock_tqdm, mock_create, mock_close):

        study = MagicMock()
        ctx = _make_ctx()

        run_study_with_callbacks(
            ctx,
            study,
            MagicMock(),
            remaining=5,
            label="Pareto",
            best_score=0.0,
            is_pareto=True,
        )

        # Verify optimize was called with callbacks that do NOT include
        # GPStoppingCallback
        args, kwargs = study.optimize.call_args
        callbacks = kwargs.get("callbacks", args[2] if len(args) > 2 else [])
        # In pareto mode, only ProgressBarUpdateCallback should be present
        assert len(callbacks) == 1


# ===================================================================
# print_phase_summary
# ===================================================================


@pytest.mark.unit
class TestPrintPhaseSummary:
    @patch("tps_pro.phases.trial_helpers.save_phase_results")
    @patch(
        "tps_pro.phases.trial_helpers.print_param_importance",
        return_value={},
    )
    @patch("tps_pro.phases.trial_helpers.get_best_trial")
    @patch(
        "tps_pro.phases.trial_helpers.trial_scalar_value",
        return_value=60.0,
    )
    def test_returns_params_and_results(
        self, mock_scalar, mock_best, mock_importance, mock_save
    ):
        import time

        best_trial = MagicMock()
        best_trial.params = {"ngl": 99}
        best_trial.user_attrs = {"tps": 55.0}
        mock_best.return_value = best_trial

        study = MagicMock()
        study.trials = []

        baseline = _make_perf(tps=40.0)
        ctx = _make_ctx()

        from tps_pro.phases.trial_helpers import PhaseSummaryContext

        params, results = print_phase_summary(
            ctx,
            PhaseSummaryContext(
                phase_name="test_phase",
                study=study,
                baseline=baseline,
                baseline_score=40.0,
                phase_start_time=time.time() - 60,
            ),
        )
        assert params == {"ngl": 99}
        assert results["phase"] == "test_phase"
        mock_save.assert_called_once()


# ===================================================================
# setup_baseline_server
# ===================================================================


@pytest.mark.unit
class TestSetupBaselineServer:
    @patch(
        "tps_pro.phases.trial_helpers.compute_score",
        return_value=45.0,
    )
    @patch("tps_pro.phases.trial_helpers.measure_perf_adaptive")
    @patch(
        "tps_pro.phases.trial_helpers.boot_server_with_jinja_recovery",
        return_value=(MagicMock(), "ok"),
    )
    @patch("tps_pro.phases.trial_helpers.kill_server")
    def test_success_returns_perf_and_score(
        self, mock_kill, mock_boot, mock_measure, mock_score
    ):

        perf = _make_perf(tps=45.0)
        mock_measure.return_value = (perf, False)

        ctx = _make_ctx()
        result_perf, result_score = setup_baseline_server(ctx, {}, "test_phase")

        assert result_perf is perf
        assert result_score == pytest.approx(45.0)
        mock_kill.assert_called_once()

    @patch(
        "tps_pro.phases.trial_helpers.boot_server_with_jinja_recovery",
        return_value=(MagicMock(), "fail"),
    )
    @patch("tps_pro.phases.trial_helpers.kill_server")
    def test_failure_returns_none(self, mock_kill, mock_boot):

        ctx = _make_ctx(fail_fast=False)
        result_perf, result_score = setup_baseline_server(ctx, {}, "test_phase")

        assert result_perf is None
        assert result_score == 0.0

    @patch(
        "tps_pro.phases.trial_helpers.boot_server_with_jinja_recovery",
        return_value=(MagicMock(), "fail"),
    )
    @patch("tps_pro.phases.trial_helpers.kill_server")
    def test_fail_fast_raises(self, mock_kill, mock_boot):
        from tps_pro.engine import BaselineFailure

        ctx = _make_ctx(fail_fast=True)
        with pytest.raises(BaselineFailure):
            setup_baseline_server(ctx, {}, "test_phase")
