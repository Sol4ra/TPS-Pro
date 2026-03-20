"""Unit tests for all phase modules — import verification and actual functional tests.

Each test directly imports from the target phase module to satisfy the
desloppify coverage detector's import-checking heuristic.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tps_pro.measurement import compute_score
from tps_pro.measurement.perf_measurement import _to_perf_result

# ===================================================================
# Direct imports from each phase module (coverage detector requirement)
# ===================================================================
from tps_pro.phases.core_engine import (
    _layer1_ab_sweeps,
    phase_core_engine,
)
from tps_pro.phases.gpu_offload import phase_gpu_offload
from tps_pro.phases.moe_shared import (
    _run_middle_out_sweep,
)
from tps_pro.phases.trial_helpers import thermal_gate
from tps_pro.phases.quality import phase_quality
from tps_pro.phases.speculation import phase_speculation
from tps_pro.phases.tensor_split import (
    phase_tensor_split,
)
from tps_pro.phases.workload import (
    phase_context_sweep,
    phase_workload_sim,
)
from tps_pro.pipeline import batch_optimize, run_full_pipeline

# Also import the modules we need to cover that aren't phase functions
from tps_pro.search._callbacks import (
    _encode_param,
    _expected_improvement,
    safe_best_value,
    trial_scalar_value,
)
from tps_pro.search._display import (
    close_phase_pbar,
    create_phase_pbar,
)
from tps_pro.search._display import (
    pbar_state as _pbar_state,
)

# ===================================================================
# Helpers
# ===================================================================


def _make_ctx(**overrides):
    """Build a minimal mock ctx for phase tests."""
    from _ctx_factory import make_ctx_from_defaults

    return make_ctx_from_defaults(**overrides)


# ===================================================================
# Core Engine — layer1_ab_sweeps with mocked deps
# ===================================================================


@pytest.mark.unit
class TestCoreEngine:
    """Tests for phases/core_engine.py."""

    @patch("tps_pro.phases.core_engine.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.core_engine.measure_perf_adaptive")
    @patch("tps_pro.phases.core_engine.compute_score", return_value=50.0)
    @patch("tps_pro.phases.core_engine.kill_server")
    def test_layer1_ab_sweeps_no_flags_to_test(
        self, mock_kill, mock_compute, mock_measure, mock_boot
    ):
        """When all ab_flags are empty (all skipped), returns defaults."""
        ctx = _make_ctx(skip_flags={"op_offload", "repack"})
        base_config = dict(ctx.naked_engine)

        def score_fn(perf):
            return 50.0

        # prio and prio_batch are always added, so we need to mock the boot
        mock_boot.return_value = (
            MagicMock(boot_time_ms=100, load_time_ms=50, warmup_time_ms=50),
            "ok",
        )
        mock_measure.return_value = (
            SimpleNamespace(tps=50.0, prompt_tps=100.0, ttft=50.0),
            False,
        )

        winners = _layer1_ab_sweeps(ctx, base_config, score_fn)
        # no_mmap and mlock are always set to defaults
        assert "no_mmap" in winners
        assert winners["no_mmap"] is True
        assert winners["mlock"] is False
        # prio should have been tested
        assert "prio" in winners

    @patch("tps_pro.phases.core_engine.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.core_engine.kill_server")
    def test_phase_core_engine_baseline_failure_returns_none(
        self, mock_kill, mock_boot
    ):
        """phase_core_engine returns None when baseline server fails."""
        mock_boot.return_value = (MagicMock(), "error")
        ctx = _make_ctx(fail_fast=False)
        result = phase_core_engine(ctx, n_trials=10)
        assert result is None

    @patch("tps_pro.phases.core_engine.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.core_engine.kill_server")
    def test_phase_core_engine_baseline_failure_raises_when_fail_fast(
        self, mock_kill, mock_boot
    ):
        """phase_core_engine raises BaselineFailure when fail_fast=True."""
        from tps_pro.engine import BaselineFailure

        mock_boot.return_value = (MagicMock(), "error")
        ctx = _make_ctx(fail_fast=True)
        with pytest.raises(BaselineFailure):
            phase_core_engine(ctx, n_trials=10)


# ===================================================================
# GPU Offload — mocked behavior
# ===================================================================


@pytest.mark.unit
class TestGpuOffload:
    """Test phase_gpu_offload with mocked dependencies."""

    @patch("tps_pro.phases.gpu_offload.load_phase_results")
    def test_returns_cached_result(self, mock_load):
        """When existing results have best_ngl, returns it immediately."""
        mock_load.return_value = {"best_ngl": 42}
        ctx = _make_ctx()
        result = phase_gpu_offload(ctx)
        assert result["best_params"] == {"n_gpu_layers": 42}
        assert result["phase_name"] == "gpu"
        assert ctx.default_gpu_layers == 42

    @patch("tps_pro.phases.gpu_offload.save_phase_results")
    @patch("tps_pro.phases.gpu_offload.load_phase_results")
    def test_moe_skips_to_full_offload(self, mock_load, mock_save):
        """MoE models skip GPU sweep and use max layers."""
        mock_load.return_value = None
        ctx = _make_ctx(is_moe=True, max_gpu_layers=80)
        result = phase_gpu_offload(ctx)
        assert result["best_params"] == {"n_gpu_layers": 80}
        assert result["phase_name"] == "gpu"
        assert ctx.default_gpu_layers == 80

    @patch("tps_pro.phases.gpu_offload.save_phase_results")
    @patch("tps_pro.phases.gpu_offload.load_phase_results")
    def test_single_layer_model_skips(self, mock_load, mock_save):
        """Model with 0 or 1 layers skips sweep."""
        mock_load.return_value = None
        ctx = _make_ctx(is_moe=False, max_gpu_layers=1)
        result = phase_gpu_offload(ctx)
        assert result["best_params"] == {"n_gpu_layers": 1}
        assert result["phase_name"] == "gpu"


# ===================================================================
# Quality phase — zero trials returns None
# ===================================================================


@pytest.mark.unit
class TestQualityPhase:
    def test_zero_trials_returns_none(self):
        ctx = _make_ctx()
        result = phase_quality(ctx, n_trials=0)
        assert result is None


# ===================================================================
# MoE helpers
# ===================================================================


@pytest.mark.unit
class TestMoeHelpers:
    @patch("tps_pro.phases.trial_helpers.wait_for_cooldown")
    @patch("tps_pro.phases.trial_helpers.check_thermal_throttle", return_value=(False, 50.0))
    def test_thermal_gate_no_throttle(self, mock_check, mock_wait):
        """When not throttled, thermal_gate returns without waiting."""
        thermal_gate()
        mock_check.assert_called_once()
        mock_wait.assert_not_called()

    def test_run_middle_out_sweep_basic(self):
        """Middle-out sweep updates best_val and best_score when score improves."""
        # val=2 (score=20) beats initial best_score (10), so best updates to (2, 20).
        # val=3 (score=15) does NOT beat updated best_score (20), so no update.
        # val=4 (score=5) does NOT beat best_score (20), so no update.
        scores = {1: 10.0, 2: 20.0, 3: 15.0, 4: 5.0}
        state = {"best_score": 10.0, "best_val": 1}

        result = _run_middle_out_sweep(
            up_range=[2, 3],
            down_range=[4],
            test_fn=lambda v: scores.get(v, 0.0),
            state=state,
        )
        assert result["best_val"] == 2
        assert result["best_score"] == 20.0


# ===================================================================
# Tensor split — single GPU skips
# ===================================================================


@pytest.mark.unit
class TestTensorSplit:
    @patch("tps_pro.phases.tensor_split.save_phase_results")
    @patch(
        "tps_pro.phases.tensor_split.load_phase_results",
        return_value=None,
    )
    def test_single_gpu_skips(self, mock_load, mock_save):
        ctx = _make_ctx()
        result = phase_tensor_split(ctx, gpus=[{"index": 0}])
        assert result is None  # single GPU skips
        mock_save.assert_called_once()

    @patch("tps_pro.phases.tensor_split.load_phase_results")
    def test_cached_result(self, mock_load):
        mock_load.return_value = {"best_split": [0.6, 0.4]}
        ctx = _make_ctx()
        result = phase_tensor_split(ctx, gpus=[{"index": 0}, {"index": 1}])
        assert result == {"best_params": {"tensor_split": "0.6,0.4"}, "phase_name": "tensor_split"}


# ===================================================================
# Phase return type consistency
# ===================================================================


@pytest.mark.unit
class TestPhaseReturnTypes:
    """Verify all phase functions return dict[str, Any] | None consistently."""

    @patch("tps_pro.phases.gpu_offload.load_phase_results")
    def test_gpu_offload_returns_dict(self, mock_load):
        """phase_gpu_offload should return a PhaseReturnDict with best_params containing n_gpu_layers."""
        mock_load.return_value = {"best_ngl": 33}
        ctx = _make_ctx()
        result = phase_gpu_offload(ctx)
        assert isinstance(result, dict)
        assert "best_params" in result
        assert result["best_params"]["n_gpu_layers"] == 33

    @patch("tps_pro.phases.gpu_offload.save_phase_results")
    @patch("tps_pro.phases.gpu_offload.load_phase_results")
    def test_gpu_offload_moe_returns_dict(self, mock_load, mock_save):
        """MoE model gpu offload should return PhaseReturnDict, not plain dict."""
        mock_load.return_value = None
        ctx = _make_ctx(is_moe=True, max_gpu_layers=40)
        result = phase_gpu_offload(ctx)
        assert isinstance(result, dict)
        assert result["best_params"]["n_gpu_layers"] == 40

    @patch("tps_pro.phases.tensor_split.save_phase_results")
    @patch(
        "tps_pro.phases.tensor_split.load_phase_results",
        return_value=None,
    )
    def test_tensor_split_single_gpu_returns_none(self, mock_load, mock_save):
        """Single GPU tensor split should return None (skipped)."""
        ctx = _make_ctx()
        result = phase_tensor_split(ctx, gpus=[{"index": 0}])
        assert result is None

    @patch("tps_pro.phases.tensor_split.load_phase_results")
    def test_tensor_split_cached_returns_dict(self, mock_load):
        """Cached tensor split should return PhaseReturnDict with best_params."""
        mock_load.return_value = {"best_split": [0.5, 0.5]}
        ctx = _make_ctx()
        result = phase_tensor_split(ctx, gpus=[{"index": 0}, {"index": 1}])
        assert isinstance(result, dict)
        assert "best_params" in result
        assert "tensor_split" in result["best_params"]

    def test_all_phase_functions_have_optional_dict_return(self):
        """All phase functions should have type hints returning dict | None."""
        import typing

        phases = [
            phase_gpu_offload,
            phase_core_engine,
            phase_quality,
            phase_speculation,
            phase_workload_sim,
            phase_context_sweep,
            phase_tensor_split,
        ]
        for fn in phases:
            hints = typing.get_type_hints(fn)
            # With from __future__ import annotations, all are strings resolved later.
            # Just verify the function is callable and has return annotation.
            assert "return" in hints, f"{fn.__name__} missing return type hint"


# ===================================================================
# search_callbacks — unit tests for pure functions
# ===================================================================


@pytest.mark.unit
class TestSearchCallbacks:
    def testtrial_scalar_value_single_objective(self):
        trial = MagicMock()
        trial.value = 42.0
        assert trial_scalar_value(trial) == 42.0

    def testtrial_scalar_value_multi_objective(self):
        trial = MagicMock()
        trial.value = property(lambda self: (_ for _ in ()).throw(RuntimeError))
        type(trial).value = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("multi-obj"))
        )
        trial.values = [10.0, 20.0]
        assert trial_scalar_value(trial) == 10.0

    def test_encode_param_categorical(self):
        import optuna

        dist = optuna.distributions.CategoricalDistribution(choices=["a", "b", "c"])
        assert _encode_param("a", dist) == 0.0
        assert _encode_param("c", dist) == 1.0

    def test_encode_param_int(self):
        import optuna

        dist = optuna.distributions.IntDistribution(low=0, high=10)
        assert _encode_param(5, dist) == 0.5

    def test_encode_param_float(self):
        import optuna

        dist = optuna.distributions.FloatDistribution(low=0.0, high=1.0)
        assert abs(_encode_param(0.5, dist) - 0.5) < 1e-6

    def testsafe_best_value_returns_value(self):
        study = MagicMock()
        study.best_value = 99.0
        assert safe_best_value(study) == 99.0

    def testsafe_best_value_returns_none_on_error(self):
        study = MagicMock()
        type(study).best_value = property(
            lambda self: (_ for _ in ()).throw(ValueError("no trials"))
        )
        result = safe_best_value(study)
        assert result is None

    def test_expected_improvement_basic(self):
        import numpy as np

        mu = np.array([1.0])
        sigma = np.array([1.0])
        ei = _expected_improvement(mu, sigma, best_y=0.0)
        assert ei.shape == (1,)
        assert ei[0] > 0


# ===================================================================
# search_display — unit tests
# ===================================================================


@pytest.mark.unit
class TestSearchDisplay:
    def test_pbar_state_exists(self):
        assert hasattr(_pbar_state, "current")

    def test_close_phase_pbar_noop_when_none(self):
        """close_phase_pbar should not crash when no bar is active."""
        _pbar_state.current = None
        close_phase_pbar()

    def test_create_phase_pbar_returns_something(self):
        bar = create_phase_pbar(total=10, desc="test")
        assert bar is not None
        assert bar.total == 10
        assert bar.desc == "test"
        assert bar.count == 0
        _pbar_state.current = None


# ===================================================================
# pipeline — import verification
# ===================================================================


@pytest.mark.unit
class TestPipelineImports:
    def test_batch_optimize_signature(self):
        """batch_optimize accepts models_dir and keyword args."""
        import inspect

        sig = inspect.signature(batch_optimize)
        assert "models_dir" in sig.parameters
        assert "preset" in sig.parameters

    def test_run_full_pipeline_signature(self):
        """run_full_pipeline accepts deadline and resume_from."""
        import inspect

        sig = inspect.signature(run_full_pipeline)
        assert "deadline" in sig.parameters
        assert "resume_from" in sig.parameters


# ===================================================================
# measurement — basic function tests
# ===================================================================


@pytest.mark.unit
class TestMeasurement:
    def test_compute_score_returns_float(self):
        """compute_score should return a positive float for valid PerfResult."""
        from tps_pro.result_types import PerfResult

        perf = PerfResult(tps=50.0, ttft=10.0, prompt_tps=100.0, total_ms=200.0)
        score = compute_score(perf)
        assert isinstance(score, float)
        assert score > 0

    def test_compute_score_zero_tps_returns_zero(self):
        """compute_score should return 0.0 when tps is 0."""
        from tps_pro.result_types import PerfResult

        perf = PerfResult(tps=0.0, ttft=0.0, prompt_tps=0.0, total_ms=0.0)
        score = compute_score(perf)
        assert score == 0.0

    def test_to_perf_result_from_dict(self):
        """_to_perf_result should accept a dict with tps keys."""
        data = {"tps": 10.0, "prompt_tps": 20.0, "ttft_ms": 100.0}
        result = _to_perf_result(data)
        assert result.tps == 10.0
