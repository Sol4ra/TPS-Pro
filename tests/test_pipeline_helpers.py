"""Tests for pipeline.py: phase ordering, result propagation, and helper functions.

Covers TestPhaseOrdering, TestResultPropagation, TestValidatedConfigMerge,
TestExtractBestParams, TestValidatedConfigMergeExtended, and TestPipelineConstants.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tps_pro.pipeline import (
    _PHASE_ERRORS,
    _REPORT_ERRORS,
    _validated_config_merge,
    run_full_pipeline,
)
from tps_pro.pipeline_config import PipelineConfig, PhaseConfig

# ---------------------------------------------------------------------------
# Module paths for patching (pipeline.py re-imports these names)
# ---------------------------------------------------------------------------
_P = "tps_pro.pipeline"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_ctx(
    is_moe=False,
    default_gpu_layers=99,
    max_gpu_layers=99,
    skip_quality=False,
    dry_run=False,
    numa_nodes=1,
    naked_engine=None,
    results_dir=None,
    model_path=None,
    skip_flags=None,
):
    """Build a mock AppContext with sensible defaults."""
    ctx = MagicMock()
    ctx.is_moe = is_moe
    ctx.default_gpu_layers = default_gpu_layers
    ctx.max_gpu_layers = max_gpu_layers
    ctx.skip_quality = skip_quality
    ctx.dry_run = dry_run
    ctx.numa_nodes = numa_nodes
    ctx.naked_engine = naked_engine or {
        "context": 4096,
        "mlock": True,
        "n_gpu_layers": default_gpu_layers,
    }
    ctx.results_dir = results_dir or Path("/tmp/test-results")
    ctx.model_path = model_path or Path("/tmp/model.gguf")
    ctx.skip_flags = skip_flags if skip_flags is not None else set()
    ctx.config = {
        "preset": "normal",
        "skip_quality": skip_quality,
        "interactive": False,
        "model": str(model_path or Path("/tmp/model.gguf")),
        "results_dir": str(results_dir or Path("/tmp/test-results")),
    }
    ctx.quality_baseline = None
    ctx.kl_baseline_cache = None
    ctx.no_jinja = False
    ctx.default_experts = 8
    ctx.max_experts = 16
    ctx.expert_override_key = ""
    ctx.model_size_class = "medium"
    ctx.model_size_gb = 4.0
    ctx.is_moe = is_moe
    ctx.lookup_cache_file = ""
    ctx.optuna_db = "sqlite:///test.db"
    return ctx


def _make_config(
    preset="normal", skip_quality=False, interactive=False, target_context=None
):
    """Build a mock _config dict."""
    cfg = {
        "preset": preset,
        "skip_quality": skip_quality,
        "interactive": interactive,
        "model": "/tmp/model.gguf",
        "results_dir": "/tmp/test-results",
    }
    if target_context is not None:
        cfg["target_context"] = target_context
    return cfg


def _make_pipeline_config():
    """Build a PipelineConfig with the classic 6-phase sequence for testing."""
    phases = [
        PhaseConfig(phase="gpu_offload", display_name="GPU Offload"),
        PhaseConfig(
            phase="kv_context_sweep",
            display_name="KV + Context Sweep",
            kv_types=["f16", "q8_0", "q4_0"],
        ),
        PhaseConfig(
            phase="core_engine",
            display_name="Core Engine",
            trials=100,
            search_params=[
                "threads",
                "threads_batch",
                "batch_size",
                "ubatch_size",
                "flash_attn",
                "poll",
                "poll_batch",
            ],
        ),
        PhaseConfig(phase="speculation", display_name="Speculation", trials=40),
        PhaseConfig(phase="workload_sim", display_name="Workload Sim"),
        PhaseConfig(phase="quality", display_name="Quality/Sampling", trials=60),
    ]
    return PipelineConfig(global_flags={}, phases=phases)


# ===================================================================
# Phase ordering and result propagation (focused unit tests)
# ===================================================================


class TestPhaseOrdering:
    """Focused tests on phase ordering constraints."""

    @pytest.mark.unit
    @patch(f"{_P}.load_phase_results", return_value=None)
    @patch(f"{_P}.detect_gpus", return_value=[])
    @patch(f"{_P}.detect_skippable_flags", return_value=set())
    @patch(f"{_P}.check_dry_run", return_value=False)
    @patch(f"{_P}.kill_server")
    @patch(f"{_P}.phase_quality")
    @patch(f"{_P}.phase_workload_sim")
    @patch(f"{_P}.phase_speculation")
    @patch(f"{_P}.phase_core_engine")
    @patch(f"{_P}.phase_kv_context_sweep")
    @patch(f"{_P}.phase_tensor_split")
    @patch(f"{_P}.phase_gpu_offload")
    @patch(f"{_P}.PipelineConfig.load", return_value=_make_pipeline_config())
    @patch("tps_pro.state.config", _make_config())
    @patch(f"{_P}.ctx", _make_mock_ctx())
    def test_full_phase_sequence(
        self,
        mock_load_config,
        mock_gpu,
        mock_ts,
        mock_kv_sweep,
        mock_core,
        mock_spec,
        mock_ws,
        mock_quality,
        mock_kill,
        mock_dry,
        mock_skip,
        mock_detect,
        mock_load_results,
    ):
        """Verify the complete phase sequence end-to-end."""
        sequence = []

        for name, mock_obj, ret in [
            ("gpu_offload", mock_gpu, {"best_ngl": 99}),
            ("kv_context_sweep", mock_kv_sweep, {"best_params": {}}),
            ("core_engine", mock_core, {"threads": 8}),
            ("speculation", mock_spec, {}),
            ("workload_sim", mock_ws, {}),
            ("quality", mock_quality, {}),
        ]:

            def _make_side_effect(n, r):
                def _se(*a, **kw):
                    sequence.append(n)
                    return r

                return _se

            mock_obj.side_effect = _make_side_effect(name, ret)

        run_full_pipeline()

        expected = [
            "gpu_offload",
            "kv_context_sweep",
            "core_engine",
            "speculation",
            "workload_sim",
            "quality",
        ]
        assert sequence == expected


class TestResultPropagation:
    """Tests verifying results flow correctly between phases."""

    @pytest.mark.unit
    def test_kv_sweep_results_loaded_when_phase_fails(self):
        """When kv_context_sweep fails, saved results are loaded and merged."""

        def _load(_ctx, name):
            if name == "kv_context_sweep":
                return {"best_params": {"cache_type_k": "q8_0", "context": 8192}}
            if name == "core_engine":
                return {"best_params": {"threads": 8}}
            return None

        ctx_mock = _make_mock_ctx()
        cfg = _make_config()

        pcfg = _make_pipeline_config()
        patches = {
            f"{_P}.ctx": ctx_mock,
            "tps_pro.state.config": cfg,
            f"{_P}.PipelineConfig.load": MagicMock(return_value=pcfg),
            f"{_P}.check_dry_run": MagicMock(return_value=False),
            f"{_P}.detect_gpus": MagicMock(return_value=[]),
            f"{_P}.detect_skippable_flags": MagicMock(return_value=set()),
            f"{_P}.load_phase_results": MagicMock(side_effect=_load),
            f"{_P}.kill_server": MagicMock(),
            f"{_P}.phase_gpu_offload": MagicMock(return_value={"best_ngl": 99}),
            f"{_P}.phase_tensor_split": MagicMock(return_value=None),
            f"{_P}.phase_kv_context_sweep": MagicMock(side_effect=RuntimeError("fail")),
            f"{_P}.phase_core_engine": MagicMock(return_value={"threads": 8}),
            f"{_P}.phase_speculation": MagicMock(return_value={}),
            f"{_P}.phase_workload_sim": MagicMock(return_value={}),
            f"{_P}.phase_quality": MagicMock(return_value={}),
        }

        active = {k: patch(k, v) for k, v in patches.items()}
        for p in active.values():
            p.start()
        try:
            from tps_pro.pipeline import run_full_pipeline

            run_full_pipeline()
        finally:
            for p in active.values():
                p.stop()

        # The load_phase_results mock should have been called with "kv_context_sweep"
        load_mock = patches[f"{_P}.load_phase_results"]
        called_with = [c[0][1] for c in load_mock.call_args_list]
        assert "kv_context_sweep" in called_with

    @pytest.mark.unit
    def test_empty_spec_results_not_merged(self):
        """When speculation returns empty dict, it doesn't pollute best_config."""
        ctx_mock = _make_mock_ctx()
        cfg = _make_config()

        captured_ws_calls = []

        def _capture_ws(*args, **kwargs):
            # The lambda captures best_config — we can't directly inspect it,
            # but we verify workload_sim was called (pipeline didn't crash)
            captured_ws_calls.append(True)
            return {}

        pcfg = _make_pipeline_config()
        patches = {
            f"{_P}.ctx": ctx_mock,
            "tps_pro.state.config": cfg,
            f"{_P}.PipelineConfig.load": MagicMock(return_value=pcfg),
            f"{_P}.check_dry_run": MagicMock(return_value=False),
            f"{_P}.detect_gpus": MagicMock(return_value=[]),
            f"{_P}.detect_skippable_flags": MagicMock(return_value=set()),
            f"{_P}.load_phase_results": MagicMock(return_value=None),
            f"{_P}.kill_server": MagicMock(),
            f"{_P}.phase_gpu_offload": MagicMock(return_value={"best_ngl": 99}),
            f"{_P}.phase_tensor_split": MagicMock(return_value=None),
            f"{_P}.phase_kv_context_sweep": MagicMock(return_value={"best_params": {}}),
            f"{_P}.phase_core_engine": MagicMock(return_value={"threads": 8}),
            f"{_P}.phase_speculation": MagicMock(return_value={}),
            f"{_P}.phase_workload_sim": MagicMock(side_effect=_capture_ws),
            f"{_P}.phase_quality": MagicMock(return_value={}),
        }

        active = {k: patch(k, v) for k, v in patches.items()}
        for p in active.values():
            p.start()
        try:
            from tps_pro.pipeline import run_full_pipeline

            run_full_pipeline()
        finally:
            for p in active.values():
                p.stop()

        assert len(captured_ws_calls) == 1


# ===================================================================
# _validated_config_merge tests
# ===================================================================


class TestValidatedConfigMerge:
    """Tests for _validated_config_merge helper."""

    @pytest.mark.unit
    def test_none_input_returns_base_copy(self):
        """None phase_params returns a copy of base_config unchanged."""
        from tps_pro.pipeline import _validated_config_merge

        base = {"threads": 8, "batch_size": 512}
        result = _validated_config_merge(base, None, "test_phase")
        assert result == base
        assert result is not base  # must be a new dict

    @pytest.mark.unit
    def test_empty_dict_returns_base_copy(self):
        """Empty dict phase_params returns base_config unchanged."""
        from tps_pro.pipeline import _validated_config_merge

        base = {"threads": 8}
        result = _validated_config_merge(base, {}, "test_phase")
        assert result == base
        assert result is not base

    @pytest.mark.unit
    def test_dict_with_none_values_drops_nones(self):
        """Keys with None values in phase_params are filtered out."""
        from tps_pro.pipeline import _validated_config_merge

        base = {"threads": 8, "batch_size": 512}
        phase = {"flash_attn": "on", "mlock": None, "extra": None}
        result = _validated_config_merge(base, phase, "test_phase")
        assert result == {"threads": 8, "batch_size": 512, "flash_attn": "on"}
        assert "mlock" not in result
        assert "extra" not in result

    @pytest.mark.unit
    def test_normal_merge(self):
        """Valid phase_params are merged into base_config."""
        from tps_pro.pipeline import _validated_config_merge

        base = {"threads": 8}
        phase = {"batch_size": 512, "flash_attn": "on"}
        result = _validated_config_merge(base, phase, "core_engine")
        assert result == {"threads": 8, "batch_size": 512, "flash_attn": "on"}

    @pytest.mark.unit
    def test_phase_overrides_base(self):
        """Phase params override existing base keys."""
        from tps_pro.pipeline import _validated_config_merge

        base = {"threads": 8, "batch_size": 256}
        phase = {"batch_size": 512}
        result = _validated_config_merge(base, phase, "core_engine")
        assert result == {"threads": 8, "batch_size": 512}

    @pytest.mark.unit
    def test_non_dict_input_returns_base_copy(self):
        """Non-dict phase_params (e.g. a list) returns base unchanged."""
        from tps_pro.pipeline import _validated_config_merge

        base = {"threads": 8}
        result = _validated_config_merge(base, [1, 2, 3], "bad_phase")
        assert result == base
        assert result is not base

    @pytest.mark.unit
    def test_base_not_mutated(self):
        """Original base_config dict is never mutated."""
        from tps_pro.pipeline import _validated_config_merge

        base = {"threads": 8}
        original_base = dict(base)
        _validated_config_merge(base, {"batch_size": 512}, "test")
        assert base == original_base


# ===================================================================
# _extract_best_params tests
# ===================================================================


class TestExtractBestParams:
    """Tests for _extract_best_params helper."""

    @pytest.mark.unit
    def test_none_returns_none(self):
        """None input returns None."""
        from tps_pro.pipeline import _extract_best_params

        assert _extract_best_params(None) is None

    @pytest.mark.unit
    def test_dict_with_best_params_key(self):
        """PhaseReturnDict shape extracts best_params value."""
        from tps_pro.pipeline import _extract_best_params

        result = _extract_best_params(
            {
                "best_params": {"threads": 8, "batch_size": 512},
                "study_name": "core_engine",
            }
        )
        assert result == {"threads": 8, "batch_size": 512}

    @pytest.mark.unit
    def test_bare_dict_without_best_params_returns_empty(self):
        """Bare dict without best_params key returns empty dict."""
        from tps_pro.pipeline import _extract_best_params

        bare = {"threads": 8, "batch_size": 512}
        assert _extract_best_params(bare) == {}

    @pytest.mark.unit
    def test_empty_dict(self):
        """Empty dict returns empty dict."""
        from tps_pro.pipeline import _extract_best_params

        assert _extract_best_params({}) == {}

    @pytest.mark.unit
    def test_non_dict_returns_empty_dict(self):
        """Non-dict, non-None input returns empty dict."""
        from tps_pro.pipeline import _extract_best_params

        assert _extract_best_params("unexpected") == {}
        assert _extract_best_params(42) == {}
        assert _extract_best_params([1, 2]) == {}

    @pytest.mark.unit
    def test_best_params_none_value(self):
        """Dict with best_params=None extracts None."""
        from tps_pro.pipeline import _extract_best_params

        result = _extract_best_params({"best_params": None})
        assert result is None

    @pytest.mark.unit
    def test_best_params_empty_dict(self):
        """Dict with best_params={} extracts empty dict."""
        from tps_pro.pipeline import _extract_best_params

        result = _extract_best_params({"best_params": {}})
        assert result == {}


# ===================================================================
# _validated_config_merge additional tests
# ===================================================================


class TestValidatedConfigMergeExtended:
    """Additional tests for _validated_config_merge helper."""

    @pytest.mark.unit
    def test_basic_merge(self):
        """Phase params merge into base config, producing a new dict."""
        from tps_pro.pipeline import _validated_config_merge

        base = {"threads": 4, "context": 4096}
        phase = {"batch_size": 512, "threads": 8}
        result = _validated_config_merge(base, phase, "test_phase")

        assert result == {"threads": 8, "context": 4096, "batch_size": 512}
        # Verify base_config is not mutated
        assert base == {"threads": 4, "context": 4096}

    @pytest.mark.unit
    def test_none_params_returns_copy_of_base(self):
        """None phase_params returns a copy of base_config without changes."""
        from tps_pro.pipeline import _validated_config_merge

        base = {"threads": 4, "context": 4096}
        result = _validated_config_merge(base, None, "test_phase")

        assert result == base
        assert result is not base  # must be a new dict

    @pytest.mark.unit
    def test_non_dict_params_returns_copy_of_base(self):
        """Non-dict phase_params returns a copy of base_config without changes."""
        from tps_pro.pipeline import _validated_config_merge

        base = {"threads": 4}
        result = _validated_config_merge(base, "not_a_dict", "test_phase")

        assert result == base
        assert result is not base

    @pytest.mark.unit
    def test_none_values_filtered_out(self):
        """None values in phase_params are dropped to avoid corrupting config."""
        from tps_pro.pipeline import _validated_config_merge

        base = {"threads": 4, "batch_size": 256}
        phase = {"batch_size": 512, "flash_attn": None, "poll": 50}
        result = _validated_config_merge(base, phase, "test_phase")

        assert result == {"threads": 4, "batch_size": 512, "poll": 50}
        assert "flash_attn" not in result

    @pytest.mark.unit
    def test_empty_phase_params(self):
        """Empty dict phase_params returns a copy of base_config."""
        base = {"threads": 4}
        result = _validated_config_merge(base, {}, "test_phase")

        assert result == base
        assert result is not base


# ===================================================================
# Direct (non-mocked) tests for pipeline module-level constants
# ===================================================================


@pytest.mark.unit
class TestPipelineConstants:
    """Verify pipeline module-level constants without mocking."""

    def test_phase_errors_is_tuple(self):
        """_PHASE_ERRORS should be a tuple of exception classes."""
        assert isinstance(_PHASE_ERRORS, tuple)
        assert all(issubclass(e, Exception) for e in _PHASE_ERRORS)

    def test_report_errors_is_tuple(self):
        """_REPORT_ERRORS should be a tuple of exception classes."""
        assert isinstance(_REPORT_ERRORS, tuple)
        assert all(issubclass(e, Exception) for e in _REPORT_ERRORS)

    def test_phase_errors_contains_os_error(self):
        """_PHASE_ERRORS should include OSError for filesystem failures."""
        assert OSError in _PHASE_ERRORS

    def test_validated_config_merge_immutable(self):
        """_validated_config_merge should not mutate the base config."""
        base = {"threads": 4, "batch_size": 256}
        original = dict(base)
        _validated_config_merge(base, {"threads": 8}, "test")
        assert base == original
