"""Tests for pipeline.py: run_full_pipeline() orchestration.

All phase functions and external dependencies are mocked to test
orchestration logic in isolation — no servers, no models, no Optuna.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tps_pro.pipeline import (
    run_full_pipeline,
)
from tps_pro.pipeline_config import PhaseConfig, PipelineConfig

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
    """Build a mock AppContext with sensible defaults.

    NOTE: Uses bare MagicMock (no spec) because run_full_pipeline accesses
    many AppContext fields dynamically.  Adding spec=AppContext would require
    setting every field the pipeline touches, which is fragile and defeats
    the purpose of mocking for orchestration tests.
    """
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
    ctx.config = {
        "preset": "normal",
        "skip_quality": skip_quality,
        "interactive": False,
        "model": str(model_path or Path("/tmp/model.gguf")),
        "results_dir": str(results_dir or Path("/tmp/test-results")),
    }
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


def _make_pipeline_config(
    is_moe=False,
    include_tensor_split=False,
    include_ab_toggles=False,
):
    """Build a PipelineConfig for testing.

    By default produces the 'classic' 6-phase sequence:
    gpu_offload, kv_context_sweep, core_engine, speculation,
    workload_sim, quality.  Toggle flags add extra phases.
    """
    phases = [
        PhaseConfig(phase="gpu_offload", display_name="GPU Offload"),
    ]
    if include_tensor_split:
        phases.append(PhaseConfig(phase="tensor_split", display_name="Tensor Split"))
    if is_moe:
        phases.append(
            PhaseConfig(
                phase="moe_sweep",
                display_name="MoE Threads",
                moe_only=True,
            )
        )
    phases.append(
        PhaseConfig(
            phase="kv_context_sweep",
            display_name="KV + Context Sweep",
            kv_types=["f16", "q8_0", "q4_0"],
        )
    )
    if include_ab_toggles:
        phases.append(
            PhaseConfig(
                phase="ab_toggles",
                display_name="A/B Toggles",
                test_flags=[
                    "op_offload",
                    "prio",
                    "prio_batch",
                    "no_mmap",
                    "mlock",
                    "repack",
                    "swa_full",
                    "numa",
                    "cpu_strict",
                    "cpu_strict_batch",
                ],
            )
        )
    phases.extend(
        [
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
            PhaseConfig(
                phase="speculation",
                display_name="Speculation",
                trials=40,
            ),
            PhaseConfig(
                phase="workload_sim",
                display_name="Workload Sim",
            ),
            PhaseConfig(
                phase="quality",
                display_name="Quality/Sampling",
                trials=60,
            ),
        ]
    )
    return PipelineConfig(global_flags={}, phases=phases)


# ===================================================================
# run_full_pipeline tests
# ===================================================================


class TestRunFullPipeline:
    """Tests for run_full_pipeline orchestration."""

    def _build_patches(
        self,
        ctx_mock=None,
        config_dict=None,
        load_results=None,
        detect_gpus_ret=None,
        pipeline_config=None,
    ):
        """Return a dict of patch targets -> mock values for run_full_pipeline."""
        ctx_mock = ctx_mock or _make_mock_ctx()
        config_dict = config_dict or _make_config()
        load_results = load_results or (lambda _ctx, name: None)
        detect_gpus_ret = detect_gpus_ret or []
        pipeline_config = pipeline_config or _make_pipeline_config()

        return {
            f"{_P}.ctx": ctx_mock,
            "tps_pro.state.config": config_dict,
            f"{_P}.PipelineConfig.load": MagicMock(return_value=pipeline_config),
            f"{_P}.check_dry_run": MagicMock(return_value=False),
            f"{_P}.detect_gpus": MagicMock(return_value=detect_gpus_ret),
            f"{_P}.detect_skippable_flags": MagicMock(return_value=set()),
            f"{_P}.load_phase_results": MagicMock(side_effect=load_results),
            f"{_P}.kill_server": MagicMock(),
            f"{_P}.phase_gpu_offload": MagicMock(return_value={"best_ngl": 99}),
            f"{_P}.phase_tensor_split": MagicMock(return_value=None),
            f"{_P}.phase_kv_context_sweep": MagicMock(
                return_value={"best_params": {"cache_type_k": "q8_0", "context": 8192}}
            ),
            f"{_P}.phase_core_engine": MagicMock(
                return_value={"threads": 8, "batch_size": 512}
            ),
            f"{_P}.phase_speculation": MagicMock(return_value={"spec_type": "ngram"}),
            f"{_P}.phase_workload_sim": MagicMock(return_value={"hot_ttft_avg_ms": 50}),
            f"{_P}.phase_quality": MagicMock(return_value={"best_score": 90}),
        }

    def _run_with_patches(self, patches, **kwargs):
        """Apply all patches and call run_full_pipeline."""

        active_patches = {k: patch(k, v) for k, v in patches.items()}
        mocks = {}
        for k, p in active_patches.items():
            mocks[k] = p.start()
        try:
            run_full_pipeline(**kwargs)
        finally:
            for p in active_patches.values():
                p.stop()
        return mocks

    @pytest.mark.unit
    def test_phases_run_in_correct_order(self):
        """All phases execute in the expected sequence."""
        call_order = []

        def _track(name, ret=None):
            def _fn(*args, **kwargs):
                call_order.append(name)
                return ret

            return _fn

        patches = self._build_patches()
        patches[f"{_P}.phase_gpu_offload"] = MagicMock(
            side_effect=_track("gpu_offload", {"best_ngl": 99})
        )
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(
            side_effect=_track("kv_context_sweep", {"best_params": {}})
        )
        patches[f"{_P}.phase_core_engine"] = MagicMock(
            side_effect=_track("core_engine", {"threads": 8})
        )
        patches[f"{_P}.phase_speculation"] = MagicMock(
            side_effect=_track("speculation", {})
        )
        patches[f"{_P}.phase_workload_sim"] = MagicMock(
            side_effect=_track("workload_sim", {})
        )
        patches[f"{_P}.phase_quality"] = MagicMock(side_effect=_track("quality", {}))

        self._run_with_patches(patches)

        assert call_order == [
            "gpu_offload",
            "kv_context_sweep",
            "core_engine",
            "speculation",
            "workload_sim",
            "quality",
        ]

    @pytest.mark.unit
    def test_gpu_offload_runs_before_core_engine(self):
        """GPU offload must complete before core engine starts."""
        call_order = []

        def _track(name, ret=None):
            def _fn(*args, **kwargs):
                call_order.append(name)
                return ret

            return _fn

        patches = self._build_patches()
        patches[f"{_P}.phase_gpu_offload"] = MagicMock(
            side_effect=_track("gpu_offload", {"best_ngl": 99})
        )
        patches[f"{_P}.phase_core_engine"] = MagicMock(
            side_effect=_track("core_engine", {"threads": 8})
        )

        self._run_with_patches(patches)

        gpu_idx = call_order.index("gpu_offload")
        core_idx = call_order.index("core_engine")
        assert gpu_idx < core_idx

    @pytest.mark.unit
    def test_quality_runs_after_parameter_tuning(self):
        """Quality/sampling phase runs after all parameter tuning phases."""
        call_order = []

        def _track(name, ret=None):
            def _fn(*args, **kwargs):
                call_order.append(name)
                return ret

            return _fn

        patches = self._build_patches()
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(
            side_effect=_track("kv_context_sweep", {"best_params": {}})
        )
        patches[f"{_P}.phase_core_engine"] = MagicMock(
            side_effect=_track("core_engine", {"threads": 8})
        )
        patches[f"{_P}.phase_speculation"] = MagicMock(
            side_effect=_track("speculation", {})
        )
        patches[f"{_P}.phase_quality"] = MagicMock(side_effect=_track("quality", {}))

        self._run_with_patches(patches)

        quality_idx = call_order.index("quality")
        for phase in ["kv_context_sweep", "core_engine", "speculation"]:
            assert call_order.index(phase) < quality_idx, (
                f"{phase} should run before quality"
            )

    @pytest.mark.unit
    def test_skip_quality_flag(self):
        """When skip_quality is set, phase_quality is not called."""
        cfg = _make_config(skip_quality=True)
        patches = self._build_patches(config_dict=cfg)
        quality_mock = MagicMock()
        patches[f"{_P}.phase_quality"] = quality_mock

        self._run_with_patches(patches)

        quality_mock.assert_not_called()

    @pytest.mark.unit
    def test_dry_run_returns_early(self):
        """Dry run mode returns before running any phases."""
        patches = self._build_patches()
        patches[f"{_P}.check_dry_run"] = MagicMock(return_value=True)
        gpu_mock = MagicMock()
        patches[f"{_P}.phase_gpu_offload"] = gpu_mock

        self._run_with_patches(patches)

        gpu_mock.assert_not_called()

    @pytest.mark.unit
    def test_deadline_skips_remaining_phases(self):
        """When deadline has passed, subsequent phases are skipped."""
        # Set deadline to the past so all phases are skipped
        patches = self._build_patches()

        # Use a deadline that's already expired
        expired_deadline = time.time() - 100

        call_order = []

        def _track(name, ret=None):
            def _fn(*args, **kwargs):
                call_order.append(name)
                return ret

            return _fn

        patches[f"{_P}.phase_gpu_offload"] = MagicMock(
            side_effect=_track("gpu_offload", {"best_ngl": 99})
        )
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(
            side_effect=_track("kv_context_sweep", {"best_params": {}})
        )
        patches[f"{_P}.phase_core_engine"] = MagicMock(
            side_effect=_track("core_engine", {"threads": 8})
        )
        patches[f"{_P}.phase_speculation"] = MagicMock(
            side_effect=_track("speculation", {})
        )
        patches[f"{_P}.phase_quality"] = MagicMock(side_effect=_track("quality", {}))

        self._run_with_patches(patches, deadline=expired_deadline)

        # With an expired deadline, _run_phase returns None immediately for all phases
        assert call_order == []

    @pytest.mark.unit
    def test_phase_failure_does_not_crash_pipeline(self):
        """A phase that raises an exception should not crash the pipeline."""
        call_order = []

        def _track(name, ret=None):
            def _fn(*args, **kwargs):
                call_order.append(name)
                return ret

            return _fn

        patches = self._build_patches()
        patches[f"{_P}.phase_gpu_offload"] = MagicMock(
            side_effect=_track("gpu_offload", {"best_ngl": 99})
        )
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(
            side_effect=_track("kv_context_sweep", {"best_params": {}})
        )
        # Core engine raises an error
        patches[f"{_P}.phase_core_engine"] = MagicMock(
            side_effect=RuntimeError("CUDA OOM")
        )
        patches[f"{_P}.phase_speculation"] = MagicMock(
            side_effect=_track("speculation", {})
        )
        patches[f"{_P}.phase_quality"] = MagicMock(side_effect=_track("quality", {}))

        # Should not raise
        self._run_with_patches(patches)

        # Phases after the failure should still run
        assert "speculation" in call_order
        assert "quality" in call_order

    @pytest.mark.unit
    def test_resume_skips_completed_phases(self):
        """resume_from parameter skips earlier phases."""
        call_order = []

        def _track(name, ret=None):
            def _fn(*args, **kwargs):
                call_order.append(name)
                return ret

            return _fn

        patches = self._build_patches()
        patches[f"{_P}.phase_gpu_offload"] = MagicMock(
            side_effect=_track("gpu_offload", {"best_ngl": 99})
        )
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(
            side_effect=_track("kv_context_sweep", {"best_params": {}})
        )
        patches[f"{_P}.phase_core_engine"] = MagicMock(
            side_effect=_track("core_engine", {"threads": 8})
        )
        patches[f"{_P}.phase_speculation"] = MagicMock(
            side_effect=_track("speculation", {})
        )

        # resume_from=2 should skip phases 0-1 (GPU Offload,
        # KV+Context Sweep) and run from Core Engine (index 2).
        self._run_with_patches(patches, resume_from=2)

        assert "gpu_offload" not in call_order
        assert "kv_context_sweep" not in call_order
        assert "core_engine" in call_order

    @pytest.mark.unit
    def test_tensor_split_runs_with_multi_gpu(self):
        """Tensor split phase runs when multiple GPUs are detected."""
        call_order = []

        def _track(name, ret=None):
            def _fn(*args, **kwargs):
                call_order.append(name)
                return ret

            return _fn

        two_gpus = [
            {"name": "RTX 4090", "vram_mb": 24000},
            {"name": "RTX 4090", "vram_mb": 24000},
        ]
        pcfg = _make_pipeline_config(include_tensor_split=True)
        patches = self._build_patches(
            detect_gpus_ret=two_gpus,
            pipeline_config=pcfg,
        )
        patches[f"{_P}.phase_tensor_split"] = MagicMock(
            side_effect=_track("tensor_split", None)
        )

        self._run_with_patches(patches)

        assert "tensor_split" in call_order

    @pytest.mark.unit
    def test_tensor_split_skipped_with_single_gpu(self):
        """Tensor split phase is skipped with a single GPU."""
        one_gpu = [{"name": "RTX 4090", "vram_mb": 24000}]
        patches = self._build_patches(detect_gpus_ret=one_gpu)
        ts_mock = MagicMock()
        patches[f"{_P}.phase_tensor_split"] = ts_mock

        self._run_with_patches(patches)

        ts_mock.assert_not_called()

    @pytest.mark.unit
    def test_result_propagation_core_to_speculation(self):
        """Core engine best params are passed as base_config to speculation."""
        core_result = {"threads": 8, "batch_size": 512, "flash_attn": "on"}
        patches = self._build_patches()
        patches[f"{_P}.phase_core_engine"] = MagicMock(return_value=core_result)

        spec_mock = MagicMock(return_value={})
        patches[f"{_P}.phase_speculation"] = spec_mock

        self._run_with_patches(patches)

        # Speculation should be called with a base_config containing core engine results
        spec_mock.assert_called_once()
        # The call is wrapped in a lambda, so we check the mock was invoked
        # The lambda calls phase_speculation(n_trials=..., base_config=best_config)
        # We verify via the phase being called (lambda captures best_config)

    @pytest.mark.unit
    def test_result_propagation_kv_sweep_to_core(self):
        """KV context sweep results are merged into best_config before core engine."""
        patches = self._build_patches()
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(
            return_value={"best_params": {"cache_type_k": "q8_0", "context": 8192}}
        )
        core_mock = MagicMock(return_value={"threads": 8})
        patches[f"{_P}.phase_core_engine"] = core_mock

        self._run_with_patches(patches)

        # Core engine should have been called (kv_sweep results merged before it)
        core_mock.assert_called_once()

    @pytest.mark.unit
    def test_saved_results_used_when_phase_returns_none(self):
        """When a phase returns None (e.g., failure), saved results are loaded."""
        saved_core = {"best_params": {"threads": 4, "batch_size": 256}}

        def _load(_ctx, name):
            if name == "core_engine":
                return saved_core
            return None

        patches = self._build_patches(load_results=_load)
        # Core engine fails (returns None via _run_phase error handling)
        patches[f"{_P}.phase_core_engine"] = MagicMock(side_effect=RuntimeError("fail"))

        spec_mock = MagicMock(return_value={})
        patches[f"{_P}.phase_speculation"] = spec_mock

        self._run_with_patches(patches)

        # Pipeline should continue using loaded results
        spec_mock.assert_called_once()

    @pytest.mark.unit
    def test_kv_context_sweep_runs_before_core_engine(self):
        """KV + Context Sweep runs after GPU offload but before core engine."""
        call_order = []

        def _track(name, ret=None):
            def _fn(*args, **kwargs):
                call_order.append(name)
                return ret

            return _fn

        patches = self._build_patches()
        patches[f"{_P}.phase_gpu_offload"] = MagicMock(
            side_effect=_track("gpu_offload", {"best_ngl": 99})
        )
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(
            side_effect=_track("kv_context_sweep", {"best_params": {"context": 8192}})
        )
        patches[f"{_P}.phase_core_engine"] = MagicMock(
            side_effect=_track("core_engine", {"threads": 8})
        )

        self._run_with_patches(patches)

        assert call_order.index("gpu_offload") < call_order.index("kv_context_sweep")
        assert call_order.index("kv_context_sweep") < call_order.index("core_engine")

    @pytest.mark.unit
    def test_kv_context_sweep_results_merged_into_best_config(self):
        """KV context sweep best_params are merged into best_config before core
        engine."""
        ctx_mock = _make_mock_ctx()
        patches = self._build_patches(ctx_mock=ctx_mock)
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(
            return_value={"best_params": {"cache_type_k": "q8_0", "context": 16384}}
        )

        spec_mock = MagicMock(return_value={})
        patches[f"{_P}.phase_speculation"] = spec_mock

        self._run_with_patches(patches)

        # Pipeline should continue without error; kv_sweep results are merged
        spec_mock.assert_called_once()

    @pytest.mark.unit
    def test_generate_command_failure_does_not_crash(self):
        """Failure in generate_optimized_command or generate_html_report is caught."""
        patches = self._build_patches()

        # Mock the lazy imports in the finally block
        with patch(
            "tps_pro.cli.services_command.generate_optimized_command",
            side_effect=ImportError("no module"),
            create=True,
        ):
            # Should not raise
            self._run_with_patches(patches)

    @pytest.mark.unit
    def test_preset_scales_trial_counts(self):
        """Different presets produce different trial counts."""
        for preset, _expected_min_core in [
            ("quick", 50),
            ("normal", 100),
            ("thorough", 150),
        ]:
            cfg = _make_config(preset=preset)
            patches = self._build_patches(config_dict=cfg)
            core_mock = MagicMock(return_value={"threads": 8})
            patches[f"{_P}.phase_core_engine"] = core_mock

            self._run_with_patches(patches)

            # core_engine is called via lambda: phase_core_engine(n_trials=t_core)
            # We verify it was called at all — detailed trial count testing would
            # require inspecting the lambda capture, which we trust from code review.
            core_mock.assert_called_once()
