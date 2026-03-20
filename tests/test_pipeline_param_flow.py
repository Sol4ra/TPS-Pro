"""Tests for pipeline parameter flow between phases.

Verifies that params accumulate correctly as they flow through the
pipeline: MoE sweep -> KV sweep -> Core Engine -> Speculation -> Workload Sim.
All phase functions are mocked to return known params, then we assert each
subsequent phase receives the accumulated config via its base_config argument.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tps_pro.pipeline import (
    _extract_best_params,
    _validated_config_merge,
    run_full_pipeline,
)
from tps_pro.pipeline_config import PhaseConfig, PipelineConfig

# ---------------------------------------------------------------------------
# Module paths for patching
# ---------------------------------------------------------------------------
_P = "tps_pro.pipeline"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_ctx(
    is_moe=False,
    naked_engine=None,
    default_gpu_layers=99,
    skip_flags=None,
):
    """Build a mock AppContext with sensible defaults."""
    ctx = MagicMock()
    ctx.is_moe = is_moe
    ctx.default_gpu_layers = default_gpu_layers
    ctx.max_gpu_layers = 99
    ctx.numa_nodes = 1
    ctx.naked_engine = naked_engine or {
        "context": 4096,
        "mlock": True,
        "n_gpu_layers": default_gpu_layers,
    }
    ctx.results_dir = Path("/tmp/test-results")
    ctx.model_path = Path("/tmp/model.gguf")
    ctx.skip_flags = skip_flags if skip_flags is not None else set()
    ctx.quality_baseline = None
    ctx.kl_baseline_cache = None
    ctx.no_jinja = False
    ctx.default_experts = 8
    ctx.max_experts = 16
    ctx.expert_override_key = ""
    ctx.model_size_class = "medium"
    ctx.model_size_gb = 4.0
    ctx.lookup_cache_file = ""
    ctx.optuna_db = "sqlite:///test.db"
    ctx.skip_quality = False
    ctx.dry_run = False
    ctx.debug = False
    ctx.config = {
        "preset": "normal",
        "interactive": False,
        "model": "/tmp/model.gguf",
        "results_dir": "/tmp/test-results",
    }
    return ctx


def _make_config(preset="normal", skip_quality=False):
    return {
        "preset": preset,
        "skip_quality": skip_quality,
        "interactive": False,
        "model": "/tmp/model.gguf",
        "results_dir": "/tmp/test-results",
    }


def _make_pipeline_config(is_moe=False):
    """Build a PipelineConfig with the classic phase sequence for testing.

    Excludes ab_toggles so phase_core_engine is only called once.
    """
    phases = [
        PhaseConfig(phase="gpu_offload", display_name="GPU Offload"),
    ]
    if is_moe:
        phases.append(
            PhaseConfig(
                phase="moe_sweep",
                display_name="MoE Threads",
                moe_only=True,
            )
        )
    phases.extend(
        [
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
    )
    return PipelineConfig(global_flags={}, phases=phases)


def _build_base_patches(ctx_mock, cfg, load_results=None, is_moe=False):
    """Return base patches dict for run_full_pipeline."""
    pcfg = _make_pipeline_config(is_moe=is_moe)
    return {
        f"{_P}.ctx": ctx_mock,
        "tps_pro.state.config": cfg,
        f"{_P}.PipelineConfig.load": MagicMock(return_value=pcfg),
        f"{_P}.check_dry_run": MagicMock(return_value=False),
        f"{_P}.detect_gpus": MagicMock(return_value=[]),
        f"{_P}.detect_skippable_flags": MagicMock(return_value=set()),
        f"{_P}.load_phase_results": MagicMock(
            side_effect=load_results or (lambda _ctx, name: None)
        ),
        f"{_P}.kill_server": MagicMock(),
        f"{_P}.phase_gpu_offload": MagicMock(return_value={"best_ngl": 99}),
        f"{_P}.phase_tensor_split": MagicMock(return_value=None),
        f"{_P}.phase_moe_sweep": MagicMock(return_value=None),
        f"{_P}.phase_kv_context_sweep": MagicMock(return_value={"best_params": {}}),
        f"{_P}.phase_core_engine": MagicMock(return_value={"threads": 8}),
        f"{_P}.phase_speculation": MagicMock(return_value={}),
        f"{_P}.phase_workload_sim": MagicMock(return_value={}),
        f"{_P}.phase_quality": MagicMock(return_value={}),
    }


def _run_with_patches(patches, **kwargs):
    """Apply all patches and call run_full_pipeline."""
    active = {k: patch(k, v) for k, v in patches.items()}
    for p in active.values():
        p.start()
    try:
        run_full_pipeline(**kwargs)
    finally:
        for p in active.values():
            p.stop()


# ===================================================================
# 1. MoE sweep results merge into best_config before KV sweep
# ===================================================================


class TestMoeSweepToKvSweep:
    """MoE sweep results must be included in KV sweep's base_config."""

    @pytest.mark.unit
    def test_moe_results_in_kv_base_config(self):
        """KV sweep receives base_config containing MoE sweep params."""
        moe_params = {"n_cpu_moe": 12, "expert_used_count": 4}
        ctx_mock = _make_mock_ctx(is_moe=True)
        cfg = _make_config()

        kv_received_configs = []

        def _capture_kv(ctx, base_config=None, **kwargs):
            kv_received_configs.append(dict(base_config) if base_config else {})
            return {"best_params": {"cache_type_k": "q8_0"}}

        patches = _build_base_patches(ctx_mock, cfg, is_moe=True)
        patches[f"{_P}.phase_moe_sweep"] = MagicMock(
            return_value={"best_params": moe_params}
        )
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(side_effect=_capture_kv)

        _run_with_patches(patches)

        assert len(kv_received_configs) == 1
        received = kv_received_configs[0]
        assert received["n_cpu_moe"] == 12
        assert received["expert_used_count"] == 4
        # naked_engine keys should also be present
        assert "context" in received
        assert "n_gpu_layers" in received


# ===================================================================
# 2. KV sweep results merge into best_config before Core Engine
# ===================================================================


class TestKvSweepToCoreEngine:
    """KV sweep results must be included in Core Engine's base_config."""

    @pytest.mark.unit
    def test_kv_results_in_core_base_config(self):
        """Core Engine receives base_config containing KV sweep params."""
        kv_params = {"cache_type_k": "q8_0", "context": 16384}
        ctx_mock = _make_mock_ctx()
        cfg = _make_config()

        core_received_configs = []

        def _capture_core(ctx, n_trials=100, base_config=None, **kwargs):
            core_received_configs.append(dict(base_config) if base_config else {})
            return {"best_params": {"threads": 8, "batch_size": 512}}

        patches = _build_base_patches(ctx_mock, cfg)
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(
            return_value={"best_params": kv_params}
        )
        patches[f"{_P}.phase_core_engine"] = MagicMock(side_effect=_capture_core)

        _run_with_patches(patches)

        assert len(core_received_configs) == 1
        received = core_received_configs[0]
        assert received["cache_type_k"] == "q8_0"
        assert received["context"] == 16384
        # naked_engine keys present
        assert received["mlock"] is True
        assert received["n_gpu_layers"] == 99


# ===================================================================
# 3. Core Engine receives base_config with all prior results
# ===================================================================


class TestCoreEngineReceivesAllPriorResults:
    """Core Engine base_config should contain MoE + KV results."""

    @pytest.mark.unit
    def test_core_has_moe_and_kv_params(self):
        """Core Engine sees accumulated MoE and KV params."""
        moe_params = {"n_cpu_moe": 16}
        kv_params = {"cache_type_k": "f16", "context": 32768}
        ctx_mock = _make_mock_ctx(is_moe=True)
        cfg = _make_config()

        core_received = []

        def _capture_core(ctx, n_trials=100, base_config=None, **kwargs):
            core_received.append(dict(base_config) if base_config else {})
            return {"best_params": {"threads": 4}}

        patches = _build_base_patches(ctx_mock, cfg, is_moe=True)
        patches[f"{_P}.phase_moe_sweep"] = MagicMock(
            return_value={"best_params": moe_params}
        )
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(
            return_value={"best_params": kv_params}
        )
        patches[f"{_P}.phase_core_engine"] = MagicMock(side_effect=_capture_core)

        _run_with_patches(patches)

        assert len(core_received) == 1
        received = core_received[0]
        assert received["n_cpu_moe"] == 16
        assert received["cache_type_k"] == "f16"
        assert received["context"] == 32768
        assert received["n_gpu_layers"] == 99


# ===================================================================
# 4. Speculation receives base_config with Core Engine results
# ===================================================================


class TestSpeculationReceivesCoreResults:
    """Speculation base_config should contain Core Engine results."""

    @pytest.mark.unit
    def test_speculation_has_core_params(self):
        """Speculation sees KV + Core Engine accumulated params."""
        kv_params = {"cache_type_k": "q8_0"}
        core_params = {"threads": 8, "batch_size": 512, "flash_attn": "on"}

        ctx_mock = _make_mock_ctx()
        cfg = _make_config()

        spec_received = []

        def _capture_spec(ctx, n_trials=40, base_config=None, **kwargs):
            spec_received.append(dict(base_config) if base_config else {})
            return {"best_params": {"spec_type": "ngram"}}

        patches = _build_base_patches(ctx_mock, cfg)
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(
            return_value={"best_params": kv_params}
        )
        patches[f"{_P}.phase_core_engine"] = MagicMock(
            return_value={"best_params": core_params}
        )
        patches[f"{_P}.phase_speculation"] = MagicMock(side_effect=_capture_spec)

        _run_with_patches(patches)

        assert len(spec_received) == 1
        received = spec_received[0]
        # Core engine results present
        assert received["threads"] == 8
        assert received["batch_size"] == 512
        assert received["flash_attn"] == "on"
        # KV results present
        assert received["cache_type_k"] == "q8_0"
        # naked_engine present
        assert received["mlock"] is True


# ===================================================================
# 5. Workload Sim receives base_config with all prior results
# ===================================================================


class TestWorkloadSimReceivesAllResults:
    """Workload Sim base_config should contain all prior phase results."""

    @pytest.mark.unit
    def test_workload_has_all_prior_params(self):
        """Workload Sim sees KV + Core + Speculation accumulated params."""
        kv_params = {"cache_type_k": "q4_0", "context": 8192}
        core_params = {"threads": 6, "batch_size": 256}
        spec_params = {"spec_type": "ngram", "ngram_min": 2}

        ctx_mock = _make_mock_ctx()
        cfg = _make_config()

        ws_received = []

        def _capture_ws(ctx, base_config=None, **kwargs):
            ws_received.append(dict(base_config) if base_config else {})
            return {"hot_ttft_avg_ms": 45}

        patches = _build_base_patches(ctx_mock, cfg)
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(
            return_value={"best_params": kv_params}
        )
        patches[f"{_P}.phase_core_engine"] = MagicMock(
            return_value={"best_params": core_params}
        )
        patches[f"{_P}.phase_speculation"] = MagicMock(
            return_value={"best_params": spec_params}
        )
        patches[f"{_P}.phase_workload_sim"] = MagicMock(side_effect=_capture_ws)

        _run_with_patches(patches)

        assert len(ws_received) == 1
        received = ws_received[0]
        # All phase results present
        assert received["cache_type_k"] == "q4_0"
        assert received["context"] == 8192
        assert received["threads"] == 6
        assert received["batch_size"] == 256
        assert received["spec_type"] == "ngram"
        assert received["ngram_min"] == 2
        # naked_engine baseline present
        assert received["mlock"] is True
        assert received["n_gpu_layers"] == 99


# ===================================================================
# 6. naked_engine values preserved when phases return None
# ===================================================================


class TestNakedEnginePreservedOnNone:
    """When phases return None, naked_engine values must remain in best_config."""

    @pytest.mark.unit
    def test_naked_engine_survives_none_phases(self):
        """All naked_engine keys remain when MoE, KV, Core all return None."""
        naked = {"context": 4096, "mlock": True, "n_gpu_layers": 99}
        ctx_mock = _make_mock_ctx(naked_engine=dict(naked))
        cfg = _make_config()

        spec_received = []

        def _capture_spec(ctx, n_trials=40, base_config=None, **kwargs):
            spec_received.append(dict(base_config) if base_config else {})
            return {}

        patches = _build_base_patches(ctx_mock, cfg)
        # All phases return None (simulating failures)
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(
            side_effect=RuntimeError("boom")
        )
        patches[f"{_P}.phase_core_engine"] = MagicMock(side_effect=RuntimeError("boom"))
        patches[f"{_P}.phase_speculation"] = MagicMock(side_effect=_capture_spec)

        _run_with_patches(patches)

        assert len(spec_received) == 1
        received = spec_received[0]
        # naked_engine values must survive
        assert received["context"] == 4096
        assert received["mlock"] is True
        assert received["n_gpu_layers"] == 99

    @pytest.mark.unit
    def test_naked_engine_not_overwritten_by_empty_phase(self):
        """Phase returning empty dict does not erase naked_engine keys."""
        naked = {"context": 4096, "mlock": True, "n_gpu_layers": 99}
        ctx_mock = _make_mock_ctx(naked_engine=dict(naked))
        cfg = _make_config()

        ws_received = []

        def _capture_ws(ctx, base_config=None):
            ws_received.append(dict(base_config) if base_config else {})
            return {}

        patches = _build_base_patches(ctx_mock, cfg)
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(
            return_value={"best_params": {}}
        )
        patches[f"{_P}.phase_core_engine"] = MagicMock(return_value={"best_params": {}})
        patches[f"{_P}.phase_speculation"] = MagicMock(return_value={"best_params": {}})
        patches[f"{_P}.phase_workload_sim"] = MagicMock(side_effect=_capture_ws)

        _run_with_patches(patches)

        assert len(ws_received) == 1
        received = ws_received[0]
        assert received["context"] == 4096
        assert received["mlock"] is True
        assert received["n_gpu_layers"] == 99


# ===================================================================
# 7. _validated_config_merge doesn't lose keys
# ===================================================================


class TestValidatedConfigMergePreservesKeys:
    """_validated_config_merge must not drop keys from base or phase dicts."""

    @pytest.mark.unit
    def test_all_base_keys_survive_merge(self):
        """All base_config keys remain after merging new phase params."""
        base = {
            "context": 4096,
            "mlock": True,
            "n_gpu_layers": 99,
            "threads": 8,
            "batch_size": 512,
        }
        phase = {"flash_attn": "on", "cache_type_k": "q8_0"}

        result = _validated_config_merge(base, phase, "test")

        for key in base:
            assert key in result, f"base key '{key}' lost after merge"
        for key in phase:
            assert key in result, f"phase key '{key}' lost after merge"
        assert len(result) == len(base) + len(phase)

    @pytest.mark.unit
    def test_overlapping_keys_use_phase_value(self):
        """When base and phase share a key, phase value wins."""
        base = {"threads": 4, "context": 4096}
        phase = {"threads": 8, "batch_size": 512}

        result = _validated_config_merge(base, phase, "test")

        assert result["threads"] == 8  # phase overrides base
        assert result["context"] == 4096  # base preserved
        assert result["batch_size"] == 512  # phase added

    @pytest.mark.unit
    def test_none_values_in_phase_do_not_erase_base_keys(self):
        """None-valued keys in phase_params must not overwrite base keys."""
        base = {"threads": 8, "batch_size": 512}
        phase = {"threads": None, "flash_attn": "on"}

        result = _validated_config_merge(base, phase, "test")

        # threads=None is filtered out, so base value preserved
        assert result["threads"] == 8
        assert result["flash_attn"] == "on"
        assert result["batch_size"] == 512

    @pytest.mark.unit
    def test_sequential_merges_accumulate(self):
        """Chained merges accumulate all keys from all phases."""
        base = {"context": 4096, "mlock": True, "n_gpu_layers": 99}

        after_moe = _validated_config_merge(base, {"n_cpu_moe": 12}, "moe")
        after_kv = _validated_config_merge(
            after_moe, {"cache_type_k": "q8_0", "context": 8192}, "kv"
        )
        after_core = _validated_config_merge(
            after_kv, {"threads": 8, "batch_size": 512}, "core"
        )
        after_spec = _validated_config_merge(after_core, {"spec_type": "ngram"}, "spec")

        expected_keys = {
            "context",
            "mlock",
            "n_gpu_layers",
            "n_cpu_moe",
            "cache_type_k",
            "threads",
            "batch_size",
            "spec_type",
        }
        assert set(after_spec.keys()) == expected_keys
        # context was overridden by KV sweep
        assert after_spec["context"] == 8192
        assert after_spec["n_cpu_moe"] == 12
        assert after_spec["spec_type"] == "ngram"


# ===================================================================
# 8. _extract_best_params handles PhaseReturnDict and bare dicts
# ===================================================================


class TestExtractBestParamsVariants:
    """_extract_best_params must handle all known return shapes."""

    @pytest.mark.unit
    def test_phase_return_dict_shape(self):
        """Standard PhaseReturnDict with best_params key."""
        result = _extract_best_params(
            {
                "best_params": {"threads": 8, "batch_size": 512},
                "best_score": 42.0,
                "study_name": "core_engine",
            }
        )
        assert result == {"threads": 8, "batch_size": 512}

    @pytest.mark.unit
    def test_bare_dict_without_best_params_returns_empty(self):
        """Bare dict without best_params key returns empty dict."""
        bare = {"threads": 8, "batch_size": 512}
        result = _extract_best_params(bare)
        assert result == {}

    @pytest.mark.unit
    def test_none_returns_none(self):
        """None input (phase failure) returns None."""
        assert _extract_best_params(None) is None

    @pytest.mark.unit
    def test_empty_dict_returns_empty(self):
        """Empty dict returns empty dict (legacy path)."""
        assert _extract_best_params({}) == {}

    @pytest.mark.unit
    def test_best_params_is_none(self):
        """Dict with best_params=None returns None."""
        assert _extract_best_params({"best_params": None}) is None

    @pytest.mark.unit
    def test_best_params_is_empty_dict(self):
        """Dict with best_params={} returns empty dict."""
        assert _extract_best_params({"best_params": {}}) == {}

    @pytest.mark.unit
    def test_non_dict_input(self):
        """Non-dict, non-None input returns empty dict."""
        assert _extract_best_params(42) == {}
        assert _extract_best_params("string") == {}
        assert _extract_best_params([1, 2, 3]) == {}

    @pytest.mark.unit
    def test_best_params_nested_dict(self):
        """best_params containing nested structures are preserved."""
        nested = {"threads": 8, "tensor_split": [0.5, 0.5]}
        result = _extract_best_params({"best_params": nested})
        assert result == nested
        assert result["tensor_split"] == [0.5, 0.5]


# ===================================================================
# Full pipeline flow: end-to-end accumulation test
# ===================================================================


class TestFullPipelineParamAccumulation:
    """End-to-end test: verify each phase receives accumulated params."""

    @pytest.mark.unit
    def test_all_phases_receive_correct_accumulated_config(self):
        """Each phase captures its base_config; verify full accumulation chain."""
        naked = {"context": 4096, "mlock": True, "n_gpu_layers": 99}
        ctx_mock = _make_mock_ctx(is_moe=True, naked_engine=dict(naked))
        cfg = _make_config()

        moe_params = {"n_cpu_moe": 12, "expert_used_count": 4}
        kv_params = {"cache_type_k": "q8_0", "context": 16384}
        core_params = {"threads": 8, "batch_size": 512}
        spec_params = {"spec_type": "ngram", "ngram_min": 2}

        kv_captured = []
        core_captured = []
        spec_captured = []
        ws_captured = []

        def _cap_kv(ctx, base_config=None, **kwargs):
            kv_captured.append(dict(base_config) if base_config else {})
            return {"best_params": kv_params}

        def _cap_core(ctx, n_trials=100, base_config=None, **kwargs):
            core_captured.append(dict(base_config) if base_config else {})
            return {"best_params": core_params}

        def _cap_spec(ctx, n_trials=40, base_config=None, **kwargs):
            spec_captured.append(dict(base_config) if base_config else {})
            return {"best_params": spec_params}

        def _cap_ws(ctx, base_config=None, **kwargs):
            ws_captured.append(dict(base_config) if base_config else {})
            return {}

        patches = _build_base_patches(ctx_mock, cfg, is_moe=True)
        patches[f"{_P}.phase_moe_sweep"] = MagicMock(
            return_value={"best_params": moe_params}
        )
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(side_effect=_cap_kv)
        patches[f"{_P}.phase_core_engine"] = MagicMock(side_effect=_cap_core)
        patches[f"{_P}.phase_speculation"] = MagicMock(side_effect=_cap_spec)
        patches[f"{_P}.phase_workload_sim"] = MagicMock(side_effect=_cap_ws)

        _run_with_patches(patches)

        # KV sweep receives: naked_engine + MoE params
        assert len(kv_captured) == 1
        kv_cfg = kv_captured[0]
        assert kv_cfg["n_cpu_moe"] == 12
        assert kv_cfg["expert_used_count"] == 4
        assert kv_cfg["context"] == 4096  # from naked_engine (not yet overridden)
        assert kv_cfg["mlock"] is True

        # Core Engine receives: naked_engine + MoE + KV
        # ab_toggles also calls phase_core_engine, so we may get 2 captures
        assert len(core_captured) >= 1
        core_cfg = core_captured[-1]  # last call is the core_engine phase
        assert core_cfg["n_cpu_moe"] == 12
        assert core_cfg["cache_type_k"] == "q8_0"
        assert core_cfg["context"] == 16384  # KV overrode naked_engine
        assert core_cfg["mlock"] is True

        # Speculation receives: naked_engine + MoE + KV + Core
        assert len(spec_captured) == 1
        spec_cfg = spec_captured[0]
        assert spec_cfg["n_cpu_moe"] == 12
        assert spec_cfg["cache_type_k"] == "q8_0"
        assert spec_cfg["context"] == 16384
        assert spec_cfg["threads"] == 8
        assert spec_cfg["batch_size"] == 512

        # Workload Sim receives: naked_engine + MoE + KV + Core + Spec
        assert len(ws_captured) == 1
        ws_cfg = ws_captured[0]
        assert ws_cfg["n_cpu_moe"] == 12
        assert ws_cfg["cache_type_k"] == "q8_0"
        assert ws_cfg["context"] == 16384
        assert ws_cfg["threads"] == 8
        assert ws_cfg["batch_size"] == 512
        assert ws_cfg["spec_type"] == "ngram"
        assert ws_cfg["ngram_min"] == 2
        assert ws_cfg["mlock"] is True
        assert ws_cfg["n_gpu_layers"] == 99

    @pytest.mark.unit
    def test_dense_model_skips_moe_but_flows_rest(self):
        """Dense model (is_moe=False) skips MoE sweep but rest flows correctly."""
        naked = {"context": 4096, "mlock": True, "n_gpu_layers": 50}
        ctx_mock = _make_mock_ctx(
            is_moe=False, naked_engine=dict(naked), default_gpu_layers=50
        )
        cfg = _make_config()

        kv_params = {"cache_type_k": "f16"}
        core_params = {"threads": 4}

        spec_captured = []

        def _cap_spec(ctx, n_trials=40, base_config=None, **kwargs):
            spec_captured.append(dict(base_config) if base_config else {})
            return {}

        patches = _build_base_patches(ctx_mock, cfg)
        patches[f"{_P}.phase_kv_context_sweep"] = MagicMock(
            return_value={"best_params": kv_params}
        )
        patches[f"{_P}.phase_core_engine"] = MagicMock(
            return_value={"best_params": core_params}
        )
        patches[f"{_P}.phase_speculation"] = MagicMock(side_effect=_cap_spec)

        _run_with_patches(patches)

        assert len(spec_captured) == 1
        received = spec_captured[0]
        # No MoE keys
        assert "n_cpu_moe" not in received
        # KV + Core present
        assert received["cache_type_k"] == "f16"
        assert received["threads"] == 4
        # naked_engine present
        assert received["n_gpu_layers"] == 50
        assert received["context"] == 4096
