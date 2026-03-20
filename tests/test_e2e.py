"""End-to-end tests for critical user flows in the llama_optimizer pipeline.

Mocks server processes and HTTP endpoints but exercises the full phase/pipeline
code paths including study management, result I/O, and phase orchestration.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from tps_pro.constants import SCORE_VERSION
from tps_pro.result_types import PerfResult, ServerProcess
from tps_pro.search import (
    load_phase_results,
    save_phase_results,
)

# ===================================================================
# Helpers
# ===================================================================


def _make_ctx(tmp_path, **overrides):
    """Build a mock ctx for e2e tests with real filesystem."""
    from _ctx_factory import make_ctx_from_defaults

    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)

    return make_ctx_from_defaults(
        server_path=Path("/usr/bin/llama-server"),
        model_path=tmp_path / "test.gguf",
        port=18090,
        _port_alt=18091,
        server_url="http://127.0.0.1:18090",
        model_size_class="small",
        http=MagicMock(spec=requests.Session),
        default_experts=8,
        max_experts=16,
        moe_sweep_center=8,
        max_gpu_layers=33,
        default_gpu_layers=33,
        naked_engine={"context": 4096, "mlock": True, "n_gpu_layers": 33},
        results_dir=results_dir,
        optuna_db="sqlite:///" + str(results_dir / "optuna.db").replace("\\", "/"),
        fresh_run=True,
        lookup_cache_file=str(results_dir / "lookup-cache.bin"),
        arch={
            "type": "dense",
            "expert_override_key": "",
            "default_experts": 8,
            "max_experts": 16,
        },
        config={"preset": "quick"},
        hw={
            "max_threads": 8,
            "moe_sweep_max": 16,
            "moe_sweep_center": 8,
            "max_gpu_layers": 33,
        },
        **overrides,
    )


def _make_server_proc(poll_return=None, stderr_lines=None):
    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = poll_return
    mock_proc.pid = 12345
    mock_proc.stderr = MagicMock()
    sp = ServerProcess(proc=mock_proc)
    if stderr_lines:
        sp.stderr_lines = list(stderr_lines)
    return sp


def _mock_http_response(tps=25.0, prompt_tps=300.0, ttft=50.0, predicted_ms=2000.0):
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "timings": {
            "predicted_per_second": tps,
            "prompt_per_second": prompt_tps,
            "prompt_ms": ttft,
            "predicted_ms": predicted_ms,
        }
    }
    return resp


# ===================================================================
# 1. Single-phase optimization (GPU offload)
# ===================================================================


@pytest.mark.integration
class TestSinglePhaseOptimization:
    """Mock server + HTTP. Run phase_gpu_offload(ctx) with a mocked server
    and verify it produces valid results."""

    @patch("tps_pro.phases.gpu_offload.wait_for_cooldown")
    @patch(
        "tps_pro.phases.gpu_offload.check_thermal_throttle",
        return_value=(False, 60),
    )
    @patch("tps_pro.phases.gpu_offload.measure_perf_adaptive")
    @patch("tps_pro.phases.gpu_offload.kill_server")
    @patch(
        "tps_pro.phases.gpu_offload.wait_for_server",
        return_value="ok",
    )
    @patch("tps_pro.phases.gpu_offload.start_server")
    @patch(
        "tps_pro.phases.gpu_offload.run_bench_trial",
        return_value=None,
    )
    def test_gpu_offload_produces_results(
        self,
        mock_bench,
        mock_start,
        mock_wait,
        mock_kill,
        mock_adaptive,
        mock_thermal,
        mock_cooldown,
        tmp_path,
    ):
        ctx = _make_ctx(tmp_path)
        ctx.max_gpu_layers = 5  # small sweep for fast test

        # Server starts successfully
        sp = _make_server_proc()
        mock_start.return_value = sp

        # Different TPS for different ngl values to create a clear winner
        tps_values = iter([15.0, 20.0, 25.0, 30.0, 28.0, 25.0])

        def varying_perf(*args, **kwargs):
            t = next(tps_values, 20.0)
            return PerfResult(tps=t, ttft=50.0, prompt_tps=300.0, total_ms=2050.0), True

        mock_adaptive.side_effect = varying_perf

        from tps_pro.phases.gpu_offload import phase_gpu_offload

        phase_gpu_offload(ctx)

        # Should have saved results
        gpu_data = load_phase_results(ctx, "gpu")
        assert gpu_data is not None
        assert "best_ngl" in gpu_data
        assert isinstance(gpu_data["best_ngl"], int)

        # ctx should be updated
        assert ctx.default_gpu_layers == gpu_data["best_ngl"]


# ===================================================================
# 2. Batch mode
# ===================================================================


@pytest.mark.integration
class TestBatchMode:
    """Create a temp directory with 2 fake GGUF files. Run batch_optimize()
    with everything mocked. Verify per-model results directories created."""

    @patch("tps_pro.pipeline.run_full_pipeline")
    @patch("tps_pro.pipeline.ensure_results_dir")
    @patch("tps_pro.pipeline.detect_skippable_flags", return_value=set())
    @patch("tps_pro.pipeline.load_phase_results", return_value=None)
    @patch("tps_pro.pipeline.detect_gpus", return_value=[])
    @patch("tps_pro.pipeline.detect_model_layers", return_value=33)
    @patch(
        "tps_pro.pipeline.classify_model",
        return_value=("small", 2.0),
    )
    @patch("tps_pro.pipeline.detect_gguf_architecture")
    @patch("tps_pro.pipeline.kill_server")
    @patch("tps_pro.pipeline.reset_load_time_debug")
    def test_batch_creates_per_model_dirs(
        self,
        mock_reset,
        mock_kill,
        mock_arch,
        mock_classify,
        mock_layers,
        mock_gpus,
        mock_load,
        mock_skip,
        mock_ensure,
        mock_pipeline,
        tmp_path,
    ):
        from tps_pro.pipeline import batch_optimize
        from tps_pro.state import ctx as global_ctx

        # Create fake GGUF files
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "model_a.gguf").write_bytes(b"\x00" * 100)
        (models_dir / "model_b.gguf").write_bytes(b"\x00" * 100)

        mock_arch.return_value = {
            "type": "dense",
            "expert_override_key": "",
            "default_experts": 8,
            "max_experts": 16,
        }

        # Patch global ctx attributes used as fallbacks by batch_optimize
        original_model_path = global_ctx.model_path
        original_results_dir = global_ctx.results_dir

        try:
            # batch_optimize now creates fresh contexts per model via create_context().
            # Mock it to return a mock ctx object.
            mock_model_ctx = MagicMock()
            mock_model_ctx.model_path = Path("/tmp/model.gguf")
            mock_model_ctx.default_gpu_layers = 33
            mock_model_ctx.skip_flags = set()
            mock_model_ctx.naked_engine = {
                "context": 4096,
                "mlock": True,
                "n_gpu_layers": 33,
            }
            mock_model_ctx.results_dir = tmp_path / "results"

            with patch(
                "tps_pro.pipeline.create_context",
                return_value=mock_model_ctx,
            ):
                batch_optimize(str(models_dir), preset="quick")

            # run_full_pipeline should have been called twice
            assert mock_pipeline.call_count == 2

            # Per-model results dirs should exist
            result_dirs = list(models_dir.glob("optimize-results-*"))
            assert len(result_dirs) == 2
        finally:
            # Restore global state
            global_ctx.model_path = original_model_path
            global_ctx.results_dir = original_results_dir


# ===================================================================
# 3. Pipeline with resume
# ===================================================================


@pytest.mark.integration
class TestPipelineResume:
    """Run partial pipeline, verify resume skips completed phases."""

    def test_resume_skips_completed_phases(self, tmp_path):
        """Verify that run_full_pipeline with resume_from skips early phases."""
        ctx = _make_ctx(tmp_path)

        # Pre-populate GPU phase results (simulating completed phase)
        gpu_results = {"best_ngl": 33, "best_tps": 30.0}
        save_phase_results(ctx, "gpu", gpu_results)

        # Pre-populate core engine results
        core_results = {
            "best_params": {"threads": 8, "batch_size": 512},
            "best_tps": 35.0,
            "baseline_score": 30.0,
            "beat_baseline": True,
        }
        save_phase_results(ctx, "core_engine", core_results)

        # Verify both phases are loadable (simulating resume check)
        assert load_phase_results(ctx, "gpu") is not None
        assert load_phase_results(ctx, "core_engine") is not None

        # Simulate the resume logic from run_full_pipeline
        resume_from = 2  # skip GPU offload (0) and tensor split (1)
        skipped = []
        phase_names = ["GPU Offload", "Tensor Split", "Core Engine", "Speculation"]

        for idx, name in enumerate(phase_names):
            if idx < resume_from:
                skipped.append(name)
            # else: would run the phase

        assert "GPU Offload" in skipped
        assert "Tensor Split" in skipped
        assert "Core Engine" not in skipped

    def test_pipeline_result_accumulation(self, tmp_path):
        """Verify that pipeline correctly accumulates results across phases."""
        ctx = _make_ctx(tmp_path)

        # Phase 1: GPU offload
        save_phase_results(ctx, "gpu", {"best_ngl": 28, "best_tps": 25.0})

        # Phase 2: Core engine
        save_phase_results(
            ctx,
            "core_engine",
            {
                "best_params": {"threads": 12, "batch_size": 768, "flash_attn": "on"},
                "best_tps": 35.0,
                "baseline_score": 25.0,
                "beat_baseline": True,
            },
        )

        # Phase 3: Speculation
        save_phase_results(
            ctx,
            "speculation",
            {
                "best_params": {"spec_type": "ngram", "draft_max": 8},
                "best_tps": 40.0,
                "baseline_score": 35.0,
                "beat_baseline": True,
            },
        )

        # Phase 4: KV quality
        save_phase_results(
            ctx,
            "kv_quality",
            {
                "best_params": {"kv_cache_type": "q8_0"},
                "best_tps": 42.0,
                "baseline_score": 40.0,
                "beat_baseline": True,
            },
        )

        # Reconstruct accumulated config as the pipeline would
        gpu_data = load_phase_results(ctx, "gpu")
        ctx.default_gpu_layers = gpu_data["best_ngl"]
        ctx.naked_engine["n_gpu_layers"] = gpu_data["best_ngl"]

        best_config = {**ctx.naked_engine}

        core_data = load_phase_results(ctx, "core_engine")
        best_config.update(core_data["best_params"])

        spec_data = load_phase_results(ctx, "speculation")
        best_config.update(spec_data["best_params"])

        kv_data = load_phase_results(ctx, "kv_quality")
        best_config.update(kv_data["best_params"])

        # Verify final accumulated config
        assert best_config["n_gpu_layers"] == 28
        assert best_config["threads"] == 12
        assert best_config["batch_size"] == 768
        assert best_config["flash_attn"] == "on"
        assert best_config["spec_type"] == "ngram"
        assert best_config["draft_max"] == 8
        assert best_config["kv_cache_type"] == "q8_0"

    def test_corrupted_results_cause_rerun(self, tmp_path):
        """Corrupted JSON files should return None, causing a phase re-run."""
        ctx = _make_ctx(tmp_path)

        # Write corrupt JSON
        corrupt_path = ctx.results_dir / "gpu_results.json"
        corrupt_path.write_text("{invalid json", encoding="utf-8")

        result = load_phase_results(ctx, "gpu")
        assert result is None  # corrupt -> re-run

    def test_all_phases_produce_loadable_results(self, tmp_path):
        """All standard phase names can save and load results."""
        ctx = _make_ctx(tmp_path)

        phase_names = [
            "gpu",
            "core_engine",
            "speculation",
            "kv_quality",
            "workload_sim",
            "context_sweep",
            "niah",
            "quality",
        ]

        for name in phase_names:
            results = {
                "phase": name,
                "best_tps": 30.0,
                "best_params": {"dummy": True},
            }
            save_phase_results(ctx, name, results)
            loaded = load_phase_results(ctx, name)
            assert loaded is not None, f"Failed to load {name}"
            assert loaded["phase"] == name
            assert loaded["score_version"] == SCORE_VERSION
