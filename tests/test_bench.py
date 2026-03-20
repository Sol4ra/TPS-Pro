"""Tests for engine/bench.py — llama-bench integration.

Mocks subprocess so tests run without a real llama-bench binary.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tps_pro.engine.bench import (
    BenchOOMError,
    _build_bench_cmd,
    _parse_bench_csv,
    run_bench_trial,
)
from tps_pro.result_types import BenchResult

# ===================================================================
# Helpers
# ===================================================================


def _make_ctx(**overrides):
    """Build a minimal mock ctx for bench tests."""
    from _ctx_factory import make_ctx_from_defaults

    bench_defaults = dict(
        bench_path=Path("/usr/bin/llama-bench"),
        model_path=Path("/models/test.gguf"),
    )
    bench_defaults.update(overrides)
    return make_ctx_from_defaults(**bench_defaults)


# ===================================================================
# _build_bench_cmd
# ===================================================================


@pytest.mark.unit
class TestBuildBenchCmd:
    def test_basic_command(self):
        ctx = _make_ctx()
        cmd = _build_bench_cmd(ctx, {}, n_prompt=256, n_gen=64, repetitions=2)
        assert cmd[0] == str(ctx.bench_path)
        assert "-m" in cmd
        assert str(ctx.model_path) in cmd
        assert "-p" in cmd
        idx_p = cmd.index("-p")
        assert cmd[idx_p + 1] == "256"
        idx_n = cmd.index("-n")
        assert cmd[idx_n + 1] == "64"
        idx_r = cmd.index("-r")
        assert cmd[idx_r + 1] == "2"
        assert "-o" in cmd
        assert "csv" in cmd

    def test_flag_map_params(self):
        ctx = _make_ctx()
        config = {"n_gpu_layers": 35, "threads": 8, "batch_size": 512}
        cmd = _build_bench_cmd(ctx, config)
        assert "-ngl" in cmd
        assert "35" in cmd
        assert "-t" in cmd
        assert "8" in cmd
        assert "-b" in cmd
        assert "512" in cmd

    def test_kv_cache_type(self):
        ctx = _make_ctx()
        config = {"kv_cache_type": "q4_0"}
        cmd = _build_bench_cmd(ctx, config)
        assert "-ctk" in cmd
        assert "-ctv" in cmd
        # Both should be set to q4_0
        idx_ctk = cmd.index("-ctk")
        assert cmd[idx_ctk + 1] == "q4_0"

    def test_flash_attn_on(self):
        ctx = _make_ctx()
        config = {"flash_attn": "on"}
        cmd = _build_bench_cmd(ctx, config)
        assert "-fa" in cmd
        idx = cmd.index("-fa")
        assert cmd[idx + 1] == "1"

    def test_flash_attn_off(self):
        ctx = _make_ctx()
        config = {"flash_attn": "off"}
        cmd = _build_bench_cmd(ctx, config)
        assert "-fa" in cmd
        idx = cmd.index("-fa")
        assert cmd[idx + 1] == "0"

    def test_no_mmap(self):
        ctx = _make_ctx()
        config = {"no_mmap": True}
        cmd = _build_bench_cmd(ctx, config)
        assert "-mmp" in cmd
        idx = cmd.index("-mmp")
        assert cmd[idx + 1] == "0"

    def test_separate_cache_types(self):
        ctx = _make_ctx()
        config = {"cache_type_k": "q8_0", "cache_type_v": "q4_0"}
        cmd = _build_bench_cmd(ctx, config)
        idx_ctk = cmd.index("-ctk")
        assert cmd[idx_ctk + 1] == "q8_0"
        idx_ctv = cmd.index("-ctv")
        assert cmd[idx_ctv + 1] == "q4_0"


# ===================================================================
# _parse_bench_csv
# ===================================================================


@pytest.mark.unit
class TestParseBenchCsv:
    def test_valid_csv(self):
        csv_data = (
            "build_commit,build_number,cuda,vulkan,opencl,gpu_info,model_filename,model_type,"
            "model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_info,gpu_info_v,"
            "n_gpu_layers,split_mode,main_gpu,no_kv_offload,flash_attn,tensor_split,"
            "use_mmap,embeddings,n_prompt,n_gen,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts\n"
            "abc,1,1,0,0,gpu,model.gguf,7B,3.5G,7000000000,512,512,8,cpu,gpu,"
            "35,none,0,0,0,,1,0,512,0,2024-01-01,1000000.0,50000.0,512.0,25.0\n"
            "abc,1,1,0,0,gpu,model.gguf,7B,3.5G,7000000000,512,512,8,cpu,gpu,"
            "35,none,0,0,0,,1,0,0,128,2024-01-01,5000000.0,200000.0,25.6,1.2\n"
        )
        result = _parse_bench_csv(csv_data)
        assert result is not None
        assert isinstance(result, BenchResult)
        assert result.tps == pytest.approx(25.6)
        assert result.prompt_tps == pytest.approx(512.0)

    def test_empty_input(self):
        result = _parse_bench_csv("")
        assert result is None

    def test_malformed_csv(self):
        result = _parse_bench_csv("not,valid,csv\ndata")
        assert result is None

    def test_no_gen_rows(self):
        """If there are no generation rows, result should be None."""
        csv_data = (
            "build_commit,build_number,cuda,vulkan,opencl,gpu_info,model_filename,model_type,"
            "model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_info,gpu_info_v,"
            "n_gpu_layers,split_mode,main_gpu,no_kv_offload,flash_attn,tensor_split,"
            "use_mmap,embeddings,n_prompt,n_gen,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts\n"
            "abc,1,1,0,0,gpu,model.gguf,7B,3.5G,7000000000,512,512,8,cpu,gpu,"
            "35,none,0,0,0,,1,0,512,0,2024-01-01,1000000.0,50000.0,512.0,25.0\n"
        )
        result = _parse_bench_csv(csv_data)
        assert result is None


# ===================================================================
# run_bench_trial
# ===================================================================


@pytest.mark.unit
class TestRunBenchTrial:
    def test_no_bench_path_returns_none(self):
        ctx = _make_ctx(bench_path=None)
        result = run_bench_trial(ctx, {})
        assert result is None

    @patch("tps_pro.engine.bench.subprocess.run")
    def test_success_returns_bench_result(self, mock_run):
        csv_output = (
            "build_commit,build_number,cuda,vulkan,opencl,gpu_info,model_filename,model_type,"
            "model_size,model_n_params,n_batch,n_ubatch,n_threads,cpu_info,gpu_info_v,"
            "n_gpu_layers,split_mode,main_gpu,no_kv_offload,flash_attn,tensor_split,"
            "use_mmap,embeddings,n_prompt,n_gen,test_time,avg_ns,stddev_ns,avg_ts,stddev_ts\n"
            "abc,1,1,0,0,gpu,model.gguf,7B,3.5G,7000000000,512,512,8,cpu,gpu,"
            "35,none,0,0,0,,1,0,512,0,2024-01-01,1000000.0,50000.0,512.0,25.0\n"
            "abc,1,1,0,0,gpu,model.gguf,7B,3.5G,7000000000,512,512,8,cpu,gpu,"
            "35,none,0,0,0,,1,0,0,128,2024-01-01,5000000.0,200000.0,25.6,1.2\n"
        )
        mock_run.return_value = MagicMock(returncode=0, stdout=csv_output, stderr="")
        ctx = _make_ctx()
        result = run_bench_trial(ctx, {"n_gpu_layers": 35})
        assert isinstance(result, BenchResult)
        assert result.tps == pytest.approx(25.6)

    @patch("tps_pro.engine.bench.is_oom", return_value=True)
    @patch("tps_pro.engine.bench.subprocess.run")
    def test_oom_raises_bench_oom_error(self, mock_run, mock_oom):
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="out of memory"
        )
        ctx = _make_ctx()
        with pytest.raises(BenchOOMError):
            run_bench_trial(ctx, {})

    @patch("tps_pro.engine.bench.subprocess.run")
    def test_timeout_returns_none(self, mock_run):
        import subprocess

        mock_run.side_effect = subprocess.TimeoutExpired(cmd="bench", timeout=300)
        ctx = _make_ctx()
        result = run_bench_trial(ctx, {})
        assert result is None

    @patch("tps_pro.engine.bench.is_oom", return_value=False)
    @patch("tps_pro.engine.bench.subprocess.run")
    def test_nonzero_exit_non_oom_returns_none(self, mock_run, mock_oom):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="some error")
        ctx = _make_ctx()
        result = run_bench_trial(ctx, {})
        assert result is None
