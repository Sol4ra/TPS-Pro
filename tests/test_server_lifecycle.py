"""Tests for engine/server.py — server lifecycle: start, wait, warmup, kill, boot.

Covers TestStartServer, TestWaitForServer, TestWarmupServer, TestKillServer,
TestSwapPort, TestBootServerWithJinjaRecovery, and TestFullArgBuild.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from tps_pro.engine.commands import (
    _add_base_args,
    _add_bool_flags,
    _add_extended_args,
    _add_kv_cache_args,
    _add_numeric_flag_pairs,
    _add_spec_args,
)
from tps_pro.engine.server import (
    _swap_port,
    boot_server_with_jinja_recovery,
    is_server_running,
    kill_server,
    start_server,
    wait_for_server,
    warmup_server,
)
from tps_pro.result_types import ServerProcess

# ===================================================================
# Helpers
# ===================================================================


def _make_ctx(**overrides):
    """Build a mock ctx object with sensible defaults for server tests."""
    from _ctx_factory import make_ctx_from_defaults

    return make_ctx_from_defaults(
        server_path=Path("/usr/bin/llama-server"),
        model_path=Path("/models/test.gguf"),
        server_url="http://127.0.0.1:8090",
        default_experts=8,
        http=MagicMock(spec=requests.Session),
        **overrides,
    )


def _make_server_proc(poll_return=None, stderr_lines=None):
    """Build a ServerProcess with a mocked subprocess.Popen."""
    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = poll_return
    mock_proc.pid = 12345
    mock_proc.stderr = MagicMock()
    sp = ServerProcess(
        proc=mock_proc, stderr_lines=list(stderr_lines) if stderr_lines else []
    )
    return sp


def _install_mock_psutil():
    """Install a mock psutil module in sys.modules if not already available."""
    import sys as _sys

    try:
        import psutil

        return psutil
    except ImportError:
        mock_psutil = MagicMock()
        mock_psutil.NoSuchProcess = type("NoSuchProcess", (Exception,), {})
        mock_psutil.TimeoutExpired = type("TimeoutExpired", (Exception,), {})
        mock_psutil.AccessDenied = type("AccessDenied", (Exception,), {})
        _sys.modules["psutil"] = mock_psutil
        return mock_psutil


_mock_psutil = _install_mock_psutil()


# ===================================================================
# start_server
# ===================================================================


@pytest.mark.unit
class TestStartServer:
    @patch("tps_pro.engine.server._assign_job_object")
    @patch("tps_pro.engine.server.subprocess.Popen")
    def test_normal_start_returns_server_process(self, mock_popen, mock_job):
        mock_ctx = _make_ctx()
        mock_ctx.server_path = Path("/bin/llama-server")
        mock_ctx.model_path = Path("/m.gguf")
        mock_ctx.port = 8090
        mock_ctx.no_jinja = False
        mock_ctx.chat_template_path = Path("")
        mock_ctx.debug = False
        mock_ctx.expert_override_key = ""
        mock_proc = MagicMock()
        mock_proc.pid = 42
        mock_proc.stderr = iter([])
        mock_popen.return_value = mock_proc
        result = start_server(mock_ctx, {"context": 4096})
        assert isinstance(result, ServerProcess)
        assert result.proc is mock_proc
        mock_popen.assert_called_once()
        assert mock_ctx.active_server_proc is result

    @patch("tps_pro.engine.server._assign_job_object")
    @patch("tps_pro.engine.server.subprocess.Popen")
    def test_cuda_visible_devices_set_when_no_tensor_split(self, mock_popen, mock_job):
        mock_ctx = _make_ctx()
        mock_ctx.server_path = Path("/bin/s")
        mock_ctx.model_path = Path("/m.gguf")
        mock_ctx.port = 8090
        mock_ctx.no_jinja = False
        mock_ctx.chat_template_path = Path("")
        mock_ctx.debug = False
        mock_ctx.expert_override_key = ""
        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.stderr = iter([])
        mock_popen.return_value = mock_proc
        with patch.dict("os.environ", {}, clear=True):
            start_server(mock_ctx, {})
        call_kwargs = mock_popen.call_args
        env = (
            call_kwargs[1]["env"]
            if "env" in call_kwargs[1]
            else call_kwargs.kwargs["env"]
        )
        assert env.get("CUDA_VISIBLE_DEVICES") == "0"

    @patch("tps_pro.engine.server._assign_job_object")
    @patch("tps_pro.engine.server.subprocess.Popen")
    def test_cuda_graph_opt_env(self, mock_popen, mock_job):
        mock_ctx = _make_ctx()
        mock_ctx.server_path = Path("/bin/s")
        mock_ctx.model_path = Path("/m.gguf")
        mock_ctx.port = 8090
        mock_ctx.no_jinja = False
        mock_ctx.chat_template_path = Path("")
        mock_ctx.debug = False
        mock_ctx.expert_override_key = ""
        mock_proc = MagicMock()
        mock_proc.pid = 1
        mock_proc.stderr = iter([])
        mock_popen.return_value = mock_proc
        start_server(mock_ctx, {"cuda_graph_opt": True})
        call_kwargs = mock_popen.call_args
        env = (
            call_kwargs[1]["env"]
            if "env" in call_kwargs[1]
            else call_kwargs.kwargs["env"]
        )
        assert env.get("GGML_CUDA_GRAPH_OPT") == "1"


# ===================================================================
# wait_for_server
# ===================================================================


@pytest.mark.unit
class TestWaitForServer:
    @patch("tps_pro.engine.server.warmup_server", return_value=True)
    @patch("tps_pro.engine.server._parse_load_time")
    @patch("tps_pro.engine.server.time")
    def test_server_becomes_ready(self, mock_time, mock_parse, mock_warmup):
        mock_ctx = _make_ctx()
        mock_ctx.model_size_class = "medium"
        mock_ctx.is_moe = False
        mock_ctx.debug = False
        mock_ctx.server_url = "http://127.0.0.1:8090"
        mock_time.time.side_effect = [0.0, 0.1, 0.2]
        mock_time.sleep = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_ctx.http.get.return_value = mock_response
        sp = _make_server_proc(poll_return=None)
        result = wait_for_server(mock_ctx, proc=sp, timeout=60)
        assert result == "ok"
        mock_warmup.assert_called_once()

    @patch("tps_pro.engine.server.warmup_server", return_value=True)
    @patch("tps_pro.engine.server._parse_load_time")
    @patch("tps_pro.engine.server.time")
    def test_server_times_out(self, mock_time, mock_parse, mock_warmup):
        mock_ctx = _make_ctx()
        mock_ctx.model_size_class = "medium"
        mock_ctx.is_moe = False
        mock_ctx.debug = False
        mock_ctx.server_url = "http://127.0.0.1:8090"
        call_count = [0]

        def advancing_time():
            call_count[0] += 1
            return call_count[0] * 100.0

        mock_time.time.side_effect = advancing_time
        mock_time.sleep = MagicMock()
        mock_ctx.http.get.side_effect = requests.ConnectionError("refused")
        sp = _make_server_proc(poll_return=None)
        result = wait_for_server(mock_ctx, proc=sp, timeout=10)
        assert result == "timeout"

    @patch("tps_pro.engine.server.time")
    def test_server_crashes_during_wait(self, mock_time):
        mock_ctx = _make_ctx()
        mock_ctx.model_size_class = "small"
        mock_ctx.is_moe = False
        mock_ctx.debug = False
        mock_time.time.side_effect = [0.0, 0.1]
        mock_time.sleep = MagicMock()
        sp = _make_server_proc(poll_return=1, stderr_lines=["some generic crash"])
        result = wait_for_server(mock_ctx, proc=sp, timeout=60)
        assert result == "died"

    @patch("tps_pro.engine.server.time")
    def test_server_crashes_with_oom(self, mock_time):
        mock_ctx = _make_ctx()
        mock_ctx.model_size_class = "large"
        mock_ctx.is_moe = False
        mock_ctx.debug = False
        mock_time.time.side_effect = [0.0, 0.1]
        mock_time.sleep = MagicMock()
        sp = _make_server_proc(
            poll_return=1,
            stderr_lines=["loading model...", "CUDA error: out of memory"],
        )
        result = wait_for_server(mock_ctx, proc=sp, timeout=60)
        assert result == "oom"

    @patch("tps_pro.engine.server.time")
    def test_server_crashes_with_jinja_error(self, mock_time):
        mock_ctx = _make_ctx()
        mock_ctx.model_size_class = "medium"
        mock_ctx.is_moe = False
        mock_ctx.debug = False
        mock_time.time.side_effect = [0.0, 0.1]
        mock_time.sleep = MagicMock()
        sp = _make_server_proc(
            poll_return=1, stderr_lines=["Jinja template error: unexpected token"]
        )
        result = wait_for_server(mock_ctx, proc=sp, timeout=60)
        assert result == "jinja_error"

    @patch("tps_pro.engine.server.time")
    def test_default_timeout_for_moe(self, mock_time):
        mock_ctx = _make_ctx()
        mock_ctx.model_size_class = "medium"
        mock_ctx.is_moe = True
        mock_ctx.debug = False
        call_count = [0]

        def time_past_timeout():
            call_count[0] += 1
            return call_count[0] * 500.0

        mock_time.time.side_effect = time_past_timeout
        mock_time.sleep = MagicMock()
        mock_ctx.http.get.side_effect = requests.ConnectionError()
        sp = _make_server_proc(poll_return=None)
        result = wait_for_server(mock_ctx, proc=sp)
        assert result == "timeout"


# ===================================================================
# warmup_server
# ===================================================================


@pytest.mark.unit
class TestWarmupServer:
    @patch("tps_pro.hardware.init_vram_info")
    @patch("tps_pro.engine.server.time")
    def test_successful_warmup(self, mock_time, mock_vram):
        mock_ctx = _make_ctx()
        mock_ctx.server_url = "http://127.0.0.1:8090"
        mock_ctx.active_server_proc = _make_server_proc()
        mock_ctx.vram_total_mb = 8192.0
        mock_time.time.side_effect = [0.0, 0.5]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_ctx.http.post.return_value = mock_response
        result = warmup_server(mock_ctx)
        assert result is True
        assert mock_ctx.http.post.call_count == 2
        mock_vram.assert_not_called()

    @patch("tps_pro.engine.server.init_vram_info")
    @patch("tps_pro.engine.server.time")
    def test_warmup_inits_vram_when_none(self, mock_time, mock_vram):
        mock_ctx = _make_ctx()
        mock_ctx.server_url = "http://127.0.0.1:8090"
        mock_ctx.active_server_proc = _make_server_proc()
        mock_ctx.vram_total_mb = None
        mock_time.time.side_effect = [0.0, 0.5]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_ctx.http.post.return_value = mock_response
        warmup_server(mock_ctx)
        mock_vram.assert_called_once()

    @patch("tps_pro.engine.server.time")
    def test_warmup_stage1_failure(self, mock_time):
        mock_ctx = _make_ctx()
        mock_ctx.server_url = "http://127.0.0.1:8090"
        mock_ctx.active_server_proc = _make_server_proc()
        mock_time.time.side_effect = [0.0]
        mock_ctx.http.post.side_effect = requests.ConnectionError("refused")
        result = warmup_server(mock_ctx)
        assert result is False

    @patch("tps_pro.engine.server.time")
    def test_warmup_stage1_500_error(self, mock_time):
        mock_ctx = _make_ctx()
        mock_ctx.server_url = "http://127.0.0.1:8090"
        mock_ctx.active_server_proc = MagicMock()
        mock_time.time.side_effect = [0.0]
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_ctx.http.post.return_value = mock_response
        result = warmup_server(mock_ctx)
        assert result is False

    @patch("tps_pro.engine.server.time")
    def test_warmup_stage2_failure(self, mock_time):
        mock_ctx = _make_ctx()
        mock_ctx.server_url = "http://127.0.0.1:8090"
        mock_ctx.active_server_proc = MagicMock()
        mock_time.time.side_effect = [0.0]
        response_ok = MagicMock()
        response_ok.status_code = 200
        mock_ctx.http.post.side_effect = [response_ok, requests.Timeout("timed out")]
        result = warmup_server(mock_ctx)
        assert result is False

    @patch("tps_pro.engine.server.time")
    def test_warmup_stage2_500(self, mock_time):
        mock_ctx = _make_ctx()
        mock_ctx.server_url = "http://127.0.0.1:8090"
        mock_ctx.active_server_proc = MagicMock()
        mock_time.time.side_effect = [0.0]
        response_ok = MagicMock()
        response_ok.status_code = 200
        response_500 = MagicMock()
        response_500.status_code = 500
        mock_ctx.http.post.side_effect = [response_ok, response_500]
        result = warmup_server(mock_ctx)
        assert result is False


# ===================================================================
# kill_server
# ===================================================================


@pytest.mark.unit
class TestKillServer:
    @patch("tps_pro.engine.server.time")
    @patch("tps_pro.engine.server.kill_process_tree")
    def test_normal_kill(self, mockkill_process_tree, mock_time):
        mock_ctx = _make_ctx()
        sp = _make_server_proc()
        mock_ctx.active_server_proc = sp
        mock_ctx._dying_server_proc = None
        mock_ctx.port = 8090
        mock_ctx.http = MagicMock(spec=requests.Session)
        mock_time.sleep = MagicMock()
        mock_sock_inst = MagicMock()
        mock_sock_inst.connect_ex.return_value = 1
        mock_sock_inst.__enter__ = MagicMock(return_value=mock_sock_inst)
        mock_sock_inst.__exit__ = MagicMock(return_value=False)
        with (
            patch("psutil.Process") as mock_ps_process,
            patch("socket.socket", return_value=mock_sock_inst),
        ):
            mock_ps_process.return_value.wait.return_value = None
            kill_server(mock_ctx, wait=True)
        mockkill_process_tree.assert_called_once_with(sp)
        assert mock_ctx.active_server_proc is None

    @patch("tps_pro.engine.server.kill_process_tree")
    def test_kill_no_wait_tracks_dying(self, mockkill_process_tree):
        mock_ctx = _make_ctx()
        sp = _make_server_proc()
        mock_ctx.active_server_proc = sp
        mock_ctx._dying_server_proc = None
        mock_ctx.http = MagicMock(spec=requests.Session)
        with patch("psutil.Process") as mock_ps:
            mock_ps.side_effect = _mock_psutil.NoSuchProcess(0)
            kill_server(mock_ctx, wait=False)
        mockkill_process_tree.assert_called_once_with(sp)
        assert mock_ctx._dying_server_proc is sp
        assert mock_ctx.active_server_proc is None

    @patch("tps_pro.engine.server.time")
    @patch("tps_pro.engine.server.kill_process_tree")
    def test_kill_when_no_active_proc(self, mockkill_process_tree, mock_time):
        mock_ctx = _make_ctx()
        mock_ctx.active_server_proc = None
        mock_ctx._dying_server_proc = None
        mock_ctx.port = 8090
        mock_ctx.http = MagicMock(spec=requests.Session)
        mock_time.sleep = MagicMock()
        mock_sock_inst = MagicMock()
        mock_sock_inst.connect_ex.return_value = 1
        mock_sock_inst.__enter__ = MagicMock(return_value=mock_sock_inst)
        mock_sock_inst.__exit__ = MagicMock(return_value=False)
        with (
            patch("psutil.Process"),
            patch("socket.socket", return_value=mock_sock_inst),
        ):
            kill_server(mock_ctx, wait=True)
        mockkill_process_tree.assert_called_once_with(None)

    @patch("tps_pro.engine.server.kill_process_tree")
    def test_kill_reaps_dying_proc(self, mockkill_process_tree):
        dying = _make_server_proc()
        active = _make_server_proc()
        mock_ctx = _make_ctx()
        mock_ctx._dying_server_proc = dying
        mock_ctx.active_server_proc = active
        mock_ctx.http = MagicMock(spec=requests.Session)
        with patch("psutil.Process") as mock_ps:
            mock_ps.return_value.wait.return_value = None
            kill_server(mock_ctx, wait=False)
        assert mock_ctx._dying_server_proc is active


# ===================================================================
# _swap_port
# ===================================================================


@pytest.mark.unit
class TestSwapPort:
    def test_port_alternation(self):
        mock_ctx = _make_ctx()
        mock_ctx.port = 8090
        mock_ctx._port_alt = 8091
        _swap_port(mock_ctx)
        assert mock_ctx.port == 8091
        assert mock_ctx._port_alt == 8090
        assert mock_ctx.server_url == "http://127.0.0.1:8091"

    def test_double_swap_restores(self):
        mock_ctx = _make_ctx()
        mock_ctx.port = 8090
        mock_ctx._port_alt = 8091
        _swap_port(mock_ctx)
        _swap_port(mock_ctx)
        assert mock_ctx.port == 8090
        assert mock_ctx._port_alt == 8091
        assert mock_ctx.server_url == "http://127.0.0.1:8090"


# ===================================================================
# boot_server_with_jinja_recovery
# ===================================================================


@pytest.mark.unit
class TestBootServerWithJinjaRecovery:
    @patch("tps_pro.engine.server.time")
    @patch("tps_pro.engine.server.wait_for_server", return_value="ok")
    @patch("tps_pro.engine.server.start_server")
    def test_normal_boot_succeeds(self, mock_start, mock_wait, mock_time):
        mock_ctx = _make_ctx()
        mock_ctx.active_server_proc = None
        mock_ctx.no_jinja = False
        mock_time.time.side_effect = [0.0, 1.0]
        mock_sp = _make_server_proc()
        mock_start.return_value = mock_sp
        proc, status = boot_server_with_jinja_recovery(mock_ctx, {"context": 4096})
        assert status == "ok"
        assert proc.proc is mock_sp.proc
        assert proc.boot_time_ms is not None
        mock_start.assert_called_once()

    @patch("tps_pro.engine.server.time")
    @patch("tps_pro.engine.server.wait_for_server")
    @patch("tps_pro.engine.server.start_server")
    @patch("tps_pro.engine.server._swap_port")
    @patch("tps_pro.engine.server.kill_server")
    def test_jinja_error_triggers_retry(
        self, mock_kill, mock_swap, mock_start, mock_wait, mock_time
    ):
        mock_ctx = _make_ctx()
        mock_ctx.active_server_proc = None
        mock_ctx.no_jinja = False
        mock_time.time.side_effect = [0.0, 2.0]
        sp1 = _make_server_proc()
        sp2 = _make_server_proc()
        mock_start.side_effect = [sp1, sp2]
        mock_wait.side_effect = ["jinja_error", "ok"]
        proc, status = boot_server_with_jinja_recovery(mock_ctx, {"context": 4096})
        assert status == "ok"
        assert mock_ctx.no_jinja is True
        assert mock_start.call_count == 2
        mock_kill.assert_called_with(mock_ctx, wait=False)
        mock_swap.assert_called_with(mock_ctx)

    @patch("tps_pro.engine.server.time")
    @patch("tps_pro.engine.server.wait_for_server")
    @patch("tps_pro.engine.server.start_server")
    @patch("tps_pro.engine.server._swap_port")
    @patch("tps_pro.engine.server.kill_server")
    def test_jinja_retry_also_fails(
        self, mock_kill, mock_swap, mock_start, mock_wait, mock_time
    ):
        mock_ctx = _make_ctx()
        mock_ctx.active_server_proc = None
        mock_ctx.no_jinja = False
        mock_time.time.side_effect = [0.0, 3.0]
        sp1 = _make_server_proc()
        sp2 = _make_server_proc()
        mock_start.side_effect = [sp1, sp2]
        mock_wait.side_effect = ["jinja_error", "died"]
        proc, status = boot_server_with_jinja_recovery(mock_ctx, {"context": 4096})
        assert status == "died"
        assert mock_start.call_count == 2

    @patch("tps_pro.engine.server.time")
    @patch("tps_pro.engine.server.wait_for_server", return_value="oom")
    @patch("tps_pro.engine.server.start_server")
    def test_oom_no_retry(self, mock_start, mock_wait, mock_time):
        mock_ctx = _make_ctx()
        mock_ctx.active_server_proc = None
        mock_ctx.no_jinja = False
        mock_time.time.side_effect = [0.0, 1.0]
        mock_start.return_value = _make_server_proc()
        proc, status = boot_server_with_jinja_recovery(mock_ctx, {"context": 4096})
        assert status == "oom"
        mock_start.assert_called_once()

    @patch("tps_pro.engine.server.time")
    @patch("tps_pro.engine.server.wait_for_server", return_value="ok")
    def test_with_existing_proc(self, mock_wait, mock_time):
        mock_ctx = _make_ctx()
        mock_ctx.no_jinja = False
        mock_time.time.side_effect = [0.0, 0.5]
        existing = _make_server_proc()
        proc, status = boot_server_with_jinja_recovery(
            mock_ctx, {"context": 4096}, proc=existing
        )
        assert status == "ok"
        assert proc.proc is existing.proc

    @patch("tps_pro.engine.server.time")
    @patch("tps_pro.engine.server.wait_for_server", return_value="ok")
    @patch("tps_pro.engine.server.start_server")
    @patch("tps_pro.engine.server._swap_port")
    @patch("tps_pro.engine.server.kill_server")
    def test_ping_pong_when_active_proc_exists(
        self, mock_kill, mock_swap, mock_start, mock_wait, mock_time
    ):
        mock_ctx = _make_ctx()
        mock_ctx.active_server_proc = _make_server_proc()
        mock_ctx.no_jinja = False
        mock_time.time.side_effect = [0.0, 0.5]
        new_sp = _make_server_proc()
        mock_start.return_value = new_sp
        proc, status = boot_server_with_jinja_recovery(mock_ctx, {"context": 4096})
        assert status == "ok"
        mock_kill.assert_called_once_with(mock_ctx, wait=False)
        mock_swap.assert_called_once_with(mock_ctx)


# ===================================================================
# Integration: full arg building pipeline
# ===================================================================


@pytest.mark.unit
class TestFullArgBuild:
    """Verify the complete command line built by start_server's internal helpers."""

    def test_comprehensive_config(self):
        mock_ctx = _make_ctx()
        mock_ctx.server_path = Path("/bin/llama-server")
        mock_ctx.model_path = Path("/model.gguf")
        mock_ctx.port = 9000
        mock_ctx.no_jinja = False
        mock_ctx.chat_template_path = Path("")
        mock_ctx.expert_override_key = ""
        mock_ctx.default_experts = 8
        config = {
            "n_gpu_layers": 40,
            "context": 8192,
            "parallel": 4,
            "batch_size": 1024,
            "threads": 16,
            "kv_cache_type": "q8_0",
            "flash_attn": True,
            "mlock": True,
            "no_mmap": True,
            "numa": "distribute",
        }
        cmd = _add_base_args(mock_ctx, config)
        cmd.extend(_add_numeric_flag_pairs(mock_ctx, config))
        cmd.extend(_add_kv_cache_args(config))
        cmd.extend(_add_spec_args(config))
        cmd.extend(_add_bool_flags(config))
        cmd.extend(_add_extended_args(config))
        assert cmd[0] == str(Path("/bin/llama-server"))
        assert "-ngl" in cmd and cmd[cmd.index("-ngl") + 1] == "40"
        assert "-b" in cmd and cmd[cmd.index("-b") + 1] == "1024"
        assert "--cache-type-k" in cmd
        assert "--flash-attn" in cmd and cmd[cmd.index("--flash-attn") + 1] == "1"
        assert "--mlock" in cmd
        assert "--no-mmap" in cmd
        assert "--numa" in cmd and cmd[cmd.index("--numa") + 1] == "distribute"


# ===================================================================
# is_server_running
# ===================================================================


@pytest.mark.unit
class TestIsServerRunning:
    @patch("tps_pro.engine.server.requests.get")
    def test_returns_true_when_healthy(self, mock_get):
        mock_ctx = _make_ctx()
        mock_ctx.server_url = "http://127.0.0.1:8090"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        assert is_server_running(mock_ctx) is True
        mock_get.assert_called_once_with("http://127.0.0.1:8090/health", timeout=0.15)

    @patch("tps_pro.engine.server.requests.get")
    def test_returns_false_when_no_server(self, mock_get):
        """Connection refused should return False quickly (under 200ms)."""
        import time

        mock_ctx = _make_ctx()
        mock_ctx.server_url = "http://127.0.0.1:8090"
        mock_get.side_effect = requests.ConnectionError("refused")
        t0 = time.monotonic()
        result = is_server_running(mock_ctx)
        elapsed_ms = (time.monotonic() - t0) * 1000
        assert result is False
        assert elapsed_ms < 200, (
            f"is_server_running took {elapsed_ms:.0f}ms, expected <200ms"
        )

    @patch("tps_pro.engine.server.requests.get")
    def test_returns_false_on_non_200(self, mock_get):
        mock_ctx = _make_ctx()
        mock_ctx.server_url = "http://127.0.0.1:8090"
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_get.return_value = mock_response
        assert is_server_running(mock_ctx) is False

    @patch("tps_pro.engine.server.requests.get")
    def test_returns_false_on_timeout(self, mock_get):
        mock_ctx = _make_ctx()
        mock_ctx.server_url = "http://127.0.0.1:8090"
        mock_get.side_effect = requests.Timeout("timed out")
        assert is_server_running(mock_ctx) is False


# ===================================================================
# kill_server -- already-dead process
# ===================================================================


@pytest.mark.unit
class TestKillServerAlreadyDead:
    @patch("tps_pro.engine.server.time")
    @patch("tps_pro.engine.server.kill_process_tree")
    def test_handles_already_dead_process(self, mock_kill_tree, mock_time):
        """kill_server should not raise when psutil.Process raises NoSuchProcess."""
        mock_ctx = _make_ctx()
        sp = _make_server_proc()
        mock_ctx.active_server_proc = sp
        mock_ctx._dying_server_proc = None
        mock_ctx.port = 8090
        mock_ctx.http = MagicMock(spec=requests.Session)
        mock_time.sleep = MagicMock()
        mock_sock_inst = MagicMock()
        mock_sock_inst.connect_ex.return_value = 1  # port free
        mock_sock_inst.__enter__ = MagicMock(return_value=mock_sock_inst)
        mock_sock_inst.__exit__ = MagicMock(return_value=False)
        with (
            patch("psutil.Process") as mock_ps_process,
            patch("socket.socket", return_value=mock_sock_inst),
        ):
            mock_ps_process.side_effect = _mock_psutil.NoSuchProcess(0)
            # Should not raise
            kill_server(mock_ctx, wait=True)
        mock_kill_tree.assert_called_once_with(sp)
        assert mock_ctx.active_server_proc is None
