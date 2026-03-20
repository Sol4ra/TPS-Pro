"""Tests for engine/server.py — ServerProcess dataclass, parsing helpers, and
command building.

Covers ServerProcess, _is_oom, _is_error_line, _parse_load_time, and all _add_*_args
command building functions.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock

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
from tps_pro.engine.parsing import (
    _is_error_line,
    _parse_load_time,
    is_oom,
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


# ===================================================================
# ServerProcess dataclass
# ===================================================================


@pytest.mark.unit
class TestServerProcess:
    def test_creation_defaults(self):
        mock_proc = MagicMock(spec=subprocess.Popen)
        sp = ServerProcess(proc=mock_proc)
        assert sp.proc is mock_proc
        assert sp.stderr_lines == []
        assert sp.load_time_ms is None
        assert sp.warmup_time_ms is None
        assert sp.boot_time_ms is None
        assert isinstance(sp.lock, type(threading.Lock()))

    def test_field_assignment(self):
        mock_proc = MagicMock(spec=subprocess.Popen)
        sp = ServerProcess(proc=mock_proc, load_time_ms=1234.0, warmup_time_ms=56.0)
        assert sp.load_time_ms == 1234.0
        assert sp.warmup_time_ms == 56.0


# ===================================================================
# _is_oom
# ===================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "text,expected",
    [
        ("CUDA error: out of memory", True),
        ("alloc failed for tensor", True),
        ("OOM killed", True),
        ("cudamalloc failed", True),
        ("insufficient memory for allocation", True),
        ("Server started successfully", False),
        ("model loaded in 1234 ms", False),
        ("", False),
    ],
)
def test_is_oom(text, expected):
    assert is_oom(text) is expected


# ===================================================================
# _is_error_line
# ===================================================================


@pytest.mark.unit
@pytest.mark.parametrize(
    "line,expected",
    [
        ("error: failed to load model", True),
        ("fatal: segfault in ggml", True),
        ("abort trap: 6", True),
        ("model loaded successfully", False),
        ("server initialized and ready", False),
        ("", False),
        ("   ", False),
        ("some random log line", False),
    ],
)
def test_is_error_line(line, expected):
    assert _is_error_line(line) is expected


# ===================================================================
# _parse_load_time
# ===================================================================


@pytest.mark.unit
class TestParseLoadTime:
    def test_classic_pattern(self):
        sp = _make_server_proc(stderr_lines=["llm_load_tensors: loaded in 2345.67 ms"])
        result = _parse_load_time(sp)
        assert result == pytest.approx(2345.67)

    def test_newer_pattern(self):
        sp = _make_server_proc(
            stderr_lines=["llama_model_load: model loaded in 1500 ms"]
        )
        result = _parse_load_time(sp)
        assert result == pytest.approx(1500.0)

    def test_load_time_pattern(self):
        sp = _make_server_proc(stderr_lines=["load_time = 800.5 ms"])
        result = _parse_load_time(sp)
        assert result == pytest.approx(800.5)

    def test_no_match_returns_none(self):
        sp = _make_server_proc(
            stderr_lines=["some unrelated log line", "another line without timing info"]
        )
        result = _parse_load_time(sp)
        assert result is None

    def test_tiny_value_filtered(self):
        sp = _make_server_proc(stderr_lines=["llm_load_tensors: loaded in 10.0 ms"])
        result = _parse_load_time(sp)
        assert result is None


# ===================================================================
# build_server_args — _add_base_args
# ===================================================================


@pytest.mark.unit
class TestAddBaseArgs:
    def test_minimal_config(self):
        mock_ctx = _make_ctx()
        mock_ctx.server_path = Path("/bin/llama-server")
        mock_ctx.model_path = Path("/model.gguf")
        mock_ctx.port = 8090
        mock_ctx.no_jinja = False
        mock_ctx.chat_template_path = Path("")
        config = {"n_gpu_layers": 33, "context": 2048, "parallel": 2}
        args = _add_base_args(mock_ctx, config)
        assert args[0] == str(Path("/bin/llama-server"))
        assert "-m" in args
        assert args[args.index("-m") + 1] == str(Path("/model.gguf"))
        assert "--port" in args
        assert args[args.index("--port") + 1] == "8090"
        assert "-ngl" in args
        assert args[args.index("-ngl") + 1] == "33"
        assert "-c" in args
        assert args[args.index("-c") + 1] == "2048"
        assert "--parallel" in args
        assert args[args.index("--parallel") + 1] == "2"

    def test_defaults_for_missing_keys(self):
        mock_ctx = _make_ctx()
        mock_ctx.server_path = Path("/bin/s")
        mock_ctx.model_path = Path("/m.gguf")
        mock_ctx.port = 9000
        mock_ctx.no_jinja = False
        mock_ctx.chat_template_path = Path("")
        args = _add_base_args(mock_ctx, {})
        assert args[args.index("-ngl") + 1] == "99"
        assert args[args.index("-c") + 1] == "4096"
        assert args[args.index("--parallel") + 1] == "1"

    def test_no_jinja_flag(self):
        mock_ctx = _make_ctx()
        mock_ctx.server_path = Path("/bin/s")
        mock_ctx.model_path = Path("/m.gguf")
        mock_ctx.port = 8090
        mock_ctx.no_jinja = True
        mock_ctx.chat_template_path = Path("")
        args = _add_base_args(mock_ctx, {})
        assert "--no-jinja" in args
        assert "--chat-template" in args
        assert args[args.index("--chat-template") + 1] == "chatml"

    def test_no_warmup_appended_when_warmup_not_true(self):
        mock_ctx = _make_ctx()
        mock_ctx.server_path = Path("/bin/s")
        mock_ctx.model_path = Path("/m.gguf")
        mock_ctx.port = 8090
        mock_ctx.no_jinja = False
        mock_ctx.chat_template_path = Path("")
        args = _add_base_args(mock_ctx, {"warmup": False})
        assert "--no-warmup" in args

    def test_warmup_true_no_flag(self):
        mock_ctx = _make_ctx()
        mock_ctx.server_path = Path("/bin/s")
        mock_ctx.model_path = Path("/m.gguf")
        mock_ctx.port = 8090
        mock_ctx.no_jinja = False
        mock_ctx.chat_template_path = Path("")
        args = _add_base_args(mock_ctx, {"warmup": True})
        assert "--no-warmup" not in args

    def test_cache_prompt_false(self):
        mock_ctx = _make_ctx()
        mock_ctx.server_path = Path("/bin/s")
        mock_ctx.model_path = Path("/m.gguf")
        mock_ctx.port = 8090
        mock_ctx.no_jinja = False
        mock_ctx.chat_template_path = Path("")
        args = _add_base_args(mock_ctx, {"cache_prompt": False})
        assert "--no-cache-prompt" in args

    def test_fit_false(self):
        mock_ctx = _make_ctx()
        mock_ctx.server_path = Path("/bin/s")
        mock_ctx.model_path = Path("/m.gguf")
        mock_ctx.port = 8090
        mock_ctx.no_jinja = False
        mock_ctx.chat_template_path = Path("")
        args = _add_base_args(mock_ctx, {"fit": False})
        assert "--fit=off" in args


# ===================================================================
# _add_numeric_flag_pairs
# ===================================================================


@pytest.mark.unit
class TestAddNumericFlagPairs:
    def test_batch_and_threads(self):
        mock_ctx = _make_ctx()
        mock_ctx.expert_override_key = ""
        config = {
            "batch_size": 512,
            "ubatch_size": 128,
            "threads": 8,
            "threads_batch": 4,
        }
        args = _add_numeric_flag_pairs(mock_ctx, config)
        assert "-b" in args and args[args.index("-b") + 1] == "512"
        assert (
            "--ubatch-size" in args and args[args.index("--ubatch-size") + 1] == "128"
        )
        assert "-t" in args and args[args.index("-t") + 1] == "8"
        assert "-tb" in args and args[args.index("-tb") + 1] == "4"

    def test_expert_override(self):
        mock_ctx = _make_ctx()
        mock_ctx.expert_override_key = "llama.expert_used_count"
        mock_ctx.default_experts = 8
        config = {"expert_used_count": 4}
        args = _add_numeric_flag_pairs(mock_ctx, config)
        assert "--override-kv" in args
        assert "llama.expert_used_count=int:4" in args[args.index("--override-kv") + 1]

    def test_expert_override_skipped_when_default(self):
        mock_ctx = _make_ctx()
        mock_ctx.expert_override_key = "llama.expert_used_count"
        mock_ctx.default_experts = 8
        config = {"expert_used_count": 8}
        args = _add_numeric_flag_pairs(mock_ctx, config)
        assert "--override-kv" not in args

    def test_toggle_pairs(self):
        mock_ctx = _make_ctx()
        mock_ctx.expert_override_key = ""
        config = {"poll": 1, "prio": 2}
        args = _add_numeric_flag_pairs(mock_ctx, config)
        assert "--poll" in args
        assert "--prio" in args

    def test_empty_config(self):
        mock_ctx = _make_ctx()
        mock_ctx.expert_override_key = ""
        assert _add_numeric_flag_pairs(mock_ctx, {}) == []


# ===================================================================
# _add_kv_cache_args
# ===================================================================


@pytest.mark.unit
class TestAddKvCacheArgs:
    def test_kv_cache_type_sets_both(self):
        args = _add_kv_cache_args({"kv_cache_type": "q8_0"})
        assert args.count("--cache-type-k") == 1
        assert args.count("--cache-type-v") == 1
        assert args[args.index("--cache-type-k") + 1] == "q8_0"
        assert args[args.index("--cache-type-v") + 1] == "q8_0"

    def test_separate_k_v_types(self):
        args = _add_kv_cache_args({"cache_type_k": "q4_0", "cache_type_v": "q8_0"})
        assert args[args.index("--cache-type-k") + 1] == "q4_0"
        assert args[args.index("--cache-type-v") + 1] == "q8_0"

    @pytest.mark.parametrize("val", ["on", True, "1", 1])
    def test_flash_attn_on(self, val):
        args = _add_kv_cache_args({"flash_attn": val})
        assert "--flash-attn" in args
        assert args[args.index("--flash-attn") + 1] == "1"

    @pytest.mark.parametrize("val", ["off", False, "0", 0])
    def test_flash_attn_off(self, val):
        args = _add_kv_cache_args({"flash_attn": val})
        assert "--flash-attn" in args
        assert args[args.index("--flash-attn") + 1] == "0"

    def test_n_predict_and_temp(self):
        args = _add_kv_cache_args({"n_predict": 512, "temp": 0.7})
        assert "--n-predict" in args and args[args.index("--n-predict") + 1] == "512"
        assert "--temp" in args and args[args.index("--temp") + 1] == "0.7"

    def test_model_draft(self, tmp_path):
        draft_file = tmp_path / "draft.gguf"
        draft_file.write_text("fake")
        args = _add_kv_cache_args({"model_draft": str(draft_file)})
        assert "--model-draft" in args

    def test_model_draft_missing_file(self):
        args = _add_kv_cache_args({"model_draft": "/nonexistent/draft.gguf"})
        assert "--model-draft" not in args

    def test_cache_reuse(self):
        args = _add_kv_cache_args({"cache_reuse": 256})
        assert "--cache-reuse" in args

    def test_empty_config(self):
        assert _add_kv_cache_args({}) == []


# ===================================================================
# _add_spec_args
# ===================================================================


@pytest.mark.unit
class TestAddSpecArgs:
    def test_speculation_params(self):
        config = {
            "spec_type": "ngram",
            "spec_ngram_n": 4,
            "draft_max": 16,
            "draft_min": 2,
            "draft_p_min": 0.5,
        }
        args = _add_spec_args(config)
        assert "--spec-type" in args and args[args.index("--spec-type") + 1] == "ngram"
        assert "--spec-ngram-size-n" in args
        assert "--draft" in args
        assert "--draft-min" in args
        assert "--draft-p-min" in args

    def test_cpu_strict(self):
        args = _add_spec_args({"cpu_strict": 1, "cpu_strict_batch": 0})
        assert "--cpu-strict" in args
        assert "--cpu-strict-batch" in args

    def test_empty_config(self):
        assert _add_spec_args({}) == []


# ===================================================================
# _add_bool_flags
# ===================================================================


@pytest.mark.unit
class TestAddBoolFlags:
    def test_mlock_true(self):
        assert "--mlock" in _add_bool_flags({"mlock": True})

    def test_mlock_false_no_flag(self):
        assert "--mlock" not in _add_bool_flags({"mlock": False})

    def test_no_mmap_true(self):
        assert "--no-mmap" in _add_bool_flags({"no_mmap": True})

    def test_flash_attn_not_in_bool_flags(self):
        args = _add_bool_flags({"flash_attn": True})
        assert "--flash-attn" not in args

    def test_repack_false_adds_no_repack(self):
        assert "--no-repack" in _add_bool_flags({"repack": False})

    def test_swa_full_true(self):
        assert "--swa-full" in _add_bool_flags({"swa_full": True})

    def test_kv_offload_false_adds_no_kv_offload(self):
        assert "--no-kv-offload" in _add_bool_flags({"kv_offload": False})

    def test_empty_config(self):
        assert _add_bool_flags({}) == []

    @pytest.mark.parametrize(
        "key,flag,trigger",
        [
            ("mlock", "--mlock", True),
            ("no_mmap", "--no-mmap", True),
            ("direct_io", "--direct-io", True),
            ("cont_batching", "--no-cont-batching", False),
            ("context_shift", "--no-context-shift", False),
        ],
    )
    def test_trigger_values(self, key, flag, trigger):
        assert flag in _add_bool_flags({key: trigger})
        opposite = not trigger
        assert flag not in _add_bool_flags({key: opposite})


# ===================================================================
# _add_extended_args
# ===================================================================


@pytest.mark.unit
class TestAddExtendedArgs:
    def test_numa(self):
        args = _add_extended_args({"numa": "distribute"})
        assert "--numa" in args and args[args.index("--numa") + 1] == "distribute"

    def test_tensor_split(self):
        args = _add_extended_args({"tensor_split": "0.5,0.5"})
        assert "--tensor-split" in args

    def test_cpu_moe(self):
        args = _add_extended_args({"cpu_moe": True})
        assert "--cpu-moe" in args

    def test_override_tensor(self):
        args = _add_extended_args({"override_tensor": "blk.0.attn=CPU"})
        assert "-ot" in args

    def test_threads_http(self):
        args = _add_extended_args({"threads_http": 4})
        assert "--threads-http" in args

    def test_empty_config(self):
        assert _add_extended_args({}) == []
