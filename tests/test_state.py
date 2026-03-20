"""Tests for tps_pro.state module.

Covers: AppContext dataclass, _detect_numa_nodes(),
create_context(), _load_config(), initialize(), _DEFAULTS.
"""

import copy
import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from tps_pro.state import (
    _DEFAULTS,
    AppContext,
    _detect_numa_nodes,
    _load_config,
    create_context,
    initialize,
)

# ============================================================
# _DEFAULTS and _PRESETS module-level dicts
# ============================================================


class TestDefaults:
    """Validate _DEFAULTS has all required keys with reasonable types."""

    @pytest.mark.unit
    def test_defaults_top_level_keys(self):
        expected_keys = {
            "server",
            "model",
            "chat_template",
            "results_dir",
            "port",
            "architecture",
            "hardware",
        }
        assert expected_keys.issubset(_DEFAULTS.keys())

    @pytest.mark.unit
    def test_defaults_architecture_keys(self):
        arch = _DEFAULTS["architecture"]
        assert "type" in arch
        assert "expert_override_key" in arch
        assert "default_experts" in arch
        assert "max_experts" in arch

    @pytest.mark.unit
    def test_defaults_hardware_keys(self):
        hw = _DEFAULTS["hardware"]
        expected = {
            "max_threads",
            "moe_sweep_max",
            "moe_sweep_center",
            "max_gpu_layers",
            "default_gpu_layers",
        }
        assert expected.issubset(hw.keys())

    @pytest.mark.unit
    def test_defaults_port_is_int(self):
        assert isinstance(_DEFAULTS["port"], int)

    @pytest.mark.unit
    def test_defaults_architecture_type_is_string(self):
        assert isinstance(_DEFAULTS["architecture"]["type"], str)

    @pytest.mark.unit
    def test_defaults_hardware_values_nullable_or_int(self):
        hw = _DEFAULTS["hardware"]
        for key in (
            "max_threads",
            "moe_sweep_max",
            "moe_sweep_center",
            "max_gpu_layers",
        ):
            assert hw[key] is None or isinstance(hw[key], int)
        assert isinstance(hw["default_gpu_layers"], int)


class TestGetPhaseTrialDefault:
    """Validate get_phase_trial_default uses PipelineConfig as source of truth."""

    @pytest.mark.unit
    def test_known_phase_normal_preset(self):
        from tps_pro.cli.services_config import get_phase_trial_default

        # core_engine has trials=100 in DEFAULT_PHASES; normal multiplier is 1.0
        assert get_phase_trial_default("core_engine", "normal") == 100

    @pytest.mark.unit
    def test_quick_preset_halves_trials(self):
        from tps_pro.cli.services_config import get_phase_trial_default

        # core_engine base=100, quick=0.5 -> 50
        assert get_phase_trial_default("core_engine", "quick") == 50

    @pytest.mark.unit
    def test_thorough_preset_increases_trials(self):
        from tps_pro.cli.services_config import get_phase_trial_default

        # core_engine base=100, thorough=1.5 -> 150
        assert get_phase_trial_default("core_engine", "thorough") == 150

    @pytest.mark.unit
    def test_unknown_phase_returns_default_60(self):
        from tps_pro.cli.services_config import get_phase_trial_default

        assert get_phase_trial_default("nonexistent_phase", "normal") == 60

    @pytest.mark.unit
    def test_unknown_preset_uses_multiplier_1(self):
        from tps_pro.cli.services_config import get_phase_trial_default

        # Unknown preset defaults to multiplier 1.0 -> base trials
        assert get_phase_trial_default("core_engine", "ultra") == 100

    @pytest.mark.unit
    def test_quick_less_than_normal_less_than_thorough(self):
        from tps_pro.cli.services_config import get_phase_trial_default

        for phase in ("core_engine", "speculation", "quality"):
            q = get_phase_trial_default(phase, "quick")
            n = get_phase_trial_default(phase, "normal")
            t = get_phase_trial_default(phase, "thorough")
            assert q <= n <= t, f"{phase}: {q} <= {n} <= {t}"


# ============================================================
# AppContext dataclass
# ============================================================


class TestAppContext:
    """Test AppContext construction, defaults, and cleanup."""

    @pytest.mark.unit
    def test_default_construction(self):
        ctx = AppContext()
        assert ctx.port == 8090
        assert ctx.is_moe is False
        assert ctx.max_threads == 8
        assert ctx.fail_fast is False
        assert ctx.skip_quality is False
        assert ctx.dry_run is False
        assert ctx.debug is False
        assert ctx.fresh_run is False
        assert ctx.model_size_class == "medium"
        assert ctx.model_size_gb == 0.0
        assert ctx.numa_nodes == 1
        assert ctx.active_server_proc is None
        assert ctx.quality_baseline is None
        assert ctx.vram_total_mb is None
        assert ctx.no_jinja is False
        assert ctx.bench_path is None

    @pytest.mark.unit
    def test_default_path_fields_are_path_objects(self):
        ctx = AppContext()
        assert isinstance(ctx.server_path, Path)
        assert isinstance(ctx.model_path, Path)
        assert isinstance(ctx.chat_template_path, Path)
        assert isinstance(ctx.results_dir, Path)

    @pytest.mark.unit
    def test_default_mutable_fields_are_independent(self):
        ctx1 = AppContext()
        ctx2 = AppContext()
        ctx1.config["test_key"] = "value"
        assert "test_key" not in ctx2.config

    @pytest.mark.unit
    def test_custom_construction(self):
        ctx = AppContext(
            server_path=Path("/usr/bin/llama-server"),
            model_path=Path("/models/test.gguf"),
            port=9000,
            is_moe=True,
            max_threads=32,
            moe_sweep_max=40,
            moe_sweep_center=20,
            max_gpu_layers=64,
            default_gpu_layers=64,
            fail_fast=True,
            skip_quality=True,
            debug=True,
            numa_nodes=4,
            model_size_class="large",
            model_size_gb=42.5,
        )
        assert ctx.port == 9000
        assert ctx.is_moe is True
        assert ctx.max_threads == 32
        assert ctx.moe_sweep_max == 40
        assert ctx.max_gpu_layers == 64
        assert ctx.fail_fast is True
        assert ctx.skip_quality is True
        assert ctx.debug is True
        assert ctx.numa_nodes == 4
        assert ctx.model_size_class == "large"
        assert ctx.model_size_gb == 42.5

    @pytest.mark.unit
    def test_http_session_is_requests_session(self):
        ctx = AppContext()
        assert isinstance(ctx.http, requests.Session)

    @pytest.mark.unit
    def test_skip_flags_default_empty_set(self):
        ctx = AppContext()
        assert isinstance(ctx.skip_flags, set)
        assert len(ctx.skip_flags) == 0

    @pytest.mark.unit
    def test_close_cleans_http_session(self):
        ctx = AppContext()
        mock_session = MagicMock()
        ctx.http = mock_session
        ctx.close()
        mock_session.close.assert_called_once()

    @pytest.mark.unit
    def test_close_handles_exception_gracefully(self):
        ctx = AppContext()
        mock_session = MagicMock()
        mock_session.close.side_effect = RuntimeError("connection error")
        ctx.http = mock_session
        # Should not raise
        ctx.close()


# ============================================================
# get_phase_trial_default() — moved to services_config, tested above
# ============================================================


# ============================================================
# _detect_numa_nodes()
# ============================================================


class TestDetectNumaNodes:
    """Test NUMA node detection across platforms."""

    @pytest.mark.unit
    @patch("tps_pro.state.sys")
    @patch("tps_pro.state.subprocess.run")
    def test_windows_wmic_returns_count(self, mock_run, mock_sys):
        mock_sys.platform = "win32"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NodeId  \n0       \n1       \n",
        )
        assert _detect_numa_nodes() == 2

    @pytest.mark.unit
    @patch("tps_pro.state.sys")
    @patch("tps_pro.state.subprocess.run")
    def test_windows_wmic_single_node(self, mock_run, mock_sys):
        mock_sys.platform = "win32"
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="NodeId  \n0       \n",
        )
        assert _detect_numa_nodes() == 1

    @pytest.mark.unit
    @patch("tps_pro.state.sys")
    @patch("tps_pro.state.subprocess.run")
    def test_windows_wmic_failure_returns_1(self, mock_run, mock_sys):
        mock_sys.platform = "win32"
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert _detect_numa_nodes() == 1

    @pytest.mark.unit
    @patch("tps_pro.state.sys")
    @patch("tps_pro.state.Path")
    def test_linux_sys_path_detects_nodes(self, mock_path_cls, mock_sys):
        mock_sys.platform = "linux"
        numa_path_instance = MagicMock()
        numa_path_instance.exists.return_value = True
        numa_path_instance.glob.return_value = [
            Path("node0"),
            Path("node1"),
            Path("node2"),
        ]

        mock_path_cls.side_effect = lambda p: (
            numa_path_instance if p == "/sys/devices/system/node" else Path(p)
        )

        assert _detect_numa_nodes() == 3

    @pytest.mark.unit
    @patch("tps_pro.state.sys")
    @patch("tps_pro.state.Path")
    @patch("tps_pro.state.subprocess.run")
    def test_linux_numactl_fallback(self, mock_run, mock_path_cls, mock_sys):
        mock_sys.platform = "linux"
        numa_path_instance = MagicMock()
        numa_path_instance.exists.return_value = False
        mock_path_cls.side_effect = lambda p: (
            numa_path_instance if p == "/sys/devices/system/node" else Path(p)
        )

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="available: 4 nodes (0-3)\nnode 0 cpus: 0 1 2 3\n",
        )
        assert _detect_numa_nodes() == 4

    @pytest.mark.unit
    @patch("tps_pro.state.sys")
    @patch("tps_pro.state.subprocess.run")
    def test_detection_oserror_returns_1(self, mock_run, mock_sys):
        mock_sys.platform = "win32"
        mock_run.side_effect = OSError("wmic not found")
        assert _detect_numa_nodes() == 1

    @pytest.mark.unit
    @patch("tps_pro.state.sys")
    @patch("tps_pro.state.subprocess.run")
    def test_detection_timeout_returns_1(self, mock_run, mock_sys):
        mock_sys.platform = "win32"
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="wmic", timeout=5)
        assert _detect_numa_nodes() == 1


# ============================================================
# create_context()
# ============================================================


class TestCreateContext:
    """Test AppContext factory from config dicts."""

    def _make_config(self, **overrides):
        """Build a valid config dict with sensible defaults, applying overrides."""
        config = copy.deepcopy(_DEFAULTS)
        config["server"] = "/path/to/llama-server"
        config["model"] = "/path/to/model.gguf"
        config["chat_template"] = ""
        config["port"] = 8090
        config["preset"] = "normal"
        config["architecture"]["type"] = "dense"
        config["hardware"]["max_threads"] = 16
        config["hardware"]["moe_sweep_max"] = 32
        config["hardware"]["moe_sweep_center"] = 16
        config["hardware"]["max_gpu_layers"] = 40
        config["hardware"]["default_gpu_layers"] = 99
        config["hardware"]["numa_nodes"] = 1
        for k, v in overrides.items():
            if isinstance(v, dict) and isinstance(config.get(k), dict):
                config[k].update(v)
            else:
                config[k] = v
        return config

    @pytest.mark.unit
    def test_basic_context_creation(self):
        config = self._make_config()
        ctx = create_context(config)
        assert ctx.server_path == Path("/path/to/llama-server")
        assert ctx.model_path == Path("/path/to/model.gguf")
        assert ctx.port == 8090
        assert ctx.max_threads == 16
        assert ctx.is_moe is False

    @pytest.mark.unit
    def test_moe_architecture_flag(self):
        config = self._make_config(
            architecture={
                "type": "moe",
                "expert_override_key": "llama.expert_used_count",
                "default_experts": 4,
                "max_experts": 8,
            }
        )
        ctx = create_context(config)
        assert ctx.is_moe is True
        assert ctx.expert_override_key == "llama.expert_used_count"
        assert ctx.default_experts == 4
        assert ctx.max_experts == 8

    @pytest.mark.unit
    def test_port_propagation(self):
        config = self._make_config(port=9999)
        ctx = create_context(config)
        assert ctx.port == 9999
        assert ctx._port_alt == 10000
        assert ctx.server_url == "http://127.0.0.1:9999"

    @pytest.mark.unit
    def test_results_dir_and_derived_paths(self, tmp_path):
        config = self._make_config(results_dir=str(tmp_path / "results"))
        ctx = create_context(config)
        assert ctx.results_dir == tmp_path / "results"
        assert ctx.lookup_cache_file == str(tmp_path / "results" / "lookup-cache.bin")
        assert "optuna.db" in ctx.optuna_db

    @pytest.mark.unit
    def test_default_gpu_layers_capped_by_max(self):
        config = self._make_config(
            hardware={
                "max_gpu_layers": 20,
                "default_gpu_layers": 99,
                "max_threads": 16,
                "moe_sweep_max": 32,
                "moe_sweep_center": 16,
                "numa_nodes": 1,
            }
        )
        ctx = create_context(config)
        assert ctx.default_gpu_layers == 20  # min(99, 20) = 20

    @pytest.mark.unit
    def test_skip_quality_flag(self):
        config = self._make_config(skip_quality=True)
        ctx = create_context(config)
        assert ctx.skip_quality is True

    @pytest.mark.unit
    def test_fail_fast_flag(self):
        config = self._make_config(fail_fast=True)
        ctx = create_context(config)
        assert ctx.fail_fast is True

    @pytest.mark.unit
    def test_no_jinja_flag(self):
        config = self._make_config(no_jinja=True)
        ctx = create_context(config)
        assert ctx.no_jinja is True

    @pytest.mark.unit
    def test_debug_flag(self):
        config = self._make_config(debug=True)
        ctx = create_context(config)
        assert ctx.debug is True

    @pytest.mark.unit
    def test_chat_template_empty_produces_empty_path(self):
        config = self._make_config(chat_template="")
        ctx = create_context(config)
        assert ctx.chat_template_path == Path("")

    @pytest.mark.unit
    def test_chat_template_nonempty_produces_path(self):
        config = self._make_config(chat_template="/templates/chatml.jinja")
        ctx = create_context(config)
        assert ctx.chat_template_path == Path("/templates/chatml.jinja").resolve()

    @pytest.mark.unit
    def test_naked_engine_has_expected_keys(self):
        config = self._make_config()
        ctx = create_context(config)
        assert "context" in ctx.naked_engine
        assert "mlock" in ctx.naked_engine
        assert "n_gpu_layers" in ctx.naked_engine

    @pytest.mark.unit
    def test_numa_nodes_from_hardware(self):
        config = self._make_config(
            hardware={
                "numa_nodes": 4,
                "max_threads": 16,
                "moe_sweep_max": 32,
                "moe_sweep_center": 16,
                "max_gpu_layers": 40,
                "default_gpu_layers": 99,
            }
        )
        ctx = create_context(config)
        assert ctx.numa_nodes == 4

    @pytest.mark.unit
    @patch(
        "tps_pro.state.find_llama_bench",
        return_value=Path("/usr/bin/llama-bench"),
    )
    def test_bench_path_populated_when_available(self, _mock_bench):
        config = self._make_config()
        ctx = create_context(config)
        assert ctx.bench_path == Path("/usr/bin/llama-bench")

    @pytest.mark.unit
    def test_bench_path_none_when_no_bench_flag(self):
        config = self._make_config(no_bench=True)
        ctx = create_context(config)
        assert ctx.bench_path is None


# ============================================================
# _load_config()
# ============================================================


class TestLoadConfig:
    """Test CLI arg parsing, config file loading, and hardware auto-detection."""

    @pytest.mark.unit
    @patch(
        "tps_pro.cli.setup_binary.ensure_llama_server", side_effect=RuntimeError("test")
    )
    @patch("tps_pro.state._detect_numa_nodes", return_value=1)
    @patch("tps_pro.state.detect_model_layers", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=8)
    @patch("sys.argv", ["optimizer"])
    def test_default_args_produce_valid_config(self, *_):
        config = _load_config()
        assert config["server"] == ""
        assert config["model"] == ""
        assert config["preset"] == "normal"
        assert config["hardware"]["max_threads"] == 8
        assert config["hardware"]["max_gpu_layers"] == 99  # fallback

    @pytest.mark.unit
    @patch("tps_pro.state._detect_numa_nodes", return_value=1)
    @patch("tps_pro.state.detect_model_layers", return_value=32)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=16)
    @patch(
        "sys.argv",
        ["optimizer", "--server", "/bin/llama-server", "--model", "/models/test.gguf"],
    )
    def test_server_and_model_paths_propagate(self, *_):
        config = _load_config()
        assert config["server"] == "/bin/llama-server"
        assert config["model"] == "/models/test.gguf"

    @pytest.mark.unit
    @patch("tps_pro.state._detect_numa_nodes", return_value=1)
    @patch("tps_pro.state.detect_model_layers", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=8)
    @patch("sys.argv", ["optimizer", "--quick"])
    def test_quick_preset_flag(self, *_):
        config = _load_config()
        assert config["preset"] == "quick"

    @pytest.mark.unit
    @patch("tps_pro.state._detect_numa_nodes", return_value=1)
    @patch("tps_pro.state.detect_model_layers", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=8)
    @patch("sys.argv", ["optimizer", "--thorough"])
    def test_thorough_preset_flag(self, *_):
        config = _load_config()
        assert config["preset"] == "thorough"

    @pytest.mark.unit
    @patch("tps_pro.state._detect_numa_nodes", return_value=1)
    @patch("tps_pro.state.detect_model_layers", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=8)
    @patch("sys.argv", ["optimizer", "--skip-quality"])
    def test_skip_quality_flag(self, *_):
        config = _load_config()
        assert config["skip_quality"] is True

    @pytest.mark.unit
    @patch("tps_pro.state._detect_numa_nodes", return_value=1)
    @patch("tps_pro.state.detect_model_layers", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=8)
    @patch("sys.argv", ["optimizer", "--port", "9999"])
    def test_custom_port(self, *_):
        config = _load_config()
        assert config["port"] == 9999

    @pytest.mark.unit
    @patch("tps_pro.state._detect_numa_nodes", return_value=1)
    @patch("tps_pro.state.detect_model_layers", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=8)
    @patch("sys.argv", ["optimizer", "--dense"])
    def test_dense_flag_overrides_architecture(self, *_):
        config = _load_config()
        assert config["architecture"]["type"] == "dense"

    @pytest.mark.unit
    def test_config_file_loading(self, tmp_path):
        config_data = {
            "server": "/custom/server",
            "model": "/custom/model.gguf",
            "port": 7777,
        }
        config_file = tmp_path / "test_config.json"
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        with (
            patch("sys.argv", ["optimizer", "--config", str(config_file)]),
            patch("tps_pro.state._detect_numa_nodes", return_value=1),
            patch(
                "tps_pro.state.detect_model_layers",
                return_value=None,
            ),
            patch("tps_pro.state.os.cpu_count", return_value=8),
            patch(
                "pathlib.Path.exists",
                autospec=True,
                side_effect=lambda self: str(self) == str(config_file),
            ),
        ):
            config = _load_config()
            assert config["server"] == "/custom/server"
            assert config["model"] == "/custom/model.gguf"
            assert config["port"] == 7777

    @pytest.mark.unit
    @patch(
        "tps_pro.cli.setup_binary.ensure_llama_server", side_effect=RuntimeError("test")
    )
    @patch("tps_pro.state._detect_numa_nodes", return_value=1)
    @patch("tps_pro.state.detect_model_layers", return_value=None)
    @patch("tps_pro.state.os.cpu_count", return_value=8)
    @patch("sys.argv", ["optimizer", "--config", "/nonexistent/config.json"])
    @patch("pathlib.Path.exists", return_value=False)
    def test_config_file_not_found_uses_defaults(self, *_):
        config = _load_config()
        assert config["server"] == ""
        assert config["preset"] == "normal"

    @pytest.mark.unit
    def test_corrupt_config_file_uses_defaults(self, tmp_path):
        config_file = tmp_path / "bad_config.json"
        config_file.write_text("{invalid json!!!", encoding="utf-8")

        with (
            patch("sys.argv", ["optimizer", "--config", str(config_file)]),
            patch("tps_pro.state._detect_numa_nodes", return_value=1),
            patch(
                "tps_pro.state.detect_model_layers",
                return_value=None,
            ),
            patch("tps_pro.state.os.cpu_count", return_value=8),
            patch(
                "pathlib.Path.exists",
                autospec=True,
                side_effect=lambda self: str(self) == str(config_file),
            ),
            patch(
                "tps_pro.cli.setup_binary.ensure_llama_server",
                side_effect=RuntimeError("test"),
            ),
        ):
            config = _load_config()
            # Corrupt file handled gracefully; defaults preserved
            assert config["preset"] == "normal"
            assert config["server"] == ""

    @pytest.mark.unit
    @patch("tps_pro.state._detect_numa_nodes", return_value=2)
    @patch("tps_pro.state.detect_model_layers", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=32)
    @patch("sys.argv", ["optimizer"])
    def test_hardware_autodetection(self, *_):
        config = _load_config()
        hw = config["hardware"]
        assert hw["max_threads"] == 32
        assert hw["numa_nodes"] == 2
        assert hw["moe_sweep_max"] == 40  # min(32*2, 40) = 40
        assert hw["moe_sweep_center"] == 20  # 40 // 2

    @pytest.mark.unit
    @patch("tps_pro.state._detect_numa_nodes", return_value=1)
    @patch("tps_pro.state.detect_model_layers", return_value=48)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=8)
    @patch("sys.argv", ["optimizer", "--model", "/models/test.gguf"])
    def test_gpu_layers_from_model_detection(self, *_):
        config = _load_config()
        assert config["hardware"]["max_gpu_layers"] == 48

    @pytest.mark.unit
    @patch("tps_pro.state._detect_numa_nodes", return_value=1)
    @patch("tps_pro.state.detect_model_layers", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=4)
    @patch("sys.argv", ["optimizer"])
    def test_moe_sweep_capped_at_40(self, *_):
        """When cpu_count * 2 exceeds 40, moe_sweep_max should cap at 40."""
        # cpu_count=4, so max_threads=4, moe_sweep_max=min(4*2, 40)=8
        config = _load_config()
        assert config["hardware"]["moe_sweep_max"] == 8  # 4*2=8 < 40

    @pytest.mark.unit
    @patch("tps_pro.state._detect_numa_nodes", return_value=1)
    @patch("tps_pro.state.detect_model_layers", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=None)
    @patch("sys.argv", ["optimizer"])
    def test_cpu_count_none_fallback(self, *_):
        """When os.cpu_count() returns None, max_threads falls back to 16."""
        config = _load_config()
        assert config["hardware"]["max_threads"] == 16

    @pytest.mark.unit
    def test_config_file_deep_merges_nested_dicts(self, tmp_path):
        config_data = {
            "architecture": {"type": "moe", "default_experts": 4},
        }
        config_file = tmp_path / "merge_config.json"
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        with (
            patch("sys.argv", ["optimizer", "--config", str(config_file)]),
            patch("tps_pro.state._detect_numa_nodes", return_value=1),
            patch(
                "tps_pro.state.detect_model_layers",
                return_value=None,
            ),
            patch("tps_pro.state.os.cpu_count", return_value=8),
            patch(
                "pathlib.Path.exists",
                autospec=True,
                side_effect=lambda self: str(self) == str(config_file),
            ),
        ):
            config = _load_config()
            # Deep merge: architecture.type updated, but expert_override_key
            # preserved from defaults
            assert config["architecture"]["type"] == "moe"
            assert config["architecture"]["default_experts"] == 4
            assert "expert_override_key" in config["architecture"]


# ============================================================
# initialize()
# ============================================================


class TestInitialize:
    """Test the module-level lazy initializer."""

    @pytest.mark.unit
    @patch("tps_pro.state._detect_numa_nodes", return_value=1)
    @patch("tps_pro.state.detect_model_layers", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=8)
    @patch("sys.argv", ["optimizer", "--port", "7070"])
    def test_initialize_returns_app_context(self, *_):
        import tps_pro.state as state_mod

        # Reset initialization state
        state_mod._initialized = False
        try:
            result = initialize()
            assert isinstance(result, AppContext)
            assert result.port == 7070
        finally:
            state_mod._initialized = False

    @pytest.mark.unit
    @patch("tps_pro.state._detect_numa_nodes", return_value=1)
    @patch("tps_pro.state.detect_model_layers", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=8)
    @patch("sys.argv", ["optimizer"])
    def test_idempotency_second_call_returns_same(self, *_):
        import tps_pro.state as state_mod

        state_mod._initialized = False
        try:
            first = initialize()
            second = initialize()
            assert first is second
        finally:
            state_mod._initialized = False

    @pytest.mark.unit
    @patch("tps_pro.state._detect_numa_nodes", return_value=4)
    @patch("tps_pro.state.detect_model_layers", return_value=64)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=32)
    @patch("sys.argv", ["optimizer"])
    def test_hardware_detection_populates_context(self, *_):
        import tps_pro.state as state_mod

        state_mod._initialized = False
        try:
            result = initialize()
            assert result.max_threads == 32
            assert result.numa_nodes == 4
            assert result.max_gpu_layers == 64
        finally:
            state_mod._initialized = False

    @pytest.mark.unit
    @patch("tps_pro.state._detect_numa_nodes", return_value=1)
    @patch("tps_pro.state.detect_model_layers", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=8)
    @patch("sys.argv", ["optimizer"])
    def test_initialize_updates_module_level_ctx(self, *_):
        import tps_pro.state as state_mod

        state_mod._initialized = False
        try:
            result = initialize()
            # The module-level ctx should be the same object, updated in place
            assert state_mod.ctx is result
            assert state_mod.ctx.max_threads == 8
        finally:
            state_mod._initialized = False

    @pytest.mark.unit
    @patch("tps_pro.state._detect_numa_nodes", return_value=1)
    @patch("tps_pro.state.detect_model_layers", return_value=None)
    @patch("pathlib.Path.exists", return_value=False)
    @patch("tps_pro.state.os.cpu_count", return_value=8)
    @patch("sys.argv", ["optimizer"])
    def test_initialize_updates_module_level_config(self, *_):
        import tps_pro.state as state_mod

        state_mod._initialized = False
        try:
            initialize()
            assert "preset" in state_mod.config
            assert state_mod.config["preset"] == "normal"
        finally:
            state_mod._initialized = False
