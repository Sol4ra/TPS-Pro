"""TUI test harness — runs the main menu loop with stubbed-out heavy operations.

Creates temporary fake server/model files so initialization succeeds, patches
out port checks, hardware detection, server management, and phase execution
so the menu rendering and dispatch logic can be exercised end-to-end.

Invoked by test_tui_integration.py via subprocess.
"""

from __future__ import annotations

import atexit
import logging
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Create temporary fake files so initialization passes path validation
# ---------------------------------------------------------------------------

_tmpdir = tempfile.mkdtemp(prefix="tui_test_")

_fake_server = Path(_tmpdir) / (
    "llama-server.exe" if sys.platform == "win32" else "llama-server"
)
_fake_server.write_bytes(b"\x00" * 64)
if sys.platform != "win32":
    _fake_server.chmod(0o755)

_fake_model = Path(_tmpdir) / "test-model.gguf"
# Write minimal GGUF header so detect_model_layers doesn't crash
_fake_model.write_bytes(b"GGUF" + b"\x03\x00\x00\x00" + b"\x00" * 16)

_fake_results = Path(_tmpdir) / "results"
_fake_results.mkdir()

# Write a minimal config file
import json as _json

_config_path = _fake_results / "optimizer-config.json"
_config_path.write_text(
    _json.dumps(
        {
            "server": str(_fake_server),
            "model": str(_fake_model),
            "chat_template": "",
            "architecture": {"type": "dense"},
            "hardware": {
                "max_threads": 8,
                "max_gpu_layers": 99,
                "default_gpu_layers": 99,
            },
            "port": 18555,
        }
    ),
    encoding="utf-8",
)


def _cleanup_tmpdir():
    import shutil

    try:
        shutil.rmtree(_tmpdir, ignore_errors=True)
    except Exception:
        pass


atexit.register(_cleanup_tmpdir)


# ---------------------------------------------------------------------------
# 1b. Keep a safe reference to stdout for writing
# ---------------------------------------------------------------------------

# Save a reference that won't be closed when LogTee replaces sys.stdout
_safe_stdout = sys.stdout


# ---------------------------------------------------------------------------
# 2. Patch sys.argv before any imports that parse CLI args
# ---------------------------------------------------------------------------

sys.argv = [
    "tps_pro",
    "--server",
    str(_fake_server),
    "--model",
    str(_fake_model),
    "--results-dir",
    str(_fake_results),
    "--port",
    "18555",
    "--dense",
    "--no-bench",
    "--config",
    str(_config_path),
]


# ---------------------------------------------------------------------------
# 3. Stub out heavy/dangerous operations BEFORE importing the app
# ---------------------------------------------------------------------------

# Suppress optuna warnings
import warnings

try:
    import optuna

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    pass


# Patch socket connect_ex to always say port is free
import socket as _socket

_orig_connect_ex = _socket.socket.connect_ex


def _patched_connect_ex(self, address):
    """Always report port as available (not in use)."""
    return 1  # 1 = connection refused = port is free


_socket.socket.connect_ex = _patched_connect_ex


# ---------------------------------------------------------------------------
# 4. Import and patch the app modules
# ---------------------------------------------------------------------------

# Ensure parent is on path
_pkg_parent = str(Path(__file__).resolve().parent.parent.parent)
if _pkg_parent not in sys.path:
    sys.path.insert(0, _pkg_parent)

# Import main first — it is the normal entry point and handles the
# circular import chain (state -> engine -> state) correctly.
# Patch hardware detection to no-op
import tps_pro.cli.setup as _setup_mod
import tps_pro.hardware as _hw_mod
import tps_pro.main as _main_mod
import tps_pro.state as _state

_orig_detect_hardware = _setup_mod.detect_hardware_and_model


def _stub_detect_hardware_and_model(kill_fn=None):
    """Skip GPU detection and competing process killing."""
    _state.ctx.model_size_class = "small"
    _state.ctx.model_size_gb = 1.0


_setup_mod.detect_hardware_and_model = _stub_detect_hardware_and_model


# Patch detect_gpus to return empty list
def _stub_detect_gpus():
    return []


_hw_mod.detect_gpus = _stub_detect_gpus

# Also patch in the menu module
import tps_pro.cli.menu as _menu_mod

_menu_mod.detect_gpus = _stub_detect_gpus


# Patch kill_server and related server management to no-ops
import tps_pro.engine as _engine_mod


def _stub_kill_server(ctx, wait=False):
    pass


_engine_mod.kill_server = _stub_kill_server
_main_mod.kill_server = _stub_kill_server


def _stub_is_server_running(ctx):
    return False


_engine_mod.is_server_running = _stub_is_server_running


# Patch LogTee to use regular stdout (avoid file creation issues)
class _FakeLogTee:
    def __init__(self, *args, **kwargs):
        self.log_path = str(Path(_tmpdir) / "test.log")
        self._out = _safe_stdout

    def write(self, data):
        try:
            self._out.write(data)
            self._out.flush()
        except (UnicodeEncodeError, ValueError, OSError):
            pass

    def flush(self):
        try:
            self._out.flush()
        except (ValueError, OSError):
            pass

    def close(self):
        pass

    def isatty(self):
        return False

    @property
    def encoding(self):
        return "utf-8"


_engine_mod.LogTee = _FakeLogTee
_main_mod.LogTee = _FakeLogTee


# Patch dashboard
import tps_pro.cli.dashboard as _dash_mod


def _stub_launch_dashboard(ctx):
    print("  [stub] Dashboard launch skipped (test mode)")


_dash_mod.launch_dashboard = _stub_launch_dashboard
_main_mod.launch_dashboard = _stub_launch_dashboard


# Patch all phase functions to be no-ops that print a stub message
import tps_pro.evals as _evals_mod
import tps_pro.phases as _phases_mod


def _make_phase_stub(name):
    def _stub(*args, **kwargs):
        print(f"  [stub] Phase {name} skipped (test mode)")

    return _stub


for phase_name in [
    "phase_gpu_offload",
    "phase_core_engine",
    "phase_speculation",
    "phase_workload_sim",
    "phase_experts",
    "phase_tensor_split",
    "phase_quality",
    "phase_context_sweep",
]:
    stub = _make_phase_stub(phase_name)
    if hasattr(_phases_mod, phase_name):
        setattr(_phases_mod, phase_name, stub)
    if hasattr(_menu_mod, phase_name):
        setattr(_menu_mod, phase_name, stub)

for eval_name in [
    "phase_niah",
    "phase_reasoning_eval",
    "phase_integrity_eval",
]:
    stub = _make_phase_stub(eval_name)
    if hasattr(_evals_mod, eval_name):
        setattr(_evals_mod, eval_name, stub)
    if hasattr(_menu_mod, eval_name):
        setattr(_menu_mod, eval_name, stub)


# Patch pipeline functions
import tps_pro.pipeline as _pipeline_mod


def _stub_run_full_pipeline():
    print("  [stub] Full pipeline skipped (test mode)")


def _stub_batch_optimize(*args, **kwargs):
    print("  [stub] Batch optimize skipped (test mode)")


_pipeline_mod.run_full_pipeline = _stub_run_full_pipeline
_pipeline_mod.batch_optimize = _stub_batch_optimize
_main_mod.run_full_pipeline = _stub_run_full_pipeline
_main_mod.batch_optimize = _stub_batch_optimize
_menu_mod.batch_optimize = _stub_batch_optimize


# Patch generate_html_report to no-op
import tps_pro.cli.report as _report_mod


def _stub_generate_html_report():
    print("  [stub] HTML report skipped (test mode)")


_report_mod.generate_html_report = _stub_generate_html_report


# Patch needs_setup to return False (we have valid fake files)
import tps_pro.cli.wizard as _wizard_mod

_wizard_mod.needs_setup = lambda: False


# Patch the signal handler installation to use a simpler one
def _stub_install_interrupt_handler():
    pass


_main_mod.install_interrupt_handler = _stub_install_interrupt_handler


# Patch ensure_results_dir
import tps_pro.search as _search_mod


def _stub_ensure_results_dir(ctx):
    ctx.results_dir.mkdir(parents=True, exist_ok=True)


_search_mod.ensure_results_dir = _stub_ensure_results_dir


# Patch kill_competing_processes
_hw_mod.kill_competing_processes = lambda *a, **kw: None


# ---------------------------------------------------------------------------
# 5. Run main()
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        stream=sys.stderr,
    )

    try:
        _main_mod.main()
    except (EOFError, KeyboardInterrupt):
        # Expected when piped input runs out
        pass
    except SystemExit as e:
        # Allow clean exits
        if e.code not in (0, None):
            sys.exit(e.code)
    finally:
        _cleanup_tmpdir()
