"""
engine subpackage -- re-exports public API symbols.

Error strategy:
    Server lifecycle functions (start/kill/wait) propagate OSError for
    unrecoverable failures (e.g. missing executable).  wait_for_server returns
    status strings ("ok", "oom", "timeout", "died", "jinja_error") -- never
    raises.  Cleanup code (kill_process_tree, kill_server) swallows errors to guarantee
    process reaping -- annotated with '# Cleanup: safe to ignore'.  Bench trials
    return None on non-OOM failure and raise BenchOOMError on OOM so callers can
    prune the trial.  Platform helpers (Job Object) are best-effort and log at
    WARNING level on failure.

Imports are lazy to avoid circular dependency with state.py.
"""

from __future__ import annotations


def __getattr__(name: str):
    """Lazy import — only load submodules when their symbols are accessed."""
    # Bench trial execution and OOM sentinel
    if name in ("BenchOOMError", "run_bench_trial"):
        from .bench import BenchOOMError, run_bench_trial

        _exports = {
            "BenchOOMError": BenchOOMError,
            "run_bench_trial": run_bench_trial,
        }
        return _exports[name]

    # Server log/output parsing helpers
    if name == "reset_load_time_debug":
        from .parsing import reset_load_time_debug

        return reset_load_time_debug

    # Server lifecycle: start, wait, kill, warmup, jinja recovery
    if name in (
        "boot_server_with_jinja_recovery",
        "is_server_running",
        "kill_server",
        "server_start_failed",
        "start_server",
        "wait_for_server",
        "warmup_server",
    ):
        from . import server

        return getattr(server, name)

    # Typed wrapper around subprocess.Popen for server processes
    if name == "ServerProcess":
        from ..result_types import ServerProcess

        return ServerProcess

    # Deprecated: import BaselineFailure directly from tps_pro.errors instead.
    # This re-export is kept only for backward compatibility with existing
    # callers (pipeline.py, phases/core_engine.py, phases/moe_*.py,
    # phases/trial_helpers.py).
    if name == "BaselineFailure":
        from ..errors import BaselineFailure

        return BaselineFailure

    # Shared utilities: logging tee, phase timer, dry-run guard, tensor math, JSON I/O
    if name in (
        "LogTee",
        "PhaseTimer",
        "check_dry_run",
        "generate_tensor_splits",
        "read_json_safe",
    ):
        from . import util

        return getattr(util, name)

    raise AttributeError(f"module 'tps_pro.engine' has no attribute {name!r}")


__all__ = [
    "ServerProcess",
    "LogTee",
    "PhaseTimer",
    "BaselineFailure",
    "BenchOOMError",
    "check_dry_run",
    "generate_tensor_splits",
    "run_bench_trial",
    "reset_load_time_debug",
    "start_server",
    "wait_for_server",
    "warmup_server",
    "kill_server",
    "is_server_running",
    "boot_server_with_jinja_recovery",
    "server_start_failed",
    "read_json_safe",
]
