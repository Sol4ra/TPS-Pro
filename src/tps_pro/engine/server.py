"""
Server lifecycle: ServerProcess, start/kill/wait/warmup, boot with Jinja recovery.

Command-building helpers are in engine/commands.py.
OOM detection and load-time parsing are in engine/parsing.py.
Windows Job Object management is in engine/platform.py.

Error strategy (see errors.py for full documentation):
    - start_server: propagates OSError if executable cannot start.
    - wait_for_server: returns status strings ("ok", "oom", "timeout", "died",
      "jinja_error") -- never raises.  Callers branch on the string.
    - warmup_server: returns bool -- False triggers retry at a higher level.
    - kill_process_tree / kill_server: cleanup code -- swallows errors to guarantee
      the process is reaped (annotated with '# Cleanup: safe to ignore').
"""

from __future__ import annotations

import dataclasses
import logging
import os
import random
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Literal

import requests  # type: ignore[import-untyped]

from ..constants import (
    BIND_HOST,
    DEFAULT_TEMPERATURE,
    HTTP_OK,
    HTTP_SERVER_ERROR,
    MOE_TIMEOUT_MULTIPLIER,
    SERVER_BOOT_TIMEOUTS,
    SERVER_HEALTH_POLL_SLEEP,
    SERVER_HEALTH_POLL_TIMEOUT,
    SERVER_KILL_WAIT_TIMEOUT,
    SERVER_PORT_RELEASE_INITIAL_DELAY,
    SERVER_PORT_RELEASE_MAX_DELAY,
    SERVER_PORT_RELEASE_RETRIES,
    TPS_TEST_PROMPT,
    WARMUP_REQUEST_TIMEOUT,
    WARMUP_TOKENS_STAGE1,
    WARMUP_TOKENS_STAGE2,
)
from ..hardware import init_vram_info
from ..result_types import EngineConfig, ServerProcess
from ..state import AppContext

# Import only what server.py itself needs (no re-exports)
from .commands import (
    _add_base_args,
    _add_bool_flags,
    _add_extended_args,
    _add_kv_cache_args,
    _add_numeric_flag_pairs,
    _add_spec_args,
)
from .parsing import (
    _is_error_line,
    _parse_load_time,
    is_oom,
)
from .platform import (
    _assign_job_object,
)

logger = logging.getLogger(__name__)

__all__ = [
    "start_server",
    "wait_for_server",
    "warmup_server",
    "kill_server",
    "is_server_running",
    "boot_server_with_jinja_recovery",
    "server_start_failed",
]


# ============================================================
# Server Lifecycle
# ============================================================


def start_server(ctx: AppContext, engine_config: EngineConfig) -> ServerProcess:
    """Start llama-server with given engine config.

    Raises:
        OSError: If the server executable cannot be started (e.g. not found,
            permission denied).
    """
    env = os.environ.copy()
    if "tensor_split" not in engine_config and "CUDA_VISIBLE_DEVICES" not in os.environ:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    if engine_config.get("cuda_graph_opt"):
        env["GGML_CUDA_GRAPH_OPT"] = "1"

    cmd = _add_base_args(ctx, engine_config)
    cmd.extend(_add_numeric_flag_pairs(ctx, engine_config))
    cmd.extend(_add_kv_cache_args(engine_config, ctx=ctx))
    cmd.extend(_add_spec_args(engine_config))
    cmd.extend(_add_bool_flags(engine_config))
    cmd.extend(_add_extended_args(engine_config))

    if ctx.debug:
        flags = cmd[3:]
        logger.debug("cmd: %s", " ".join(str(f) for f in flags))

    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        if sys.platform == "win32"
        else 0,
    )
    # On Windows, assign to a Job Object so child dies when parent dies
    _assign_job_object(proc)
    server_proc = ServerProcess(proc=proc)
    ctx.active_server_proc = server_proc

    def _drain_stderr():
        try:
            for line in server_proc.proc.stderr or []:
                with server_proc.lock:
                    server_proc.stderr_lines.append(
                        line.decode("utf-8", errors="replace").rstrip()
                    )
        except (OSError, ValueError) as e:
            logger.debug("stderr drain ended: %s", e)

    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()

    return server_proc


def wait_for_server(  # noqa: C901, PLR0912
    ctx: AppContext, proc: ServerProcess | None = None, timeout: float | None = None
) -> Literal["ok", "oom", "died", "timeout", "jinja_error"]:
    """Wait for llama-server /health to return ok.

    Args:
        proc: A ServerProcess instance returned by start_server(), or None.
        timeout: Optional timeout in seconds.
    """
    if timeout is None:
        size_cls = getattr(ctx, "model_size_class", "medium")
        timeout = SERVER_BOOT_TIMEOUTS.get(size_cls, 300)
        if ctx.is_moe:
            timeout = int(timeout * MOE_TIMEOUT_MULTIPLIER)

    start = time.time()
    last_status = start
    while True:
        if proc is not None and proc.proc.poll() is not None:
            with proc.lock:
                stderr_text = "\n".join(proc.stderr_lines)
            if any(
                kw in stderr_text.lower()
                for kw in ["jinja", "template error", "template parse"]
            ):
                logger.warning("Server died with Jinja/template error.")
                return "jinja_error"
            if is_oom(stderr_text):
                logger.warning("Server died with OOM error.")
                return "oom"
            return "died"
        now = time.time()
        if now - start > timeout:
            return "timeout"
        if now - last_status >= 5:  # noqa: PLR2004
            elapsed = int(now - start)
            if ctx.debug:
                logger.debug("Waiting for server... (%ds)", elapsed)
            last_status = now
        try:
            r = ctx.http.get(
                f"{ctx.server_url}/health", timeout=SERVER_HEALTH_POLL_TIMEOUT
            )
            if r.status_code == HTTP_OK and r.json().get("status") == "ok":
                if proc is not None:
                    with proc.lock:
                        stderr_text = "\n".join(proc.stderr_lines)
                    # Parse model load time from stderr
                    parsed_load_ms = _parse_load_time(proc)
                    if parsed_load_ms is not None:
                        proc = dataclasses.replace(proc, load_time_ms=parsed_load_ms)
                    # Wall-clock fallback if stderr parsing found nothing
                    elif proc.load_time_ms is None:
                        proc = dataclasses.replace(
                            proc, load_time_ms=(time.time() - start) * 1000
                        )
                    ctx.active_server_proc = proc
                if warmup_server(ctx):
                    return "ok"
        except (requests.ConnectionError, requests.Timeout, ValueError) as e:
            logger.debug("Health poll failed (retrying): %s", e)
        time.sleep(SERVER_HEALTH_POLL_SLEEP)


def warmup_server(
    ctx: AppContext,
    vram_init_fn: Callable[[AppContext], None] | None = None,
) -> bool:
    """Two-stage warmup: CUDA pipeline warm + speculation cache prime.

    Args:
        vram_init_fn: Optional callback ``(ctx) -> None`` that populates
            ``ctx.vram_total_mb``.  When *None* (the default), the function
            falls back to a lazy import of ``hardware.init_vram_info``.
            Accepting the callback explicitly makes the hardware dependency
            injectable and testable without import-cycle gymnastics.
    """
    warmup_start = time.time()
    try:
        r = ctx.http.post(
            f"{ctx.server_url}/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": f"warmup {random.randint(1, 99999)}"}
                ],
                "max_tokens": WARMUP_TOKENS_STAGE1,
                "temperature": DEFAULT_TEMPERATURE,
            },
            timeout=WARMUP_REQUEST_TIMEOUT,
        )
        if r.status_code >= HTTP_SERVER_ERROR:
            return False
    except requests.RequestException as e:
        logger.debug("Warmup request 1 failed: %s", e)
        return False
    # Second warmup: prime speculation cache with tokens (enough to populate
    # n-gram lookup tables without wasting 5-10s on full 30-token generation)
    try:
        r = ctx.http.post(
            f"{ctx.server_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": TPS_TEST_PROMPT}],
                "max_tokens": WARMUP_TOKENS_STAGE2,
                "temperature": DEFAULT_TEMPERATURE,
            },
            timeout=WARMUP_REQUEST_TIMEOUT,
        )
        if r.status_code >= HTTP_SERVER_ERROR:
            return False
    except requests.RequestException as e:
        logger.debug("Warmup request 2 failed: %s", e)
        return False
    warmup_ms = (time.time() - warmup_start) * 1000
    if ctx.active_server_proc:
        ctx.active_server_proc = dataclasses.replace(
            ctx.active_server_proc, warmup_time_ms=warmup_ms
        )
    if ctx.vram_total_mb is None:
        if vram_init_fn is not None:
            vram_init_fn(ctx)
        else:
            init_vram_info(ctx)
    return True


def kill_process_tree(server_proc: ServerProcess | None) -> None:
    """Kill a server process and all its children -- does NOT wait for them to die.

    Uses psutil to walk the process tree recursively, ensuring child
    processes (e.g. CUDA workers) are terminated before the parent.
    Falls back to subprocess.kill() if psutil cannot access the process.
    """
    import psutil

    if server_proc is None:
        return
    try:
        parent = psutil.Process(server_proc.proc.pid)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass  # Cleanup: child already exited -- safe to ignore
        parent.kill()
    except psutil.NoSuchProcess:
        pass  # Cleanup: process already exited -- safe to ignore
    except (psutil.AccessDenied, OSError):
        # Cleanup: fall back to subprocess.kill() if psutil cannot access
        try:
            server_proc.proc.kill()
        except OSError:
            pass  # Cleanup: process already dead -- safe to ignore
    try:
        if server_proc.proc.stderr:
            server_proc.proc.stderr.close()
    except OSError:
        pass  # Cleanup: stderr pipe already closed -- safe to ignore


def _swap_port(ctx: AppContext) -> None:
    """Swap ctx.port and ctx._port_alt (ping-pong)."""
    ctx.port, ctx._port_alt = ctx._port_alt, ctx.port
    ctx.server_url = f"http://{BIND_HOST}:{ctx.port}"


def _reset_server_state(ctx: AppContext) -> None:
    """Reset all server-related state on ctx atomically.

    Clears active_server_proc, resets the HTTP session, clears the
    dying-server tracker, and leaves the port unchanged (caller decides
    whether to swap).
    """
    ctx.active_server_proc = None
    ctx.http.close()
    ctx.http = requests.Session()
    ctx._dying_server_proc = None


def kill_server(ctx: AppContext, wait: bool = True) -> None:
    """Kill the server process WE spawned.

    Args:
        wait: If True (default), wait for process death + port release.
              If False, fire-and-forget -- caller must swap ports.

    Raises:
        OSError: If socket operations fail during port-release polling.
    """
    # First, reap any previous dying process (from prior ping-pong)
    if ctx._dying_server_proc is not None:
        import psutil

        try:
            psutil.Process(ctx._dying_server_proc.proc.pid).wait(timeout=0.1)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired, OSError, AttributeError):
            pass  # Cleanup: dying process already gone or unreachable -- safe to ignore

    old_proc = ctx.active_server_proc

    kill_process_tree(old_proc)

    _reset_server_state(ctx)

    if not wait:
        # Track the dying process so atexit can clean it up
        ctx._dying_server_proc = old_proc
        return  # caller will swap to alt port -- no need to wait

    # Wait for process death + port release
    import psutil

    if old_proc is not None:
        try:
            psutil.Process(old_proc.proc.pid).wait(timeout=SERVER_KILL_WAIT_TIMEOUT)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired, OSError):
            pass  # Cleanup: process already exited or wait timed out -- safe to ignore

    import socket

    delay = SERVER_PORT_RELEASE_INITIAL_DELAY
    for _ in range(SERVER_PORT_RELEASE_RETRIES):
        time.sleep(delay)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((BIND_HOST, ctx.port)) != 0:
                return
        delay = min(delay * 2, SERVER_PORT_RELEASE_MAX_DELAY)


def is_server_running(ctx: AppContext) -> bool:
    """Quick health check -- returns True if server responds.

    Uses a direct request (no retries) to avoid 2s delay when server is offline.
    """
    try:
        r = requests.get(
            f"{ctx.server_url}/health",
            timeout=0.15,  # very short — just checking if port responds
        )
        return r.status_code == HTTP_OK
    except (requests.RequestException, OSError, ValueError):
        return False


def boot_server_with_jinja_recovery(
    ctx: AppContext,
    config: EngineConfig,
    proc: ServerProcess | None = None,
    timeout: float | None = None,
) -> tuple[ServerProcess, str]:
    """Boot server with automatic Jinja template crash recovery.

    Uses ping-pong ports: fires async kill on old port, immediately starts
    new server on the alternate port. Old process releases VRAM while new
    process does CPU-side init (GGUF parsing, buffer allocation).

    Returns (proc, status) where status is "ok", "oom", "timeout", etc.

    Raises:
        OSError: If the server executable cannot be started.
    """
    boot_start = time.time()
    if proc is None:
        if ctx.active_server_proc is not None:
            # Ping-pong: fire kill (don't wait), swap to alt port, start immediately
            kill_server(ctx, wait=False)
            _swap_port(ctx)
        proc = start_server(ctx, config)

    status = wait_for_server(ctx, proc=proc, timeout=timeout)

    if status == "jinja_error" and not ctx.no_jinja:
        logger.warning(
            "Jinja template crash detected -- auto-recovering with --no-jinja..."
        )
        ctx.no_jinja = True
        kill_server(ctx, wait=False)
        _swap_port(ctx)
        proc = start_server(ctx, config)
        status = wait_for_server(ctx, proc=proc, timeout=timeout)

    # Total time from kill to ready (includes VRAM release wait)
    proc = dataclasses.replace(proc, boot_time_ms=(time.time() - boot_start) * 1000)
    ctx.active_server_proc = proc
    return proc, status


def server_start_failed(
    ctx: AppContext, trial_num: int, params_short: str, proc: ServerProcess
) -> None:
    """Handle server start failure -- extract reason from stderr and report."""
    # Give stderr drain thread time to catch up before reading
    try:
        proc.proc.wait(timeout=2)
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("server_start_failed: proc.wait interrupted: %s", e)
    reason = ""
    try:
        with proc.lock:
            lines = list(proc.stderr_lines)
        for line in reversed(lines):
            line = line.strip()
            if _is_error_line(line):
                reason = f" -> {line[:120]}"
                break
        if not reason and lines:
            last_lines = [line.strip() for line in lines if line.strip()]
            if last_lines:
                reason = f" -> {last_lines[-1][:120]}"
    except (AttributeError, IndexError) as e:
        logger.debug("Could not extract failure reason from stderr: %s", e)
    logger.info("Trial %d: FAILED | %s%s", trial_num, params_short, reason)
    kill_server(ctx)
