"""
GPU detection, VRAM management, thermal monitoring, and competing process cleanup.

Error strategy (see errors.py for full documentation):
    Hardware detection is OPTIONAL -- the optimizer must work on CPU-only
    systems or when pynvml is not installed.  Therefore every public function
    in this module returns a safe default on failure:

    - detect_gpus() -> empty list
    - get_vram_used_mb() -> None
    - init_vram_info() -> no-op (leaves ctx.vram_total_mb as None)
    - check_thermal_throttle() -> (False, 0) (assume not throttling)
    - wait_for_cooldown() -> True (assume cool enough)
    - kill_competing_processes() -> empty list

    ImportError (pynvml absent) is caught silently (returns default).
    pynvml.NVMLError is logged at warning/debug depending on context.
    The atexit nvmlShutdown handler catches broad Exception because it
    must never prevent process exit.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable

from ._types import PynvmlModule
from .constants import (
    THERMAL_COOLDOWN_TARGET,
    THERMAL_COOLDOWN_TIMEOUT,
    THERMAL_THROTTLE_THRESHOLD,
)
from .result_types import GpuInfo, KilledProcessInfo
from .state import AppContext

logger = logging.getLogger(__name__)

__all__ = [
    "detect_gpus",
    "kill_competing_processes",
    "check_thermal_throttle",
    "wait_for_cooldown",
]

# GPU processes using more than this amount of memory will be
# killed by kill_competing_processes
_COMPETING_PROCESS_VRAM_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MB
# Seconds to wait after killing competing processes before
# resuming (allows VRAM to free)
_COMPETING_PROCESS_KILL_WAIT_SEC = 3


# ============================================================
# Persistent NVML Session
# ============================================================

_nvml_state = types.SimpleNamespace(initialized=False)
_nvml_lock = threading.RLock()


def _nvml_shutdown_atexit() -> None:
    """Called by atexit to cleanly release the NVML driver handle."""
    with _nvml_lock:
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception as e:  # Cleanup: atexit must never raise
            logger.warning("nvmlShutdown at exit failed: %s", e)
        _nvml_state.initialized = False


def _ensure_nvml() -> None:
    """Ensure NVML is initialized, initializing it at most once per process.

    Re-initializes automatically if the driver reports it is no longer active
    (NVMLError_DriverNotLoaded / NVMLError_Uninitialized). Registers
    nvmlShutdown via atexit on the first successful init so the driver handle
    is always released cleanly on process exit.

    Raises ImportError if pynvml is not installed.
    Raises pynvml.NVMLError if the driver cannot be reached.
    """
    import pynvml  # raises ImportError if absent — callers handle this

    with _nvml_lock:
        if _nvml_state.initialized:
            return
        pynvml.nvmlInit()
        _nvml_state.initialized = True
        atexit.register(_nvml_shutdown_atexit)


def _ensure_nvml_with_retry() -> None:
    """Like _ensure_nvml(), but resets the flag and retries once on NVMLError.

    Handles the case where the driver was unloaded mid-session (e.g. after a
    suspend/resume cycle) so the persistent session can recover transparently.
    """
    import pynvml

    try:
        _ensure_nvml()
    except pynvml.NVMLError:
        # Driver may have died mid-session; force re-initialization.
        with _nvml_lock:
            _nvml_state.initialized = False
        _ensure_nvml()  # propagates NVMLError if driver is truly unavailable


# ============================================================
# GPU Detection
# ============================================================


def detect_gpus() -> list[GpuInfo]:
    """Auto-detect NVIDIA GPUs via pynvml.

    Returns list of GpuInfo dicts or empty list.
    """
    try:
        import pynvml

        _ensure_nvml_with_retry()
        count = pynvml.nvmlDeviceGetCount()
        gpus: list[GpuInfo] = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_info: GpuInfo = {
                "index": i,
                "name": name,
                "vram_total_gb": round(mem.total / (1024**3), 2),
                "vram_free_gb": round(mem.free / (1024**3), 2),
            }
            gpus.append(gpu_info)
        return gpus
    except ImportError:
        return []
    except (pynvml.NVMLError, OSError, UnicodeDecodeError) as e:
        logger.warning("GPU detection failed: %s", e)
        return []


def init_vram_info(ctx: AppContext) -> None:
    """Cache total VRAM once after first successful server warmup."""
    gpus = detect_gpus()
    if gpus:
        ctx.vram_total_mb = sum(g["vram_total_gb"] * 1024 for g in gpus)


_vram_cache = types.SimpleNamespace(time=0.0, value=None)
_vram_cache_lock = threading.Lock()


def get_vram_used_mb(cache_seconds: float = 2.0) -> float | None:
    """Snapshot current VRAM usage in MB. Returns float or None.

    Caches the result for `cache_seconds` to avoid reinitializing pynvml
    on every call (detect_gpus does nvmlInit/Shutdown each time).
    During adaptive measurement (3-5 runs per trial), this saves ~10 pynvml cycles.
    """
    now = time.time()
    with _vram_cache_lock:
        if now - _vram_cache.time < cache_seconds and _vram_cache.value is not None:
            return _vram_cache.value
        gpus = detect_gpus()
        if gpus:
            value = sum((g["vram_total_gb"] - g["vram_free_gb"]) * 1024 for g in gpus)
            _vram_cache.value = value
            _vram_cache.time = now
            return value
        return None


# ============================================================
# Competing Process Cleanup
# ============================================================


# Default whitelist of system processes that must not be killed.
_DEFAULT_WHITELIST: frozenset[str] = frozenset(
    {
        "dwm.exe",
        "csrss.exe",
        "winlogon.exe",
        "explorer.exe",
        "svchost.exe",
        "nvidia-smi",
        "nvidia-persistenced",
        "Xorg",
        "Xwayland",
        "gnome-shell",
        "kwin_wayland",
        "kwin_x11",
        "plasmashell",
        "sddm",
        "gdm",
        "gdm3",
        "lightdm",
        "cinnamon",
        "mate-panel",
        "xfwm4",
        "compiz",
        "mutter",
        "sway",
        "weston",
        "wayfire",
        "hyprland",
    }
)


def _get_gpu_processes(
    pynvml_mod: PynvmlModule, handle: object, gpu_idx: int,
) -> list:
    """Collect compute + graphics processes for one GPU."""
    procs: list = []
    try:
        procs = pynvml_mod.nvmlDeviceGetComputeRunningProcesses(handle)
    except pynvml_mod.NVMLError as e:
        logger.debug("Could not get compute processes for GPU %d: %s", gpu_idx, e)
    try:
        procs += pynvml_mod.nvmlDeviceGetGraphicsRunningProcesses(handle)
    except pynvml_mod.NVMLError as e:
        logger.debug("Could not get graphics processes for GPU %d: %s", gpu_idx, e)
    return procs


def _resolve_process_name(pynvml_mod: PynvmlModule, pid: int) -> str | None:
    """Resolve a PID to a lowercase basename, or None if unresolvable."""
    try:
        raw = pynvml_mod.nvmlSystemGetProcessName(pid).decode(
            "utf-8", errors="replace"
        )
        name = os.path.basename(raw).lower()
        if name:
            return name
    except (OSError, UnicodeDecodeError, AttributeError, ValueError) as e:
        logger.debug("Could not get process name for PID %d: %s", pid, e)
    except pynvml_mod.NVMLError as e:
        logger.debug("Could not get process name for PID %d: %s", pid, e)
    return None


def _kill_process(pid: int, name: str, gpu_mb: int) -> KilledProcessInfo | None:
    """Attempt to kill a single process; return info dict on success."""
    logger.warning("Killing %s (PID %d) — %d MB GPU memory", name, pid, gpu_mb)
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                timeout=10,
            )
        else:
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        return {"pid": pid, "name": name, "gpu_mb": gpu_mb}
    except (
        ProcessLookupError,
        PermissionError,
        OSError,
        subprocess.SubprocessError,
    ) as e:
        logger.warning("Failed to kill PID %d: %s", pid, e)
        return None


def _gather_killable_gpu_processes(
    pynvml_mod: PynvmlModule,
    gpu_count: int,
    whitelist_lower: set[str],
    threshold_bytes: int,
) -> list[tuple[int, str, int]]:
    """Scan all GPUs and return non-whitelisted processes exceeding the VRAM threshold.

    Returns a list of (pid, process_name, gpu_mb) tuples — candidates for killing.
    Does NOT kill anything.
    """
    candidates: list[tuple[int, str, int]] = []
    for i in range(gpu_count):
        handle = pynvml_mod.nvmlDeviceGetHandleByIndex(i)
        procs = _get_gpu_processes(pynvml_mod, handle, i)
        seen: set[int] = set()
        for proc in procs:
            if proc.pid in seen:
                continue
            seen.add(proc.pid)
            used = proc.usedGpuMemory or 0
            if used < threshold_bytes:
                continue
            base_name = _resolve_process_name(pynvml_mod, proc.pid)
            if base_name is None:
                logger.debug(
                    "Skipping PID %d — could not resolve process name", proc.pid
                )
                continue
            if base_name in whitelist_lower:
                continue
            candidates.append((proc.pid, base_name, used // (1024 * 1024)))
    return candidates


def kill_competing_processes(
    whitelist: set[str] | None = None,
    confirm_callback: Callable[[list[str]], bool] | None = None,
) -> list[KilledProcessInfo]:
    """Kill non-whitelisted processes using >500MB of GPU memory.

    Args:
        whitelist: Process names to never kill. Defaults to built-in system whitelist.
        confirm_callback: Optional callback that receives a list of human-readable
            process descriptions (e.g. ``["chrome.exe (PID 1234) — 1024 MB GPU"]``)
            and returns True to proceed with killing or False to abort.
            If None, processes are killed without confirmation (backward compatible).
    """
    whitelist_lower = {w.lower() for w in (whitelist or _DEFAULT_WHITELIST)}
    try:
        import pynvml
    except ImportError:
        logger.warning("pynvml not installed — cannot detect GPU processes")
        return []

    try:
        _ensure_nvml_with_retry()
        candidates = _gather_killable_gpu_processes(
            pynvml,
            pynvml.nvmlDeviceGetCount(),
            whitelist_lower,
            _COMPETING_PROCESS_VRAM_THRESHOLD_BYTES,
        )
    except pynvml.NVMLError as e:
        logger.error("NVML error: %s", e)
        return []

    if not candidates:
        return []

    # If a confirmation callback is provided, ask before killing.
    if confirm_callback is not None:
        descriptions = [
            f"{name} (PID {pid}) — {gpu_mb} MB GPU"
            for pid, name, gpu_mb in candidates
        ]
        if not confirm_callback(descriptions):
            logger.info("User declined to kill competing processes")
            return []

    killed: list[KilledProcessInfo] = []
    for pid, name, gpu_mb in candidates:
        result = _kill_process(pid, name, gpu_mb)
        if result is not None:
            killed.append(result)

    if killed:
        logger.info(
            "Killed %d competing process(es). Waiting %ds...",
            len(killed),
            _COMPETING_PROCESS_KILL_WAIT_SEC,
        )
        time.sleep(_COMPETING_PROCESS_KILL_WAIT_SEC)
    return killed


# ============================================================
# Thermal Monitoring
# ============================================================


def check_thermal_throttle(
    threshold: int = THERMAL_THROTTLE_THRESHOLD,
) -> tuple[bool, int]:
    """Check if any GPU is at/above thermal threshold.

    Returns (is_throttling, temp_c).
    """
    try:
        import pynvml

        _ensure_nvml_with_retry()
        max_temp = 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                max_temp = max(max_temp, temp)
            except pynvml.NVMLError as e:
                logger.debug("Could not read GPU %d temperature: %s", i, e)
        return max_temp >= threshold, max_temp
    except ImportError:
        return False, 0
    except pynvml.NVMLError as e:
        logger.warning("Thermal check failed: %s", e)
        return False, 0


def wait_for_cooldown(
    target_temp: int = THERMAL_COOLDOWN_TARGET, timeout: int = THERMAL_COOLDOWN_TIMEOUT
) -> bool:
    """Wait for GPU to cool below target temperature."""
    throttling, current_temp = check_thermal_throttle(threshold=target_temp)
    if not throttling:
        return True
    logger.info("GPU at %d°C — waiting for cooldown to %d°C", current_temp, target_temp)
    start = time.time()
    while time.time() - start < timeout:
        _, current_temp = check_thermal_throttle(threshold=target_temp)
        if current_temp < target_temp:
            logger.info("GPU cooled to %d°C — resuming", current_temp)
            return True
        time.sleep(5)
    logger.warning(
        "Cooldown timeout — GPU still at %d°C. Proceeding anyway.", current_temp
    )
    return False
