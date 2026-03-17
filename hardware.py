"""
GPU detection, VRAM management, thermal monitoring, and competing process cleanup.
"""

import os
import signal
import subprocess
import sys
import time

from .state import ctx


# ============================================================
# GPU Detection
# ============================================================

def detect_gpus():
    """Auto-detect NVIDIA GPUs via pynvml. Returns list of dicts or empty list."""
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append({
                "index": i, "name": name,
                "vram_total_gb": round(mem.total / (1024**3), 2),
                "vram_free_gb": round(mem.free / (1024**3), 2),
            })
        pynvml.nvmlShutdown()
        return gpus
    except Exception:
        return []


def _init_vram_info():
    """Cache total VRAM once after first successful server warmup."""
    gpus = detect_gpus()
    if gpus:
        ctx.vram_total_mb = sum(g["vram_total_gb"] * 1024 for g in gpus)


def _get_vram_used_mb():
    """Snapshot current VRAM usage in MB. Returns float or None."""
    gpus = detect_gpus()
    if gpus:
        return sum((g["vram_total_gb"] - g["vram_free_gb"]) * 1024 for g in gpus)
    return None


# ============================================================
# Competing Process Cleanup
# ============================================================

def kill_competing_processes(whitelist=None):
    """Kill non-whitelisted processes using >500MB of GPU memory."""
    if whitelist is None:
        whitelist = {"dwm.exe", "csrss.exe", "winlogon.exe", "explorer.exe",
                     "svchost.exe", "nvidia-smi", "nvidia-persistenced",
                     "Xorg", "Xwayland", "gnome-shell", "kwin_wayland",
                     "kwin_x11", "plasmashell", "sddm", "gdm", "gdm3",
                     "lightdm", "cinnamon", "mate-panel", "xfwm4",
                     "compiz", "mutter", "sway", "weston", "wayfire",
                     "hyprland"}
    whitelist_lower = {w.lower() for w in whitelist}
    try:
        import pynvml
    except ImportError:
        print("[!] pynvml not installed — cannot detect GPU processes")
        return []
    killed = []
    threshold_bytes = 500 * 1024 * 1024
    try:
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            procs = []
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except Exception:
                pass
            try:
                procs += pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            except Exception:
                pass
            seen = set()
            for proc in procs:
                if proc.pid in seen:
                    continue
                seen.add(proc.pid)
                used = proc.usedGpuMemory or 0
                if used < threshold_bytes:
                    continue
                proc_name = f"pid-{proc.pid}"
                try:
                    proc_name = pynvml.nvmlSystemGetProcessName(proc.pid).decode("utf-8", errors="replace")
                except Exception:
                    pass
                base_name = os.path.basename(proc_name).lower()
                if any(w in base_name for w in whitelist_lower):
                    continue
                print(f"  [!] Killing {base_name} (PID {proc.pid}) — {used // (1024*1024)} MB GPU memory")
                try:
                    if sys.platform == "win32":
                        subprocess.run(["taskkill", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=10)
                    else:
                        os.kill(proc.pid, signal.SIGKILL)
                    killed.append({"pid": proc.pid, "name": base_name, "gpu_mb": used // (1024*1024)})
                except Exception as e:
                    print(f"    [!] Failed to kill PID {proc.pid}: {e}")
    except Exception as e:
        print(f"[!] NVML error: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
    if killed:
        print(f"  Killed {len(killed)} competing process(es). Waiting 3s...")
        time.sleep(3)
    return killed


# ============================================================
# Thermal Monitoring
# ============================================================

def check_thermal_throttle(threshold=85):
    """Check if any GPU is at/above thermal threshold. Returns (is_throttling, temp_c)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        max_temp = 0
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                max_temp = max(max_temp, temp)
            except Exception:
                pass
        pynvml.nvmlShutdown()
        return max_temp >= threshold, max_temp
    except Exception:
        return False, 0


def wait_for_cooldown(target_temp=75, timeout=120):
    """Wait for GPU to cool below target temperature."""
    throttling, current_temp = check_thermal_throttle(threshold=target_temp)
    if not throttling:
        return True
    print(f"\n  [*] GPU at {current_temp}°C — waiting for cooldown to {target_temp}°C")
    start = time.time()
    while time.time() - start < timeout:
        _, current_temp = check_thermal_throttle(threshold=target_temp)
        if current_temp < target_temp:
            print(f"  [*] GPU cooled to {current_temp}°C — resuming")
            return True
        time.sleep(5)
    print(f"  [!] Cooldown timeout — GPU still at {current_temp}°C. Proceeding anyway.")
    return False
