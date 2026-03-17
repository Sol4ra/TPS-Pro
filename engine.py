"""
Server lifecycle (start/kill/wait), llama-bench integration, LogTee, PhaseTimer.
"""

import csv
import io
import logging
import os
import random
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

from .state import ctx, _config, get_preset_trials
from .constants import TPS_TEST_PROMPT
from .hardware import _init_vram_info

# Shared OOM keyword list — used by wait_for_server, run_bench_trial, _server_start_failed
_OOM_KEYWORDS = [
    "out of memory", "oom", "alloc failed", "cuda error",
    "not enough memory", "failed to allocate", "ggml_cuda_op_mul_mat",
    "cudamalloc failed", "insufficient memory",
]

_ERROR_KEYWORDS = [
    "error", "failed", "abort", "oom", "alloc",
    "cuda", "memory", "unknown", "invalid",
]


def _is_oom(text):
    """Check if text contains OOM-related keywords."""
    lower = text.lower()
    return any(kw in lower for kw in _OOM_KEYWORDS)


def _is_error_line(line):
    """Check if a stderr line looks like an error."""
    lower = line.lower()
    return bool(line.strip()) and any(kw in lower for kw in _ERROR_KEYWORDS)


# ============================================================
# LogTee — duplicate stdout to log file
# ============================================================

class LogTee:
    """Tees stdout to both the console and a timestamped log file."""

    def __init__(self, results_dir):
        import atexit
        self._original_stdout = sys.stdout
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(results_dir) / f"optimize_{ts}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered
        self.log_path = str(log_path)
        atexit.register(self.close)

    def write(self, data):
        self._original_stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._original_stdout.flush()
        self._file.flush()

    def close(self):
        sys.stdout = self._original_stdout
        self._file.close()


# ============================================================
# Dry-run mode
# ============================================================

def check_dry_run(phase_name, config, n_trials):
    """If dry-run mode is active, print what WOULD happen and return True."""
    if not ctx.dry_run:
        return False
    trial_str = "full sweep" if n_trials == "full" else f"{n_trials} trials"
    print(f"\n{'=' * 50}")
    print(f"  [DRY RUN] Phase: {phase_name}")
    print(f"  Trials: {trial_str}")
    if config:
        print(f"  Config keys: {sorted(config.keys())}")
    print(f"{'=' * 50}")
    return True


# ============================================================
# Multi-GPU Tensor Split Generation
# ============================================================

def generate_tensor_splits(gpu_count):
    """Generate candidate tensor split ratio tuples for 2-4 GPUs."""
    step = 0.05
    splits = []
    if gpu_count == 2:
        ratio = step
        while ratio <= 1.0 - step:
            splits.append((round(ratio, 2), round(1.0 - ratio, 2)))
            ratio += step
    elif gpu_count == 3:
        for a_int in range(1, 19):
            a = round(a_int * step, 2)
            for b_int in range(1, 19):
                b = round(b_int * step, 2)
                c = round(1.0 - a - b, 2)
                if c >= step:
                    splits.append((a, b, c))
    elif gpu_count >= 4:
        even = round(1.0 / gpu_count, 2)
        splits.append(tuple([even] * gpu_count))
        for primary in range(gpu_count):
            for boost in [0.10, 0.20, 0.30]:
                split = [even] * gpu_count
                split[primary] = round(even + boost, 2)
                deficit = boost / (gpu_count - 1)
                for j in range(gpu_count):
                    if j != primary:
                        split[j] = round(even - deficit, 2)
                if all(v > 0.02 for v in split):
                    total = sum(split)
                    splits.append(tuple(round(v / total, 2) for v in split))
                split2 = [even] * gpu_count
                split2[primary] = round(even - boost, 2)
                surplus = boost / (gpu_count - 1)
                for j in range(gpu_count):
                    if j != primary:
                        split2[j] = round(even + surplus, 2)
                if all(v > 0.02 for v in split2):
                    total = sum(split2)
                    splits.append(tuple(round(v / total, 2) for v in split2))
        splits = list(set(splits))
    return splits


# ============================================================
# BaselineFailure Exception
# ============================================================

class BaselineFailure(Exception):
    """Raised when a baseline server fails to start with --fail-fast enabled."""
    pass


# ============================================================
# PhaseTimer — ETA and timing tracking
# ============================================================

class PhaseTimer:
    """Track timing for phases and individual trials to provide ETAs."""

    def __init__(self):
        self._phases = {}
        self._trial_durations = []

    def start_phase(self, name):
        self._phases[name] = {"start": time.time(), "end": None, "duration": None}
        self._trial_durations = []

    def end_phase(self, name):
        if name in self._phases and self._phases[name]["start"]:
            self._phases[name]["end"] = time.time()
            self._phases[name]["duration"] = self._phases[name]["end"] - self._phases[name]["start"]

    def record_trial(self, duration):
        self._trial_durations.append(duration)
        if len(self._trial_durations) > 20:
            self._trial_durations = self._trial_durations[-20:]

    def eta_pipeline(self, remaining_phases):
        """Estimate remaining time based on completed phase durations."""
        FALLBACK_SEC_PER_TRIAL = 90
        display_to_key = {"GPU Offload": "gpu", "Tensor Split": "tensor_split",
                          "MoE": "moe", "Compute": "compute", "Memory": "memory",
                          "MoE Audit": "moe_audit", "Compute Audit": "compute_audit",
                          "Memory Audit": "memory_audit", "Quality": "quality"}
        total_completed_sec = 0.0
        total_completed_trials = 0
        for phase_name, p_data in self._phases.items():
            if p_data.get("duration") and p_data["duration"] > 0:
                preset_key = display_to_key.get(phase_name, phase_name)
                trial_count = get_preset_trials(_config.get("preset", "normal"), preset_key)
                if trial_count > 0:
                    total_completed_sec += p_data["duration"]
                    total_completed_trials += trial_count
        if total_completed_trials > 0:
            sec_per_trial = total_completed_sec / total_completed_trials
        else:
            sec_per_trial = FALLBACK_SEC_PER_TRIAL
        total_remaining_trials = sum(get_preset_trials(_config.get("preset", "normal"), p) for p in remaining_phases)
        remaining_sec = total_remaining_trials * sec_per_trial
        if remaining_sec < 60: return f"{remaining_sec:.0f}s"
        elif remaining_sec < 3600: return f"{remaining_sec / 60:.0f}m"
        else: return f"{remaining_sec / 3600:.1f}h"

    def eta(self, remaining_trials):
        if not self._trial_durations or remaining_trials <= 0:
            return "unknown"
        avg = sum(self._trial_durations) / len(self._trial_durations)
        remaining_sec = avg * remaining_trials
        if remaining_sec < 60: return f"{remaining_sec:.0f}s"
        elif remaining_sec < 3600: return f"{remaining_sec / 60:.1f}m"
        else: return f"{remaining_sec / 3600:.1f}h"

    def summary(self):
        print("\n  Phase Timing Summary")
        print("  " + "-" * 45)
        total = 0.0
        for name, info in self._phases.items():
            dur = info.get("duration")
            if dur is not None:
                total += dur
                dur_str = f"{dur:.1f}s" if dur < 60 else (f"{dur / 60:.1f}m" if dur < 3600 else f"{dur / 3600:.1f}h")
                print(f"  {name:20s}  {dur_str:>8s}")
            else:
                print(f"  {name:20s}  {'skipped':>8s}")
        print("  " + "-" * 45)
        t_str = f"{total / 60:.1f}m" if total < 3600 else f"{total / 3600:.1f}h"
        print(f"  {'Total':20s}  {t_str}")


# ============================================================
# llama-bench integration
# ============================================================

def _build_bench_cmd(engine_config, n_prompt=512, n_gen=128, repetitions=3):
    """Map engine_config to llama-bench CLI args. Returns command list."""
    cmd = [
        str(ctx.bench_path),
        "-m", str(ctx.model_path),
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-r", str(repetitions),
        "-o", "csv",
    ]
    flag_map = {
        "n_gpu_layers": "-ngl", "threads": "-t", "batch_size": "-b",
        "ubatch_size": "-ub", "n_cpu_moe": "-ncmoe", "poll": "--poll",
        "cpu_strict": "--cpu-strict", "prio": "--prio",
        "tensor_split": "-ts", "override_tensor": "-ot",
    }
    for key, flag in flag_map.items():
        if key in engine_config:
            cmd.extend([flag, str(engine_config[key])])
    if "kv_cache_type" in engine_config:
        cmd.extend(["-ctk", engine_config["kv_cache_type"]])
        cmd.extend(["-ctv", engine_config["kv_cache_type"]])
    if "cache_type_k" in engine_config:
        cmd.extend(["-ctk", engine_config["cache_type_k"]])
    if "cache_type_v" in engine_config:
        cmd.extend(["-ctv", engine_config["cache_type_v"]])
    if engine_config.get("flash_attn") in ("on", True, "1", 1):
        cmd.extend(["-fa", "1"])
    elif "flash_attn" in engine_config:
        cmd.extend(["-fa", "0"])
    if engine_config.get("no_mmap"):
        cmd.extend(["-mmp", "0"])
    if engine_config.get("direct_io"):
        cmd.extend(["-dio", "1"])
    if engine_config.get("op_offload") is False:
        cmd.extend(["-nopo", "1"])
    if engine_config.get("kv_offload") is False:
        cmd.extend(["-nkvo", "1"])
    if engine_config.get("no_host"):
        cmd.extend(["--no-host", "1"])
    return cmd


def _parse_bench_csv(csv_output):
    """Parse llama-bench CSV output into a perf dict."""
    try:
        reader = csv.DictReader(io.StringIO(csv_output))
        pp_ts = pp_ns = gen_ts = gen_ns = None
        pp_n_tokens = gen_n_tokens = 0
        for row in reader:
            n_prompt = int(row.get("n_prompt", 0))
            n_gen = int(row.get("n_gen", 0))
            avg_ts = float(row.get("avg_ts", 0))
            avg_ns = float(row.get("avg_ns", 0))
            if n_gen == 0 and n_prompt > 0:
                pp_ts, pp_ns, pp_n_tokens = avg_ts, avg_ns, n_prompt
            elif n_prompt == 0 and n_gen > 0:
                gen_ts, gen_ns, gen_n_tokens = avg_ts, avg_ns, n_gen
        if gen_ts and gen_ts > 0:
            pp_total_ns = (pp_ns * pp_n_tokens) if pp_ns and pp_n_tokens else 0
            gen_total_ns = (gen_ns * gen_n_tokens) if gen_ns and gen_n_tokens else 0
            return {
                "tps": gen_ts,
                "prompt_tps": pp_ts or 0.0,
                "ttft": pp_total_ns / 1e6 if pp_total_ns else 0.0,
                "total_ms": (pp_total_ns + gen_total_ns) / 1e6,
            }
    except (subprocess.SubprocessError, csv.Error, ValueError, KeyError) as e:
        logger.debug("Bench CSV parsing failed: %s", e)
    return None


def run_bench_trial(engine_config, n_prompt=512, n_gen=128, repetitions=3):
    """Run a single llama-bench trial. Returns perf dict, {"error": "oom"}, or None."""
    if not ctx.bench_path:
        return None
    cmd = _build_bench_cmd(engine_config, n_prompt, n_gen, repetitions)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )
        if result.returncode != 0:
            if _is_oom(result.stderr):
                return {"error": "oom"}
            if ctx.debug:
                print(f"    [bench] returncode={result.returncode}")
                print(f"    [bench] stderr: {result.stderr[:500]}")
            return None
        return _parse_bench_csv(result.stdout)
    except subprocess.TimeoutExpired:
        return None
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("Bench trial failed: %s", e)
        return None


# ============================================================
# Server Lifecycle
# ============================================================

def start_server(engine_config):
    """Start llama-server with given engine config."""
    env = os.environ.copy()
    if engine_config.get("cuda_graph_opt"):
        env["GGML_CUDA_GRAPH_OPT"] = "1"

    cmd = [
        str(ctx.server_path),
        "-m", str(ctx.model_path),
        "--port", str(ctx.port),
        "--host", "127.0.0.1",
        "-ngl", str(engine_config.get("n_gpu_layers", 99)),
        "-c", str(engine_config.get("context", 4096)),
        "--parallel", str(engine_config.get("parallel", 1)),
    ]
    if ctx.no_jinja:
        cmd.extend(["--no-jinja", "--chat-template", "chatml"])
    elif ctx.chat_template_path and ctx.chat_template_path.is_file():
        cmd.extend(["--chat-template-file", str(ctx.chat_template_path)])
    if engine_config.get("warmup") is False:
        cmd.append("--no-warmup")
    if engine_config.get("cache_prompt") is False:
        cmd.append("--no-cache-prompt")
    if engine_config.get("fit") is False:
        cmd.append("--fit=off")

    # Only add flags that are explicitly in the config
    flag_pairs = [
        ("batch_size", "-b"), ("ubatch_size", "--ubatch-size"),
        ("threads", "-t"), ("threads_batch", "-tb"),
        ("n_cpu_moe", "--n-cpu-moe"),
    ]
    for key, flag in flag_pairs:
        if key in engine_config:
            cmd.extend([flag, str(engine_config[key])])

    if "expert_used_count" in engine_config and ctx.expert_override_key:
        if engine_config["expert_used_count"] != ctx.default_experts:
            cmd.extend(["--override-kv", f"{ctx.expert_override_key}=int:{engine_config['expert_used_count']}"])

    toggle_pairs = [
        ("poll", "--poll"), ("poll_batch", "--poll-batch"),
        ("prio", "--prio"), ("prio_batch", "--prio-batch"),
    ]
    for key, flag in toggle_pairs:
        if key in engine_config:
            cmd.extend([flag, str(engine_config[key])])

    # KV cache types
    if "kv_cache_type" in engine_config:
        cmd.extend(["--cache-type-k", engine_config["kv_cache_type"]])
        cmd.extend(["--cache-type-v", engine_config["kv_cache_type"]])
    if "cache_type_k" in engine_config:
        cmd.extend(["--cache-type-k", engine_config["cache_type_k"]])
    if "cache_type_v" in engine_config:
        cmd.extend(["--cache-type-v", engine_config["cache_type_v"]])
    if engine_config.get("flash_attn") in ("on", True, "1", 1):
        cmd.extend(["--flash-attn", "1"])
    elif "flash_attn" in engine_config:
        cmd.extend(["--flash-attn", "0"])
    if "n_predict" in engine_config:
        cmd.extend(["--n-predict", str(engine_config["n_predict"])])
    if "temp" in engine_config:
        cmd.extend(["--temp", str(engine_config["temp"])])
    if "model_draft" in engine_config:
        cmd.extend(["--model-draft", str(engine_config["model_draft"])])
    if "cache_reuse" in engine_config:
        cmd.extend(["--cache-reuse", str(engine_config["cache_reuse"])])

    # Speculation params
    spec_pairs = [
        ("spec_type", "--spec-type"), ("spec_ngram_n", "--spec-ngram-size-n"),
        ("spec_ngram_m", "--spec-ngram-size-m"), ("spec_ngram_min_hits", "--spec-ngram-min-hits"),
        ("draft_max", "--draft"), ("draft_min", "--draft-min"),
        ("draft_p_min", "--draft-p-min"),
    ]
    for key, flag in spec_pairs:
        if key in engine_config:
            cmd.extend([flag, str(engine_config[key])])

    # CPU placement
    if "cpu_strict" in engine_config:
        cmd.extend(["--cpu-strict", str(engine_config["cpu_strict"])])
    if "cpu_strict_batch" in engine_config:
        cmd.extend(["--cpu-strict-batch", str(engine_config["cpu_strict_batch"])])

    # Boolean flags
    bool_flags = [
        ("swa_full", "--swa-full", True), ("repack", "--no-repack", False),
        ("op_offload", "--no-op-offload", False), ("kv_unified", "--kv-unified", True),
        ("mlock", "--mlock", True), ("no_mmap", "--no-mmap", True),
        ("kv_offload", "--no-kv-offload", False), ("no_host", "--no-host", True),
        ("direct_io", "--direct-io", True), ("cont_batching", "--no-cont-batching", False),
        ("backend_sampling", "--backend-sampling", True), ("context_shift", "--no-context-shift", False),
    ]
    for key, flag, trigger_val in bool_flags:
        if engine_config.get(key) is trigger_val:
            cmd.append(flag)

    # Extended params
    ext_pairs = [
        ("ctx_checkpoints", "--ctx-checkpoints"), ("checkpoint_every_n", "--checkpoint-every-n-tokens"),
        ("cache_ram", "--cache-ram"), ("threads_http", "--threads-http"),
    ]
    for key, flag in ext_pairs:
        if key in engine_config:
            cmd.extend([flag, str(engine_config[key])])
    if engine_config.get("lookup_cache_dynamic"):
        cmd.extend(["--lookup-cache-dynamic", str(engine_config["lookup_cache_dynamic"])])
    if "numa" in engine_config:
        cmd.extend(["--numa", str(engine_config["numa"])])
    if engine_config.get("cpu_moe"):
        cmd.append("--cpu-moe")
    if engine_config.get("override_tensor"):
        cmd.extend(["-ot", engine_config["override_tensor"]])
    if "tensor_split" in engine_config:
        cmd.extend(["--tensor-split", engine_config["tensor_split"]])

    if ctx.debug:
        flags = cmd[3:]
        print(f"    [debug] cmd: {' '.join(str(f) for f in flags)}")

    proc = subprocess.Popen(
        cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
    )
    ctx.active_server_proc = proc

    proc._stderr_lines = []
    def _drain_stderr():
        try:
            for line in proc.stderr:
                proc._stderr_lines.append(line.decode("utf-8", errors="replace").rstrip())
        except Exception:
            pass
    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()

    return proc


def wait_for_server(proc=None, timeout=None):
    """Wait for llama-server /health to return ok."""
    if timeout is None:
        size_timeouts = {"tiny": 60, "small": 120, "medium": 300, "large": 600}
        timeout = size_timeouts.get(getattr(ctx, "model_size_class", "medium"), 300)
        if ctx.is_moe:
            timeout = int(timeout * 1.5)

    start = time.time()
    while True:
        if proc is not None and proc.poll() is not None:
            stderr_text = "\n".join(getattr(proc, "_stderr_lines", []))
            if any(kw in stderr_text.lower() for kw in ["jinja", "template error", "template parse"]):
                proc._jinja_error = True
                print("[!] Server died with Jinja/template error.")
                return "jinja_error"
            if _is_oom(stderr_text):
                proc._oom_error = True
                print("[!] Server died with OOM error.")
                return "oom"
            return "died"
        if time.time() - start > timeout:
            return "timeout"
        try:
            r = ctx.http.get(f"{ctx.server_url}/health", timeout=0.5)
            if r.status_code == 200 and r.json().get("status") == "ok":
                if proc is not None:
                    stderr_text = "\n".join(getattr(proc, "_stderr_lines", []))
                    if "flash attention not supported" in stderr_text.lower():
                        ctx._flash_attn_disabled_for_kv = True
                if warmup_server():
                    return "ok"
        except (requests.ConnectionError, requests.Timeout, ValueError):
            pass
        time.sleep(0.1)


def warmup_server():
    """Two-stage warmup: CUDA pipeline warm + speculation cache prime."""
    try:
        r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": f"warmup {random.randint(1, 99999)}"}],
            "max_tokens": 5, "temperature": 0.4,
        }, timeout=60)
        if r.status_code >= 500:
            return False
    except requests.RequestException as e:
        logger.debug("Warmup request 1 failed: %s", e)
        return False
    # Second warmup: prime speculation cache with 10 tokens (enough to populate
    # n-gram lookup tables without wasting 5-10s on full 30-token generation)
    try:
        r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": TPS_TEST_PROMPT}],
            "max_tokens": 10, "temperature": 0.4,
        }, timeout=60)
        if r.status_code >= 500:
            return False
    except requests.RequestException as e:
        logger.debug("Warmup request 2 failed: %s", e)
        return False
    if ctx.vram_total_mb is None:
        _init_vram_info()
    return True


def kill_server():
    """Kill only the server process WE spawned (by PID)."""
    import psutil

    if ctx.active_server_proc is not None:
        pid = ctx.active_server_proc.pid
        try:
            parent = psutil.Process(pid)
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            parent.kill()
            parent.wait(timeout=5)
        except psutil.NoSuchProcess:
            pass
        except Exception:
            try:
                ctx.active_server_proc.kill()
                ctx.active_server_proc.wait(timeout=5)
            except Exception:
                pass
        try:
            if ctx.active_server_proc.stderr:
                ctx.active_server_proc.stderr.close()
        except Exception:
            pass
        ctx.active_server_proc = None

    # Reset HTTP session — close() invalidates the urllib3 pool, so we must
    # create a new Session. Connection pooling benefit is negligible for localhost.
    ctx.http.close()
    ctx.http = requests.Session()

    import socket
    delay = 0.05  # start at 50ms, exponential backoff
    for _ in range(12):
        time.sleep(delay)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", ctx.port)) != 0:
                return
        delay = min(delay * 2, 0.5)  # cap at 500ms


def is_server_running():
    """Quick health check — returns True if server responds."""
    try:
        r = ctx.http.get(f"{ctx.server_url}/health", timeout=0.5)
        return r.status_code == 200
    except Exception:
        return False


def _boot_server_with_jinja_recovery(config, proc=None, timeout=None):
    """Boot server with automatic Jinja template crash recovery.

    If wait_for_server returns "jinja_error" and ctx.no_jinja is not already set,
    automatically flips ctx.no_jinja=True and retries once.

    Returns (proc, status) where status is "ok", "oom", "timeout", etc.
    """
    if proc is None:
        kill_server()
        proc = start_server(config)

    kwargs = {"proc": proc}
    if timeout is not None:
        kwargs["timeout"] = timeout
    status = wait_for_server(**kwargs)

    if status == "jinja_error" and not ctx.no_jinja:
        print("    [!] Jinja template crash detected — auto-recovering with --no-jinja...")
        ctx.no_jinja = True
        kill_server()
        proc = start_server(config)
        kwargs["proc"] = proc
        status = wait_for_server(**kwargs)

    return proc, status


def _server_start_failed(trial_num, params_short, proc):
    """Handle server start failure — extract reason from stderr and report."""
    # Give stderr drain thread time to catch up before reading
    try:
        proc.wait(timeout=2)
    except Exception:
        pass
    reason = ""
    try:
        lines = getattr(proc, "_stderr_lines", [])
        for line in reversed(lines):
            line = line.strip()
            if _is_error_line(line):
                reason = f" → {line[:120]}"
                break
        if not reason and lines:
            last_lines = [l.strip() for l in lines if l.strip()]
            if last_lines:
                reason = f" → {last_lines[-1][:120]}"
    except Exception:
        pass
    print(f"  Trial {trial_num}: FAILED | {params_short}{reason}")
    kill_server()


def start_naked_server():
    """Start llama-server with the current naked_engine config. Returns proc or None."""
    from .state import ctx
    kill_server()
    proc = start_server(ctx.naked_engine)
    status = wait_for_server(proc=proc)
    if status != "ok":
        print("[!] Server failed to start.")
        kill_server()
        return None
    return proc
