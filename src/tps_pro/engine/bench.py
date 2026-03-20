"""
llama-bench integration: _build_bench_cmd, _parse_bench_csv, run_bench_trial.

Error strategy (see errors.py for full documentation):
    - run_bench_trial(): returns None on non-OOM failure (timeout, parse
      error, process error).  Raises BenchOOMError on OOM so the caller
      can prune the trial.  The None-return pattern is intentional:
      a single bench failure should not abort the whole phase.
    - _parse_bench_csv(): returns None on parse failure (logged at debug).
"""

from __future__ import annotations

import csv
import io
import logging
import subprocess
import sys
from typing import Any, cast

from ..constants import BENCH_SUBPROCESS_TIMEOUT
from ..errors import BenchOOMError
from ..result_types import BenchResult, EngineConfig
from ..state import AppContext
from .parsing import is_oom

logger = logging.getLogger(__name__)

__all__ = [
    "BenchOOMError",
    "run_bench_trial",
]


def _build_bench_cmd(  # noqa: C901
    ctx: AppContext,
    engine_config: EngineConfig,
    n_prompt: int = 512,
    n_gen: int = 128,
    repetitions: int = 3,
) -> list[str]:
    """Map engine_config to llama-bench CLI args. Returns command list."""
    cmd = [
        str(ctx.bench_path),
        "-m",
        str(ctx.model_path),
        "-p",
        str(n_prompt),
        "-n",
        str(n_gen),
        "-r",
        str(repetitions),
        "-o",
        "csv",
    ]
    flag_map = {
        "n_gpu_layers": "-ngl",
        "threads": "-t",
        "batch_size": "-b",
        "ubatch_size": "-ub",
        "n_cpu_moe": "-ncmoe",
        "poll": "--poll",
        "cpu_strict": "--cpu-strict",
        "prio": "--prio",
        "tensor_split": "-ts",
        "override_tensor": "-ot",
    }
    _ec = cast(dict[str, Any], engine_config)
    for key, flag in flag_map.items():
        if key in engine_config:
            cmd.extend([flag, str(_ec[key])])
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


def _parse_bench_csv(csv_output: str) -> BenchResult | None:
    """Parse llama-bench CSV output into a BenchResult."""
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
            return BenchResult(
                tps=gen_ts,
                prompt_tps=pp_ts or 0.0,
                ttft=pp_total_ns / 1e6 if pp_total_ns else 0.0,
                total_ms=(pp_total_ns + gen_total_ns) / 1e6,
            )
    except (csv.Error, ValueError, KeyError, IndexError, TypeError) as e:
        logger.debug("Bench CSV parsing failed: %s", e)
    return None


def run_bench_trial(
    ctx: AppContext,
    engine_config: EngineConfig,
    n_prompt: int = 512,
    n_gen: int = 128,
    repetitions: int = 3,
) -> BenchResult | None:
    """Run a single llama-bench trial.

    Returns BenchResult on success, None on non-OOM failure.

    Raises:
        BenchOOMError: When the llama-bench process fails due to
            out-of-memory (detected via stderr OOM patterns).
    """
    if not ctx.bench_path:
        return None
    cmd = _build_bench_cmd(ctx, engine_config, n_prompt, n_gen, repetitions)
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=BENCH_SUBPROCESS_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            if sys.platform == "win32"
            else 0,
        )
        if result.returncode != 0:
            if is_oom(result.stderr):
                raise BenchOOMError(f"OOM during bench trial: {result.stderr[:200]}")
            if ctx.debug:
                logger.debug("[bench] returncode=%s", result.returncode)
                logger.debug("[bench] stderr: %s", result.stderr[:500])
            return None
        return _parse_bench_csv(result.stdout)
    except subprocess.TimeoutExpired:
        return None
    except BenchOOMError:
        raise
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("Bench trial failed: %s", e)
        return None
