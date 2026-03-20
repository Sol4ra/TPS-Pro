"""Command-building helpers for llama-server CLI arguments."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from ..constants import BIND_HOST
from ..result_types import EngineConfig
from ..state import AppContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Numeric bounds for engine config values (clamped with warning on violation)
# ---------------------------------------------------------------------------

_NUMERIC_BOUNDS: dict[str, tuple[int, int]] = {
    "threads": (1, 256),
    "threads_batch": (1, 256),
    "batch_size": (1, 65536),
    "ubatch_size": (1, 65536),
    "context": (1, 2_097_152),
    "n_gpu_layers": (0, 999),
    "parallel": (1, 256),
    "expert_used_count": (1, 256),
}


def _clamp_numeric(engine_config: EngineConfig) -> EngineConfig:
    """Return a copy of *engine_config* with numeric values clamped to valid ranges.

    Out-of-range values are clamped and a warning is logged for each violation.
    """
    clamped = dict(engine_config)
    for key, (lo, hi) in _NUMERIC_BOUNDS.items():
        if key in clamped:
            raw = clamped[key]
            try:
                val = int(raw)
            except (TypeError, ValueError):
                continue
            if val < lo or val > hi:
                new_val = max(lo, min(hi, val))
                logger.warning(
                    "Clamped %s=%s to valid range [%s, %s] -> %s",
                    key,
                    val,
                    lo,
                    hi,
                    new_val,
                )
                clamped[key] = new_val
    return clamped


# Allowlist regex for override_tensor values: e.g. "q4_0", "q8_0:0,1", "blk.0.attn=CPU"
_OVERRIDE_TENSOR_RE = re.compile(r"^[a-zA-Z0-9_.,:=/+-]+$")

# Allowlist regex for tensor_split values: comma-separated numbers, e.g. "50,50"
_TENSOR_SPLIT_RE = re.compile(r"^[0-9.,]+$")

# Enum allowlists for validated CLI arguments
_VALID_KV_CACHE_TYPES: frozenset[str] = frozenset(
    {"f16", "f32", "q8_0", "q5_0", "q5_1", "q4_0", "q4_1", "iq4_nl"}
)
_VALID_NUMA_MODES: frozenset[str] = frozenset({"", "distribute", "isolate", "numactl"})


def _add_base_args(ctx: AppContext, engine_config: EngineConfig) -> list[str]:
    """Return the initial cmd list: server path, model, port, host, core flags."""
    engine_config = _clamp_numeric(engine_config)
    args = [
        str(ctx.server_path),
        "-m",
        str(ctx.model_path),
        "--port",
        str(ctx.port),
        "--host",
        BIND_HOST,
        "-ngl",
        str(engine_config.get("n_gpu_layers", 99)),
        "-c",
        str(engine_config.get("context", 4096)),
        "--parallel",
        str(engine_config.get("parallel", 1)),
    ]
    if ctx.no_jinja:
        args.extend(["--no-jinja", "--chat-template", "chatml"])
    elif ctx.chat_template_path and ctx.chat_template_path.is_file():
        args.extend(["--chat-template-file", str(ctx.chat_template_path)])
    if engine_config.get("warmup") is not True:
        args.append("--no-warmup")
    if engine_config.get("cache_prompt") is False:
        args.append("--no-cache-prompt")
    if engine_config.get("fit") is False:
        args.append("--fit=off")
    return args


def _add_numeric_flag_pairs(ctx: AppContext, engine_config: EngineConfig) -> list[str]:
    """Return args for batch_size, ubatch_size, threads, threads_batch, n_cpu_moe,
    expert_used_count override, and poll/prio toggle pairs.

    Note: callers must ensure *engine_config* has already been clamped via
    ``_clamp_numeric`` (done once in ``_add_base_args``).
    """
    args: list[str] = []
    flag_pairs = [
        ("batch_size", "-b"),
        ("ubatch_size", "--ubatch-size"),
        ("threads", "-t"),
        ("threads_batch", "-tb"),
        ("n_cpu_moe", "--n-cpu-moe"),
    ]
    for key, flag in flag_pairs:
        if key in engine_config:
            args.extend([flag, str(engine_config[key])])

    if "expert_used_count" in engine_config and ctx.expert_override_key:
        if engine_config["expert_used_count"] != ctx.default_experts:
            args.extend(
                [
                    "--override-kv",
                    f"{ctx.expert_override_key}=int:{engine_config['expert_used_count']}",
                ]
            )

    toggle_pairs = [
        ("poll", "--poll"),
        ("poll_batch", "--poll-batch"),
        ("prio", "--prio"),
        ("prio_batch", "--prio-batch"),
    ]
    for key, flag in toggle_pairs:
        if key in engine_config:
            args.extend([flag, str(engine_config[key])])
    return args


def _add_kv_cache_args(  # noqa: C901, PLR0912
    engine_config: EngineConfig, *, ctx: AppContext | None = None
) -> list[str]:
    """Return args for kv_cache_type, cache_type_k/v, flash_attn, n_predict, temp,
    model_draft, and cache_reuse.

    Args:
        ctx: Optional application context.  When provided, ``model_draft``
            is validated to reside under ``ctx.model_path.parent``.
    """
    args: list[str] = []
    if "kv_cache_type" in engine_config:
        kv_val = str(engine_config["kv_cache_type"])
        if kv_val in _VALID_KV_CACHE_TYPES:
            args.extend(["--cache-type-k", kv_val])
            args.extend(["--cache-type-v", kv_val])
        else:
            logger.warning("Rejected invalid kv_cache_type value: %r", kv_val)
    if "cache_type_k" in engine_config:
        ctk = str(engine_config["cache_type_k"])
        if ctk in _VALID_KV_CACHE_TYPES:
            args.extend(["--cache-type-k", ctk])
        else:
            logger.warning("Rejected invalid cache_type_k value: %r", ctk)
    if "cache_type_v" in engine_config:
        ctv = str(engine_config["cache_type_v"])
        if ctv in _VALID_KV_CACHE_TYPES:
            args.extend(["--cache-type-v", ctv])
        else:
            logger.warning("Rejected invalid cache_type_v value: %r", ctv)
    if engine_config.get("flash_attn") in ("on", True, "1", 1):
        args.extend(["--flash-attn", "1"])
    elif "flash_attn" in engine_config:
        args.extend(["--flash-attn", "0"])
    if "n_predict" in engine_config:
        args.extend(["--n-predict", str(engine_config["n_predict"])])
    if "temp" in engine_config:
        args.extend(["--temp", str(engine_config["temp"])])
    if "model_draft" in engine_config:
        draft_path = Path(str(engine_config["model_draft"])).resolve()
        if ctx is not None:
            allowed_dir = ctx.model_path.parent.resolve()
            starts_under = str(draft_path).startswith(
                str(allowed_dir) + os.sep
            )
            if not starts_under and draft_path.parent != allowed_dir:
                logger.warning(
                    "Rejected model_draft — path %s is not under model directory %s",
                    draft_path,
                    allowed_dir,
                )
            elif draft_path.is_file():
                args.extend(["--model-draft", str(draft_path)])
            else:
                logger.warning("Rejected model_draft — file not found: %s", draft_path)
        elif draft_path.is_file():
            args.extend(["--model-draft", str(draft_path)])
        else:
            logger.warning("Rejected model_draft — file not found: %s", draft_path)
    if "cache_reuse" in engine_config:
        args.extend(["--cache-reuse", str(engine_config["cache_reuse"])])
    return args


def _add_spec_args(engine_config: EngineConfig) -> list[str]:
    """Return args for speculation and cpu placement params."""
    args: list[str] = []
    spec_pairs = [
        ("spec_type", "--spec-type"),
        ("spec_ngram_n", "--spec-ngram-size-n"),
        ("spec_ngram_m", "--spec-ngram-size-m"),
        ("spec_ngram_min_hits", "--spec-ngram-min-hits"),
        ("draft_max", "--draft"),
        ("draft_min", "--draft-min"),
        ("draft_p_min", "--draft-p-min"),
    ]
    for key, flag in spec_pairs:
        if key in engine_config:
            args.extend([flag, str(engine_config[key])])
    if "cpu_strict" in engine_config:
        args.extend(["--cpu-strict", str(engine_config["cpu_strict"])])
    if "cpu_strict_batch" in engine_config:
        args.extend(["--cpu-strict-batch", str(engine_config["cpu_strict_batch"])])
    return args


def _add_bool_flags(engine_config: EngineConfig) -> list[str]:
    """Return args for boolean on/off flags."""
    args: list[str] = []
    bool_flags = [
        ("swa_full", "--swa-full", True),
        ("repack", "--no-repack", False),
        ("op_offload", "--no-op-offload", False),
        ("kv_unified", "--kv-unified", True),
        ("mlock", "--mlock", True),
        ("no_mmap", "--no-mmap", True),
        ("kv_offload", "--no-kv-offload", False),
        ("no_host", "--no-host", True),
        ("direct_io", "--direct-io", True),
        ("cont_batching", "--no-cont-batching", False),
        ("backend_sampling", "--backend-sampling", True),
        ("context_shift", "--no-context-shift", False),
    ]
    for key, flag, trigger_val in bool_flags:
        if engine_config.get(key) is trigger_val:
            args.append(flag)
    return args


def _add_extended_args(engine_config: EngineConfig) -> list[str]:  # noqa: C901, PLR0912
    """Return args for ctx_checkpoints, cache_ram, numa, tensor_split, etc."""
    args: list[str] = []
    ext_pairs = [
        ("ctx_checkpoints", "--ctx-checkpoints"),
        ("checkpoint_every_n", "--checkpoint-every-n-tokens"),
        ("cache_ram", "--cache-ram"),
        ("threads_http", "--threads-http"),
    ]
    for key, flag in ext_pairs:
        if key in engine_config:
            args.extend([flag, str(engine_config[key])])
    if engine_config.get("lookup_cache_dynamic"):
        args.extend(
            ["--lookup-cache-dynamic", str(engine_config["lookup_cache_dynamic"])]
        )
    if "numa" in engine_config:
        numa_val = str(engine_config["numa"])
        if numa_val in _VALID_NUMA_MODES:
            if numa_val:  # don't pass empty string as flag
                args.extend(["--numa", numa_val])
        else:
            logger.warning("Rejected invalid numa value: %r", numa_val)
    if engine_config.get("cpu_moe"):
        args.append("--cpu-moe")
    if engine_config.get("override_tensor"):
        ot_val = engine_config["override_tensor"]
        if _OVERRIDE_TENSOR_RE.match(ot_val):
            args.extend(["-ot", ot_val])
        else:
            logger.warning("Rejected invalid override_tensor value: %r", ot_val)
    if "tensor_split" in engine_config:
        ts_val = engine_config["tensor_split"]
        if _TENSOR_SPLIT_RE.match(str(ts_val)):
            args.extend(["--tensor-split", str(ts_val)])
        else:
            logger.warning("Rejected invalid tensor_split value: %r", ts_val)
    return args
