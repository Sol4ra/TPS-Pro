"""KV Sweep boot scanning — binary-search for max bootable context per KV type.

Extracted from kv_context_sweep.py for module size management.
"""

from __future__ import annotations

import logging
from typing import Any

from ..constants import DEFAULT_CONTEXT_SIZE
from ..engine import (
    boot_server_with_jinja_recovery,
    kill_server,
)
from ..state import AppContext

logger = logging.getLogger(__name__)

__all__ = [
    "discover_bootable_contexts",
    "generate_test_points",
    "get_model_metadata",
    "get_model_max_context",
]

# KV types to sweep, ordered from highest to lowest fidelity.
KV_TYPES = ["f16", "q8_0", "q4_0"]

# Binary-search convergence threshold (tokens).
_BISECT_THRESHOLD = 2048

# Round context sizes down to this alignment.
_CTX_ALIGNMENT = 1024


def _round_down(value: int, alignment: int) -> int:
    """Round *value* down to the nearest multiple of *alignment*."""
    return (value // alignment) * alignment


def get_model_metadata(ctx: AppContext) -> dict[str, Any]:
    """Read model metadata from GGUF for context limits and VRAM estimation."""
    from ..models import read_gguf_metadata

    try:
        return read_gguf_metadata(str(ctx.model_path))
    except (OSError, ValueError, KeyError) as e:
        logger.debug("Could not read GGUF metadata: %s", e)
        return {}


def get_model_max_context(meta: dict) -> int:
    """Extract model's trained max context from GGUF metadata."""
    for k, v in meta.items():
        if k.endswith(".context_length") and isinstance(v, int):
            return v
    return 0


def _try_boot_at_context(
    ctx: AppContext, config: dict[str, Any], ctx_size: int,
) -> str:
    """Attempt to boot the server at a given context size.

    Returns 'ok' if boot succeeds, 'oom' if OOM detected, 'fail' otherwise.
    Always kills the server after the attempt.
    """
    boot_config = {**config, "context": ctx_size}
    _proc, status = boot_server_with_jinja_recovery(ctx, boot_config)
    kill_server(ctx)
    if status == "ok":
        return "ok"
    if status == "oom":
        return "oom"
    return "fail"


def _find_max_bootable(  # noqa: C901, PLR0912
    ctx: AppContext,
    kv_type: str,
    model_max_ctx: int = 0,
    safe_max_ctx: int = 0,
    base_config: dict | None = None,
) -> int | None:
    """Binary-search for the maximum context that boots without OOM.

    Returns the max bootable context (rounded down to _CTX_ALIGNMENT),
    or None if even 4096 fails.
    """
    start_ctx = DEFAULT_CONTEXT_SIZE
    source = base_config if base_config is not None else ctx.naked_engine
    config = {**source, "kv_cache_type": kv_type, "flash_attn": "on"}

    if model_max_ctx > 0 and safe_max_ctx > 0:
        model_max_ctx = min(model_max_ctx, safe_max_ctx)
    elif safe_max_ctx > 0:
        model_max_ctx = safe_max_ctx

    if model_max_ctx > 0:
        logger.info("  Model context limit: %dk", model_max_ctx // 1024)
    if safe_max_ctx > 0:
        logger.info(
            "  VRAM estimate: ~%dk (advisory, actual boot test is authoritative)",
            safe_max_ctx // 1024,
        )

    if _try_boot_at_context(ctx, config, start_ctx) != "ok":
        return None

    hard_cap = model_max_ctx if model_max_ctx > 0 else 0
    last_pass = start_ctx
    test_ctx = start_ctx * 2
    while True:
        if hard_cap > 0 and test_ctx > hard_cap:
            test_ctx = hard_cap
            if test_ctx <= last_pass:
                return _round_down(last_pass, _CTX_ALIGNMENT)
            if _try_boot_at_context(ctx, config, test_ctx) == "ok":
                return _round_down(test_ctx, _CTX_ALIGNMENT)
            break

        if _try_boot_at_context(ctx, config, test_ctx) == "ok":
            last_pass = test_ctx
            test_ctx *= 2
        else:
            break

    first_fail = test_ctx

    lo, hi = last_pass, first_fail
    while hi - lo > _BISECT_THRESHOLD:
        mid = (lo + hi) // 2
        if _try_boot_at_context(ctx, config, mid) == "ok":
            lo = mid
        else:
            hi = mid

    return _round_down(lo, _CTX_ALIGNMENT)


def generate_test_points(max_bootable: int) -> list[int]:
    """Generate doubling test points from DEFAULT_CONTEXT_SIZE up to max_bootable."""
    points: list[int] = []
    ctx_size = DEFAULT_CONTEXT_SIZE
    while ctx_size <= max_bootable:
        points.append(ctx_size)
        ctx_size *= 2
    if points and points[-1] < max_bootable:
        rounded = _round_down(max_bootable, _CTX_ALIGNMENT)
        if rounded > points[-1]:
            points.append(rounded)
    return points


def discover_bootable_contexts(
    ctx: AppContext,
    model_max_ctx: int,
    base_config: dict,
) -> tuple[dict[str, int | None], set[int]]:
    """Scan each KV type to find max bootable context sizes.

    Returns (kv_bootable map, union of all test points).
    """
    kv_bootable: dict[str, int | None] = {}
    all_test_points: set[int] = set()

    for kv_type in KV_TYPES:
        logger.info("  Scanning %s context limits...", kv_type)
        max_bootable = _find_max_bootable(
            ctx,
            kv_type,
            model_max_ctx,
            0,
            base_config=base_config,
        )
        kv_bootable[kv_type] = max_bootable
        if max_bootable is None:
            logger.warning(
                "  KV Sweep: %s OOM at 4096 -- skipping",
                kv_type,
            )
        else:
            logger.info(
                "  KV Sweep: %s max bootable: %d",
                kv_type,
                max_bootable,
            )
            for pt in generate_test_points(max_bootable):
                all_test_points.add(pt)

    return kv_bootable, all_test_points
