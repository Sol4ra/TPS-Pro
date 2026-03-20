"""Hardware detection and first-time setup helpers.

Extracted from main.py. main.py delegates to these functions
so it only handles: parse args, initialize, dispatch.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from typing import Any

from ..constants import DEFAULT_MAX_GPU_LAYERS
from ..hardware import detect_gpus
from ..models import classify_model, detect_model_layers
from ..state import (
    _DEFAULTS,
    _detect_numa_nodes,
    ctx,
    get_config,
    rebuild_ctx,
    update_naked_engine,
)

logger = logging.getLogger(__name__)


def run_first_time_setup() -> None:
    """Run first-time setup wizard and rebuild the application context.

    Merges wizard output with defaults, auto-detects hardware parameters,
    and rebuilds the global ctx.
    """
    from .wizard import first_run_setup

    new_config = first_run_setup()
    merged: dict[str, Any] = copy.deepcopy(_DEFAULTS)
    for k, v in new_config.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k].update(v)
        else:
            merged[k] = v
    # Auto-detect hardware (normally done in _load_config, skipped on first-run path)
    hw = merged.setdefault("hardware", {})
    if hw.get("max_threads") is None:
        hw["max_threads"] = os.cpu_count() or 16
    if "numa_nodes" not in hw:
        hw["numa_nodes"] = _detect_numa_nodes()
    if hw.get("moe_sweep_max") is None:
        hw["moe_sweep_max"] = min(hw["max_threads"] * 2, 40)
    if hw.get("moe_sweep_center") is None:
        hw["moe_sweep_center"] = hw["moe_sweep_max"] // 2
    if hw.get("max_gpu_layers") is None:
        detected = detect_model_layers(merged.get("model", ""))
        hw["max_gpu_layers"] = detected or DEFAULT_MAX_GPU_LAYERS
    rebuild_ctx(merged)


def _confirm_kill_processes(descriptions: list[str]) -> bool:
    """CLI confirmation callback for kill_competing_processes.

    Shows the user which processes will be killed and asks for confirmation.
    Returns True to proceed, False to abort.
    """
    print("  The following GPU processes will be killed to free VRAM:")
    for desc in descriptions:
        print(f"    - {desc}")
    answer = input("  Proceed? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def detect_hardware_and_model(kill_competing_processes) -> None:
    """Detect GPUs, classify model, and hydrate GPU state from past runs.

    Args:
        kill_competing_processes: Callable to kill competing GPU processes.
    """
    from ..search import load_phase_results

    # Kill competing GPU processes to free VRAM (opt-in only)
    if get_config("kill_competing"):
        callback = None
        if get_config("interactive") and sys.stdin.isatty():
            callback = _confirm_kill_processes
        kill_competing_processes(confirm_callback=callback)
    else:
        pass  # Skip VRAM pre-check to avoid double GPU detection

    # Detect GPUs (once)
    gpus = detect_gpus()
    if gpus:
        for g in gpus:
            logger.debug(
                "GPU %s: %s (%s GB, %s GB free)",
                g["index"],
                g["name"],
                g["vram_total_gb"],
                g["vram_free_gb"],
            )
    else:
        logger.debug("No GPUs detected (pynvml not installed or no NVIDIA GPU)")

    # Classify model and set size-based timeouts
    model_class, model_size = classify_model(str(ctx.model_path))
    ctx.model_size_class = model_class
    ctx.model_size_gb = model_size
    logger.debug(
        "Model: %s (%.1f GB, class: %s)", ctx.model_path.name, model_size, model_class
    )

    # Hydrate GPU state from past runs so direct-menu launches don't OOM
    gpu_data = load_phase_results(ctx, "gpu")
    if gpu_data and "best_ngl" in gpu_data:
        ctx.default_gpu_layers = gpu_data["best_ngl"]
        update_naked_engine(ctx, n_gpu_layers=ctx.default_gpu_layers)
