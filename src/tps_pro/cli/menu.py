"""Interactive terminal menus — pure UI, all logic via services.py."""

from __future__ import annotations

import os
import subprocess
from typing import Any

from ..constants import DEFAULT_CONTEXT_SIZE
from ..state import ctx
from . import services
from ._helpers import pause as _pause
from ._helpers import safe_input as _input
from ._helpers import show_error as _show_error

_MIN_CONTEXT = 512


def clear_screen() -> None:
    """Clear terminal."""
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "cls"], shell=False, timeout=5)
    else:
        subprocess.run(["clear"], shell=False, timeout=5)


# ── Header ───────────────────────────────────────────────────────


_cached_info = None


def invalidate_header_cache() -> None:
    """Call after model switch, config change, etc."""
    global _cached_info
    _cached_info = None


def print_header() -> None:
    """Print compact system info header. Caches expensive calls."""
    global _cached_info

    if _cached_info is None:
        try:
            _cached_info = services.get_system_info(ctx)
        except Exception:
            print("  [!] Could not load system info")
            return

    info = _cached_info

    # Only re-check cheap fields that change between menu draws
    preset = services.get_config("preset", "normal")
    ctx_val = services.get_config("target_context")
    ctx_label = f"{ctx_val:,}" if ctx_val else "auto"

    print("=" * 60)
    print("  llama-server Parameter Optimizer")
    print("  GP-Bayesian Coordinate Descent")
    print("=" * 60)
    print()
    print(f"  Python:  {info.python_version}")
    print(f"  Server:  {info.server_url}")
    print(f"  Model:   {info.model_name}")

    if info.arch_detail:
        print(f"  Arch:    {info.arch_type} ({info.arch_detail})")
    else:
        print(f"  Arch:    {info.arch_type}")

    print(f"  GPU:     {info.gpu_layers} layers offloaded")
    for gpu in info.gpus:
        name = gpu.get("name", "Unknown")
        free_gb = gpu.get("vram_free_gb", 0)
        total_gb = gpu.get("vram_total_gb", 0)
        idx = gpu.get("index", 0)
        if total_gb > 0:
            print(f"  GPU {idx}:  {name} ({free_gb:.1f}/{total_gb:.1f}GB free)")
        else:
            print(f"  GPU {idx}:  {name}")

    print(f"  CPU:     {info.cpu_threads} threads (auto-detected)")
    print(f"  Size:    {info.model_size_gb:.1f} GB ({info.model_size_class})")
    print(f"  Context: {ctx_label}")
    print(f"  Preset:  {preset}")
    print(f"  Results: {info.results_dir}")
    print()
    print("=" * 60)


# ── Main Menu ────────────────────────────────────────────────────


def print_menu() -> None:
    """Print main menu options."""
    preset = services.get_config("preset", "normal")
    ctx_val = services.get_config("target_context")
    ctx_label = f"{ctx_val:,}" if ctx_val else "auto"

    print()
    print(f"  [o]   Optimize (full pipeline)  [{preset}]")
    print("  [v]   View results")
    print("  [a]   Advanced...")
    print("  [m]   Switch model")
    print(f"  [p]   Preset ({preset})")
    print(f"  [c]   Context ({ctx_label})")
    print("  [q]   Quit")
    print()


# ── Advanced Menu ────────────────────────────────────────────────


def _print_advanced_menu() -> set[str]:
    """Print advanced menu, return valid keys."""
    is_moe = ctx.is_moe

    print("  " + "=" * 50)
    print("  Advanced")
    print("  " + "=" * 50)

    print("  Phases:")
    print("    [g]   GPU offload        [kv]  KV + context sweep")
    print("    [ce]  Core engine        [sp]  Speculation")
    print("    [wl]  Workload sim       [s]   Quality/Sampling")

    valid = {"g", "ce", "sp", "kv", "wl", "s", "html", "r", "d", "b", ""}

    if is_moe:
        print("  MoE Phases:")
        print("    [moe] MoE sweep          [ex]  Expert count")
        valid.update({"moe", "ex"})

    print("  Other:")
    print("    [html] HTML report       [r]   Reset DB")
    print("    [d]   Dashboard          [cfg] Pipeline config")
    print("  [b] <- Back")
    print()

    valid.add("cfg")
    return valid


def advanced_menu() -> None:
    """Advanced submenu loop."""
    while True:
        clear_screen()
        valid = _print_advanced_menu()

        choice = _input("  > ").strip().lower()
        if not choice or choice == "b":
            return

        if choice not in valid:
            print("  Invalid choice.")
            _pause()
            continue

        try:
            _dispatch_advanced(choice)
        except Exception as e:
            _show_error(str(e))


def _dispatch_advanced(choice: str) -> None:
    """Handle an advanced menu choice."""
    phase_map = {
        "g": ("phase_gpu_offload", False, "GPU Offload"),
        "kv": ("phase_kv_context_sweep", False, "KV + Context Sweep"),
        "ce": ("phase_core_engine", True, "Core Engine"),
        "sp": ("phase_speculation", True, "Speculation"),
        "wl": ("phase_workload_sim", False, "Workload Sim"),
        "moe": ("phase_moe_sweep", False, "MoE Sweep"),
        "ex": ("phase_experts", False, "Expert Count"),
        "s": ("phase_quality", True, "Quality"),
    }

    if choice in phase_map:
        func_name, needs_trials, label = phase_map[choice]
        _run_single_phase(func_name, needs_trials, label)
    elif choice == "r":
        _do_reset_db()
    elif choice == "html":
        _do_html_report()
    elif choice == "d":
        _do_dashboard()
    elif choice == "cfg":
        _do_config_editor()


def _do_config_editor() -> None:
    """Open the pipeline config editor."""
    from .config_editor import config_editor_menu

    config_editor_menu()


def _run_single_phase(func_name: str, needs_trials: bool, label: str) -> None:
    """Run a single optimization phase."""
    from ..phases import (
        phase_core_engine,
        phase_experts,
        phase_gpu_offload,
        phase_kv_context_sweep,
        phase_moe_sweep,
        phase_quality,
        phase_speculation,
        phase_workload_sim,
    )

    funcs: dict[str, Any] = {
        "phase_gpu_offload": phase_gpu_offload,
        "phase_core_engine": phase_core_engine,
        "phase_kv_context_sweep": phase_kv_context_sweep,
        "phase_speculation": phase_speculation,
        "phase_workload_sim": phase_workload_sim,
        "phase_moe_sweep": phase_moe_sweep,
        "phase_experts": phase_experts,
        "phase_quality": phase_quality,
    }

    n_trials = None
    if needs_trials:
        preset = services.get_config("preset", "normal")
        default = services.get_phase_trial_default(func_name, preset)
        raw = _input(f"  {label} trials [{default}]: ").strip()
        n_trials = int(raw) if raw.isdigit() else default

    base_config = services.build_phase_base_config(ctx, func_name)

    clear_screen()
    print(f"  Starting {label}...")

    fn = funcs[func_name]
    if needs_trials and n_trials:
        fn(ctx, n_trials=n_trials, base_config=base_config)
    elif func_name == "phase_gpu_offload":
        fn(ctx)
    elif func_name == "phase_kv_context_sweep":
        fn(ctx, force=True, base_config=base_config)
    elif func_name == "phase_moe_sweep":
        fn(ctx, force=True)
    elif func_name == "phase_experts":
        fn(ctx)
    else:
        fn(ctx, base_config=base_config)

    print(f"\n  {label} complete.")
    _pause()


def _do_reset_db() -> None:
    """Reset the Optuna database and results."""
    confirm = _input("  Delete all saved trial progress and results? [y/N]: ")
    if confirm.strip().lower() != "y":
        print("  Cancelled.")
        _pause()
        return

    try:
        ok = services.reset_database(ctx)
        if ok:
            print("  DB and results deleted. All phases will start fresh.")
        else:
            print("  [!] Could not fully delete DB.")
    except services.DatabaseResetError as e:
        _show_error(f"Reset failed: {e}")
        return
    _pause()


def _do_html_report() -> None:
    """Generate HTML report."""
    try:
        path = services.generate_html_report(ctx)
        if path:
            print(f"  Report saved: {path}")
        else:
            print("  No results to report.")
    except Exception as e:
        _show_error(f"Report failed: {e}")
    _pause()


def _do_dashboard() -> None:
    """Launch Optuna dashboard."""
    try:
        from .dashboard import launch_dashboard

        launch_dashboard(ctx)
        print("  Dashboard launched.")
    except Exception as e:
        _show_error(f"Dashboard failed: {e}")
    _pause()


# ── Context Menu ─────────────────────────────────────────────────


def context_menu() -> None:
    """Set context size."""
    clear_screen()
    print("  Context size:")
    print(f"    [auto]  Auto ({DEFAULT_CONTEXT_SIZE} -> sweep)")
    print(f"    [N]     Fixed (e.g. {DEFAULT_CONTEXT_SIZE}, 8192)")
    print("    [b]     Back")

    choice = _input("  > ").strip().lower()
    if not choice or choice == "b":
        return

    try:
        services.set_context_size(ctx, ctx.config, choice)
    except services.ConfigValidationError as e:
        print(f"  {e}")
        _pause()
        return

    if choice == "auto":
        print("  Context set to auto.")
    else:
        print(f"  Context set to {int(choice):,}.")
    _pause()
