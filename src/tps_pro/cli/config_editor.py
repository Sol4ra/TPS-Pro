"""TUI editor for pipeline configuration.

Provides an interactive menu for viewing and modifying PipelineConfig:
global flags, phase ordering, enabling/disabling phases, editing phase
details, and saving/resetting the configuration.
"""

from __future__ import annotations

import dataclasses
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..pipeline_config import PhaseConfig, PipelineConfig
from ..state import ctx
from ._helpers import safe_input as _input

# Local wrappers so tests can patch config_editor._input and intercept
# all input calls (including those inside _pause/_show_error).


def _pause() -> None:
    _input("\n  Press Enter to continue...")


def _show_error(msg: str) -> None:
    print(f"\n  [!] {msg}")
    _pause()


# ── Constants ─────────────────────────────────────────────────────

_MAX_DISPLAY_PARAMS = 3


# ── Display ──────────────────────────────────────────────────────


def _format_phase_line(idx: int, phase: PhaseConfig) -> str:
    """Format a single phase line for display."""
    mark = "\u2713" if phase.enabled else "\u2717"
    label = f"  [{idx + 1}] {mark} {phase.display_name}"

    details: list[str] = []
    if not phase.enabled:
        details.append("disabled")
    else:
        if phase.range:
            details.append(
                f"range: {phase.range[0]}-{phase.range[1]}, step: {phase.step}"
            )
        if phase.test_flags:
            details.append(f"flags: {', '.join(phase.test_flags)}")
        if phase.trials is not None:
            details.append(f"{phase.trials} trials")
        if phase.search_params:
            params_str = ", ".join(phase.search_params[:_MAX_DISPLAY_PARAMS])
            if len(phase.search_params) > _MAX_DISPLAY_PARAMS:
                params_str += "..."
            details.append(f"searching: {params_str}")
        if phase.kv_types:
            details.append(f"kv: {', '.join(phase.kv_types)}")

    if details:
        return f"  {label:<30s}({', '.join(details)})"
    return f"  {label}"


def _print_config_view(config: PipelineConfig) -> None:
    """Print the full configuration overview."""
    print("=" * 60)
    print("  Pipeline Configuration")
    print("=" * 60)

    # Global flags
    if config.global_flags:
        flags_str = ", ".join(f"{k}={v}" for k, v in config.global_flags.items())
        print(f"  Global Flags: {flags_str}")
    else:
        print("  Global Flags: (none)")

    print()
    print("  Pipeline Order:")
    for i, phase in enumerate(config.phases):
        print(_format_phase_line(i, phase))

    print()
    print("  [g] Set global flag    [r] Remove global flag")
    print("  [t] Toggle phase       [m] Move phase")
    print("  [e] Edit phase         [d] Reset defaults")
    print("  [s] Save               [b] Back")
    print()


# ── Actions ──────────────────────────────────────────────────────


def _parse_flag_value(raw: str) -> Any:
    """Parse a flag value string into an appropriate Python type."""
    lower = raw.lower()
    if lower in ("true", "on", "yes", "1"):
        return True
    if lower in ("false", "off", "no", "0"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _do_set_global_flag(config: PipelineConfig) -> PipelineConfig:
    """Set a global flag (returns new config)."""
    print("\n  Set global flag — e.g. 'no_mmap true', 'flash_attn on'")
    raw = _input("  > ").strip()
    if not raw:
        return config

    parts = raw.split(None, 1)
    if len(parts) < 2:  # noqa: PLR2004
        _show_error("Expected: <flag_name> <value>")
        return config

    key, val_str = parts[0], parts[1]
    value = _parse_flag_value(val_str)

    new_flags = dict(config.global_flags)
    new_flags[key] = value
    new_config = PipelineConfig(
        global_flags=new_flags,
        phases=deepcopy(config.phases),
    )
    print(f"\n  Set {key}={value}")
    print(f"  Note: '{key}' will be removed from phase search pools.")
    _pause()
    return new_config


def _do_remove_global_flag(config: PipelineConfig) -> PipelineConfig:
    """Remove a global flag (returns new config)."""
    if not config.global_flags:
        _show_error("No global flags to remove.")
        return config

    print("\n  Current global flags:")
    for k, v in config.global_flags.items():
        print(f"    {k}={v}")
    print()

    key = _input("  Flag to remove: ").strip()
    if not key:
        return config

    if key not in config.global_flags:
        _show_error(f"Flag '{key}' not found.")
        return config

    new_flags = {k: v for k, v in config.global_flags.items() if k != key}
    new_config = PipelineConfig(
        global_flags=new_flags,
        phases=deepcopy(config.phases),
    )
    print(f"\n  Removed '{key}'.")
    _pause()
    return new_config


def _do_toggle_phase(config: PipelineConfig) -> PipelineConfig:
    """Toggle a phase on/off (returns new config)."""
    print("\n  Toggle phase — enter phase number:")
    for i, p in enumerate(config.phases):
        mark = "\u2713" if p.enabled else "\u2717"
        print(f"    [{i + 1}] {mark} {p.display_name}")

    raw = _input("  > ").strip()
    if not raw or not raw.isdigit():
        return config

    idx = int(raw) - 1
    if idx < 0 or idx >= len(config.phases):
        _show_error(f"Invalid phase number. Choose 1-{len(config.phases)}.")
        return config

    new_phases = deepcopy(config.phases)
    new_phases[idx] = dataclasses.replace(
        new_phases[idx],
        enabled=not new_phases[idx].enabled,
    )
    state = "enabled" if new_phases[idx].enabled else "disabled"
    new_config = PipelineConfig(
        global_flags=dict(config.global_flags),
        phases=new_phases,
    )
    print(f"\n  {config.phases[idx].display_name} is now {state}.")
    _pause()
    return new_config


def _do_move_phase(config: PipelineConfig) -> PipelineConfig:
    """Move a phase up or down (returns new config)."""
    print("\n  Move phase — enter phase number:")
    for i, p in enumerate(config.phases):
        print(f"    [{i + 1}] {p.display_name}")

    raw = _input("  Phase number: ").strip()
    if not raw or not raw.isdigit():
        return config

    idx = int(raw) - 1
    if idx < 0 or idx >= len(config.phases):
        _show_error(f"Invalid phase number. Choose 1-{len(config.phases)}.")
        return config

    direction = _input("  Direction [u]p / [d]own: ").strip().lower()
    if direction not in ("u", "d", "up", "down"):
        _show_error("Enter 'u' for up or 'd' for down.")
        return config

    new_phases = list(config.phases)
    if direction in ("u", "up"):
        if idx == 0:
            _show_error("Already at the top.")
            return config
        new_phases[idx - 1], new_phases[idx] = new_phases[idx], new_phases[idx - 1]
    else:
        if idx == len(new_phases) - 1:
            _show_error("Already at the bottom.")
            return config
        new_phases[idx], new_phases[idx + 1] = new_phases[idx + 1], new_phases[idx]

    new_config = PipelineConfig(
        global_flags=dict(config.global_flags),
        phases=new_phases,
    )
    direction_label = "up" if direction in ("u", "up") else "down"
    print(f"\n  Moved {config.phases[idx].display_name} {direction_label}.")
    _pause()
    return new_config


# ── Edit-phase command handlers ──────────────────────────────────


def _handle_trials_cmd(phase: PhaseConfig, parts: list[str]) -> PhaseConfig:
    """Handle the 'trials <N>' edit command."""
    if len(parts) >= 2 and parts[1].isdigit():  # noqa: PLR2004
        new_phase = dataclasses.replace(phase, trials=int(parts[1]))
        print(f"    Trials set to {parts[1]}")
        return new_phase
    return phase


def _handle_param_cmd(phase: PhaseConfig, parts: list[str], add: bool) -> PhaseConfig:
    """Handle '+param <name>' or '-param <name>' edit commands."""
    if len(parts) < 2:  # noqa: PLR2004
        return phase
    name = parts[1]
    if add:
        if name not in phase.search_params:
            new_params = list(phase.search_params) + [name]
            new_phase = dataclasses.replace(phase, search_params=new_params)
            print(f"    Added search param: {name}")
            return new_phase
        print(f"    '{name}' already in search params")
    else:
        if name in phase.search_params:
            new_params = [param for param in phase.search_params if param != name]
            new_phase = dataclasses.replace(phase, search_params=new_params)
            print(f"    Removed search param: {name}")
            return new_phase
        print(f"    '{name}' not in search params")
    return phase


def _handle_flag_cmd(phase: PhaseConfig, parts: list[str], add: bool) -> PhaseConfig:
    """Handle '+flag <name>' or '-flag <name>' edit commands."""
    if len(parts) < 2:  # noqa: PLR2004
        return phase
    name = parts[1]
    if add:
        if name not in phase.test_flags:
            new_flags = list(phase.test_flags) + [name]
            new_phase = dataclasses.replace(phase, test_flags=new_flags)
            print(f"    Added test flag: {name}")
            return new_phase
        print(f"    '{name}' already in test flags")
    else:
        if name in phase.test_flags:
            new_flags = [flag for flag in phase.test_flags if flag != name]
            new_phase = dataclasses.replace(phase, test_flags=new_flags)
            print(f"    Removed test flag: {name}")
            return new_phase
        print(f"    '{name}' not in test flags")
    return phase


def _handle_lock_cmd(phase: PhaseConfig, parts: list[str]) -> PhaseConfig:
    """Handle 'lock <key> <value>' edit command."""
    if len(parts) < 3:  # noqa: PLR2004
        return phase
    key, val_str = parts[1], parts[2]
    value = _parse_flag_value(val_str)
    new_lock = dict(phase.lock)
    new_lock[key] = value
    new_phase = dataclasses.replace(phase, lock=new_lock)
    print(f"    Locked {key}={value}")
    return new_phase


def _handle_unlock_cmd(phase: PhaseConfig, parts: list[str]) -> PhaseConfig:
    """Handle 'unlock <key>' edit command."""
    if len(parts) < 2:  # noqa: PLR2004
        return phase
    key = parts[1]
    if key in phase.lock:
        new_lock = {k: v for k, v in phase.lock.items() if k != key}
        new_phase = dataclasses.replace(phase, lock=new_lock)
        print(f"    Unlocked {key}")
        return new_phase
    print(f"    '{key}' not locked")
    return phase


# Dispatch table for edit-phase commands
_EDIT_DISPATCH: dict[str, Any] = {
    "trials": lambda phase, parts: _handle_trials_cmd(phase, parts),
    "+param": lambda phase, parts: _handle_param_cmd(phase, parts, add=True),
    "-param": lambda phase, parts: _handle_param_cmd(phase, parts, add=False),
    "+flag": lambda phase, parts: _handle_flag_cmd(phase, parts, add=True),
    "-flag": lambda phase, parts: _handle_flag_cmd(phase, parts, add=False),
    "lock": lambda phase, parts: _handle_lock_cmd(phase, parts),
    "unlock": lambda phase, parts: _handle_unlock_cmd(phase, parts),
}


def _do_edit_phase(config: PipelineConfig) -> PipelineConfig:
    """Edit a phase's details (returns new config)."""
    print("\n  Edit phase — enter phase number:")
    for i, p in enumerate(config.phases):
        print(f"    [{i + 1}] {p.display_name}")

    raw = _input("  Phase number: ").strip()
    if not raw or not raw.isdigit():
        return config

    idx = int(raw) - 1
    if idx < 0 or idx >= len(config.phases):
        _show_error(f"Invalid phase number. Choose 1-{len(config.phases)}.")
        return config

    phase = config.phases[idx]
    print(f"\n  Editing: {phase.display_name}")
    print(f"    Trials:        {phase.trials if phase.trials is not None else '(n/a)'}")
    sp = ", ".join(phase.search_params) if phase.search_params else "(none)"
    tf = ", ".join(phase.test_flags) if phase.test_flags else "(none)"
    print(f"    Search params: {sp}")
    print(f"    Test flags:    {tf}")
    print(f"    Locked params: {phase.lock if phase.lock else '(none)'}")
    print()
    print("  Commands:")
    print("    trials <N>          — set trial count")
    print("    +param <name>       — add search param")
    print("    -param <name>       — remove search param")
    print("    +flag <name>        — add test flag")
    print("    -flag <name>        — remove test flag")
    print("    lock <key> <value>  — lock a param to a value")
    print("    unlock <key>        — remove a locked param")
    print("    done                — finish editing")
    print()

    new_phase = dataclasses.replace(phase)

    while True:
        cmd = _input("  edit> ").strip()
        if not cmd or cmd == "done":
            break

        parts = cmd.split(None, 2)
        action = parts[0].lower()

        handler = _EDIT_DISPATCH.get(action)
        if handler is not None:
            new_phase = handler(new_phase, parts)
        else:
            print("    Unknown command. Type 'done' to finish.")

    new_phases = list(config.phases)
    new_phases[idx] = new_phase
    return PipelineConfig(
        global_flags=dict(config.global_flags),
        phases=new_phases,
    )


def _do_reset_defaults(config: PipelineConfig) -> PipelineConfig:
    """Reset config to defaults (returns new config)."""
    confirm = _input("  Reset pipeline config to defaults? [y/N]: ").strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        _pause()
        return config

    new_config = PipelineConfig.default(is_moe=ctx.is_moe)
    print("  Reset to defaults.")
    _pause()
    return new_config


def _do_save(config: PipelineConfig) -> None:
    """Save config to results/<model>/pipeline-config.json."""
    save_path = Path(ctx.results_dir) / "pipeline-config.json"
    config.save(save_path)
    print(f"\n  Saved to {save_path}")
    _pause()


# ── Main Menu Loop ───────────────────────────────────────────────


def config_editor_menu() -> None:
    """Interactive config editor loop."""
    config_path = Path(ctx.results_dir) / "pipeline-config.json"
    config = PipelineConfig.load(config_path, is_moe=ctx.is_moe)

    while True:
        # Clear screen using menu.py's pattern
        from .menu import clear_screen

        clear_screen()

        _print_config_view(config)

        choice = _input("  > ").strip().lower()
        if not choice or choice == "b":
            return

        if choice == "g":
            config = _do_set_global_flag(config)
        elif choice == "r":
            config = _do_remove_global_flag(config)
        elif choice == "t":
            config = _do_toggle_phase(config)
        elif choice == "m":
            config = _do_move_phase(config)
        elif choice == "e":
            config = _do_edit_phase(config)
        elif choice == "d":
            config = _do_reset_defaults(config)
        elif choice == "s":
            _do_save(config)
        else:
            print("  Invalid choice.")
            _pause()
