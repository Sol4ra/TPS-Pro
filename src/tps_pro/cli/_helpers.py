"""Shared CLI helpers used by menu.py, config_editor.py, and other TUI modules."""

from __future__ import annotations

__all__ = ["safe_input", "pause", "show_error"]


def safe_input(prompt: str = "") -> str:
    """Safe input that returns empty string on EOF."""
    try:
        return input(prompt)
    except EOFError:
        return ""


def pause() -> None:
    """Wait for Enter."""
    safe_input("\n  Press Enter to continue...")


def show_error(msg: str) -> None:
    """Print an error message and pause."""
    print(f"\n  [!] {msg}")
    pause()
