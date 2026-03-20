"""CLI subpackage — menus, wizard, results display.

All business logic is in services.py. TUI modules handle only print/input.
"""

from .display import view_results
from .menu import (
    advanced_menu,
    clear_screen,
    context_menu,
    print_header,
    print_menu,
)
from .wizard import first_run_setup, needs_setup, switch_model

__all__ = [
    "advanced_menu",
    "clear_screen",
    "context_menu",
    "first_run_setup",
    "needs_setup",
    "print_header",
    "print_menu",
    "switch_model",
    "view_results",
]
