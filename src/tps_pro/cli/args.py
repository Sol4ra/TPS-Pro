"""CLI argument parsing for the optimizer.

Extracted from state.py to separate concerns: state.py handles AppContext
and config loading, this module handles argparse definitions.
"""

from __future__ import annotations

import argparse

_MAX_PORT = 65535


def _valid_port(value):
    """Argparse type validator: ensures port is in the valid range 1-65535."""
    try:
        port = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Port must be an integer, got {value!r}"
        ) from e
    if not (1 <= port <= _MAX_PORT):
        raise argparse.ArgumentTypeError(f"Port must be 1-65535, got {port}")
    return port


def parse_cli_args() -> argparse.Namespace:
    """Parse CLI arguments and return the namespace.

    Returns the parsed args and silently ignores unknown arguments
    (for forward compatibility with new flags).
    """
    parser = argparse.ArgumentParser(
        description="llama-server Parameter Optimizer", add_help=False
    )
    parser.add_argument("--server", help="Path to llama-server executable")
    parser.add_argument("--model", help="Path to GGUF model file")
    parser.add_argument("--chat-template", help="Path to chat template file")
    parser.add_argument("--results-dir", help="Path to results directory")
    parser.add_argument("--port", type=_valid_port, help="Server port (1-65535)")
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument(
        "--dense", action="store_true", help="Dense model (skip MoE phases)"
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit immediately if baseline server fails to start",
    )
    parser.add_argument(
        "--skip-quality", action="store_true", help="Skip Quality/sampling phase"
    )
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument(
        "--quick",
        action="store_const",
        const="quick",
        dest="preset",
        help="Quick optimization: fewer trials per phase",
    )
    preset_group.add_argument(
        "--normal",
        action="store_const",
        const="normal",
        dest="preset",
        help="Normal optimization: balanced trial counts (default)",
    )
    preset_group.add_argument(
        "--thorough",
        action="store_const",
        const="thorough",
        dest="preset",
        help="Thorough optimization: more trials per phase",
    )
    parser.set_defaults(preset="normal")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing",
    )
    parser.add_argument("--debug", action="store_true", help="Show debug output")
    parser.add_argument(
        "--batch", metavar="DIR", help="Optimize all GGUF models in directory"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Pause between phases for inspection"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        metavar="MINUTES",
        default=0,
        help="Per-model timeout in minutes",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip models with existing results"
    )
    parser.add_argument(
        "--no-jinja", action="store_true", help="Disable Jinja template parsing"
    )
    parser.add_argument(
        "--no-bench", action="store_true", help="Disable llama-bench acceleration"
    )
    parser.add_argument(
        "--dashboard", action="store_true", help="Launch optuna-dashboard web UI"
    )
    parser.add_argument(
        "--pareto",
        action="store_true",
        help="Enable multi-objective Pareto optimization",
    )
    parser.add_argument(
        "--simulate-users",
        type=int,
        default=0,
        metavar="N",
        help="Concurrent user load test with N simultaneous users",
    )
    parser.add_argument(
        "--kill-competing",
        action="store_true",
        help="Kill competing GPU processes (>500MB VRAM) before optimizing",
    )
    args, _ = parser.parse_known_args()
    return args
