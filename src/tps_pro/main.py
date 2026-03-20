"""Entry point — initialize, validate, run menu loop.

Heavy imports (optuna, pipeline, engine, phases) are deferred until needed
to keep the startup time under 1 second for the menu screen.
"""

from __future__ import annotations

import atexit
import logging
import signal
import socket
import sys
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)
__all__ = ["main"]


# ── Cleanup ──────────────────────────────────────────────────────


def _safe_kill_server(kill_fn: Callable, app_ctx: object) -> None:
    """atexit -- kill server, never raise."""
    try:
        kill_fn(app_ctx, wait=True)
    except Exception:
        logger.debug("atexit kill_server failed", exc_info=True)


def _install_interrupt_handler() -> None:
    """Ctrl+C — kill server, print summary, re-raise."""

    def _handler(_sig, _frame):
        print("\n\n  INTERRUPT — cleaning up...")
        try:
            from .engine import kill_server
            from .state import ctx

            kill_server(ctx)
        except Exception:
            logger.debug("Interrupt handler kill_server failed", exc_info=True)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handler)


# ── Menu handlers ────────────────────────────────────────────────


def _pause() -> None:
    try:
        input("\n  Press Enter to continue...")
    except EOFError:
        pass


def _run_optimize() -> None:
    """Run full pipeline, auto-detect resume."""
    from .cli import services
    from .pipeline import run_full_pipeline
    from .state import ctx

    progress = services.get_pipeline_progress(ctx)
    resume_idx = services.find_resume_point(progress)
    resume_from = None

    has_any_progress = any(p.status in ("done", "partial") for p in progress)
    if has_any_progress:
        done = [p for p in progress if p.status == "done"]
        partial = [p for p in progress if p.status == "partial"]
        print("\n  Found existing progress:")
        print(f"    {len(done)} phases complete, {len(partial)} in progress")
        print("  [1] Continue from where you left off")
        print("  [2] Start fresh (reset all)")
        try:
            choice = input("  > ").strip()
        except EOFError:
            return
        if choice == "2":
            services.reset_database(ctx)
            print("  Reset complete.")
        else:
            resume_from = resume_idx

    from .cli.menu import clear_screen

    clear_screen()
    ctx.fresh_run = resume_from is None
    import optuna

    try:
        if resume_from is not None:
            run_full_pipeline(resume_from=resume_from)
        else:
            run_full_pipeline()
    except (KeyboardInterrupt, SystemExit):
        raise
    except (OSError, RuntimeError, ValueError, optuna.exceptions.OptunaError) as e:
        logger.error("Pipeline error: %s", e, exc_info=True)
        print(f"\n  [!] Pipeline error: {e}")
    ctx.fresh_run = False
    _pause()


def _cycle_preset() -> None:
    """Cycle quick -> normal -> thorough."""
    from .cli import services
    from .state import config as _config
    from .state import get_config

    current = get_config("preset", "normal")
    new = services.cycle_preset(_config)
    print(f"  Preset changed: {current} -> {new}")
    _pause()


def _view_results() -> None:
    from .cli.display import view_results
    view_results()


def _advanced_menu() -> None:
    from .cli.menu import advanced_menu
    advanced_menu()


def _switch_model() -> None:
    from .cli.wizard import switch_model
    switch_model()


def _context_menu() -> None:
    from .cli.menu import context_menu
    context_menu()


_CHOICE_DISPATCH: dict[str, Callable[[], None]] = {
    "o": _run_optimize,
    "v": _view_results,
    "a": _advanced_menu,
    "m": _switch_model,
    "p": _cycle_preset,
    "c": _context_menu,
}


def _handle_choice(choice: str) -> None:
    """Lazy dispatch — heavy imports happen only when user picks an option."""
    try:
        handler = _CHOICE_DISPATCH.get(choice)
        if handler is not None:
            handler()
        else:
            print("  Invalid choice.")
            time.sleep(0.5)
    except (KeyboardInterrupt, SystemExit):
        raise
    except (OSError, RuntimeError, ValueError) as e:
        print(f"\n  [!] Error: {e}")
        _pause()


# ── Main ─────────────────────────────────────────────────────────


def main() -> None:  # noqa: C901, PLR0915
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("urllib3.util.retry").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.WARNING)

    # Lazy imports — only load what's needed for startup
    from .constants import BIND_HOST
    from .state import ctx, get_config, initialize

    initialize()

    from .engine import kill_server as _kill_fn

    atexit.register(_safe_kill_server, _kill_fn, ctx)

    from .cli.setup import detect_hardware_and_model, run_first_time_setup
    from .cli.wizard import needs_setup

    if needs_setup():
        run_first_time_setup()

    # Port check (short timeout to avoid delay when server isn't running)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.1)
        if s.connect_ex((BIND_HOST, ctx.port)) == 0:
            print(f"  [!] Port {ctx.port} already in use.")
            sys.exit(1)

    # Path check
    if not ctx.server_path.is_file():
        print(f"  [!] llama-server not found: {ctx.server_path}")
        sys.exit(1)
    if not ctx.model_path.is_file():
        print(f"  [!] Model not found: {ctx.model_path}")
        sys.exit(1)

    from .search import ensure_results_dir

    ensure_results_dir(ctx)

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if get_config("dry_run"):
        ctx.dry_run = True

    from .engine import LogTee

    log_tee = LogTee(str(ctx.results_dir))
    sys.stdout = log_tee

    _install_interrupt_handler()

    if get_config("dashboard"):
        from .cli.dashboard import launch_dashboard
        launch_dashboard(ctx)

    from .hardware import kill_competing_processes

    detect_hardware_and_model(kill_competing_processes)

    # Batch mode
    if get_config("batch_dir"):
        from .pipeline import batch_optimize
        try:
            batch_optimize(
                models_dir=get_config("batch_dir"),
                preset=get_config("preset", "normal"),
                skip_existing=get_config("skip_existing", False),
                timeout_minutes=get_config("timeout_minutes", 0),
                interactive=get_config("interactive", False),
            )
        finally:
            log_tee.close()
        return

    # Menu loop — these imports are cheap (already loaded by detect_hardware)
    from .cli.menu import clear_screen, print_header, print_menu

    while True:
        clear_screen()
        print_header()
        print_menu()

        try:
            choice = input("  > ").strip().lower()
        except EOFError:
            break

        if choice == "q":
            break
        _handle_choice(choice)
