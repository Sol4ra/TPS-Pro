"""Optuna dashboard launcher — background subprocess for live trial visualization."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time

import optuna

from ..constants import BIND_HOST
from ..state import AppContext

logger = logging.getLogger(__name__)

_dashboard_proc = None

# Inline launcher script passed to ``python -c``.  The DB URL is injected via
# an environment variable so it is not visible to other users via ``ps`` output.
_LAUNCHER_CODE = (
    "import os, sys, optuna_dashboard; "
    "optuna_dashboard.run_server(os.environ['OPTUNA_DB_URL'], "
    "host=sys.argv[1], port=int(sys.argv[2]))"
)


def launch_dashboard(ctx: AppContext) -> subprocess.Popen | None:
    """Launch optuna-dashboard as a background subprocess.

    The dashboard provides live parallel coordinate plots, contour maps,
    and hyperparameter importance graphs -- all reading from the existing
    Optuna SQLite DB. Zero code changes needed; it's a free visualization layer.

    Raises:
        subprocess.CalledProcessError: If pip install of optuna-dashboard fails.
    """
    global _dashboard_proc

    try:
        import optuna_dashboard  # noqa: F401
    except ImportError:
        logger.info("Installing optuna-dashboard...")
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "optuna-dashboard",
                    "-q",
                    "--no-warn-script-location",
                ],
                timeout=120,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Failed to install optuna-dashboard: %s", e)
            return None
        import optuna_dashboard  # noqa: F401

    db_path = ctx.results_dir / "optuna.db"
    db_url = "sqlite:///" + str(db_path).replace("\\", "/")
    dashboard_port = ctx.port + 100  # e.g., 8190 if server is 8090

    # Ensure DB has Optuna tables -- dashboard connects with
    # skip_table_creation=True so it will crash if the DB is empty.
    # Creating a dummy study initializes the schema.
    try:
        _init_study = optuna.create_study(
            storage=db_url,
            study_name="_dashboard_init",
            load_if_exists=True,
            direction="maximize",
        )
        optuna.delete_study(study_name="_dashboard_init", storage=db_url)
    except (RuntimeError, OSError, ValueError) as e:
        # Cleanup: best-effort schema init -- tables may already exist or
        # the dummy study may have been concurrently deleted.  Non-fatal.
        logger.debug(
            "Dashboard DB init skipped"
            " (tables exist or study deleted): %s",
            e,
        )

    logger.info(
        "\n[*] Launching optuna-dashboard on http://%s:%s", BIND_HOST, dashboard_port
    )
    logger.info("    DB: %s", db_path)
    logger.info("    Open in your browser: http://%s:%s", BIND_HOST, dashboard_port)
    logger.info(
        "    This runs in the background — close it with Ctrl+C or kill the process.\n"
    )

    # Long-running background process -- no timeout needed (monitored below)
    safe_url = db_url.replace("\\", "/")
    launcher_env = {**os.environ, "OPTUNA_DB_URL": safe_url}
    proc = subprocess.Popen(
        [sys.executable, "-c", _LAUNCHER_CODE, BIND_HOST, str(dashboard_port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=launcher_env,
    )
    # Quick check — did it crash immediately?
    startup_check_delay = 3.0
    time.sleep(startup_check_delay)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode("utf-8", errors="replace")
        logger.warning("  [!] Dashboard failed to start: %s", stderr[:500])
        return None

    # Auto-open in browser
    import webbrowser

    webbrowser.open(f"http://{BIND_HOST}:{dashboard_port}")

    _dashboard_proc = proc
    return proc
