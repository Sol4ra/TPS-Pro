"""Pipeline progress, results browsing, and database management.

Pure business-logic — ZERO print/input calls.
Split from services.py for cohesion; re-exported via services.py for
backward compatibility.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, TypedDict, Union

from typing_extensions import TypeAlias

from ..constants import SCORE_VERSION
from ..engine.util import read_json_safe
from ..search import load_phase_results
from ..state import AppContext, get_config
from .services_config import DatabaseResetError

logger = logging.getLogger(__name__)

#: Phase config values are heterogeneous JSON-derived scalars.
PhaseConfigValue: TypeAlias = Union[str, int, float, bool, None]
PhaseConfig: TypeAlias = Dict[str, PhaseConfigValue]

#: Phase detail dicts loaded from results JSON files.
PhaseDetailValue: TypeAlias = Union[
    str, int, float, bool, List[object], Dict[str, object], None
]
PhaseDetail: TypeAlias = Dict[str, PhaseDetailValue]


class ModelResultSummary(TypedDict):
    """Summary of optimization results for a single model."""

    name: str
    path: str
    phase_count: int
    best_tps: float
    last_modified: str


# ============================================================
# Data Classes
# ============================================================


@dataclass(frozen=True)
class PhaseProgress:
    """Progress state for a single pipeline phase."""

    name: str
    display_name: str
    status: str  # "done", "partial", "pending"
    completed_trials: int
    results_key: str = ""
    study_key: str | None = None
    preset_key: str | None = None


@dataclass(frozen=True)
class PhaseDisplayResult:
    """Summary of a completed (or partially completed) phase's results."""

    name: str
    display_name: str
    best_tps: float | None = None
    duration_seconds: float | None = None
    beat_baseline: bool | None = None
    trial_count: int = 0
    data: PhaseDetail = field(default_factory=dict)


# ============================================================
# Pipeline phase ordering (shared between progress & base-config)
# ============================================================

_PIPELINE_PHASES: list[tuple[str, str, str | None, str | None]] = [
    # (display_name, results_key, study_key, preset_key)
    ("GPU Offload", "gpu", None, None),
    ("MoE Sweep", "moe_sweep", None, None),
    ("KV + Context Sweep", "kv_context_sweep", None, None),
    ("A/B Toggles", "ab_toggles", None, None),
    ("Core Engine", "core_engine", "core_engine", None),
    ("Speculation", "speculation", "speculation", None),
    ("Workload Sim", "workload_sim", None, None),
    ("Quality", "quality", "quality", "quality"),
]

# Display ordering for result browsing.
_DISPLAY_PHASE_ORDER: list[str] = [
    "gpu",
    "moe_sweep",
    "kv_context_sweep",
    "ab_toggles",
    "core_engine",
    "speculation",
    "workload_sim",
    "quality",
]


# ============================================================
# Pipeline & Phase Progress
# ============================================================


def get_pipeline_progress(ctx: AppContext) -> list[PhaseProgress]:
    """Scan saved results and Optuna studies to determine pipeline progress.

    Returns:
        Ordered list of PhaseProgress for each pipeline phase.
    """
    is_pareto = get_config("pareto", False)
    progress: list[PhaseProgress] = []

    for display_name, results_key, study_key, preset_key in _PIPELINE_PHASES:
        has_results = load_phase_results(ctx, results_key) is not None

        completed_trials = 0
        if study_key:
            import optuna

            versioned = f"{study_key}_{SCORE_VERSION}" + (
                "_pareto" if is_pareto else ""
            )
            try:
                study = optuna.load_study(study_name=versioned, storage=ctx.optuna_db)
                completed_trials = len(study.trials)
            except (KeyError, RuntimeError, ValueError, OSError) as exc:
                # Optuna/alembic can raise varied errors on corrupt or missing DBs
                logger.debug(
                    "Could not load Optuna study %s: %s",
                    versioned,
                    exc,
                    exc_info=True,
                )

        if has_results:
            status = "done"
        elif completed_trials > 0:
            status = "partial"
        else:
            status = "pending"

        progress.append(
            PhaseProgress(
                name=results_key,
                display_name=display_name,
                status=status,
                completed_trials=completed_trials,
                results_key=results_key,
                study_key=study_key,
                preset_key=preset_key,
            )
        )
    return progress


def find_resume_point(progress: list[PhaseProgress]) -> int | None:
    """Find the index of the first phase that isn't fully done.

    Args:
        progress: Ordered list of PhaseProgress.

    Returns:
        Index of the first non-done phase, or None if all complete.
    """
    for i, p in enumerate(progress):
        if p.status != "done":
            return i
    return None


def build_phase_base_config(ctx: AppContext, phase_name: str) -> Dict[str, object]:
    """Build a base config by merging naked_engine with saved phase results.

    The merge order follows the pipeline dependency chain so that later
    phases inherit parameters discovered by earlier ones.

    Args:
        phase_name: Target phase name -- determines which prior results
                    to merge.

    Returns:
        Dict suitable for passing as ``base_config`` to a phase function.
    """
    _phase_deps: dict[str, list[str]] = {
        "speculation": ["core_engine", "io_toggles"],
        "kv_quality": ["core_engine", "io_toggles", "speculation"],
        "workload_sim": [
            "core_engine",
            "io_toggles",
            "speculation",
            "kv_quality",
        ],
        "tensor_split": ["core_engine"],
        "topology_sweep": ["core_engine"],
        "context_sweep": ["core_engine"],
    }
    sources = _phase_deps.get(phase_name, [])

    base = dict(ctx.naked_engine)
    for src in sources:
        data = load_phase_results(ctx, src)
        if data and "best_params" in data:
            base.update(data["best_params"])
    return base


# ============================================================
# Results
# ============================================================


def get_model_results(ctx: AppContext) -> list[ModelResultSummary]:
    """List all models that have optimization results on disk.

    Returns:
        List of dicts: ``name``, ``path``, ``phase_count``, ``best_tps``,
        ``last_modified``.
    """
    base_results_dir = ctx.results_dir
    if not base_results_dir.exists():
        return []

    model_dirs: list[Path] = []
    for d in sorted(base_results_dir.iterdir()):
        if d.is_dir() and any(d.glob("*_results.json")):
            model_dirs.append(d)

    results: list[ModelResultSummary] = []
    for d in model_dirs:
        phase_files = list(d.glob("*_results.json"))
        best_tps = 0.0
        for pf in phase_files:
            data = read_json_safe(pf)
            if data:
                bt = data.get("best_tps", 0)
                best_tps = max(best_tps, bt)

        newest = max(phase_files, key=lambda f: f.stat().st_mtime)
        mtime = datetime.fromtimestamp(
            newest.stat().st_mtime, tz=timezone.utc
        ).isoformat()

        results.append(
            {
                "name": d.name,
                "path": str(d),
                "phase_count": len(phase_files),
                "best_tps": best_tps,
                "last_modified": mtime,
            }
        )
    return results


def get_phase_results(model_dir: Path) -> list[PhaseDisplayResult]:
    """Load all phase results for a specific model directory.

    Args:
        model_dir: Path to the model's results directory.

    Returns:
        Ordered list of PhaseDisplayResult.
    """
    available: list[tuple[str, Path]] = []
    for phase in _DISPLAY_PHASE_ORDER:
        path = model_dir / f"{phase}_results.json"
        if path.exists():
            available.append((phase, path))
    seen = {a[0] for a in available}
    for path in sorted(model_dir.glob("*_results.json")):
        phase = path.stem.replace("_results", "")
        if phase not in seen:
            available.append((phase, path))

    results: list[PhaseDisplayResult] = []
    for phase, path in available:
        data = read_json_safe(path)
        if data is None:
            results.append(PhaseDisplayResult(name=phase, display_name=phase, data={}))
            continue

        tps = data.get("best_tps")
        dur_min = data.get("duration_minutes")
        dur_sec = dur_min * 60 if dur_min else None
        beat = data.get("beat_baseline")
        trials = data.get("all_trials", [])

        results.append(
            PhaseDisplayResult(
                name=phase,
                display_name=phase,
                best_tps=tps,
                duration_seconds=dur_sec,
                beat_baseline=beat,
                trial_count=len(trials),
                data=data,
            )
        )
    return results


def get_phase_detail(phase_path: Path) -> PhaseDetail:
    """Load full detail for a single phase result file.

    Args:
        phase_path: Path to a ``*_results.json`` file.

    Returns:
        Parsed JSON dict, or empty dict on read failure.
    """
    data = read_json_safe(phase_path)
    return data if data is not None else {}


# ============================================================
# Database Management
# ============================================================


def _safe_delete_file(path: Path) -> bool:
    """Try to delete a file; fall back to renaming on PermissionError.

    Args:
        path: File to delete.

    Returns:
        True if removed or renamed, False on failure.
    """
    try:
        path.unlink()
        return True
    except PermissionError:
        stale = path.with_suffix(".db.old")
        try:
            if stale.exists():
                stale.unlink()
        except (PermissionError, OSError) as exc:
            logger.debug("Could not remove stale file %s: %s", stale, exc)
        try:
            path.rename(stale)
            return True
        except (PermissionError, OSError) as exc:
            logger.debug("Could not rename locked file %s: %s", path, exc)
            return False


def reset_database(ctx: AppContext) -> bool:
    """Delete the Optuna DB and result JSON files so all phases start fresh.

    Returns:
        True if successful.

    Raises:
        DatabaseResetError: If the database file cannot be deleted or
                            renamed.
    """
    db_path = ctx.results_dir / "optuna.db"

    if db_path.exists():
        import optuna

        try:
            gc.collect()
            storage = optuna.storages.RDBStorage(ctx.optuna_db)
            for s in optuna.study.get_all_study_summaries(storage=storage):
                optuna.delete_study(study_name=s.study_name, storage=storage)
            del storage
            gc.collect()
        except (OSError, RuntimeError, ImportError) as exc:
            logger.warning(
                "Could not close Optuna DB connections before deletion: %s",
                exc,
            )

        if not _safe_delete_file(db_path):
            raise DatabaseResetError(
                "Could not delete DB -- close the optimizer and delete it manually."
            )

    for p in ctx.results_dir.glob("*_results.json"):
        try:
            p.unlink()
        except OSError as exc:
            logger.warning("Could not delete result file %s: %s", p, exc)

    logger.info("DB and results deleted. All phases will start fresh.")
    return True


def delete_study(ctx: AppContext, study_key: str) -> bool:
    """Delete a single Optuna study to allow a phase to restart.

    Args:
        study_key: The base study key (e.g. ``"core_engine"``).

    Returns:
        True if deleted or already absent, False on error.
    """
    import optuna

    is_pareto = get_config("pareto", False)
    versioned = f"{study_key}_{SCORE_VERSION}" + ("_pareto" if is_pareto else "")
    try:
        optuna.delete_study(study_name=versioned, storage=ctx.optuna_db)
        return True
    except KeyError:
        return True
    except (RuntimeError, OSError) as exc:
        logger.debug("Could not delete Optuna study %s: %s", versioned, exc)
        return False
