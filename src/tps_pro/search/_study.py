"""Optuna study management — create/resume studies, persist results, duplicate checking.

This module is the study management core.  Callback implementations live in
search_callbacks.py; display/output helpers live in search_display.py.

Functions:
    setup_study, save_phase_results, load_phase_results, ensure_results_dir,
    check_and_mark_duplicate_trial, update_param_cache, clear_param_cache.

Error strategy (see errors.py for full documentation):
    - setup_study(): catches KeyError when deleting a non-existent study
      (expected during fresh_run on first invocation).
    - load_phase_results(): returns None on corrupt JSON (logged at warning).
      Stale results (wrong score_version) also return None with a warning.
    - save_phase_results(): write errors propagate -- partial writes are
      prevented by the tmp-file-then-rename atomic pattern.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
import types
from pathlib import Path
from typing import Any

import optuna

from ..constants import SCORE_VERSION
from ..result_types import PhaseResult
from ..state import AppContext

logger = logging.getLogger(__name__)

__all__ = [
    "ensure_results_dir",
    "check_and_mark_duplicate_trial",
    "get_positive_completed_trials",
    "setup_study",
    "save_phase_results",
    "load_phase_results",
    "clear_param_cache",
    "update_param_cache",
]

# ---------------------------------------------------------------------------
# Parameter cache — thread-safe deduplication across resumed studies
# ---------------------------------------------------------------------------

_param_cache: dict[str, dict] = {}
_param_cache_lock = threading.Lock()


def clear_param_cache(study_name: str) -> None:
    """Remove cached trial params for a completed study."""
    with _param_cache_lock:
        _param_cache.pop(study_name, None)


def update_param_cache(trial: optuna.Trial, value: float | tuple[float, ...]) -> None:
    """Add a just-completed trial's params and value to the in-memory cache.

    Must be called before the objective function returns so that subsequent
    trials with identical params can skip re-running the benchmark.

    Args:
        trial: The live Optuna trial object (params and user_attrs already set).
        value: The return value of the objective — a scalar float for
               single-objective studies, or a tuple/list for Pareto studies.
    """
    study_key = trial.study.study_name
    with _param_cache_lock:
        if study_key not in _param_cache:
            # cache not yet initialised --
            # check_and_mark_duplicate_trial will build it
            return
        key = tuple(sorted(trial.params.items()))
        if isinstance(value, (tuple, list)):
            entry = types.SimpleNamespace(
                user_attrs=dict(trial.user_attrs),
                value=None,
                values=list(value),
            )
        else:
            entry = types.SimpleNamespace(
                user_attrs=dict(trial.user_attrs),
                value=value,
                values=None,
            )
        _param_cache[study_key][key] = entry


# ---------------------------------------------------------------------------
# Shared trial filters
# ---------------------------------------------------------------------------


def get_positive_completed_trials(
    study: optuna.Study,
) -> list[optuna.trial.FrozenTrial]:
    """Return completed trials with a positive scalar value.

    Filters ``study.trials`` for ``COMPLETE`` state and a scalar value > 0.
    Used by callbacks, display helpers, and anywhere that needs the set of
    "successful" trials from a study.
    """
    from ._callbacks import trial_scalar_value

    result = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        v = trial_scalar_value(t)
        if v is not None and v > 0:
            result.append(t)
    return result


# ---------------------------------------------------------------------------
# Study management — create/resume studies, persist results
# ---------------------------------------------------------------------------


def ensure_results_dir(ctx: AppContext) -> None:
    ctx.results_dir.mkdir(parents=True, exist_ok=True)


def check_and_mark_duplicate_trial(trial: optuna.Trial) -> float | list[float] | None:
    """Check if this exact param combo was already tested and copy cached user_attrs.

    If a duplicate is found, the cached user_attrs are copied onto the current
    trial (side effect) so the caller can skip re-running the benchmark.

    In multi-objective (Pareto) mode, returns the list of values instead of a scalar.
    Uses a frozen-params dict for O(1) lookup instead of scanning all past trials.
    """
    study = trial.study
    study_key = study.study_name
    with _param_cache_lock:
        if study_key not in _param_cache:
            _param_cache[study_key] = {}
            for past in study.trials:
                if past.state == optuna.trial.TrialState.COMPLETE:
                    key = tuple(sorted(past.params.items()))
                    _param_cache[study_key][key] = past

        key = tuple(sorted(trial.params.items()))
        past = _param_cache[study_key].get(key)  # type: ignore[assignment]
    if past is not None:
        for k, v in past.user_attrs.items():
            trial.set_user_attr(k, v)
        is_multi = len(study.directions) > 1
        if is_multi and past.values is not None:
            return past.values
        return past.value
    return None


def setup_study(  # noqa: PLR0913
    ctx: AppContext,
    study_name: str,
    n_trials: int,
    seed: int = 42,
    sampler_override: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    is_pareto: bool = False,
) -> tuple[optuna.Study, int, int]:
    """Create/resume an Optuna study. Returns (study, remaining_trials, completed).

    Study names are versioned with SCORE_VERSION so that formula changes
    automatically start fresh studies instead of resuming stale data.

    Args:
        is_pareto: If True, creates a multi-objective study with NSGA-II
                   (3 directions: TPS, -VRAM, quality). Phases must opt in
                   explicitly — single-objective phases must leave this False.
        pruner: Optional Optuna pruner for multi-fidelity trials (e.g., MedianPruner).
    """
    ensure_results_dir(ctx)
    versioned_name = f"{study_name}_{SCORE_VERSION}" + ("_pareto" if is_pareto else "")

    # If fresh run requested, delete any existing study with this name
    if ctx.fresh_run:
        try:
            optuna.delete_study(study_name=versioned_name, storage=ctx.optuna_db)
        except KeyError:
            pass  # Expected: study doesn't exist yet on first run

    # Default pruner: WilcoxonPruner uses statistical testing on paired measurements,
    # which is better than MedianPruner for noisy server benchmarks where measurements
    # don't have a natural ordering (prompt 3 isn't "later" than prompt 1).
    if pruner is None:
        pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.1)

    if is_pareto:
        sampler: optuna.samplers.BaseSampler = optuna.samplers.NSGAIISampler(seed=seed)
        study = optuna.create_study(
            directions=["maximize", "maximize", "maximize"],
            study_name=versioned_name,
            storage=ctx.optuna_db,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
        )
    else:
        sampler = (
            sampler_override
            if sampler_override is not None
            else optuna.samplers.TPESampler(
                multivariate=True,
                seed=seed,
                warn_independent_sampling=False,
            )
        )
        study = optuna.create_study(
            direction="maximize",
            study_name=versioned_name,
            storage=ctx.optuna_db,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
        )

    completed = len(study.trials)
    remaining = n_trials
    if completed > 0:
        logger.info(
            "Resuming from trial %d/%d (%d completed)", completed, n_trials, completed
        )
        remaining = max(0, n_trials - completed)
        if remaining == 0:
            logger.info("All trials already completed. Use more trials or reset DB.")

    return study, remaining, completed


def save_phase_results(
    ctx: AppContext, phase_name: str, results: dict[str, Any] | PhaseResult
) -> None:
    """Save phase results to JSON atomically (write tmp, then rename).

    Accepts a plain dict.  If a PhaseResult dataclass is passed, it is
    automatically converted via .to_dict().

    Prevents data corruption if user hits Ctrl+C mid-write.
    Tags results with SCORE_VERSION so load_phase_results can reject stale data.
    """
    ensure_results_dir(ctx)
    # Convert dataclass to dict if needed (PhaseResult, NIAHPhaseResult, etc.)
    if hasattr(results, "to_dict"):
        results = results.to_dict()
    results["score_version"] = SCORE_VERSION
    final_path = ctx.results_dir / f"{phase_name}_results.json"
    temp_fd = None
    temp_path = None
    try:
        temp_fd, temp_path_str = tempfile.mkstemp(
            dir=str(ctx.results_dir), suffix=".json.tmp", prefix=f"{phase_name}_"
        )
        temp_path = Path(temp_path_str)
        with open(temp_fd, "w", encoding="utf-8", closefd=True) as f:
            temp_fd = None  # ownership transferred to the file object
            json.dump(results, f, indent=2)
        temp_path.replace(final_path)
        logger.debug("Results saved to %s", final_path)
    except KeyboardInterrupt:
        # Clean up temp file on Ctrl+C
        if temp_fd is not None:
            import os as _os

            _os.close(temp_fd)
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    except Exception:
        # Clean up temp file on any failure
        if temp_fd is not None:
            import os as _os

            _os.close(temp_fd)
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def load_phase_results(ctx: AppContext, phase_name: str) -> dict[str, Any] | None:
    """Load saved results from a completed phase.

    Returns None if the results were saved under a different SCORE_VERSION,
    forcing a re-run so stale configs from an old formula aren't reused.
    """
    path = ctx.results_dir / f"{phase_name}_results.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, ValueError):
            logger.warning("%s results corrupted — will re-run", phase_name)
            return None
        if data.get("score_version") != SCORE_VERSION:
            logger.warning(
                "%s results from score %s (current: %s) — will re-run",
                phase_name,
                data.get("score_version", "v1"),
                SCORE_VERSION,
            )
            return None
        return data
    return None
