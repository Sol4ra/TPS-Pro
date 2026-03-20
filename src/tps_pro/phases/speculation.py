"""Speculative decoding phase."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import optuna

from ..constants import VRAM_FAILURE_PENALTY
from ..measurement import compute_score, get_best_trial
from ..pipeline_config import PhaseConfig
from ..result_types import EngineConfig, PhaseReturnDict
from ..search import (
    check_and_mark_duplicate_trial,
    clear_param_cache,
    print_param_importance,
    setup_study,
    trial_scalar_value,
    update_param_cache,
)
from ..state import AppContext
from .trial_helpers import (
    BestScoreTracker,
    PhaseSummaryContext,
    finalize_trial,
    print_phase_summary,
    record_trial_attrs,
    recover_best_score,
    run_server_trial,
    run_study_with_callbacks,
    setup_baseline_server,
    suggest_or_lock,
    thermal_gate,
)

logger = logging.getLogger(__name__)

__all__ = ["phase_speculation"]


def _suggest_spec_params(
    trial: optuna.Trial,
    draft_model: str | None,
    search_params: frozenset | None = None,
    lock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Suggest all speculation hyperparameters from an Optuna trial.

    Raises:
        optuna.exceptions.TrialPruned: When draft_min >= draft_max.
    """
    _lock = lock or {}

    spec_opts = [
        "ngram-simple",
        "ngram-cache",
        "ngram-map-k",
        "ngram-map-k4v",
        "ngram-mod",
    ]
    if draft_model:
        spec_opts.append("draft")

    def _sol(name, suggest_fn, default=None):
        return suggest_or_lock(name, suggest_fn, _lock, search_params, default)

    params = {
        "spec_type": _sol(
            "spec_type",
            lambda: trial.suggest_categorical("spec_type", spec_opts),
            default="ngram-simple",
        ),
        "spec_ngram_n": _sol(
            "spec_ngram_n",
            lambda: trial.suggest_int("spec_ngram_n", 2, 24),
            default=4,
        ),
        "spec_ngram_m": _sol(
            "spec_ngram_m",
            lambda: trial.suggest_int("spec_ngram_m", 8, 96),
            default=16,
        ),
        "spec_ngram_min_hits": _sol(
            "spec_ngram_min_hits",
            lambda: trial.suggest_int("spec_ngram_min_hits", 1, 5),
            default=2,
        ),
        "draft_max": _sol(
            "draft_max",
            lambda: trial.suggest_int("draft_max", 4, 48),
            default=16,
        ),
        "draft_min": _sol(
            "draft_min",
            lambda: trial.suggest_int("draft_min", 0, 8),
            default=0,
        ),
        "draft_p_min": _sol(
            "draft_p_min",
            lambda: trial.suggest_float("draft_p_min", 0.3, 0.99),
            default=0.5,
        ),
        "use_lookup_cache": _sol(
            "use_lookup_cache",
            lambda: trial.suggest_categorical("use_lookup_cache", [True, False]),
            default=True,
        ),
    }
    if params["draft_min"] >= params["draft_max"]:
        raise optuna.exceptions.TrialPruned()
    return params


def _build_spec_config(
    base_config: dict, params: dict, lookup_cache_file: str, draft_model: str | None
) -> tuple[dict, str]:
    """Build the server config dict and short description from suggested params.

    Returns (config dict, params_short string) tuple.
    """
    config = {
        **base_config,
        "spec_type": params["spec_type"],
        "spec_ngram_n": params["spec_ngram_n"],
        "spec_ngram_m": params["spec_ngram_m"],
        "spec_ngram_min_hits": params["spec_ngram_min_hits"],
        "draft_max": params["draft_max"],
        "draft_min": params["draft_min"],
        "draft_p_min": params["draft_p_min"],
    }
    if params["use_lookup_cache"] and lookup_cache_file:
        config["lookup_cache_dynamic"] = lookup_cache_file
    if draft_model and params["spec_type"] == "draft":
        config["model_draft"] = draft_model
    config = {k: v for k, v in config.items() if v is not None}

    params_short = (
        f"{params['spec_type']},"
        f" n={params['spec_ngram_n']},"
        f" m={params['spec_ngram_m']},"
        f" draft={params['draft_max']}"
        f"/{params['draft_min']},"
        f" p={params['draft_p_min']:.2f},"
        f" hits={params['spec_ngram_min_hits']},"
        f" lc={int(params['use_lookup_cache'])}"
    )
    return config, params_short


def _clear_lookup_cache_if_needed(
    use_lookup_cache: bool, lookup_cache_file: str
) -> None:
    """Remove existing lookup cache file if this trial uses lookup cache."""
    if use_lookup_cache and lookup_cache_file:
        Path(lookup_cache_file).unlink(missing_ok=True)


def phase_speculation(  # noqa: C901, PLR0915
    ctx: AppContext,
    n_trials: int = 40,
    base_config: EngineConfig | None = None,
    phase_config: PhaseConfig | None = None,
) -> PhaseReturnDict | None:
    """Phase 4: Speculative decoding sweep (N-gram or draft model).

    Isolated after core engine is locked so:
    1. N-gram cache isn't corrupted by batch/thread changes between trials
    2. Speculation is a multiplier on top of already-optimized base TPS
    3. Measurements are stable (only spec params vary)

    Parameters: spec_type, spec_ngram_n, spec_ngram_m, spec_ngram_min_hits,
                draft_max, draft_min, draft_p_min, lookup_cache_dynamic.
    If a draft model is configured, sweeps model_draft params instead.
    """
    phase_start_time = time.time()
    label = "Speculative Decoding"

    logger.info("=" * 60)
    logger.info("  %s", label)
    logger.info("=" * 60)
    logger.info("")

    # Use phase_config.trials if provided, otherwise use n_trials
    effective_trials = n_trials
    if phase_config is not None and phase_config.trials is not None:
        effective_trials = phase_config.trials

    is_pareto = ctx.config.get("pareto", False)
    study, remaining, completed = setup_study(
        ctx, "speculation", effective_trials, is_pareto=is_pareto
    )
    if remaining == 0:
        best = get_best_trial(ctx, study)
        logger.info(
            "Best Score: %.1f | TPS: %.1f",
            trial_scalar_value(best),
            best.user_attrs.get("tps", 0),
        )
        print_param_importance(study)
        clear_param_cache(study.study_name)
        return PhaseReturnDict(best_params=best.params, phase_name="speculation")

    if base_config is None:
        base_config = dict(ctx.naked_engine)

    logger.debug("Speculation base_config keys: %s", list(base_config.keys()))
    baseline, baseline_score = setup_baseline_server(ctx, base_config, "Speculation")
    if baseline is None:
        return None

    draft_model = ctx.config.get("draft_model")
    if draft_model:
        logger.info("Draft model: %s", Path(draft_model).name)

    # No seed trials — let TPE explore from scratch

    total_trials = completed + remaining
    best = BestScoreTracker(
        max(
            baseline_score,
            recover_best_score(study, compute_score) if completed > 0 else 0,
        )
    )

    # Build search_params and lock from phase_config if provided
    _cfg_search: frozenset | None = None
    _cfg_lock: dict = {}
    if phase_config is not None:
        if phase_config.search_params:
            _cfg_search = frozenset(phase_config.search_params)
        if phase_config.lock:
            _cfg_lock = dict(phase_config.lock)

    def objective(trial):
        thermal_gate()

        params = _suggest_spec_params(
            trial, draft_model, search_params=_cfg_search, lock=_cfg_lock
        )
        config, params_short = _build_spec_config(
            base_config, params, ctx.lookup_cache_file, draft_model
        )

        cached = check_and_mark_duplicate_trial(trial)
        if cached is not None:
            return cached

        _clear_lookup_cache_if_needed(params["use_lookup_cache"], ctx.lookup_cache_file)

        perf, score = run_server_trial(
            ctx, trial, config, params_short, best.value, is_pareto
        )
        if perf is None:
            return (0.0, VRAM_FAILURE_PENALTY, 0.0) if is_pareto else 0.0
        record_trial_attrs(ctx, trial, perf)
        result, best.value = finalize_trial(
            ctx, trial, perf, params_short, best.value, total_trials, is_pareto
        )
        update_param_cache(trial, result)
        return result

    logger.info("  Running %d trials...", remaining)
    run_study_with_callbacks(
        ctx, study, objective, remaining, label, best.value, is_pareto
    )

    summary_ctx = PhaseSummaryContext(
        phase_name="speculation",
        study=study,
        baseline=baseline,
        baseline_score=baseline_score,
        phase_start_time=phase_start_time,
        is_pareto=is_pareto,
    )
    returned_params, _ = print_phase_summary(ctx, summary_ctx)
    clear_param_cache(study.study_name)
    return PhaseReturnDict(best_params=returned_params, phase_name="speculation")
