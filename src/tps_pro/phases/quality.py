"""Quality / Sampling phase."""

from __future__ import annotations

import logging
from typing import Any, cast

from ..constants import (
    HAS_AIOHTTP,
    QUALITY_TASKS,
    QUALITY_WEIGHT_CONFIDENCE,
    QUALITY_WEIGHT_CORRECTNESS,
    QUALITY_WEIGHT_EFFICIENCY,
)
from ..engine import boot_server_with_jinja_recovery
from ..evals import measure_quality
from ..measurement import get_best_trial
from ..pipeline_config import PhaseConfig
from ..result_types import EngineConfig, PhaseReturnDict, SamplingParams
from ..search import (
    ProgressBarUpdateCallback,
    check_and_mark_duplicate_trial,
    clear_param_cache,
    close_phase_pbar,
    create_phase_pbar,
    load_phase_results,
    safe_best_value,
    save_phase_results,
    setup_study,
    update_param_cache,
)
from ..state import AppContext
from ._helpers import build_phase_config, get_moe_config
from .trial_helpers import BestScoreTracker, suggest_or_lock

logger = logging.getLogger(__name__)

__all__ = ["phase_quality"]


def _quality_objective(  # noqa: PLR0913, PLR0915
    trial,
    ctx,
    best,
    total_trials,
    search_params: frozenset | None = None,
    lock: dict | None = None,
):
    """Optuna objective for a single quality/sampling trial.

    Args:
        trial: Optuna trial object.
        best: BestScoreTracker tracking the best score so far.
        total_trials: Total number of trials for progress display.
        search_params: If provided, only suggest params in this set.
        lock: If provided, use locked values instead of suggesting.

    Returns:
        float | list[float]: Quality score for this trial.
    """
    _lock = lock or {}

    def _sol(name, suggest_fn, default=None):
        return suggest_or_lock(name, suggest_fn, _lock, search_params, default)

    # Mirostat overrides temperature/top_p/top_k/min_p -- conditional search space
    mirostat = _sol(
        "mirostat", lambda: trial.suggest_categorical("mirostat", [0, 1, 2]), default=0
    )
    params = {"mirostat": mirostat}

    if mirostat == 0:
        # Standard samplers (only active when mirostat is off)
        params["temperature"] = _sol(
            "temperature",
            lambda: trial.suggest_float("temperature", 0.0, 1.5),
            default=0.4,
        )
        params["top_p"] = _sol(
            "top_p",
            lambda: trial.suggest_float("top_p", 0.5, 1.0),
            default=0.95,
        )
        params["top_k"] = _sol(
            "top_k",
            lambda: trial.suggest_int("top_k", 1, 100),
            default=40,
        )
        params["min_p"] = _sol(
            "min_p",
            lambda: trial.suggest_float("min_p", 0.0, 0.3),
            default=0.05,
        )
        params["typical_p"] = _sol(
            "typical_p",
            lambda: trial.suggest_float("typical_p", 0.5, 1.0),
            default=1.0,
        )
        params["top_n_sigma"] = _sol(
            "top_n_sigma",
            lambda: trial.suggest_float("top_n_sigma", 0.0, 3.0),
            default=0.0,
        )
        params["dynatemp_range"] = _sol(
            "dynatemp_range",
            lambda: trial.suggest_float("dynatemp_range", 0.0, 1.0),
            default=0.0,
        )
        params["dynatemp_exp"] = _sol(
            "dynatemp_exp",
            lambda: trial.suggest_float("dynatemp_exp", 0.5, 2.0),
            default=1.0,
        )
    else:
        # Mirostat-specific params (only active when mirostat is on)
        params["mirostat_lr"] = _sol(
            "mirostat_lr",
            lambda: trial.suggest_float("mirostat_lr", 0.01, 0.5),
            default=0.1,
        )
        params["mirostat_ent"] = _sol(
            "mirostat_ent",
            lambda: trial.suggest_float("mirostat_ent", 1.0, 10.0),
            default=5.0,
        )

    # Penalties and repetition control -- always active
    params["repeat_penalty"] = _sol(
        "repeat_penalty",
        lambda: trial.suggest_float("repeat_penalty", 1.0, 1.3),
        default=1.0,
    )
    params["repeat_last_n"] = _sol(
        "repeat_last_n",
        lambda: trial.suggest_categorical("repeat_last_n", [0, 32, 64, 128, 256]),
        default=64,
    )
    params["presence_penalty"] = _sol(
        "presence_penalty",
        lambda: trial.suggest_float("presence_penalty", 0.0, 0.5),
        default=0.0,
    )
    params["frequency_penalty"] = _sol(
        "frequency_penalty",
        lambda: trial.suggest_float("frequency_penalty", 0.0, 0.5),
        default=0.0,
    )

    # XTC and DRY samplers -- always active
    params["xtc_probability"] = _sol(
        "xtc_probability",
        lambda: trial.suggest_float("xtc_probability", 0.0, 0.5),
        default=0.0,
    )
    params["xtc_threshold"] = _sol(
        "xtc_threshold",
        lambda: trial.suggest_float("xtc_threshold", 0.01, 0.5),
        default=0.1,
    )
    params["dry_multiplier"] = _sol(
        "dry_multiplier",
        lambda: trial.suggest_float("dry_multiplier", 0.0, 1.0),
        default=0.0,
    )
    params["dry_base"] = _sol(
        "dry_base", lambda: trial.suggest_float("dry_base", 1.0, 3.0), default=1.75
    )
    params["dry_allowed_length"] = _sol(
        "dry_allowed_length",
        lambda: trial.suggest_int("dry_allowed_length", 1, 5),
        default=2,
    )
    params["dry_penalty_last_n"] = _sol(
        "dry_penalty_last_n",
        lambda: trial.suggest_categorical(
            "dry_penalty_last_n",
            [-1, 0, 64, 128, 256, 512],
        ),
        default=-1,
    )

    # Adaptive sampling
    params["adaptive_target"] = _sol(
        "adaptive_target",
        lambda: trial.suggest_float("adaptive_target", 0.0, 1.0),
        default=0.0,
    )
    params["adaptive_decay"] = _sol(
        "adaptive_decay",
        lambda: trial.suggest_float("adaptive_decay", 0.0, 1.0),
        default=0.0,
    )

    # Check for duplicate config before running quality eval
    cached = check_and_mark_duplicate_trial(trial)
    if cached is not None:
        logger.debug("  Quality: (duplicate, score:%.1f)", cached)
        return cached

    score = measure_quality(
        ctx, cast(SamplingParams, params), target_to_beat=best.value
    ).score

    marker = ""
    if score > best.value:
        best.value = score
        marker = " *** NEW BEST ***"

    done = trial.number + 1
    pct = done / total_trials * 100
    bar_len = 20
    filled = int(bar_len * done / total_trials)
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

    if mirostat == 0:
        line1 = (
            f"miro=0 temp={params['temperature']:.2f} top_p={params['top_p']:.2f} "
            f"top_k={params['top_k']:3d} min_p={params['min_p']:.3f} "
            f"typ_p={params['typical_p']:.2f} tns={params['top_n_sigma']:.2f} "
            f"dtr={params['dynatemp_range']:.2f} dte={params['dynatemp_exp']:.2f}"
        )
    else:
        line1 = (
            f"miro={mirostat} lr={params['mirostat_lr']:.3f} "
            f"ent={params['mirostat_ent']:.1f}"
        )
    line2 = (
        f"rpen={params['repeat_penalty']:.2f} rln={params['repeat_last_n']} "
        f"pres={params['presence_penalty']:.2f} freq={params['frequency_penalty']:.2f} "
        f"xtcp={params['xtc_probability']:.2f} xtct={params['xtc_threshold']:.2f} "
        f"dry={params['dry_multiplier']:.2f} dryb={params['dry_base']:.2f} "
        f"dryal={params['dry_allowed_length']} drypln={params['dry_penalty_last_n']} "
        f"atgt={params['adaptive_target']:.2f} adcy={params['adaptive_decay']:.2f}"
    )
    logger.info(
        "[%s] %5.1f%%  Trial %3d/%d: %5.0f%% | %s%s",
        bar,
        pct,
        trial.number,
        total_trials,
        score,
        line1,
        marker,
    )
    logger.info("%s  %s  %s", " " * 20, " " * (8 + len(str(total_trials)) * 2), line2)

    update_param_cache(trial, score)
    return score


def phase_quality(  # noqa: C901, PLR0912, PLR0915
    ctx: AppContext,
    n_trials: int = 80,
    phase_config: PhaseConfig | None = None,
) -> PhaseReturnDict | None:
    """Optimize sampling params. Server runs with best compute + memory config."""
    if n_trials <= 0:
        return None
    logger.info("=" * 60)
    logger.info("  Quality / Sampling")
    logger.info("=" * 60)
    logger.info("")

    # Build server config from all completed pipeline phases
    server_config = build_phase_config(ctx)

    # Also merge MoE config if available
    p1a = load_phase_results(ctx, "moe_combined")
    if p1a:
        moe_cfg = get_moe_config(ctx, p1a)
        cast(dict[str, Any], server_config).update(moe_cfg)

    # Remove None values
    server_config = cast(
        EngineConfig, {k: v for k, v in server_config.items() if v is not None}
    )

    phases_loaded = [
        p
        for p in ["gpu", "core_engine", "io_toggles", "speculation", "kv_quality"]
        if load_phase_results(ctx, p)
    ]
    if phases_loaded:
        logger.info("Config merged from phases: %s", ", ".join(phases_loaded))
    else:
        logger.warning("No phase results found \u2014 running with naked engine.")

    # Enable concurrent eval slots for async quality measurement
    # Cap parallel to 4: llama-server partitions context across slots, so
    # context=4096 / parallel=10 = only 409 tokens per slot (too small for prompts).
    # With parallel=4 and context>=4096, each slot gets 1024+ tokens.
    if HAS_AIOHTTP:
        max_parallel = 4
        server_config["parallel"] = min(len(QUALITY_TASKS), max_parallel)
        server_config["context"] = max(server_config.get("context", 4096), 4096)

    # Start server
    logger.info("Starting server with best config...")
    proc, status = boot_server_with_jinja_recovery(ctx, server_config)
    if status != "ok":
        logger.warning("Server failed to start with combined config")
        return None

    load_ms = proc.load_time_ms
    if load_ms:
        logger.info("Model loaded in %.0fms", load_ms)

    # Baseline quality
    if HAS_AIOHTTP:
        logger.info(
            "[async] Concurrent quality evals enabled (--parallel %s)",
            len(QUALITY_TASKS),
        )
    logger.info("Measuring baseline quality...")
    baseline_score = measure_quality(
        ctx,
        {
            "temp": 0.4,
            "top_p": 0.95,
            "top_k": 40,
            "min_p": 0.05,
            "repeat_penalty": 1.0,
        },
    ).score
    logger.info(
        "Baseline: %.1f%% (3-signal:"
        " correctness\u00d7%.0f%%"
        " + confidence\u00d7%.0f%%"
        " + efficiency\u00d7%.0f%%)",
        baseline_score,
        QUALITY_WEIGHT_CORRECTNESS * 100,
        QUALITY_WEIGHT_CONFIDENCE * 100,
        QUALITY_WEIGHT_EFFICIENCY * 100,
    )

    # Use phase_config.trials if provided, otherwise use n_trials
    effective_trials = n_trials
    if phase_config is not None and phase_config.trials is not None:
        effective_trials = phase_config.trials

    # Build search_params and lock from phase_config if provided
    _cfg_search: frozenset | None = None
    _cfg_lock: dict = {}
    if phase_config is not None:
        if phase_config.search_params:
            _cfg_search = frozenset(phase_config.search_params)
        if phase_config.lock:
            _cfg_lock = dict(phase_config.lock)

    # Multivariate TPE learns correlations between sampling
    # params (e.g., temperature x top_p). It handles mirostat's
    # conditional search space natively -- missing params ignored.
    study, remaining, completed = setup_study(ctx, "quality", effective_trials)
    if remaining == 0:
        clear_param_cache(study.study_name)
        return PhaseReturnDict(
            best_params=study.best_trial.params, phase_name="quality"
        )

    total_trials = completed + remaining
    best = BestScoreTracker(baseline_score)
    _sbv = safe_best_value(study)
    if completed > 0 and _sbv is not None:
        best.value = max(best.value, _sbv)

    def objective(trial):
        return _quality_objective(
            trial, ctx, best, total_trials, search_params=_cfg_search, lock=_cfg_lock
        )

    logger.info("  Running %s trials...", remaining)
    create_phase_pbar(remaining, desc="Quality")
    callbacks = [
        ProgressBarUpdateCallback(),
        # GP early stopping disabled — run all trials
    ]
    study.optimize(
        objective, n_trials=remaining, callbacks=callbacks, show_progress_bar=False
    )
    close_phase_pbar()

    best_trial = get_best_trial(ctx, study)
    beat_baseline = best_trial.value > baseline_score

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Quality — RESULTS")
    logger.info("=" * 60)
    logger.info("")
    logger.info("  Baseline: %.0f%%", baseline_score)
    if beat_baseline:
        logger.info(
            "  Optimal:  %.0f%% — beats baseline by %.0f%%",
            best_trial.value,
            best_trial.value - baseline_score,
        )
    else:
        logger.info(
            "  Optimal:  %.0f%% — BELOW baseline (%.0f%%)",
            best_trial.value,
            baseline_score,
        )
        logger.info("  No trial beat baseline. Using default sampling params.")
    logger.info("")
    logger.info(
        "  Params:   %s",
        ", ".join(f"{k}={v}" for k, v in best_trial.params.items()),
    )

    returned_params = best_trial.params if beat_baseline else {}

    results = {
        "phase": "quality",
        "baseline_score": baseline_score,
        "beat_baseline": beat_baseline,
        "best_score": best_trial.value,
        "best_params": returned_params,
        "eval_tasks": [
            {"prompt": p, "answer": a, "category": c} for p, a, c in QUALITY_TASKS
        ],
        "all_trials": [
            {"number": t.number, "score": t.value, "params": t.params}
            for t in study.trials
        ],
    }
    save_phase_results(ctx, "quality", results)
    clear_param_cache(study.study_name)

    return PhaseReturnDict(best_params=returned_params, phase_name="quality")
