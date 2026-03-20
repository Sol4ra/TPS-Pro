"""
Shared trial execution helpers for Optuna-based optimization phases.

Reduces ~40-50 lines of duplicated boilerplate per phase by extracting the
common patterns: server boot, measurement, attribute recording, result printing,
and post-study summary.

Error strategy (see errors.py for full documentation):
    - run_server_trial(): raises TrialPruned on OOM (Optuna handles this
      as a normal pruned trial).  Returns (None, 0.0) on non-OOM boot
      failure -- the caller treats this as a failed trial with score 0.
    - setup_baseline_server(): raises BaselineFailure when --fail-fast is
      enabled and baseline cannot start.  Otherwise returns (None, 0.0).
    - print_phase_summary(): catches nothing -- errors propagate to the
      pipeline orchestrator which handles them per _run_phase().
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Callable, cast

# Note: trial_helpers uses both optuna.Trial (live trials in objectives) and
# optuna.trial.FrozenTrial (completed trials in summaries). Functions that
# accept either use the X | Y type syntax below.
import optuna

from ..constants import (
    STUDY_SAFETY_TIMEOUT_SEC,
    THERMAL_COOLDOWN_TARGET,
    THERMAL_COOLDOWN_TIMEOUT,
    THERMAL_THROTTLE_THRESHOLD,
)
from ..engine import (
    BaselineFailure,
    boot_server_with_jinja_recovery,
    kill_server,
    server_start_failed,
)
from ..hardware import check_thermal_throttle, get_vram_used_mb, wait_for_cooldown
from ..measurement import (
    compute_pareto_objectives,
    compute_score,
    extract_pareto_front,
    get_best_trial,
    measure_perf_adaptive,
    measure_perf_once,
    print_pareto_front,
)
from ..result_types import EngineConfig, PerfResult
from ..search import (
    ProgressBarUpdateCallback,
    clear_param_cache,
    close_phase_pbar,
    create_phase_pbar,
    print_param_importance,
    print_trial_result,
    save_phase_results,
    trial_scalar_value,
)
from ..state import AppContext

logger = logging.getLogger(__name__)

__all__ = [
    "BestScoreTracker",
    "suggest_or_lock",
    "thermal_gate",
    "run_server_trial",
    "record_trial_attrs",
    "finalize_trial",
    "recover_best_score",
    "run_study_with_callbacks",
    "print_phase_summary",
    "PhaseSummaryContext",
    "setup_baseline_server",
]


def suggest_or_lock(
    name: str,
    suggest_fn,
    lock: dict,
    search_params: frozenset | None = None,
    default=None,
):
    """Use locked value if present, skip if not in search_params, else suggest.

    Args:
        name: Parameter name.
        suggest_fn: Callable that suggests a value via Optuna trial.
        lock: Dict of locked parameter values.
        search_params: If provided, only suggest params in this set.
            Params not in the set use *default*.
        default: Value to use when the param is not in *search_params*.

    Returns:
        The locked, default, or suggested value.
    """
    if name in lock:
        return lock[name]
    if search_params is not None and name not in search_params:
        return default
    return suggest_fn()


class BestScoreTracker:
    """Mutable tracker for the best score seen across trials.

    Replaces the ``nonlocal best_score`` pattern used in objective closures.
    """

    __slots__ = ("value",)

    def __init__(self, initial: float = 0.0) -> None:
        self.value: float = initial


def thermal_gate() -> None:
    """Check GPU thermals and wait for cooldown if needed."""
    if check_thermal_throttle(threshold=THERMAL_THROTTLE_THRESHOLD)[0]:
        wait_for_cooldown(
            target_temp=THERMAL_COOLDOWN_TARGET, timeout=THERMAL_COOLDOWN_TIMEOUT
        )


def run_server_trial(  # noqa: PLR0913
    ctx: AppContext,
    trial: optuna.Trial,
    config: EngineConfig,
    params_short: str,
    best_score: float,
    is_pareto: bool = False,
) -> tuple[PerfResult | None, float]:
    """Execute one server-based trial: boot, gate, measure, record attrs.

    Returns (PerfResult, score) on success, or raises TrialPruned / returns
    (None, 0.0) on failure.

    Raises:
        optuna.exceptions.TrialPruned: When the server OOMs at boot or the
            multi-fidelity gate determines the trial is unpromising.
    """
    proc, status = boot_server_with_jinja_recovery(ctx, config)
    if status == "oom":
        logger.info("Trial %d: pruned (OOM)", trial.number)
        kill_server(ctx)
        raise optuna.exceptions.TrialPruned()
    elif status != "ok":
        server_start_failed(ctx, trial.number, params_short, proc)
        return None, 0.0

    # Multi-fidelity gate
    gate = measure_perf_once(ctx, n_predict=5)
    if gate and best_score > 0:
        gate_score = compute_score(gate)
        trial.report(gate_score, step=0)
        if trial.should_prune():
            logger.info("Trial %d: pruned by gate (%.1f)", trial.number, gate_score)
            raise optuna.exceptions.TrialPruned()

    perf, promoted = measure_perf_adaptive(ctx, best_score)
    score = compute_score(perf)
    trial.report(score, step=1)

    return perf, score


def record_trial_attrs(
    ctx: AppContext, trial: optuna.Trial, perf: PerfResult
) -> PerfResult:
    """Record standard performance attributes on a trial.

    Returns the (possibly updated) PerfResult — if VRAM was measured,
    a new PerfResult is returned with vram_used_mb set.
    """
    if perf.tps_std is not None:
        trial.set_user_attr("tps_std", perf.tps_std)
        trial.set_user_attr("tps_cv", perf.tps_cv or 0)

    vram_mb = get_vram_used_mb()
    if vram_mb is not None:
        trial.set_user_attr("vram_used_mb", vram_mb)
        perf = dataclasses.replace(perf, vram_used_mb=vram_mb)

    trial.set_user_attr("tps", perf.tps or 0)
    trial.set_user_attr("ttft", perf.ttft or 0)
    trial.set_user_attr("prompt_tps", perf.prompt_tps or 0)
    trial.set_user_attr("total_ms", perf.total_ms or 0)

    # Record model load time if captured from server stderr
    if perf.load_time_ms is not None:
        trial.set_user_attr("load_time_ms", perf.load_time_ms)

    return perf


def finalize_trial(  # noqa: PLR0913
    ctx: AppContext,
    trial: optuna.Trial,
    perf: PerfResult,
    params_short: str,
    best_score: float,
    total_trials: int,
    is_pareto: bool = False,
    score: float | None = None,
) -> tuple[float | tuple[float, ...], float]:
    """Print result and return the appropriate value (scalar or pareto tuple).

    Returns (return_value, new_best_score).
    """
    tps = perf.tps or 0
    if score is None:
        score = compute_score(perf)

    if not is_pareto:
        new_best = print_trial_result(
            ctx,
            trial.number,
            total_trials,
            tps,
            perf,
            params_short,
            best_score,
            final_score=score,
        )
        return score, new_best
    else:
        vram_mb = perf.vram_used_mb
        quality = perf.quality_factor if perf.quality_factor is not None else 1.0
        objectives = compute_pareto_objectives(perf, quality_factor=quality)
        logger.info(
            "Trial %d: TPS=%.1f VRAM=%.0fMB | %s",
            trial.number,
            tps,
            vram_mb or 0,
            params_short,
        )
        return cast(tuple[float, ...], objectives), best_score


def recover_best_score(study: optuna.Study, score_fn: Callable[..., float]) -> float:
    """Recover the best score from completed trials in a resumed study.

    Reconstructs PerfResult from trial user_attrs for scoring.
    """
    best = 0.0
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs:
            perf = PerfResult(
                tps=t.user_attrs.get("tps", 0),
                ttft=t.user_attrs.get("ttft", 0),
                prompt_tps=t.user_attrs.get("prompt_tps", 0),
                total_ms=t.user_attrs.get("total_ms", 0),
            )
            if perf.tps > 0:
                best = max(best, score_fn(perf))
    return best


def run_study_with_callbacks(  # noqa: PLR0913
    ctx: AppContext,
    study: optuna.Study,
    objective: Callable[..., Any],
    remaining: int,
    label: str,
    best_score: float,
    is_pareto: bool = False,
    timeout_seconds: int | None = None,
) -> None:
    """Run study.optimize with standard callbacks (progress bar + GP stopping).

    If *timeout_seconds* is None, defaults to 90 minutes (enough for 100+
    trials but prevents infinite hangs if a single trial blocks forever).
    """
    if timeout_seconds is None:
        timeout_seconds = STUDY_SAFETY_TIMEOUT_SEC
    create_phase_pbar(remaining, desc=label)
    callbacks = [ProgressBarUpdateCallback()]
    # GP early stopping disabled — run all trials
    study.optimize(
        objective,
        n_trials=remaining,
        callbacks=callbacks,
        show_progress_bar=False,
        timeout=timeout_seconds,
    )
    close_phase_pbar()
    clear_param_cache(study.study_name)


@dataclasses.dataclass
class PhaseSummaryContext:
    """Groups parameters for print_phase_summary to reduce argument count."""

    phase_name: str
    study: optuna.Study
    baseline: Any
    baseline_score: float
    phase_start_time: float
    is_pareto: bool = False
    score_fn: Callable[..., float] | None = None
    param_keys: list[str] | None = None


def print_phase_summary(
    ctx: AppContext,
    summary: PhaseSummaryContext,
) -> tuple[EngineConfig, dict[str, Any]]:
    """Print standard phase results and save to disk.

    Accepts baseline as PerfResult or dict. Returns (best_params_dict, results_dict).

    Must be called with a PhaseSummaryContext:
        print_phase_summary(ctx, PhaseSummaryContext(...))
    """
    phase_name = summary.phase_name
    study = summary.study
    baseline = summary.baseline
    baseline_score = summary.baseline_score
    phase_start_time = summary.phase_start_time
    is_pareto = summary.is_pareto
    score_fn = summary.score_fn
    param_keys = summary.param_keys

    if score_fn is None:
        score_fn = compute_score

    best = get_best_trial(ctx, study)
    phase_elapsed = time.time() - phase_start_time

    logger.info("")
    logger.info("=" * 60)
    logger.info("  %s — RESULTS", phase_name)
    logger.info("=" * 60)
    logger.info("")

    bl_tps = getattr(baseline, "tps", 0)
    bl_pp = getattr(baseline, "prompt_tps", 0)
    bl_ttft = getattr(baseline, "ttft", 0)
    logger.info(
        "  Baseline: %.1f t/s | pp: %.0f | TTFT: %.0fms | score: %.1f",
        bl_tps,
        bl_pp,
        bl_ttft,
        baseline_score,
    )

    if is_pareto:
        pareto = extract_pareto_front(study)
        logger.info("  Pareto Front: %d configs", len(pareto))
        print_pareto_front(pareto)
        returned_params = best.params
    else:
        bv = trial_scalar_value(best) or 0
        beat_baseline = bv > baseline_score
        if beat_baseline:
            returned_params = best.params
        else:
            if param_keys:
                returned_params = {
                    k: getattr(baseline, k)
                    for k in param_keys
                    if getattr(baseline, k, None) is not None
                }
            else:
                returned_params = best.params

    best_tps = best.user_attrs.get("tps", 0)
    best_pp = best.user_attrs.get("prompt_tps", 0)
    best_ttft = best.user_attrs.get("ttft", 0)
    best_score = trial_scalar_value(best) or 0
    logger.info(
        "  Optimal:  %.1f t/s | pp: %.0f | TTFT: %.0fms | score: %.1f",
        best_tps,
        best_pp,
        best_ttft,
        best_score,
    )
    logger.info("")
    logger.info(
        "  Params:    %s", ", ".join(f"{k}={v}" for k, v in returned_params.items())
    )

    importances = print_param_importance(study)

    # Serialize baseline for JSON storage
    baseline_dict = baseline.to_dict() if hasattr(baseline, "to_dict") else baseline

    results = {
        "phase": phase_name,
        "baseline": baseline_dict,
        "baseline_score": baseline_score,
        "beat_baseline": (trial_scalar_value(best) or 0) > baseline_score
        if not is_pareto
        else True,
        "best_tps": best.user_attrs.get("tps", 0),
        "best_metrics": best.user_attrs,
        "best_params": returned_params,
        "param_importance": {k: round(v * 100, 1) for k, v in importances.items()},
        "duration_minutes": round(phase_elapsed / 60, 1),
        "all_trials": [
            {
                "number": t.number,
                "tps": trial_scalar_value(t),
                "metrics": t.user_attrs,
                "params": t.params,
            }
            for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ],
    }
    save_phase_results(ctx, phase_name, results)
    return cast(EngineConfig, returned_params), results


def setup_baseline_server(
    ctx: AppContext,
    base_config: EngineConfig,
    phase_name: str,
    runs: int = 3,
    start_label: str = "baseline server",
) -> tuple[PerfResult | None, float]:
    """Boot a baseline server and measure initial performance.

    Args:
        base_config: Server configuration dict.
        phase_name: Human-readable phase name used in failure messages.
        runs: Number of measurement runs passed to measure_perf.
        start_label: Subject used in the "[*] Starting ... ..." message, e.g.
                     "baseline server (f16 KV)".

    Returns:
        (PerfResult, baseline_score) on success.
        (None, 0.0) when the baseline server fails to start and
            ``ctx.fail_fast`` is False.

    Raises:
        BaselineFailure: When the baseline server fails to start and
            ``ctx.fail_fast`` is True.
    """
    logger.debug("Starting %s...", start_label)
    kill_server(ctx)
    proc, status = boot_server_with_jinja_recovery(ctx, base_config)
    if status != "ok":
        logger.warning("Baseline failed in %s", phase_name)
        if ctx.fail_fast:
            raise BaselineFailure(f"Baseline server failed in {phase_name}.")
        return None, 0.0
    baseline, _ = measure_perf_adaptive(ctx, runs=runs)
    baseline_score = compute_score(baseline)
    logger.info(
        "  Baseline: %.1f t/s | pp: %.0f | TTFT: %.0fms | score: %.1f",
        baseline.tps,
        baseline.prompt_tps,
        baseline.ttft,
        baseline_score,
    )
    logger.info("")
    return baseline, baseline_score
