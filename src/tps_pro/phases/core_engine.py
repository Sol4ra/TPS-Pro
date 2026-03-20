"""Core Engine phase: 2-layer optimization of all engine parameters.

Layer 1: Quick A/B sweeps for independent toggles (~8-12 trials, ~3 min)
Layer 2: Focused TPE co-optimization for correlated params (~40-60 trials)

Research-backed grouping:
- Layer 1 flags are truly independent (mlock, no_mmap, repack, op_offload, prio)
- Layer 2 flags have strong cross-correlations (threads×batch, threads×poll,
  threads×n_cpu_moe, batch×flash_attn, etc.) that TPE must learn together.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable
from typing import Any

import optuna

from ..constants import DEFAULT_CONTEXT_SIZE, VRAM_FAILURE_PENALTY
from ..engine import (
    BaselineFailure,
    boot_server_with_jinja_recovery,
    kill_server,
    server_start_failed,
)
from ..measurement import (
    compute_score,
    get_best_trial,
    measure_perf_adaptive,
    measure_perf_once,
)
from ..pipeline_config import PhaseConfig
from ..result_types import EngineConfig, PhaseReturnDict
from ..search import (
    check_and_mark_duplicate_trial,
    clear_param_cache,
    pbar_state,
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
    run_study_with_callbacks,
    suggest_or_lock,
    thermal_gate,
)

logger = logging.getLogger(__name__)

__all__ = ["phase_core_engine"]


# ============================================================
# Dataclass for objective function parameters
# ============================================================


@dataclasses.dataclass(frozen=True)
class _ObjectiveParams:
    """Explicit parameters for the Layer 2 objective function.

    Replaces the closure that previously captured these values from the
    enclosing _layer2_tpe_search scope.
    """

    ctx: AppContext
    layer2_base: dict[str, Any]
    score_fn: Callable[..., float]
    is_pareto: bool
    total_trials: int
    thread_opts: list[int]
    batch_opts: list[int]
    ubatch_opts: list[int]
    skip_flags: set[str]
    search_params: frozenset[str] | None = None
    lock: dict[str, Any] = dataclasses.field(default_factory=dict)


def _suggest_layer2_params(
    trial: optuna.Trial, params: _ObjectiveParams
) -> dict[str, Any]:
    """Suggest Optuna trial parameters for the Layer 2 search space.

    Builds the full config dict from trial suggestions, locked values,
    and skip flags. Raises TrialPruned for invalid combinations.
    """
    ctx = params.ctx
    _skip = params.skip_flags
    _search = params.search_params
    _lock = params.lock

    def _sol(name: str, suggest_fn: Callable[..., Any]) -> Any:
        return suggest_or_lock(name, suggest_fn, _lock, _search)

    # Core compute params (strong correlations)
    threads = _sol(
        "threads", lambda: trial.suggest_categorical("threads", params.thread_opts)
    )
    threads_batch = _sol(
        "threads_batch",
        lambda: trial.suggest_categorical("threads_batch", params.thread_opts),
    )
    batch_size = _sol(
        "batch_size",
        lambda: trial.suggest_categorical("batch_size", params.batch_opts),
    )
    ubatch_size = _sol(
        "ubatch_size",
        lambda: trial.suggest_categorical("ubatch_size", params.ubatch_opts),
    )

    # Flash attention (interacts with batch_size, required for KV quant)
    flash_attn = _sol(
        "flash_attn",
        lambda: trial.suggest_categorical("flash_attn", ["on", "off"]),
    )

    # Thread scheduling (interact with threads)
    poll = _sol(
        "poll", lambda: trial.suggest_categorical("poll", [0, 10, 25, 50, 100])
    )
    poll_batch = _sol(
        "poll_batch",
        lambda: trial.suggest_categorical("poll_batch", [0, 10, 25, 50, 100]),
    )

    # CPU affinity (interacts with threads, incompatible with numa)
    cpu_strict = _sol(
        "cpu_strict",
        lambda: (
            0
            if "cpu_strict" in _skip
            else trial.suggest_categorical("cpu_strict", [0, 1])
        ),
    )
    cpu_strict_batch = _sol(
        "cpu_strict_batch",
        lambda: (
            0
            if "cpu_strict_batch" in _skip
            else trial.suggest_categorical("cpu_strict_batch", [0, 1])
        ),
    )

    # Enforce ubatch <= batch (only when both are actively searched)
    if ubatch_size is not None and batch_size is not None and ubatch_size > batch_size:
        raise optuna.exceptions.TrialPruned()

    config: dict[str, Any] = {
        **params.layer2_base,
        "threads": threads,
        "threads_batch": threads_batch,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "flash_attn": flash_attn,
        "kv_cache_type": "f16",
        "poll": poll,
        "poll_batch": poll_batch,
        "cpu_strict": cpu_strict,
        "cpu_strict_batch": cpu_strict_batch,
        "n_gpu_layers": ctx.default_gpu_layers,
        "fit": True,
    }

    # NUMA (multi-NUMA only, incompatible with cpu_strict)
    if ctx.numa_nodes > 1 and "numa" not in _skip:
        numa = trial.suggest_categorical("numa", ["distribute", "isolate", "numactl"])
        config["numa"] = numa
        # Enforce: numa and cpu_strict conflict
        if numa != "disabled" and (cpu_strict == 1 or cpu_strict_batch == 1):
            raise optuna.exceptions.TrialPruned()

    # MoE: co-optimize n_cpu_moe with threads (shared thread pool)
    # Skip if moe_sweep phase already locked the optimal value
    if ctx.is_moe and "n_cpu_moe" not in params.layer2_base:
        n_cpu_moe = trial.suggest_int(
            "n_cpu_moe",
            8,
            min(24, ctx.moe_sweep_max, ctx.max_threads * 2),
        )
        config["n_cpu_moe"] = n_cpu_moe
        config["expert_used_count"] = ctx.default_experts
    elif ctx.is_moe:
        config["n_cpu_moe"] = params.layer2_base["n_cpu_moe"]
        config["expert_used_count"] = ctx.default_experts

    return {k: v for k, v in config.items() if v is not None}


def _layer2_objective(  # noqa: C901, PLR0915
    trial: optuna.Trial, params: _ObjectiveParams, best: BestScoreTracker
):
    """Objective function for Layer 2 TPE search.

    Extracted as a module-level function to eliminate the 137-line nested closure
    inside _layer2_tpe_search. All state is passed explicitly via *params* and
    *best* rather than captured from a closure.
    """
    ctx = params.ctx

    thermal_gate()

    config = _suggest_layer2_params(trial, params)

    # Duplicate check
    cached = check_and_mark_duplicate_trial(trial)
    if cached is not None:
        score_val = cached if isinstance(cached, (int, float)) else cached[0]
        desc = pbar_state.current.desc if pbar_state.current else "Core Engine"
        logger.debug("  %s: (duplicate, score:%.1f)", desc, score_val)
        return cached

    moe_str = f" moe={config.get('n_cpu_moe', '-')}" if ctx.is_moe else ""
    numa_str = f" numa={config['numa']}" if "numa" in config else ""
    params_short = (
        f"t={config.get('threads')}/{config.get('threads_batch')}, "
        f"b={config.get('batch_size')}, ub={config.get('ubatch_size')}, "
        f"fa={config.get('flash_attn')}, "
        f"poll={config.get('poll')}/{config.get('poll_batch')}, "
        f"cs={config.get('cpu_strict')}/{config.get('cpu_strict_batch')}"
        f"{moe_str}{numa_str}"
    )

    # Note: This objective inlines the boot+gate+measure pattern rather
    # than calling run_server_trial() because it needs to:
    # 1. Capture proc.load_time_ms and attach it to the PerfResult
    # 2. Use record_trial_attrs + finalize_trial (separate building blocks)
    # 3. Return Pareto tuples for multi-objective mode
    proc, status = boot_server_with_jinja_recovery(ctx, config)
    if status == "oom":
        logger.info("Trial %s: pruned (OOM)", trial.number)
        kill_server(ctx)
        raise optuna.exceptions.TrialPruned()
    elif status != "ok":
        server_start_failed(ctx, trial.number, params_short, proc)
        return 0.0 if not params.is_pareto else (0.0, VRAM_FAILURE_PENALTY, 0.0)

    # Multi-fidelity gate
    gate = measure_perf_once(ctx, n_predict=5)
    if gate and best.value > 0:
        gate_score = compute_score(gate)
        trial.report(gate_score, step=0)
        if trial.should_prune():
            logger.info("Trial %s: pruned by gate (%.1f)", trial.number, gate_score)
            raise optuna.exceptions.TrialPruned()

    perf, promoted = measure_perf_adaptive(ctx, best.value)

    # Capture model load time from server stderr
    if proc.load_time_ms is not None:
        perf = dataclasses.replace(perf, load_time_ms=proc.load_time_ms)

    score = params.score_fn(perf)

    trial.report(score, step=1)

    record_trial_attrs(ctx, trial, perf)
    result, best.value = finalize_trial(
        ctx,
        trial,
        perf,
        params_short,
        best.value,
        params.total_trials,
        params.is_pareto,
        score=score,
    )
    update_param_cache(trial, result)
    return result


# ============================================================
# Layer 1: Quick A/B Sweeps
# ============================================================


def _build_ab_flags(skip_flags: set) -> tuple[dict, list]:
    """Build the list of flags to A/B test and their pre-determined winners.

    Returns (winners, ab_flags) where winners contains flags that are
    pre-determined (skipped or hardcoded) and ab_flags contains tuples
    of (flag_name, candidates, default_if_skipped) to test.
    """
    winners = {}
    ab_flags = []

    # Candidate order matters: first value wins ties, so list the
    # simpler/safer default first (off before on, False before True)
    if "op_offload" not in skip_flags:
        ab_flags.append(("op_offload", [False, True], False))
    else:
        winners["op_offload"] = False  # MoE default: OFF

    # no_mmap always ON for faster loading -- skip A/B test
    winners["no_mmap"] = True
    # mlock does nothing when no_mmap=True, so skip it too
    winners["mlock"] = False

    if "repack" not in skip_flags:
        ab_flags.append(("repack", [False, True], False))
    else:
        winners["repack"] = True  # ON by default in llama.cpp

    # prio: test 0 (normal) vs 2 (high) -- the main lever on Windows
    ab_flags.append(("prio", [0, 2], 0))
    ab_flags.append(("prio_batch", [0, 2], 0))

    return winners, ab_flags


def _run_single_ab_test(  # noqa: PLR0913
    ctx,
    base_config,
    winners,
    flag_name,
    candidates,
    default_val,
    score_fn,
    baseline_score=0.0,
):
    """Run A/B test for a single flag.

    Returns a 4-tuple ``(flag_name, best_val, best_perf, best_score)``.
    *best_perf* is ``None`` and *best_score* is ``-1.0`` when no candidate
    produced a valid measurement.

    If neither candidate beats *baseline_score*, *default_val* is kept so
    that noise in a single A/B pair cannot regress overall performance.
    """
    # mlock dependency: if no_mmap already won as True, mlock is a no-op
    if flag_name == "mlock" and winners.get("no_mmap", False):
        logger.debug("mlock: skipped (no_mmap=True makes mlock a no-op)")
        return "mlock", False, None, -1.0

    results = []
    for val in candidates:
        config = {**base_config, **winners, flag_name: val}

        proc, status = boot_server_with_jinja_recovery(ctx, config)
        if status != "ok":
            logger.warning("%s=%s: server failed", flag_name, val)
            kill_server(ctx)
            continue

        perf, _ = measure_perf_adaptive(ctx, runs=2)
        score = score_fn(perf)
        results.append((val, perf, score))

    if not results:
        return flag_name, default_val, None, -1.0

    best_val, best_perf, best_score = max(results, key=lambda r: r[2])

    # If no candidate beats the baseline, keep the default value
    if best_score < baseline_score:
        logger.debug(
            "%s: no candidate beat baseline (%.1f < %.1f), keeping default=%s",
            flag_name,
            best_score,
            baseline_score,
            default_val,
        )
        best_val = default_val

    for val, perf, score in results:
        marker = " ** BEST **" if val == best_val else ""
        logger.info(
            "  A/B Toggles: %.1f t/s | pp: %.0f | TTFT: %.0fms | score: %.1f%s",
            perf.tps,
            perf.prompt_tps,
            perf.ttft,
            score,
            marker,
        )
        logger.info("    %s=%s", flag_name, val)
        logger.info("")

    return flag_name, best_val, best_perf, best_score


def _layer1_ab_sweeps(  # noqa: PLR0913
    ctx: AppContext,
    base_config,
    score_fn,
    baseline=None,
    baseline_score=0.0,
    phase_config: PhaseConfig | None = None,
):
    """Test independent toggle flags with quick A/B comparisons.

    Each flag is tested on vs off (or a small set of values) while holding
    everything else at baseline. Winner is kept. ~2 trials per flag.

    Returns dict of {flag_name: best_value} for all tested flags.

    If *phase_config* is provided, only flags listed in
    ``phase_config.test_flags`` are tested. Flags not in the list are skipped.

    Note: Cannot use run_server_trial() here because Layer 1 has no Optuna
    trial object -- it's a simple A/B sweep outside the study, with no pruning
    gate and only 2 measurement runs per candidate.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("  A/B Toggles")
    logger.info("=" * 60)
    logger.info("")

    if baseline is not None:
        logger.info(
            "  Baseline: %5.1f t/s | pp:%5.0f | TTFT:%4.0fms | score:%5.1f",
            baseline.tps,
            baseline.prompt_tps,
            baseline.ttft,
            baseline_score,
        )
        logger.info("")

    winners, ab_flags = _build_ab_flags(ctx.skip_flags)

    # Filter A/B flags by phase_config.test_flags if provided
    if phase_config is not None and phase_config.test_flags:
        allowed = set(phase_config.test_flags)
        ab_flags = [
            (name, cands, dflt)
            for name, cands, dflt in ab_flags
            if name in allowed
        ]

    if not ab_flags:
        logger.info("No flags to A/B test (all GGUF-skipped)")
        return winners

    overall_best_perf = None
    overall_best_score = baseline_score

    for flag_name, candidates, default_val in ab_flags:
        _, best_val, best_perf, best_score = _run_single_ab_test(
            ctx,
            base_config,
            winners,
            flag_name,
            candidates,
            default_val,
            score_fn,
            baseline_score=baseline_score,
        )
        winners[flag_name] = best_val
        if best_score > overall_best_score:
            overall_best_score = best_score
            overall_best_perf = best_perf

    # Apply mlock dependency: if winner no_mmap=True, force mlock=False
    if winners.get("no_mmap", False) and "mlock" in winners:
        winners["mlock"] = False

    logger.info("")
    logger.info("=" * 60)
    logger.info("  A/B Toggles — RESULTS")
    logger.info("=" * 60)
    logger.info("")
    if baseline is not None:
        logger.info(
            "  Baseline: %.1f t/s | pp: %.0f | TTFT: %.0fms | score: %.1f",
            baseline.tps,
            baseline.prompt_tps,
            baseline.ttft,
            baseline_score,
        )
    if overall_best_perf:
        logger.info(
            "  Optimal:  %.1f t/s | pp: %.0f | TTFT: %.0fms | score: %.1f",
            overall_best_perf.tps,
            overall_best_perf.prompt_tps,
            overall_best_perf.ttft,
            overall_best_score,
        )
    logger.info("")
    logger.info("  Params:    %s", ", ".join(f"{k}={v}" for k, v in winners.items()))
    return winners


# ============================================================
# Layer 2: Focused TPE Co-optimization
# ============================================================


def _setup_layer2_study(
    ctx: AppContext,
    n_trials: int,
    is_pareto: bool,
) -> tuple[optuna.Study, int, int]:
    """Create or resume the Layer 2 Optuna study with TPE sampler."""
    n_startup = 25 if ctx.is_moe else 10
    sampler = optuna.samplers.TPESampler(
        multivariate=True,
        n_startup_trials=n_startup,
        warn_independent_sampling=False,
    )
    pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.1)
    return setup_study(
        ctx,
        "core_engine",
        n_trials,
        sampler_override=sampler,
        pruner=pruner,
        is_pareto=is_pareto,
    )


def _measure_layer2_baseline(
    ctx: AppContext,
    layer2_base: dict,
    score_fn: Callable[..., float],
) -> float:
    """Measure performance baseline with Layer 1 winners applied.

    Returns the baseline score, or 0.0 on failure.
    """
    logger.debug("Layer 2 baseline (with Layer 1 winners)...")
    _proc, status = boot_server_with_jinja_recovery(ctx, layer2_base)
    if status != "ok":
        logger.warning("Baseline server failed -- starting from scratch")
        return 0.0
    layer2_baseline, _ = measure_perf_adaptive(ctx, runs=3)
    baseline_score = score_fn(layer2_baseline)
    logger.info(
        "  Baseline: %.1f t/s | pp: %.0f | TTFT: %.0fms | score: %.1f",
        layer2_baseline.tps,
        layer2_baseline.prompt_tps,
        layer2_baseline.ttft,
        baseline_score,
    )
    logger.info("")
    return baseline_score


def _compute_search_space_options(
    ctx: AppContext,
    ctx_size: int,
) -> tuple[list[int], list[int], list[int]]:
    """Compute thread, batch, and ubatch options for Layer 2 search space.

    Returns (thread_opts, batch_opts, ubatch_opts).
    """
    all_batch_opts = [256, 512, 1024, 2048, 4096]
    if ctx.vram_total_mb and ctx.model_size_gb:
        vram_for_kv_mb = ctx.vram_total_mb - (ctx.model_size_gb * 1024) - 512
        if vram_for_kv_mb > 0:
            max_batch = (
                int(vram_for_kv_mb * 1024 / (ctx_size * 0.5)) if ctx_size > 0 else 4096
            )
            max_batch = max(256, min(max_batch, 4096))
            all_batch_opts = [v for v in all_batch_opts if v <= max_batch]
    batch_opts = sorted(
        [v for v in all_batch_opts if v <= ctx_size] or [min(all_batch_opts)]
    )
    ubatch_opts = [128, 256, 512, 1024]
    thread_opts = sorted(
        set(list(range(2, ctx.max_threads + 1, 2)) + [ctx.max_threads])
    )
    return thread_opts, batch_opts, ubatch_opts


def _layer2_tpe_search(  # noqa: PLR0913
    ctx: AppContext,
    n_trials,
    base_config,
    layer1_winners,
    score_fn,
    is_pareto,
    phase_config: PhaseConfig | None = None,
):
    """Multivariate TPE search over correlated parameters.

    Uses Layer 1 winners as fixed context. Searches only the params
    that have strong cross-correlations: threads, batch, flash_attn,
    poll, cpu_strict, n_cpu_moe.

    If *phase_config* is provided, only params in ``phase_config.search_params``
    are searched. Params in ``phase_config.lock`` are fixed to their locked values.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Core Engine")
    logger.info("=" * 60)
    logger.info("")

    _skip = ctx.skip_flags
    ctx_size = ctx.naked_engine.get("context", DEFAULT_CONTEXT_SIZE)

    study, remaining, completed = _setup_layer2_study(ctx, n_trials, is_pareto)
    if remaining == 0:
        best = get_best_trial(ctx, study)
        bv = trial_scalar_value(best) or 0
        logger.info("Best Score: %.1f | TPS: %.1f", bv, best.user_attrs.get("tps", 0))
        print_param_importance(study)
        clear_param_cache(study.study_name)
        return best.params, study

    layer2_base = {**base_config, **layer1_winners}

    layer2_baseline_score = 0.0
    if completed == 0:
        layer2_baseline_score = _measure_layer2_baseline(ctx, layer2_base, score_fn)

    total_trials = completed + remaining
    best_score = max(
        layer2_baseline_score if completed == 0 else 0,
        recover_best_score(study, score_fn) if completed > 0 else 0,
    )

    thread_opts, batch_opts, ubatch_opts = _compute_search_space_options(ctx, ctx_size)

    # Build search_params and lock from phase_config if provided
    _cfg_search = None
    _cfg_lock: dict = {}
    if phase_config is not None:
        if phase_config.search_params:
            _cfg_search = frozenset(phase_config.search_params)
        if phase_config.lock:
            _cfg_lock = dict(phase_config.lock)

    obj_params = _ObjectiveParams(
        ctx=ctx,
        layer2_base=layer2_base,
        score_fn=score_fn,
        is_pareto=is_pareto,
        total_trials=total_trials,
        thread_opts=thread_opts,
        batch_opts=batch_opts,
        ubatch_opts=ubatch_opts,
        skip_flags=_skip,
        search_params=_cfg_search,
        lock=_cfg_lock,
    )
    best_tracker = BestScoreTracker(best_score)

    def objective(trial):
        return _layer2_objective(trial, obj_params, best_tracker)

    logger.info("  Running %s trials...", remaining)
    dims = (
        7
        + (1 if ctx.is_moe else 0)
        + (1 if ctx.numa_nodes > 1 and "numa" not in _skip else 0)
    )
    logger.debug("Sampler: Multivariate TPE (%sD search space)", dims)
    run_study_with_callbacks(
        ctx, study, objective, remaining, "Core Engine", best_score, is_pareto
    )

    return None, study  # params extracted by caller from study


# ============================================================
# Main Entry Point
# ============================================================


def phase_core_engine(  # noqa: C901
    ctx: AppContext,
    n_trials: int = 100,
    base_config: EngineConfig | None = None,
    phase_config: PhaseConfig | None = None,
) -> PhaseReturnDict | None:
    """Phase 2: 2-layer optimization of all engine parameters.

    Layer 1: Quick A/B sweeps for independent toggles (~8-12 trials)
    Layer 2: Focused TPE for correlated params (remaining trials)

    This replaces the old single-pass approach that threw all 16+ params
    into one TPE search. The layered approach reduces the TPE search space
    from ~16D to ~7-9D, dramatically improving convergence.

    Args:
        base_config: Upstream config from prior phases (MoE sweep, KV sweep).
                     If None, falls back to ctx.naked_engine.
    """
    phase_start_time = time.time()
    is_pareto = ctx.config.get("pareto", False)

    score_fn = compute_score

    # Baseline — apply GGUF-skipped defaults so baseline runs in the same
    # environment as trials (e.g., op_offload=False for MoE)
    _skip = ctx.skip_flags
    gguf_defaults = {}
    if "op_offload" in _skip:
        gguf_defaults["op_offload"] = False
    if "no_mmap" in _skip:
        gguf_defaults["no_mmap"] = False
    if "swa_full" in _skip:
        gguf_defaults["swa_full"] = False
    if "repack" in _skip:
        gguf_defaults["repack"] = True  # ON by default in llama.cpp
    _upstream = base_config if base_config is not None else dict(ctx.naked_engine)
    base_config = {
        **_upstream,
        "flash_attn": "on",
        "kv_cache_type": "f16",
        "no_mmap": True,
        **gguf_defaults,
    }

    logger.debug("Starting baseline server...")
    kill_server(ctx)
    proc, status = boot_server_with_jinja_recovery(ctx, base_config)
    if status != "ok":
        logger.warning("Baseline server failed to start")
        if ctx.fail_fast:
            raise BaselineFailure("Baseline server failed in Core Engine.")
        return None
    baseline, _ = measure_perf_adaptive(ctx, runs=3)
    baseline_score = score_fn(baseline)

    # Use phase_config.trials if provided, otherwise use n_trials
    effective_trials = n_trials
    if phase_config is not None and phase_config.trials is not None:
        effective_trials = phase_config.trials

    # Layer 1: Quick A/B sweeps (~8-12 trials)
    layer1_winners = _layer1_ab_sweeps(
        ctx, base_config, score_fn, baseline, baseline_score, phase_config=phase_config
    )

    # Layer 2: Focused TPE (remaining trials allocated to correlated params)
    # Reserve ~12 trials for Layer 1, rest goes to Layer 2
    layer2_trials = max(40, effective_trials - 12)
    _, study = _layer2_tpe_search(
        ctx,
        layer2_trials,
        base_config,
        layer1_winners,
        score_fn,
        is_pareto,
        phase_config=phase_config,
    )

    # Combine results: Layer 1 winners + Layer 2 best params
    param_keys = [
        "threads",
        "threads_batch",
        "batch_size",
        "ubatch_size",
        "flash_attn",
        "poll",
        "poll_batch",
        "cpu_strict",
        "cpu_strict_batch",
        "n_cpu_moe",
        # Layer 1 keys (stored in study as fixed attrs)
        "mlock",
        "no_mmap",
        "repack",
        "op_offload",
        "prio",
        "prio_batch",
    ]
    summary_ctx = PhaseSummaryContext(
        phase_name="core_engine",
        study=study,
        baseline=baseline,
        baseline_score=baseline_score,
        phase_start_time=phase_start_time,
        is_pareto=is_pareto,
        score_fn=score_fn,
        param_keys=param_keys,
    )
    returned_params, _ = print_phase_summary(ctx, summary_ctx)
    clear_param_cache(study.study_name)

    # Merge Layer 1 winners into returned params
    if returned_params is None:
        returned_params = {}
    for k, v in layer1_winners.items():
        if k not in returned_params:
            returned_params[k] = v

    return PhaseReturnDict(best_params=returned_params, phase_name="core_engine")
