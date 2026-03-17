"""
Shared trial execution helpers for Optuna-based optimization phases.

Reduces ~40-50 lines of duplicated boilerplate per phase by extracting the
common patterns: server boot, measurement, attribute recording, result printing,
and post-study summary.
"""

import time

import optuna

from .state import ctx, _config
from .hardware import _get_vram_used_mb, check_thermal_throttle, wait_for_cooldown
from .engine import (
    start_server, wait_for_server, kill_server,
    _boot_server_with_jinja_recovery, _server_start_failed,
    run_bench_trial, BaselineFailure,
)
from .measurement import (
    compute_score, compute_pareto_objectives, extract_pareto_front,
    print_pareto_front, get_best_trial, get_best_value,
    measure_perf, measure_perf_adaptive, measure_perf_quick_gate,
)
from .search import (
    _trial_scalar_value, TqdmUpdateCallback, GPStoppingCallback,
    check_duplicate_trial, setup_study, save_phase_results,
    create_phase_pbar, close_phase_pbar,
    print_trial_result, print_param_importance,
)


def thermal_gate():
    """Check GPU thermals and wait for cooldown if needed."""
    if check_thermal_throttle(threshold=85)[0]:
        wait_for_cooldown(target_temp=75, timeout=120)


def run_server_trial(trial, config, params_short, best_score, is_pareto=False):
    """Execute one server-based trial: boot, gate, measure, record attrs.

    Returns (perf_dict, score) on success, or raises TrialPruned / returns
    (None, 0.0) on failure.
    """
    print(f"\n  Trial {trial.number}: server | {params_short}")
    kill_server()
    proc = start_server(config)
    status = wait_for_server(proc=proc)
    if status == "oom":
        print(f"  Trial {trial.number}: pruned (OOM)")
        kill_server()
        raise optuna.exceptions.TrialPruned()
    elif status != "ok":
        _server_start_failed(trial.number, params_short, proc)
        return None, 0.0

    # Multi-fidelity gate
    gate = measure_perf_quick_gate(n_predict=5)
    if gate and best_score > 0:
        gate_score = gate.get("gate_score", 0)
        trial.report(gate_score, step=0)
        if trial.should_prune():
            print(f"  Trial {trial.number}: pruned by gate ({gate_score:.1f})")
            raise optuna.exceptions.TrialPruned()

    perf, promoted = measure_perf_adaptive(best_score)
    score = compute_score(perf)
    trial.report(score, step=1)

    return perf, score


def record_trial_attrs(trial, perf):
    """Record standard performance attributes on a trial."""
    if perf.get("tps_std") is not None:
        trial.set_user_attr("tps_std", perf["tps_std"])
        trial.set_user_attr("tps_cv", perf.get("tps_cv", 0))

    vram_mb = _get_vram_used_mb()
    if vram_mb is not None:
        trial.set_user_attr("vram_used_mb", vram_mb)
        perf["vram_used_mb"] = vram_mb

    trial.set_user_attr("tps", perf.get("tps", 0))
    trial.set_user_attr("ttft", perf.get("ttft", 0))
    trial.set_user_attr("prompt_tps", perf.get("prompt_tps", 0))
    trial.set_user_attr("total_ms", perf.get("total_ms", 0))


def finalize_trial(trial, perf, params_short, best_score, total_trials, is_pareto=False, score=None):
    """Print result and return the appropriate value (scalar or pareto tuple).

    Returns (return_value, new_best_score).
    """
    tps = perf.get("tps", 0)
    if score is None:
        score = compute_score(perf)

    if not is_pareto:
        new_best = print_trial_result(trial.number, total_trials, tps, perf, params_short, best_score, final_score=score)
        return score, new_best
    else:
        vram_mb = perf.get("vram_used_mb")
        quality = perf.get("quality_factor", 1.0)
        objectives = compute_pareto_objectives(perf, quality_factor=quality)
        print(f"  Trial {trial.number}: TPS={tps:.1f} VRAM={vram_mb or 0:.0f}MB | {params_short}")
        return objectives, best_score


def recover_best_score(study, score_fn):
    """Recover the best score from completed trials in a resumed study."""
    best = 0.0
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs:
            perf = {k: t.user_attrs.get(k, 0) for k in ["tps", "ttft", "prompt_tps", "total_ms"]}
            if perf["tps"] > 0:
                best = max(best, score_fn(perf))
    return best


def run_study_with_callbacks(study, objective, remaining, label, best_score, is_pareto=False):
    """Run study.optimize with standard callbacks (progress bar + GP stopping)."""
    pbar = create_phase_pbar(remaining, desc=label)
    callbacks = [TqdmUpdateCallback()]
    if not is_pareto:
        callbacks.append(GPStoppingCallback(baseline_score=best_score))
    study.optimize(objective, n_trials=remaining, callbacks=callbacks, show_progress_bar=False)
    close_phase_pbar()


def print_phase_summary(phase_name, study, baseline, baseline_score, phase_start_time,
                        is_pareto=False, score_fn=None, param_keys=None):
    """Print standard phase results and save to disk.

    Returns (best_params_dict, results_dict).
    """
    if score_fn is None:
        score_fn = compute_score

    best = get_best_trial(study)
    phase_elapsed = time.time() - phase_start_time

    print(f"\n{'=' * 60}")
    print(f"  {phase_name} — RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline:    {baseline['tps']:.1f} t/s | Score: {baseline_score:.1f}")

    if is_pareto:
        pareto = extract_pareto_front(study)
        print(f"\n  Pareto Front: {len(pareto)} configs")
        print_pareto_front(pareto)
        returned_params = best.params
    else:
        bv = _trial_scalar_value(best) or 0
        beat_baseline = bv > baseline_score
        if beat_baseline:
            print(f"  Best Score:  {bv:.1f} — beats baseline by {bv - baseline_score:.1f}")
            returned_params = best.params
        else:
            print(f"  Best Score:  {bv:.1f} — below baseline ({baseline_score:.1f})")
            if param_keys:
                returned_params = {k: baseline.get(k) for k in param_keys if k in baseline}
            else:
                returned_params = best.params

    print(f"  Best TPS:    {best.user_attrs.get('tps', 0):.1f} t/s")
    print(f"  Best params:")
    for k, v in returned_params.items():
        print(f"    {k}: {v}")
    importances = print_param_importance(study)

    print(f"\n  Duration:    {phase_elapsed / 60:.1f} min")

    results = {
        "phase": phase_name,
        "baseline": baseline,
        "baseline_score": baseline_score,
        "beat_baseline": (_trial_scalar_value(best) or 0) > baseline_score if not is_pareto else True,
        "best_tps": best.user_attrs.get("tps", 0),
        "best_metrics": best.user_attrs,
        "best_params": returned_params,
        "param_importance": {k: round(v * 100, 1) for k, v in importances.items()},
        "duration_minutes": round(phase_elapsed / 60, 1),
        "all_trials": [
            {"number": t.number, "tps": _trial_scalar_value(t), "metrics": t.user_attrs, "params": t.params}
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ],
    }
    save_phase_results(phase_name, results)
    return returned_params, results


def setup_baseline_server(base_config, phase_name):
    """Boot a baseline server and measure initial performance.

    Returns (baseline_perf, baseline_score) or raises BaselineFailure.
    """
    print("\n[*] Starting baseline server...")
    kill_server()
    proc, status = _boot_server_with_jinja_recovery(base_config)
    if status != "ok":
        print(f"[!] Baseline failed in {phase_name}")
        if ctx.fail_fast:
            raise BaselineFailure(f"Baseline server failed in {phase_name}.")
        return None, 0.0
    baseline = measure_perf(runs=3)
    baseline_score = compute_score(baseline)
    print(f"    Baseline: {baseline['tps']:.1f} t/s | pp: {baseline['prompt_tps']:.0f} t/s | "
          f"TTFT: {baseline['ttft']:.0f}ms | Score: {baseline_score:.1f}")
    return baseline, baseline_score
