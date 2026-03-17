"""
Optimization phases and pipeline orchestrator.

Each phase_* function runs an Optuna study over a specific parameter group.
batch_optimize() and run_full_pipeline() orchestrate the full optimization pipeline.
"""

import random
import time
from datetime import datetime
from pathlib import Path

import optuna

from .constants import (
    SCORE_VERSION, SCORE_TTFT_BASELINE, SCORE_PP_BASELINE,
    QUALITY_WEIGHT_CORRECTNESS, QUALITY_WEIGHT_CONFIDENCE, QUALITY_WEIGHT_EFFICIENCY,
    QUALITY_TASKS,
    ADAPTIVE_THRESHOLD, CV_TARGET, CV_MIN_RUNS, CV_MAX_RUNS,
    PPL_DEGRADATION_WARN, PPL_DEGRADATION_FAIL,
    _NIAH_NEEDLES, _NIAH_FILLER_BLOCKS,
)
from .state import ctx, _config, get_preset_trials
from .models import classify_model, detect_model_layers, _detect_gguf_architecture
from .hardware import (
    detect_gpus, _get_vram_used_mb, kill_competing_processes,
    check_thermal_throttle, wait_for_cooldown,
)
from .engine import (
    check_dry_run, generate_tensor_splits, BaselineFailure, PhaseTimer,
    _build_bench_cmd, _parse_bench_csv, run_bench_trial,
    start_server, wait_for_server, warmup_server, kill_server, is_server_running,
    _boot_server_with_jinja_recovery, _server_start_failed,
)
from .measurement import (
    compute_score, compute_pareto_objectives, extract_pareto_front,
    print_pareto_front, get_best_trial, get_best_value,
    measure_perf, measure_perf_adaptive, measure_perf_quick_gate,
    measure_concurrent_load, measure_token_uncertainty,
    _get_actual_ctx,
)
from .search import (
    _trial_scalar_value, TqdmUpdateCallback, GPStoppingCallback,
    check_duplicate_trial, setup_study, save_phase_results, load_phase_results,
    _safe_best_value, create_phase_pbar, close_phase_pbar,
    print_trial_result, print_param_importance, ensure_results_dir,
)
from .evals import (
    measure_quality_gate, measure_kl_divergence, kl_quality_factor,
    measure_true_perplexity, ppl_quality_factor,
    measure_quality, niah_test, phase_niah,
    phase_reasoning_eval, phase_integrity_eval,
    _tokenize_count, _build_niah_prompt,
)

# Optional: aiohttp for concurrent quality evals
try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

# ============================================================
# Helpers
# ============================================================


def _bench_score(p, baseline=None):
    """Bench scoring: gen_tps with prompt_tps as a relative tiebreaker.

    llama-bench prompt_tps/ttft are not comparable to HTTP in absolute terms,
    but the RELATIVE differences between configs are real (faster bench pp =
    faster server pp). We use prompt_tps normalized against the bench baseline
    as a bounded tiebreaker so two configs with equal gen_tps can be separated
    by their prompt processing speed.

    Score ≈ gen_tps * (0.85 + 0.15 * pp_ratio), capped at 2x baseline pp.
    Without a baseline, falls back to gen_tps only.
    """
    tps = p.get("tps", 0)
    if tps <= 0:
        return 0.0
    pp = p.get("prompt_tps", 0)
    if baseline and baseline.get("prompt_tps", 0) > 0 and pp > 0:
        pp_ratio = min(pp / baseline["prompt_tps"], 2.0)
    else:
        pp_ratio = 1.0
    return tps * (0.85 + 0.15 * pp_ratio)


def _get_moe_config(p1a_results=None):
    """Extract MoE config dict from MoE phase results, with defaults."""
    if not ctx.is_moe:
        return {}
    if p1a_results and "best_params" in p1a_results:
        bp = p1a_results["best_params"]
        return {
            "n_cpu_moe": bp.get("n_cpu_moe", ctx.moe_sweep_center),
            "expert_used_count": bp.get("expert_used_count", ctx.default_experts),
        }
    moe_data = load_phase_results("moe")
    if moe_data and "best_params" in moe_data:
        moe_threads = moe_data["best_params"].get("n_cpu_moe", ctx.moe_sweep_center)
        print(f"  [*] Using MoE threads from MoE sweep: {moe_threads} (expert sweep was skipped)")
        return {"n_cpu_moe": moe_threads, "expert_used_count": ctx.default_experts}
    return {"n_cpu_moe": ctx.moe_sweep_center, "expert_used_count": ctx.default_experts}


# ============================================================
# GPU Offload Phase
# ============================================================

def phase_gpu_offload():
    """GPU Offload: Find optimal GPU layer offload.

    For MoE models: skips entirely, locks to ctx.max_gpu_layers (all on GPU).
    MoE models use n_cpu_moe for smart CPU offloading in the MoE phase instead.

    For dense models: sweeps n_gpu_layers using middle-out approach with
    adaptive measurement and per-direction early stopping.

    Updates ctx.naked_engine and ctx.default_gpu_layers for all subsequent phases.
    Returns int: best n_gpu_layers value.
    """

    label = "GPU Offload"
    max_ngl = ctx.max_gpu_layers

    # Check for existing results
    existing = load_phase_results("gpu")
    if existing and "best_ngl" in existing:
        best_ngl = existing["best_ngl"]
        print(f"\n[*] GPU Offload already complete — n_gpu_layers={best_ngl} (from previous run)")
        ctx.default_gpu_layers = best_ngl
        ctx.naked_engine["n_gpu_layers"] = best_ngl
        return best_ngl

    # MoE models: always full GPU offload — MoE phase handles smart CPU offloading
    if ctx.is_moe:
        print(f"\n[*] MoE model — all {max_ngl} layers on GPU (MoE phase handles CPU offloading)")
        ctx.default_gpu_layers = max_ngl
        ctx.naked_engine["n_gpu_layers"] = max_ngl
        save_phase_results("gpu", {"phase": "gpu", "best_ngl": max_ngl, "skipped": "moe"})
        return max_ngl

    # Skip if max_gpu_layers is 0 or 1 — nothing to sweep
    if max_ngl <= 1:
        print(f"\n[*] Model has {max_ngl} layers — skipping GPU offload sweep.")
        save_phase_results("gpu", {"phase": "gpu", "best_ngl": max_ngl})
        return max_ngl

    # Binary search for OOM boundary, then score sweep above it.
    # GPU offload is monotonic: more layers on GPU = faster until OOM.
    # Step 1: Bisect to find the exact OOM boundary in O(log N) restarts.
    # Step 2: Score the working range above the boundary.

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)

    use_bench = ctx.bench_path is not None
    if use_bench:
        print(f"    [bench] Using llama-bench for fast GPU sweep\n")

    # --- Step 1: Binary search for OOM boundary ---
    print(f"\n[*] Binary search: finding OOM boundary in {max_ngl} layers...")

    def _test_ngl(ngl):
        """Returns True if ngl value works (no OOM), False otherwise."""
        config = {**ctx.naked_engine, "n_gpu_layers": ngl}
        if use_bench:
            perf = run_bench_trial(config, repetitions=1)
            return perf is not None and perf.get("error") != "oom"
        else:
            kill_server()
            proc = start_server(config)
            ok = wait_for_server(proc=proc) == "ok"
            kill_server()
            return ok

    # First check if max works — common case (small model / big GPU)
    if _test_ngl(max_ngl):
        oom_boundary = max_ngl  # no OOM at max
        print(f"    Max layers ({max_ngl}) fits in VRAM — no OOM boundary")
    else:
        # Binary search: find highest ngl that works
        lo, hi = 0, max_ngl
        # Quick check: does 0 work at all?
        if not _test_ngl(0):
            print("[!] Even 0 GPU layers fails — model may be too large for system RAM")
            ctx.default_gpu_layers = 0
            ctx.naked_engine["n_gpu_layers"] = 0
            save_phase_results("gpu", {"best_ngl": 0, "reason": "all_failed"})
            return 0

        bisect_steps = 0
        while lo < hi - 1:
            mid = (lo + hi) // 2
            bisect_steps += 1
            works = _test_ngl(mid)
            status = "OK" if works else "OOM"
            print(f"    Bisect [{bisect_steps}]: ngl={mid} → {status}  (range: {lo}..{hi})")
            if works:
                lo = mid
            else:
                hi = mid

        oom_boundary = lo
        # Safety margin: verify the boundary works, drop down if it doesn't
        for retry in range(3):
            if _test_ngl(oom_boundary):
                break
            oom_boundary = max(0, oom_boundary - 1)
            print(f"    Safety margin: dropped to ngl={oom_boundary}")
        print(f"    OOM boundary: ngl={oom_boundary} (found in {bisect_steps} bisections vs {max_ngl - oom_boundary} linear)")

    # --- Step 2: Smart score sweep ---
    # If max layers fits, quick 2-point check. Only full sweep near OOM boundary.

    def _score_ngl(ngl):
        """Measure score for a given ngl. Returns (perf, score) or (None, 0)."""
        config = {**ctx.naked_engine, "n_gpu_layers": ngl}
        if use_bench:
            perf = run_bench_trial(config, repetitions=3)
            if perf is None or perf.get("error") == "oom":
                return None, 0.0
        else:
            kill_server()
            proc = start_server(config)
            if wait_for_server(proc=proc) != "ok":
                kill_server()
                return None, 0.0
            perf = measure_perf(runs=3)
            kill_server()
        return perf, _bench_score(perf) if use_bench else compute_score(perf)

    results = []
    best_score = 0.0
    best_ngl = oom_boundary

    if oom_boundary == max_ngl:
        # Model fits entirely — sweep across the full range in even steps
        n_points = min(16, max_ngl + 1)
        step = max(1, max_ngl // (n_points - 1)) if n_points > 1 else 1
        checkpoints = sorted(set([max_ngl] + [max(0, max_ngl - i * step) for i in range(n_points)]), reverse=True)
        # Remove duplicates for small layer counts
        checkpoints = list(dict.fromkeys(checkpoints))

        print(f"\n[*] GPU sweep: {len(checkpoints)} points across {max_ngl} layers")

        for i, ngl in enumerate(checkpoints):
            if check_thermal_throttle(threshold=85)[0]:
                wait_for_cooldown(target_temp=75, timeout=120)
            perf, score = _score_ngl(ngl)
            if not perf:
                print(f"  [{i+1}] ngl={ngl:3d}: FAILED")
                continue
            results.append({"ngl": ngl, "perf": perf, "score": score, "promoted": True})
            marker = ""
            if score > best_score:
                best_score = score
                best_ngl = ngl
                marker = " *NEW BEST*"
            print(f"  [{i+1}] ngl={ngl:3d}: {perf['tps']:.1f} t/s | "
                  f"pp: {perf['prompt_tps']:.0f} t/s | TTFT: {perf['ttft']:.0f}ms | "
                  f"Score: {score:.1f}{marker}")

            # Early stop: if we're already at <50% of best, lower points won't help
            if best_score > 0 and score < best_score * 0.50 and len(results) >= 3:
                print(f"    Score below 50% of best — stopping early")
                break
    else:
        # OOM boundary below max — sweep from boundary down, stop at 90% drop
        sweep_floor = max(0, oom_boundary - 8)
        print(f"\n[*] Scoring sweep: ngl={oom_boundary}→{sweep_floor}")

        for ngl in range(oom_boundary, sweep_floor - 1, -1):
            if check_thermal_throttle(threshold=85)[0]:
                wait_for_cooldown(target_temp=75, timeout=120)
            perf, score = _score_ngl(ngl)
            if not perf:
                print(f"  ngl={ngl:3d}: FAILED")
                continue
            results.append({"ngl": ngl, "perf": perf, "score": score, "promoted": True})
            marker = ""
            if score > best_score:
                best_score = score
                best_ngl = ngl
                marker = " *NEW BEST*"
            print(f"  ngl={ngl:3d}: {perf['tps']:.1f} t/s | "
                  f"pp: {perf['prompt_tps']:.0f} t/s | TTFT: {perf['ttft']:.0f}ms | "
                  f"Score: {score:.1f}{marker}")
            if best_score > 0 and score < best_score * 0.90 and len(results) > 2:
                print(f"    Score dropped below 90% of best — stopping sweep")
                break

    if not results:
        print("[!] All offload levels failed. Using default.")
        return ctx.default_gpu_layers

    # Show ranking (top 10)
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    show_n = min(10, len(sorted_results))
    print(f"\n  {'ngl':>5s}  {'t/s':>7s}  {'pp':>7s}  {'TTFT':>7s}  {'Score':>7s}")
    print("  " + "-" * 42)
    for r in sorted_results[:show_n]:
        marker = " <<<" if r["ngl"] == best_ngl else ""
        print(f"  {r['ngl']:5d}  {r['perf']['tps']:7.1f}  {r['perf']['prompt_tps']:7.0f}  "
              f"{r['perf']['ttft']:7.0f}  {r['score']:7.1f}{marker}")

    print(f"\n  Winner: n_gpu_layers={best_ngl} (score {best_score:.1f})")

    # Update globals for all subsequent phases
    ctx.default_gpu_layers = best_ngl
    ctx.naked_engine["n_gpu_layers"] = ctx.default_gpu_layers

    save_phase_results("gpu", {
        "phase": "gpu",
        "best_ngl": best_ngl,
        "best_score": best_score,
        "all_results": [{"ngl": r["ngl"], "tps": r["perf"]["tps"], "score": r["score"]} for r in results],
    })

    return best_ngl


# ============================================================
# MoE Thread Sweep
# ============================================================

def phase_moe_threads(n_trials=40, base_memory_config=None):
    """MoE Thread Sweep: Sweep n_cpu_moe to find optimal MoE thread count.

    Sequential sweep 0-40 with adaptive measurement:
    - Pass 1: test every value with 1 run (bad configs filtered fast)
    - Automatically promotes competitive configs to 3 runs via adaptive measurement

    Returns int (best n_cpu_moe) or None on failure.
    """
    # Check for existing results
    existing = load_phase_results("moe")
    if existing and "best_params" in existing:
        best_moe = existing["best_params"]["n_cpu_moe"]
        print(f"\n[*] MoE thread sweep already complete — n_cpu_moe={best_moe} (from previous run)")
        return best_moe

    phase_start_time = time.time()
    label = "MoE Thread Sweep"
    # Middle-out sweep: each direction stops independently at 50% of best
    up_range = list(range(ctx.moe_sweep_center, ctx.moe_sweep_max + 1))
    down_range = list(range(ctx.moe_sweep_center - 1, -1, -1))

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)

    base_config = {**ctx.naked_engine}
    if base_memory_config:
        base_config.update(base_memory_config)

    # MoE sweep needs --override-kv which bench can't do
    use_bench = ctx.bench_path is not None and not ctx.is_moe

    # Baseline measurement
    if use_bench:
        print("\n[*] Running baseline via llama-bench...")
        baseline = run_bench_trial(base_config, repetitions=3)
        if not baseline or baseline.get("error"):
            print("[!] Baseline bench failed, falling back to HTTP server...")
            use_bench = False  # fall back for entire phase
    if not use_bench:
        print("\n[*] Starting baseline server...")
        kill_server()
        proc = start_server(base_config)
        if wait_for_server(proc=proc) != "ok":
            print("[!] Baseline server failed to start")
            if ctx.fail_fast:
                raise BaselineFailure("Baseline server failed in MoE Thread Sweep.")
            return None
        baseline = measure_perf(runs=3)

    score_fn = (lambda p: _bench_score(p, baseline)) if use_bench else compute_score
    print(f"    Baseline: {baseline['tps']:.1f} t/s | pp: {baseline['prompt_tps']:.0f} t/s | "
          f"TTFT: {baseline['ttft']:.0f}ms | Score: {score_fn(baseline):.1f}")
    if use_bench:
        print(f"    [bench] Using llama-bench for fast MoE sweep")

    total = len(up_range) + len(down_range)
    best_score = score_fn(baseline)
    best_moe = 0

    print(f"\n[*] Sweeping 0-{ctx.moe_sweep_max} (middle-out from {ctx.moe_sweep_center})")
    print(f"    Each direction stops when score drops below 50% of best\n")

    results_by_val = {}  # moe_val -> {moe, score, perf, promoted}
    trial_num = 0

    def _test_moe(moe_val, force_3runs=False):
        """Test a single MoE value. Returns score."""
        nonlocal best_score, trial_num
        trial_num += 1
        if check_thermal_throttle(threshold=85)[0]:
            wait_for_cooldown(target_temp=75, timeout=120)
        config = {**base_config, "n_cpu_moe": moe_val}
        params_short = f"moe={moe_val}"

        if use_bench:
            reps = 5 if force_3runs else 3
            print(f"\n  Trial {trial_num}: bench (r={reps}) | {params_short}")
            perf = run_bench_trial(config, repetitions=reps)
            if perf is None or perf.get("error"):
                print(f"    FAILED ({'OOM' if perf and perf.get('error') else 'bench error'})")
                results_by_val[moe_val] = {"moe": moe_val, "score": 0.0, "perf": None, "promoted": False}
                return 0.0
            promoted = True
        else:
            label_str = "restarting server..." if not force_3runs else "re-testing (3runs)..."
            print(f"\n  Trial {trial_num}: {label_str} | {params_short}")
            kill_server()
            proc = start_server(config)

            if wait_for_server(proc=proc) != "ok":
                _server_start_failed(trial_num, params_short, proc)
                results_by_val[moe_val] = {"moe": moe_val, "score": 0.0, "perf": None, "promoted": False}
                return 0.0

            if force_3runs:
                perf = measure_perf(runs=3)
                promoted = True
            else:
                perf, promoted = measure_perf_adaptive(best_score)

        tps = perf["tps"]
        score = score_fn(perf)

        results_by_val[moe_val] = {"moe": moe_val, "score": score, "perf": perf, "promoted": promoted}

        runs_label = "bench" if use_bench else ("3runs" if promoted else "1run")
        best_score = print_trial_result(trial_num, total, tps, perf, f"{params_short} ({runs_label})", best_score)
        return score

    # Pass 1: middle-out sweep with directional stopping
    up_stopped = False
    down_stopped = False
    up_idx = 0
    down_idx = 0

    while not (up_stopped and down_stopped):
        if not up_stopped and up_idx < len(up_range):
            score = _test_moe(up_range[up_idx])
            up_idx += 1
            if score >= best_score:
                best_moe = up_range[up_idx - 1]
            if best_score > 0 and score < best_score * 0.50 and up_idx > 2:
                print(f"    ↑ Upward direction stopped (score dropped below 50% of best)")
                up_stopped = True
        else:
            up_stopped = True

        if not down_stopped and down_idx < len(down_range):
            score = _test_moe(down_range[down_idx])
            down_idx += 1
            if score >= best_score:
                best_moe = down_range[down_idx - 1]
            if best_score > 0 and score < best_score * 0.50 and down_idx > 2:
                print(f"    ↓ Downward direction stopped (score dropped below 50% of best)")
                down_stopped = True
        else:
            down_stopped = True

    # Pass 2: re-test best ±2 neighbors with fresh 3 runs (always, even if already promoted)
    best_entry = max(results_by_val.values(), key=lambda x: x["score"])
    best_moe = best_entry["moe"]
    retest_range = 2  # ±2 neighbors
    retests = []
    for offset in range(-retest_range, retest_range + 1):
        neighbor = best_moe + offset
        if neighbor in results_by_val:
            retests.append(neighbor)

    print(f"\n  [*] Re-testing best ±{retest_range} neighbors ({len(retests)} values) with fresh 3 runs...")
    for moe_val in retests:
        _test_moe(moe_val, force_3runs=True)

    # Find final best after all retests
    best_entry = max(results_by_val.values(), key=lambda x: x["score"])
    best_moe = best_entry["moe"]
    all_results = [results_by_val[v] for v in sorted(results_by_val.keys())]
    best_perf = best_entry["perf"] or baseline

    print(f"\n{'=' * 60}")
    print(f"  {label} — RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline:        {baseline['tps']:.1f} t/s | TTFT: {baseline['ttft']:.0f}ms")
    print(f"  Best MoE threads: {best_moe}")
    print(f"  Best Score:      {best_entry['score']:.1f} (composite)")
    print(f"  Best TPS:        {best_perf['tps']:.1f} t/s")
    print(f"  Best TTFT:       {best_perf['ttft']:.0f}ms")

    # Histogram
    max_score = max(r["score"] for r in all_results) if all_results else 0
    bar_max = 30
    print(f"\n  Score by n_cpu_moe:")
    print(f"  {'Value':>6}  {'Score':>7}  {'Runs':>4}  {'':}")
    print(f"  {'─' * 6}  {'─' * 7}  {'─' * 4}  {'─' * bar_max}")
    for r in all_results:
        score = r["score"]
        bar_len = int(score / max_score * bar_max) if max_score > 0 else 0
        bar = "█" * bar_len
        marker = " ◄ best" if r["moe"] == best_moe else ""
        runs = "3" if r["promoted"] else "1"
        print(f"  {r['moe']:>6}  {score:>7.1f}  {runs:>4}  {bar}{marker}")

    phase_elapsed = time.time() - phase_start_time
    print(f"\n  Duration:        {phase_elapsed / 60:.1f} min")

    results = {
        "phase": "moe",
        "baseline": baseline,
        "best_tps": best_entry["score"],
        "best_metrics": {"tps": best_perf["tps"], "ttft": best_perf["ttft"],
                         "prompt_tps": best_perf["prompt_tps"], "total_ms": best_perf["total_ms"]},
        "best_params": {"n_cpu_moe": best_moe},
        "duration_seconds": round(phase_elapsed, 1),
        "all_trials": [
            {"number": i, "tps": r["score"], "metrics": r["perf"], "params": {"n_cpu_moe": r["moe"]}}
            for i, r in enumerate(all_results)
        ],
    }
    save_phase_results("moe", results)

    return best_moe


def phase_experts(n_trials=20, locked_moe_threads=18, base_memory_config=None):
    """Expert Count Sweep: Sweep expert_used_count with perplexity quality gate.

    Sequential sweep 1-16 with adaptive measurement + quality gate.
    MoE threads are locked from MoE Thread Sweep.

    Returns int (best expert_used_count) or 8 (default) on failure.
    """
    # Check for existing results
    existing = load_phase_results("experts")
    if existing and "best_params" in existing:
        best_exp = existing["best_params"]["expert_used_count"]
        print(f"\n[*] Expert sweep already complete — experts={best_exp} (from previous run)")
        return best_exp

    phase_start_time = time.time()
    label = "Expert Count Sweep"
    # Middle-out: each direction stops independently at 50% of best
    up_range = list(range(ctx.default_experts, ctx.max_experts + 1))
    down_range = list(range(ctx.default_experts - 1, 0, -1))

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"\n[*] Locked MoE threads: {locked_moe_threads}")

    base_config = {**ctx.naked_engine, "n_cpu_moe": locked_moe_threads}
    if base_memory_config:
        base_config.update(base_memory_config)

    # Start with default experts (8) to establish baseline
    print(f"\n[*] Starting baseline server (default {ctx.default_experts} experts)...")
    kill_server()
    proc = start_server(base_config)
    if wait_for_server(proc=proc) != "ok":
        print("[!] Baseline server failed to start")
        if ctx.fail_fast:
            raise BaselineFailure("Baseline server failed in Expert Count Sweep.")
        return ctx.default_experts
    baseline = measure_perf(runs=3)
    print(f"    Baseline: {baseline['tps']:.1f} t/s | pp: {baseline['prompt_tps']:.0f} t/s | "
          f"TTFT: {baseline['ttft']:.0f}ms | Score: {compute_score(baseline):.1f}")

    # Establish quality baseline with full experts
    print("[*] Measuring baseline quality (token uncertainty calibration)...")
    baseline_qf = measure_quality_gate(is_baseline=True)

    # Populate KL-divergence baseline for expert count penalty
    if ctx.is_moe:
        print("    Measuring baseline KL-Divergence distribution...")
        ctx.kl_baseline_cache, _ = measure_kl_divergence()

    if ctx.quality_baseline is None:
        print("[!] WARNING: Could not measure baseline quality!")
        print("    The server may not support n_probs / completion_probabilities.")
        print(f"    Falling back to default {ctx.default_experts} experts (no quality gate available).")
        return ctx.default_experts

    total = len(up_range) + len(down_range)
    best_score = compute_score(baseline)
    best_experts = ctx.default_experts
    results_by_val = {}  # expert_count -> result dict
    trial_num = 0

    print(f"\n[*] Sweeping 1-{ctx.max_experts} (middle-out from {ctx.default_experts})")
    print(f"    Each direction stops when score drops below 50% of best\n")

    def _test_expert(expert_count, force_3runs=False):
        """Test a single expert count. Returns score."""
        nonlocal best_score, trial_num
        trial_num += 1
        if check_thermal_throttle(threshold=85)[0]:
            wait_for_cooldown(target_temp=75, timeout=120)
        config = {**base_config, "expert_used_count": expert_count}
        params_short = f"experts={expert_count}"

        lbl = "restarting server..." if not force_3runs else "re-testing (3runs)..."
        print(f"\n  Trial {trial_num}: {lbl} | {params_short}")
        kill_server()
        proc = start_server(config)

        if wait_for_server(proc=proc) != "ok":
            _server_start_failed(trial_num, params_short, proc)
            results_by_val[expert_count] = {"experts": expert_count, "score": 0.0, "speed_score": 0.0,
                                            "perf": None, "quality_factor": 0.0, "promoted": False}
            return 0.0

        if force_3runs:
            perf = measure_perf(runs=3)
            promoted = True
        else:
            perf, promoted = measure_perf_adaptive(best_score)
        tps = perf["tps"]
        speed_score = compute_score(perf)

        # Always measure quality in expert phase — that's the whole point
        quality_factor = measure_quality_gate()
        score = speed_score * quality_factor

        results_by_val[expert_count] = {"experts": expert_count, "score": score, "speed_score": speed_score,
                                        "perf": perf, "quality_factor": quality_factor, "promoted": promoted}

        qf_label = f" q={quality_factor:.2f}" if quality_factor < 1.0 else ""
        runs_label = "3runs" if promoted else "1run"
        best_score = print_trial_result(trial_num, total, tps, perf, f"{params_short} ({runs_label}){qf_label}",
                                        best_score, final_score=score)
        return score

    # Pass 1: middle-out sweep with directional stopping
    up_stopped = False
    down_stopped = False
    up_idx = 0
    down_idx = 0

    while not (up_stopped and down_stopped):
        if not up_stopped and up_idx < len(up_range):
            score = _test_expert(up_range[up_idx])
            up_idx += 1
            if score >= best_score:
                best_experts = up_range[up_idx - 1]
            if best_score > 0 and score < best_score * 0.50 and up_idx > 2:
                print(f"    ↑ Upward direction stopped (score dropped below 50% of best)")
                up_stopped = True
        else:
            up_stopped = True

        if not down_stopped and down_idx < len(down_range):
            score = _test_expert(down_range[down_idx])
            down_idx += 1
            if score >= best_score:
                best_experts = down_range[down_idx - 1]
            if best_score > 0 and score < best_score * 0.50 and down_idx > 2:
                print(f"    ↓ Downward direction stopped (score dropped below 50% of best)")
                down_stopped = True
        else:
            down_stopped = True

    # Pass 2: re-test neighbors of the best with 3 runs if they only got 1
    best_entry = max(results_by_val.values(), key=lambda x: x["score"])
    best_experts = best_entry["experts"]
    retest_range = 2  # ±2 neighbors
    retests = []
    for offset in range(-retest_range, retest_range + 1):
        neighbor = best_experts + offset
        if neighbor in results_by_val:
            retests.append(neighbor)

    print(f"\n  [*] Re-testing best ±{retest_range} neighbors ({len(retests)} values) with fresh 3 runs...")
    for expert_count in retests:
        _test_expert(expert_count, force_3runs=True)

    # Find final best after all retests
    best_entry = max(results_by_val.values(), key=lambda x: x["score"])
    best_experts = best_entry["experts"]
    all_results = [results_by_val[v] for v in sorted(results_by_val.keys())]
    best_perf = best_entry["perf"] or baseline

    print(f"\n{'=' * 60}")
    print(f"  {label} — RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline:     {baseline['tps']:.1f} t/s ({ctx.default_experts} experts)")
    print(f"  Best experts: {best_experts}")
    print(f"  Best Score:   {best_entry['score']:.1f} (speed × quality)")
    print(f"  Best TPS:     {best_perf['tps']:.1f} t/s")
    print(f"  Quality:      {best_entry['quality_factor']:.2f}")

    # Histogram
    max_score = max(r["score"] for r in all_results) if all_results else 0
    bar_max = 30
    print(f"\n  Score by expert_used_count (quality-adjusted):")
    print(f"  {'Value':>6}  {'Score':>7}  {'QF':>5}  {'Runs':>4}  {'':}")
    print(f"  {'─' * 6}  {'─' * 7}  {'─' * 5}  {'─' * 4}  {'─' * bar_max}")
    for r in all_results:
        score = r["score"]
        bar_len = int(score / max_score * bar_max) if max_score > 0 else 0
        bar = "█" * bar_len
        marker = " ◄ best" if r["experts"] == best_experts else ""
        runs = "3" if r["promoted"] else "1"
        qf = f"{r['quality_factor']:.2f}"
        print(f"  {r['experts']:>6}  {score:>7.1f}  {qf:>5}  {runs:>4}  {bar}{marker}")

    phase_elapsed = time.time() - phase_start_time
    print(f"\n  Duration:     {phase_elapsed / 60:.1f} min")

    results = {
        "phase": "experts",
        "baseline": baseline,
        "baseline_quality": ctx.quality_baseline,
        "best_tps": best_entry["score"],
        "best_metrics": {"tps": best_perf["tps"], "ttft": best_perf["ttft"],
                         "prompt_tps": best_perf["prompt_tps"], "total_ms": best_perf["total_ms"],
                         "quality_factor": best_entry["quality_factor"]},
        "best_params": {"expert_used_count": best_experts},
        "duration_seconds": round(phase_elapsed, 1),
        "all_trials": [
            {"number": i, "tps": r["score"], "metrics": r["perf"], "params": {"expert_used_count": r["experts"]},
             "quality_factor": r["quality_factor"]}
            for i, r in enumerate(all_results)
        ],
    }
    save_phase_results("experts", results)

    return best_experts


def phase_moe(n_trials=60, base_memory_config=None, include_experts=False):
    """MoE: Find optimal MoE config (threads + optionally experts).

    Runs MoE thread sweep (mandatory), then expert count sweep (optional).
      MoE thread sweep — sequential sweep, adaptive measurement
      Expert count sweep — optional, sequential sweep + quality gate

    For dense models (ctx.is_moe=False), skips entirely and returns empty config.
    n_trials is ignored — both sub-phases sweep their full ranges.

    Returns dict: {"n_cpu_moe": int, "expert_used_count": int} or None on failure.
    """
    if not ctx.is_moe:
        print("\n[*] Dense model detected — skipping MoE phase.")
        save_phase_results("moe_combined", {"phase": "moe_combined", "best_params": {}})
        return {}

    # Sub-phase 1: MoE threads — full sweep (always)
    best_moe_threads = phase_moe_threads(base_memory_config=base_memory_config)
    if best_moe_threads is None:
        return None

    # Sub-phase 2: Expert count — optional
    best_experts = ctx.default_experts
    if include_experts:
        best_experts = phase_experts(locked_moe_threads=best_moe_threads,
                                    base_memory_config=base_memory_config)

    # Save combined result so _get_moe_config() can load it
    combined = {"n_cpu_moe": best_moe_threads, "expert_used_count": best_experts}
    save_phase_results("moe_combined", {
        "phase": "moe_combined",
        "best_params": combined,
    })

    return combined


# ============================================================
# MoE Audit (re-validate MoE with compute params locked)
# ============================================================

def phase_moe_revalidate(locked_compute=None, locked_moe=None, base_memory_config=None):
    """MoE Audit: Re-test best ±2 MoE thread values with compute params locked.

    After Compute Audit finds optimal compute, the MoE sweet spot may shift.
    This does a quick focused sweep (~5 values) instead of the full 0-40.

    Args:
        locked_compute: Compute params from Compute Audit (threads, speculation, etc.)
        locked_moe: Current MoE config from MoE phase {"n_cpu_moe": int, "expert_used_count": int}
        base_memory_config: Memory params from Memory phase (if available)

    Returns int (best n_cpu_moe) or None on failure.
    """
    if not ctx.is_moe:
        print("\n[*] Dense model detected — skipping MoE re-validation.")
        return None

    existing = load_phase_results("moe_audit")
    if existing and "best_params" in existing:
        best_moe = existing["best_params"]["n_cpu_moe"]
        print(f"\n[*] MoE re-validation already complete — n_cpu_moe={best_moe} (from previous run)")
        return best_moe

    if locked_moe is None:
        locked_moe = {"n_cpu_moe": ctx.moe_sweep_center, "expert_used_count": ctx.default_experts}

    current_best = locked_moe["n_cpu_moe"]
    expert_count = locked_moe["expert_used_count"]
    retest_range = 2  # ±2 neighbors

    phase_start_time = time.time()
    label = "MoE Audit"

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)
    print(f"\n[*] Current MoE threads: {current_best} (from MoE phase)")
    if locked_compute:
        print(f"[*] Locked compute from Compute Audit: {len(locked_compute)} params")

    # Build base config with compute + memory params locked
    base_config = {**ctx.naked_engine}
    if base_memory_config:
        base_config.update(base_memory_config)
    if locked_compute:
        base_config.update(locked_compute)
    base_config["expert_used_count"] = expert_count

    # Measure baseline with current MoE setting
    print(f"\n[*] Starting baseline server (moe={current_best})...")
    kill_server()
    baseline_config = {**base_config, "n_cpu_moe": current_best}
    proc = start_server(baseline_config)
    if wait_for_server(proc=proc) != "ok":
        print("[!] Baseline server failed to start")
        if ctx.fail_fast:
            raise BaselineFailure("Baseline server failed in MoE Audit.")
        return None
    baseline = measure_perf(runs=3)
    baseline_score = compute_score(baseline)
    print(f"    Baseline (moe={current_best}): {baseline['tps']:.1f} t/s | "
          f"pp: {baseline['prompt_tps']:.0f} t/s | TTFT: {baseline['ttft']:.0f}ms | "
          f"Score: {baseline_score:.1f}")

    # Build test values: best ±2, clamped to valid range
    test_values = []
    for offset in range(-retest_range, retest_range + 1):
        val = current_best + offset
        if 1 <= val <= ctx.moe_sweep_max:
            test_values.append(val)

    print(f"\n[*] Re-testing MoE threads {test_values} with compute params locked...")

    results = {}
    best_score = baseline_score
    best_moe = current_best

    for i, moe_val in enumerate(test_values):
        print(f"\n  Test {i + 1}/{len(test_values)}: moe={moe_val}")
        kill_server()
        config = {**base_config, "n_cpu_moe": moe_val}
        proc = start_server(config)
        if wait_for_server(proc=proc) != "ok":
            print(f"    [!] Server failed to start for moe={moe_val}")
            results[moe_val] = {"score": 0.0, "perf": None}
            continue

        perf = measure_perf(runs=3)
        score = compute_score(perf)
        results[moe_val] = {"score": score, "perf": perf}

        marker = " *** NEW BEST ***" if score > best_score else ""
        print(f"    moe={moe_val}: {perf['tps']:.1f} t/s | pp: {perf['prompt_tps']:.0f} t/s | "
              f"TTFT: {perf['ttft']:.0f}ms | Score: {score:.1f}{marker}")

        if score > best_score:
            best_score = score
            best_moe = moe_val

    # Results summary
    phase_elapsed = time.time() - phase_start_time
    changed = best_moe != current_best

    print(f"\n{'=' * 60}")
    print(f"  {label} — RESULTS")
    print(f"{'=' * 60}")
    print(f"  Previous best: moe={current_best} (Score: {baseline_score:.1f})")
    print(f"  New best:      moe={best_moe} (Score: {best_score:.1f})")
    if changed:
        print(f"  ** MoE threads changed from {current_best} to {best_moe} **")
    else:
        print(f"  MoE threads confirmed at {best_moe}")

    # Score table
    print(f"\n  {'Value':>6}  {'Score':>7}  {'TPS':>6}  {'':}")
    print(f"  {'─' * 6}  {'─' * 7}  {'─' * 6}  {'─' * 20}")
    for val in sorted(results.keys()):
        r = results[val]
        perf = r["perf"]
        if perf:
            marker = " ◄ best" if val == best_moe else ""
            tps_str = f"{perf['tps']:.1f}"
            print(f"  {val:>6}  {r['score']:>7.1f}  {tps_str:>6}  {marker}")

    print(f"\n  Duration: {phase_elapsed / 60:.1f} min")

    best_perf = results.get(best_moe, {}).get("perf") or baseline
    save_phase_results("moe_audit", {
        "phase": "moe_audit",
        "baseline": baseline,
        "previous_moe": current_best,
        "best_tps": best_score,
        "best_metrics": {"tps": best_perf["tps"], "ttft": best_perf["ttft"],
                         "prompt_tps": best_perf["prompt_tps"], "total_ms": best_perf["total_ms"]},
        "best_params": {"n_cpu_moe": best_moe},
        "changed": changed,
        "duration_seconds": round(phase_elapsed, 1),
        "all_trials": [
            {"params": {"n_cpu_moe": val}, "score": results[val]["score"],
             "metrics": results[val]["perf"]}
            for val in sorted(results.keys()) if results[val]["perf"]
        ],
    })

    return best_moe


# ============================================================
# Compute / Compute Audit (with MoE locked)
# ============================================================

def phase_compute(n_trials=60, phase_name="compute", base_memory_config=None, seed_params=None, locked_moe=None):
    """Optimize compute allocation params (with MoE locked from MoE phase).

    Args:
        n_trials: Number of trials to run.
        phase_name: "compute" or "compute_audit" — used for study name and result file.
        base_memory_config: If provided (Compute Audit), use these memory/throughput
                           params as the base. Otherwise starts naked.
        locked_moe: Dict from MoE phase {"n_cpu_moe": int, "expert_used_count": int} (locked, not tuned).
    """
    if not locked_moe:
        locked_moe = {"n_cpu_moe": ctx.moe_sweep_center, "expert_used_count": ctx.default_experts}
    phase_start_time = time.time()
    is_pareto = _config.get("pareto", False)
    labels = {
        "compute": "Compute",
        "compute_audit": "Compute Audit",
    }
    label = labels.get(phase_name, f"{phase_name}: Compute Allocation")
    if is_pareto:
        label += " [PARETO]"

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)
    if ctx.is_moe:
        print(f"\n[*] Locked MoE: {locked_moe.get('n_cpu_moe', 'N/A')} | Experts: {locked_moe.get('expert_used_count', ctx.default_experts)}")

    if base_memory_config:
        mem_src = "Memory" if phase_name == "compute_audit" else "previous"
        print(f"\n[*] Base memory config from {mem_src}: {len(base_memory_config)} params")

    # Naked baseline (with memory config if revalidating)
    # Check if study is already complete before starting baseline server
    is_pareto = _config.get("pareto", False)
    study, remaining, completed = setup_study(phase_name, n_trials, is_pareto=is_pareto)
    if remaining == 0:
        best = get_best_trial(study)
        if is_pareto and best.values:
            print(f"\n  Pareto Best: TPS={best.values[0]:.1f} | VRAM={-best.values[1]:.0f}MB | Q={best.values[2]:.2f}")
            pareto = extract_pareto_front(study)
            print_pareto_front(pareto)
        else:
            print(f"\n  Best Score:  {best.value:.1f} | TPS: {best.user_attrs.get('tps', 0):.1f} | "
                  f"TTFT: {best.user_attrs.get('ttft', 0):.0f}ms")
        print_param_importance(study)
        return best.params

    base_config = {**ctx.naked_engine}
    if base_memory_config:
        base_config.update(base_memory_config)
    # Always include locked MoE params in baseline
    if locked_moe:
        base_config.update(locked_moe)

    print("\n[*] Starting baseline server...")
    proc, status = _boot_server_with_jinja_recovery(base_config)
    if status != "ok":
        print("[!] Baseline server failed to start")
        if ctx.fail_fast:
            raise BaselineFailure(f"Baseline server failed in {label}.")
        return None
    baseline = measure_perf(runs=3)
    print(f"    Baseline: {baseline['tps']:.1f} t/s | pp: {baseline['prompt_tps']:.0f} t/s | "
          f"TTFT: {baseline['ttft']:.0f}ms | Score: {compute_score(baseline):.1f}")

    # Seed with previous phase's best params so TPE starts from a known good point
    if seed_params and completed == 0:
        print(f"[*] Seeding Trial 0 with previous best config")
        study.enqueue_trial(seed_params)
    elif completed == 0:
        # Seed with proven best compute config from previous runs
        print(f"[*] Seeding Trial 0 with known-good compute config")
        study.enqueue_trial({
            "threads": 4,
            "threads_batch": 12,
            "poll": 0, "poll_batch": 50,
            "prio": 0, "prio_batch": 0,
            "cpu_strict": 1, "cpu_strict_batch": 1,
            "spec_type": "ngram-map-k4v",
            "spec_ngram_n": 23, "spec_ngram_m": 20,
            "spec_ngram_min_hits": 2,
            "draft_max": 9, "draft_min": 8,
            "draft_p_min": 0.9,
            "lookup_cache_dynamic": str(ctx.results_dir / "lookup-cache.bin"),
        })
        # Second seed: alternate config with higher threads + speculation
        study.enqueue_trial({
            "threads": 16,
            "threads_batch": 16,
            "poll": 50, "poll_batch": 50,
            "prio": 0, "prio_batch": 3,
            "cpu_strict": 0, "cpu_strict_batch": 1,
            "spec_type": "ngram-cache",
            "spec_ngram_n": 14, "spec_ngram_m": 64,
            "spec_ngram_min_hits": 4,
            "draft_max": 47, "draft_min": 4,
            "draft_p_min": 0.52,
            "lookup_cache_dynamic": None,
        })

    total_trials = completed + remaining
    best_score = compute_score(baseline)
    # When resuming, recalculate best score from existing trials using CURRENT formula
    # (stored values may use an old formula and poison adaptive measurement thresholds)
    if completed > 0:
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs:
                perf = {k: t.user_attrs.get(k, 0) for k in ["tps", "ttft", "prompt_tps", "total_ms"]}
                if perf["tps"] > 0:
                    recalc = compute_score(perf)
                    best_score = max(best_score, recalc)

    def objective(trial):
        nonlocal best_score

        # Ensure GPU isn't thermally throttled before this trial
        if check_thermal_throttle(threshold=85)[0]:
            wait_for_cooldown(target_temp=75, timeout=120)

        config = {
            **base_config,
            # MoE locked from MoE phase
            **locked_moe,
            # Compute allocation params
            "threads": trial.suggest_categorical("threads", sorted(set(list(range(4, ctx.max_threads + 1, 4)) + [ctx.max_threads]))),
            "threads_batch": trial.suggest_categorical("threads_batch", sorted(set(list(range(4, ctx.max_threads + 1, 4)) + [ctx.max_threads]))),
            "poll": trial.suggest_categorical("poll", [0, 10, 25, 50, 100]),
            "poll_batch": trial.suggest_categorical("poll_batch", [0, 10, 25, 50, 100]),
            "prio": trial.suggest_int("prio", 0, 3),
            "prio_batch": trial.suggest_int("prio_batch", 0, 3),
            "cpu_strict": trial.suggest_categorical("cpu_strict", [0, 1]),
            "cpu_strict_batch": trial.suggest_categorical("cpu_strict_batch", [0, 1]),
            # NUMA awareness — only exposed if multi-NUMA system detected
            **({"numa": trial.suggest_categorical("numa", ["distribute", "isolate", "numactl"])} if ctx.numa_nodes > 1 else {}),
            # Speculation params
            "spec_type": trial.suggest_categorical("spec_type", ["ngram-simple", "ngram-cache", "ngram-map-k", "ngram-map-k4v", "ngram-mod"]),
            "spec_ngram_n": trial.suggest_int("spec_ngram_n", 2, 24),
            "spec_ngram_m": trial.suggest_int("spec_ngram_m", 8, 96),
            "spec_ngram_min_hits": trial.suggest_int("spec_ngram_min_hits", 1, 5),
            "draft_max": trial.suggest_int("draft_max", 4, 48),
            "draft_min": trial.suggest_int("draft_min", 0, 8),
            "draft_p_min": trial.suggest_float("draft_p_min", 0.3, 0.99),
            "lookup_cache_dynamic": trial.suggest_categorical("lookup_cache_dynamic", [None, ctx.lookup_cache_file]),
        }
        # Remove None values so start_server doesn't see the key
        config = {k: v for k, v in config.items() if v is not None}

        # Cross-phase parameter injection: in Compute Audit, also wiggle memory params
        # slightly (±20%) around the Memory phase best. This catches threads×batch_size
        # interactions that coordinate descent misses.
        if phase_name == "compute_audit" and base_memory_config:
            # Batch size: allow neighbor values
            mem_batch = base_memory_config.get("batch_size", 512)
            batch_opts = sorted(set(v for v in [256, 512, 1024, 2048, 4096]
                                    if abs(v - mem_batch) / max(1, mem_batch) <= 0.5 or v == mem_batch))
            if batch_opts:
                config["batch_size"] = trial.suggest_categorical("cross_batch_size", batch_opts)
            # Ubatch size
            mem_ubatch = base_memory_config.get("ubatch_size", 256)
            ubatch_opts = sorted(set(v for v in [128, 256, 512, 1024]
                                     if abs(v - mem_ubatch) / max(1, mem_ubatch) <= 0.5 or v == mem_ubatch))
            if ubatch_opts:
                config["ubatch_size"] = trial.suggest_categorical("cross_ubatch_size", ubatch_opts)

        # Pre-boot pruning
        if config.get("draft_min", 0) >= config.get("draft_max", 4):
            print(f"\n  Trial {trial.number}: pruned (draft_min >= draft_max)")
            raise optuna.exceptions.TrialPruned()

        # Check for duplicate config before restarting server
        cached = check_duplicate_trial(trial)
        if cached is not None:
            if isinstance(cached, (list, tuple)):
                print(f"\n  Trial {trial.number}: duplicate config — cached TPS: {cached[0]:.1f}")
            else:
                print(f"\n  Trial {trial.number}: duplicate config — cached score: {cached:.1f}")
            return cached

        params_short = (f"t={config['threads']}/{config['threads_batch']} "
                        f"moe={config['n_cpu_moe']} experts={config['expert_used_count']} "
                        f"poll={config['poll']} prio={config['prio']} "
                        f"spec_n={config['spec_ngram_n']} spec_m={config['spec_ngram_m']} "
                        f"draft={config['draft_max']}")

        # Delete dynamic lookup cache before each trial to prevent temporal leakage —
        # otherwise later trials get free TPS from earlier trials' cached n-grams,
        # corrupting the GP's parameter importance estimates.
        cache_file = config.get("lookup_cache_dynamic")
        if cache_file and Path(cache_file).exists():
            Path(cache_file).unlink()

        print(f"\n  Trial {trial.number}: restarting server... | {params_short}")
        kill_server()
        proc = start_server(config)

        status = wait_for_server(proc=proc)
        if status == "oom":
            print(f"  Trial {trial.number}: pruned (OOM — config too large for VRAM)")
            kill_server()
            raise optuna.exceptions.TrialPruned()
        elif status != "ok":
            _server_start_failed(trial.number, params_short, proc)
            if is_pareto:
                return (0.0, -99999.0, 0.0)
            return 0.0

        # Multi-fidelity gate: 5-token quick test for TTFT/prompt speed
        # If the gate score is in the bottom 50% of known configs, prune early
        gate = measure_perf_quick_gate(n_predict=5)
        if gate and best_score > 0:
            gate_score = gate.get("gate_score", 0)
            trial.report(gate_score, step=0)
            if trial.should_prune():
                print(f"  Trial {trial.number}: pruned by multi-fidelity gate (gate_score={gate_score:.1f})")
                raise optuna.exceptions.TrialPruned()

        perf, promoted = measure_perf_adaptive(best_score)
        tps = perf["tps"]
        score = compute_score(perf)

        # Report full score for successive halving step 1
        trial.report(score, step=1)

        # Store measurement variance for noise-aware GP
        if perf.get("tps_std") is not None:
            trial.set_user_attr("tps_std", perf["tps_std"])
            trial.set_user_attr("tps_cv", perf.get("tps_cv", 0))

        # Capture VRAM for Pareto objective
        vram_mb = _get_vram_used_mb()
        if vram_mb is not None:
            trial.set_user_attr("vram_used_mb", vram_mb)
            perf["vram_used_mb"] = vram_mb

        trial.set_user_attr("tps", tps)
        trial.set_user_attr("ttft", perf["ttft"])
        trial.set_user_attr("prompt_tps", perf["prompt_tps"])
        trial.set_user_attr("total_ms", perf["total_ms"])

        if not is_pareto:
            best_score = print_trial_result(trial.number, total_trials, tps, perf, params_short, best_score)
            return score
        else:
            # Pareto mode: return (tps, -vram, quality_factor=1.0 for speed phases)
            objectives = compute_pareto_objectives(perf, quality_factor=1.0)
            print(f"  Trial {trial.number}: TPS={tps:.1f} VRAM={vram_mb or 0:.0f}MB | {params_short}")
            return objectives

    est_minutes = remaining * 20 // 60
    print(f"\n[*] Running {remaining} trials (+ {completed} completed, ~{est_minutes} min)...\n")
    pbar = create_phase_pbar(remaining, desc=label)
    pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.1)
    callbacks = [TqdmUpdateCallback()]
    if not is_pareto:
        callbacks.append(GPStoppingCallback(baseline_score=best_score))
    study.optimize(objective, n_trials=remaining, callbacks=callbacks, show_progress_bar=False)
    close_phase_pbar()

    # Results — handle both single-objective and Pareto
    best = get_best_trial(study)
    baseline_score = compute_score(baseline)

    print(f"\n{'=' * 60}")
    print(f"  {label} — RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline:    {baseline['tps']:.1f} t/s | TTFT: {baseline['ttft']:.0f}ms | Score: {baseline_score:.1f}")

    if is_pareto:
        pareto = extract_pareto_front(study)
        print(f"\n  Pareto Front: {len(pareto)} optimal configs")
        print_pareto_front(pareto)
        beat_baseline = best.values[0] > baseline["tps"] if best.values else False
        returned_params = best.params
    else:
        beat_baseline = best.value > baseline_score
        if beat_baseline:
            print(f"  Best Score:  {best.value:.1f} (composite) — beats baseline by {best.value - baseline_score:.1f}")
            returned_params = best.params
        else:
            print(f"  Best Score:  {best.value:.1f} (composite) — BELOW baseline ({baseline_score:.1f})")
            print(f"  [!] No trial beat baseline. Passing through baseline config.")
            # Return the baseline config keys that downstream phases need, not empty dict
            returned_params = {k: base_config[k] for k in [
                "threads", "threads_batch", "poll", "poll_batch",
                "prio", "prio_batch", "cpu_strict", "cpu_strict_batch",
                "n_cpu_moe", "expert_used_count",
            ] if k in base_config}

    print(f"  Best TPS:    {best.user_attrs.get('tps', 0):.1f} t/s")
    print(f"  Best TTFT:   {best.user_attrs.get('ttft', 0):.0f}ms")
    print(f"  Best Prompt: {best.user_attrs.get('prompt_tps', 0):.0f} t/s")
    print(f"  Best params:")
    for k, v in (best.params if beat_baseline else returned_params).items():
        print(f"    {k}: {v}")

    importances = print_param_importance(study)

    phase_elapsed = time.time() - phase_start_time
    phase_mins = phase_elapsed / 60
    print(f"\n  Duration:    {phase_mins:.1f} min")

    results = {
        "phase": phase_name,
        "baseline": baseline,
        "baseline_score": baseline_score,
        "beat_baseline": beat_baseline,
        "best_tps": best.user_attrs.get("tps", 0),
        "best_metrics": best.user_attrs,
        "best_params": returned_params,
        "base_memory_config": base_memory_config,
        "param_importance": {k: round(v * 100, 1) for k, v in importances.items()},
        "duration_seconds": round(phase_elapsed, 1),
        "duration_minutes": round(phase_mins, 1),
        "all_trials": [
            {"number": t.number,
             "tps": t.values[0] if is_pareto and t.values else t.value,
             "metrics": t.user_attrs, "params": t.params}
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ],
    }
    if is_pareto:
        results["pareto_front"] = [
            {"tps": t.values[0], "vram_mb": -t.values[1], "quality": t.values[2],
             "params": t.params}
            for t in extract_pareto_front(study)
        ]
    save_phase_results(phase_name, results)

    return returned_params


# ============================================================
# Memory / Memory Audit
# ============================================================

def _mini_niah_recall_test():
    """Quick 1-shot needle-in-a-haystack recall test.

    Sends a short prompt (~500 tokens of filler) with an embedded fact,
    then asks the model to retrieve it. Used inside the Memory phase objective
    to catch KV cache quantization levels that destroy attention recall.

    Returns True if the model correctly retrieved the needle, False otherwise.
    """
    needle_fact = "The emergency evacuation code is Bravo Tango 7742."
    needle_query = "What is the emergency evacuation code?"
    needle_expected = "bravo tango 7742"

    # Build a short filler (~500 tokens worth, ~1500 chars at 3 chars/token)
    filler_parts = []
    for i, block in enumerate(_NIAH_FILLER_BLOCKS[:4]):
        filler_parts.append(f"\nParagraph {i + 1}:\n{block}")
    filler = "\n".join(filler_parts)

    # Inject needle at ~50% depth
    mid = len(filler) // 2
    newline_pos = filler.rfind("\n\n", 0, mid)
    if newline_pos == -1:
        newline_pos = mid
    prompt = filler[:newline_pos] + f"\n\nIMPORTANT NOTE: {needle_fact}\n\n" + filler[newline_pos:]
    prompt += f"\n\nBased on everything above, answer precisely:\n{needle_query}"

    try:
        r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json={
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }, timeout=60)
        if r.status_code == 200:
            content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            return needle_expected in content.lower()
    except Exception:
        pass
    return False


def phase_memory(n_trials=60, phase_name="memory", base_compute_config=None, seed_params=None):
    """Optimize memory & throughput params. Each trial restarts the server.

    Args:
        n_trials: Number of trials to run.
        phase_name: "memory" or "memory_audit" — used for study name and result file.
        base_compute_config: Compute allocation params to use as base.
    """
    phase_start_time = time.time()
    is_pareto = _config.get("pareto", False)
    is_revalidation = phase_name == "memory_audit"
    label = "Memory Audit" if is_revalidation else "Memory"
    if is_pareto:
        label += " [PARETO]"

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)

    # Check if study is already complete before starting baseline server
    is_pareto = _config.get("pareto", False)
    study, remaining, completed = setup_study(phase_name, n_trials, is_pareto=is_pareto)
    if remaining == 0:
        best = get_best_trial(study)
        if is_pareto and best.values:
            print(f"\n  Pareto Best: TPS={best.values[0]:.1f} | VRAM={-best.values[1]:.0f}MB | Q={best.values[2]:.2f}")
            pareto = extract_pareto_front(study)
            print_pareto_front(pareto)
        else:
            print(f"\n  Best Score:  {best.value:.1f} | TPS: {best.user_attrs.get('tps', 0):.1f} | "
                  f"TTFT: {best.user_attrs.get('ttft', 0):.0f}ms")
        print_param_importance(study)
        return best.params

    # Build base config from compute results
    base_config = {**ctx.naked_engine}
    if base_compute_config:
        # Map compute trial params to server config keys
        base_config["threads"] = base_compute_config.get("threads")
        base_config["threads_batch"] = base_compute_config.get("threads_batch")
        base_config["n_cpu_moe"] = base_compute_config.get("n_cpu_moe")
        base_config["expert_used_count"] = base_compute_config.get("expert_used_count")
        base_config["poll"] = base_compute_config.get("poll")
        base_config["poll_batch"] = base_compute_config.get("poll_batch")
        base_config["prio"] = base_compute_config.get("prio")
        base_config["prio_batch"] = base_compute_config.get("prio_batch")
        base_config["cpu_strict"] = base_compute_config.get("cpu_strict")
        base_config["cpu_strict_batch"] = base_compute_config.get("cpu_strict_batch")
        base_config["spec_type"] = base_compute_config.get("spec_type", "ngram-simple")
        base_config["spec_ngram_n"] = base_compute_config.get("spec_ngram_n")
        base_config["spec_ngram_m"] = base_compute_config.get("spec_ngram_m")
        base_config["spec_ngram_min_hits"] = base_compute_config.get("spec_ngram_min_hits")
        base_config["draft_max"] = base_compute_config.get("draft_max")
        base_config["draft_min"] = base_compute_config.get("draft_min")
        base_config["draft_p_min"] = base_compute_config.get("draft_p_min")
        if base_compute_config.get("lookup_cache_dynamic"):
            base_config["lookup_cache_dynamic"] = base_compute_config["lookup_cache_dynamic"]
        # Remove None values
        base_config = {k: v for k, v in base_config.items() if v is not None}
        print(f"\n[*] Base compute config: t={base_compute_config.get('threads')}/{base_compute_config.get('threads_batch')} "
              f"moe={base_compute_config.get('n_cpu_moe')} experts={base_compute_config.get('expert_used_count', 8)} "
              f"spec_n={base_compute_config.get('spec_ngram_n')} spec_m={base_compute_config.get('spec_ngram_m')} "
              f"draft={base_compute_config.get('draft_max')}")
    else:
        print("\n[!] No compute config — running with naked engine.")

    # Baseline — use llama-bench if available for speed
    # BUT: if base config includes speculation params, bench can't measure those
    # (llama-bench doesn't support speculative decoding), so fall back to HTTP
    has_speculation = any(k in base_config for k in ["spec_type", "spec_ngram_n", "draft_max"])
    use_bench = ctx.bench_path is not None and not has_speculation
    if has_speculation and ctx.bench_path:
        print("    [bench] Skipped — base config uses speculative decoding (HTTP only)")
    if use_bench:
        print("\n[*] Running baseline via llama-bench...")
        baseline = run_bench_trial(base_config, repetitions=3)
        if not baseline or baseline.get("error"):
            print("[!] Baseline bench failed, falling back to HTTP server...")
            use_bench = False
    if not use_bench:
        print("\n[*] Starting baseline server...")
        print(f"    Config keys: {sorted(base_config.keys())}")
        kill_server()
        proc = start_server(base_config)
        if wait_for_server(proc=proc) != "ok":
            print("[!] Baseline server failed to start")
            if ctx.fail_fast:
                raise BaselineFailure(f"Baseline server failed in {label}.")
            return None
        baseline = measure_perf(runs=3)
    score_fn = (lambda p: _bench_score(p, baseline)) if use_bench else compute_score
    print(f"    Baseline: {baseline['tps']:.1f} t/s | pp: {baseline['prompt_tps']:.0f} t/s | "
          f"TTFT: {baseline['ttft']:.0f}ms | Score: {score_fn(baseline):.1f}")
    if use_bench:
        print(f"    [bench] Using llama-bench for fast Memory sweep")

    ctx_size = base_config.get("context", 4096)

    # Seed with previous phase's best params so TPE starts from a known good point
    if seed_params and completed == 0:
        print(f"[*] Seeding Trial 0 with previous best config")
        study.enqueue_trial(seed_params)
    elif completed == 0:
        # No seed — enqueue a known-good baseline config (f16 KV, small batch, fa on)
        # Clamp batch_size to context so the seed value is always in batch_opts
        seed_batch = min(512, ctx_size) if ctx_size > 0 else 512
        print(f"[*] Seeding Trial 0 with known-good f16 config")
        study.enqueue_trial({
            "batch_size": seed_batch, "ubatch_size": min(128, seed_batch), "flash_attn": "on",
            "kv_cache_type": "f16", "swa_full": False, "repack": False,
            "op_offload": False, "mlock": True, "no_mmap": True,
        })

    # Measure baseline PPL for quant degradation comparison.
    # Only meaningful when using HTTP server (bench can't do logprobs).
    baseline_ppl = None
    if not use_bench:
        print("    Measuring baseline perplexity...")
        baseline_ppl = measure_true_perplexity()
        if baseline_ppl != float('inf'):
            print(f"    Baseline PPL: {baseline_ppl:.2f}")
        else:
            print("    [!] PPL measurement failed — will skip PPL quality gate")
            baseline_ppl = None

    total_trials = completed + remaining
    best_score = score_fn(baseline)
    # When resuming, recalculate best score from existing trials using CURRENT formula
    if completed > 0:
        for t in study.trials:
            if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs:
                perf = {k: t.user_attrs.get(k, 0) for k in ["tps", "ttft", "prompt_tps", "total_ms"]}
                if perf["tps"] > 0:
                    recalc = score_fn(perf)
                    best_score = max(best_score, recalc)

    def objective(trial):
        nonlocal best_score

        # Ensure GPU isn't thermally throttled before this trial
        if check_thermal_throttle(threshold=85)[0]:
            wait_for_cooldown(target_temp=75, timeout=120)

        # VRAM-aware batch_size bounds: prevent wasted trials on infeasible sizes.
        # KV cache per batch slot ≈ 2 * n_layers * d_head * n_kv_heads * kv_bytes * ctx / 1e9 GB
        # Simplified heuristic: model_size + batch_size * ctx * kv_cost_per_token must fit in VRAM.
        all_batch_opts = [512, 1024, 2048, 4096]
        if ctx.vram_total_mb and ctx.model_size_gb:
            vram_for_kv_mb = ctx.vram_total_mb - (ctx.model_size_gb * 1024) - 512  # 512MB headroom
            if vram_for_kv_mb > 0:
                # Conservative: ~0.5 KB per token per batch slot for KV cache (covers most architectures)
                max_batch = int(vram_for_kv_mb * 1024 / (ctx_size * 0.5)) if ctx_size > 0 else 4096
                max_batch = max(256, min(max_batch, 4096))  # clamp to sane range
                all_batch_opts = [v for v in all_batch_opts if v <= max_batch]
        batch_opts = [v for v in all_batch_opts if v <= ctx_size]
        if not batch_opts:
            batch_opts = [min(all_batch_opts) if all_batch_opts else ctx_size]
        batch_size = trial.suggest_categorical("batch_size", batch_opts)
        ubatch_size = trial.suggest_categorical("ubatch_size", [128, 256, 512, 1024])

        # Pre-boot pruning: skip logically impossible configs
        if ubatch_size > batch_size:
            print(f"\n  Trial {trial.number}: pruned (ubatch {ubatch_size} > batch {batch_size})")
            raise optuna.exceptions.TrialPruned()

        flash_attn = trial.suggest_categorical("flash_attn", ["on", "off"])
        kv_cache_type = trial.suggest_categorical("kv_cache_type", ["f16", "bf16", "q8_0", "q5_1", "q4_0"])

        # Quantized KV cache requires flash attention — skip impossible combos
        if flash_attn == "off" and kv_cache_type not in ("f16", "bf16"):
            print(f"\n  Trial {trial.number}: pruned (quantized KV {kv_cache_type} requires flash_attn=on)")
            raise optuna.exceptions.TrialPruned()

        config = {
            **base_config,
            "context": ctx_size,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            # Core memory params that actually affect single-user inference
            "flash_attn": flash_attn,
            "kv_cache_type": kv_cache_type,
            "swa_full": trial.suggest_categorical("swa_full", [True, False]),
            "repack": trial.suggest_categorical("repack", [True, False]),
            "op_offload": trial.suggest_categorical("op_offload", [True, False]),
            "mlock": trial.suggest_categorical("mlock", [True, False]),
            "no_mmap": trial.suggest_categorical("no_mmap", [True, False]),
            # GPU layers locked from GPU Offload
            "n_gpu_layers": ctx.default_gpu_layers,
            "fit": True,
        }
        config = {k: v for k, v in config.items() if v is not None}

        # Check for duplicate config before restarting server
        cached = check_duplicate_trial(trial)
        if cached is not None:
            if isinstance(cached, (list, tuple)):
                print(f"\n  Trial {trial.number}: duplicate config — cached TPS: {cached[0]:.1f}")
            else:
                print(f"\n  Trial {trial.number}: duplicate config — cached score: {cached:.1f}")
            return cached

        params_short = (f"b={config['batch_size']} ub={config['ubatch_size']} "
                        f"fa={config['flash_attn']} "
                        f"kv={config.get('kv_cache_type', 'f16')}")

        if use_bench:
            print(f"\n  Trial {trial.number}: bench | {params_short}")
            perf = run_bench_trial(config, repetitions=3)
            if perf is None or perf.get("error") == "oom":
                err = "OOM" if perf and perf.get("error") else "bench error"
                print(f"  Trial {trial.number}: pruned ({err})")
                raise optuna.exceptions.TrialPruned()
        else:
            print(f"\n  Trial {trial.number}: restarting server... | {params_short}")
            kill_server()
            proc = start_server(config)

            status = wait_for_server(proc=proc)
            if status == "oom":
                print(f"  Trial {trial.number}: pruned (OOM — config too large for VRAM)")
                kill_server()
                raise optuna.exceptions.TrialPruned()
            elif status != "ok":
                _server_start_failed(trial.number, params_short, proc)
                if is_pareto:
                    return (0.0, -99999.0, 0.0)
                return 0.0

            # Check if flash_attn was silently disabled by the server
            # (happens when quantized KV cache isn't compatible with flash attention)
            ctx._flash_attn_disabled_for_kv = False  # reset per trial
            if flash_attn == "on" and kv_cache_type not in ("f16", "bf16"):
                # Check stderr after warmup — server may have silently disabled flash attn
                stderr_text = "\n".join(getattr(proc, "_stderr_lines", []))
                if "flash attention not supported" in stderr_text.lower():
                    ctx._flash_attn_disabled_for_kv = True
                    print(f"  Trial {trial.number}: [!] flash_attn silently disabled for {kv_cache_type} KV cache")
                    trial.set_user_attr("flash_attn_disabled", True)

            # Multi-fidelity gate: 5-token quick test
            gate = measure_perf_quick_gate(n_predict=5)
            if gate and best_score > 0:
                gate_score = gate.get("gate_score", 0)
                trial.report(gate_score, step=0)
                if trial.should_prune():
                    print(f"  Trial {trial.number}: pruned by multi-fidelity gate (gate_score={gate_score:.1f})")
                    raise optuna.exceptions.TrialPruned()

            perf, promoted = measure_perf_adaptive(best_score)

        tps = perf["tps"]
        score = score_fn(perf)

        # Penalize configs where flash_attn was silently disabled — the GP would
        # otherwise falsely associate the KV type with bad performance instead of
        # realizing flash attention was turned off.
        if ctx._flash_attn_disabled_for_kv:
            score *= 0.7  # 30% penalty
            trial.set_user_attr("flash_attn_penalty", True)

        # Quality gates for quantized KV cache types (f16/bf16 are lossless → skip).
        # Two complementary signals:
        #   1. PPL: mathematical measure of language comprehension degradation
        #   2. Mini-NIAH: functional recall test (catches attention mechanism failure)
        if not use_bench and kv_cache_type not in ("f16", "bf16"):
            # PPL quality gate — primary signal for quant degradation
            if baseline_ppl is not None:
                trial_ppl = measure_true_perplexity()
                trial.set_user_attr("ppl", round(trial_ppl, 2) if trial_ppl != float('inf') else -1)
                if trial_ppl != float('inf'):
                    ppl_factor = ppl_quality_factor(baseline_ppl, trial_ppl)
                    trial.set_user_attr("ppl_factor", round(ppl_factor, 3))
                    ppl_pct = ((trial_ppl - baseline_ppl) / baseline_ppl * 100) if baseline_ppl > 0 else 0
                    print(f"  Trial {trial.number}: PPL {trial_ppl:.2f} (baseline {baseline_ppl:.2f}, "
                          f"{ppl_pct:+.1f}%) → factor {ppl_factor:.2f}")
                    score *= ppl_factor
                else:
                    print(f"  Trial {trial.number}: [!] PPL measurement failed for {kv_cache_type}")

            # Mini-NIAH: functional recall test (catches attention failure PPL might miss)
            niah_passed = _mini_niah_recall_test()
            trial.set_user_attr("niah_recall", niah_passed)
            if not niah_passed:
                score *= 0.15  # 85% penalty — GP learns this KV type breaks the model
                print(f"  Trial {trial.number}: [!] mini-NIAH FAIL — {kv_cache_type} broke recall")

        # Report full score for successive halving step 1
        if not use_bench:
            trial.report(score, step=1)

        # Store measurement variance for noise-aware GP
        if perf.get("tps_std") is not None:
            trial.set_user_attr("tps_std", perf["tps_std"])
            trial.set_user_attr("tps_cv", perf.get("tps_cv", 0))

        # Capture VRAM for Pareto objective
        vram_mb = _get_vram_used_mb()
        if vram_mb is not None:
            trial.set_user_attr("vram_used_mb", vram_mb)
            perf["vram_used_mb"] = vram_mb

        trial.set_user_attr("tps", tps)
        trial.set_user_attr("ttft", perf["ttft"])
        trial.set_user_attr("prompt_tps", perf["prompt_tps"])
        trial.set_user_attr("total_ms", perf["total_ms"])

        if not is_pareto:
            best_score = print_trial_result(trial.number, total_trials, tps, perf, params_short, best_score)
            return score
        else:
            objectives = compute_pareto_objectives(perf, quality_factor=1.0)
            print(f"  Trial {trial.number}: TPS={tps:.1f} VRAM={vram_mb or 0:.0f}MB | {params_short}")
            return objectives

    secs_per_trial = 5 if use_bench else 20
    est_minutes = remaining * secs_per_trial // 60
    mode_label = "llama-bench" if use_bench else "HTTP"
    print(f"\n[*] Running {remaining} trials via {mode_label} (+ {completed} completed, ~{est_minutes} min)...\n")
    pbar = create_phase_pbar(remaining, desc=label)
    callbacks = [TqdmUpdateCallback()]
    if not is_pareto:
        callbacks.append(GPStoppingCallback(baseline_score=best_score))
    study.optimize(objective, n_trials=remaining, callbacks=callbacks, show_progress_bar=False)
    close_phase_pbar()

    # Restore best config for verification
    best = get_best_trial(study)
    if use_bench:
        print(f"\n[*] Verifying best config via llama-bench (3 reps)...")
        verify_config_mapped = {**base_config, **best.params}
        verify_config_mapped = {k: v for k, v in verify_config_mapped.items() if v is not None}
        verify = run_bench_trial(verify_config_mapped, repetitions=3)
        if not verify or verify.get("error"):
            print("[!] Bench verification failed, falling back to HTTP...")
            kill_server()
            best_proc = start_server(verify_config_mapped)
            wait_for_server(proc=best_proc)
            verify = measure_perf(runs=3)
    else:
        print(f"\n[*] Restarting with best config for verification...")
        kill_server()
        verify_config_mapped = {**base_config, **best.params}
        verify_config_mapped = {k: v for k, v in verify_config_mapped.items() if v is not None}
        best_proc = start_server(verify_config_mapped)
        wait_for_server(proc=best_proc)
        verify = measure_perf(runs=3)

    baseline_score = score_fn(baseline)

    print(f"\n{'=' * 60}")
    print(f"  {label} — RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline:    {baseline['tps']:.1f} t/s | TTFT: {baseline['ttft']:.0f}ms | Score: {baseline_score:.1f}")

    if is_pareto:
        pareto = extract_pareto_front(study)
        print(f"\n  Pareto Front: {len(pareto)} optimal configs")
        print_pareto_front(pareto)
        beat_baseline = best.values[0] > baseline["tps"] if best.values else False
        returned_params = best.params
    else:
        beat_baseline = best.value > baseline_score
        if beat_baseline:
            print(f"  Best Score:  {best.value:.1f} (composite) — beats baseline by {best.value - baseline_score:.1f}")
            returned_params = best.params
        else:
            print(f"  Best Score:  {best.value:.1f} (composite) — BELOW baseline ({baseline_score:.1f})")
            print(f"  [!] No trial beat baseline. Passing through baseline config.")
            # Return baseline memory keys so downstream phases have something to work with
            returned_params = {k: base_config[k] for k in [
                "batch_size", "ubatch_size", "flash_attn", "kv_cache_type",
                "no_mmap", "mlock",
            ] if k in base_config}

    print(f"  Best TPS:    {best.user_attrs.get('tps', 0):.1f} t/s")
    print(f"  Best TTFT:   {best.user_attrs.get('ttft', 0):.0f}ms")
    print(f"  Verified:    {verify['tps']:.1f} t/s | TTFT: {verify['ttft']:.0f}ms | Score: {score_fn(verify):.1f}")
    print(f"  Best params:")
    for k, v in (best.params if beat_baseline else returned_params).items():
        print(f"    {k}: {v}")

    importances = print_param_importance(study)

    phase_elapsed = time.time() - phase_start_time
    phase_mins = phase_elapsed / 60
    print(f"\n  Duration:    {phase_mins:.1f} min")

    results = {
        "phase": phase_name,
        "baseline": baseline,
        "baseline_score": baseline_score,
        "beat_baseline": beat_baseline,
        "best_tps": best.user_attrs.get("tps", 0),
        "best_metrics": best.user_attrs,
        "verified": verify,
        "best_params": returned_params,
        "base_compute_config": base_compute_config,
        "param_importance": {k: round(v * 100, 1) for k, v in importances.items()},
        "duration_seconds": round(phase_elapsed, 1),
        "duration_minutes": round(phase_mins, 1),
        "all_trials": [
            {"number": t.number,
             "tps": t.values[0] if is_pareto and t.values else t.value,
             "metrics": t.user_attrs, "params": t.params}
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ],
    }
    if is_pareto:
        results["pareto_front"] = [
            {"tps": t.values[0], "vram_mb": -t.values[1], "quality": t.values[2],
             "params": t.params}
            for t in extract_pareto_front(study)
        ]
    save_phase_results(phase_name, results)

    return returned_params


# ============================================================
# Quality / Sampling
# ============================================================

def phase3(n_trials=80):
    """Optimize sampling params. Server runs with best compute + memory config."""
    if n_trials <= 0:
        return None
    print("\n" + "=" * 60)
    print("  Quality / Sampling")
    print("=" * 60)

    # Build server config from best available results
    # Prefer revalidation results (1b/2b) over initial (1/2)
    server_config = {**ctx.naked_engine}

    # Load compute config (prefer 1c over 1b)
    compute_src = load_phase_results("compute_audit") or load_phase_results("compute")
    if compute_src:
        cp = compute_src["best_params"]
        server_config["threads"] = cp.get("threads")
        server_config["threads_batch"] = cp.get("threads_batch")
        server_config["poll"] = cp.get("poll")
        server_config["poll_batch"] = cp.get("poll_batch")
        server_config["prio"] = cp.get("prio")
        server_config["prio_batch"] = cp.get("prio_batch")
        server_config["cpu_strict"] = cp.get("cpu_strict")
        server_config["cpu_strict_batch"] = cp.get("cpu_strict_batch")
        server_config["spec_type"] = cp.get("spec_type", "ngram-simple")
        server_config["spec_ngram_n"] = cp.get("spec_ngram_n")
        server_config["spec_ngram_m"] = cp.get("spec_ngram_m")
        server_config["spec_ngram_min_hits"] = cp.get("spec_ngram_min_hits")
        server_config["draft_max"] = cp.get("draft_max")
        server_config["draft_min"] = cp.get("draft_min")
        server_config["draft_p_min"] = cp.get("draft_p_min")
        if cp.get("lookup_cache_dynamic"):
            server_config["lookup_cache_dynamic"] = cp["lookup_cache_dynamic"]
        # Also load MoE + expert count from MoE phase
        p1a = load_phase_results("moe_combined")
        if p1a:
            moe_cfg = _get_moe_config(p1a)
            server_config.update(moe_cfg)
        src_name = "Compute Audit" if load_phase_results("compute_audit") else "Compute"
        print(f"\n[*] Compute from {src_name}: t={cp.get('threads')}/{cp.get('threads_batch')} "
              f"moe={server_config.get('n_cpu_moe')} experts={server_config.get('expert_used_count', 8)} "
              f"draft={cp.get('draft_max')}")
    else:
        print("\n[!] No compute results — running without compute tuning.")

    # Load memory config (prefer 2b over 2)
    memory_src = load_phase_results("memory_audit") or load_phase_results("memory")
    if memory_src:
        mp = memory_src["best_params"]
        server_config.update(mp)
        src_name = "Memory Audit" if load_phase_results("memory_audit") else "Memory"
        print(f"[*] Memory from {src_name}: {len(mp)} params")
    else:
        print("[!] No memory results — running with naked engine.")

    # Remove None values
    server_config = {k: v for k, v in server_config.items() if v is not None}

    # Enable concurrent eval slots for async quality measurement
    # Cap parallel to 4: llama-server partitions context across slots, so
    # context=4096 / parallel=10 = only 409 tokens per slot (too small for prompts).
    # With parallel=4 and context>=4096, each slot gets 1024+ tokens.
    if _HAS_AIOHTTP:
        max_parallel = 4
        server_config["parallel"] = min(len(QUALITY_TASKS), max_parallel)
        server_config["context"] = max(server_config.get("context", 4096), 4096)

    # Start server
    print("\n[*] Starting server with best config...")
    proc, status = _boot_server_with_jinja_recovery(server_config)
    if status != "ok":
        print("[!] Server failed to start with combined config")
        return None

    # Baseline quality
    if _HAS_AIOHTTP:
        print(f"    [async] Concurrent quality evals enabled (--parallel {len(QUALITY_TASKS)})")
    print("\n[*] Measuring baseline quality...")
    baseline_score = measure_quality({
        "temperature": 0.4,
        "top_p": 0.95,
        "top_k": 40,
        "min_p": 0.05,
        "repeat_penalty": 1.0,
    })
    print(f"    Baseline: {baseline_score:.1f}% (3-signal: correctness×{QUALITY_WEIGHT_CORRECTNESS:.0%} + confidence×{QUALITY_WEIGHT_CONFIDENCE:.0%} + efficiency×{QUALITY_WEIGHT_EFFICIENCY:.0%})")

    # Multivariate TPE learns correlations between sampling params (e.g., temperature × top_p).
    # It handles mirostat's conditional search space natively — missing params are ignored.
    study, remaining, completed = setup_study("quality", n_trials)
    if remaining == 0:
        return study.best_trial.params

    total_trials = completed + remaining
    best_score = baseline_score
    _sbv = _safe_best_value(study)
    if completed > 0 and _sbv is not None:
        best_score = max(best_score, _sbv)

    def objective(trial):
        nonlocal best_score

        # Mirostat overrides temperature/top_p/top_k/min_p — conditional search space
        mirostat = trial.suggest_categorical("mirostat", [0, 1, 2])
        params = {"mirostat": mirostat}

        if mirostat == 0:
            # Standard samplers (only active when mirostat is off)
            params["temperature"] = trial.suggest_float("temperature", 0.0, 1.5)
            params["top_p"] = trial.suggest_float("top_p", 0.5, 1.0)
            params["top_k"] = trial.suggest_int("top_k", 1, 100)
            params["min_p"] = trial.suggest_float("min_p", 0.0, 0.3)
            params["typical_p"] = trial.suggest_float("typical_p", 0.5, 1.0)
            params["top_n_sigma"] = trial.suggest_float("top_n_sigma", 0.0, 3.0)
            params["dynatemp_range"] = trial.suggest_float("dynatemp_range", 0.0, 1.0)
            params["dynatemp_exp"] = trial.suggest_float("dynatemp_exp", 0.5, 2.0)
        else:
            # Mirostat-specific params (only active when mirostat is on)
            params["mirostat_lr"] = trial.suggest_float("mirostat_lr", 0.01, 0.5)
            params["mirostat_ent"] = trial.suggest_float("mirostat_ent", 1.0, 10.0)

        # Penalties and repetition control — always active
        params["repeat_penalty"] = trial.suggest_float("repeat_penalty", 1.0, 1.3)
        params["repeat_last_n"] = trial.suggest_categorical("repeat_last_n", [0, 32, 64, 128, 256])
        params["presence_penalty"] = trial.suggest_float("presence_penalty", 0.0, 0.5)
        params["frequency_penalty"] = trial.suggest_float("frequency_penalty", 0.0, 0.5)

        # XTC and DRY samplers — always active
        params["xtc_probability"] = trial.suggest_float("xtc_probability", 0.0, 0.5)
        params["xtc_threshold"] = trial.suggest_float("xtc_threshold", 0.01, 0.5)
        params["dry_multiplier"] = trial.suggest_float("dry_multiplier", 0.0, 1.0)
        params["dry_base"] = trial.suggest_float("dry_base", 1.0, 3.0)
        params["dry_allowed_length"] = trial.suggest_int("dry_allowed_length", 1, 5)
        params["dry_penalty_last_n"] = trial.suggest_categorical("dry_penalty_last_n", [-1, 0, 64, 128, 256, 512])

        # Adaptive sampling
        params["adaptive_target"] = trial.suggest_float("adaptive_target", 0.0, 1.0)
        params["adaptive_decay"] = trial.suggest_float("adaptive_decay", 0.0, 1.0)

        # Check for duplicate config before running quality eval
        cached = check_duplicate_trial(trial)
        if cached is not None:
            print(f"  Trial {trial.number}: duplicate config — cached score: {cached:.1f}")
            return cached

        score = measure_quality(params, target_to_beat=best_score)

        marker = ""
        if score > best_score:
            best_score = score
            marker = " *** NEW BEST ***"

        done = trial.number + 1
        pct = done / total_trials * 100
        bar_len = 20
        filled = int(bar_len * done / total_trials)
        bar = "█" * filled + "░" * (bar_len - filled)

        if mirostat == 0:
            detail = (f"temp={params['temperature']:.2f} top_p={params['top_p']:.2f} "
                      f"top_k={params['top_k']:3d} min_p={params['min_p']:.3f}")
        else:
            detail = (f"mirostat={mirostat} lr={params['mirostat_lr']:.3f} "
                      f"ent={params['mirostat_ent']:.1f}")
        print(f"  [{bar}] {pct:5.1f}%  Trial {trial.number:3d}/{total_trials}: {score:5.0f}% | "
              f"{detail}{marker}")

        return score

    print(f"\n[*] Running {remaining} trials (+ {completed} completed)...\n")
    pbar = create_phase_pbar(remaining, desc="Quality")
    callbacks = [TqdmUpdateCallback(), GPStoppingCallback(baseline_score=best_score)]
    study.optimize(objective, n_trials=remaining, callbacks=callbacks, show_progress_bar=False)
    close_phase_pbar()

    best = get_best_trial(study)
    beat_baseline = best.value > baseline_score

    print(f"\n{'=' * 60}")
    print(f"  Quality — RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline:  {baseline_score:.0f}%")
    if beat_baseline:
        print(f"  Best:      {best.value:.0f}% — beats baseline by {best.value - baseline_score:.0f}%")
    else:
        print(f"  Best:      {best.value:.0f}% — BELOW baseline ({baseline_score:.0f}%)")
        print(f"  [!] No trial beat baseline. Using default sampling params.")
    print(f"  Best params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    returned_params = best.params if beat_baseline else {}

    results = {
        "phase": "quality",
        "baseline_score": baseline_score,
        "beat_baseline": beat_baseline,
        "best_score": best.value,
        "best_params": returned_params,
        "eval_tasks": [{"prompt": p, "answer": a, "category": c} for p, a, c in QUALITY_TASKS],
        "all_trials": [
            {"number": t.number, "score": t.value, "params": t.params}
            for t in study.trials
        ],
    }
    save_phase_results("quality", results)

    return returned_params


# ============================================================
# Tensor Split Phase (Multi-GPU)
# ============================================================

def phase_tensor_split(gpus, base_config=None, n_trials=20):
    """Tensor Split: Sweep split ratios across multiple GPUs."""
    gpu_count = len(gpus)
    existing = load_phase_results("tensor_split")
    if existing and "best_split" in existing:
        best_split = tuple(existing["best_split"])
        print(f"\n[*] Tensor Split already complete — split={best_split}")
        return best_split

    if gpu_count < 2:
        print(f"\n[*] Single GPU — skipping tensor split sweep")
        save_phase_results("tensor_split", {"phase": "tensor_split", "best_split": [1.0], "skipped": "single_gpu"})
        return (1.0,)

    if base_config is None:
        base_config = dict(ctx.naked_engine)

    print("\n" + "=" * 60)
    print("  Tensor Split")
    print("=" * 60)

    candidates = generate_tensor_splits(gpu_count)
    even_split = tuple([round(1.0 / gpu_count, 2)] * gpu_count)
    if len(candidates) > n_trials:
        random.shuffle(candidates)
        candidates = candidates[:n_trials - 1]
        if even_split not in candidates:
            candidates.insert(0, even_split)

    print(f"\n[*] Testing {len(candidates)} split ratios across {gpu_count} GPUs\n")

    results = []
    best_score = 0.0
    best_split = even_split

    for trial_num, split in enumerate(candidates, 1):
        split_str = ",".join(str(s) for s in split)
        config = {**base_config, "tensor_split": split_str}
        kill_server()
        proc = start_server(config)
        if wait_for_server(proc=proc) != "ok":
            _server_start_failed(trial_num, f"split={split_str}", proc)
            kill_server()
            continue
        perf, promoted = measure_perf_adaptive(best_score)
        score = compute_score(perf)
        results.append({"split": list(split), "split_str": split_str, "perf": perf, "score": score, "promoted": promoted})
        marker = " *NEW BEST*" if score > best_score else ""
        if score > best_score:
            best_score = score
            best_split = split
        runs_label = "3 runs" if promoted else "1 run"
        print(f"  [{trial_num}] split={split_str}: {perf['tps']:.1f} t/s | Score: {score:.1f} ({runs_label}){marker}")
        kill_server()

    if not results:
        print("[!] All tensor split configs failed. Using even split.")
        best_split = even_split

    best_split_str = ",".join(str(s) for s in best_split)
    print(f"\n  >>> Best tensor split: {best_split_str} (score: {best_score:.1f})")
    save_phase_results("tensor_split", {
        "phase": "tensor_split", "best_split": list(best_split),
        "best_split_str": best_split_str, "best_score": best_score,
        "gpu_count": gpu_count, "all_results": results,
    })
    ctx.naked_engine["tensor_split"] = best_split_str
    return best_split


def phase_topology_sweep(gpus, base_split, base_config=None, step=0.02, n_runs=2):
    """Fine-grained topology sweep: test splits +-10% around the best coarse split.

    Takes the best split from phase_tensor_split and explores nearby ratios
    at finer granularity to find the true optimum.
    """
    gpu_count = len(gpus)
    if gpu_count < 2:
        return base_split

    existing = load_phase_results("topology_sweep")
    if existing and "best_split" in existing:
        best = tuple(existing["best_split"])
        print(f"\n[*] Topology Sweep already complete - split={best}")
        return best

    if base_config is None:
        base_config = dict(ctx.naked_engine)

    print("\n" + "=" * 60)
    print("  Topology Sweep (fine-grained)")
    print("=" * 60)
    print(f"  Base split: {base_split}")

    # Generate fine-grained candidates: vary each GPU's share by +-10% in small steps
    candidates = set()
    candidates.add(tuple(base_split))

    for gpu_idx in range(gpu_count):
        base_val = base_split[gpu_idx]
        low = max(0.05, base_val - 0.10)
        high = min(0.95, base_val + 0.10)
        val = low
        while val <= high + 0.001:
            new_split = list(base_split)
            delta = val - base_val
            new_split[gpu_idx] = round(val, 3)
            # Redistribute the delta across other GPUs proportionally
            other_total = sum(new_split[j] for j in range(gpu_count) if j != gpu_idx)
            if other_total > 0:
                for j in range(gpu_count):
                    if j != gpu_idx:
                        new_split[j] = round(new_split[j] - delta * (new_split[j] / other_total), 3)
                # Ensure all values are positive and sum to ~1.0
                if all(v > 0.01 for v in new_split):
                    total = sum(new_split)
                    normed = tuple(round(v / total, 3) for v in new_split)
                    candidates.add(normed)
            val += step

    candidates = sorted(candidates)
    print(f"  Testing {len(candidates)} fine-grained splits\n")

    best_score = 0.0
    best_split_result = tuple(base_split)
    results = []

    for trial_num, split in enumerate(candidates, 1):
        split_str = ",".join(str(s) for s in split)
        config = {**base_config, "tensor_split": split_str}
        kill_server()
        proc = start_server(config)
        if wait_for_server(proc=proc) != "ok":
            _server_start_failed(trial_num, f"split={split_str}", proc)
            kill_server()
            continue

        perf = measure_perf(runs=n_runs)
        score = compute_score(perf)
        results.append({"split": list(split), "score": score, "perf": perf})
        marker = " *NEW BEST*" if score > best_score else ""
        if score > best_score:
            best_score = score
            best_split_result = split
        print(f"  [{trial_num}/{len(candidates)}] split={split_str}: {perf['tps']:.1f} t/s | Score: {score:.1f}{marker}")
        kill_server()

    best_str = ",".join(str(s) for s in best_split_result)
    print(f"\n  >>> Best topology: {best_str} (score: {best_score:.1f})")
    save_phase_results("topology_sweep", {
        "phase": "topology_sweep", "best_split": list(best_split_result),
        "best_split_str": best_str, "best_score": best_score,
        "gpu_count": gpu_count, "all_results": results,
    })
    ctx.naked_engine["tensor_split"] = best_str
    return best_split_result


# ============================================================
# ============================================================
# Pyramid Pipeline Phases
# Phase 2: Core Engine + I/O — multivariate TPE co-optimizes all correlated params
# Phase 3: Speculative Decoding — isolated after core engine is locked
# Phase 4: KV Cache + Quality — degradation testing with PPL/NIAH
# Phase 5: Workload Simulation — hot-cache TTFT + concurrent load
# ============================================================

def phase_core_engine(n_trials=100):
    """Phase 2: Multivariate co-optimization of all correlated engine parameters.

    Combines compute, memory, and I/O toggles into one phase using Optuna's
    multivariate TPE sampler, which learns cross-parameter correlations
    (e.g., threads×batch_size, flash_attn×batch_size, poll×threads).

    Merging I/O toggles here (instead of a separate phase) lets TPE discover
    interactions like flash_attn×kv_cache_type and cpu_strict×threads that
    coordinate descent would miss.

    For dense models, uses llama-bench for core params (fast), but falls back
    to HTTP server when I/O toggles need real server measurement.
    For MoE, always uses HTTP server (n_cpu_moe requires --override-kv).

    Parameters tuned: threads, threads_batch, batch_size, ubatch_size,
                       flash_attn, mlock, no_mmap, swa_full, repack, op_offload,
                       poll, poll_batch, prio, prio_batch, cpu_strict, cpu_strict_batch,
                       numa (multi-NUMA only), n_cpu_moe (MoE only).
    """
    from .trial_helpers import thermal_gate, recover_best_score
    phase_start_time = time.time()
    is_pareto = _config.get("pareto", False)
    label = "Core Engine (Multivariate TPE)"
    if is_pareto:
        label += " [PARETO]"

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)

    ctx_size = ctx.naked_engine.get("context", 4096)

    # Multivariate TPE learns parameter correlations — the key upgrade
    sampler = optuna.samplers.TPESampler(multivariate=True, seed=42, warn_independent_sampling=False)
    pruner = optuna.pruners.WilcoxonPruner(p_threshold=0.1)
    is_pareto = _config.get("pareto", False)
    study, remaining, completed = setup_study("core_engine", n_trials, sampler_override=sampler, pruner=pruner, is_pareto=is_pareto)
    if remaining == 0:
        best = get_best_trial(study)
        bv = _trial_scalar_value(best) or 0
        print(f"\n  Best Score: {bv:.1f} | TPS: {best.user_attrs.get('tps', 0):.1f}")
        print_param_importance(study)
        return best.params

    # I/O toggles need real server measurement (bench can't test mlock, poll, etc.)
    # so we always use HTTP server for this merged phase.
    use_bench = False
    print("    [*] Using HTTP server (I/O toggles require live server measurement)")

    # Baseline — start with flash_attn on and f16 KV as reference point
    base_config = {**ctx.naked_engine, "flash_attn": "on", "kv_cache_type": "f16"}
    print("\n[*] Starting baseline server...")
    kill_server()
    proc, status = _boot_server_with_jinja_recovery(base_config)
    if status != "ok":
        print("[!] Baseline server failed to start")
        if ctx.fail_fast:
            raise BaselineFailure("Baseline server failed in Core Engine.")
        return None
    baseline = measure_perf(runs=3)
    score_fn = compute_score
    print(f"    Baseline: {baseline['tps']:.1f} t/s | pp: {baseline['prompt_tps']:.0f} t/s | "
          f"TTFT: {baseline['ttft']:.0f}ms | Score: {score_fn(baseline):.1f}")

    # Seed trial 0 with a known-good starting config (includes I/O toggles)
    if completed == 0:
        seed = {
            "threads": min(8, ctx.max_threads),
            "threads_batch": ctx.max_threads,
            "batch_size": 512,
            "ubatch_size": 256,
            # I/O toggles — sensible defaults
            "flash_attn": "on",
            "mlock": True,
            "no_mmap": True,
            "swa_full": False,
            "repack": False,
            "op_offload": False,
            "poll": 0,
            "poll_batch": 50,
            "prio": 0,
            "prio_batch": 0,
            "cpu_strict": 1,
            "cpu_strict_batch": 1,
        }
        if ctx.is_moe:
            seed["n_cpu_moe"] = ctx.moe_sweep_center
        print(f"[*] Seeding Trial 0 with balanced config")
        study.enqueue_trial(seed)

    total_trials = completed + remaining
    best_score = max(score_fn(baseline), recover_best_score(study, score_fn) if completed > 0 else 0)

    # VRAM-aware batch bounds (computed once, not per trial)
    all_batch_opts = [256, 512, 1024, 2048, 4096]
    if ctx.vram_total_mb and ctx.model_size_gb:
        vram_for_kv_mb = ctx.vram_total_mb - (ctx.model_size_gb * 1024) - 512
        if vram_for_kv_mb > 0:
            max_batch = int(vram_for_kv_mb * 1024 / (ctx_size * 0.5)) if ctx_size > 0 else 4096
            max_batch = max(256, min(max_batch, 4096))
            all_batch_opts = [v for v in all_batch_opts if v <= max_batch]
    batch_opts = sorted([v for v in all_batch_opts if v <= ctx_size] or [min(all_batch_opts)])
    ubatch_opts = [128, 256, 512, 1024]

    # Thread options
    thread_opts = sorted(set(list(range(2, ctx.max_threads + 1, 2)) + [ctx.max_threads]))

    def objective(trial):
        nonlocal best_score
        thermal_gate()

        # Core compute params
        threads = trial.suggest_categorical("threads", thread_opts)
        threads_batch = trial.suggest_categorical("threads_batch", thread_opts)
        batch_size = trial.suggest_categorical("batch_size", batch_opts)
        ubatch_size = trial.suggest_categorical("ubatch_size", ubatch_opts)

        # I/O toggles — co-optimized with core params so TPE can learn
        # interactions like flash_attn×batch_size, cpu_strict×threads, poll×threads
        flash_attn = trial.suggest_categorical("flash_attn", ["on", "off"])
        mlock = trial.suggest_categorical("mlock", [True, False])
        no_mmap = trial.suggest_categorical("no_mmap", [True, False])
        swa_full = trial.suggest_categorical("swa_full", [True, False])
        repack = trial.suggest_categorical("repack", [True, False])
        op_offload = trial.suggest_categorical("op_offload", [True, False])
        poll = trial.suggest_categorical("poll", [0, 10, 25, 50, 100])
        poll_batch = trial.suggest_categorical("poll_batch", [0, 10, 25, 50, 100])
        prio = trial.suggest_int("prio", 0, 3)
        prio_batch = trial.suggest_int("prio_batch", 0, 3)
        cpu_strict = trial.suggest_categorical("cpu_strict", [0, 1])
        cpu_strict_batch = trial.suggest_categorical("cpu_strict_batch", [0, 1])

        # Pre-boot pruning
        if ubatch_size > batch_size:
            raise optuna.exceptions.TrialPruned()

        config = {
            **base_config,
            "threads": threads,
            "threads_batch": threads_batch,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            "flash_attn": flash_attn,
            "kv_cache_type": "f16",     # Keep lossless during core (tested in KV Quality phase)
            "mlock": mlock,
            "no_mmap": no_mmap,
            "swa_full": swa_full,
            "repack": repack,
            "op_offload": op_offload,
            "poll": poll,
            "poll_batch": poll_batch,
            "prio": prio,
            "prio_batch": prio_batch,
            "cpu_strict": cpu_strict,
            "cpu_strict_batch": cpu_strict_batch,
            "n_gpu_layers": ctx.default_gpu_layers,
            "fit": True,
        }

        # NUMA awareness — only on multi-NUMA systems
        if ctx.numa_nodes > 1:
            config["numa"] = trial.suggest_categorical("numa", ["distribute", "isolate", "numactl"])

        # MoE: co-optimize n_cpu_moe with threads (the key correlation)
        if ctx.is_moe:
            n_cpu_moe = trial.suggest_int("n_cpu_moe", 1, min(ctx.moe_sweep_max, ctx.max_threads * 2))
            config["n_cpu_moe"] = n_cpu_moe
            config["expert_used_count"] = ctx.default_experts

        config = {k: v for k, v in config.items() if v is not None}

        # Duplicate check
        cached = check_duplicate_trial(trial)
        if cached is not None:
            if isinstance(cached, (list, tuple)):
                print(f"\n  Trial {trial.number}: duplicate — cached TPS: {cached[0]:.1f}")
            else:
                print(f"\n  Trial {trial.number}: duplicate — cached score: {cached:.1f}")
            return cached

        moe_str = f" moe={config.get('n_cpu_moe', '-')}" if ctx.is_moe else ""
        params_short = (f"t={threads}/{threads_batch} b={batch_size} ub={ubatch_size} "
                        f"fa={flash_attn} poll={poll}{moe_str}")

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
            return 0.0 if not is_pareto else (0.0, -99999.0, 0.0)

        # Multi-fidelity gate
        gate = measure_perf_quick_gate(n_predict=5)
        if gate and best_score > 0:
            gate_score = gate.get("gate_score", 0)
            trial.report(gate_score, step=0)
            if trial.should_prune():
                print(f"  Trial {trial.number}: pruned by gate ({gate_score:.1f})")
                raise optuna.exceptions.TrialPruned()

        perf, promoted = measure_perf_adaptive(best_score)

        from .trial_helpers import record_trial_attrs, finalize_trial
        tps = perf["tps"]
        score = score_fn(perf)

        trial.report(score, step=1)

        record_trial_attrs(trial, perf)
        result, best_score = finalize_trial(trial, perf, params_short, best_score, total_trials, is_pareto, score=score)
        return result

    est_minutes = remaining * 25 // 60
    print(f"\n[*] Running {remaining} trials (+ {completed} done, ~{est_minutes} min)...")
    print(f"    Sampler: Multivariate TPE (learns threads×batch×flash_attn×poll correlations)")
    from .trial_helpers import run_study_with_callbacks, print_phase_summary
    run_study_with_callbacks(study, objective, remaining, label, best_score, is_pareto)

    baseline_score = score_fn(baseline)
    param_keys = ["threads", "threads_batch", "batch_size", "ubatch_size",
                  "flash_attn", "mlock", "no_mmap", "swa_full", "repack", "op_offload",
                  "poll", "poll_batch", "prio", "prio_batch",
                  "cpu_strict", "cpu_strict_batch", "n_cpu_moe"]
    returned_params, _ = print_phase_summary(
        "core_engine", study, baseline, baseline_score, phase_start_time,
        is_pareto, score_fn=score_fn, param_keys=param_keys)
    return returned_params


def phase_io_toggles(n_trials=20, base_core_config=None):
    """Phase 3: Categorical sweep of binary OS/memory toggles.

    These flags don't interact heavily with batch/thread sizing —
    flash_attn is on or off, mlock either prevents page faults or doesn't.
    20 trials is plenty to test the important combinations.

    Parameters: flash_attn, mlock, no_mmap, numa, swa_full, repack, op_offload,
                poll, poll_batch, prio, prio_batch, cpu_strict, cpu_strict_batch.
    """
    from .trial_helpers import (
        thermal_gate, run_server_trial, record_trial_attrs, finalize_trial,
        recover_best_score, run_study_with_callbacks, print_phase_summary,
        setup_baseline_server,
    )

    phase_start_time = time.time()
    label = "I/O Toggles"

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)

    is_pareto = _config.get("pareto", False)
    study, remaining, completed = setup_study("io_toggles", n_trials, is_pareto=is_pareto)
    if remaining == 0:
        best = get_best_trial(study)
        bv = _trial_scalar_value(best) or 0
        print(f"\n  Best Score: {bv:.1f} | TPS: {best.user_attrs.get('tps', 0):.1f}")
        print_param_importance(study)
        return best.params

    base_config = {**ctx.naked_engine}
    if base_core_config:
        base_config.update(base_core_config)
        print(f"\n[*] Locked core engine: t={base_core_config.get('threads')}/{base_core_config.get('threads_batch')} "
              f"b={base_core_config.get('batch_size')} ub={base_core_config.get('ubatch_size')}")

    baseline, baseline_score = setup_baseline_server(base_config, "I/O Toggles")
    if baseline is None:
        return None

    if completed == 0:
        study.enqueue_trial({
            "flash_attn": "on", "mlock": True, "no_mmap": True,
            "swa_full": False, "repack": False, "op_offload": False,
            "poll": 0, "poll_batch": 50, "prio": 0, "prio_batch": 0,
            "cpu_strict": 1, "cpu_strict_batch": 1,
        })

    total_trials = completed + remaining
    best_score = max(baseline_score, recover_best_score(study, compute_score) if completed > 0 else 0)

    def objective(trial):
        nonlocal best_score
        thermal_gate()

        config = {
            **base_config,
            "flash_attn": trial.suggest_categorical("flash_attn", ["on", "off"]),
            "mlock": trial.suggest_categorical("mlock", [True, False]),
            "no_mmap": trial.suggest_categorical("no_mmap", [True, False]),
            "swa_full": trial.suggest_categorical("swa_full", [True, False]),
            "repack": trial.suggest_categorical("repack", [True, False]),
            "op_offload": trial.suggest_categorical("op_offload", [True, False]),
            "poll": trial.suggest_categorical("poll", [0, 10, 25, 50, 100]),
            "poll_batch": trial.suggest_categorical("poll_batch", [0, 10, 25, 50, 100]),
            "prio": trial.suggest_int("prio", 0, 3),
            "prio_batch": trial.suggest_int("prio_batch", 0, 3),
            "cpu_strict": trial.suggest_categorical("cpu_strict", [0, 1]),
            "cpu_strict_batch": trial.suggest_categorical("cpu_strict_batch", [0, 1]),
            **({"numa": trial.suggest_categorical("numa", ["distribute", "isolate", "numactl"])}
               if ctx.numa_nodes > 1 else {}),
        }
        config = {k: v for k, v in config.items() if v is not None}

        cached = check_duplicate_trial(trial)
        if cached is not None:
            return cached

        params_short = (f"fa={config['flash_attn']} mlock={config['mlock']} "
                        f"poll={config['poll']}/{config['poll_batch']} "
                        f"prio={config.get('prio', 0)}/{config.get('prio_batch', 0)}")

        perf, score = run_server_trial(trial, config, params_short, best_score, is_pareto)
        if perf is None:
            return (0.0, -99999.0, 0.0) if is_pareto else 0.0
        record_trial_attrs(trial, perf)
        result, best_score = finalize_trial(trial, perf, params_short, best_score, total_trials, is_pareto)
        return result

    print(f"\n[*] Running {remaining} trials (~{remaining * 25 // 60} min)...")
    run_study_with_callbacks(study, objective, remaining, label, best_score, is_pareto)

    returned_params, _ = print_phase_summary(
        "io_toggles", study, baseline, baseline_score, phase_start_time, is_pareto)
    return returned_params


def phase_speculation(n_trials=40, base_config=None):
    """Phase 4: Speculative decoding sweep (N-gram or draft model).

    Isolated after core engine is locked so:
    1. N-gram cache isn't corrupted by batch/thread changes between trials
    2. Speculation is a multiplier on top of already-optimized base TPS
    3. Measurements are stable (only spec params vary)

    Parameters: spec_type, spec_ngram_n, spec_ngram_m, spec_ngram_min_hits,
                draft_max, draft_min, draft_p_min, lookup_cache_dynamic.
    If a draft model is configured, sweeps model_draft params instead.
    """
    from .trial_helpers import (
        thermal_gate, run_server_trial, record_trial_attrs, finalize_trial,
        recover_best_score, run_study_with_callbacks, print_phase_summary,
        setup_baseline_server,
    )

    phase_start_time = time.time()
    label = "Speculative Decoding"

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)

    is_pareto = _config.get("pareto", False)
    study, remaining, completed = setup_study("speculation", n_trials, is_pareto=is_pareto)
    if remaining == 0:
        best = get_best_trial(study)
        print(f"\n  Best Score: {_trial_scalar_value(best):.1f} | TPS: {best.user_attrs.get('tps', 0):.1f}")
        print_param_importance(study)
        return best.params

    if base_config is None:
        base_config = dict(ctx.naked_engine)

    baseline, baseline_score = setup_baseline_server(base_config, "Speculation")
    if baseline is None:
        return None

    draft_model = _config.get("draft_model")
    if draft_model:
        print(f"    Draft model: {Path(draft_model).name}")

    if completed == 0:
        study.enqueue_trial({
            "spec_type": "ngram-map-k4v",
            "spec_ngram_n": 23, "spec_ngram_m": 20,
            "spec_ngram_min_hits": 2,
            "draft_max": 9, "draft_min": 8,
            "draft_p_min": 0.9,
            "use_lookup_cache": True,
        })
        study.enqueue_trial({
            "spec_type": "ngram-cache",
            "spec_ngram_n": 14, "spec_ngram_m": 64,
            "spec_ngram_min_hits": 4,
            "draft_max": 47, "draft_min": 4,
            "draft_p_min": 0.52,
            "use_lookup_cache": False,
        })

    total_trials = completed + remaining
    best_score = max(baseline_score, recover_best_score(study, compute_score) if completed > 0 else 0)

    def objective(trial):
        nonlocal best_score
        thermal_gate()

        spec_opts = ["ngram-simple", "ngram-cache", "ngram-map-k", "ngram-map-k4v", "ngram-mod"]
        if draft_model:
            spec_opts.append("draft")
        spec_type = trial.suggest_categorical("spec_type", spec_opts)
        spec_ngram_n = trial.suggest_int("spec_ngram_n", 2, 24)
        spec_ngram_m = trial.suggest_int("spec_ngram_m", 8, 96)
        spec_ngram_min_hits = trial.suggest_int("spec_ngram_min_hits", 1, 5)
        draft_max = trial.suggest_int("draft_max", 4, 48)
        draft_min = trial.suggest_int("draft_min", 0, 8)
        draft_p_min = trial.suggest_float("draft_p_min", 0.3, 0.99)
        use_lookup_cache = trial.suggest_categorical("use_lookup_cache", [True, False])

        if draft_min >= draft_max:
            raise optuna.exceptions.TrialPruned()

        config = {
            **base_config,
            "spec_type": spec_type,
            "spec_ngram_n": spec_ngram_n,
            "spec_ngram_m": spec_ngram_m,
            "spec_ngram_min_hits": spec_ngram_min_hits,
            "draft_max": draft_max,
            "draft_min": draft_min,
            "draft_p_min": draft_p_min,
        }
        if use_lookup_cache and ctx.lookup_cache_file:
            config["lookup_cache_dynamic"] = ctx.lookup_cache_file
        if draft_model and spec_type == "draft":
            config["model_draft"] = draft_model
        config = {k: v for k, v in config.items() if v is not None}

        cached = check_duplicate_trial(trial)
        if cached is not None:
            return cached

        if use_lookup_cache and ctx.lookup_cache_file and Path(ctx.lookup_cache_file).exists():
            Path(ctx.lookup_cache_file).unlink()

        params_short = (f"{spec_type} n={spec_ngram_n} m={spec_ngram_m} "
                        f"draft={draft_max}/{draft_min} p={draft_p_min:.2f}")

        perf, score = run_server_trial(trial, config, params_short, best_score, is_pareto)
        if perf is None:
            return (0.0, -99999.0, 0.0) if is_pareto else 0.0
        record_trial_attrs(trial, perf)
        result, best_score = finalize_trial(trial, perf, params_short, best_score, total_trials, is_pareto)
        return result

    print(f"\n[*] Running {remaining} trials (~{remaining * 20 // 60} min)...")
    run_study_with_callbacks(study, objective, remaining, label, best_score, is_pareto)

    returned_params, _ = print_phase_summary(
        "speculation", study, baseline, baseline_score, phase_start_time, is_pareto)
    return returned_params


def phase_kv_quality(n_trials=15, base_config=None):
    """Phase 5: KV cache quantization + expert count with PPL/NIAH quality gates.

    Tries to squeeze out extra VRAM/speed by dropping KV cache precision,
    and (for MoE) reducing active experts. Strictly punishes quality loss
    using True Perplexity and mini-NIAH recall tests.

    Parameters: kv_cache_type, expert_used_count (if MoE).
    """
    phase_start_time = time.time()
    label = "KV Cache + Quality"

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)

    is_pareto = _config.get("pareto", False)
    study, remaining, completed = setup_study("kv_quality", n_trials, is_pareto=is_pareto)
    if remaining == 0:
        best = get_best_trial(study)
        bv = _trial_scalar_value(best) or 0
        print(f"\n  Best Score: {bv:.1f}")
        return best.params

    if base_config is None:
        base_config = dict(ctx.naked_engine)

    # Baseline with f16 KV (lossless reference)
    base_config["kv_cache_type"] = "f16"
    base_config["flash_attn"] = base_config.get("flash_attn", "on")

    print("\n[*] Starting baseline server (f16 KV)...")
    kill_server()
    proc, status = _boot_server_with_jinja_recovery(base_config)
    if status != "ok":
        print("[!] Baseline failed")
        if ctx.fail_fast:
            raise BaselineFailure("Baseline server failed in KV Quality.")
        return None
    baseline = measure_perf(runs=3)
    baseline_score = compute_score(baseline)
    print(f"    Baseline (f16 KV): {baseline['tps']:.1f} t/s | Score: {baseline_score:.1f}")

    # Measure baseline PPL for degradation comparison
    print("    Measuring baseline perplexity...")
    baseline_ppl = measure_true_perplexity()
    if baseline_ppl != float('inf'):
        print(f"    Baseline PPL: {baseline_ppl:.2f}")
    else:
        print("    [!] PPL measurement failed — will use NIAH only")
        baseline_ppl = None

    # Populate KL-divergence baseline for MoE expert count penalty
    if ctx.is_moe:
        print("    Measuring baseline KL-Divergence distribution...")
        ctx.kl_baseline_cache, _ = measure_kl_divergence()

    if completed == 0:
        study.enqueue_trial({"kv_cache_type": "q8_0",
                             **({"expert_used_count": ctx.default_experts} if ctx.is_moe else {})})

    total_trials = completed + remaining
    best_score = baseline_score

    # Expert count options for MoE
    expert_opts = None
    if ctx.is_moe:
        expert_opts = sorted(set([ctx.default_experts,
                                  max(1, ctx.default_experts // 2),
                                  max(1, ctx.default_experts - 2),
                                  min(ctx.max_experts, ctx.default_experts + 2),
                                  min(ctx.max_experts, ctx.default_experts * 2)]))

    def objective(trial):
        nonlocal best_score

        if check_thermal_throttle(threshold=85)[0]:
            wait_for_cooldown(target_temp=75, timeout=120)

        kv_cache_type = trial.suggest_categorical("kv_cache_type", ["f16", "q8_0", "q5_1", "q4_0"])

        config = {**base_config, "kv_cache_type": kv_cache_type}

        # Quantized KV requires flash attention
        if kv_cache_type not in ("f16", "bf16") and config.get("flash_attn") == "off":
            config["flash_attn"] = "on"

        # MoE: sweep expert count
        if ctx.is_moe and expert_opts:
            expert_count = trial.suggest_categorical("expert_used_count", expert_opts)
            config["expert_used_count"] = expert_count

        config = {k: v for k, v in config.items() if v is not None}

        cached = check_duplicate_trial(trial)
        if cached is not None:
            return cached

        expert_str = f" experts={config.get('expert_used_count', '-')}" if ctx.is_moe else ""
        params_short = f"kv={kv_cache_type}{expert_str}"

        print(f"\n  Trial {trial.number}: server | {params_short}")
        kill_server()
        proc = start_server(config)
        status = wait_for_server(proc=proc)
        if status == "oom":
            kill_server()
            raise optuna.exceptions.TrialPruned()
        elif status != "ok":
            _server_start_failed(trial.number, params_short, proc)
            return 0.0

        perf, promoted = measure_perf_adaptive(best_score)
        tps = perf["tps"]
        score = compute_score(perf)
        quality_factor = 1.0

        # Quality gates for quantized KV types
        if kv_cache_type not in ("f16", "bf16"):
            # PPL quality gate
            if baseline_ppl is not None:
                trial_ppl = measure_true_perplexity()
                trial.set_user_attr("ppl", round(trial_ppl, 2) if trial_ppl != float('inf') else -1)
                if trial_ppl != float('inf'):
                    pf = ppl_quality_factor(baseline_ppl, trial_ppl)
                    trial.set_user_attr("ppl_factor", round(pf, 3))
                    ppl_pct = ((trial_ppl - baseline_ppl) / baseline_ppl * 100) if baseline_ppl > 0 else 0
                    print(f"  Trial {trial.number}: PPL {trial_ppl:.2f} ({ppl_pct:+.1f}%) → factor {pf:.2f}")
                    score *= pf
                    quality_factor *= pf

            # Mini-NIAH recall gate
            niah_passed = _mini_niah_recall_test()
            trial.set_user_attr("niah_recall", niah_passed)
            if not niah_passed:
                score *= 0.15
                quality_factor *= 0.15
                print(f"  Trial {trial.number}: [!] mini-NIAH FAIL — {kv_cache_type} broke recall")

        # MoE: penalize reduced experts via KL-divergence
        if ctx.is_moe and config.get("expert_used_count", ctx.default_experts) < ctx.default_experts:
            if ctx.kl_baseline_cache is not None:
                _, kl_div = measure_kl_divergence(baseline_cache=ctx.kl_baseline_cache)
                if kl_div is not None:
                    kf = kl_quality_factor(kl_div)
                    trial.set_user_attr("kl_div", round(kl_div, 4))
                    trial.set_user_attr("kl_factor", round(kf, 3))
                    score *= kf
                    quality_factor *= kf
                    print(f"  Trial {trial.number}: KL-div {kl_div:.4f} → factor {kf:.2f}")

        trial.set_user_attr("tps", tps)
        trial.set_user_attr("ttft", perf.get("ttft", 0))
        trial.set_user_attr("prompt_tps", perf.get("prompt_tps", 0))
        trial.set_user_attr("total_ms", perf.get("total_ms", 0))

        if is_pareto:
            objectives = compute_pareto_objectives(perf, quality_factor=quality_factor)
            print(f"  Trial {trial.number}: TPS={tps:.1f} VRAM={perf.get('vram_used_mb', 0):.0f}MB QF={quality_factor:.2f} | {params_short}")
            return objectives

        best_score = print_trial_result(trial.number, total_trials, tps, perf, params_short, best_score,
                                        final_score=score)
        return score

    print(f"\n[*] Running {remaining} trials (~{remaining * 45 // 60} min)...")
    pbar = create_phase_pbar(remaining, desc=label)
    is_pareto = _config.get("pareto", False)
    callbacks = [TqdmUpdateCallback()]
    if not is_pareto:
        callbacks.append(GPStoppingCallback(baseline_score=best_score))
    study.optimize(objective, n_trials=remaining, callbacks=callbacks, show_progress_bar=False)
    close_phase_pbar()

    best = get_best_trial(study)

    print(f"\n{'=' * 60}")
    print(f"  KV Cache + Quality — RESULTS")
    print(f"{'=' * 60}")
    print(f"  Baseline (f16): {baseline['tps']:.1f} t/s | PPL: {baseline_ppl:.2f}" if baseline_ppl else
          f"  Baseline (f16): {baseline['tps']:.1f} t/s")

    best_val = best.values[0] if is_pareto else best.value
    beat_baseline = best_val > baseline_score
    if beat_baseline:
        print(f"  Best Score:  {best_val:.1f} — beats baseline by {best_val - baseline_score:.1f}")
        returned_params = best.params
    else:
        print(f"  Best Score:  {best_val:.1f} — below baseline, keeping f16 KV defaults")
        returned_params = {"kv_cache_type": "f16"}
        if ctx.is_moe:
            returned_params["expert_used_count"] = ctx.default_experts

    best_kv = returned_params.get("kv_cache_type", "f16")
    print(f"  Best KV type: {best_kv}")
    print(f"  Best TPS:     {best.user_attrs.get('tps', 0):.1f}")
    if best.user_attrs.get("ppl"):
        print(f"  Best PPL:     {best.user_attrs['ppl']}")
    for k, v in returned_params.items():
        print(f"    {k}: {v}")

    phase_elapsed = time.time() - phase_start_time
    print(f"\n  Duration:    {phase_elapsed / 60:.1f} min")

    results = {
        "phase": "kv_quality",
        "baseline": baseline,
        "baseline_ppl": baseline_ppl,
        "best_tps": best.user_attrs.get("tps", 0),
        "best_params": returned_params,
        "duration_minutes": round(phase_elapsed / 60, 1),
        "all_trials": [
            {"number": t.number, "tps": _trial_scalar_value(t), "metrics": t.user_attrs, "params": t.params}
            for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ],
    }
    save_phase_results("kv_quality", results)
    return returned_params


def phase_workload_sim(base_config=None):
    """Phase 6: Workload simulation — hot-cache TTFT + concurrent load test.

    No Optuna trials. Measures production-readiness metrics:
    1. Hot-cache TTFT: Time to first token when 95% of prompt is already cached
    2. Concurrent load: System throughput with N simultaneous users
    """
    label = "Workload Simulation"

    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)

    if base_config is None:
        base_config = dict(ctx.naked_engine)

    # Start server with full optimized config + cache-reuse enabled
    config = {**base_config, "cache_reuse": 256}
    print("\n[*] Starting server with cache-reuse enabled...")
    kill_server()
    proc, status = _boot_server_with_jinja_recovery(config)
    if status != "ok":
        print("[!] Server failed to start for workload simulation")
        return None

    results = {}

    # 1. Hot-cache TTFT test
    print("\n  [HOT CACHE] Measuring TTFT on cached prompt...")
    system_prompt = (
        "You are a helpful AI assistant specialized in software engineering. "
        "You write clean, efficient, well-documented code. You follow best practices "
        "for error handling, testing, and performance optimization. You communicate "
        "clearly and concisely, focusing on practical solutions."
    )
    # First request: prime the cache (cold)
    try:
        r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Explain the difference between a mutex and a semaphore."},
            ],
            "max_tokens": 100,
            "temperature": 0.0,
        }, timeout=120)
        if r.status_code == 200:
            timings = r.json().get("timings", {})
            cold_ttft = timings.get("prompt_ms", 0)
            print(f"    Cold TTFT: {cold_ttft:.0f}ms (prompt processing)")
            results["cold_ttft_ms"] = round(cold_ttft, 1)
    except Exception as e:
        print(f"    [!] Cold request failed: {e}")

    # Second request: same system prompt, different user query (hot cache)
    hot_ttfts = []
    for i, query in enumerate([
        "How do I implement a binary search tree in Python?",
        "What are the SOLID principles? Give a brief example of each.",
        "Explain the CAP theorem and its practical implications.",
    ]):
        try:
            r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                "max_tokens": 100,
                "temperature": 0.0,
            }, timeout=120)
            if r.status_code == 200:
                timings = r.json().get("timings", {})
                hot_ttft = timings.get("prompt_ms", 0)
                hot_ttfts.append(hot_ttft)
                print(f"    Hot TTFT #{i + 1}: {hot_ttft:.0f}ms")
        except Exception:
            pass

    if hot_ttfts:
        avg_hot = sum(hot_ttfts) / len(hot_ttfts)
        results["hot_ttft_avg_ms"] = round(avg_hot, 1)
        results["hot_ttft_min_ms"] = round(min(hot_ttfts), 1)
        results["hot_ttft_max_ms"] = round(max(hot_ttfts), 1)
        print(f"    Hot TTFT avg: {avg_hot:.0f}ms (min={min(hot_ttfts):.0f}, max={max(hot_ttfts):.0f})")
        if results.get("cold_ttft_ms"):
            speedup = results["cold_ttft_ms"] / avg_hot if avg_hot > 0 else 0
            print(f"    Cache speedup: {speedup:.1f}x")
            results["cache_speedup"] = round(speedup, 2)

    # 2. Concurrent load test
    n_users = _config.get("simulate_users", 4)
    if n_users > 0 and _HAS_AIOHTTP:
        print(f"\n  [LOAD TEST] Simulating {n_users} concurrent users...")
        load_results = measure_concurrent_load(n_users=n_users, n_predict=50)
        if load_results:
            results["concurrent"] = load_results
            print(f"    Total throughput: {load_results.get('concurrent_total_tps', 0):.1f} t/s")
            print(f"    Per-user avg:     {load_results.get('concurrent_avg_tps', 0):.1f} t/s")
            print(f"    Avg latency:      {load_results.get('concurrent_avg_wall_ms', 0):.0f}ms")
    elif n_users > 0:
        print(f"\n  [LOAD TEST] Skipped — aiohttp not available")

    kill_server()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Workload Simulation — RESULTS")
    print(f"{'=' * 60}")
    if results.get("hot_ttft_avg_ms"):
        verdict = "EXCELLENT" if results["hot_ttft_avg_ms"] < 200 else \
                  "GOOD" if results["hot_ttft_avg_ms"] < 500 else \
                  "SLOW" if results["hot_ttft_avg_ms"] < 1000 else "POOR"
        print(f"  Hot-cache TTFT: {results['hot_ttft_avg_ms']:.0f}ms [{verdict}]")
    if results.get("concurrent"):
        print(f"  Concurrent ({n_users} users): {results['concurrent'].get('concurrent_total_tps', 0):.1f} t/s total")

    save_phase_results("workload_sim", results)
    return results


# ============================================================
# Context Size Sweep
# ============================================================

def phase_context_sweep(base_config=None, contexts=None, n_runs=3):
    """Test which context sizes the model can handle and measure TPS for each."""
    if contexts is None:
        contexts = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    if base_config is None:
        base_config = dict(ctx.naked_engine)

    print("\n" + "=" * 60)
    print("  CONTEXT SIZE SWEEP")
    print(f"  Testing: {contexts}")
    print("=" * 60)

    # Build filler sentence for KV cache pressure testing
    _FILLER_SENTENCE = "The quick brown fox jumps over the lazy dog. "  # ~10 tokens

    results = {}
    for ctx_size in contexts:
        # Build a prompt that fills ~75% of the context window so KV cache is under real pressure
        fill_tokens = int(ctx_size * 0.75)
        repeats = max(1, fill_tokens // 10)
        filler = _FILLER_SENTENCE * repeats
        fill_prompt = f"Summarize the following text in one sentence:\n{filler}\nSummary:"
        print(f"\n  Testing context={ctx_size:,} (filling ~{fill_tokens:,} tokens)...")

        engine_config = {**base_config, "context": ctx_size}
        # Clamp batch params to context — llama-server asserts batch_size <= context
        if engine_config.get("batch_size", 512) > ctx_size:
            engine_config["batch_size"] = ctx_size
        if engine_config.get("ubatch_size", 128) > engine_config.get("batch_size", ctx_size):
            engine_config["ubatch_size"] = engine_config["batch_size"]
        # Clamp speculation draft tokens — draft >= context crashes llama.cpp
        if engine_config.get("draft_max", 0) >= ctx_size:
            engine_config["draft_max"] = max(1, ctx_size // 4)
            # Safety: clamp draft_min so it never exceeds draft_max (C++ assert crash)
            if engine_config.get("draft_min", 0) >= engine_config["draft_max"]:
                engine_config["draft_min"] = max(0, engine_config["draft_max"] - 1)
        kill_server()
        proc = start_server(engine_config)
        if wait_for_server(proc=proc, timeout=600) != "ok":
            print(f"    context={ctx_size:,}: DOES NOT FIT")
            kill_server()
            results[ctx_size] = {"fits": False, "tps": 0.0, "score": 0.0, "prompt_tps": 0.0}
            continue
        perf = measure_perf(n_predict=50, runs=n_runs, prompt=fill_prompt)
        score = compute_score(perf)
        pp_tps = perf.get("prompt_tps", 0.0)
        results[ctx_size] = {"fits": True, "tps": round(perf["tps"], 2), "score": round(score, 2), "prompt_tps": round(pp_tps, 2)}
        print(f"    context={ctx_size:,}: {perf['tps']:.1f} t/s gen | {pp_tps:.1f} t/s PP (score: {score:.1f})")
        kill_server()

    print("\n  Context Sweep Results:")
    print(f"  {'Context':>10s}  {'Fits':>5s}  {'Gen t/s':>8s}  {'PP t/s':>8s}  {'Score':>8s}")
    print("  " + "-" * 50)
    for ctx_size in contexts:
        r = results[ctx_size]
        fits_str = "yes" if r["fits"] else "NO"
        tps_str = f"{r['tps']:.1f}" if r["fits"] else "-"
        pp_str = f"{r.get('prompt_tps', 0):.1f}" if r["fits"] else "-"
        score_str = f"{r['score']:.1f}" if r["fits"] else "-"
        print(f"  {ctx_size:>10,}  {fits_str:>5s}  {tps_str:>8s}  {pp_str:>8s}  {score_str:>8s}")

    save_phase_results("context_sweep", {"contexts": {str(k): v for k, v in results.items()}, "n_runs": n_runs})
    return results


# ============================================================
# Batch Pipeline — optimize multiple models
# ============================================================

def batch_optimize(models_dir, preset="normal", skip_existing=False, timeout_minutes=0, interactive=False):
    """Run full optimization pipeline on every GGUF in a directory."""

    models_path = Path(models_dir)
    if not models_path.is_dir():
        print(f"[!] Not a directory: {models_dir}")
        return

    gguf_files = sorted(models_path.rglob("*.gguf"))
    gguf_files = [f for f in gguf_files
                  if "mmproj" not in f.name.lower()
                  and "embedding" not in f.parent.name.lower()
                  and "reranker" not in f.parent.name.lower()]

    if not gguf_files:
        print(f"[!] No GGUF model files found in {models_dir}")
        return

    total = len(gguf_files)
    print("\n" + "=" * 60)
    print("  BATCH OPTIMIZATION")
    print(f"  Models: {total} | Skip existing: {skip_existing}")
    if timeout_minutes: print(f"  Per-model timeout: {timeout_minutes} min")
    print("=" * 60)

    batch_timer = PhaseTimer()
    batch_timer.start_phase("batch_total")
    results_summary = []

    for idx, gguf_path in enumerate(gguf_files):
        model_name = gguf_path.name
        print(f"\n{'=' * 60}")
        print(f"  [Model {idx + 1}/{total}] {model_name}")
        print(f"{'=' * 60}")

        model_stem = gguf_path.stem.lower().replace(" ", "-")
        per_model_results = gguf_path.parent / f"optimize-results-{model_stem}"

        if skip_existing and per_model_results.is_dir() and list(per_model_results.glob("*_results.json")):
            print(f"  Skipping — results already exist")
            results_summary.append({"model": model_name, "status": "skipped"})
            continue

        model_start = time.time()
        ctx.quality_baseline = None  # CRITICAL: Reset quality baseline for each model
        ctx.kl_baseline_cache = None  # Reset KL baseline — Model #2 must not inherit Model #1's distribution
        ctx._flash_attn_disabled_for_kv = False  # Reset flash attn state for clean start
        ctx.no_jinja = False  # Reset jinja recovery state for clean model
        ctx.naked_engine = {"context": 4096, "mlock": True, "n_gpu_layers": ctx.default_gpu_layers}  # Reset to defaults
        try:
            arch_info = _detect_gguf_architecture(str(gguf_path))
            print(f"  Architecture: {arch_info['type']}")

            ctx.model_path = gguf_path
            _config["model"] = str(ctx.model_path)
            model_class, model_size = classify_model(str(ctx.model_path))
            ctx.model_size_class = model_class
            ctx.model_size_gb = model_size
            ctx.arch = arch_info
            _config["architecture"] = ctx.arch
            ctx.is_moe = ctx.arch["type"] == "moe"
            ctx.expert_override_key = ctx.arch.get("expert_override_key", "")
            ctx.default_experts = ctx.arch.get("default_experts", 8)
            ctx.max_experts = ctx.arch.get("max_experts", 16)

            detected = detect_model_layers(str(ctx.model_path))
            ctx.max_gpu_layers = detected or 99
            ctx.default_gpu_layers = ctx.max_gpu_layers
            ctx.naked_engine["n_gpu_layers"] = ctx.default_gpu_layers

            per_model_results.mkdir(parents=True, exist_ok=True)
            ctx.results_dir = per_model_results
            ctx.lookup_cache_file = str(ctx.results_dir / "lookup-cache.bin")
            ctx.optuna_db = "sqlite:///" + str(ctx.results_dir / "optuna.db").replace("\\", "/")
            _config["results_dir"] = str(ctx.results_dir)
            ensure_results_dir()

            gpu_data = load_phase_results("gpu")
            if gpu_data and "best_ngl" in gpu_data:
                ctx.default_gpu_layers = gpu_data["best_ngl"]
                ctx.naked_engine["n_gpu_layers"] = ctx.default_gpu_layers

            p = _config.get("preset", "normal")
            deadline = (time.time() + (timeout_minutes * 60)) if timeout_minutes and timeout_minutes > 0 else None
            run_full_pipeline(deadline=deadline)
            results_summary.append({"model": model_name, "status": "completed", "duration": time.time() - model_start})
        except KeyboardInterrupt:
            print(f"\n  [!] Batch interrupted at {model_name}")
            kill_server()
            results_summary.append({"model": model_name, "status": "interrupted"})
            break
        except BaselineFailure as e:
            print(f"\n  [!] {e} Skipping model.")
            kill_server()
            results_summary.append({"model": model_name, "status": "failed (baseline)"})
        except Exception as e:
            print(f"\n  [!] Error optimizing {model_name}: {e}")
            kill_server()
            results_summary.append({"model": model_name, "status": "error", "error": str(e)})
        finally:
            kill_server()

        dur = time.time() - model_start
        batch_timer.record_trial(dur)
        remaining = total - (idx + 1)
        if remaining > 0:
            print(f"\n  ETA for remaining {remaining} model(s): {batch_timer.eta(remaining)}")
        if interactive and idx < total - 1:
            input("\n  Press Enter to continue to next model...")

    batch_timer.end_phase("batch_total")
    print("\n" + "=" * 60)
    print("  BATCH OPTIMIZATION COMPLETE")
    print("=" * 60)
    for entry in results_summary:
        dur_str = ""
        if "duration" in entry:
            d = entry["duration"]
            dur_str = f" ({d / 60:.1f}m)" if d >= 60 else f" ({d:.0f}s)"
        print(f"  {entry['model']:40s} {entry['status']}{dur_str}")
    batch_timer.summary()


# ============================================================
# Full Pipeline
# ============================================================

def run_full_pipeline(trials_moe=50, trials_p1b=60, trials_p2=60, trials_p1c=60, trials_p2b=60, trials_p3=80, deadline=None, resume_from=0):
    """Run the Pyramid Pipeline: Hardware → Core Engine → Accelerators → Quality.

    This replaces the old sequential Compute → Memory → Audit flow with a
    multivariate co-optimization that finds the global optimum in fewer trials.

    Phase 1: VRAM Boundaries (GPU offload, tensor split)
    Phase 2: Core Engine + I/O (threads × batch × flash_attn × poll × ..., multivariate TPE)
    Phase 3: Speculative Decoding (N-gram or draft model)
    Phase 4: KV Cache + Quality (PPL/NIAH quality gates)
    Phase 5: Workload Simulation (hot-cache TTFT, concurrent load)
    Phase 6: Quality/Sampling (temperature, top_p, mirostat)
    """
    p = _config.get("preset", "normal")
    # Scale trial counts by preset
    preset_scale = {"quick": 0.5, "normal": 1.0, "thorough": 1.5}
    scale = preset_scale.get(p, 1.0)
    # Core engine now includes I/O toggles (merged) — more params needs more trials
    t_core = max(60, int(100 * scale))
    t_spec = max(20, int(40 * scale))
    t_kv = max(8, int(15 * scale))
    t_quality = max(30, int(60 * scale))
    total_est = t_core + t_spec + t_kv + t_quality

    print("\n" + "=" * 60)
    print("  PYRAMID OPTIMIZATION PIPELINE")
    print(f"  Phase 1: VRAM Boundaries    (GPU offload + tensor split)")
    print(f"  Phase 2: Core Engine + I/O  ({t_core} trials, multivariate TPE)")
    print(f"  Phase 3: Speculation         ({t_spec} trials)")
    print(f"  Phase 4: KV Cache + Quality  ({t_kv} trials, PPL/NIAH)")
    print(f"  Phase 5: Workload Sim        (hot-cache + load test)")
    print(f"  Phase 6: Quality/Sampling    ({t_quality} trials)")
    print(f"  Total:   ~{total_est} trials  [{p}]")
    print("=" * 60)

    if check_dry_run("Full Pipeline", {"phases": 7, "total_trials": total_est}, "all"):
        return

    print("\n  Tip: Press Ctrl+C to skip the current phase\n")

    interactive = _config.get("interactive", False)
    pipeline_timer = PhaseTimer()
    pipeline_timer.start_phase("pipeline_total")

    phase_names = ["GPU Offload", "Tensor Split", "Core Engine",
                   "Speculation", "KV Quality", "Workload Sim", "Context Sweep",
                   "NIAH", "Quality"]

    def _run_phase(name, fn):
        """Run a phase, catching Ctrl+C to skip. Tracks timing."""
        if deadline and time.time() > deadline:
            print(f"\n[!] Model timeout reached. Skipping {name}.")
            return None
        pipeline_timer.start_phase(name)
        phase_start = time.time()
        try:
            result = fn()
            return result
        except KeyboardInterrupt:
            print(f"\n\n[!] {name} skipped (Ctrl+C)")
            kill_server()
            return None
        except Exception as e:
            print(f"\n\n[!] {name} failed with error: {e}")
            kill_server()
            return None
        finally:
            pipeline_timer.end_phase(name)
            dur = time.time() - phase_start
            pipeline_timer.record_trial(dur)
            phase_idx = phase_names.index(name) if name in phase_names else -1
            remaining_names = phase_names[phase_idx + 1:] if phase_idx >= 0 else []
            if remaining_names:
                eta_str = pipeline_timer.eta(len(remaining_names))
                print(f"\n  [{name} done in {dur / 60:.1f}m] "
                      f"ETA for {len(remaining_names)} remaining: {eta_str}")
            if interactive:
                input(f"\n  --interactive: Press Enter to continue...")

    def _skip(phase_idx, name):
        if phase_idx < resume_from:
            print(f"\n  [skip] {name} — already complete (resuming)")
            return True
        return False

    # ── Phase 1: VRAM Boundaries ──
    if not _skip(0, "GPU Offload"):
        _run_phase("GPU Offload", phase_gpu_offload)

    gpus = detect_gpus()
    if len(gpus) >= 2 and not _skip(0, "Tensor Split"):
        _run_phase("Tensor Split", lambda: phase_tensor_split(gpus))

    # ── Phase 2: Core Engine + I/O Toggles (merged) ──
    core_best = None
    if not _skip(1, "Core Engine"):
        core_best = _run_phase("Core Engine", lambda: phase_core_engine(n_trials=t_core))
    if core_best is None:
        core_data = load_phase_results("core_engine")
        core_best = core_data["best_params"] if core_data else {}
        # Also check for legacy io_toggles results from previous runs
        io_data = load_phase_results("io_toggles")
        if io_data and "best_params" in io_data:
            core_best.update(io_data["best_params"])

    # Build accumulating config
    best_config = {**ctx.naked_engine}
    best_config.update(core_best)

    # ── Phase 3: Speculative Decoding ──
    spec_best = None
    if not _skip(2, "Speculation"):
        spec_best = _run_phase("Speculation", lambda: phase_speculation(n_trials=t_spec, base_config=best_config))
    if spec_best is None:
        spec_data = load_phase_results("speculation")
        spec_best = spec_data["best_params"] if spec_data else {}
    # Only apply spec params if they beat baseline (empty dict = no spec)
    if spec_best:
        best_config.update(spec_best)

    # ── Phase 4: KV Cache + Quality ──
    kv_best = None
    if not _skip(3, "KV Quality"):
        kv_best = _run_phase("KV Quality", lambda: phase_kv_quality(n_trials=t_kv, base_config=best_config))
    if kv_best is None:
        kv_data = load_phase_results("kv_quality")
        kv_best = kv_data["best_params"] if kv_data else {}
    best_config.update(kv_best)

    # ── Phase 5: Workload Simulation ──
    if not _skip(4, "Workload Sim"):
        _run_phase("Workload Sim", lambda: phase_workload_sim(base_config=best_config))

    # ── Context Sweep (auto mode) ──
    if _config.get("target_context"):
        print(f"\n[*] Context locked to {_config['target_context']:,} — skipping context sweep.")
    elif not _skip(5, "Context Sweep"):
        sweep_results = _run_phase("Context Sweep", lambda: phase_context_sweep(base_config=best_config))
        if sweep_results:
            peak_tps = max((r["tps"] for r in sweep_results.values() if r.get("fits")), default=0)
            if peak_tps > 0:
                viable = [(int(k), r) for k, r in sweep_results.items()
                          if r.get("fits") and r["tps"] >= peak_tps * 0.80]
                if viable:
                    best_ctx = max(viable, key=lambda x: x[0])[0]
                    ctx.naked_engine["context"] = best_ctx
                    best_config["context"] = best_ctx
                    print(f"  [*] Auto-selected context: {best_ctx:,} (≥80% of peak {peak_tps:.1f} t/s)")

    # ── NIAH ──
    if not _skip(6, "NIAH"):
        _run_phase("NIAH", lambda: phase_niah(base_config=best_config))

    # ── Phase 6: Quality/Sampling ──
    if _config.get("skip_quality") or t_quality <= 0:
        print("\n[*] Skipping Quality/sampling phase.")
    elif not _skip(7, "Quality"):
        _run_phase("Quality", lambda: phase3(n_trials=t_quality))

    pipeline_timer.end_phase("pipeline_total")

    # Final summary
    print("\n" + "=" * 60)
    print("  PYRAMID PIPELINE COMPLETE")
    print("=" * 60)
    for name in ["gpu", "core_engine", "io_toggles", "speculation", "kv_quality",
                  "workload_sim", "context_sweep", "niah", "quality"]:
        data = load_phase_results(name)
        if data:
            if "best_ngl" in data:
                print(f"  {name:16s}: n_gpu_layers={data['best_ngl']}")
            elif "contexts" in data:
                ctxs = data["contexts"]
                viable = [int(k) for k, v in ctxs.items() if v.get("fits")]
                max_ctx = max(viable) if viable else 0
                print(f"  {name:16s}: max context={max_ctx:,}")
            elif "kv_results" in data:
                safe_kvs = [r["kv_type"] for r in data["kv_results"]
                            if r.get("pass_rate", 0) >= data.get("reference_pass_rate", 100) - 5]
                print(f"  {name:16s}: safe KV types: {', '.join(safe_kvs)}")
            elif "hot_ttft_avg_ms" in data:
                print(f"  {name:16s}: hot TTFT={data['hot_ttft_avg_ms']:.0f}ms")
            elif "best_tps" in data:
                print(f"  {name:16s}: {data['best_tps']:.1f} t/s")
            elif "best_score" in data:
                print(f"  {name:16s}: {data['best_score']:.0f}% quality")

    pipeline_timer.summary()

    # Generate optimized command and HTML report (lazy import to avoid circular dependency)
    try:
        from .main import generate_command
        generate_command()
    except Exception as e:
        print(f"  [!] Could not generate command: {e}")

    try:
        from .main import generate_html_report
        generate_html_report()
    except Exception as e:
        print(f"  [!] Could not generate HTML report: {e}")

    print("=" * 60)


# ============================================================
# Interactive Terminal Menu
# ============================================================
