"""CLI entry point, interactive menus, and HTML report generation."""
import copy
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import optuna
import requests

from .state import ctx, _config, _DEFAULTS, create_context, _detect_numa_nodes, initialize
from .models import detect_model_layers
from .constants import SCORE_VERSION
from .hardware import detect_gpus
from .engine import (
    start_server, kill_server, wait_for_server, LogTee, PhaseTimer,
    check_dry_run, BaselineFailure,
)
from .models import classify_model, _detect_gguf_architecture
from .search import load_phase_results, save_phase_results, _trial_scalar_value
from .pipeline import (
    phase_gpu_offload, phase_tensor_split, phase_core_engine,
    phase_io_toggles, phase_speculation, phase_kv_quality,
    phase_workload_sim, phase_context_sweep, phase_moe,
    phase_experts, phase_moe_revalidate, phase_compute,
    phase_memory, phase3, batch_optimize, run_full_pipeline,
    phase_moe_threads, _get_moe_config,
)
from .evals import phase_niah, phase_reasoning_eval, phase_integrity_eval

_dashboard_proc = None


# ============================================================
# Pipeline Resume
# ============================================================

# Ordered pipeline phases: (display_name, results_key, study_key, preset_key)
_PIPELINE_PHASES = [
    # (display_name, results_key, study_key, preset_key)
    ("GPU Offload",    "gpu",            None,              None),
    ("Core Engine",    "core_engine",    "core_engine",     None),
    ("I/O Toggles",   "io_toggles",     "io_toggles",      None),
    ("Speculation",    "speculation",    "speculation",      None),
    ("KV Quality",    "kv_quality",     "kv_quality",       None),
    ("Workload Sim",  "workload_sim",   None,               None),
    ("Context Sweep", "context_sweep",  None,               None),
    ("NIAH",           "niah",           None,              None),
    ("Quality",        "quality",        "quality",         "quality"),
]


def _get_pipeline_progress():
    """Scan saved results and Optuna studies to determine pipeline progress.

    Returns list of dicts: [{name, results_key, status, completed_trials, total_trials}]
    Status is one of: "done", "partial", "pending"
    """
    is_pareto = _config.get("pareto", False)
    progress = []
    for display_name, results_key, study_key, preset_key in _PIPELINE_PHASES:
        info = {"name": display_name, "results_key": results_key, "status": "pending",
                "completed_trials": 0, "total_trials": 0, "study_key": study_key, "preset_key": preset_key}

        # Check if final results file exists
        has_results = load_phase_results(results_key) is not None

        # Check Optuna study for partial progress
        if study_key:
            versioned = f"{study_key}_{SCORE_VERSION}" + ("_pareto" if is_pareto else "")
            try:
                study = optuna.load_study(study_name=versioned, storage=ctx.optuna_db)
                info["completed_trials"] = len(study.trials)
            except Exception:
                pass

        if has_results:
            info["status"] = "done"
        elif info["completed_trials"] > 0:
            info["status"] = "partial"

        progress.append(info)
    return progress


def _find_resume_point(progress):
    """Find the first phase that isn't fully done. Returns index or None if all done."""
    for i, p in enumerate(progress):
        if p["status"] != "done":
            return i
    return None


def _delete_study(study_key):
    """Delete an Optuna study to allow a phase to restart from scratch."""
    is_pareto = _config.get("pareto", False)
    versioned = f"{study_key}_{SCORE_VERSION}" + ("_pareto" if is_pareto else "")
    try:
        optuna.delete_study(study_name=versioned, storage=ctx.optuna_db)
        print(f"    Cleared study: {versioned}")
    except Exception:
        pass  # study doesn't exist yet, that's fine


def resume_pipeline():
    """Show pipeline progress and let user choose how to resume."""
    progress = _get_pipeline_progress()
    resume_idx = _find_resume_point(progress)

    print("\n" + "=" * 60)
    print("  Pipeline Progress")
    print("=" * 60)

    for i, p in enumerate(progress):
        if p["status"] == "done":
            marker = "  [done]"
        elif p["status"] == "partial":
            marker = f"  [{p['completed_trials']} trials]"
        else:
            marker = ""

        arrow = "  >>>" if i == resume_idx else "     "
        status_icon = {"done": "+", "partial": "~", "pending": " "}[p["status"]]
        print(f"  {arrow} [{status_icon}] {p['name']}{marker}")

    if resume_idx is None:
        print("\n  All phases complete! Nothing to resume.")
        print("  Use [r] Reset in Advanced to start fresh, or [o] to re-run.")
        input("\n  Press Enter to continue...")
        return

    phase = progress[resume_idx]
    print(f"\n  Resume point: {phase['name']}")

    if phase["status"] == "partial" and phase["completed_trials"] > 0:
        print(f"  This phase has {phase['completed_trials']} completed trials.")
        print()
        print("  [1] Continue from last trial (keep existing progress)")
        print("  [2] Restart this phase (clear trials, start fresh)")
        print("  [b] Back")
        print()
        choice = input("  > ").strip().lower()
        if choice == "b" or choice == "":
            return
        elif choice == "2":
            if phase["study_key"]:
                _delete_study(phase["study_key"])
                # Also delete the results file if it exists
                results_path = ctx.results_dir / f"{phase['results_key']}_results.json"
                if results_path.exists():
                    results_path.unlink()
                    print(f"    Cleared results: {results_path.name}")
            print(f"    Phase {phase['name']} reset. Starting fresh.")
        elif choice != "1":
            print("  Invalid choice.")
            return
    else:
        print(f"  This phase hasn't started yet. Resuming pipeline from here.")
        input("\n  Press Enter to start...")

    # Run the pipeline starting from resume_idx
    p = _config.get("preset", "normal")
    print(f"\n  Using preset: {p}")
    run_full_pipeline(resume_from=resume_idx)


# ============================================================
# Interactive Terminal Menu
# ============================================================

def clear_screen():
    os.system("cls" if sys.platform == "win32" else "clear")


def print_header():
    print("=" * 60)
    print("  llama-server Parameter Optimizer")
    print("  GP-Bayesian Coordinate Descent")
    print("=" * 60)
    from .engine import is_server_running
    status = "ONLINE" if is_server_running() else "OFFLINE"
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"  Python: {py_ver}")
    print(f"  Server: {ctx.server_url} [{status}]")
    print(f"  Model:  {ctx.model_path.name}")
    arch_label = f"MoE ({ctx.default_experts} experts, {ctx.max_experts} max)" if ctx.is_moe else "Dense"
    print(f"  Arch:   {arch_label}")
    print(f"  GPU:    {ctx.default_gpu_layers}/{ctx.max_gpu_layers} layers offloaded")
    gpus = detect_gpus()
    if gpus:
        for g in gpus:
            print(f"  GPU {g['index']}: {g['name']} ({g['vram_free_gb']:.1f}/{g['vram_total_gb']}GB free)")
    numa_info = f", {ctx.numa_nodes} NUMA nodes" if ctx.numa_nodes > 1 else ""
    print(f"  CPU:    {ctx.max_threads} threads (auto-detected{numa_info})")
    model_class, model_size = classify_model(str(ctx.model_path))
    print(f"  Size:   {model_size:.1f} GB ({model_class})")
    ctx_val = _config.get("target_context")
    ctx_label = f"{ctx_val:,}" if ctx_val else "auto (4096 → sweep)"
    print(f"  Context: {ctx_label}")
    print(f"  Preset: {_config.get('preset', 'normal')}")
    if ctx.bench_path:
        print(f"  Bench:  {ctx.bench_path.name} (GPU/MoE phases accelerated)")
    if _config.get("draft_model"):
        print(f"  Draft:  {Path(_config['draft_model']).name}")
    try:
        import aiohttp
        _HAS_AIOHTTP = True
    except ImportError:
        _HAS_AIOHTTP = False
    if _HAS_AIOHTTP:
        print(f"  Async:  aiohttp (concurrent quality evals)")
    # Show active toggles
    active_toggles = []
    if _config.get("pareto"): active_toggles.append("pareto")
    if _config.get("debug"): active_toggles.append("debug")
    if _config.get("no_jinja"): active_toggles.append("no-jinja")
    if _config.get("no_bench"): active_toggles.append("no-bench")
    if _config.get("fail_fast"): active_toggles.append("fail-fast")
    if _config.get("skip_quality"): active_toggles.append("skip-quality")
    if _config.get("interactive"): active_toggles.append("interactive")
    if active_toggles:
        print(f"  Flags:  {', '.join(active_toggles)}")
    print(f"  Results: {ctx.results_dir}/")
    print("=" * 60)


def switch_model():
    """Scan models directory for GGUFs and let user pick one."""

    models_dir = ctx.model_path.parent.parent  # go up from model subdir to models/
    # Safety: if we ended up at a root or very shallow dir, stay in the model's own folder
    if len(models_dir.parts) <= 2:
        models_dir = ctx.model_path.parent
    gguf_files = sorted(models_dir.rglob("*.gguf"))
    # Filter out mmproj files (vision projectors, not language models)
    gguf_files = [f for f in gguf_files if "mmproj" not in f.name.lower()
                  and "reranker" not in f.parent.name.lower()
                  and "embedding" not in f.parent.name.lower()]

    if not gguf_files:
        print(f"\n  No GGUF files found in {models_dir}")
        input("\n  Press Enter to continue...")
        return

    print(f"\n  Available models in {models_dir}:\n")
    for i, f in enumerate(gguf_files):
        current = " ← current" if f == ctx.model_path else ""
        size_gb = f.stat().st_size / (1024**3)
        print(f"    [{i+1}] {f.parent.name}/{f.name} ({size_gb:.1f} GB){current}")

    print(f"\n    [0] Enter custom path")
    raw = input("\n  > ").strip()

    if raw == "0":
        path = input("  Path to GGUF: ").strip().strip('"').strip("'")
        if not Path(path).is_file():
            print(f"  File not found: {path}")
            input("\n  Press Enter to continue...")
            return
        new_model = Path(path)
    elif raw.isdigit() and 1 <= int(raw) <= len(gguf_files):
        new_model = gguf_files[int(raw) - 1]
    else:
        return

    # Detect architecture
    print(f"\n  Architecture for {new_model.name}?")
    print("    [1] MoE (Mixture of Experts)")
    print("    [2] Dense")
    arch_choice = input("  > ").strip()
    if arch_choice not in ("1", "2"):
        return

    ctx.model_path = new_model
    _config["model"] = str(ctx.model_path)

    # Update architecture
    if arch_choice == "1":
        key = input("  Expert override key (e.g., qwen35moe.expert_used_count): ").strip()
        default_exp = input("  Default active experts [8]: ").strip()
        max_exp = input("  Max experts [16]: ").strip()
        try:
            parsed_default = int(default_exp) if default_exp else 8
        except ValueError:
            parsed_default = 8
        try:
            parsed_max = int(max_exp) if max_exp else 16
        except ValueError:
            parsed_max = 16
        ctx.arch = {
            "type": "moe",
            "expert_override_key": key,
            "default_experts": parsed_default,
            "max_experts": parsed_max,
        }
        ctx.is_moe = True
        ctx.expert_override_key = key
        ctx.default_experts = ctx.arch["default_experts"]
        ctx.max_experts = ctx.arch["max_experts"]
    else:
        ctx.arch = {"type": "dense"}
        ctx.is_moe = False
        ctx.expert_override_key = ""
        ctx.default_experts = 8
        ctx.max_experts = 16

    _config["architecture"] = ctx.arch

    # Re-detect layers
    detected = detect_model_layers(str(ctx.model_path))
    ctx.max_gpu_layers = detected or 99
    ctx.default_gpu_layers = ctx.max_gpu_layers
    ctx.naked_engine["n_gpu_layers"] = ctx.default_gpu_layers
    if "tensor_split" in ctx.naked_engine:
        del ctx.naked_engine["tensor_split"]
    ctx.quality_baseline = None

    # Recalculate size class for new model's timeouts
    model_class, model_size = classify_model(str(ctx.model_path))
    ctx.model_size_class = model_class
    ctx.model_size_gb = model_size

    # Reset results dir for new model — always inside the package
    model_stem = ctx.model_path.stem.lower().replace(" ", "-")
    base_results_dir = Path(__file__).resolve().parent / "results"
    ctx.results_dir = base_results_dir / model_stem
    ctx.results_dir.mkdir(parents=True, exist_ok=True)
    ctx.lookup_cache_file = str(ctx.results_dir / "lookup-cache.bin")
    ctx.optuna_db = "sqlite:///" + str(ctx.results_dir / "optuna.db").replace("\\", "/")
    _config["results_dir"] = str(ctx.results_dir)

    # Persist config so the model switch survives restarts
    config_path = Path(_config.get("_config_path", "optimizer-config.json"))
    try:
        save_data = {k: v for k, v in _config.items() if not k.startswith("_")}
        temp_path = config_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)
        temp_path.replace(config_path)
        print(f"  Config saved to {config_path}")
    except Exception as e:
        print(f"  [!] Warning: couldn't save config: {e}")

    print(f"\n  Switched to: {ctx.model_path.name}")
    print(f"  Arch: {'MoE' if ctx.is_moe else 'Dense'} | Layers: {ctx.max_gpu_layers}")
    print(f"  Results: {ctx.results_dir}/")
    input("\n  Press Enter to continue...")


def print_menu():
    p = _config.get("preset", "normal")
    # Check for resumable progress
    progress = _get_pipeline_progress()
    resume_idx = _find_resume_point(progress)
    done_count = sum(1 for x in progress if x["status"] == "done")
    partial = [x for x in progress if x["status"] == "partial"]

    print()
    print(f"  [o]   Optimize (full pipeline)  [{p}]")
    if done_count > 0 or partial:
        resume_hint = ""
        if partial:
            ph = partial[0]
            resume_hint = f"  ({ph['name']}: {ph['completed_trials']} trials done)"
        elif resume_idx is not None:
            resume_hint = f"  (next: {progress[resume_idx]['name']})"
        print(f"  [re]  Resume pipeline{resume_hint}")
    print("  [cmd] Generate launch command")
    print("  [v]   View results")
    print()
    print("  [a]   Advanced...")
    print("  [m]   Switch model")
    print(f"  [p]   Preset ({p})")
    ctx_val = _config.get("target_context")
    ctx_label = f"{ctx_val:,}" if ctx_val else "auto"
    print(f"  [c]   Context ({ctx_label})")
    draft = _config.get("draft_model")
    draft_label = Path(draft).name if draft else "none"
    print(f"  [d]   Draft model ({draft_label})")
    print("  [t]   Toggles")
    print("  [q]   Quit")
    print()


def print_advanced_menu():
    print()
    print("  " + "=" * 50)
    print("  Advanced")
    print("  " + "=" * 50)
    print()
    print("  Pyramid Phases:")
    print("    [g]   GPU offload        [ce]  Core engine")
    print("    [io]  I/O toggles        [sp]  Speculation")
    print("    [kv]  KV quality         [wl]  Workload sim")
    print()
    print("  Legacy Phases (individual):")
    print("    [moe] MoE threads        [ex]  Expert count")
    print("    [c]   Compute            [me]  Memory")
    print("    [mo]  MoE audit          [ca]  Compute audit")
    print("    [ma]  Memory audit")
    print()
    print("  Multi-GPU:")
    print("    [ts]  Tensor split       [tp]  Topology")
    print()
    print("  Evals:")
    print("    [s]   Quality            [ni]  NIAH")
    print("    [re]  Reasoning          [ie]  Integrity")
    print("    [cs]  Context sweep      [qr]  Quant rec")
    print()
    print("  Other:")
    print("    [cd]  Coord descent (no quality)")
    print("    [ba]  Batch optimize")
    print("    [html] HTML report")
    print("    [d]   Dashboard (web UI)")
    print("    [r]   Reset DB")
    print()
    print("  [b] ← Back")
    print()


def _advanced_menu():
    """Submenu for individual phases, multi-GPU, evals, and other tools."""
    from .engine import is_server_running, start_naked_server
    from .state import get_preset_trials, _find_llama_bench
    from .pipeline import phase_topology_sweep
    from .models import recommend_quant

    while True:
        clear_screen()
        print_header()
        print_advanced_menu()

        choice = input("  > ").strip().lower()

        if choice in ("b", ""):
            return
        elif choice == "g":
            phase_gpu_offload()
            input("\n  Press Enter to continue...")
        elif choice == "ce":
            n = ask_trials("Core Engine", 80)
            phase_core_engine(n_trials=n)
            input("\n  Press Enter to continue...")
        elif choice == "io":
            n = ask_trials("I/O Toggles", 20)
            core_data = load_phase_results("core_engine")
            base = core_data["best_params"] if core_data else {}
            phase_io_toggles(n_trials=n, base_core_config={**ctx.naked_engine, **base})
            input("\n  Press Enter to continue...")
        elif choice == "sp":
            n = ask_trials("Speculation", 40)
            base = dict(ctx.naked_engine)
            for src in ["core_engine", "io_toggles"]:
                data = load_phase_results(src)
                if data and "best_params" in data:
                    base.update(data["best_params"])
            phase_speculation(n_trials=n, base_config=base)
            input("\n  Press Enter to continue...")
        elif choice == "kv":
            n = ask_trials("KV Quality", 15)
            base = dict(ctx.naked_engine)
            for src in ["core_engine", "io_toggles", "speculation"]:
                data = load_phase_results(src)
                if data and "best_params" in data:
                    base.update(data["best_params"])
            phase_kv_quality(n_trials=n, base_config=base)
            input("\n  Press Enter to continue...")
        elif choice == "wl":
            base = dict(ctx.naked_engine)
            for src in ["core_engine", "io_toggles", "speculation", "kv_quality"]:
                data = load_phase_results(src)
                if data and "best_params" in data:
                    base.update(data["best_params"])
            phase_workload_sim(base_config=base)
            input("\n  Press Enter to continue...")
        elif choice == "moe":
            phase_moe(include_experts=False)
            input("\n  Press Enter to continue...")
        elif choice == "ex":
            p1a = load_phase_results("moe_combined")
            moe = _get_moe_config(p1a)
            phase_experts(locked_moe_threads=moe["n_cpu_moe"])
            input("\n  Press Enter to continue...")
        elif choice == "c":
            n = ask_trials("Compute", 60)
            p1a = load_phase_results("moe_combined")
            moe = _get_moe_config(p1a)
            phase_compute(n_trials=n, phase_name="compute", locked_moe=moe)
            input("\n  Press Enter to continue...")
        elif choice == "me":
            n = ask_trials("Memory", 60)
            p1a = load_phase_results("moe_combined")
            p1b = load_phase_results("compute")
            moe = _get_moe_config(p1a)
            base = {**p1b["best_params"], **moe} if p1b else None
            phase_memory(n_trials=n, phase_name="memory", base_compute_config=base)
            input("\n  Press Enter to continue...")
        elif choice == "ca":
            n = ask_trials("Compute Audit", 60)
            p1a = load_phase_results("moe_combined")
            p1d = load_phase_results("moe_audit")
            p2 = load_phase_results("memory")
            p1b = load_phase_results("compute")
            moe = _get_moe_config(p1a)
            if p1d and "best_params" in p1d:
                moe["n_cpu_moe"] = p1d["best_params"]["n_cpu_moe"]
            base = p2["best_params"] if p2 else None
            seed = p1b["best_params"] if p1b else None
            if base is None:
                print("  [!] No Memory results found. Run Memory first.")
                input("  Press Enter to continue...")
            else:
                phase_compute(n_trials=n, phase_name="compute_audit", base_memory_config=base, seed_params=seed, locked_moe=moe)
                input("\n  Press Enter to continue...")
        elif choice == "mo":
            p1a = load_phase_results("moe_combined")
            p1b = load_phase_results("compute")
            p2 = load_phase_results("memory")
            moe = _get_moe_config(p1a)
            compute = p1b["best_params"] if p1b else None
            mem = p2["best_params"] if p2 else None
            if compute is None:
                print("  [!] No Compute results found. Run Compute first.")
                input("  Press Enter to continue...")
            else:
                phase_moe_revalidate(locked_compute=compute, locked_moe=moe, base_memory_config=mem)
                input("\n  Press Enter to continue...")
        elif choice == "ma":
            n = ask_trials("Memory Audit", 60)
            p1a = load_phase_results("moe_combined")
            p1c = load_phase_results("compute_audit")
            p1d = load_phase_results("moe_audit")
            p2 = load_phase_results("memory")
            moe = _get_moe_config(p1a)
            if p1d and "best_params" in p1d:
                moe["n_cpu_moe"] = p1d["best_params"]["n_cpu_moe"]
            base = {**p1c["best_params"], **moe} if p1c else None
            seed = p2["best_params"] if p2 else None
            if base is None:
                print("  [!] No Compute Audit results found. Run Compute Audit first.")
                input("  Press Enter to continue...")
            else:
                phase_memory(n_trials=n, phase_name="memory_audit", base_compute_config=base, seed_params=seed)
                input("\n  Press Enter to continue...")
        elif choice == "ts":
            gpus = detect_gpus()
            if len(gpus) < 2:
                print("  [!] Tensor split requires 2+ GPUs.")
            else:
                p1c = load_phase_results("compute_audit") or load_phase_results("compute")
                p2b = load_phase_results("memory_audit") or load_phase_results("memory")
                base = dict(ctx.naked_engine)
                if p1c and "best_params" in p1c: base.update(p1c["best_params"])
                if p2b and "best_params" in p2b: base.update(p2b["best_params"])
                phase_tensor_split(gpus, base_config=base)
            input("\n  Press Enter to continue...")
        elif choice == "tp":
            gpus = detect_gpus()
            ts_data = load_phase_results("tensor_split")
            if len(gpus) < 2:
                print("  [!] Topology sweep requires 2+ GPUs.")
            elif not ts_data or "best_split" not in ts_data:
                print("  [!] Run tensor split [ts] first to get a base split.")
            else:
                base_split = tuple(ts_data["best_split"])
                p1c = load_phase_results("compute_audit") or load_phase_results("compute")
                p2b = load_phase_results("memory_audit") or load_phase_results("memory")
                base = dict(ctx.naked_engine)
                if p1c and "best_params" in p1c: base.update(p1c["best_params"])
                if p2b and "best_params" in p2b: base.update(p2b["best_params"])
                phase_topology_sweep(gpus, base_split=base_split, base_config=base)
            input("\n  Press Enter to continue...")
        elif choice == "s":
            if ctx.skip_quality:
                print("  [*] --skip-quality is set — Quality/sampling phase skipped (embedding/reranker model).")
                input("\n  Press Enter to continue...")
            else:
                n = ask_trials("Quality", 80)
                phase3(n_trials=n)
                input("\n  Press Enter to continue...")
        elif choice == "ni":
            phase_niah()
            input("\n  Press Enter to continue...")
        elif choice == "re":
            if not is_server_running():
                proc = start_naked_server()
                if proc is None:
                    input("\n  Press Enter to continue...")
                    continue
            phase_reasoning_eval()
            input("\n  Press Enter to continue...")
        elif choice == "ie":
            if not is_server_running():
                proc = start_naked_server()
                if proc is None:
                    input("\n  Press Enter to continue...")
                    continue
            phase_integrity_eval()
            input("\n  Press Enter to continue...")
        elif choice == "cs":
            ctx_input = input("  Context sizes (comma-sep, or Enter for default): ").strip()
            contexts = [int(x.strip()) for x in ctx_input.split(",") if x.strip().isdigit()] if ctx_input else None
            p1c = load_phase_results("compute_audit") or load_phase_results("compute")
            p2b = load_phase_results("memory_audit") or load_phase_results("memory")
            p1a = load_phase_results("moe_combined")
            moe = _get_moe_config(p1a)
            base = dict(ctx.naked_engine)
            if p1c and "best_params" in p1c: base.update(p1c["best_params"])
            if p2b and "best_params" in p2b: base.update(p2b["best_params"])
            base.update(moe)
            phase_context_sweep(base_config=base, contexts=contexts)
            input("\n  Press Enter to continue...")
        elif choice == "qr":
            gpus = detect_gpus()
            rec = recommend_quant(str(ctx.model_path), gpus)
            print(f"\n  Recommended: {rec['recommended']}")
            print(f"  Reasoning:   {rec['reasoning']}")
            print(f"  Est. VRAM:   {rec['estimated_vram_gb']:.1f} GB")
            if gpus:
                print(f"  GPU(s):      {', '.join(g['name'] for g in gpus)}")
            input("\n  Press Enter to continue...")
        elif choice == "cd":
            p = _config.get("preset", "normal")
            print(f"  Using preset: {p} (change with [p])")
            run_full_pipeline(
                trials_moe=get_preset_trials(p, "moe"),
                trials_p1b=get_preset_trials(p, "compute"),
                trials_p2=get_preset_trials(p, "memory"),
                trials_p1c=get_preset_trials(p, "compute_audit"),
                trials_p2b=get_preset_trials(p, "memory_audit"),
                trials_p3=0,
            )
            input("\n  Press Enter to continue...")
        elif choice == "ba":
            batch_dir = input("  Path to models directory: ").strip().strip('"').strip("'")
            if batch_dir and Path(batch_dir).is_dir():
                skip = input("  Skip models with existing results? [y/N]: ").strip().lower() == "y"
                timeout_str = input("  Per-model timeout in minutes (0=none) [0]: ").strip()
                timeout_min = int(timeout_str) if timeout_str.isdigit() else 0
                batch_optimize(batch_dir, skip_existing=skip, timeout_minutes=timeout_min, interactive=True)
            else:
                print(f"  Not a valid directory: {batch_dir}")
            input("\n  Press Enter to continue...")
        elif choice == "html":
            generate_html_report()
            input("\n  Press Enter to continue...")
        elif choice == "d":
            launch_dashboard()
            input("\n  Press Enter to continue...")
        elif choice == "r":
            reset_db()
        else:
            print("  Invalid choice.")
            time.sleep(1)


def _context_menu():
    """Set target context size for optimization, or leave on auto."""
    current = _config.get("target_context")
    print()
    print("  " + "=" * 50)
    print("  Context Size")
    print("  " + "=" * 50)
    print()
    if current:
        print(f"  Current: {current:,} tokens (locked)")
        print("  All phases will optimize around this context size.")
    else:
        print("  Current: auto")
        print("  Phases optimize at 4096, then context sweep finds your max.")
    print()
    print("  Common sizes: 4096, 8192, 16384, 32768, 65536, 131072")
    print("  Enter a value, or 'auto' to reset to automatic.")
    print()

    val = input("  Context > ").strip().lower()
    if not val or val == "b":
        return
    if val == "auto":
        _config.pop("target_context", None)
        ctx.naked_engine["context"] = 4096
        print("  Context: auto (optimize at 4096, sweep for max after)")
    else:
        try:
            ctx_val = int(val)
            if ctx_val < 512:
                print("  [!] Minimum context is 512.")
                input("  Press Enter to continue...")
                return
            if ctx_val > 1048576:
                print("  [!] That's over 1M tokens. Are you sure? (y/n)")
                if input("  > ").strip().lower() != "y":
                    return
            _config["target_context"] = ctx_val
            ctx.naked_engine["context"] = ctx_val
            print(f"  Context: {ctx_val:,} tokens (all phases will use this)")
        except ValueError:
            print("  [!] Invalid number.")
            input("  Press Enter to continue...")
            return
    input("  Press Enter to continue...")


def _draft_model_menu():
    """Set a draft model for true speculative decoding (Phase 4)."""
    current = _config.get("draft_model")
    print()
    print("  " + "=" * 50)
    print("  Draft Model (Speculative Decoding)")
    print("  " + "=" * 50)
    print()
    if current:
        print(f"  Current: {Path(current).name}")
        print(f"  Path:    {current}")
    else:
        print("  Current: none (will use N-gram speculation)")
    print()
    print("  A draft model is a small model (e.g., 1.5B) that guesses")
    print("  tokens for the main model to verify. Much faster than N-gram")
    print("  for long-form generation.")
    print()
    print("  Enter path to a GGUF draft model, or 'none' to disable.")
    print()

    val = input("  Draft model > ").strip()
    if not val or val.lower() == "b":
        return
    if val.lower() == "none":
        _config.pop("draft_model", None)
        print("  Draft model: disabled (will use N-gram speculation)")
    else:
        path = Path(val)
        if not path.exists():
            print(f"  [!] File not found: {val}")
            input("  Press Enter to continue...")
            return
        if not path.suffix.lower() == ".gguf":
            print(f"  [!] Not a GGUF file: {val}")
            input("  Press Enter to continue...")
            return
        _config["draft_model"] = str(path)
        size_gb = path.stat().st_size / (1024**3)
        print(f"  Draft model: {path.name} ({size_gb:.1f} GB)")
    input("  Press Enter to continue...")


def toggle_menu():
    """Interactive submenu for toggling runtime flags."""
    from .state import _find_llama_bench

    while True:
        # Build toggle state display
        def _on_off(key, invert=False):
            val = _config.get(key, False)
            if invert:
                val = not val
            return "ON" if val else "OFF"

        print()
        print("  " + "=" * 50)
        print("  Toggles")
        print("  " + "=" * 50)
        print(f"  [1] Pareto mode         [{_on_off('pareto')}]  Multi-objective optimization (TPS × VRAM × Quality)")
        print(f"  [2] Debug output        [{_on_off('debug')}]  Show server cmd flags, extra diagnostics")
        print(f"  [3] No Jinja            [{_on_off('no_jinja')}]  Disable Jinja template parsing")
        print(f"  [4] No llama-bench      [{_on_off('no_bench')}]  Force HTTP-only mode for all phases")
        print(f"  [5] Fail fast           [{_on_off('fail_fast')}]  Exit immediately if baseline server fails")
        print(f"  [6] Skip quality        [{_on_off('skip_quality')}]  Skip Quality/sampling phase")
        print(f"  [7] Interactive         [{_on_off('interactive')}]  Pause between phases for inspection")
        print()
        print("  [b] Back")
        print()

        choice = input("  > ").strip().lower()

        toggle_map = {
            "1": ("pareto", "Pareto mode"),
            "2": ("debug", "Debug output"),
            "3": ("no_jinja", "No Jinja"),
            "4": ("no_bench", "No llama-bench"),
            "5": ("fail_fast", "Fail fast"),
            "6": ("skip_quality", "Skip quality"),
            "7": ("interactive", "Interactive"),
        }

        if choice == "b" or choice == "":
            return
        elif choice in toggle_map:
            key, label = toggle_map[choice]
            _config[key] = not _config.get(key, False)
            new_state = "ON" if _config[key] else "OFF"
            print(f"  {label}: {new_state}")

            # Apply runtime side effects
            if key == "no_jinja":
                ctx.no_jinja = _config["no_jinja"]
            elif key == "debug":
                ctx.debug = _config["debug"]
            elif key == "fail_fast":
                ctx.fail_fast = _config["fail_fast"]
            elif key == "skip_quality":
                ctx.skip_quality = _config["skip_quality"]
            elif key == "no_bench":
                if _config["no_bench"]:
                    ctx.bench_path = None
                else:
                    ctx.bench_path = _find_llama_bench(str(ctx.server_path))
        else:
            print("  Invalid choice.")


## _get_moe_config is imported from .pipeline


def ask_trials(phase_label, default):
    """Ask for trial count, return default if user just hits enter."""
    raw = input(f"  {phase_label} trials [{default}]: ").strip()
    if not raw:
        return default
    try:
        n = int(raw)
        return n if n > 0 else default
    except ValueError:
        print(f"  Invalid number, using {default}")
        return default


def reset_db():
    """Delete the Optuna DB and result files so all phases start fresh."""
    db_path = ctx.results_dir / "optuna.db"
    confirm = input("  Delete all saved trial progress and results? [y/N]: ").strip().lower()
    if confirm == "y":
        if db_path.exists():
            # Close any open Optuna storage connections first
            try:
                import gc
                gc.collect()  # Force garbage collection to release DB handles
                import optuna
                # Delete all studies to release connections
                storage = optuna.storages.RDBStorage(ctx.optuna_db)
                for s in optuna.study.get_all_study_summaries(storage=storage):
                    optuna.delete_study(study_name=s.study_name, storage=storage)
                del storage
                gc.collect()
            except Exception:
                pass
            try:
                db_path.unlink()
            except PermissionError:
                # Nuclear option: rename it so it's effectively gone
                stale = db_path.with_suffix(".db.old")
                try:
                    if stale.exists():
                        stale.unlink()
                except Exception:
                    pass
                try:
                    db_path.rename(stale)
                    print("  (DB was locked — renamed to optuna.db.old)")
                except Exception:
                    print("  [!] Could not delete DB — close the optimizer and delete it manually.")
        # Also clean up result JSONs
        for name in ["gpu", "core_engine", "io_toggles", "speculation", "kv_quality", "workload_sim", "context_sweep", "moe_combined", "moe", "experts", "compute", "memory", "compute_audit", "moe_audit", "memory_audit", "niah", "quality"]:
            p = ctx.results_dir / f"{name}_results.json"
            if p.exists():
                p.unlink()
        print("  DB and results deleted. All phases will start fresh.")
    else:
        print("  Cancelled.")
    input("  Press Enter to continue...")


def view_results():
    """Browse saved results organized by model."""
    base_results_dir = Path(__file__).resolve().parent / "results"
    if not base_results_dir.exists():
        print("\n  No results directory found.")
        input("  Press Enter to continue...")
        return

    # Find all model folders (dirs that contain at least one *_results.json)
    model_dirs = []
    for d in sorted(base_results_dir.iterdir()):
        if d.is_dir() and any(d.glob("*_results.json")):
            model_dirs.append(d)

    # Migrate legacy flat results into a model subfolder
    root_results = list(base_results_dir.glob("*_results.json"))
    if root_results:
        # Try to get model name from config
        cfg_path = base_results_dir / "optimizer-config.json"
        model_label = "unknown-model"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                model_path = cfg.get("model", "")
                if model_path:
                    model_label = Path(model_path).stem.lower().replace(" ", "-")
            except Exception:
                pass
        dest = base_results_dir / model_label
        dest.mkdir(parents=True, exist_ok=True)
        for f in root_results:
            target = dest / f.name
            if not target.exists():
                f.rename(target)
            else:
                f.unlink()  # duplicate, remove from root
        # Move optuna.db too if present
        root_db = base_results_dir / "optuna.db"
        if root_db.exists() and not (dest / "optuna.db").exists():
            root_db.rename(dest / "optuna.db")
        # Rescan after migration
        model_dirs = []
        for d in sorted(base_results_dir.iterdir()):
            if d.is_dir() and any(d.glob("*_results.json")):
                model_dirs.append(d)

    if not model_dirs:
        print("\n  No optimization results found.")
        input("  Press Enter to continue...")
        return

    # Model selection
    while True:
        print(f"\n{'=' * 60}")
        print(f"  Saved Results — Select Model")
        print(f"{'=' * 60}")
        for i, d in enumerate(model_dirs):
            label = "(root)" if d == base_results_dir else d.name
            # Count phases and find best TPS across all results
            phase_files = list(d.glob("*_results.json"))
            best_tps = 0
            for pf in phase_files:
                try:
                    data = json.loads(pf.read_text(encoding="utf-8"))
                    bt = data.get("best_tps", 0)
                    if bt > best_tps:
                        best_tps = bt
                except Exception:
                    pass
            # Get modification time of most recent result
            newest = max(phase_files, key=lambda f: f.stat().st_mtime)
            from datetime import datetime
            mtime = datetime.fromtimestamp(newest.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            tps_str = f" | Best: {best_tps:.1f} t/s" if best_tps > 0 else ""
            print(f"  [{i + 1}] {label}")
            print(f"      {len(phase_files)} phases | {mtime}{tps_str}")
        print(f"\n  [b] Back")
        choice = input("\n  > ").strip().lower()
        if choice == "b":
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(model_dirs):
                _view_model_results(model_dirs[idx])
            else:
                print("  Invalid choice.")
        except ValueError:
            print("  Invalid choice.")


def _view_model_results(model_dir):
    """Show all phase results for a specific model."""
    phase_order = [
        "gpu", "tensor_split", "topology_sweep",
        "moe_combined", "moe", "experts", "moe_audit",
        "core_engine", "io_toggles", "speculation",
        "kv_quality", "niah", "workload_sim", "context_sweep",
        "compute", "memory", "compute_audit", "memory_audit",
        "quality",
    ]
    label = "(root)" if model_dir.name == "results" else model_dir.name

    while True:
        # Gather available phases in order
        available = []
        for phase in phase_order:
            path = model_dir / f"{phase}_results.json"
            if path.exists():
                available.append((phase, path))
        # Also catch any phases not in the predefined order
        for path in sorted(model_dir.glob("*_results.json")):
            phase = path.stem.replace("_results", "")
            if phase not in [a[0] for a in available]:
                available.append((phase, path))

        if not available:
            print(f"\n  No phase results found in {label}.")
            input("  Press Enter to continue...")
            return

        print(f"\n{'=' * 60}")
        print(f"  {label} — Phase Results")
        print(f"{'=' * 60}")

        for i, (phase, path) in enumerate(available):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                tps = data.get("best_tps", 0)
                score = data.get("best_score", data.get("baseline_score", 0))
                dur = data.get("duration_minutes", 0)
                beat = data.get("beat_baseline", False)
                status = "+" if beat else "="
                tps_str = f"{tps:.1f} t/s" if tps > 0 else ""
                dur_str = f"{dur:.1f}m" if dur else ""
                detail_parts = [s for s in [tps_str, dur_str] if s]
                detail = f" ({', '.join(detail_parts)})" if detail_parts else ""
                print(f"  [{i + 1}] {status} {phase}{detail}")
            except Exception:
                print(f"  [{i + 1}]   {phase} (error reading)")

        print(f"\n  [b] Back")
        choice = input("\n  > ").strip().lower()
        if choice == "b":
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(available):
                _view_phase_detail(available[idx][0], available[idx][1])
            else:
                print("  Invalid choice.")
        except ValueError:
            print("  Invalid choice.")


def _view_phase_detail(phase_name, path):
    """Show detailed results for a single phase."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"\n  Error reading {path}: {e}")
        input("  Press Enter to continue...")
        return

    print(f"\n{'=' * 60}")
    print(f"  {phase_name} — Details")
    print(f"{'=' * 60}")

    # Baseline
    bl = data.get("baseline", {})
    if bl:
        print(f"  Baseline:  {bl.get('tps', 0):.1f} t/s | pp: {bl.get('prompt_tps', 0):.0f} t/s | TTFT: {bl.get('ttft', 0):.0f}ms")

    bs = data.get("baseline_score")
    if bs:
        print(f"  Baseline Score: {bs:.1f}")

    # Best
    bm = data.get("best_metrics", {})
    if bm:
        print(f"  Best:      {bm.get('tps', 0):.1f} t/s | pp: {bm.get('prompt_tps', 0):.0f} t/s | TTFT: {bm.get('ttft', 0):.0f}ms")
    elif data.get("best_tps"):
        print(f"  Best TPS:  {data['best_tps']:.1f} t/s")

    if data.get("best_score"):
        print(f"  Best Score: {data['best_score']:.1f}")

    beat = data.get("beat_baseline")
    if beat is not None:
        print(f"  Beat baseline: {'Yes' if beat else 'No'}")

    # Verified
    vf = data.get("verified", {})
    if vf:
        print(f"  Verified:  {vf.get('tps', 0):.1f} t/s | pp: {vf.get('prompt_tps', 0):.0f} t/s | TTFT: {vf.get('ttft', 0):.0f}ms")

    # Duration + trials
    dur = data.get("duration_minutes")
    if dur:
        print(f"  Duration:  {dur:.1f} min")
    trials = data.get("all_trials", [])
    if trials:
        print(f"  Trials:    {len(trials)}")

    # Best params
    bp = data.get("best_params", {})
    if bp:
        print(f"\n  Best Params:")
        for k, v in bp.items():
            print(f"    {k}: {v}")

    # Param importance
    pi = data.get("param_importance", {})
    if pi:
        print(f"\n  Parameter Importance:")
        for k, v in sorted(pi.items(), key=lambda x: x[1], reverse=True):
            bar = "#" * int(v * 20)
            print(f"    {k:28s} {v * 100:5.1f}%  {bar}")

    # Context sweep special display
    contexts = data.get("contexts", {})
    if contexts:
        print(f"\n  Context Results:")
        for ctx_str, info in sorted(contexts.items(), key=lambda x: int(x[0])):
            tps = info.get("tps", 0)
            score = info.get("score", 0)
            fits = "yes" if info.get("fits", True) else "OOM"
            print(f"    {int(ctx_str):>8,}: {tps:.1f} t/s | score: {score:.1f} | {fits}")

    # NIAH special display
    kv_results = data.get("kv_results", [])
    if kv_results:
        print(f"\n  KV Cache Results:")
        for r in kv_results:
            ppl_str = f" | PPL: {r['ppl']:.2f}" if r.get("ppl") else ""
            print(f"    {r['kv_type']:>6}: {r['pass_rate']:.0f}% recall{ppl_str}")

    print()
    input("  Press Enter to continue...")


def _find_file(pattern_name, extensions, search_hints=None):
    """Try to find a file by searching common locations. Returns path or None."""
    search_hints = search_hints or []
    for hint in search_hints:
        hint = Path(hint).expanduser()
        if hint.is_file():
            return str(hint)
        if hint.is_dir():
            for ext in extensions:
                for f in hint.rglob(f"*{ext}"):
                    return str(f)
    return None


def first_run_setup():
    """Interactive setup wizard for first-time users. Returns config dict."""
    print("=" * 60)
    print("  llama-server Parameter Optimizer — First Run Setup")
    print("=" * 60)
    print()
    print("  This wizard will help you configure the optimizer.")
    print("  Your settings will be saved so you only do this once.")
    print()

    config = {}

    # 1. llama-server path
    print("  [1/6] Path to llama-server executable")
    print("        (e.g., /usr/local/bin/llama-server or C:\\...\\llama-server.exe)")
    while True:
        path = input("        > ").strip().strip('"').strip("'")
        if Path(path).is_file():
            config["server"] = path
            break
        print(f"        File not found: {path}")
        print("        Please enter the full path to the llama-server executable.")

    # 2. Model path
    print()
    print("  [2/6] Path to GGUF model file")
    print("        (e.g., /models/my-model.gguf)")
    while True:
        path = input("        > ").strip().strip('"').strip("'")
        if Path(path).is_file():
            config["model"] = path
            break
        print(f"        File not found: {path}")

    # 3. Chat template
    print()
    print("  [3/6] Path to chat template (.jinja)")
    print("        (press Enter to skip — server will use its default)")
    path = input("        > ").strip().strip('"').strip("'")
    if path and Path(path).is_file():
        config["chat_template"] = path
    else:
        config["chat_template"] = ""

    # 4. Architecture
    print()
    print("  [4/6] Model architecture")
    print("        [1] MoE (Mixture of Experts) — e.g., Qwen 3.5 MoE, Mixtral, DeepSeek")
    print("        [2] Dense — e.g., Llama, Qwen dense, Gemma, Phi")
    while True:
        choice = input("        > ").strip()
        if choice in ("1", "2"):
            break
        print("        Enter 1 or 2.")

    if choice == "1":
        config["architecture"] = {"type": "moe"}
        print()
        print("        GGUF override key for expert count")
        print("        (e.g., qwen35moe.expert_used_count, deepseek2.expert_used_count)")
        print("        Check your model's GGUF metadata if unsure.")
        key = input("        > ").strip()
        config["architecture"]["expert_override_key"] = key

        print()
        print("        Default active experts (how many the model was trained with)")
        default_exp = input("        [8] > ").strip()
        try:
            config["architecture"]["default_experts"] = int(default_exp) if default_exp else 8
        except ValueError:
            config["architecture"]["default_experts"] = 8

        print()
        print("        Max experts to sweep")
        max_exp = input("        [16] > ").strip()
        try:
            config["architecture"]["max_experts"] = int(max_exp) if max_exp else 16
        except ValueError:
            config["architecture"]["max_experts"] = 16
    else:
        config["architecture"] = {"type": "dense"}

    # 5. GPU offload starting point
    print()
    print("  [5/6] GPU layer offload starting point (-ngl)")
    print("        GPU Offload will automatically sweep all offload levels to find the fastest.")
    print("        This sets the starting default (99 = try full GPU first).")
    print()
    print("        - 99 = start with full GPU (recommended)")
    print("        - 0  = CPU only (no GPU)")
    ngl = input("        [99] > ").strip()
    default_ngl = int(ngl) if ngl else 99
    config["hardware"] = {"default_gpu_layers": default_ngl}

    # 6. Port
    print()
    print("  [6/6] Server port")
    port = input("        [8090] > ").strip()
    config["port"] = int(port) if port else 8090

    # Results dir
    config["results_dir"] = str(Path(__file__).resolve().parent / "results")

    # Save config
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    config_path = results_dir / "optimizer-config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    config["_config_path"] = str(config_path)

    print()
    print(f"  Config saved to: {config_path}")
    print("  You can edit this file to change settings later.")
    print()
    input("  Press Enter to start the optimizer...")

    return config


def _needs_setup():
    """Check if the current config points to valid files."""
    return not (Path(_config["server"]).is_file() and Path(_config["model"]).is_file())


# ============================================================
# Command Generation & Reporting
# ============================================================

def generate_command(results_dir=None, server_path=None, model_path=None,
                     chat_template_path=None, port=None):
    """Generate a ready-to-use llama-server command from optimization results."""
    from .search import ensure_results_dir

    results_dir = Path(results_dir or ctx.results_dir)
    server_path = server_path or str(ctx.server_path)
    model_path = model_path or str(ctx.model_path)
    chat_template_path = chat_template_path or str(ctx.chat_template_path)
    port = port or ctx.port

    # Load all phase results. New pyramid phases take priority over old legacy phases.
    merged_params = {}
    for phase_name in ["gpu", "tensor_split", "topology_sweep",
                       # Legacy phases (loaded first, overwritten by pyramid if both exist)
                       "moe_combined", "compute", "memory", "compute_audit", "memory_audit",
                       # New pyramid phases (loaded last, take priority)
                       "core_engine", "io_toggles", "speculation", "kv_quality",
                       "quality"]:
        path = results_dir / f"{phase_name}_results.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, ValueError):
                print(f"  [!] {phase_name} results corrupted — skipping")
                continue
            # Skip stale results from a different scoring formula
            if "score_version" in data and data["score_version"] != SCORE_VERSION:
                print(f"  [!] Skipping stale {phase_name} results (score v{data['score_version']} != v{SCORE_VERSION})")
                continue
            if phase_name == "gpu" and "best_ngl" in data:
                merged_params["n_gpu_layers"] = data["best_ngl"]
            elif phase_name in ("tensor_split", "topology_sweep") and "best_split_str" in data:
                merged_params["tensor_split"] = data["best_split_str"]
            elif "best_params" in data:
                merged_params.update(data["best_params"])

    if not merged_params:
        print("[!] No optimization results found. Run the optimizer first.")
        return None

    # Build command parts — quote paths with spaces for shell safety
    def _q(path_str):
        return f'"{path_str}"' if " " in str(path_str) else str(path_str)

    parts = [_q(server_path), "-m", _q(model_path), "--port", str(port)]
    if chat_template_path and Path(chat_template_path).is_file():
        parts.extend(["--chat-template-file", _q(chat_template_path)])

    p = merged_params
    flag_map = [
        ("n_gpu_layers", "-ngl"), ("context", "-c"), ("threads", "-t"),
        ("threads_batch", "-tb"), ("n_cpu_moe", "--n-cpu-moe"),
        ("batch_size", "-b"), ("ubatch_size", "--ubatch-size"),
        ("poll", "--poll"), ("poll_batch", "--poll-batch"),
        ("prio", "--prio"), ("prio_batch", "--prio-batch"),
        ("cpu_strict", "--cpu-strict"), ("cpu_strict_batch", "--cpu-strict-batch"),
        ("spec_type", "--spec-type"), ("spec_ngram_n", "--spec-ngram-size-n"),
        ("spec_ngram_m", "--spec-ngram-size-m"), ("spec_ngram_min_hits", "--spec-ngram-min-hits"),
        ("draft_max", "--draft"), ("draft_min", "--draft-min"),
        ("draft_p_min", "--draft-p-min"), ("model_draft", "--model-draft"),
        ("cache_reuse", "--cache-reuse"),
        # Quality/Sampling — so the final command boots with optimized defaults
        ("temperature", "--temp"), ("top_p", "--top-p"), ("top_k", "--top-k"),
        ("min_p", "--min-p"), ("typical_p", "--typical-p"),
        ("repeat_penalty", "--repeat-penalty"), ("presence_penalty", "--presence-penalty"),
        ("frequency_penalty", "--frequency-penalty"),
        ("mirostat", "--mirostat"), ("mirostat_lr", "--mirostat-lr"), ("mirostat_ent", "--mirostat-ent"),
        ("repeat_last_n", "--repeat-last-n"), ("top_n_sigma", "--top-n-sigma"),
        ("dynatemp_range", "--dynatemp-range"), ("dynatemp_exp", "--dynatemp-exponent"),
        ("xtc_probability", "--xtc-probability"), ("xtc_threshold", "--xtc-threshold"),
        ("dry_multiplier", "--dry-multiplier"), ("dry_base", "--dry-base"),
        ("dry_allowed_length", "--dry-allowed-length"), ("dry_penalty_last_n", "--dry-penalty-last-n"),
    ]
    for key, flag in flag_map:
        if key in p:
            parts.extend([flag, str(p[key])])

    if "tensor_split" in p:
        parts.extend(["--tensor-split", p["tensor_split"]])
    if p.get("flash_attn") in ("on", True, "1", 1):
        parts.append("--flash-attn")
    if "kv_cache_type" in p:
        parts.extend(["--cache-type-k", p["kv_cache_type"], "--cache-type-v", p["kv_cache_type"]])
    if "expert_used_count" in p and ctx.expert_override_key and p["expert_used_count"] != ctx.default_experts:
        parts.extend(["--override-kv", f"{ctx.expert_override_key}=int:{p['expert_used_count']}"])

    # Boolean flags
    if p.get("swa_full"): parts.append("--swa-full")
    if p.get("repack") is False: parts.append("--no-repack")
    if p.get("op_offload") is False: parts.append("--no-op-offload")
    if p.get("mlock"): parts.append("--mlock")
    if p.get("no_mmap"): parts.append("--no-mmap")

    if p.get("lookup_cache_dynamic") or p.get("use_lookup_cache"):
        cache_path = p.get("lookup_cache_dynamic") or str(results_dir / "lookup-cache.bin")
        parts.extend(["--lookup-cache-dynamic", cache_path])
    if "numa" in p:
        parts.extend(["--numa", str(p["numa"])])
    if ctx.no_jinja:
        parts.append("--no-jinja")
        parts.extend(["--chat-template", "chatml"])

    # Format with line continuations
    lines = [parts[0]]
    i = 1
    while i < len(parts):
        token = parts[i]
        if token.startswith("-") and i + 1 < len(parts) and not parts[i + 1].startswith("-"):
            lines.append(f"  {token} {parts[i + 1]}")
            i += 2
        else:
            lines.append(f"  {token}")
            i += 1
    command_str = " \\\n".join(lines)

    print("\n" + "=" * 60)
    print("  OPTIMIZED COMMAND")
    print("=" * 60)

    if p.get("context_shift") is False or "--no-context-shift" in command_str:
        print()
        print("  [!] WARNING: This config disables context shift (--no-context-shift).")
        print("      This improves speed but BREAKS long multi-turn chats.")
        print("      Remove --no-context-shift if using this for a conversational UI.")

    print()
    print(command_str)
    print()

    # Save
    ensure_results_dir()
    cmd_path = results_dir / "command.txt"
    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write(command_str + "\n")
    print(f"  Saved to {cmd_path}")

    cmd_json_path = results_dir / "command.json"
    with open(cmd_json_path, "w", encoding="utf-8") as f:
        json.dump({"server": server_path, "model": model_path, "port": port, "params": merged_params}, f, indent=2)
    print(f"  Saved to {cmd_json_path}")
    return command_str


def install_interrupt_handler():
    """Install Ctrl+C handler that prints summary and generates command on interrupt."""
    def _handler(signum, frame):
        print("\n\n" + "=" * 60)
        print("  INTERRUPT — cleaning up...")
        print("=" * 60)
        try:
            kill_server()
        except Exception:
            pass
        # Print completed phases
        print("\n  Completed phases:")
        any_results = False
        for name in ["gpu", "moe_combined", "compute", "memory", "moe_audit", "compute_audit", "memory_audit", "quality"]:
            path = ctx.results_dir / f"{name}_results.json"
            if path.exists():
                try:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    if "best_ngl" in data:
                        print(f"    {name:16s}: ngl={data['best_ngl']}")
                    elif "best_tps" in data:
                        print(f"    {name:16s}: {data['best_tps']:.1f} t/s")
                    elif "best_score" in data:
                        print(f"    {name:16s}: {data['best_score']:.0f}%")
                    any_results = True
                except Exception:
                    pass
        if any_results:
            try:
                generate_command()
            except Exception:
                pass
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, _handler)


# ============================================================
# HTML Report Generation
# ============================================================

def generate_html_report(results_dir=None, model_name=None, gpus=None):
    """Generate a comprehensive HTML report from optimization results."""
    from .constants import SCORE_PP_BASELINE, SCORE_TTFT_BASELINE

    results_dir = Path(results_dir or ctx.results_dir)
    model_name = model_name or ctx.model_path.name
    if gpus is None:
        gpus = detect_gpus()

    # Load all phase results
    phases = {}
    for name in ["gpu", "tensor_split", "topology_sweep", "moe_combined", "moe", "experts",
                  "compute", "memory", "moe_audit", "compute_audit", "memory_audit",
                  "quality", "context_sweep"]:
        path = results_dir / f"{name}_results.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    phases[name] = json.load(f)
            except (json.JSONDecodeError, ValueError):
                print(f"  [!] {name} results corrupted — skipping")

    if not phases:
        print("[!] No results found for HTML report.")
        return None

    gpu_info = ", ".join(f"{g['name']} ({g['vram_total_gb']}GB)" for g in gpus) if gpus else "Unknown"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build phase rows
    phase_rows = ""
    for name, data in phases.items():
        if name in ("context_sweep",):
            continue
        score = data.get("best_tps", data.get("best_score", data.get("best_ngl", "-")))
        dur = data.get("duration_minutes", data.get("duration_seconds", 0))
        if isinstance(dur, (int, float)) and dur > 0 and dur < 1:
            dur_str = f"{dur * 60:.0f}s"
        elif isinstance(dur, (int, float)):
            dur_str = f"{dur:.1f}m" if dur < 60 else f"{dur / 60:.1f}h"
        else:
            dur_str = "-"
        trials = len(data.get("all_trials", []))
        beat = data.get("beat_baseline", True)
        color = "#4ade80" if beat else "#f87171"
        phase_rows += f"""
        <tr>
            <td>{name}</td>
            <td style="color:{color}">{score}</td>
            <td>{trials}</td>
            <td>{dur_str}</td>
        </tr>"""

    # Build importance sections
    importance_html = ""
    for name in ["compute", "memory", "compute_audit", "memory_audit", "quality"]:
        data = phases.get(name, {})
        imp = data.get("param_importance", {})
        if imp:
            bars = ""
            max_val = max(imp.values()) if imp else 1
            for param, pct in sorted(imp.items(), key=lambda x: -x[1]):
                width = pct / max_val * 100 if max_val > 0 else 0
                bars += f"""
                <div class="imp-row">
                    <span class="imp-name">{param}</span>
                    <div class="imp-bar" style="width:{width:.0f}%"></div>
                    <span class="imp-pct">{pct:.1f}%</span>
                </div>"""
            importance_html += f"<h3>{name}</h3>{bars}"

    # Get command
    cmd_path = results_dir / "command.txt"
    command_text = ""
    if cmd_path.exists():
        command_text = cmd_path.read_text()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Optimize Report — {model_name}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: #1a1a2e; color: #e0e0e0; font-family: 'JetBrains Mono', 'Cascadia Code', 'Consolas', monospace; font-size: 14px; padding: 2rem; }}
  h1 {{ color: #00d4ff; margin-bottom: 0.5rem; }}
  h2 {{ color: #00d4ff; margin: 2rem 0 1rem; border-bottom: 1px solid #333; padding-bottom: 0.5rem; }}
  h3 {{ color: #a0a0a0; margin: 1.5rem 0 0.5rem; }}
  .header {{ margin-bottom: 2rem; }}
  .meta {{ color: #888; margin-bottom: 0.25rem; }}
  table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; }}
  th, td {{ padding: 0.5rem 1rem; text-align: left; border-bottom: 1px solid #2a2a4a; }}
  th {{ color: #00d4ff; }}
  .imp-row {{ display: flex; align-items: center; margin: 0.25rem 0; }}
  .imp-name {{ width: 200px; color: #a0a0a0; }}
  .imp-bar {{ height: 16px; background: linear-gradient(90deg, #00d4ff, #0066ff); border-radius: 2px; margin: 0 0.5rem; min-width: 2px; }}
  .imp-pct {{ color: #888; width: 60px; }}
  pre {{ background: #0d0d1a; padding: 1rem; border-radius: 4px; overflow-x: auto; border: 1px solid #333; }}
  .score-formula {{ background: #0d0d1a; padding: 1rem; border-radius: 4px; border: 1px solid #333; margin: 1rem 0; }}
</style>
</head>
<body>
<div class="header">
  <h1>Optimization Report</h1>
  <div class="meta">Model: {model_name}</div>
  <div class="meta">GPU: {gpu_info}</div>
  <div class="meta">Generated: {ts}</div>
  <div class="meta">Score Version: {SCORE_VERSION}</div>
</div>

<h2>Phase Summary</h2>
<table>
  <tr><th>Phase</th><th>Best Score</th><th>Trials</th><th>Duration</th></tr>
  {phase_rows}
</table>

<h2>Parameter Importance</h2>
{importance_html if importance_html else "<p>No importance data available.</p>"}

<h2>Score Formula</h2>
<div class="score-formula">
  <p><b>Full mode</b> (promoted configs with large-prompt data):<br>
  score = gen×0.35 + large_tps×0.25 + pp_norm×gen×0.15 + ttft_norm×gen×0.15 + vram_eff×gen×0.10</p>
  <p><b>Lightweight mode</b> (quick filter):<br>
  score = gen × (0.60 + 0.25×pp/{SCORE_PP_BASELINE} + 0.15×{SCORE_TTFT_BASELINE}/ttft)</p>
  <p style="color:#888;margin-top:0.5rem">Full mode rewards configs that maintain TPS under heavy context load.</p>
</div>

<h2>Optimized Command</h2>
<pre>{command_text if command_text else "Run generate_command() to produce this."}</pre>

<footer style="margin-top:3rem;color:#555;text-align:center">
  Generated by llama-server Parameter Optimizer
</footer>
</body>
</html>"""

    report_path = results_dir / "report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML report saved to {report_path}")
    return str(report_path)


def launch_dashboard():
    """Launch optuna-dashboard as a background subprocess.

    The dashboard provides live parallel coordinate plots, contour maps,
    and hyperparameter importance graphs — all reading from the existing
    Optuna SQLite DB. Zero code changes needed; it's a free visualization layer.
    """
    global _dashboard_proc

    try:
        import optuna_dashboard as _od
    except ImportError:
        print("[*] Installing optuna-dashboard...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna-dashboard", "-q", "--no-warn-script-location"])
        import optuna_dashboard as _od

    db_path = ctx.results_dir / "optuna.db"
    db_url = "sqlite:///" + str(db_path).replace("\\", "/")
    dashboard_port = ctx.port + 100  # e.g., 8190 if server is 8090

    # Ensure DB has Optuna tables — dashboard connects with skip_table_creation=True
    # so it will crash if the DB is empty. Creating a dummy study initializes the schema.
    try:
        _init_study = optuna.create_study(
            storage=db_url, study_name="_dashboard_init", load_if_exists=True,
            direction="maximize",
        )
        optuna.delete_study(study_name="_dashboard_init", storage=db_url)
    except Exception:
        pass  # tables already exist or study already deleted

    print(f"\n[*] Launching optuna-dashboard on http://127.0.0.1:{dashboard_port}")
    print(f"    DB: {db_path}")
    print(f"    Open in your browser: http://127.0.0.1:{dashboard_port}")
    print(f"    This runs in the background — close it with Ctrl+C or kill the process.\n")

    # Write a launcher script — always use forward slashes to avoid \U SyntaxError on Windows
    safe_url = db_url.replace("\\", "/")
    launcher = ctx.results_dir / "_dashboard_launcher.py"
    launcher.write_text(
        f"import optuna_dashboard\n"
        f"optuna_dashboard.run_server('{safe_url}', host='127.0.0.1', port={dashboard_port})\n",
        encoding="utf-8",
    )

    proc = subprocess.Popen(
        [sys.executable, str(launcher)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    # Quick check — did it crash immediately?
    time.sleep(3)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode("utf-8", errors="replace")
        print(f"  [!] Dashboard failed to start: {stderr[:500]}")
        return None

    # Auto-open in browser
    import webbrowser
    webbrowser.open(f"http://127.0.0.1:{dashboard_port}")

    _dashboard_proc = proc
    return proc


def main():
    global _dashboard_proc

    initialize()  # Parse CLI args & populate ctx/_config (no-op if already done)

    from .search import ensure_results_dir
    from .engine import is_server_running
    from .hardware import kill_competing_processes

    # First-run setup if config is missing or paths are invalid
    if _needs_setup():
        new_config = first_run_setup()
        # Merge new config into defaults and rebuild context
        _config.clear()
        _config.update(copy.deepcopy(_DEFAULTS))
        for k, v in new_config.items():
            if isinstance(v, dict) and isinstance(_config.get(k), dict):
                _config[k].update(v)
            else:
                _config[k] = v
        # Auto-detect hardware (normally done in _load_config, skipped on first-run path)
        hw = _config["hardware"]
        if hw.get("max_threads") is None:
            hw["max_threads"] = os.cpu_count() or 16
        if "numa_nodes" not in hw:
            hw["numa_nodes"] = _detect_numa_nodes()
        if hw.get("moe_sweep_max") is None:
            hw["moe_sweep_max"] = min(hw["max_threads"] * 2, 40)
        if hw.get("moe_sweep_center") is None:
            hw["moe_sweep_center"] = hw["moe_sweep_max"] // 2
        if hw.get("max_gpu_layers") is None:
            hw["max_gpu_layers"] = detect_model_layers(_config.get("model", ""))
        if hw.get("max_gpu_layers") is None:
            hw["max_gpu_layers"] = 99
        # Rebuild application context from updated config
        new_ctx = create_context(_config)
        # Copy all fields to the module-level ctx
        for f in ctx.__dataclass_fields__:
            setattr(ctx, f, getattr(new_ctx, f))

    # Check if the target port is already in use (e.g. a production server)
    import socket as _socket
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as _sock:
        if _sock.connect_ex(('127.0.0.1', ctx.port)) == 0:
            print(f"[!] Port {ctx.port} is already in use.")
            print(f"    A server may already be running. Use --port to pick a different port,")
            print(f"    or stop the existing process first.")
            sys.exit(1)

    # Validate critical file paths before proceeding
    if not ctx.server_path.is_file():
        print(f"[ERROR] llama-server executable not found: {ctx.server_path}")
        print("        Use --server to specify the correct path, or run first-time setup.")
        sys.exit(1)
    if not ctx.model_path.is_file():
        print(f"[ERROR] Model file not found: {ctx.model_path}")
        print("        Use --model to specify the correct path, or run first-time setup.")
        sys.exit(1)

    ensure_results_dir()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Enable dry-run mode if requested
    if _config.get("dry_run"):
        ctx.dry_run = True
        print("[*] DRY RUN mode — no phases will execute")

    # Start log tee
    log_tee = LogTee(str(ctx.results_dir))
    sys.stdout = log_tee
    print(f"[*] Logging to {log_tee.log_path}")

    # Install interrupt handler
    install_interrupt_handler()

    # Launch optuna-dashboard if requested
    _dashboard_proc = None
    if _config.get("dashboard"):
        _dashboard_proc = launch_dashboard()

    # Kill competing GPU processes to free VRAM (opt-in only)
    if _config.get("kill_competing"):
        kill_competing_processes()
    else:
        gpus_pre = detect_gpus()
        if gpus_pre and any(g["vram_free_gb"] < g["vram_total_gb"] * 0.5 for g in gpus_pre):
            print("  [*] Tip: Use --kill-competing to free GPU memory from other processes")

    # Detect GPUs
    gpus = detect_gpus()
    if gpus:
        for g in gpus:
            print(f"  GPU {g['index']}: {g['name']} ({g['vram_total_gb']} GB, {g['vram_free_gb']} GB free)")
    else:
        print("  [!] No GPUs detected (pynvml not installed or no NVIDIA GPU)")

    # Classify model and set size-based timeouts
    model_class, model_size = classify_model(str(ctx.model_path))
    ctx.model_size_class = model_class
    ctx.model_size_gb = model_size
    print(f"  Model: {ctx.model_path.name} ({model_size:.1f} GB, class: {model_class})")

    # Hydrate GPU state from past runs so direct-menu launches don't OOM
    gpu_data = load_phase_results("gpu")
    if gpu_data and "best_ngl" in gpu_data:
        ctx.default_gpu_layers = gpu_data["best_ngl"]
        ctx.naked_engine["n_gpu_layers"] = ctx.default_gpu_layers

    # Batch mode — if --batch was provided, run batch and exit
    if _config.get("batch_dir"):
        try:
            batch_optimize(
                models_dir=_config["batch_dir"],
                preset=_config.get("preset", "normal"),
                skip_existing=_config.get("skip_existing", False),
                timeout_minutes=_config.get("timeout_minutes", 0),
                interactive=_config.get("interactive", False),
            )
        finally:
            log_tee.close()
        return

    while True:
        clear_screen()
        print_header()
        print_menu()

        choice = input("  > ").strip().lower()

        if choice == "q":
            break
        elif choice == "re":
            resume_pipeline()
            input("\n  Press Enter to continue...")
        elif choice in ("o", "all", "12345"):
            p = _config.get("preset", "normal")
            print(f"  Using preset: {p} (change with [p])")
            ctx.fresh_run = True
            run_full_pipeline()
            ctx.fresh_run = False
            input("\n  Press Enter to continue...")
        elif choice == "cmd":
            generate_command()
            input("\n  Press Enter to continue...")
        elif choice == "v":
            view_results()
        elif choice == "m":
            switch_model()
        elif choice == "p":
            presets = ["quick", "normal", "thorough"]
            current = _config.get("preset", "normal")
            idx = presets.index(current) if current in presets else 1
            new_idx = (idx + 1) % len(presets)
            _config["preset"] = presets[new_idx]
            print(f"  Preset changed: {current} → {presets[new_idx]}")
            input("  Press Enter to continue...")
        elif choice == "c":
            _context_menu()
        elif choice == "d":
            _draft_model_menu()
        elif choice == "t":
            toggle_menu()
        elif choice == "a":
            _advanced_menu()
        else:
            print("  Invalid choice.")
            time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Optimization aborted by user.")
    finally:
        print("\n[*] Cleaning up...")
        kill_server()
        if _dashboard_proc and _dashboard_proc.poll() is None:
            _dashboard_proc.terminate()
        print("    Done.")
