"""
Optimization phases and pipeline orchestrator.

Phase functions are implemented in the phases/ subpackage.
Import them directly from phases/ (e.g. ``from .phases import phase_gpu_offload``).
This module provides the orchestration functions:
batch_optimize() and run_full_pipeline().

Error strategy (see errors.py for full documentation):
    - _run_phase(): catches broad (OSError, ValueError, ...) so a single
      phase failure does not abort the entire pipeline.  Errors are logged
      at ERROR level and kill_server() is called to clean up.
    - batch_optimize(): catches per-model errors so the batch continues.
      BaselineFailure and KeyboardInterrupt get special handling.
    - _generate_reports(): catches and logs report generation errors at
      WARNING level -- a report failure should not hide a successful run.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from .constants import DEFAULT_MAX_GPU_LAYERS
from .engine import (
    BaselineFailure,
    PhaseTimer,
    check_dry_run,
    kill_server,
)
from .engine.parsing import reset_load_time_debug
from .hardware import detect_gpus
from .models import (
    classify_model,
    detect_gguf_architecture,
    detect_model_layers,
    detect_skippable_flags,
)
from .phases import (
    phase_core_engine,
    phase_gpu_offload,
    phase_kv_context_sweep,
    phase_moe_sweep,
    phase_quality,
    phase_speculation,
    phase_tensor_split,
    phase_workload_sim,
)
from .pipeline_config import PhaseConfig, PipelineConfig
from .result_types import EngineConfig, PhaseReturnDict
from .search import ensure_results_dir, load_phase_results
from .state import (
    AppContext,
    create_context,
    ctx,
    get_config,
    update_naked_engine,
)

logger = logging.getLogger(__name__)

# Split phase errors into two categories for differentiated logging:
# - Operational errors: environment/runtime failures (network, file I/O, concurrency)
# - Bug errors: likely code defects that should be investigated
_OPERATIONAL_ERRORS = (OSError, RuntimeError)
_BUG_ERRORS = (TypeError, KeyError, ValueError)
_PHASE_ERRORS = (*_OPERATIONAL_ERRORS, *_BUG_ERRORS)
_REPORT_ERRORS = (
    ImportError,
    OSError,
    json.JSONDecodeError,
    KeyError,
    ValueError,
    TypeError,
)

_MIN_TPS_RATIO = 0.80

__all__ = [
    "batch_optimize",
    "run_full_pipeline",
]


# ============================================================
# Batch Pipeline — optimize multiple models
# ============================================================


def _detect_architecture(gguf_path: Path) -> dict[str, Any]:
    """Detect and log the GGUF model architecture."""
    arch_info = detect_gguf_architecture(str(gguf_path))
    logger.info("Architecture: %s", arch_info["type"])
    return arch_info


def _build_model_config(
    gguf_path: Path, arch_info: dict[str, Any], per_model_results: Path
) -> dict[str, Any]:
    """Build a per-model config dict from the global config and GGUF metadata."""
    model_config = copy.deepcopy(ctx.config)
    # Fall back to global ctx for paths set during initialize().
    model_config.setdefault("server", str(ctx.server_path))
    model_config.setdefault("port", ctx.port)
    model_config.setdefault(
        "chat_template",
        str(ctx.chat_template_path) if ctx.chat_template_path else "",
    )
    model_config["model"] = str(gguf_path)
    model_config["architecture"] = arch_info

    detected = detect_model_layers(str(gguf_path))
    max_ngl = detected or DEFAULT_MAX_GPU_LAYERS
    model_config["hardware"] = dict(model_config.get("hardware", {}))
    model_config["hardware"]["max_gpu_layers"] = max_ngl
    model_config["hardware"]["default_gpu_layers"] = max_ngl
    hw = model_config["hardware"]
    hw.setdefault("max_threads", ctx.max_threads)
    hw.setdefault("moe_sweep_max", ctx.moe_sweep_max)
    hw.setdefault("moe_sweep_center", ctx.moe_sweep_center)

    per_model_results.mkdir(parents=True, exist_ok=True)
    model_config["results_dir"] = str(per_model_results)
    return model_config


def _load_gpu_results(model_ctx: AppContext) -> None:
    """Restore GPU offload results from a previous run if available."""
    gpu_data = load_phase_results(model_ctx, "gpu")
    if gpu_data and "best_ngl" in gpu_data:
        model_ctx.default_gpu_layers = gpu_data["best_ngl"]
        update_naked_engine(model_ctx, n_gpu_layers=model_ctx.default_gpu_layers)


def _apply_skip_flags(model_ctx: AppContext) -> None:
    """Detect and apply GGUF-aware skippable flags on the context."""
    model_ctx.skip_flags = detect_skippable_flags(
        str(model_ctx.model_path), model_ctx.default_gpu_layers
    )
    if model_ctx.skip_flags:
        logger.debug(
            "GGUF-aware: skipping irrelevant flags: %s",
            ", ".join(sorted(model_ctx.skip_flags)),
        )


def _build_model_context(gguf_path: Path, per_model_results: Path) -> AppContext:
    """Create a fresh AppContext for one model in a batch run.

    Detects architecture, layer count, and skippable flags, then returns a
    fully initialized context ready for ``run_full_pipeline``.
    """
    arch_info = _detect_architecture(gguf_path)
    model_config = _build_model_config(gguf_path, arch_info, per_model_results)

    model_ctx = create_context(model_config)
    model_ctx.config = model_config

    model_class, model_size = classify_model(str(gguf_path))
    model_ctx.model_size_class = model_class
    model_ctx.model_size_gb = model_size

    ensure_results_dir(model_ctx)
    _load_gpu_results(model_ctx)
    _apply_skip_flags(model_ctx)

    model_ctx.bench_path = ctx.bench_path
    return model_ctx


def _optimize_single_model(
    gguf_path: Path,
    per_model_results: Path,
    timeout_minutes: int,
) -> dict[str, Any]:
    """Run the pipeline on one model and return a result summary dict.

    Returns a dict with at least ``model`` and ``status`` keys.
    Raises ``KeyboardInterrupt`` so the caller can break the batch.
    """
    model_name = gguf_path.name
    model_start = time.time()
    reset_load_time_debug()

    model_ctx: AppContext | None = None
    try:
        model_ctx = _build_model_context(gguf_path, per_model_results)

        deadline = (
            (time.time() + (timeout_minutes * 60))
            if timeout_minutes and timeout_minutes > 0
            else None
        )
        run_full_pipeline(deadline=deadline, ctx=model_ctx)
        return {
            "model": model_name,
            "status": "completed",
            "duration": time.time() - model_start,
        }
    except KeyboardInterrupt:
        logger.warning("Batch interrupted at %s", model_name)
        kill_server(model_ctx or ctx)
        raise
    except BaselineFailure as e:
        logger.warning("%s Skipping model.", e)
        return {"model": model_name, "status": "failed (baseline)"}
    except _BUG_ERRORS as e:
        logger.warning(
            "Likely code bug optimizing %s: %s: %s",
            model_name,
            type(e).__name__,
            e,
        )
        logger.debug("Bug error details:", exc_info=True)
        return {"model": model_name, "status": "error", "error": str(e)}
    except _OPERATIONAL_ERRORS as e:
        logger.info("Operational error optimizing %s: %s", model_name, e)
        logger.debug("Operational error details:", exc_info=True)
        return {"model": model_name, "status": "error", "error": str(e)}
    finally:
        kill_server(model_ctx or ctx)
        if model_ctx is not None:
            model_ctx.close()


def batch_optimize(  # noqa: C901, PLR0915
    models_dir: str,
    preset: str = "normal",
    skip_existing: bool = False,
    timeout_minutes: int = 0,
    interactive: bool = False,
) -> None:
    """Run full optimization pipeline on every GGUF in a directory.

    Raises:
        KeyboardInterrupt: Propagated if the user aborts during batch run
            (after cleanup of the current model).
    """

    models_path = Path(models_dir).resolve()
    if not models_path.is_dir():
        logger.error("Not a directory: %s", models_dir)
        return

    gguf_files = sorted(models_path.rglob("*.gguf"))
    gguf_files = [
        f
        for f in gguf_files
        if f.is_file()
        and not f.is_symlink()
        and "mmproj" not in f.name.lower()
        and "embedding" not in f.parent.name.lower()
        and "reranker" not in f.parent.name.lower()
    ]

    if not gguf_files:
        logger.error("No GGUF model files found in %s", models_dir)
        return

    total = len(gguf_files)
    logger.info("=" * 60)
    logger.info("BATCH OPTIMIZATION")
    logger.info("Models: %d | Skip existing: %s", total, skip_existing)
    if timeout_minutes:
        logger.info("Per-model timeout: %d min", timeout_minutes)
    logger.info("=" * 60)

    batch_timer = PhaseTimer()
    batch_timer.start_phase("batch_total")
    results_summary: list[dict] = []

    for idx, gguf_path in enumerate(gguf_files):
        model_name = gguf_path.name
        logger.info("=" * 60)
        logger.info("[Model %d/%d] %s", idx + 1, total, model_name)
        logger.info("=" * 60)

        model_stem = gguf_path.stem.lower().replace(" ", "-")
        per_model_results = gguf_path.parent / f"optimize-results-{model_stem}"

        if (
            skip_existing
            and per_model_results.is_dir()
            and list(per_model_results.glob("*_results.json"))
        ):
            logger.info("Skipping — results already exist")
            results_summary.append({"model": model_name, "status": "skipped"})
            continue

        try:
            result = _optimize_single_model(
                gguf_path, per_model_results, timeout_minutes
            )
            results_summary.append(result)
        except KeyboardInterrupt:
            results_summary.append({"model": model_name, "status": "interrupted"})
            break

        dur = result.get("duration", 0.0)
        batch_timer.record_trial(dur)
        remaining = total - (idx + 1)
        if remaining > 0:
            logger.info(
                "ETA for remaining %d model(s): %s",
                remaining,
                batch_timer.eta(remaining),
            )
        if interactive and idx < total - 1:
            input("\n  Press Enter to continue to next model...")

    batch_timer.end_phase("batch_total")
    logger.info("=" * 60)
    logger.info("BATCH OPTIMIZATION COMPLETE")
    logger.info("=" * 60)
    for entry in results_summary:
        dur_str = ""
        if "duration" in entry:
            d = entry["duration"]
            dur_str = f" ({d / 60:.1f}m)" if d >= 60 else f" ({d:.0f}s)"  # noqa: PLR2004
        logger.info("  %-40s %s%s", entry["model"], entry["status"], dur_str)
    logger.info(batch_timer.summary())


# ============================================================
# Pipeline helpers
# ============================================================


def _validated_config_merge(
    base_config: EngineConfig,
    phase_params: dict[str, Any] | None,
    phase_name: str,
) -> EngineConfig:
    """Merge phase results into best_config with validation.

    Checks that phase_params is a non-None dict before merging.
    Warns if the merge introduces None values for any keys.

    Args:
        base_config: The accumulating best configuration dict.
        phase_params: Parameters from a completed phase (dict or None).
        phase_name: Name of the phase for diagnostic logging.

    Returns:
        New merged config dict (base_config is not mutated).
    """
    if phase_params is None:
        logger.warning("Phase %s returned None params -- skipping merge", phase_name)
        return {**base_config}
    if not isinstance(phase_params, dict):
        logger.warning(
            "Phase %s returned non-dict params (%s) -- skipping merge",
            phase_name,
            type(phase_params).__name__,
        )
        return {**base_config}

    # Filter out None values from phase params to avoid corrupting config
    clean_params = {k: v for k, v in phase_params.items() if v is not None}
    if len(clean_params) < len(phase_params):
        dropped = [k for k, v in phase_params.items() if v is None]
        logger.debug("Phase %s: dropped None-valued keys: %s", phase_name, dropped)

    return {**base_config, **clean_params}


def _extract_best_params(
    phase_result: PhaseReturnDict | None,
) -> dict[str, Any] | None:
    """Extract best_params dict from a phase return value.

    All phase functions now return PhaseReturnDict (with best_params key).

    Returns:
        dict of best parameters, or None if phase_result is None/invalid.
    """
    if phase_result is None:
        return None
    if isinstance(phase_result, dict) and "best_params" in phase_result:
        return phase_result["best_params"]
    return {}


def _select_context_size(
    ctx: AppContext, sweep_results: dict[str, Any], best_config: EngineConfig
) -> EngineConfig:
    """Select the best context size from sweep results.

    Picks the largest context that achieves at least 80% of peak throughput.
    Updates ctx.naked_engine in place (baseline state) and returns a new
    best_config with the context key set.

    Args:
        ctx: Application context (naked_engine is updated).
        sweep_results: Dict of context_size_str -> {tps, fits, ...}.
        best_config: Accumulating config dict (not mutated).

    Returns:
        Updated best_config dict with context set, or the original if no viable context.
    """
    peak_tps = max(
        (r["tps"] for r in sweep_results.values() if r.get("fits")), default=0
    )
    if peak_tps <= 0:
        return best_config
    viable = [
        (int(k), r)
        for k, r in sweep_results.items()
        if k.isdigit() and r.get("fits") and r["tps"] >= peak_tps * _MIN_TPS_RATIO
    ]
    if not viable:
        return best_config
    best_ctx = max(viable, key=lambda x: x[0])[0]
    update_naked_engine(ctx, context=best_ctx)
    logger.info(
        "Auto-selected context: %s (>=80%% of peak %.1f t/s)",
        "{:,}".format(best_ctx),
        peak_tps,
    )
    return {**best_config, "context": best_ctx}


def _print_pipeline_summary(ctx: AppContext, pipeline_timer: PhaseTimer) -> None:
    """Print the final pipeline summary with results from each phase.

    Args:
        ctx: Application context for loading phase results.
        pipeline_timer: PhaseTimer with timing data.
    """
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("")
    for name in [
        "gpu",
        "moe_sweep",
        "kv_context_sweep",
        "core_engine",
        "speculation",
        "workload_sim",
        "quality",
    ]:
        data = load_phase_results(ctx, name)
        if data:
            _log_phase_result_line(name, data)

    logger.info(pipeline_timer.summary())


def _log_phase_result_line(name: str, data: dict[str, Any]) -> None:
    """Log a single phase result summary line."""
    if "best_ngl" in data:
        logger.info("  %-16s: n_gpu_layers=%s", name, data["best_ngl"])
    elif "contexts" in data:
        ctxs = data["contexts"]
        viable = [int(k) for k, v in ctxs.items() if v.get("fits")]
        max_ctx = max(viable) if viable else 0
        logger.info("  %-16s: max context=%s", name, "{:,}".format(max_ctx))
    elif "kv_results" in data:
        safe_kvs = [
            r["kv_type"]
            for r in data["kv_results"]
            if r.get("pass_rate", 0) >= data.get("reference_pass_rate", 100) - 5
        ]
        logger.info("  %-16s: safe KV types: %s", name, ", ".join(safe_kvs))
    elif "hot_ttft_avg_ms" in data:
        logger.info("  %-16s: hot TTFT=%.0fms", name, data["hot_ttft_avg_ms"])
    elif "best_tps" in data:
        logger.info("  %-16s: %.1f t/s", name, data["best_tps"])
    elif "best_score" in data:
        logger.info("  %-16s: %.0f%% quality", name, data["best_score"])


def _generate_reports(ctx: AppContext) -> None:
    """Generate the optimized command and HTML report."""

    def _safe_report(label, fn):
        try:
            fn()
        except _REPORT_ERRORS as e:
            logger.warning("Could not generate %s: %s", label, e)
            logger.debug("%s generation error details:", label, exc_info=True)

    from .cli.report import generate_html_report
    from .cli.services_command import generate_optimized_command

    _safe_report(
        "command",
        lambda: generate_optimized_command(ctx),
    )
    _safe_report(
        "HTML report",
        lambda: generate_html_report(
            results_dir=str(ctx.results_dir),
            model_name=ctx.model_path.name,
        ),
    )


# ============================================================
# Phase Dispatcher — maps phase names to execution functions
# ============================================================


_PhaseHandlerReturn = PhaseReturnDict | None

# Type alias for phase dispatch handlers
_PhaseHandler = Callable[
    [AppContext, PhaseConfig, EngineConfig, PipelineConfig],
    _PhaseHandlerReturn,
]


def _handle_gpu_offload(
    ctx: AppContext,
    pc: PhaseConfig,
    base: EngineConfig,
    pipeline_config: PipelineConfig,
) -> _PhaseHandlerReturn:
    return phase_gpu_offload(ctx)


def _handle_tensor_split(
    ctx: AppContext,
    pc: PhaseConfig,
    base: EngineConfig,
    pipeline_config: PipelineConfig,
) -> _PhaseHandlerReturn:
    gpus = detect_gpus()
    if len(gpus) >= 2:  # noqa: PLR2004
        return phase_tensor_split(ctx, gpus)
    return None


def _handle_moe_sweep(
    ctx: AppContext,
    pc: PhaseConfig,
    base: EngineConfig,
    pipeline_config: PipelineConfig,
) -> _PhaseHandlerReturn:
    if not ctx.is_moe:
        return None
    return phase_moe_sweep(ctx, phase_config=pc)


def _handle_kv_context_sweep(
    ctx: AppContext,
    pc: PhaseConfig,
    base: EngineConfig,
    pipeline_config: PipelineConfig,
) -> _PhaseHandlerReturn:
    return phase_kv_context_sweep(
        ctx,
        base_config=base,
        phase_config=pc,
        scoring_weights=pipeline_config.scoring_weights,
    )


def _handle_ab_toggles(
    ctx: AppContext,
    pc: PhaseConfig,
    base: EngineConfig,
    pipeline_config: PipelineConfig,
) -> _PhaseHandlerReturn:
    test_flags = pipeline_config.strip_globals_from_flags(
        pc.test_flags
        or [
            "op_offload",
            "prio",
            "prio_batch",
            "no_mmap",
            "mlock",
            "repack",
            "swa_full",
            "numa",
            "cpu_strict",
            "cpu_strict_batch",
        ]
    )
    stripped_pc = dataclasses.replace(pc, test_flags=test_flags)
    return phase_core_engine(
        ctx,
        n_trials=0,
        base_config=base,
        phase_config=stripped_pc,
    )


def _handle_core_engine(
    ctx: AppContext,
    pc: PhaseConfig,
    base: EngineConfig,
    pipeline_config: PipelineConfig,
) -> _PhaseHandlerReturn:
    trials = pc.trials or 100
    return phase_core_engine(
        ctx,
        n_trials=trials,
        base_config=base,
        phase_config=pc,
    )


def _handle_speculation(
    ctx: AppContext,
    pc: PhaseConfig,
    base: EngineConfig,
    pipeline_config: PipelineConfig,
) -> _PhaseHandlerReturn:
    trials = pc.trials or 40
    return phase_speculation(
        ctx,
        n_trials=trials,
        base_config=base,
        phase_config=pc,
    )


def _handle_workload_sim(
    ctx: AppContext,
    pc: PhaseConfig,
    base: EngineConfig,
    pipeline_config: PipelineConfig,
) -> _PhaseHandlerReturn:
    return phase_workload_sim(ctx, base_config=base)


def _handle_quality(
    ctx: AppContext,
    pc: PhaseConfig,
    base: EngineConfig,
    pipeline_config: PipelineConfig,
) -> _PhaseHandlerReturn:
    trials = pc.trials or 60
    if get_config("skip_quality") or trials <= 0:
        logger.info("Skipping Quality/sampling phase.")
        return None
    return phase_quality(ctx, n_trials=trials, phase_config=pc)


_PHASE_DISPATCH: dict[str, _PhaseHandler] = {
    "gpu_offload": _handle_gpu_offload,
    "tensor_split": _handle_tensor_split,
    "moe_sweep": _handle_moe_sweep,
    "kv_context_sweep": _handle_kv_context_sweep,
    "ab_toggles": _handle_ab_toggles,
    "core_engine": _handle_core_engine,
    "speculation": _handle_speculation,
    "workload_sim": _handle_workload_sim,
    "quality": _handle_quality,
}


def _dispatch_phase(
    phase_cfg: PhaseConfig,
    ctx: AppContext,
    best_config: EngineConfig,
    pipeline_config: PipelineConfig,
) -> _PhaseHandlerReturn:
    """Execute a single phase based on its config.

    Routes to the correct phase function via _PHASE_DISPATCH and passes
    config-specific params.
    Returns the phase result (PhaseReturnDict | None).
    """
    name = phase_cfg.phase

    # Build config with global flags applied
    base = {**best_config, **pipeline_config.global_flags}

    handler = _PHASE_DISPATCH.get(name)
    if handler is None:
        logger.warning("Unknown phase: %s — skipping", name)
        return None
    return handler(ctx, phase_cfg, base, pipeline_config)


# ============================================================
# Full Pipeline — helpers
# ============================================================


def _print_pipeline_banner(
    enabled_phases: list[PhaseConfig],
    pipeline_cfg: PipelineConfig,
) -> int:
    """Print the pipeline banner and return the estimated total trials.

    Args:
        enabled_phases: List of enabled PhaseConfig objects.
        pipeline_cfg: The loaded PipelineConfig.

    Returns:
        Estimated total trial count across all enabled phases.
    """
    p = get_config("preset", "normal")
    total_est = sum(pc.trials or 0 for pc in enabled_phases)
    logger.info("=" * 60)
    logger.info("PYRAMID OPTIMIZATION PIPELINE")
    for i, pc in enumerate(enabled_phases, 1):
        desc = pc.description or pc.display_name
        if pc.trials:
            desc = f"{desc} ({pc.trials} trials)"
        logger.info(
            "Phase %d: %-18s %s",
            i,
            pc.display_name,
            f"({desc})" if pc.trials else "",
        )
    if pipeline_cfg.global_flags:
        logger.info(
            "Global:  %s",
            ", ".join(f"{k}={v}" for k, v in pipeline_cfg.global_flags.items()),
        )
    logger.info("Total:   ~%d trials  [%s]", total_est, p)
    logger.info("=" * 60)
    return total_est


def _should_skip_phase(
    name: str,
    skip_to: int,
    resumed: int,
) -> bool:
    """Return True if the phase at index *resumed* should be skipped.

    Args:
        name: Display name of the phase (for logging).
        skip_to: The resume_from index (phases before this are skipped).
        resumed: Current phase index in the enabled list.

    Returns:
        True if the phase should be skipped.
    """
    if resumed < skip_to:
        logger.info("[skip] %s — already complete (resuming)", name)
        return True
    return False


def _merge_phase_result(
    best_config: EngineConfig,
    result_data: dict[str, Any] | None,
    phase_name: str,
    ctx: AppContext,
) -> EngineConfig:
    """Extract best_params from a phase result and merge into best_config.

    Falls back to loading cached results from disk when the phase returned
    None (the phase may have saved results internally).

    Args:
        best_config: Accumulating best configuration dict (not mutated).
        result_data: Raw return value from a phase function.
        phase_name: Internal phase name (used for disk lookup and logging).
        ctx: Application context for loading cached phase results.

    Returns:
        New merged config dict.
    """
    phase_best = _extract_best_params(result_data)
    if phase_best is None:
        cached = load_phase_results(ctx, phase_name)
        if cached and "best_params" in cached:
            phase_best = cached["best_params"]
    if phase_best:
        return _validated_config_merge(best_config, phase_best, phase_name)
    return {**best_config}


# ============================================================
# Full Pipeline
# ============================================================


def run_full_pipeline(  # noqa: C901, PLR0912, PLR0915
    deadline: float | None = None,
    resume_from: int = 0,
    *,
    ctx: AppContext | None = None,
) -> None:
    """Run the configurable Pyramid Pipeline.

    Pipeline phases, order, and search parameters are all controlled by
    the PipelineConfig loaded from results/<model>/pipeline-config.json.
    If no config exists, defaults are used.

    Each phase receives the accumulated best_config from all prior phases,
    plus global_flags from the pipeline config.
    """
    # Use module-level ctx as fallback for backward compatibility.
    if ctx is None:
        import sys as _sys

        ctx = _sys.modules[__name__].ctx

    # Load pipeline config
    config_path = ctx.results_dir / "pipeline-config.json"
    pipeline_cfg = PipelineConfig.load(config_path, is_moe=ctx.is_moe)

    p = get_config("preset", "normal")
    scale = pipeline_cfg.presets.get(p, 1.0)
    for pc in pipeline_cfg.phases:
        if pc.trials is not None:
            pc.trials = max(10, int(pc.trials * scale))

    enabled = pipeline_cfg.enabled_phases()
    total_est = _print_pipeline_banner(enabled, pipeline_cfg)

    if check_dry_run(
        ctx, "Full Pipeline", {"phases": len(enabled), "total_trials": total_est}, "all"
    ):
        return

    logger.info("")
    logger.info("Tip: Press Ctrl+C to skip the current phase")

    interactive = get_config("interactive", False)
    pipeline_timer = PhaseTimer()
    pipeline_timer.start_phase("pipeline_total")

    phase_names = [pc.display_name for pc in enabled]

    def _run_phase(name: str, fn: Callable[[], Any]) -> Any:
        """Run a phase, catching Ctrl+C to skip. Tracks timing."""
        if deadline and time.time() > deadline:
            logger.warning("Model timeout reached. Skipping %s.", name)
            return None
        logger.info("")
        pipeline_timer.start_phase(name)
        phase_start = time.time()
        try:
            return fn()
        except KeyboardInterrupt:
            logger.warning("%s skipped (Ctrl+C)", name)
            kill_server(ctx)
            return None
        except _BUG_ERRORS as e:
            logger.warning(
                "%s failed (likely code bug — %s): %s",
                name,
                type(e).__name__,
                e,
            )
            logger.debug("Bug error details:", exc_info=True)
            kill_server(ctx)
            return None
        except _OPERATIONAL_ERRORS as e:
            logger.info("%s failed (operational): %s", name, e)
            logger.debug("Operational error details:", exc_info=True)
            kill_server(ctx)
            return None
        finally:
            pipeline_timer.end_phase(name)
            dur = time.time() - phase_start
            pipeline_timer.record_trial(dur)
            p_idx = phase_names.index(name) if name in phase_names else -1
            remaining_names = phase_names[p_idx + 1 :] if p_idx >= 0 else []
            if remaining_names:
                eta_str = pipeline_timer.eta(len(remaining_names))
                logger.debug(
                    "[%s done in %.1fm] ETA for %d remaining: %s",
                    name,
                    dur / 60,
                    len(remaining_names),
                    eta_str,
                )
            if interactive:
                input("\n  --interactive: Press Enter to continue...")

    # Detect GGUF-skippable flags early (used by A/B toggles)
    if not ctx.skip_flags:
        ctx.skip_flags = detect_skippable_flags(
            str(ctx.model_path), ctx.default_gpu_layers
        )
    if ctx.numa_nodes <= 1 and "numa" not in ctx.skip_flags:
        ctx.skip_flags.add("numa")
    if ctx.skip_flags:
        logger.debug(
            "GGUF-aware: skipping irrelevant flags: %s",
            ", ".join(sorted(ctx.skip_flags)),
        )

    # Build initial best_config from naked_engine + global flags
    best_config = pipeline_cfg.build_base_config(dict(ctx.naked_engine))

    # --- Config-driven phase loop ---
    for phase_idx, phase_cfg in enumerate(enabled):
        name = phase_cfg.display_name

        if _should_skip_phase(name, resume_from, phase_idx):
            cached = load_phase_results(ctx, phase_cfg.phase)
            if cached and "best_params" in cached:
                best_config = _validated_config_merge(
                    best_config, cached["best_params"], phase_cfg.phase
                )
            continue

        if deadline and time.time() > deadline:
            logger.warning("Model timeout reached. Skipping %s.", name)
            continue

        result = _run_phase(
            name,
            lambda pc=phase_cfg, bc=best_config: _dispatch_phase(
                pc, ctx, bc, pipeline_cfg
            ),
        )

        best_config = _merge_phase_result(best_config, result, phase_cfg.phase, ctx)

    pipeline_timer.end_phase("pipeline_total")

    _print_pipeline_summary(ctx, pipeline_timer)
    _generate_reports(ctx)

    cmd_path = ctx.results_dir / "command.txt"
    report_path = ctx.results_dir / "report.html"
    if cmd_path.exists():
        logger.info("  Best command saved to: %s", cmd_path)
    if report_path.exists():
        logger.info("  HTML report: %s", report_path)
    logger.info("")
