"""Config, preset, toggle, and model management business logic.

Pure business-logic — ZERO print/input calls.
Split from services.py for cohesion; re-exported via services.py for
backward compatibility.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from ..constants import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_EXPERTS,
    DEFAULT_MAX_GPU_LAYERS,
    DEFAULT_TRIAL_COUNT,
    HAS_AIOHTTP,
    MAX_EXPERTS,
)
from ..hardware import detect_gpus
from ..models import classify_model, detect_model_layers
from ..state import (
    AppContext,
    find_llama_bench,
    get_config,
    rebuild_ctx,
    set_config,
    update_naked_engine,
)

logger = logging.getLogger(__name__)

_MIN_CONTEXT = 512
_MAX_CONTEXT = 1_048_576

# Single source of truth: maps config_key -> display label for all toggles.
_TOGGLE_REGISTRY: dict[str, str] = {
    "pareto": "pareto",
    "debug": "debug",
    "no_jinja": "no-jinja",
    "no_bench": "no-bench",
    "fail_fast": "fail-fast",
    "skip_quality": "skip-quality",
    "interactive": "interactive",
}

# ============================================================
# Type Aliases
# ============================================================

#: Application-wide config dict -- values are heterogeneous by design.
ConfigDict = dict[str, str | float | int | bool | None]


class MoeArchConfig(TypedDict):
    """Architecture config for Mixture-of-Experts models."""

    type: str
    expert_override_key: str
    default_experts: int
    max_experts: int


class DenseArchConfig(TypedDict):
    """Architecture config for dense (non-MoE) models."""

    type: str


class DraftModelInfo(TypedDict, total=False):
    """Info returned after setting a draft model."""

    name: str
    size_gb: float


# ============================================================
# Data Classes
# ============================================================


@dataclass(frozen=True)
class SystemInfo:
    """Snapshot of current system / configuration state for display."""

    python_version: str
    server_url: str
    server_online: bool
    model_name: str
    arch_type: str  # "MoE" or "Dense"
    arch_detail: str  # e.g. "8 experts, 16 max"
    gpu_layers: str  # e.g. "32/32"
    gpus: list[dict] = field(default_factory=list)
    cpu_threads: int = 0
    numa_nodes: int = 1
    model_size_gb: float = 0.0
    model_size_class: str = "unknown"
    context_label: str = "auto"
    preset: str = "normal"
    bench_path: str | None = None
    draft_model: str | None = None
    has_aiohttp: bool = False
    results_dir: str = ""
    active_toggles: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ModelInfo:
    """Represents a single GGUF model discovered on disk."""

    path: Path
    name: str
    size_gb: float
    is_current: bool


# ============================================================
# Exceptions
# ============================================================


class ServiceError(Exception):
    """Base exception for service-layer errors."""


class DatabaseResetError(ServiceError):
    """Raised when the Optuna DB or results cannot be deleted."""


class ModelSwitchError(ServiceError):
    """Raised when switching to a new model fails."""


class ConfigValidationError(ServiceError):
    """Raised when a configuration value is invalid."""


# ============================================================
# System Info
# ============================================================


def get_system_info(ctx: AppContext) -> SystemInfo:
    """Build a read-only snapshot of the current system configuration."""
    from ..engine import is_server_running

    py_ver = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    if ctx.is_moe:
        arch_type = "MoE"
        arch_detail = f"{ctx.default_experts} experts, {ctx.max_experts} max"
    else:
        arch_type = "Dense"
        arch_detail = ""

    gpu_layers = f"{ctx.default_gpu_layers}/{ctx.max_gpu_layers}"
    gpus = detect_gpus()
    model_class, model_size = classify_model(str(ctx.model_path))

    ctx_val = get_config("target_context")
    ctx_label = f"{ctx_val:,}" if ctx_val else "auto (4096 -> sweep)"

    active_toggles: list[str] = [
        label
        for config_key, label in _TOGGLE_REGISTRY.items()
        if get_config(config_key)
    ]

    return SystemInfo(
        python_version=py_ver,
        server_url=ctx.server_url,
        server_online=is_server_running(ctx),
        model_name=ctx.model_path.name,
        arch_type=arch_type,
        arch_detail=arch_detail,
        gpu_layers=gpu_layers,
        gpus=gpus,
        cpu_threads=ctx.max_threads,
        numa_nodes=ctx.numa_nodes,
        model_size_gb=model_size,
        model_size_class=model_class,
        context_label=ctx_label,
        preset=get_config("preset", "normal"),
        bench_path=str(ctx.bench_path) if ctx.bench_path else None,
        draft_model=get_config("draft_model"),
        has_aiohttp=HAS_AIOHTTP,
        results_dir=str(ctx.results_dir),
        active_toggles=active_toggles,
    )


# ============================================================
# Model Management
# ============================================================


def get_available_models(ctx: AppContext) -> list[ModelInfo]:
    """Scan the model directory tree for GGUF files.

    Returns sorted list of ModelInfo, excluding mmproj/reranker/embedding files.
    Searches upward from ctx.model_path to find the models root directory.
    """
    models_dir = ctx.model_path.parent.parent
    if len(models_dir.parts) <= 2:  # noqa: PLR2004
        models_dir = ctx.model_path.parent

    gguf_files = sorted(models_dir.rglob("*.gguf"))
    gguf_files = [
        f
        for f in gguf_files
        if f.is_file() and not f.is_symlink()
        and "mmproj" not in f.name.lower()
        and "reranker" not in f.parent.name.lower()
        and "embedding" not in f.parent.name.lower()
    ]

    results: list[ModelInfo] = []
    for f in gguf_files:
        try:
            size_gb = f.stat().st_size / (1024**3)
        except OSError:
            continue
        results.append(
            ModelInfo(
                path=f,
                name=f"{f.parent.name}/{f.name}",
                size_gb=round(size_gb, 1),
                is_current=(f == ctx.model_path),
            )
        )
    return results


def detect_architecture(model_path: Path) -> MoeArchConfig | DenseArchConfig | None:
    """Auto-detect model architecture from GGUF metadata if possible.

    Returns architecture config dict, or None if detection is not possible.
    """
    try:
        from ..models import detect_gguf_architecture

        return detect_gguf_architecture(str(model_path))
    except (ImportError, AttributeError):
        return None


def switch_to_model(
    ctx: AppContext,
    config: ConfigDict,
    model_path: Path,
    arch_config: MoeArchConfig | DenseArchConfig,
) -> None:
    """Switch the active model, rebuilding ctx and persisting config.

    Args:
        ctx: Application context (mutated in-place via rebuild_ctx).
        config: The module-level config dict.
        model_path: Path to the new GGUF model file.
        arch_config: Architecture dict, e.g. ``{"type": "moe", ...}``
                     or ``{"type": "dense"}``.

    Raises:
        ModelSwitchError: If the model file does not exist or config
                          cannot be saved.
    """
    if not model_path.is_file():
        raise ModelSwitchError(f"Model file not found: {model_path}")

    detected = detect_model_layers(str(model_path))
    max_ngl = detected or DEFAULT_MAX_GPU_LAYERS

    model_stem = model_path.stem.lower().replace(" ", "-")
    base_results_dir = Path(__file__).resolve().parent.parent / "results"
    new_results_dir = base_results_dir / model_stem
    new_results_dir.mkdir(parents=True, exist_ok=True)

    config["model"] = str(model_path)
    config["architecture"] = arch_config
    config["hardware"] = dict(get_config("hardware", {}))
    config["hardware"]["max_gpu_layers"] = max_ngl
    config["hardware"]["default_gpu_layers"] = max_ngl
    config["results_dir"] = str(new_results_dir)
    rebuild_ctx(config)

    model_class, model_size = classify_model(str(model_path))
    ctx.model_size_class = model_class
    ctx.model_size_gb = model_size
    ctx.quality_baseline = None

    config_path = Path(get_config("_config_path", "optimizer-config.json"))
    try:
        save_config_to_disk(config, config_path)
    except ServiceError as exc:
        raise ModelSwitchError(str(exc)) from exc


def build_arch_config_moe(
    expert_override_key: str,
    default_experts: int = DEFAULT_EXPERTS,
    max_experts: int = MAX_EXPERTS,
) -> MoeArchConfig:
    """Build a MoE architecture config dict.

    Args:
        expert_override_key: GGUF key for expert count override.
        default_experts: Trained default active experts.
        max_experts: Max experts to sweep.

    Returns:
        Architecture config dict with type="moe".

    Raises:
        ConfigValidationError: If expert_override_key contains invalid
                               characters.
    """
    if expert_override_key and not re.match(r"^[a-zA-Z0-9_.-]+$", expert_override_key):
        raise ConfigValidationError(
            "Invalid expert override key — only alphanumeric, underscore, "
            "hyphen, and dot allowed."
        )
    return {
        "type": "moe",
        "expert_override_key": expert_override_key,
        "default_experts": default_experts,
        "max_experts": max_experts,
    }


def build_arch_config_dense() -> DenseArchConfig:
    """Build a Dense architecture config dict.

    Returns:
        Architecture config dict with type="dense".
    """
    return {"type": "dense"}


# ============================================================
# Config / Preset / Toggle Management
# ============================================================


def cycle_preset(config: ConfigDict) -> str:
    """Cycle through quick/normal/thorough presets. Mutates *config*."""
    presets = ["quick", "normal", "thorough"]
    current = config.get("preset", "normal")
    idx = presets.index(current) if current in presets else 1
    new_idx = (idx + 1) % len(presets)
    new_preset = presets[new_idx]
    set_config("preset", new_preset)
    return new_preset


def apply_toggle(
    ctx: AppContext,
    config: ConfigDict,
    toggle_name: str,
    value: bool,
) -> None:
    """Apply a runtime toggle flag, updating config and ctx side-effects.

    Args:
        ctx: Application context (mutated for runtime flags).
        config: The config dict (mutated: toggle key is updated).
        toggle_name: One of: pareto, debug, no_jinja, no_bench,
                     fail_fast, skip_quality, interactive.
        value: New boolean value for the toggle.

    Raises:
        ConfigValidationError: If toggle_name is not recognized.
    """
    if toggle_name not in _TOGGLE_REGISTRY:
        raise ConfigValidationError(
            f"Unknown toggle: {toggle_name!r}. "
            f"Valid toggles: {', '.join(sorted(_TOGGLE_REGISTRY))}"
        )

    set_config(toggle_name, value)

    # Apply runtime side effects on ctx
    if toggle_name == "no_jinja":
        ctx.no_jinja = value
    elif toggle_name == "debug":
        ctx.debug = value
    elif toggle_name == "fail_fast":
        ctx.fail_fast = value
    elif toggle_name == "skip_quality":
        ctx.skip_quality = value
    elif toggle_name == "no_bench":
        if value:
            ctx.bench_path = None
        else:
            ctx.bench_path = find_llama_bench(str(ctx.server_path))


def set_context_size(
    ctx: AppContext,
    config: ConfigDict,
    size: int | str,
) -> None:
    """Set the target context size for optimization.

    Args:
        ctx: Application context (naked_engine is updated).
        config: The config dict (mutated).
        size: Integer context size, or the string ``"auto"`` to reset.

    Raises:
        ConfigValidationError: If size is below minimum or invalid.
    """
    if isinstance(size, str):
        if size.lower() == "auto":
            set_config("target_context", None)
            update_naked_engine(ctx, context=DEFAULT_CONTEXT_SIZE)
            return
        try:
            size = int(size)
        except ValueError as exc:
            raise ConfigValidationError(
                f"Invalid context size: {size!r}. Must be an integer or 'auto'."
            ) from exc

    if size < _MIN_CONTEXT:
        raise ConfigValidationError(f"Minimum context size is {_MIN_CONTEXT}.")
    if size > _MAX_CONTEXT:
        raise ConfigValidationError(
            f"Context size {size:,} exceeds 1M tokens. "
            "Confirm via the caller if intentional."
        )

    set_config("target_context", size)
    update_naked_engine(ctx, context=size)


def set_draft_model(
    ctx: AppContext,
    config: ConfigDict,
    path: str | None,
) -> DraftModelInfo:
    """Set or clear the draft model for speculative decoding.

    Args:
        ctx: Application context.
        config: The config dict (mutated).
        path: Path to a GGUF draft model, or None to disable.

    Returns:
        Dict with ``name`` and ``size_gb`` keys if a model was set,
        or empty dict if disabled.

    Raises:
        ConfigValidationError: If the file doesn't exist or isn't GGUF.
    """
    if path is None:
        set_config("draft_model", None)
        return {}

    p = Path(path)
    if not p.exists():
        raise ConfigValidationError(f"File not found: {path}")
    if p.suffix.lower() != ".gguf":
        raise ConfigValidationError(f"Not a GGUF file: {path}")

    set_config("draft_model", str(p))
    size_gb = p.stat().st_size / (1024**3)
    return {"name": p.name, "size_gb": round(size_gb, 1)}


def get_toggle_states(ctx: AppContext, config: ConfigDict) -> dict[str, bool]:
    """Return the current state of all runtime toggle flags."""
    return {key: bool(get_config(key, False)) for key in _TOGGLE_REGISTRY}


def save_config_to_disk(config: ConfigDict, config_path: Path) -> None:
    """Persist the config dict to a JSON file (atomic write).

    Args:
        config: Config dict to save (keys starting with ``_`` are excluded).
        config_path: Destination file path.

    Raises:
        ServiceError: If the file cannot be written.
    """
    import json
    import tempfile

    try:
        save_data = {k: v for k, v in config.items() if not k.startswith("_")}
        temp_fd, temp_path_str = tempfile.mkstemp(
            dir=str(config_path.parent), suffix=".json.tmp", prefix="config_"
        )
        temp_path = Path(temp_path_str)
        try:
            with open(temp_fd, "w", encoding="utf-8", closefd=True) as f:
                json.dump(save_data, f, indent=2)
            temp_path.replace(config_path)
        except KeyboardInterrupt:
            temp_path.unlink(missing_ok=True)
            raise
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    except (OSError, TypeError, ValueError) as exc:
        raise ServiceError(f"Could not save config to {config_path}: {exc}") from exc


def get_phase_trial_default(phase_name: str, preset: str) -> int:
    """Get the default trial count for a phase given the current preset.

    Trial counts come from ``PipelineConfig.default()`` — the single source
    of truth for per-phase trial values.  The *preset* acts as a multiplier:
    ``quick`` halves the base count, ``thorough`` adds 50%.

    Args:
        phase_name: Phase identifier (e.g. ``"core_engine"``, ``"quality"``).
        preset: Preset name (``"quick"``, ``"normal"``, ``"thorough"``).

    Returns:
        Default number of trials.
    """
    from ..pipeline_config import PipelineConfig

    cfg = PipelineConfig.default()
    phase = cfg.get_phase(phase_name)
    if phase is None or phase.trials is None:
        return DEFAULT_TRIAL_COUNT
    base = phase.trials
    multiplier = cfg.presets.get(preset, 1.0)
    return int(base * multiplier)
