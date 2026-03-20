"""Command generation and HTML report delegation.

Pure business-logic — ZERO print/input calls.
Split from services.py for cohesion; re-exported via services.py for
backward compatibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..constants import DEFAULT_EXPERTS, SCORE_VERSION
from ..engine.util import read_json_safe
from ..hardware import detect_gpus
from ..search import ensure_results_dir
from ..state import AppContext

logger = logging.getLogger(__name__)

# Boolean toggle flags for llama-server command generation.
_BOOL_FLAGS: list[tuple[str, str, bool]] = [
    ("swa_full", "--swa-full", True),
    ("mlock", "--mlock", True),
    ("no_mmap", "--no-mmap", True),
]


# Phase ordering used by _merge_phase_results / command generation.
_MERGE_PHASE_ORDER: list[str] = [
    "gpu",
    "tensor_split",
    "topology_sweep",
    "moe_combined",
    "core_engine",
    "io_toggles",
    "kv_context_sweep",
    "speculation",
    "kv_quality",
    "quality",
]


# Parameter-to-CLI-flag mapping for llama-server command generation.
_FLAG_MAP: list[tuple[str, str]] = [
    ("n_gpu_layers", "-ngl"),
    ("context", "-c"),
    ("threads", "-t"),
    ("threads_batch", "-tb"),
    ("n_cpu_moe", "--n-cpu-moe"),
    ("batch_size", "-b"),
    ("ubatch_size", "--ubatch-size"),
    ("poll", "--poll"),
    ("poll_batch", "--poll-batch"),
    ("prio", "--prio"),
    ("prio_batch", "--prio-batch"),
    ("cpu_strict", "--cpu-strict"),
    ("cpu_strict_batch", "--cpu-strict-batch"),
    ("spec_type", "--spec-type"),
    ("spec_ngram_n", "--spec-ngram-size-n"),
    ("spec_ngram_m", "--spec-ngram-size-m"),
    ("spec_ngram_min_hits", "--spec-ngram-min-hits"),
    ("draft_max", "--draft"),
    ("draft_min", "--draft-min"),
    ("draft_p_min", "--draft-p-min"),
    ("model_draft", "--model-draft"),
    ("cache_reuse", "--cache-reuse"),
    ("temperature", "--temp"),
    ("top_p", "--top-p"),
    ("top_k", "--top-k"),
    ("min_p", "--min-p"),
    ("typical_p", "--typical-p"),
    ("repeat_penalty", "--repeat-penalty"),
    ("presence_penalty", "--presence-penalty"),
    ("frequency_penalty", "--frequency-penalty"),
    ("mirostat", "--mirostat"),
    ("mirostat_lr", "--mirostat-lr"),
    ("mirostat_ent", "--mirostat-ent"),
    ("repeat_last_n", "--repeat-last-n"),
    ("top_n_sigma", "--top-n-sigma"),
    ("dynatemp_range", "--dynatemp-range"),
    ("dynatemp_exp", "--dynatemp-exponent"),
    ("xtc_probability", "--xtc-probability"),
    ("xtc_threshold", "--xtc-threshold"),
    ("dry_multiplier", "--dry-multiplier"),
    ("dry_base", "--dry-base"),
    ("dry_allowed_length", "--dry-allowed-length"),
    ("dry_penalty_last_n", "--dry-penalty-last-n"),
]


def _load_phase_data(results_dir: Path, phase_name: str) -> dict | None:
    """Load and validate a single phase result file.

    Returns:
        Parsed data dict, or None if the file is missing, corrupt, or stale.
    """
    path = results_dir / f"{phase_name}_results.json"
    if not path.exists():
        return None
    data = read_json_safe(path, logger)
    if data is None:
        logger.warning("%s results corrupted -- skipping", phase_name)
        return None
    if "score_version" in data and data["score_version"] != SCORE_VERSION:
        logger.warning(
            "Skipping stale %s results (score v%s != v%s)",
            phase_name,
            data["score_version"],
            SCORE_VERSION,
        )
        return None
    return data


def _extract_phase_params(phase_name: str, data: dict) -> dict[str, Any]:
    """Extract the relevant parameters from a validated phase result.

    Returns:
        Dict of parameters to merge, or empty dict.
    """
    if phase_name == "gpu" and "best_ngl" in data:
        return {"n_gpu_layers": data["best_ngl"]}
    if phase_name in ("tensor_split", "topology_sweep") and "best_split_str" in data:
        return {"tensor_split": data["best_split_str"]}
    if "best_params" in data:
        return dict(data["best_params"])
    return {}


def _merge_phase_results(results_dir: Path) -> dict[str, Any]:
    """Load and merge best_params from all pipeline phase result files.

    Args:
        results_dir: Path to the results directory.

    Returns:
        Merged parameters from all completed phases, or empty dict.
    """
    merged_params: dict[str, Any] = {}
    for phase_name in _MERGE_PHASE_ORDER:
        data = _load_phase_data(results_dir, phase_name)
        if data is None:
            continue
        merged_params.update(_extract_phase_params(phase_name, data))
    return merged_params


def _quote_path(path_str: str) -> str:
    """Quote a path string if it contains spaces."""
    return f'"{path_str}"' if " " in str(path_str) else str(path_str)


def _append_special_params(
    parts: list[str],
    p: dict[str, Any],
    expert_override_key: str,
    default_experts: int,
) -> None:
    """Append tensor-split, flash-attn, kv-cache, and expert override flags."""
    if "tensor_split" in p:
        parts.extend(["--tensor-split", p["tensor_split"]])
    if p.get("flash_attn") in ("on", True, "1", 1):
        parts.append("--flash-attn")
    if "kv_cache_type" in p:
        parts.extend(
            ["--cache-type-k", p["kv_cache_type"], "--cache-type-v", p["kv_cache_type"]]
        )
    if (
        "expert_used_count" in p
        and expert_override_key
        and p["expert_used_count"] != default_experts
    ):
        parts.extend(
            ["--override-kv", f"{expert_override_key}=int:{p['expert_used_count']}"]
        )


def _append_boolean_flags(
    parts: list[str],
    p: dict[str, Any],
    results_dir: Path,
    no_jinja: bool,
) -> None:
    """Append boolean toggle flags and optional lookup cache / numa / jinja flags."""
    for key, flag, truthy in _BOOL_FLAGS:
        if p.get(key) == truthy:
            parts.append(flag)
    if p.get("repack") is False:
        parts.append("--no-repack")
    if p.get("op_offload") is False:
        parts.append("--no-op-offload")

    if p.get("lookup_cache_dynamic") or p.get("use_lookup_cache"):
        cache_path = p.get("lookup_cache_dynamic") or str(
            results_dir / "lookup-cache.bin"
        )
        parts.extend(["--lookup-cache-dynamic", cache_path])
    if "numa" in p:
        parts.extend(["--numa", str(p["numa"])])
    if no_jinja:
        parts.extend(["--no-jinja", "--chat-template", "chatml"])


def _build_command_parts(  # noqa: PLR0913
    params: dict[str, Any],
    server_path: str,
    model_path: str,
    chat_template_path: str,
    port: int,
    results_dir: Path,
    *,
    no_jinja: bool = False,
    expert_override_key: str = "",
    default_experts: int = DEFAULT_EXPERTS,
) -> list[str]:
    """Build the list of CLI tokens for a llama-server command.

    Args:
        params: Merged optimization parameters dict.
        server_path: Path to the llama-server executable.
        model_path: Path to the model file.
        chat_template_path: Path to the chat template file.
        port: Server port number.
        results_dir: Path to the results directory.
        no_jinja: Whether ``--no-jinja`` flag should be appended.
        expert_override_key: GGUF key for expert count override.
        default_experts: Default number of active experts.

    Returns:
        Ordered list of command-line tokens.
    """
    parts = [
        _quote_path(server_path),
        "-m",
        _quote_path(model_path),
        "--port",
        str(port),
    ]
    if chat_template_path and Path(chat_template_path).is_file():
        parts.extend(["--chat-template-file", _quote_path(chat_template_path)])

    for key, flag in _FLAG_MAP:
        if key in params:
            parts.extend([flag, str(params[key])])

    _append_special_params(parts, params, expert_override_key, default_experts)
    _append_boolean_flags(parts, params, results_dir, no_jinja)

    return parts


def _format_command(parts: list[str]) -> str:
    """Format a list of command tokens into a human-readable string.

    Args:
        parts: List of CLI tokens.

    Returns:
        Formatted command string with line continuations.
    """
    lines = [parts[0]]
    i = 1
    while i < len(parts):
        token = parts[i]
        if (
            token.startswith("-")
            and i + 1 < len(parts)
            and not parts[i + 1].startswith("-")
        ):
            lines.append(f"  {token} {parts[i + 1]}")
            i += 2
        else:
            lines.append(f"  {token}")
            i += 1
    return " \\\n".join(lines)


def generate_optimized_command(ctx: AppContext) -> str | None:
    """Generate a ready-to-use llama-server command from optimization results.

    Merges all phase results, builds the CLI command, and persists both
    ``command.txt`` and ``command.json`` to the results directory.

    Args:
        ctx: Application context.

    Returns:
        Formatted command string, or None if no results exist.
    """
    results_dir = ctx.results_dir
    merged_params = _merge_phase_results(results_dir)
    if not merged_params:
        return None

    parts = _build_command_parts(
        merged_params,
        str(ctx.server_path),
        str(ctx.model_path),
        str(ctx.chat_template_path),
        ctx.port,
        results_dir,
        no_jinja=ctx.no_jinja,
        expert_override_key=ctx.expert_override_key,
        default_experts=ctx.default_experts,
    )
    command_str = _format_command(parts)

    if (
        merged_params.get("context_shift") is False
        or "--no-context-shift" in command_str
    ):
        logger.warning(
            "[!] WARNING: This config disables context shift (--no-context-shift). "
            "This improves speed but BREAKS long multi-turn chats. "
            "Remove --no-context-shift if using this for a conversational UI."
        )

    ensure_results_dir(ctx)
    cmd_path = results_dir / "command.txt"
    with open(cmd_path, "w", encoding="utf-8") as f:
        f.write(command_str + "\n")

    cmd_json_path = results_dir / "command.json"
    with open(cmd_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "server": str(ctx.server_path),
                "model": str(ctx.model_path),
                "port": ctx.port,
                "params": merged_params,
            },
            f,
            indent=2,
        )

    return command_str


def generate_html_report(ctx: AppContext) -> str | None:
    """Generate a comprehensive HTML report from optimization results.

    Delegates to the existing report module's generate_html_report
    function.

    Args:
        ctx: Application context.

    Returns:
        Path to the generated HTML report file, or None if no results.
    """
    from .report import generate_html_report as _gen_html

    return _gen_html(
        results_dir=str(ctx.results_dir),
        model_name=ctx.model_path.name,
        gpus=detect_gpus(),
    )
