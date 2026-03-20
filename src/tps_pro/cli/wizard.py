"""First-run setup wizard and model switching.

first_run_setup() is self-contained (runs before services are available).
switch_model() delegates to services.py for business logic.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ..constants import (
    DEFAULT_EXPERTS,
    DEFAULT_MAX_GPU_LAYERS,
    DEFAULT_PORT,
    MAX_EXPERTS,
)
from ..state import ctx, get_config
from .services import (
    ModelSwitchError,
    build_arch_config_dense,
    build_arch_config_moe,
    detect_architecture,
    get_available_models,
    switch_to_model,
)
from .setup_binary import SetupBinaryError, ensure_llama_server

# GGUF filename patterns to skip when scanning for models
# (vision projectors, embeddings, etc.)
_SKIP_PATTERNS = (
    "mmproj",
    "embedding",
    "reranker",
)


def needs_setup() -> bool:
    """Check if current config points to valid files."""
    server = get_config("server", "")
    model = get_config("model", "")
    return not (server and Path(server).is_file() and model and Path(model).is_file())


def switch_model() -> None:
    """Let user pick a new GGUF model from disk."""
    from .menu import clear_screen

    try:
        clear_screen()
        models = get_available_models(ctx)
        if not models:
            print("  No GGUF files found near current model directory.")
            return

        print("  Available models:\n")
        for i, m in enumerate(models):
            tag = " <- current" if m.is_current else ""
            print(f"    [{i + 1}] {m.name} ({m.size_gb:.1f} GB){tag}")
        print("\n    [0] Enter custom path")

        try:
            raw = input("\n  > ").strip()
        except EOFError:
            return

        if not raw or raw.lower() == "b":
            return

        new_model = _resolve_model_choice(raw, models)
        if new_model is None:
            return

        arch = _resolve_architecture(new_model)
        if arch is None:
            return

        switch_to_model(ctx, ctx.config, new_model, arch)
        from .menu import invalidate_header_cache

        invalidate_header_cache()
        print(f"\n  Switched to: {ctx.model_path.name}")
        kind = "MoE" if ctx.is_moe else "Dense"
        print(f"  Arch: {kind} | Layers: {ctx.max_gpu_layers}")
        print(f"  Results: {ctx.results_dir}/")
    except ModelSwitchError as exc:
        print(f"  [!] {exc}")
    except EOFError:
        return
    except Exception as exc:
        print(f"  [!] Error switching model: {exc}")


def _resolve_model_choice(raw: str, models: list) -> Path | None:
    """Turn user input into a Path, or None to cancel."""
    if raw == "0":
        try:
            path = input("  Path to GGUF: ").strip().strip('"').strip("'")
        except EOFError:
            return None
        if not path or not Path(path).is_file():
            print(f"  File not found: {path}")
            return None
        return Path(path)

    if raw.isdigit() and 1 <= int(raw) <= len(models):
        return models[int(raw) - 1].path

    print("  Invalid choice.")
    return None


def _resolve_architecture(model_path: Path) -> dict[str, Any] | None:
    """Detect or ask user for architecture config. Returns dict or None."""
    auto = detect_architecture(model_path)
    if auto:
        arch_type = auto.get("type", "dense")
        print(f"\n  Auto-detected architecture: {arch_type}")
        try:
            confirm = input("  Use auto-detected? [Y/n] ").strip().lower()
        except EOFError:
            confirm = "y"
        if confirm not in ("n", "no"):
            if arch_type == "moe":
                return build_arch_config_moe(
                    expert_override_key=auto.get("expert_override_key", ""),
                    default_experts=auto.get("default_experts", DEFAULT_EXPERTS),
                    max_experts=auto.get("max_experts", MAX_EXPERTS),
                )
            return build_arch_config_dense()

    return _ask_architecture(model_path)


def _ask_architecture(
    model_path: Path | None = None,
    *,
    first_run: bool = False,
) -> dict[str, Any] | None:
    """Prompt user to choose Dense or MoE. Returns config dict or None.

    Args:
        model_path: Optional model path (used to display the filename).
        first_run: When True, uses the compact first-run wizard display format
                   and returns raw dicts instead of typed config builders.
    """
    try:
        if first_run:
            print("\n  [3/5] Model architecture")
            print("        [1] MoE    [2] Dense")
            while True:
                choice = input("        > ").strip()
                if choice in ("1", "2"):
                    break
                print("        Enter 1 or 2.")
        else:
            label = f" for {model_path.name}" if model_path else ""
            print(f"\n  Architecture{label}?")
            print("    [1] MoE (Mixture of Experts)")
            print("    [2] Dense")
            choice = input("  > ").strip()
    except EOFError:
        return None

    if choice == "2":
        return {"type": "dense"} if first_run else build_arch_config_dense()
    if choice != "1":
        return None

    try:
        if first_run:
            print("\n        GGUF override key for expert count")
            while True:
                key = input("        > ").strip()
                if not key or re.match(r"^[a-zA-Z0-9_.-]+$", key):
                    break
                print("        Invalid key.")
            de = input(f"        Default active experts [{DEFAULT_EXPERTS}] > ").strip()
            me = input(f"        Max experts [{MAX_EXPERTS}] > ").strip()
        else:
            key = _ask_valid_key()
            de = input(f"  Default active experts [{DEFAULT_EXPERTS}]: ").strip()
            me = input(f"  Max experts [{MAX_EXPERTS}]: ").strip()
    except EOFError:
        return None

    if first_run:
        return {
            "type": "moe",
            "expert_override_key": key,
            "default_experts": _safe_int(de, DEFAULT_EXPERTS),
            "max_experts": _safe_int(me, MAX_EXPERTS),
        }
    return build_arch_config_moe(
        expert_override_key=key,
        default_experts=_safe_int(de, DEFAULT_EXPERTS),
        max_experts=_safe_int(me, MAX_EXPERTS),
    )


def _ask_valid_key() -> str:
    """Ask for a valid GGUF expert override key."""
    while True:
        key = input(
            "  Expert override key (e.g., qwen35moe.expert_used_count): "
        ).strip()
        if not key or re.match(r"^[a-zA-Z0-9_.-]+$", key):
            return key
        print("  Invalid key.")


def _safe_int(raw: str, default: int) -> int:
    """Parse int from string, return default on failure."""
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _resolve_server_path() -> str:
    """Auto-detect or download llama-server, falling back to manual input.

    Returns:
        Path string to llama-server executable.
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    try:
        server_path = ensure_llama_server(project_root)
        return str(server_path)
    except SetupBinaryError as exc:
        print(f"\n  [!] Auto-download failed: {exc}")
        print("  Please provide the path manually.")
        return _ask_path("1/5", "Path to llama-server executable", must_exist=True)


def first_run_setup() -> dict[str, Any]:
    """Interactive setup wizard for first-time users. Returns config dict."""
    print("=" * 60)
    print("  llama-server Parameter Optimizer -- First Run Setup")
    print("=" * 60)
    print()
    print("  This wizard will help you configure the optimizer.")
    print("  Your settings will be saved so you only do this once.")
    print()

    cfg = {}

    # Auto-detect / download llama-server (no user interaction needed)
    cfg["server"] = _resolve_server_path()

    cfg["model"] = _ask_models_folder()

    # Auto-detect chat template next to model
    model_dir = Path(cfg["model"]).parent
    jinja_files = list(model_dir.glob("*.jinja"))
    if jinja_files:
        cfg["chat_template"] = str(jinja_files[0])
        print(f"  [*] Found chat template: {jinja_files[0].name}")
    else:
        cfg["chat_template"] = ""

    # Auto-detect architecture from GGUF metadata
    model_path = Path(cfg["model"])
    auto_arch = detect_architecture(model_path)
    if auto_arch:
        arch_type = auto_arch.get("type", "dense")
        print(f"  [*] Auto-detected architecture: {arch_type}")
        if arch_type == "moe":
            cfg["architecture"] = {
                "type": "moe",
                "expert_override_key": auto_arch.get("expert_override_key", ""),
                "default_experts": auto_arch.get("default_experts", DEFAULT_EXPERTS),
                "max_experts": auto_arch.get("max_experts", MAX_EXPERTS),
            }
        else:
            cfg["architecture"] = {"type": "dense"}
    else:
        cfg["architecture"] = _ask_architecture(first_run=True)

    # Default to full GPU offload
    cfg["hardware"] = {"default_gpu_layers": DEFAULT_MAX_GPU_LAYERS}

    # Default port
    cfg["port"] = DEFAULT_PORT

    cfg["results_dir"] = str(Path(__file__).resolve().parent.parent / "results")
    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    config_path = results_dir / "optimizer-config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    cfg["_config_path"] = str(config_path)

    print(f"\n  Config saved to: {config_path}")
    print("  Edit this file to change settings later.")
    input("\n  Press Enter to start the optimizer...")

    return cfg


def _ask_models_folder() -> str:
    """Ask for models folder, scan for GGUFs, let user pick."""
    print("\n  [1/2] Path to your models folder")
    print("        (folder containing .gguf files)")
    while True:
        try:
            folder = input("        > ").strip().strip('"').strip("'")
        except EOFError as err:
            raise SystemExit(0) from err
        folder_path = Path(folder)
        if not folder_path.is_dir():
            print(f"        Not a valid folder: {folder}")
            continue

        ggufs = sorted(
            [
                f
                for f in folder_path.rglob("*.gguf")
                if f.is_file()
                and not f.is_symlink()
                and not any(s in f.name.lower() for s in _SKIP_PATTERNS)
            ],
            key=lambda f: f.stat().st_size,
            reverse=True,
        )
        if not ggufs:
            print("        No .gguf files found in that folder.")
            continue

        print(f"\n  Found {len(ggufs)} model(s):\n")
        for i, gguf in enumerate(ggufs[:20]):  # Show max 20
            size_gb = gguf.stat().st_size / (1024**3)
            rel = gguf.relative_to(folder_path)
            print(f"    [{i + 1}] {rel} ({size_gb:.1f} GB)")

        if len(ggufs) > 20:  # noqa: PLR2004
            print(f"    ... and {len(ggufs) - 20} more")

        print(f"\n    Default: [{1}] {ggufs[0].name}")
        try:
            choice = input("    > ").strip()
        except EOFError:
            choice = ""

        if not choice:
            selected = ggufs[0]
        elif choice.isdigit() and 1 <= int(choice) <= len(ggufs):
            selected = ggufs[int(choice) - 1]
        else:
            print("    Invalid choice, using first model.")
            selected = ggufs[0]

        size_gb = selected.stat().st_size / (1024**3)
        print(f"\n  [*] Selected: {selected.name} ({size_gb:.1f} GB)")
        return str(selected)


def _ask_path(step: str, prompt: str, must_exist: bool = False) -> str:
    """Ask for a file path, loop until valid if must_exist."""
    print(f"\n  [{step}] {prompt}")
    while True:
        path = input("        > ").strip().strip('"').strip("'")
        if must_exist and not Path(path).is_file():
            print(f"        File not found: {path}")
            continue
        return path
