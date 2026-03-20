"""
Application state: AppContext dataclass, _DEFAULTS, config loading.

Initialization Lifecycle
========================
This module uses a **module-level mutable singleton** pattern (``ctx`` and
``_config``).  The design is intentional and appropriate for a CLI tool (not a
library):

1. **Import time** — ``ctx`` and ``_config`` are created as *empty sentinels*
   (``AppContext()`` and ``{}``).  Other modules do
   ``from .state import ctx`` which binds their local name to the *same object*.

2. **Startup** — ``main.py`` calls ``initialize()`` *once*.  This parses CLI
   args, loads config.json, auto-detects hardware, and *mutates* the existing
   ``ctx`` / ``_config`` objects in-place (via ``rebuild_ctx``).  In-place
   mutation is required so that every module that already imported ``ctx`` sees
   the real values without a second import.

3. **Runtime** — ``ctx`` is fully populated; the optimizer pipeline reads and
   (in a few places) writes fields on it.

4. **Shutdown** — ``atexit`` closes the HTTP session.

Why singletons and not dependency injection?
    This is a single-entry-point CLI tool.  Every phase, measurement function,
    and display helper needs the same ``ctx``.  Threading it through 40+ call
    sites would add boilerplate with no testability gain — tests already
    construct ``AppContext()`` or ``SimpleNamespace`` mocks directly.

Guard against pre-init access
    ``AppContext._initialized`` is ``False`` until ``initialize()`` completes.
    Code that depends on real configuration should assert
    ``ctx._initialized`` or call ``initialize()`` first.

This is the foundation module — almost everything imports from here.
Only imports from .constants and .models (both are leaf modules
with no internal imports).

Error strategy (see errors.py for full documentation):
    - _load_config(): config file parse errors are logged at warning and
      fall back to defaults.  CLI arg validation uses argparse types.
    - _detect_numa_nodes(): returns 1 on any failure (safe default for
      single-NUMA systems which are the vast majority of consumer PCs).
    - AppContext.close(): catches OSError/RuntimeError on HTTP session
      close (logged at debug) -- must not prevent process exit.
"""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .constants import (
    BIND_HOST,
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_EXPERTS,
    DEFAULT_MAX_GPU_LAYERS,
    DEFAULT_PORT,
    MAX_EXPERTS,
)
from .models import detect_model_layers
from .result_types import (
    NakedEngineConfig,
    ServerProcess,
    TokenUncertaintyResult,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AppContext",
    "config",
    "create_context",
    "ctx",
    "get_config",
    "initialize",
    "rebuild_ctx",
    "replace_config",
    "set_config",
    "update_naked_engine",
]


_MAX_PORT = 65535

# ============================================================
# Default Configuration
# ============================================================

_DEFAULTS = {
    # Paths — set via first-run wizard or CLI args (--server, --model, --chat-template)
    "server": "",
    "model": "",
    "chat_template": "",
    "results_dir": str(Path(__file__).resolve().parent / "results"),
    "port": DEFAULT_PORT,
    # Model architecture — set via first-run wizard or config file
    "architecture": {
        "type": "dense",  # "moe" or "dense"
        "expert_override_key": "",  # GGUF key for expert count override
        "default_experts": DEFAULT_EXPERTS,  # trained default active experts
        "max_experts": MAX_EXPERTS,  # max experts to sweep
    },
    # Hardware — auto-detected if not set
    "hardware": {
        "max_threads": None,  # auto: os.cpu_count()
        "moe_sweep_max": None,  # auto: max_threads * 2 (capped at 40)
        "moe_sweep_center": None,  # auto: moe_sweep_max // 2
        "max_gpu_layers": None,  # auto-detected or DEFAULT_MAX_GPU_LAYERS
        "default_gpu_layers": DEFAULT_MAX_GPU_LAYERS,  # -ngl (all GPU)
    },
}


# ============================================================
# Known config keys — used by _load_config for schema validation
# ============================================================

_KNOWN_CONFIG_KEYS = frozenset(
    {
        "server",
        "model",
        "chat_template",
        "results_dir",
        "port",
        "architecture",
        "hardware",
        "_config_path",
        "fail_fast",
        "skip_quality",
        "no_jinja",
        "debug",
        "no_bench",
        "dashboard",
        "pareto",
        "simulate_users",
        "kill_competing",
        "preset",
        "dry_run",
        "batch_dir",
        "interactive",
        "timeout_minutes",
        "skip_existing",
    }
)


# ============================================================
# NUMA Detection (needed by _load_config)
# ============================================================


def _detect_numa_nodes() -> int:
    """Detect number of NUMA nodes on the system.

    Returns 1 for uniform-memory systems (most consumer PCs).
    Returns >1 for dual-socket Xeons, Threadrippers, or
    NUMA-aware Intel P/E core configs.
    """
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["wmic", "path", "Win32_NumaNode", "get", "NodeId"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                lines = [
                    line.strip()
                    for line in result.stdout.strip().splitlines()
                    if line.strip()
                    and line.strip() != "NodeId"
                    and line.strip().isdigit()
                ]
                return max(1, len(lines))
        else:
            numa_path = Path("/sys/devices/system/node")
            if numa_path.exists():
                nodes = list(numa_path.glob("node[0-9]*"))
                return max(1, len(nodes))
            result = subprocess.run(
                ["numactl", "--hardware"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if result.returncode == 0:
                avail_line = next(
                    (
                        line
                        for line in result.stdout.split("\n")
                        if "available:" in line
                    ),
                    None,
                )
                if avail_line is not None:
                    parts = avail_line.split()
                    idx = next(
                        (i for i, p in enumerate(parts) if p == "available:"),
                        None,
                    )
                    if idx is not None and idx + 1 < len(parts):
                        return int(parts[idx + 1])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass  # NUMA detection is optional -- default to 1 (uniform memory)
    return 1


# ============================================================
# llama-bench discovery (needed by create_context)
# ============================================================


def find_llama_bench(server_path: str | Path) -> Path | None:
    """Discover llama-bench binary next to llama-server. Returns Path or None."""
    if not server_path:
        return None
    server_dir = Path(server_path).parent
    if sys.platform == "win32":
        bench = server_dir / "llama-bench.exe"
    else:
        bench = server_dir / "llama-bench"
    return bench if bench.is_file() else None


# ============================================================
# AppContext Dataclass
# ============================================================


@dataclass
class AppContext:
    """All mutable application state in one place (37 fields).

    Must be initialized via ``initialize()`` before use by the pipeline.
    Accessing a non-initialized context will work (sentinel defaults) but
    callers that depend on real configuration should check ``_initialized``.

    Field Groups
    ============
    - **Initialization guard**: _initialized
    - **Paths & network**: server_path, model_path, chat_template_path,
      results_dir, lookup_cache_file, optuna_db, port, _port_alt, server_url
    - **Raw config**: config
    - **Architecture**: is_moe, expert_override_key, default_experts, max_experts
    - **Hardware limits**: max_threads, moe_sweep_max, moe_sweep_center,
      max_gpu_layers, default_gpu_layers
    - **Runtime flags**: naked_engine, fail_fast, skip_quality, dry_run,
      debug, fresh_run, no_jinja
    - **Model metadata**: model_size_class, model_size_gb
    - **Session / server**: http, active_server_proc, _dying_server_proc,
      quality_baseline, vram_total_mb
    - **Tooling**: bench_path, numa_nodes
    - **Pipeline state**: skip_flags, kl_baseline_cache
    - **Internal**: _PRE_INIT_SAFE
    """

    # Initialization guard — set to True by initialize() after config is loaded.
    _initialized: bool = False

    # Paths
    server_path: Path = field(default_factory=lambda: Path(""))
    model_path: Path = field(default_factory=lambda: Path(""))
    chat_template_path: Path = field(default_factory=lambda: Path(""))
    results_dir: Path = field(default_factory=lambda: Path(""))
    lookup_cache_file: str = ""
    optuna_db: str = ""
    port: int = DEFAULT_PORT
    _port_alt: int = DEFAULT_PORT + 1  # ping-pong alternate port
    server_url: str = f"http://{BIND_HOST}:{DEFAULT_PORT}"

    # Config dict (raw)
    config: dict[str, Any] = field(default_factory=dict)

    # Architecture
    is_moe: bool = False
    expert_override_key: str = ""
    default_experts: int = DEFAULT_EXPERTS
    max_experts: int = MAX_EXPERTS

    # Hardware
    max_threads: int = 8
    moe_sweep_max: int = 16
    moe_sweep_center: int = 8
    max_gpu_layers: int = DEFAULT_MAX_GPU_LAYERS
    default_gpu_layers: int = DEFAULT_MAX_GPU_LAYERS

    # Runtime
    naked_engine: NakedEngineConfig = field(default_factory=NakedEngineConfig)
    fail_fast: bool = False
    skip_quality: bool = False
    dry_run: bool = False
    debug: bool = False
    fresh_run: bool = False

    # Model classification
    model_size_class: str = (
        "medium"  # tiny/small/medium/large — drives wait_for_server timeouts
    )
    model_size_gb: float = (
        0.0  # model file size in GB — used for VRAM budget calculations
    )

    # Session
    http: requests.Session = field(default_factory=requests.Session)
    active_server_proc: ServerProcess | None = None
    _dying_server_proc: ServerProcess | None = (
        None  # ping-pong: process being killed async
    )
    quality_baseline: TokenUncertaintyResult | None = None

    # VRAM cache (populated once after first server warmup)
    vram_total_mb: float | None = None

    # Server flags
    no_jinja: bool = (
        False  # --no-jinja escape hatch for models that crash on Jinja parsing
    )

    # llama-bench integration
    bench_path: Path | None = None  # path to llama-bench binary (None = HTTP-only mode)

    # NUMA detection
    numa_nodes: int = 1

    # GGUF-aware flag filtering: flags detected as irrelevant for this model/config
    # Populated by pipeline init from GGUF metadata + GPU offload results.
    # core_engine skips suggesting these params and uses their default values instead.
    skip_flags: set[str] = field(default_factory=set)

    # KL-divergence baseline cache
    kl_baseline_cache: list | None = None

    # Names that are safe to access before initialize() — prevents warning noise
    # during construction, import-time attribute checks, and cleanup.
    _PRE_INIT_SAFE = frozenset(
        {
            "_initialized",
            "_PRE_INIT_SAFE",
            "_pre_init_warned",
            "close",
            "http",
            "config",
            "__class__",
            "__dict__",
            "__dataclass_fields__",
            "__dataclass_params__",
            "__repr__",
            "__eq__",
            "__hash__",
        }
    )

    def close(self) -> None:
        """Clean up resources."""
        try:
            self.http.close()
        except (OSError, RuntimeError, AttributeError) as e:
            logger.debug("HTTP session close failed: %s", e)


# ============================================================
# Context Factory
# ============================================================


_EXPERT_KEY_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")


def create_context(config: dict) -> AppContext:
    """Build AppContext from a loaded config dict."""
    arch = config["architecture"]
    hw = config["hardware"]

    # Validate expert_override_key (also validated in wizard, but enforce here too)
    expert_key = arch.get("expert_override_key", "")
    if expert_key and not _EXPERT_KEY_RE.match(expert_key):
        logger.warning(
            "Invalid expert_override_key %r — resetting to empty", expert_key
        )
        arch = {**arch, "expert_override_key": ""}
    results_dir = Path(config["results_dir"])
    default_gpu_layers_capped = min(
        hw.get("default_gpu_layers", DEFAULT_MAX_GPU_LAYERS), hw["max_gpu_layers"]
    )

    naked_engine_typed = NakedEngineConfig(
        context=DEFAULT_CONTEXT_SIZE,
        mlock=True,
        n_gpu_layers=default_gpu_layers_capped,
    )

    ctx = AppContext(
        server_path=Path(config["server"]),
        model_path=Path(config["model"]),
        chat_template_path=Path(config["chat_template"]).resolve()
        if config.get("chat_template")
        else Path(""),
        results_dir=results_dir,
        lookup_cache_file=str(results_dir / "lookup-cache.bin"),
        optuna_db="sqlite:///" + str(results_dir / "optuna.db").replace("\\", "/"),
        port=config["port"],
        _port_alt=config["port"] + 1,
        server_url="http://" + BIND_HOST + ":" + str(config["port"]),
        config=config,
        is_moe=arch.get("type", "dense") == "moe",
        expert_override_key=arch.get("expert_override_key", ""),
        default_experts=arch.get("default_experts", DEFAULT_EXPERTS),
        max_experts=arch.get("max_experts", MAX_EXPERTS),
        max_threads=hw["max_threads"],
        moe_sweep_max=hw["moe_sweep_max"],
        moe_sweep_center=hw["moe_sweep_center"],
        max_gpu_layers=hw["max_gpu_layers"],
        default_gpu_layers=default_gpu_layers_capped,
        naked_engine=naked_engine_typed,
        fail_fast=config.get("fail_fast", False),
        skip_quality=config.get("skip_quality", False),
        no_jinja=config.get("no_jinja", False),
        debug=config.get("debug", False),
        bench_path=None
        if config.get("no_bench")
        else find_llama_bench(config["server"]),
        numa_nodes=hw.get("numa_nodes", 1),
    )

    # Configure HTTP session with retry adapter for transient errors
    retry = Retry(total=2, backoff_factor=0.1, status_forcelist=[502])
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=10)
    ctx.http.mount("http://", adapter)

    return ctx


# ============================================================
# Config Loading
# ============================================================


def _load_file_config(path: Path) -> dict[str, Any]:
    """Read JSON configuration from disk.

    Returns the parsed dict, or an empty dict if the file does not exist
    or cannot be decoded.

    Args:
        path: Path to the JSON config file.
    """
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Config file corrupted (%s) — using defaults", e)
        return {}


def _merge_cli_args(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Merge CLI argument overrides into *config* and return a new dict.

    Args:
        config: Base configuration dict (will not be mutated).
        args: Parsed CLI namespace.
    """
    merged = copy.deepcopy(config)

    if args.server:
        merged["server"] = args.server
    if args.model:
        merged["model"] = args.model
    if args.chat_template:
        merged["chat_template"] = args.chat_template
    if args.results_dir:
        merged["results_dir"] = args.results_dir
    if args.port:
        merged["port"] = args.port
    if args.dense:
        merged["architecture"]["type"] = "dense"
    if args.fail_fast:
        merged["fail_fast"] = True
    if args.skip_quality:
        merged["skip_quality"] = True
    merged["preset"] = args.preset
    merged["dry_run"] = args.dry_run
    merged["debug"] = args.debug
    if args.batch:
        merged["batch_dir"] = args.batch
    merged["interactive"] = args.interactive
    merged["timeout_minutes"] = args.timeout
    merged["skip_existing"] = args.skip_existing
    merged["no_jinja"] = args.no_jinja
    merged["no_bench"] = args.no_bench
    merged["dashboard"] = args.dashboard
    merged["pareto"] = args.pareto
    merged["simulate_users"] = args.simulate_users
    merged["kill_competing"] = args.kill_competing

    return merged


def _auto_detect_hardware(config: dict[str, Any]) -> dict[str, Any]:
    """Fill in hardware defaults that were not explicitly set.

    Auto-detects NUMA topology, CPU thread count, and GPU layer count.
    Returns a new config dict with the hardware section populated.

    Args:
        config: Configuration dict (will not be mutated).
    """
    result = copy.deepcopy(config)
    hw = result["hardware"]

    if hw["max_threads"] is None:
        hw["max_threads"] = os.cpu_count() or 16
    if "numa_nodes" not in hw:
        hw["numa_nodes"] = _detect_numa_nodes()
    if hw["moe_sweep_max"] is None:
        hw["moe_sweep_max"] = min(hw["max_threads"] * 2, 40)
    if hw["moe_sweep_center"] is None:
        hw["moe_sweep_center"] = hw["moe_sweep_max"] // 2
    if hw["max_gpu_layers"] is None:
        detected = detect_model_layers(result.get("model", ""))
        hw["max_gpu_layers"] = detected or DEFAULT_MAX_GPU_LAYERS

    return result


def _resolve_server_binary(config: dict[str, Any]) -> dict[str, Any]:
    """Find or download llama-server when no server path is configured.

    If a binary is found, the path is persisted to the config file so
    auto-download only happens once.  Returns a new config dict.

    Args:
        config: Configuration dict (will not be mutated).
    """
    result = copy.deepcopy(config)

    if result.get("server"):
        return result

    try:
        from .cli.setup_binary import SetupBinaryError, ensure_llama_server

        project_root = Path(__file__).resolve().parent.parent.parent
        server_path = ensure_llama_server(project_root)
        result["server"] = str(server_path)
        # Persist to config file so this only happens once
        config_file_path = result.get("_config_path")
        if config_file_path and Path(config_file_path).exists():
            try:
                with open(config_file_path, encoding="utf-8") as f:
                    persisted = json.load(f)
                persisted["server"] = str(server_path)
                with open(config_file_path, "w", encoding="utf-8") as f:
                    json.dump(persisted, f, indent=2)
            except (json.JSONDecodeError, OSError):
                pass  # Non-fatal: config file will be written by wizard
    except (OSError, RuntimeError, ImportError, ValueError, SetupBinaryError) as exc:
        logger.warning("Auto-download of llama-server failed: %s", exc)

    return result


def _load_config() -> dict[str, Any]:
    """Load config from CLI args, config.json, or defaults.

    Orchestrates four steps:
    1. Read JSON from disk via ``_load_file_config``
    2. Merge CLI overrides via ``_merge_cli_args``
    3. Find / download llama-server via ``_resolve_server_binary``
    4. Auto-detect hardware via ``_auto_detect_hardware``
    """
    from .cli.args import parse_cli_args

    args = parse_cli_args()

    config = copy.deepcopy(_DEFAULTS)

    # Layer 1: config.json file (deep merge for nested dicts)
    config_path = args.config or str(
        Path(__file__).resolve().parent / "results" / "optimizer-config.json"
    )
    config["_config_path"] = config_path
    file_config = _load_file_config(Path(config_path))
    # Basic schema validation: only merge known top-level keys
    for k, v in file_config.items():
        if k not in _KNOWN_CONFIG_KEYS and not k.startswith("_"):
            logger.warning("Ignoring unknown config key: %r", k)
            continue
        if isinstance(v, dict) and isinstance(config.get(k), dict):
            config[k].update(v)
        else:
            config[k] = v

    # Validate port from config file (CLI port is validated by argparse)
    config_port = config.get("port")
    if config_port is not None:
        if not isinstance(config_port, int) or not (1 <= config_port <= _MAX_PORT):
            logger.warning(
                "Invalid port %r in config file — falling back to default %d",
                config_port,
                _DEFAULTS["port"],
            )
            config["port"] = _DEFAULTS["port"]

    # Canonicalize results_dir early for consistent downstream paths
    config["results_dir"] = str(Path(config["results_dir"]).resolve())

    # Layer 2: CLI args override everything
    config = _merge_cli_args(config, args)

    # Layer 2.5: Auto-resolve server binary if not set
    config = _resolve_server_binary(config)

    # Layer 3: Auto-detect hardware if not explicitly set
    config = _auto_detect_hardware(config)

    return config


# ============================================================
# Module-level initialization (lazy)
# ============================================================

# Sentinel instances — populated in-place by initialize().
# Other modules do `from .state import ctx, config`, so we must update
# the *same objects* rather than rebinding the names.
#
# See module docstring "Initialization Lifecycle" for full rationale.
ctx = AppContext()
config: dict[str, Any] = {}
_initialized = False



def replace_config(new_config: dict[str, Any]) -> None:
    """Replace the contents of the module-level ``config`` dict.

    Clears and repopulates the *same dict object* so that all modules
    which imported ``config`` via ``from .state import config`` see the
    new values without rebinding their local names.

    Args:
        new_config: The new configuration dict whose contents will replace config.
    """
    if new_config is not config:
        config.clear()
        config.update(new_config)


def get_config(key: str, default: Any = None) -> Any:
    """Read a value from the module-level config dict.

    Provides controlled access to the mutable ``config`` singleton so
    callers do not need to import and index the raw dict directly.

    Args:
        key: Configuration key to look up.
        default: Value returned when *key* is absent (default ``None``).

    Returns:
        The config value, or *default* if the key is not present.
    """
    return config.get(key, default)


def set_config(key: str, value: Any) -> None:
    """Write a value into the module-level config dict.

    Provides controlled mutation of the mutable ``config`` singleton so
    callers do not need to import and index the raw dict directly.

    Args:
        key: Configuration key to set.
        value: New value for the key.
    """
    config[key] = value


def update_naked_engine(ctx: AppContext, **kwargs: Any) -> None:
    """Update ctx.naked_engine by replacing it with a merged copy.

    Creates a new dict from the current naked_engine merged with ``kwargs``
    and assigns it back to ``ctx.naked_engine``.  Replaces ctx.naked_engine
    with a new merged dict. ctx itself is mutated.

    Args:
        ctx: Application context whose naked_engine will be replaced.
        **kwargs: Key-value pairs to merge into the new naked_engine dict.
    """
    ctx.naked_engine = {**ctx.naked_engine, **kwargs}


def rebuild_ctx(config: dict) -> AppContext:
    """Rebuild the module-level ctx from a new config dict.

    Replaces ``config`` contents and copies all fields from a freshly created
    AppContext into the existing module-level ``ctx`` so that all modules
    which imported ``ctx`` via ``from .state import ctx`` see the new state.

    Args:
        config: The new configuration dict.

    Returns:
        The updated module-level ctx.
    """
    replace_config(config)
    fresh = create_context(config)
    # Temporarily mark BOTH as initialized to suppress
    # pre-init warnings during field copy (getattr on fresh
    # also triggers __getattribute__ which warns if not init)
    was_initialized = object.__getattribute__(ctx, "_initialized")
    object.__setattr__(ctx, "_initialized", True)
    object.__setattr__(fresh, "_initialized", True)
    for f in ctx.__dataclass_fields__:
        setattr(ctx, f, getattr(fresh, f))
    # Restore original state (initialize() will set _initialized=True permanently)
    object.__setattr__(ctx, "_initialized", was_initialized)
    return ctx


def initialize() -> AppContext:
    """Parse CLI args, load config, populate module-level ctx/config.

    Safe to call multiple times — subsequent calls are no-ops.
    Must be called once before the optimizer pipeline runs
    (typically from main.py's main()).
    """
    global _initialized
    if _initialized:
        return ctx
    config = _load_config()
    rebuild_ctx(config)
    atexit.register(ctx.close)
    _initialized = True
    ctx._initialized = True
    return ctx
