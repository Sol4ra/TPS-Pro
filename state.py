"""
Application state: AppContext dataclass, _DEFAULTS, config loading.

This is the foundation module — almost everything imports from here.
Only imports from .constants and .models (both are leaf modules with no internal imports).
"""

import argparse
import copy
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

from .constants import SCORE_VERSION
from .models import detect_model_layers


# ============================================================
# Default Configuration
# ============================================================

_DEFAULTS = {
    # Paths — set via first-run wizard or CLI args (--server, --model, --chat-template)
    "server": "",
    "model": "",
    "chat_template": "",
    "results_dir": str(Path(__file__).resolve().parent / "results"),
    "port": 8090,

    # Model architecture — set via first-run wizard or config file
    "architecture": {
        "type": "dense",                                     # "moe" or "dense"
        "expert_override_key": "",                           # GGUF key for expert count override
        "default_experts": 8,                                # trained default active experts
        "max_experts": 16,                                   # max experts to sweep
    },

    # Hardware — auto-detected if not set
    "hardware": {
        "max_threads": None,          # auto: os.cpu_count()
        "moe_sweep_max": None,        # auto: max_threads * 2 (capped at 40)
        "moe_sweep_center": None,     # auto: moe_sweep_max // 2
        "max_gpu_layers": None,       # auto-detected from model metadata, or 99
        "default_gpu_layers": 99,     # default -ngl for naked engine (99 = all GPU)
    },
}


# ============================================================
# Presets — trial counts per phase
# ============================================================

_PRESETS = {
    "quick": {"moe": 15, "compute": 15, "memory": 15, "compute_audit": 10, "moe_audit": 10, "memory_audit": 10, "quality": 20},
    "normal": {"moe": 30, "compute": 40, "memory": 40, "compute_audit": 30, "moe_audit": 30, "memory_audit": 30, "quality": 40},
    "thorough": {"moe": 80, "compute": 100, "memory": 100, "compute_audit": 80, "moe_audit": 80, "memory_audit": 80, "quality": 120},
}


def get_preset_trials(preset, phase_name):
    """Get the trial count for a phase from a preset."""
    if preset not in _PRESETS:
        preset = "normal"
    return _PRESETS[preset].get(phase_name, 60)


# ============================================================
# NUMA Detection (needed by _load_config)
# ============================================================

def _detect_numa_nodes():
    """Detect number of NUMA nodes on the system.

    Returns 1 for uniform-memory systems (most consumer PCs).
    Returns >1 for dual-socket Xeons, Threadrippers, or NUMA-aware Intel P/E core configs.
    """
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["wmic", "path", "Win32_NumaNode", "get", "NodeId"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip() and l.strip() != "NodeId"]
                return max(1, len(lines))
        else:
            numa_path = Path("/sys/devices/system/node")
            if numa_path.exists():
                nodes = list(numa_path.glob("node[0-9]*"))
                return max(1, len(nodes))
            result = subprocess.run(["numactl", "--hardware"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "available:" in line:
                        parts = line.split()
                        for i, p in enumerate(parts):
                            if p == "available:" and i + 1 < len(parts):
                                return int(parts[i + 1])
    except Exception:
        pass
    return 1


# ============================================================
# llama-bench discovery (needed by create_context)
# ============================================================

def _find_llama_bench(server_path):
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
    """All mutable application state in one place."""

    # Paths
    server_path: Path = field(default_factory=lambda: Path(""))
    model_path: Path = field(default_factory=lambda: Path(""))
    chat_template_path: Path = field(default_factory=lambda: Path(""))
    results_dir: Path = field(default_factory=lambda: Path(""))
    lookup_cache_file: str = ""
    optuna_db: str = ""
    port: int = 8090
    server_url: str = "http://127.0.0.1:8090"

    # Config dict (raw)
    config: dict = field(default_factory=dict)

    # Architecture
    arch: dict = field(default_factory=dict)
    is_moe: bool = False
    expert_override_key: str = ""
    default_experts: int = 8
    max_experts: int = 16

    # Hardware
    hw: dict = field(default_factory=dict)
    max_threads: int = 8
    moe_sweep_max: int = 16
    moe_sweep_center: int = 8
    max_gpu_layers: int = 99
    default_gpu_layers: int = 99

    # Runtime
    naked_engine: dict = field(default_factory=dict)
    fail_fast: bool = False
    skip_quality: bool = False
    dry_run: bool = False
    debug: bool = False
    fresh_run: bool = False

    # Model classification
    model_size_class: str = "medium"  # tiny/small/medium/large — drives wait_for_server timeouts
    model_size_gb: float = 0.0  # model file size in GB — used for VRAM budget calculations

    # Session
    http: requests.Session = field(default_factory=requests.Session)
    active_server_proc: Optional[subprocess.Popen] = None
    quality_baseline: Optional[dict] = None

    # VRAM cache (populated once after first server warmup)
    vram_total_mb: Optional[float] = None

    # Server flags
    no_jinja: bool = False  # --no-jinja escape hatch for models that crash on Jinja parsing

    # llama-bench integration
    bench_path: Optional[Path] = None  # path to llama-bench binary (None = HTTP-only mode)

    # Flash attention compatibility (detected from server stderr)
    _flash_attn_disabled_for_kv: bool = False

    # NUMA detection
    numa_nodes: int = 1

    # KL-divergence baseline cache
    kl_baseline_cache: Optional[list] = None


# ============================================================
# Context Factory
# ============================================================

def create_context(config: dict) -> AppContext:
    """Build AppContext from a loaded config dict."""
    arch = config["architecture"]
    hw = config["hardware"]
    results_dir = Path(config["results_dir"])
    dgl = min(hw.get("default_gpu_layers", 99), hw["max_gpu_layers"])

    return AppContext(
        server_path=Path(config["server"]),
        model_path=Path(config["model"]),
        chat_template_path=Path(config["chat_template"]) if config.get("chat_template") else Path(""),
        results_dir=results_dir,
        lookup_cache_file=str(results_dir / "lookup-cache.bin"),
        optuna_db="sqlite:///" + str(results_dir / "optuna.db").replace("\\", "/"),
        port=config["port"],
        server_url="http://127.0.0.1:" + str(config["port"]),
        config=config,
        arch=arch,
        is_moe=arch["type"] == "moe",
        expert_override_key=arch.get("expert_override_key", ""),
        default_experts=arch.get("default_experts", 8),
        max_experts=arch.get("max_experts", 16),
        hw=hw,
        max_threads=hw["max_threads"],
        moe_sweep_max=hw["moe_sweep_max"],
        moe_sweep_center=hw["moe_sweep_center"],
        max_gpu_layers=hw["max_gpu_layers"],
        default_gpu_layers=dgl,
        naked_engine={"context": 4096, "mlock": True, "n_gpu_layers": dgl},
        fail_fast=config.get("fail_fast", False),
        skip_quality=config.get("skip_quality", False),
        no_jinja=config.get("no_jinja", False),
        debug=config.get("debug", False),
        bench_path=None if config.get("no_bench") else _find_llama_bench(config["server"]),
        numa_nodes=hw.get("numa_nodes", 1),
    )


# ============================================================
# Config Loading
# ============================================================

def _load_config():
    """Load config from CLI args, config.json, or defaults."""
    parser = argparse.ArgumentParser(description="llama-server Parameter Optimizer", add_help=False)
    parser.add_argument("--server", help="Path to llama-server executable")
    parser.add_argument("--model", help="Path to GGUF model file")
    parser.add_argument("--chat-template", help="Path to chat template file")
    parser.add_argument("--results-dir", help="Path to results directory")
    parser.add_argument("--port", type=int, help="Server port")
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument("--dense", action="store_true", help="Dense model (skip MoE phases)")
    parser.add_argument("--fail-fast", action="store_true", help="Exit immediately if baseline server fails to start")
    parser.add_argument("--skip-quality", action="store_true", help="Skip Quality/sampling phase")
    preset_group = parser.add_mutually_exclusive_group()
    preset_group.add_argument("--quick", action="store_const", const="quick", dest="preset",
                              help="Quick optimization: fewer trials per phase")
    preset_group.add_argument("--normal", action="store_const", const="normal", dest="preset",
                              help="Normal optimization: balanced trial counts (default)")
    preset_group.add_argument("--thorough", action="store_const", const="thorough", dest="preset",
                              help="Thorough optimization: more trials per phase")
    parser.set_defaults(preset="normal")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without executing")
    parser.add_argument("--debug", action="store_true", help="Show debug output")
    parser.add_argument("--batch", metavar="DIR", help="Optimize all GGUF models in directory")
    parser.add_argument("--interactive", action="store_true", help="Pause between phases for inspection")
    parser.add_argument("--timeout", type=int, metavar="MINUTES", default=0, help="Per-model timeout in minutes")
    parser.add_argument("--skip-existing", action="store_true", help="Skip models with existing results")
    parser.add_argument("--no-jinja", action="store_true", help="Disable Jinja template parsing")
    parser.add_argument("--no-bench", action="store_true", help="Disable llama-bench acceleration")
    parser.add_argument("--dashboard", action="store_true", help="Launch optuna-dashboard web UI")
    parser.add_argument("--pareto", action="store_true", help="Enable multi-objective Pareto optimization")
    parser.add_argument("--simulate-users", type=int, default=0, metavar="N",
                        help="Concurrent user load test with N simultaneous users")
    args, _ = parser.parse_known_args()

    config = copy.deepcopy(_DEFAULTS)

    # Layer 1: config.json file (deep merge for nested dicts)
    config_path = args.config or os.path.join(str(Path(__file__).resolve().parent / "results"), "optimizer-config.json")
    config["_config_path"] = config_path
    if os.path.exists(config_path):
        try:
            with open(config_path, encoding="utf-8") as f:
                file_config = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[!] Config file corrupted ({e}) — using defaults")
            file_config = {}
        for k, v in file_config.items():
            if isinstance(v, dict) and isinstance(config.get(k), dict):
                config[k].update(v)
            else:
                config[k] = v

    # Layer 2: CLI args override everything
    if args.server:
        config["server"] = args.server
    if args.model:
        config["model"] = args.model
    if args.chat_template:
        config["chat_template"] = args.chat_template
    if args.results_dir:
        config["results_dir"] = args.results_dir
    if args.port:
        config["port"] = args.port
    if args.dense:
        config["architecture"]["type"] = "dense"
    if args.fail_fast:
        config["fail_fast"] = True
    if args.skip_quality:
        config["skip_quality"] = True
    config["preset"] = args.preset
    config["dry_run"] = args.dry_run
    config["debug"] = args.debug
    if args.batch:
        config["batch_dir"] = args.batch
    config["interactive"] = args.interactive
    config["timeout_minutes"] = args.timeout
    config["skip_existing"] = args.skip_existing
    config["no_jinja"] = args.no_jinja
    config["no_bench"] = args.no_bench
    config["dashboard"] = args.dashboard
    config["pareto"] = args.pareto
    config["simulate_users"] = args.simulate_users

    # Layer 3: Auto-detect hardware if not explicitly set
    hw = config["hardware"]
    if hw["max_threads"] is None:
        hw["max_threads"] = os.cpu_count() or 16
    if "numa_nodes" not in hw:
        hw["numa_nodes"] = _detect_numa_nodes()
    if hw["moe_sweep_max"] is None:
        hw["moe_sweep_max"] = min(hw["max_threads"] * 2, 40)
    if hw["moe_sweep_center"] is None:
        hw["moe_sweep_center"] = hw["moe_sweep_max"] // 2
    if hw["max_gpu_layers"] is None:
        hw["max_gpu_layers"] = detect_model_layers(config.get("model", ""))
    if hw["max_gpu_layers"] is None:
        hw["max_gpu_layers"] = 99  # fallback

    return config


# ============================================================
# Module-level initialization
# ============================================================

_config = _load_config()
ctx = create_context(_config)
