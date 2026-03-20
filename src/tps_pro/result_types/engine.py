"""Engine configuration, hardware shape contracts (TypedDicts), and ServerProcess.

ServerProcess lives here (in result_types, a leaf package) so that state.py
can import it without creating a circular dependency through the engine
subpackage:
    engine/__init__.py -> engine/server.py -> state.py -> engine (old cycle)
    result_types/engine.py <- state.py  (no cycle -- result_types is a leaf)
"""

from __future__ import annotations

import dataclasses
import subprocess
import threading
from typing import TypedDict

# ---------------------------------------------------------------------------
# Server process wrapper (used by engine/server.py and state.py)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ServerProcess:
    """Wraps subprocess.Popen with typed server state attributes.

    Frozen dataclass -- fields cannot be reassigned after construction.
    Mutable containers (stderr_lines, lock) are still mutated in-place
    (frozen only prevents field reassignment, not container mutation).
    Use dataclasses.replace() to produce a new instance with updated
    timing fields (load_time_ms, warmup_time_ms, boot_time_ms).
    """

    proc: subprocess.Popen
    stderr_lines: list = dataclasses.field(default_factory=list)
    load_time_ms: float | None = None
    warmup_time_ms: float | None = None
    boot_time_ms: float | None = None  # total: kill -> load -> warmup -> ready
    lock: threading.Lock = dataclasses.field(
        default_factory=threading.Lock, compare=False, hash=False
    )

# ---------------------------------------------------------------------------
# GPU and hardware detection shape contracts (hardware.py)
# ---------------------------------------------------------------------------


class GpuInfo(TypedDict):
    """Shape contract for a single GPU info dict from detect_gpus()."""

    index: int
    name: str
    vram_total_gb: float
    vram_free_gb: float


class KilledProcessInfo(TypedDict):
    """Shape contract for a killed process entry from kill_competing_processes()."""

    pid: int
    name: str
    gpu_mb: int


class SamplingParams(TypedDict, total=False):
    """Shape contract for sampling parameter dicts passed to measure_quality.

    All keys are optional since different callers populate different subsets.
    """

    temp: float
    top_p: float
    top_k: int
    min_p: float
    repeat_penalty: float
    mirostat: int
    mirostat_tau: float
    mirostat_eta: float


# ---------------------------------------------------------------------------
# Architecture and hardware configuration shape contracts (state.py)
# ---------------------------------------------------------------------------


class ArchConfig(TypedDict, total=False):
    """Shape contract for the architecture config dict in AppContext.

    Describes model architecture properties used to decide which
    optimization phases to run (MoE vs dense) and their parameter ranges.
    """

    type: str  # "dense" or "moe"
    expert_override_key: str
    default_experts: int
    max_experts: int


class HardwareConfig(TypedDict, total=False):
    """Shape contract for the hardware config dict in AppContext.

    Auto-detected or user-specified hardware capabilities that constrain
    the search space for thread counts, GPU layers, and MoE parameters.
    """

    max_threads: int
    moe_sweep_max: int
    moe_sweep_center: int
    max_gpu_layers: int
    default_gpu_layers: int
    numa_nodes: int


class NakedEngineConfig(TypedDict, total=False):
    """Shape contract for the naked_engine baseline config in AppContext.

    "Naked engine" means the llama-server launched with only the bare
    minimum flags (model path, GPU layers, context size, mlock) and no
    optimization parameters applied.  It serves as the untuned baseline
    that every phase measures against -- the engine "stripped naked" of
    all optional tuning knobs.

    Minimal server configuration used as the starting point before any
    optimization phases layer on their discovered parameters.
    """

    context: int
    mlock: bool
    n_gpu_layers: int


# ---------------------------------------------------------------------------
# Engine configuration shape contract (engine/server.py, engine/bench.py)
# ---------------------------------------------------------------------------


class EngineConfig(TypedDict, total=False):
    """Shape contract for the engine_config dict passed through server/bench functions.

    All keys are optional (``total=False``) because configs are built incrementally
    by each pipeline phase -- no single call site populates every field.  Using
    ``.get(key, default)`` on an ``EngineConfig`` dict is the *correct* access
    pattern: it mirrors the ``total=False`` contract and handles absent keys safely.

    At runtime these are plain ``dict`` instances (TypedDict is structural, not
    nominal), so ``.get()`` behaves exactly like ``dict.get()`` and does **not**
    indicate a trust violation.

    Type-safety note: ``total=False`` TypedDicts have a known mypy/pyright limitation
    where ``.get()`` returns ``value_type | None`` but type checkers sometimes widen
    to ``Any``.  This is a Python typing ecosystem limitation (python/mypy#7981), not
    a bug in this codebase.
    """

    # Core server flags (_add_base_args)
    n_gpu_layers: int
    context: int
    parallel: int
    warmup: bool
    cache_prompt: bool
    fit: bool

    # Numeric flag pairs (_add_numeric_flag_pairs)
    batch_size: int
    ubatch_size: int
    threads: int
    threads_batch: int
    n_cpu_moe: int
    expert_used_count: int
    poll: int
    poll_batch: int
    prio: int
    prio_batch: int

    # KV cache args (_add_kv_cache_args)
    kv_cache_type: str
    cache_type_k: str
    cache_type_v: str
    flash_attn: str | bool | int
    n_predict: int
    temp: float
    model_draft: str
    cache_reuse: int

    # Speculation args (_add_spec_args)
    spec_type: str
    spec_ngram_n: int
    spec_ngram_m: int
    spec_ngram_min_hits: int
    draft_max: int
    draft_min: int
    draft_p_min: float
    cpu_strict: int
    cpu_strict_batch: int

    # Boolean flags (_add_bool_flags)
    swa_full: bool
    repack: bool
    op_offload: bool
    kv_unified: bool
    mlock: bool
    no_mmap: bool
    kv_offload: bool
    no_host: bool
    direct_io: bool
    cont_batching: bool
    backend_sampling: bool
    context_shift: bool

    # Extended args (_add_extended_args)
    ctx_checkpoints: int
    checkpoint_every_n: int
    cache_ram: str
    threads_http: int
    lookup_cache_dynamic: str
    numa: str
    cpu_moe: bool
    override_tensor: str
    tensor_split: str

    # Environment variable flags (start_server)
    cuda_graph_opt: bool
