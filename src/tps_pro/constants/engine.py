"""Engine defaults, batch sizes, GPU sweep parameters, and aiohttp detection."""

from __future__ import annotations

# Import-time probe: detect whether aiohttp is installed so that callers can
# fall back to synchronous HTTP without paying the import cost at every call
# site.  This is a deliberate import-time side-effect -- the flag is read-only
# after module load and does not depend on any runtime application state.
try:
    import aiohttp as _aiohttp  # noqa: F401

    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

# ============================================================
# GPU Offload Sweep
# ============================================================

GPU_SWEEP_MAX_POINTS = (
    16  # maximum number of NGL checkpoints when model fits entirely in VRAM
)
GPU_SWEEP_OOM_DEPTH = (
    8  # how many layers below the OOM boundary to sweep when boundary < max
)

# Minimum fraction for any single GPU in a tensor split.
# Splits with any element below this threshold are discarded as degenerate.
MIN_SPLIT_FRACTION = 0.01

# ============================================================
# Engine Defaults
# ============================================================

DEFAULT_MAX_GPU_LAYERS = 99  # "all layers on GPU" sentinel used when auto-detect fails
DEFAULT_CONTEXT_SIZE = 4096  # baseline context window (tokens)
DEFAULT_PORT = 8090  # default llama-server port
DEFAULT_TEMPERATURE = 0.4  # temperature for measurement / warmup requests
DEFAULT_EXPERTS = 8  # trained default active experts for MoE models
MAX_EXPERTS = 16  # maximum experts to sweep for MoE models
