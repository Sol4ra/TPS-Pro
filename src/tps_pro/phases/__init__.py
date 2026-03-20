"""Optimization phases — re-exports all phase functions for backward compatibility.

Error strategy:
    Each phase function is called via pipeline._run_phase(), which catches
    _PHASE_ERRORS (OSError, ValueError, KeyError, TypeError, RuntimeError) so a
    single phase failure does not abort the entire pipeline.  Within a phase,
    individual trial failures return None (non-OOM) or raise BenchOOMError (OOM)
    to let Optuna prune the trial.  KeyboardInterrupt skips the current phase.
"""

from ._helpers import get_moe_config
from .core_engine import phase_core_engine
from .gpu_offload import phase_gpu_offload
from .kv_context_sweep import phase_kv_context_sweep
from .moe_experts import phase_experts
from .moe_sweep import phase_moe_sweep
from .quality import phase_quality
from .speculation import phase_speculation
from .tensor_split import phase_tensor_split
from .workload import phase_context_sweep, phase_workload_sim

__all__ = [
    # Helpers
    "get_moe_config",
    # Phase functions
    "phase_gpu_offload",
    "phase_experts",
    "phase_moe_sweep",
    "phase_core_engine",
    "phase_speculation",
    "phase_kv_context_sweep",
    "phase_workload_sim",
    "phase_context_sweep",
    "phase_quality",
    "phase_tensor_split",
]
