"""Typed dataclasses replacing dict[str, Any] result types throughout the codebase.

Type Strategies
===============
This package uses three deliberate type strategies:

1. **TypedDict for configuration shapes** (EngineConfig, ArchConfig, HardwareConfig,
   NakedEngineConfig, PhaseReturnDict, SamplingParams, GpuInfo, etc.)
   These are structural type annotations over plain dicts.  They use ``total=False``
   because configs are built incrementally by different pipeline phases -- no single
   call site populates every field.  ``.get(key, default)`` is the correct access
   pattern and mirrors the ``total=False`` contract.

2. **Frozen dataclass + _DictAccessMixin for measurement results** (PerfResult,
   PerfSample, ConcurrentLoadResult, TokenUncertaintyResult, PhaseResult, etc.)
   These are immutable value objects created once and never mutated.  Freezing
   prevents accidental field changes after construction, which is critical for
   measurement data that flows through scoring, caching, and serialization.
   The _DictAccessMixin provides backward-compatible dict-style access (``obj["field"]``
   and ``obj.get()``) for callers not yet migrated to attribute access.

3. **Rationale**: TypedDicts are used where the runtime object must be a plain dict
   (e.g., JSON-serializable configs passed to subprocess CLI builders).  Frozen
   dataclasses are used where immutability and attribute access are more important
   than dict compatibility, and where the object lifecycle is create-once-read-many.

Each frozen dataclass provides:
  - from_dict(cls, data) for backward-compatible construction from existing dicts
  - to_dict(self) for serialization via dataclasses.asdict
  - dict-style access via [] and .get() for backward compatibility with callers
    that haven't been migrated to attribute access yet
"""

from .engine import (
    ArchConfig,
    EngineConfig,
    GpuInfo,
    HardwareConfig,
    KilledProcessInfo,
    NakedEngineConfig,
    SamplingParams,
    ServerProcess,
)
from .perf import (
    BenchResult,
    ConcurrentLoadResult,
    ConcurrentUserResult,
    ParetoObjectives,
    PerfResult,
    PerfSample,
    TokenUncertaintyResult,
)
from .phase import (
    PhaseFunction,
    PhaseResult,
    PhaseReturnDict,
    TrialSummary,
)
from .quality import (
    KLResult,
    NIAHPhaseResult,
    NIAHResult,
    NIAHTestResult,
    PPLResult,
    QualityResult,
    QualityTaskResult,
)

__all__ = [
    # engine
    "ArchConfig",
    "EngineConfig",
    "GpuInfo",
    "HardwareConfig",
    "KilledProcessInfo",
    "NakedEngineConfig",
    "SamplingParams",
    "ServerProcess",
    # perf
    "BenchResult",
    "ConcurrentLoadResult",
    "ConcurrentUserResult",
    "ParetoObjectives",
    "PerfResult",
    "PerfSample",
    "TokenUncertaintyResult",
    # phase
    "PhaseFunction",
    "PhaseResult",
    "PhaseReturnDict",
    "TrialSummary",
    # quality
    "KLResult",
    "NIAHPhaseResult",
    "NIAHResult",
    "NIAHTestResult",
    "PPLResult",
    "QualityResult",
    "QualityTaskResult",
]
