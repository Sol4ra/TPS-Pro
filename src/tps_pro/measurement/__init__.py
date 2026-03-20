"""Performance measurement, scoring, and concurrent load testing.

This package groups the three measurement-related modules:
    - scoring.py: compute_score, compute_pareto_objectives, _clamp_finite
    - perf_measurement.py: measure_perf_once, measure_perf_adaptive, _aggregate_samples
    - concurrent.py: measure_concurrent_load, measure_token_uncertainty

All public names are re-exported here so existing ``from .measurement import X``
and ``from ..measurement import X`` statements continue to work without changes.

Error strategy (see errors.py for full documentation):
    - measure_perf_once(): returns None on HTTP failure (logged at warning).
      The adaptive measurement loop collects multiple samples and tolerates
      individual failures gracefully.
    - measure_perf_adaptive(): never raises.  Returns empty PerfResult on
      total failure (all zeros, was_promoted=False).
    - measure_concurrent_load(): returns None on failure (logged at debug).
    - measure_token_uncertainty(): returns None on failure.  Individual
      prompt failures are logged at warning but do not abort the batch.
    - extract_pareto_front(): returns empty list on Optuna RuntimeError.
"""

from __future__ import annotations

# Re-export concurrent/uncertainty functions
from .concurrent import (  # noqa: F401
    measure_concurrent_load,
    measure_token_uncertainty,
)

# Re-export performance measurement functions
from .perf_measurement import (  # noqa: F401
    measure_perf_adaptive,
    measure_perf_once,
)

# Re-export scoring functions
from .scoring import (  # noqa: F401
    compute_pareto_objectives,
    compute_score,
    extract_pareto_front,
    get_best_trial,
    print_pareto_front,
)

__all__ = [
    "compute_score",
    "compute_pareto_objectives",
    "extract_pareto_front",
    "print_pareto_front",
    "get_best_trial",
    "measure_perf_adaptive",
    "measure_perf_once",
    "measure_concurrent_load",
    "measure_token_uncertainty",
]
