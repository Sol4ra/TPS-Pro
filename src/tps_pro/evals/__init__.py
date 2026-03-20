"""Quality evaluation: quality gate, KL-divergence, perplexity, NIAH, MCQ.

Error strategy:
    Evaluation functions propagate exceptions to the calling phase, which catches
    them via _PHASE_ERRORS.  Network errors (requests.RequestException) during
    server communication are caught locally and cause the evaluation to return a
    failure score (0 or None) rather than aborting the phase.  Parsing errors in
    quality/perplexity measurement are logged at DEBUG level and yield None so the
    trial is marked as failed without crashing the optimizer.
"""

from .integrity import phase_integrity_eval, phase_reasoning_eval  # noqa: F401
from .kl_divergence import (  # noqa: F401
    kl_quality_factor,
    measure_kl_divergence,
)
from .mcq import (  # noqa: F401
    measure_quality,
)
from .niah import (  # noqa: F401
    build_niah_prompt,
    niah_test,
    phase_niah,
    tokenize_count,
)
from .perplexity import measure_true_perplexity, ppl_quality_factor  # noqa: F401
from .quality_gate import measure_quality_gate  # noqa: F401

__all__ = [
    "measure_quality_gate",
    "measure_kl_divergence",
    "kl_quality_factor",
    "measure_true_perplexity",
    "ppl_quality_factor",
    "measure_quality",
    "tokenize_count",
    "build_niah_prompt",
    "niah_test",
    "phase_niah",
    "phase_reasoning_eval",
    "phase_integrity_eval",
]
