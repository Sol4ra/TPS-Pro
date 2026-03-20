"""Search subpackage — study management, callbacks, and display helpers.

Re-exports the public API so existing ``from .search import ...`` imports
continue to work unchanged.
"""

from ._callbacks import (  # noqa: F401
    GPStoppingCallback,
    ProgressBarUpdateCallback,
    _encode_param,
    _expected_improvement,
    safe_best_value,
    trial_scalar_value,
)
from ._display import (  # noqa: F401
    close_phase_pbar,
    create_phase_pbar,
    pbar_state,
    print_param_importance,
    print_trial_result,
)
from ._study import (  # noqa: F401
    check_and_mark_duplicate_trial,
    clear_param_cache,
    ensure_results_dir,
    get_positive_completed_trials,
    load_phase_results,
    save_phase_results,
    setup_study,
    update_param_cache,
)

__all__ = [
    # _study
    "check_and_mark_duplicate_trial",
    "clear_param_cache",
    "ensure_results_dir",
    "load_phase_results",
    "save_phase_results",
    "setup_study",
    "update_param_cache",
    # _callbacks (public API only; private helpers like _encode_param,
    # _expected_improvement, safe_best_value are importable from
    # search._callbacks for tests/internals but not part of __all__)
    "GPStoppingCallback",
    "ProgressBarUpdateCallback",
    "trial_scalar_value",
    # _display
    "close_phase_pbar",
    "create_phase_pbar",
    "pbar_state",
    "print_param_importance",
    "print_trial_result",
]
