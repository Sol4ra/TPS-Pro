"""Display helpers — progress bars and formatted trial/importance output.

Module-level state
------------------
``pbar_state`` is the only module-level mutable object.  It is a
``types.SimpleNamespace(current=None)`` that tracks the currently active
tqdm progress bar.  Using a namespace instead of a bare global lets
functions assign to ``pbar_state.current`` without the ``global`` keyword.

This state exists at module level because the progress bar is shared across
``create_phase_pbar``, ``close_phase_pbar``, ``print_trial_result``, and the
``ProgressBarUpdateCallback`` in ``search_callbacks``.  It is reset to ``None``
whenever a phase bar is closed, so there is no stale cross-phase leakage.

Public API:
- create_phase_pbar — create a tqdm progress bar writing to raw stdout.
- close_phase_pbar — close the active progress bar.
- print_trial_result — formatted trial result line (tqdm-aware).
- print_param_importance — ranked fANOVA importance table.
- pbar_state — namespace tracking the active progress bar.

Error strategy (see errors.py for full documentation):
    - print_param_importance(): catches RuntimeError/ValueError from
      fANOVA and falls back to MeanDecreaseImpurity.  If both fail,
      returns empty dict with a warning.  Importance analysis is
      informational and must never abort a phase.
"""

from __future__ import annotations

import logging
import types

import optuna

from ..result_types import PerfResult, PerfSample
from ..state import AppContext

logger = logging.getLogger(__name__)

__all__ = [
    "create_phase_pbar",
    "close_phase_pbar",
    "pbar_state",
    "print_trial_result",
    "print_param_importance",
]

# Module-level progress bar state — encapsulated in a namespace so functions
# can mutate pbar_state.current without the `global` keyword.
# See module docstring for rationale.
pbar_state = types.SimpleNamespace(current=None)


def create_phase_pbar(total: int, desc: str = "") -> types.SimpleNamespace | None:
    """Create a phase counter (no progress bar). Stores desc and total for formatting.

    Returns a SimpleNamespace tracker, or None.
    """
    tracker = types.SimpleNamespace(
        total=total,
        desc=desc,
        count=0,
        current=None,
    )
    pbar_state.current = tracker
    return tracker


def close_phase_pbar() -> None:
    """Close the active phase tracker."""
    pbar_state.current = None


def print_trial_result(  # noqa: PLR0913
    ctx: AppContext,
    trial_num: int,
    total_trials: int,
    tps: float,
    perf: PerfResult | PerfSample | None,
    params_short: str,
    best_score: float,
    final_score: float | None = None,
) -> float:
    """Print a formatted trial result line. Returns new best_score.

    Format: PhaseName  N/Total:  XX.X t/s | pp: XXX | TTFT: XXms
            | score: XXX.X | params ** BEST **
    """
    from ..measurement import compute_score

    score = final_score if final_score is not None else compute_score(perf)
    is_new_best = score > best_score
    if is_new_best:
        best_score = score

    prompt_tps = perf.prompt_tps if perf else 0
    ttft = perf.ttft if perf else 0
    marker = " ** BEST **" if is_new_best else ""

    # Get phase name from tracker
    desc = ""
    if pbar_state.current is not None:
        desc = pbar_state.current.desc
        pbar_state.current.count = trial_num + 1

    # Build counter prefix
    counter = f"{trial_num + 1:>3d}/{total_trials}"

    metrics = (
        f"  {desc} {counter}: {tps:.1f} t/s | pp: {prompt_tps:.0f} | "
        f"TTFT: {ttft:.0f}ms | score: {score:.1f}{marker}"
    )
    logger.info(metrics)
    logger.info("    %s", params_short)
    logger.info("")
    return best_score


def print_param_importance(study: optuna.Study) -> dict[str, float]:
    """Print a ranked table of parameter importances using fANOVA."""
    # Filter out failed trials (score=0) so fANOVA has clean data
    from ._study import get_positive_completed_trials

    completed = get_positive_completed_trials(study)
    if len(completed) < 10:  # noqa: PLR2004
        logger.info(
            "Only %d successful trials — need 10+ for importance analysis",
            len(completed),
        )
        return {}

    try:
        importances = optuna.importance.get_param_importances(study)
    except (RuntimeError, ValueError) as e:
        # Fallback to mean decrease impurity if fANOVA fails
        try:
            from optuna.importance import MeanDecreaseImpurityImportanceEvaluator

            importances = optuna.importance.get_param_importances(
                study, evaluator=MeanDecreaseImpurityImportanceEvaluator()
            )
        except (RuntimeError, ValueError, ImportError):
            logger.warning("Could not compute parameter importance: %s", e)
            return {}

    if not importances or len(importances) <= 1:
        return importances

    logger.info("")
    logger.info("  Parameter Importance:")
    max_bar = 20
    for param, importance in importances.items():
        pct = importance * 100
        bar_len = (
            int(importance / max(importances.values()) * max_bar)
            if max(importances.values()) > 0
            else 0
        )
        bar = "\u2588" * bar_len
        logger.info("    %-24s %5.1f%%  %s", param, pct, bar)

    return importances
