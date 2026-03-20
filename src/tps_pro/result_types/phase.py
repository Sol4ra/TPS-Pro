"""Phase function contracts and phase result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, cast

from .base import _DictAccessMixin

if TYPE_CHECKING:
    from ..state import AppContext


# ---------------------------------------------------------------------------
# Phase function return shape contract
# ---------------------------------------------------------------------------


class PhaseReturnDict(TypedDict, total=False):
    """Standard return shape for phase functions."""

    best_params: dict[str, Any]
    best_score: float
    study_name: str
    phase_name: str


class PhaseFunction(Protocol):
    """Structural type for phase entry-point functions.

    Documents the expected calling convention shared by all ``phase_*``
    functions in ``phases/``.  This is *not* enforced at runtime -- it
    exists purely for static type-checking and IDE support.

    All phase functions accept ``ctx`` as the first positional argument
    and may accept additional keyword arguments (``n_trials``,
    ``base_config``, etc.) depending on the phase.
    """

    def __call__(
        self,
        ctx: AppContext,
        *,
        n_trials: int = ...,
        base_config: dict[str, Any] | None = ...,
    ) -> PhaseReturnDict | None: ...


# ---------------------------------------------------------------------------
# Phase result types (search.py / trial_helpers.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialSummary(_DictAccessMixin):
    """Summary of a single completed Optuna trial.

    Stored inside PhaseResult.all_trials.
    """

    number: int
    tps: float | None = None
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseResult(_DictAccessMixin):
    """Saved phase output from print_phase_summary / save_phase_results."""

    phase: str
    baseline: dict = field(default_factory=dict)
    baseline_score: float = 0.0
    beat_baseline: bool = False
    best_tps: float = 0.0
    best_metrics: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)
    param_importance: dict[str, float] = field(default_factory=dict)
    duration_minutes: float = 0.0
    all_trials: list[TrialSummary] = field(default_factory=list)
    score_version: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> PhaseResult:
        raw_trials = data.get("all_trials", [])
        trials = [
            TrialSummary.from_dict(t) if isinstance(t, dict) else t for t in raw_trials
        ]
        return cls(
            phase=data.get("phase", ""),
            baseline=data.get("baseline", {}),
            baseline_score=data.get("baseline_score", 0.0),
            beat_baseline=data.get("beat_baseline", False),
            best_tps=data.get("best_tps", 0.0),
            best_metrics=data.get("best_metrics", {}),
            best_params=data.get("best_params", {}),
            param_importance=data.get("param_importance", {}),
            duration_minutes=data.get("duration_minutes", 0.0),
            all_trials=cast(list[TrialSummary], trials),
            score_version=data.get("score_version"),
        )
