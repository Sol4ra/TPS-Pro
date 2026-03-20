"""Quality evaluation result types (frozen dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import _DictAccessMixin

# ---------------------------------------------------------------------------
# Quality evaluation types (evals/)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityTaskResult(_DictAccessMixin):
    """Result of a single MCQ evaluation task from _eval_single_task."""

    correct: bool
    logprob: float | None = None
    ttft_ms: float | None = None
    answer: str | None = None
    category: str | None = None



@dataclass(frozen=True)
class QualityResult(_DictAccessMixin):
    """Composite quality evaluation output from measure_quality.

    The score is a float 0-100 computed from correctness, confidence, and
    efficiency signals.  Individual task results are kept for diagnostics.
    """

    score: float
    task_results: list[QualityTaskResult] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> QualityResult:
        raw_tasks = data.get("task_results", [])
        tasks = [
            QualityTaskResult.from_dict(t) if isinstance(t, dict) else t
            for t in raw_tasks
        ]
        return cls(
            score=data.get("score", 0.0),
            task_results=tasks,
        )



@dataclass(frozen=True)
class NIAHTestResult(_DictAccessMixin):
    """Single needle-in-a-haystack probe at a given context size and depth."""

    context: int
    depth: float
    passed: bool
    needle_idx: int = 0
    error: str | None = None



@dataclass(frozen=True)
class NIAHResult(_DictAccessMixin):
    """Aggregate NIAH result for one KV cache type from niah_test."""

    kv_type: str
    results: list[NIAHTestResult] = field(default_factory=list)
    pass_rate: float = 0.0
    ppl: float | None = None
    oom: bool = False
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> NIAHResult:
        raw_results = data.get("results", [])
        results = [
            NIAHTestResult.from_dict(r) if isinstance(r, dict) else r
            for r in raw_results
        ]
        return cls(
            kv_type=data.get("kv_type", ""),
            results=results,
            pass_rate=data.get("pass_rate", 0.0),
            ppl=data.get("ppl"),
            oom=data.get("oom", False),
            error=data.get("error"),
        )



@dataclass(frozen=True)
class NIAHPhaseResult(_DictAccessMixin):
    """Full NIAH phase output from phase_niah, saved via save_phase_results."""

    phase: str = "niah"
    kv_results: list[NIAHResult] = field(default_factory=list)
    reference_kv: str = "f16"
    reference_pass_rate: float = 100.0
    reference_ppl: float | None = None
    score_version: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> NIAHPhaseResult:
        raw_kv = data.get("kv_results", [])
        kv_results = [
            NIAHResult.from_dict(r) if isinstance(r, dict) else r for r in raw_kv
        ]
        return cls(
            phase=data.get("phase", "niah"),
            kv_results=kv_results,
            reference_kv=data.get("reference_kv", "f16"),
            reference_pass_rate=data.get("reference_pass_rate", 100.0),
            reference_ppl=data.get("reference_ppl"),
            score_version=data.get("score_version"),
        )



@dataclass(frozen=True)
class KLResult(_DictAccessMixin):
    """KL-divergence measurement from measure_kl_divergence.

    Stores both the raw per-token logprob distributions (for reuse as a
    baseline cache) and the scalar divergence score.
    """

    distributions: list[dict[str, float] | None] | None = None
    kl_divergence: float | None = None

    def __iter__(self):
        """Support tuple unpacking: dists, kl_div = measure_kl_divergence(...)"""
        return iter((self.distributions, self.kl_divergence))



@dataclass(frozen=True)
class PPLResult(_DictAccessMixin):
    """Perplexity measurement from measure_true_perplexity.

    Wraps the scalar perplexity value with optional baseline comparison
    metadata used by ppl_quality_factor.
    """

    perplexity: float
    baseline_ppl: float | None = None
    quality_factor: float | None = None

    @classmethod
    def from_dict(cls, data: dict) -> PPLResult:
        """Override: perplexity defaults to inf (not 0.0) when missing."""
        return cls(
            perplexity=data.get("perplexity", float("inf")),
            baseline_ppl=data.get("baseline_ppl"),
            quality_factor=data.get("quality_factor"),
        )

