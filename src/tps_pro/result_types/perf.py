"""Performance measurement result types (frozen dataclasses)."""

from __future__ import annotations

from dataclasses import dataclass

from .base import _DictAccessMixin

# ---------------------------------------------------------------------------
# Performance measurement types (measurement.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerfSample(_DictAccessMixin):
    """Single raw measurement run from measure_perf_once."""

    tps: float
    ttft: float
    prompt_tps: float
    total_ms: float
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None


@dataclass(frozen=True)
class PerfResult(_DictAccessMixin):
    """Aggregated performance result from measure_perf_adaptive.

    Extends PerfSample with CV-stabilization metadata, optional large-prompt
    data, and concurrent-load metrics.
    """

    tps: float
    ttft: float
    prompt_tps: float
    total_ms: float
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    # CV-stabilization metadata
    tps_std: float | None = None
    tps_cv: float | None = None
    n_runs: int | None = None
    # Large-prompt data (full-mode scoring)
    large_tps: float | None = None
    # Concurrent-load metrics (merged from measure_concurrent_load)
    concurrent_total_tps: float | None = None
    concurrent_avg_tps: float | None = None
    concurrent_avg_ttft: float | None = None
    concurrent_avg_wall_ms: float | None = None
    concurrent_max_wall_ms: float | None = None
    concurrent_success_rate: float | None = None
    concurrent_users: int | None = None
    # Quality factor (attached by some phases)
    quality_factor: float | None = None
    # Model load time captured from server stderr
    load_time_ms: float | None = None


@dataclass(frozen=True)
class ConcurrentLoadResult(_DictAccessMixin):
    """Aggregate metrics from measure_concurrent_load."""

    concurrent_total_tps: float
    concurrent_avg_tps: float
    concurrent_avg_ttft: float
    concurrent_avg_wall_ms: float
    concurrent_max_wall_ms: float
    concurrent_success_rate: float
    concurrent_users: int


@dataclass(frozen=True)
class TokenUncertaintyResult(_DictAccessMixin):
    """Token-level uncertainty measurement from measure_token_uncertainty."""

    uncertain_count: int
    tail_avg: float
    total_tokens: int


# ---------------------------------------------------------------------------
# Pareto multi-objective tuple (measurement.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParetoObjectives(_DictAccessMixin):
    """Frozen dataclass for multi-objective Pareto optimization values.

    All three objectives are maximized:
      - tps: generation tokens/sec
      - neg_vram: negative VRAM usage in MB (higher = less VRAM)
      - quality_factor: 0.0-1.0 quality gate score

    Supports tuple unpacking and indexing for backward compatibility
    with callers that treat this as a tuple (e.g., Optuna multi-objective return).
    """

    tps: float
    neg_vram: float
    quality_factor: float

    @property
    def _as_tuple(self) -> tuple[float, float, float]:
        """Return the three objectives as a tuple."""
        return (self.tps, self.neg_vram, self.quality_factor)

    def __iter__(self):
        """Support tuple unpacking: tps, neg_vram, qf = objectives."""
        return iter(self._as_tuple)

    def __getitem__(self, index: int):  # type: ignore[override]
        """Support positional indexing: objectives[0], objectives[1], objectives[2]."""
        return self._as_tuple[index]

    def __len__(self) -> int:
        return len(self._as_tuple)


# ---------------------------------------------------------------------------
# llama-bench result (engine/bench.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchResult(_DictAccessMixin):
    """Parsed llama-bench CSV output.

    Fields mirror the perf-dict returned by _parse_bench_csv:
      - tps: generation tokens/sec (avg_ts from gen rows)
      - prompt_tps: prompt-processing tokens/sec (avg_ts from pp rows)
      - ttft: time-to-first-token in ms (pp_total_ns / 1e6)
      - total_ms: total wall time in ms (pp + gen)
    """

    tps: float
    prompt_tps: float
    ttft: float
    total_ms: float


# ---------------------------------------------------------------------------
# Concurrent single-user result (inner result from _single_request)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConcurrentUserResult(_DictAccessMixin):
    """Result of a single user request during concurrent load testing."""

    user_id: int
    success: bool
    tps: float = 0.0
    ttft: float = 0.0
    prompt_tps: float = 0.0
    wall_time: float = 0.0
    error: str | None = None
