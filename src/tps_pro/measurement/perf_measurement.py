"""Single-run and adaptive performance measurement.

Handles HTTP communication with the inference server, warmup gating,
CV-based stabilization, and sample aggregation.

Error strategy (see errors.py for full documentation):
    - measure_perf_once(): returns None on HTTP failure (logged at warning).
      The adaptive measurement loop collects multiple samples and tolerates
      individual failures gracefully.
    - measure_perf_adaptive(): never raises.  Returns empty PerfResult on
      total failure (all zeros, was_promoted=False).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import requests

from ..constants import (
    ADAPTIVE_THRESHOLD,
    ADAPTIVE_WARMUP_RUNS,
    CV_MAX_RUNS,
    CV_MIN_RUNS,
    CV_TARGET,
    DEFAULT_TEMPERATURE,
    HTTP_OK,
    TPS_TEST_PROMPT,
    WARMUP_REQUEST_TIMEOUT,
)
from ..result_types import (
    EngineConfig,
    PerfResult,
    PerfSample,
)
from ..state import AppContext, get_config
from .scoring import compute_score

_MIN_SAMPLES = 2

logger = logging.getLogger(__name__)

__all__ = [
    "measure_perf_once",
    "measure_perf_adaptive",
    "_aggregate_samples",
    "_attach_variance",
    "_median_by_score",
    "_to_perf_result",
]


def _tps_stats(samples: list[PerfSample]) -> tuple[float, float, float]:
    """Compute mean, std, and CV for TPS values from a list of samples.

    Returns:
        (mean, std, cv) tuple. Returns (0.0, 0.0, 0.0) if samples is empty.
    """
    if not samples:
        return 0.0, 0.0, 0.0
    tps_values = [x.tps for x in samples]
    mean_tps = sum(tps_values) / len(tps_values)
    if mean_tps <= 0:
        return mean_tps, 0.0, 0.0
    std_tps = (sum((v - mean_tps) ** 2 for v in tps_values) / len(tps_values)) ** 0.5
    cv = std_tps / mean_tps
    return mean_tps, std_tps, cv


def _to_perf_result(obj: PerfResult | PerfSample | dict[str, Any]) -> PerfResult:
    """Convert any perf-like object (dict, PerfSample, PerfResult) to PerfResult."""
    if isinstance(obj, PerfResult):
        return obj
    if isinstance(obj, dict):
        return PerfResult.from_dict(obj)
    # PerfSample or other dataclass with to_dict
    return PerfResult.from_dict(obj.to_dict())


def _capture_vram(ctx: AppContext) -> tuple[float | None, float | None]:
    """Capture current VRAM usage if GPU info is available.

    Returns:
        (vram_used_mb, vram_total_mb) or (None, None) if unavailable.
    """
    from ..hardware import get_vram_used_mb

    if not ctx.vram_total_mb:
        return None, None
    vram_used = get_vram_used_mb()
    if vram_used is None:
        return None, None
    return vram_used, ctx.vram_total_mb


def _parse_perf_response(data: dict[str, Any], ctx: AppContext) -> PerfSample | None:
    """Extract PerfSample from a successful chat completions response."""
    timings = data.get("timings", {})
    tps = timings.get("predicted_per_second", 0)
    if tps <= 0:
        return None

    vram_used_mb, vram_total_mb = _capture_vram(ctx)
    return PerfSample(
        tps=tps,
        ttft=timings.get("prompt_ms", 0),
        prompt_tps=timings.get("prompt_per_second", 0),
        total_ms=timings.get("prompt_ms", 0) + timings.get("predicted_ms", 0),
        vram_used_mb=vram_used_mb,
        vram_total_mb=vram_total_mb,
    )


def measure_perf_once(
    ctx: AppContext,
    n_predict: int = 50,
    spec_params: EngineConfig | None = None,
    prompt: str | None = None,
) -> PerfSample | None:
    """Single measurement run. Returns PerfSample or None on failure.

    Uses /v1/chat/completions so the server applies its --chat-template
    automatically -- works with any model, not just ChatML.
    """
    payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt or TPS_TEST_PROMPT}],
        "max_tokens": n_predict,
        "temperature": DEFAULT_TEMPERATURE,
    }
    if spec_params:
        payload["speculative"] = spec_params
    try:
        prompt_len = len(prompt or TPS_TEST_PROMPT)
        est_tokens = prompt_len // 4
        timeout = max(
            WARMUP_REQUEST_TIMEOUT, WARMUP_REQUEST_TIMEOUT + est_tokens // 500
        )
        r = ctx.http.post(
            f"{ctx.server_url}/v1/chat/completions", json=payload, timeout=timeout
        )
        if r.status_code == HTTP_OK:
            return _parse_perf_response(r.json(), ctx)
    except (requests.RequestException, requests.Timeout, ValueError, KeyError) as e:
        logger.warning("Request failed: %s", e)
    return None


def _aggregate_samples(samples: list[PerfSample]) -> PerfResult:
    """Return the median run as PerfResult from PerfSample list.

    Sorting by composite score and returning the middle sample avoids synthesizing
    an impossible combination of fast TTFT from one run and fast TPS from another.
    """
    if not samples:
        return PerfResult(tps=0.0, ttft=0.0, prompt_tps=0.0, total_ms=0.0)
    if len(samples) == 1:
        return _to_perf_result(samples[0])
    # Sort by composite score, return the middle (median) run as a whole
    ranked = sorted(samples, key=lambda s: compute_score(s))
    return _to_perf_result(ranked[len(ranked) // 2])


def _run_warmup(
    ctx: AppContext,
    n_predict: int,
    spec_params: EngineConfig | None,
    prompt: str | None,
) -> list[PerfSample]:
    """Run warmup passes and return collected samples."""
    results: list[PerfSample] = []
    for _ in range(ADAPTIVE_WARMUP_RUNS):
        run = measure_perf_once(
            ctx, n_predict=n_predict, spec_params=spec_params, prompt=prompt
        )
        if run is not None:
            results.append(run)
    return results


def _median_by_score(samples: list[PerfSample]) -> PerfSample:
    """Return the median sample when sorted by composite score."""
    ranked = sorted(samples, key=lambda s: compute_score(s))
    return ranked[len(ranked) // 2]


def _cv_stabilize(
    ctx: AppContext,
    seed_samples: list[PerfSample],
    n_predict: int,
    spec_params: EngineConfig | None,
    prompt: str | None,
) -> list[PerfSample]:
    """Run additional samples until TPS CV is below threshold."""
    samples = list(seed_samples)
    extra_needed = CV_MAX_RUNS - len(samples)
    for _ in range(extra_needed):
        s = measure_perf_once(
            ctx, n_predict=n_predict, spec_params=spec_params, prompt=prompt
        )
        if s:
            samples.append(s)
        if len(samples) >= CV_MIN_RUNS:
            _mean, _std, cv = _tps_stats(samples)
            if _mean > 0 and cv <= CV_TARGET:
                break
    return samples


def _attach_variance(result: PerfResult, samples: list[PerfSample]) -> PerfResult:
    """Attach TPS variance stats to the result for noise-aware GP."""
    if len(samples) < _MIN_SAMPLES:
        return result
    _mean, tps_std, tps_cv = _tps_stats(samples)
    return dataclasses.replace(
        result, tps_std=tps_std, tps_cv=tps_cv, n_runs=len(samples)
    )


def _attach_concurrent_load(
    result: PerfResult, ctx: AppContext, n_predict: int
) -> PerfResult:
    """Run concurrent load test if configured and attach results."""
    from .concurrent import measure_concurrent_load

    n_users = get_config("simulate_users", 0)
    if n_users <= 0:
        return result
    load = measure_concurrent_load(ctx, n_users=n_users, n_predict=n_predict)
    if load is None:
        return result
    return dataclasses.replace(
        result,
        concurrent_total_tps=load.concurrent_total_tps,
        concurrent_avg_tps=load.concurrent_avg_tps,
        concurrent_avg_ttft=load.concurrent_avg_ttft,
        concurrent_avg_wall_ms=load.concurrent_avg_wall_ms,
        concurrent_max_wall_ms=load.concurrent_max_wall_ms,
        concurrent_success_rate=load.concurrent_success_rate,
        concurrent_users=load.concurrent_users,
    )


def measure_perf_adaptive(  # noqa: PLR0913
    ctx: AppContext,
    best_score: float = 0.0,
    n_predict: int = 50,
    spec_params: EngineConfig | None = None,
    prompt: str | None = None,
    runs: int | None = None,
) -> tuple[PerfResult, bool]:
    """Unified measurement: fixed-run or adaptive CV-based stability.

    When ``runs`` is set (not None):
      Simple fixed-run mode.  Do exactly N runs with ``measure_perf_once()``,
      aggregate with ``_aggregate_samples()``.  Always returns ``(result, True)``.

    When ``runs`` is None (default):
      Full adaptive mode:
        1. Run ADAPTIVE_WARMUP_RUNS quick runs as a gate (median score decides).
        2. If competitive (>= 50% of best), run CV-stabilized measurement.
        3. Runs 3-5 times until CV <= 5% (or max reached).

    ``prompt`` is forwarded to every ``measure_perf_once()`` call (warmup and
    CV runs alike) so callers can supply a workload-specific context.

    Returns ``(PerfResult, was_promoted)`` -- *was_promoted* is True when the
    config received stable / full measurement.

    Raises:
        This function does not raise exceptions.  On measurement failure it
        returns an empty ``PerfResult`` (all zeros) with ``was_promoted=False``.
    """
    _empty_result = PerfResult(tps=0.0, ttft=0.0, prompt_tps=0.0, total_ms=0.0)

    # ------------------------------------------------------------------
    # Fixed-run mode (replaces the old measure_perf())
    # ------------------------------------------------------------------
    if runs is not None:
        samples: list[PerfSample] = []
        for _ in range(runs):
            s = measure_perf_once(
                ctx, n_predict=n_predict, spec_params=spec_params, prompt=prompt
            )
            if s:
                samples.append(s)
        result = _aggregate_samples(samples)

        return result, True

    # ------------------------------------------------------------------
    # Adaptive mode (warmup gate -> CV stabilisation -> large prompt)
    # ------------------------------------------------------------------
    warmup_results = _run_warmup(ctx, n_predict, spec_params, prompt)
    if not warmup_results:
        return _empty_result, False

    best_warmup = _median_by_score(warmup_results)
    quick_score = compute_score(best_warmup)

    if best_score > 0 and quick_score < best_score * ADAPTIVE_THRESHOLD:
        return _to_perf_result(best_warmup), False

    samples = _cv_stabilize(ctx, warmup_results, n_predict, spec_params, prompt)
    if not samples:
        return _to_perf_result(best_warmup), False

    result = _aggregate_samples(samples)
    result = _attach_variance(result, samples)
    result = _attach_concurrent_load(result, ctx, n_predict)
    return result, True
