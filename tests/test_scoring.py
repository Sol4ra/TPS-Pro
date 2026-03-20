"""Tests for scoring and measurement functions from measurement.py.

measurement.py imports from state.py, but this is safe -- state.py only
creates empty sentinel objects at module level. CLI arg parsing only runs
when initialize() is explicitly called.
"""

import math

import pytest

from tps_pro.constants import SCORE_PP_BASELINE, TTFT_BASELINE_MS
from tps_pro.measurement import (
    compute_pareto_objectives,
    compute_score,
)
from tps_pro.measurement.perf_measurement import _aggregate_samples
from tps_pro.result_types import PerfResult, PerfSample

pytestmark = pytest.mark.unit


def _p(
    tps=50.0,
    prompt_tps=SCORE_PP_BASELINE,
    ttft=TTFT_BASELINE_MS,
    total_ms=1000.0,
    **kw,
) -> PerfResult:
    """Shorthand for building a PerfResult with sensible defaults."""
    return PerfResult(
        tps=tps, prompt_tps=prompt_tps, ttft=ttft, total_ms=total_ms, **kw
    )


def _s(
    tps=50.0, prompt_tps=SCORE_PP_BASELINE, ttft=TTFT_BASELINE_MS, total_ms=1000.0
) -> PerfSample:
    """Shorthand for building a PerfSample."""
    return PerfSample(tps=tps, prompt_tps=prompt_tps, ttft=ttft, total_ms=total_ms)


# ===================================================================
# compute_score
# ===================================================================


def test_zero_tps_returns_zero():
    assert compute_score(_p(tps=0)) == 0.0


def test_negative_tps_returns_zero():
    assert compute_score(_p(tps=-10)) == 0.0


def test_lightweight_mode_basic():
    """Lightweight mode: gen_tps=50, prompt_tps=PP_BASELINE, ttft=TTFT_BASELINE.

    pp_factor = 1.0, ttft_factor = 1.0
    score = 50 * (0.60 + 0.25*1.0 + 0.15*1.0) = 50 * 1.0 = 50.0
    """
    assert compute_score(_p()) == pytest.approx(50.0, abs=0.5)


def test_lightweight_mode_high_pp():
    """Higher prompt_tps should boost the score."""
    perf = _p(prompt_tps=SCORE_PP_BASELINE * 2)
    # pp_factor = 2.0 -> score = 50 * (0.60 + 0.25*2.0 + 0.15*1.0) = 50 * 1.25 = 62.5
    assert compute_score(perf) == pytest.approx(62.5, abs=0.5)


def test_lightweight_mode_fast_ttft():
    """Faster TTFT should boost the score."""
    perf = _p(ttft=TTFT_BASELINE_MS / 2)
    # ttft_factor = 2.0 -> score = 50 * (0.60 + 0.25*1.0 + 0.15*2.0) = 50 * 1.15 = 57.5
    assert compute_score(perf) == pytest.approx(57.5, abs=0.5)


def test_lightweight_pp_factor_capped_at_3():
    """pp_factor is capped at 3.0."""
    perf = _p(prompt_tps=SCORE_PP_BASELINE * 30)
    # pp_factor = min(30, 3.0) = 3.0
    # score = 50 * (0.60 + 0.25*3.0 + 0.15*1.0) = 50 * 1.50 = 75.0
    assert compute_score(perf) == pytest.approx(75.0, abs=0.5)


def test_full_mode_basic():
    """Full mode: gen_tps=50, large_tps=40, prompt_tps=PP_BASELINE, ttft=TTFT_BASELINE.

    pp_norm = 1.0, ttft_norm = 1.0
    score = 50*0.35 + 40*0.25 + 1.0*50*0.15 + 1.0*50*0.15 = 42.5
    No VRAM data -> add gen_tps * 0.10 = 5.0 -> Total = 47.5
    """
    perf = _p(large_tps=40)
    assert compute_score(perf) == pytest.approx(47.5, abs=0.5)


def test_full_mode_with_vram():
    """Full mode with VRAM data.

    Base = 42.5, VRAM efficiency = 0.5, vram_bonus = 0.5 * 50 * 0.10 = 2.5
    Total = 45.0
    """
    perf = _p(large_tps=40)
    assert compute_score(perf, vram_used_mb=4000, vram_total_mb=8000) == pytest.approx(
        45.0, abs=0.5
    )


def test_lightweight_vram_efficiency_bonus():
    """Lightweight mode with VRAM data gets up to 5% boost.

    Base = 50.0, utilization = 0.5, headroom_bonus = 0.05
    Final = 50.0 * 1.05 = 52.5
    """
    assert compute_score(_p(), vram_used_mb=4000, vram_total_mb=8000) == pytest.approx(
        52.5, abs=0.5
    )


def test_lightweight_vram_high_utilization():
    """High VRAM utilization gives smaller bonus.

    utilization = 7000/8000 = 0.875, headroom_bonus = 0.0125
    score = 50.0 * 1.0125 = 50.625
    """
    assert compute_score(_p(), vram_used_mb=7000, vram_total_mb=8000) == pytest.approx(
        50.625, abs=0.05
    )


def test_concurrent_load_bonus():
    """Concurrent load bonus scales score by (0.85 + 0.15 * efficiency).

    Base = 50.0, efficiency = 150/(50*4) = 0.75
    score *= 0.9625 -> 48.125
    """
    perf = _p(concurrent_total_tps=150, concurrent_users=4)
    assert compute_score(perf) == pytest.approx(48.125, abs=0.05)


def test_concurrent_load_perfect_scaling():
    """Perfect scaling (efficiency=1.0) gives full 15% bonus.

    score *= 1.0 -> 50.0
    """
    perf = _p(concurrent_total_tps=200, concurrent_users=4)
    assert compute_score(perf) == pytest.approx(50.0, abs=0.5)


def test_nan_tps_returns_zero():
    """NaN gen_tps: final isfinite check catches the NaN score."""
    assert compute_score(_p(tps=float("nan"))) == 0.0


def test_inf_prompt_tps_clamped():
    """Inf prompt_tps is clamped to 50000 -> pp_factor capped at 3.0.

    score = 50 * (0.60 + 0.25*3.0 + 0.15*1.0) = 75.0
    """
    assert compute_score(_p(prompt_tps=float("inf"))) == pytest.approx(75.0, abs=0.5)


def test_inf_ttft_clamped():
    """Inf ttft is clamped to 1.0ms -> ttft_factor capped at 3.0.

    score = 50 * (0.60 + 0.25*1.0 + 0.15*3.0) = 65.0
    """
    assert compute_score(_p(ttft=float("inf"))) == pytest.approx(65.0, abs=0.5)


def test_nan_prompt_tps_treated_as_50000():
    """NaN prompt_tps is treated as 50000 (same as inf case)."""
    assert compute_score(_p(prompt_tps=float("nan"))) == pytest.approx(75.0, abs=0.5)


def test_result_is_always_finite():
    """Score must always be finite (not inf, not NaN)."""
    test_cases = [
        _p(tps=50, prompt_tps=0, ttft=0.001),
        _p(tps=50, prompt_tps=float("inf")),
        _p(tps=50, prompt_tps=float("nan"), ttft=float("nan")),
    ]
    for perf in test_cases:
        score = compute_score(perf)
        assert math.isfinite(score), f"Non-finite score for {perf}: {score}"


# ===================================================================
# compute_pareto_objectives
# ===================================================================


def test_pareto_returns_3_tuple():
    result = compute_pareto_objectives(_p(vram_used_mb=4000), quality_factor=0.9)
    assert len(result) == 3
    # Supports tuple unpacking and indexing
    tps, neg_vram, qf = result
    assert result[0] == tps


def test_pareto_negative_vram_conversion():
    """VRAM should be negated so maximize(-VRAM) = minimize VRAM."""
    tps, neg_vram, qf = compute_pareto_objectives(_p(vram_used_mb=4000))
    assert tps == 50.0
    assert neg_vram == -4000
    assert qf == 1.0


def test_pareto_missing_vram_penalized():
    """Missing VRAM data gets a large penalty."""
    _, neg_vram, _ = compute_pareto_objectives(_p())
    assert neg_vram == -99999.0


def test_pareto_quality_factor_passthrough():
    _, _, qf = compute_pareto_objectives(
        _p(tps=30.0, vram_used_mb=2000), quality_factor=0.85
    )
    assert qf == 0.85


# ===================================================================
# _aggregate_samples
# ===================================================================


def test_aggregate_empty_returns_zeros():
    result = _aggregate_samples([])
    assert result.tps == 0.0
    assert result.ttft == 0.0
    assert result.prompt_tps == 0.0
    assert result.total_ms == 0.0


def test_aggregate_single_returns_itself():
    sample = _s(tps=42.0, ttft=100, prompt_tps=500, total_ms=200)
    result = _aggregate_samples([sample])
    assert result.tps == sample.tps
    assert result.ttft == sample.ttft
    assert result.prompt_tps == sample.prompt_tps
    assert result.total_ms == sample.total_ms


def test_aggregate_multiple_returns_median_by_score():
    """Three samples sorted by score; median (middle) is returned."""
    low = _s(tps=10, prompt_tps=100, ttft=1000, total_ms=2000)
    mid = _s(tps=50, prompt_tps=SCORE_PP_BASELINE, ttft=TTFT_BASELINE_MS, total_ms=1000)
    high = _s(
        tps=90,
        prompt_tps=SCORE_PP_BASELINE * 2,
        ttft=TTFT_BASELINE_MS / 2.5,
        total_ms=500,
    )
    # Verify ordering
    assert compute_score(low) < compute_score(mid)
    assert compute_score(mid) < compute_score(high)
    # Median should be the mid sample; verify by tps value
    result = _aggregate_samples([high, low, mid])
    assert result.tps == mid.tps


def test_aggregate_even_number_returns_lower_median():
    """With 4 samples, index 2 (0-indexed) is returned = 3rd element."""
    s1 = _s(tps=10, prompt_tps=100, ttft=1000, total_ms=2000)
    s2 = _s(tps=30, prompt_tps=200, ttft=700, total_ms=1500)
    s3 = _s(tps=50, prompt_tps=SCORE_PP_BASELINE, ttft=TTFT_BASELINE_MS, total_ms=1000)
    s4 = _s(tps=70, prompt_tps=400, ttft=300, total_ms=700)
    result = _aggregate_samples([s4, s1, s3, s2])
    # sorted by score: s1, s2, s3, s4 -> index 2 = s3
    assert result.tps == s3.tps
