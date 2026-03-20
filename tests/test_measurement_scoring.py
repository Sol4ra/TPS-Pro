"""Tests for measurement.py: compute_score and compute_pareto_objectives.

Covers NaN/Inf sanitization, concurrent scaling bonus, full/lightweight mode
branching, normalization boundaries, and Pareto objectives.
"""

from __future__ import annotations

import math

import pytest

from tps_pro.constants import (
    CONCURRENT_BASE_FACTOR,
    CONCURRENT_BONUS_WEIGHT,
    LITE_BASE_MULTIPLIER,
    LITE_MULTIPLIER_CAP,
    LITE_VRAM_BONUS_CAP,
    LITE_WEIGHT_TTFT,
    PROMPT_TPS_CLAMP_MAX,
    SCORE_PP_BASELINE,
    TTFT_BASELINE_MS,
    TTFT_FLOOR_MS,
    WEIGHT_VRAM,
)
from tps_pro.measurement import (
    compute_pareto_objectives,
    compute_score,
)
from tps_pro.result_types import PerfResult


def _make_perf(
    tps=50.0,
    prompt_tps=SCORE_PP_BASELINE,
    ttft=TTFT_BASELINE_MS,
    total_ms=1000.0,
    **extra,
) -> PerfResult:
    """Build a PerfResult with sensible defaults."""
    return PerfResult(
        tps=tps, prompt_tps=prompt_tps, ttft=ttft, total_ms=total_ms, **extra
    )


# ===================================================================
# compute_score — NaN / Inf sanitization
# ===================================================================


class TestComputeScoreNanInfSanitization:
    """Edge cases for NaN/Inf sanitization in compute_score."""

    @pytest.mark.unit
    def test_nan_gen_tps_returns_zero(self):
        perf = _make_perf(tps=float("nan"))
        assert compute_score(perf) == 0.0

    @pytest.mark.unit
    def test_inf_gen_tps_full_mode_returns_finite(self):
        perf = _make_perf(tps=float("inf"), large_tps=40)
        result = compute_score(perf)
        assert result == 0.0

    @pytest.mark.unit
    def test_nan_prompt_tps_clamped_to_max(self):
        perf = _make_perf(prompt_tps=float("nan"))
        score_nan = compute_score(perf)
        perf_clamped = _make_perf(prompt_tps=PROMPT_TPS_CLAMP_MAX)
        score_clamped = compute_score(perf_clamped)
        assert score_nan == pytest.approx(score_clamped, abs=0.01)

    @pytest.mark.unit
    def test_nan_ttft_clamped_to_floor(self):
        perf = _make_perf(ttft=float("nan"))
        score_nan = compute_score(perf)
        perf_floor = _make_perf(ttft=TTFT_FLOOR_MS)
        score_floor = compute_score(perf_floor)
        assert score_nan == pytest.approx(score_floor, abs=0.01)

    @pytest.mark.unit
    def test_negative_prompt_tps_clamped_to_zero(self):
        perf = _make_perf(prompt_tps=-500)
        score = compute_score(perf)
        expected = 50.0 * (LITE_BASE_MULTIPLIER + LITE_WEIGHT_TTFT * 1.0)
        assert score == pytest.approx(expected, abs=0.5)

    @pytest.mark.unit
    def test_inf_prompt_tps_in_full_mode(self):
        perf = _make_perf(prompt_tps=float("inf"), large_tps=40)
        score = compute_score(perf)
        assert math.isfinite(score)
        assert score > 0

    @pytest.mark.unit
    def test_negative_ttft_clamped_to_floor(self):
        perf = _make_perf(ttft=-100)
        score = compute_score(perf)
        assert math.isfinite(score)
        assert score > 0


# ===================================================================
# compute_score — concurrent scaling bonus
# ===================================================================


class TestComputeScoreConcurrentBonus:
    """Concurrent load bonus integration in compute_score."""

    @pytest.mark.unit
    def test_no_concurrent_data_no_bonus(self):
        perf = _make_perf()
        score = compute_score(perf)
        assert score == pytest.approx(50.0, abs=0.5)

    @pytest.mark.unit
    def test_zero_concurrent_tps_no_bonus(self):
        perf = _make_perf(concurrent_total_tps=0, concurrent_users=4)
        score = compute_score(perf)
        base = compute_score(_make_perf())
        assert score == pytest.approx(base, abs=0.01)

    @pytest.mark.unit
    def test_scaling_efficiency_capped_at_one(self):
        perf = _make_perf(concurrent_total_tps=400, concurrent_users=4)
        base = compute_score(_make_perf())
        expected = base * (CONCURRENT_BASE_FACTOR + CONCURRENT_BONUS_WEIGHT * 1.0)
        assert compute_score(perf) == pytest.approx(expected, abs=0.1)

    @pytest.mark.unit
    def test_poor_scaling_reduces_score(self):
        perf = _make_perf(concurrent_total_tps=50, concurrent_users=4)
        base = compute_score(_make_perf())
        score = compute_score(perf)
        expected = base * (CONCURRENT_BASE_FACTOR + CONCURRENT_BONUS_WEIGHT * 0.25)
        assert score == pytest.approx(expected, abs=0.1)
        assert score < base

    @pytest.mark.unit
    def test_default_concurrent_users_is_4(self):
        perf = _make_perf(concurrent_total_tps=200)
        base = compute_score(_make_perf())
        expected = base * (CONCURRENT_BASE_FACTOR + CONCURRENT_BONUS_WEIGHT * 1.0)
        assert compute_score(perf) == pytest.approx(expected, abs=0.1)


# ===================================================================
# compute_score — full mode vs lightweight mode branching
# ===================================================================


class TestComputeScoreModeBranching:
    """Full mode vs lightweight mode branching."""

    @pytest.mark.unit
    def test_large_tps_zero_stays_lightweight(self):
        perf = _make_perf(large_tps=0)
        base = compute_score(_make_perf())
        assert compute_score(perf) == pytest.approx(base, abs=0.01)

    @pytest.mark.unit
    def test_large_tps_none_stays_lightweight(self):
        perf = _make_perf()
        assert perf.large_tps is None
        score = compute_score(perf)
        assert score > 0

    @pytest.mark.unit
    def test_large_tps_negative_stays_lightweight(self):
        perf = _make_perf(large_tps=-1)
        base = compute_score(_make_perf())
        assert compute_score(perf) == pytest.approx(base, abs=0.01)

    @pytest.mark.unit
    def test_full_mode_triggered_with_positive_large_tps(self):
        lite_score = compute_score(_make_perf())
        full_score = compute_score(_make_perf(large_tps=40))
        assert lite_score != pytest.approx(full_score, abs=0.01)

    @pytest.mark.unit
    def test_full_mode_vram_redistribution_without_data(self):
        perf = _make_perf(large_tps=40)
        score_no_vram = compute_score(perf)
        score_with_vram = compute_score(perf, vram_used_mb=4000, vram_total_mb=8000)
        assert score_no_vram > score_with_vram


# ===================================================================
# compute_score — boundary values for normalization caps
# ===================================================================


class TestComputeScoreNormalizationBoundaries:
    """Boundary values for normalization caps."""

    @pytest.mark.unit
    def test_pp_norm_at_exactly_3x_baseline(self):
        perf = _make_perf(prompt_tps=SCORE_PP_BASELINE * 3, large_tps=40)
        score_3x = compute_score(perf)
        perf_10x = _make_perf(prompt_tps=SCORE_PP_BASELINE * 10, large_tps=40)
        score_10x = compute_score(perf_10x)
        assert score_3x == pytest.approx(score_10x, abs=0.01)

    @pytest.mark.unit
    def test_ttft_norm_at_exactly_3x_baseline(self):
        perf = _make_perf(ttft=TTFT_BASELINE_MS / 3, large_tps=40)
        score_3x = compute_score(perf)
        perf_tiny = _make_perf(ttft=1.0, large_tps=40)
        score_tiny = compute_score(perf_tiny)
        assert score_3x == pytest.approx(score_tiny, abs=0.01)

    @pytest.mark.unit
    def test_lite_multiplier_cap_reached(self):
        perf = _make_perf(prompt_tps=SCORE_PP_BASELINE * 100, ttft=1.0)
        score = compute_score(perf)
        expected = 50.0 * LITE_MULTIPLIER_CAP
        assert score == pytest.approx(expected, abs=0.5)

    @pytest.mark.unit
    def test_vram_efficiency_full_mode_boundaries(self):
        perf = _make_perf(large_tps=40)
        score_full = compute_score(perf, vram_used_mb=8000, vram_total_mb=8000)
        score_empty = compute_score(perf, vram_used_mb=0, vram_total_mb=8000)
        assert score_empty > score_full
        diff = score_empty - score_full
        assert diff == pytest.approx(50.0 * WEIGHT_VRAM, abs=0.1)

    @pytest.mark.unit
    def test_lite_vram_bonus_cap(self):
        perf = _make_perf()
        base = compute_score(perf)
        score_max_bonus = compute_score(perf, vram_used_mb=0, vram_total_mb=8000)
        assert score_max_bonus == pytest.approx(
            base * (1.0 + LITE_VRAM_BONUS_CAP), abs=0.1
        )


# ===================================================================
# compute_pareto_objectives
# ===================================================================


class TestComputeParetoObjectives:
    """Tests for compute_pareto_objectives."""

    @pytest.mark.unit
    def test_returns_3_tuple(self):
        perf = _make_perf(tps=80.0, vram_used_mb=6000)
        result = compute_pareto_objectives(perf, quality_factor=0.9)
        assert len(result) == 3
        tps, neg_vram, qf = result
        assert result[0] == tps

    @pytest.mark.unit
    def test_tps_passthrough(self):
        perf = _make_perf(tps=42.5, vram_used_mb=3000)
        tps, _, _ = compute_pareto_objectives(perf)
        assert tps == 42.5

    @pytest.mark.unit
    def test_vram_negated(self):
        perf = _make_perf(tps=50, vram_used_mb=5000)
        _, neg_vram, _ = compute_pareto_objectives(perf)
        assert neg_vram == -5000

    @pytest.mark.unit
    def test_missing_vram_heavily_penalized(self):
        perf = _make_perf(tps=50)
        _, neg_vram, _ = compute_pareto_objectives(perf)
        assert neg_vram == -99999.0

    @pytest.mark.unit
    def test_quality_factor_default_is_one(self):
        perf = _make_perf(tps=50, vram_used_mb=1000)
        _, _, qf = compute_pareto_objectives(perf)
        assert qf == 1.0

    @pytest.mark.unit
    def test_quality_factor_passed_through(self):
        perf = _make_perf(tps=50, vram_used_mb=1000)
        _, _, qf = compute_pareto_objectives(perf, quality_factor=0.42)
        assert qf == 0.42

    @pytest.mark.unit
    def test_zero_tps_defaults(self):
        perf = _make_perf(tps=0.0)
        tps, _, _ = compute_pareto_objectives(perf)
        assert tps == 0.0

    @pytest.mark.unit
    def test_zero_vram_negated_correctly(self):
        perf = _make_perf(tps=50, vram_used_mb=0)
        _, neg_vram, _ = compute_pareto_objectives(perf)
        assert neg_vram == 0
