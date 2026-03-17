"""Tests for scoring and measurement functions from measurement.py.

Functions are tested without importing from the package to avoid
triggering state.py's module-level side effects (CLI arg parsing).
Instead, we replicate the pure scoring logic directly.
"""
import math
import unittest


# ---------------------------------------------------------------------------
# Constants copied from constants.py (leaf module, no side effects)
# ---------------------------------------------------------------------------
SCORE_TTFT_BASELINE = 500   # ms
SCORE_PP_BASELINE = 300     # t/s


# ---------------------------------------------------------------------------
# compute_score — copied verbatim from measurement.py (pure function)
# ---------------------------------------------------------------------------
def compute_score(perf, vram_used_mb=None, vram_total_mb=None):
    if vram_used_mb is None:
        vram_used_mb = perf.get("vram_used_mb")
    if vram_total_mb is None:
        vram_total_mb = perf.get("vram_total_mb")
    gen_tps = perf["tps"]
    prompt_tps = perf["prompt_tps"]
    ttft = perf["ttft"]

    if gen_tps <= 0:
        return 0.0

    prompt_tps = min(50000.0, max(0.0, prompt_tps)) if math.isfinite(prompt_tps) else 50000.0
    ttft = max(1.0, ttft) if math.isfinite(ttft) else 1.0

    large_tps = perf.get("large_tps")

    if large_tps and large_tps > 0:
        pp_norm = min(prompt_tps / SCORE_PP_BASELINE, 3.0)
        ttft_norm = min(SCORE_TTFT_BASELINE / ttft, 3.0)

        score = (gen_tps * 0.35 +
                 large_tps * 0.25 +
                 pp_norm * gen_tps * 0.15 +
                 ttft_norm * gen_tps * 0.15)

        if vram_used_mb is not None and vram_total_mb is not None and vram_total_mb > 0:
            vram_efficiency = 1.0 - (vram_used_mb / vram_total_mb)
            vram_bonus = max(0.0, min(1.0, vram_efficiency))
            score += vram_bonus * gen_tps * 0.10
        else:
            score += gen_tps * 0.10
    else:
        pp_factor = min((prompt_tps / SCORE_PP_BASELINE), 3.0) if prompt_tps > 0 else 0.0
        ttft_factor = min((SCORE_TTFT_BASELINE / ttft), 3.0) if ttft > 0 else 0.0

        score = gen_tps * (0.60 + 0.25 * pp_factor + 0.15 * ttft_factor)

        if vram_used_mb is not None and vram_total_mb is not None and vram_total_mb > 0:
            utilization = vram_used_mb / vram_total_mb
            headroom_bonus = min(0.05, max(0.0, (1.0 - utilization) * 0.10))
            score *= (1.0 + headroom_bonus)

    concurrent_tps = perf.get("concurrent_total_tps")
    if concurrent_tps and concurrent_tps > 0:
        n_users = perf.get("concurrent_users", 4)
        scaling_efficiency = concurrent_tps / (gen_tps * n_users) if gen_tps > 0 else 0
        scaling_efficiency = min(1.0, max(0.0, scaling_efficiency))
        score *= (0.85 + 0.15 * scaling_efficiency)

    return score if math.isfinite(score) else 0.0


# ---------------------------------------------------------------------------
# compute_pareto_objectives — copied from measurement.py
# ---------------------------------------------------------------------------
def compute_pareto_objectives(perf, quality_factor=1.0):
    tps = perf.get("tps", 0.0)
    vram_mb = perf.get("vram_used_mb")
    neg_vram = -vram_mb if vram_mb is not None else -99999.0
    return (tps, neg_vram, quality_factor)


# ---------------------------------------------------------------------------
# _aggregate_samples — needs compute_score, which we already have above
# ---------------------------------------------------------------------------
def _aggregate_samples(samples):
    if not samples:
        return {"tps": 0.0, "ttft": 0.0, "prompt_tps": 0.0, "total_ms": 0.0}
    if len(samples) == 1:
        return samples[0]
    ranked = sorted(samples, key=lambda s: compute_score(s))
    return ranked[len(ranked) // 2]


# ===================================================================
# Tests
# ===================================================================

class TestComputeScore(unittest.TestCase):
    """Tests for compute_score()."""

    def test_zero_tps_returns_zero(self):
        perf = {"tps": 0, "prompt_tps": 300, "ttft": 500}
        self.assertEqual(compute_score(perf), 0.0)

    def test_negative_tps_returns_zero(self):
        perf = {"tps": -10, "prompt_tps": 300, "ttft": 500}
        self.assertEqual(compute_score(perf), 0.0)

    def test_lightweight_mode_basic(self):
        """Lightweight mode: gen_tps=50, prompt_tps=300, ttft=500.

        pp_factor = min(300/300, 3.0) = 1.0
        ttft_factor = min(500/500, 3.0) = 1.0
        score = 50 * (0.60 + 0.25*1.0 + 0.15*1.0)
             = 50 * (0.60 + 0.25 + 0.15)
             = 50 * 1.0
             = 50.0
        """
        perf = {"tps": 50, "prompt_tps": 300, "ttft": 500}
        score = compute_score(perf)
        self.assertAlmostEqual(score, 50.0, places=1)

    def test_lightweight_mode_high_pp(self):
        """Higher prompt_tps should boost the score."""
        perf = {"tps": 50, "prompt_tps": 600, "ttft": 500}
        # pp_factor = min(600/300, 3.0) = 2.0
        # score = 50 * (0.60 + 0.25*2.0 + 0.15*1.0) = 50 * 1.25 = 62.5
        score = compute_score(perf)
        self.assertAlmostEqual(score, 62.5, places=1)

    def test_lightweight_mode_fast_ttft(self):
        """Faster TTFT should boost the score."""
        perf = {"tps": 50, "prompt_tps": 300, "ttft": 250}
        # ttft_factor = min(500/250, 3.0) = 2.0
        # score = 50 * (0.60 + 0.25*1.0 + 0.15*2.0) = 50 * 1.15 = 57.5
        score = compute_score(perf)
        self.assertAlmostEqual(score, 57.5, places=1)

    def test_lightweight_pp_factor_capped_at_3(self):
        """pp_factor is capped at 3.0."""
        perf = {"tps": 50, "prompt_tps": 9000, "ttft": 500}
        # pp_factor = min(9000/300, 3.0) = 3.0
        # score = 50 * (0.60 + 0.25*3.0 + 0.15*1.0) = 50 * 1.50 = 75.0
        score = compute_score(perf)
        self.assertAlmostEqual(score, 75.0, places=1)

    def test_full_mode_basic(self):
        """Full mode: gen_tps=50, large_tps=40, prompt_tps=300, ttft=500, no VRAM.

        pp_norm = min(300/300, 3.0) = 1.0
        ttft_norm = min(500/500, 3.0) = 1.0
        score = 50*0.35 + 40*0.25 + 1.0*50*0.15 + 1.0*50*0.15
             = 17.5 + 10.0 + 7.5 + 7.5
             = 42.5
        No VRAM data -> add gen_tps * 0.10 = 5.0
        Total = 47.5
        """
        perf = {"tps": 50, "prompt_tps": 300, "ttft": 500, "large_tps": 40}
        score = compute_score(perf)
        self.assertAlmostEqual(score, 47.5, places=1)

    def test_full_mode_with_vram(self):
        """Full mode with VRAM data.

        Base score (same as above without VRAM redistribution):
        50*0.35 + 40*0.25 + 1.0*50*0.15 + 1.0*50*0.15 = 42.5

        VRAM efficiency = 1.0 - (4000/8000) = 0.5
        vram_bonus = 0.5 (clamped to [0,1])
        score += 0.5 * 50 * 0.10 = 2.5
        Total = 45.0
        """
        perf = {"tps": 50, "prompt_tps": 300, "ttft": 500, "large_tps": 40}
        score = compute_score(perf, vram_used_mb=4000, vram_total_mb=8000)
        self.assertAlmostEqual(score, 45.0, places=1)

    def test_lightweight_vram_efficiency_bonus(self):
        """Lightweight mode with VRAM data gets up to 5% boost.

        Base score = 50.0 (from test_lightweight_mode_basic)
        utilization = 4000/8000 = 0.5
        headroom_bonus = min(0.05, max(0.0, 0.5 * 0.10)) = min(0.05, 0.05) = 0.05
        score *= 1.05
        Final = 50.0 * 1.05 = 52.5
        """
        perf = {"tps": 50, "prompt_tps": 300, "ttft": 500}
        score = compute_score(perf, vram_used_mb=4000, vram_total_mb=8000)
        self.assertAlmostEqual(score, 52.5, places=1)

    def test_lightweight_vram_high_utilization(self):
        """High VRAM utilization gives smaller bonus.

        utilization = 7000/8000 = 0.875
        headroom_bonus = min(0.05, max(0.0, 0.125 * 0.10)) = min(0.05, 0.0125) = 0.0125
        score = 50.0 * 1.0125 = 50.625
        """
        perf = {"tps": 50, "prompt_tps": 300, "ttft": 500}
        score = compute_score(perf, vram_used_mb=7000, vram_total_mb=8000)
        self.assertAlmostEqual(score, 50.625, places=2)

    def test_concurrent_load_bonus(self):
        """Concurrent load bonus scales score by (0.85 + 0.15 * efficiency).

        Base score = 50.0
        scaling_efficiency = 150 / (50 * 4) = 0.75
        score *= (0.85 + 0.15 * 0.75) = 0.85 + 0.1125 = 0.9625
        Final = 50.0 * 0.9625 = 48.125
        """
        perf = {
            "tps": 50, "prompt_tps": 300, "ttft": 500,
            "concurrent_total_tps": 150, "concurrent_users": 4,
        }
        score = compute_score(perf)
        self.assertAlmostEqual(score, 48.125, places=2)

    def test_concurrent_load_perfect_scaling(self):
        """Perfect scaling (efficiency=1.0) gives full 15% bonus.

        Base = 50.0, scaling_efficiency = 200/(50*4) = 1.0
        score *= (0.85 + 0.15*1.0) = 1.0
        Final = 50.0
        """
        perf = {
            "tps": 50, "prompt_tps": 300, "ttft": 500,
            "concurrent_total_tps": 200, "concurrent_users": 4,
        }
        score = compute_score(perf)
        self.assertAlmostEqual(score, 50.0, places=1)

    def test_nan_tps_input(self):
        """NaN gen_tps: gen_tps is checked <= 0, NaN fails that, but final
        isfinite check catches the NaN score."""
        perf = {"tps": float("nan"), "prompt_tps": 300, "ttft": 500}
        score = compute_score(perf)
        self.assertEqual(score, 0.0)

    def test_inf_prompt_tps_clamped(self):
        """Inf prompt_tps is clamped to 50000.

        pp_factor = min(50000/300, 3.0) = 3.0
        score = 50 * (0.60 + 0.25*3.0 + 0.15*1.0) = 50 * 1.50 = 75.0
        """
        perf = {"tps": 50, "prompt_tps": float("inf"), "ttft": 500}
        score = compute_score(perf)
        self.assertAlmostEqual(score, 75.0, places=1)

    def test_inf_ttft_clamped(self):
        """Inf ttft is clamped to 1.0ms.

        ttft_factor = min(500/1.0, 3.0) = 3.0
        score = 50 * (0.60 + 0.25*1.0 + 0.15*3.0) = 50 * 1.30 = 65.0
        """
        perf = {"tps": 50, "prompt_tps": 300, "ttft": float("inf")}
        score = compute_score(perf)
        # inf ttft -> clamped to 1.0, so ttft_factor = min(500/1, 3.0) = 3.0
        self.assertAlmostEqual(score, 65.0, places=1)

    def test_nan_prompt_tps_treated_as_50000(self):
        """NaN prompt_tps is treated as 50000 (same as inf case)."""
        perf = {"tps": 50, "prompt_tps": float("nan"), "ttft": 500}
        score = compute_score(perf)
        self.assertAlmostEqual(score, 75.0, places=1)

    def test_result_is_always_finite(self):
        """Score must always be finite (not inf, not NaN)."""
        test_cases = [
            {"tps": 50, "prompt_tps": 0, "ttft": 0},
            {"tps": 50, "prompt_tps": float("inf"), "ttft": float("inf")},
            {"tps": 50, "prompt_tps": float("nan"), "ttft": float("nan")},
        ]
        for perf in test_cases:
            score = compute_score(perf)
            self.assertTrue(math.isfinite(score), f"Non-finite score for {perf}: {score}")


class TestComputeParetoObjectives(unittest.TestCase):
    """Tests for compute_pareto_objectives()."""

    def test_returns_3_tuple(self):
        perf = {"tps": 50.0, "vram_used_mb": 4000}
        result = compute_pareto_objectives(perf, quality_factor=0.9)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_negative_vram_conversion(self):
        """VRAM should be negated so maximize(-VRAM) = minimize VRAM."""
        perf = {"tps": 50.0, "vram_used_mb": 4000}
        tps, neg_vram, qf = compute_pareto_objectives(perf)
        self.assertEqual(tps, 50.0)
        self.assertEqual(neg_vram, -4000)
        self.assertEqual(qf, 1.0)

    def test_missing_vram_penalized(self):
        """Missing VRAM data gets a large penalty."""
        perf = {"tps": 50.0}
        _, neg_vram, _ = compute_pareto_objectives(perf)
        self.assertEqual(neg_vram, -99999.0)

    def test_quality_factor_passthrough(self):
        perf = {"tps": 30.0, "vram_used_mb": 2000}
        _, _, qf = compute_pareto_objectives(perf, quality_factor=0.85)
        self.assertEqual(qf, 0.85)


class TestAggregateSamples(unittest.TestCase):
    """Tests for _aggregate_samples()."""

    def test_empty_list_returns_zeros(self):
        result = _aggregate_samples([])
        self.assertEqual(result["tps"], 0.0)
        self.assertEqual(result["ttft"], 0.0)
        self.assertEqual(result["prompt_tps"], 0.0)
        self.assertEqual(result["total_ms"], 0.0)

    def test_single_sample_returns_itself(self):
        sample = {"tps": 42.0, "ttft": 100, "prompt_tps": 500, "total_ms": 200}
        result = _aggregate_samples([sample])
        self.assertIs(result, sample)

    def test_multiple_samples_returns_median_by_score(self):
        """Three samples sorted by score; median (middle) is returned."""
        low = {"tps": 10, "prompt_tps": 100, "ttft": 1000, "total_ms": 2000}
        mid = {"tps": 50, "prompt_tps": 300, "ttft": 500, "total_ms": 1000}
        high = {"tps": 90, "prompt_tps": 600, "ttft": 200, "total_ms": 500}
        # Verify ordering: low < mid < high
        self.assertLess(compute_score(low), compute_score(mid))
        self.assertLess(compute_score(mid), compute_score(high))
        # Median should be the mid sample
        result = _aggregate_samples([high, low, mid])
        self.assertIs(result, mid)

    def test_even_number_returns_lower_median(self):
        """With 4 samples, index 2 (0-indexed) is returned = 3rd element."""
        s1 = {"tps": 10, "prompt_tps": 100, "ttft": 1000, "total_ms": 2000}
        s2 = {"tps": 30, "prompt_tps": 200, "ttft": 700, "total_ms": 1500}
        s3 = {"tps": 50, "prompt_tps": 300, "ttft": 500, "total_ms": 1000}
        s4 = {"tps": 70, "prompt_tps": 400, "ttft": 300, "total_ms": 700}
        result = _aggregate_samples([s4, s1, s3, s2])
        # sorted by score: s1, s2, s3, s4 -> index 2 = s3
        self.assertIs(result, s3)


if __name__ == "__main__":
    unittest.main()
