"""Tests for eval functions from evals.py.

Functions are copied directly to avoid importing from the package
(which would trigger state.py's module-level CLI arg parsing).
"""
import math
import re
import unittest


# ---------------------------------------------------------------------------
# Constants from constants.py
# ---------------------------------------------------------------------------
KL_DIV_THRESHOLD = 0.5
KL_DIV_HARD_FAIL = 1.5
PPL_DEGRADATION_WARN = 0.10
PPL_DEGRADATION_FAIL = 0.30


# ---------------------------------------------------------------------------
# _extract_answer_letter — copied from evals.py (pure function)
# ---------------------------------------------------------------------------
def _extract_answer_letter(content):
    """Extract the MC answer letter (A/B/C/D) from model response."""
    content = content.strip()
    patterns = [
        r'(?:answer|choice)\s*(?:is|:)\s*\(?([A-D])\)?',
        r'\(?([A-D])\)?\s*$',
        r'^([A-D])\b',
        r'\(([A-D])\)',
    ]
    for pattern in patterns:
        m = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


# ---------------------------------------------------------------------------
# ppl_quality_factor — copied from evals.py
# ---------------------------------------------------------------------------
def ppl_quality_factor(baseline_ppl, trial_ppl):
    if baseline_ppl <= 0 or baseline_ppl == float('inf'):
        return 1.0
    if trial_ppl == float('inf'):
        return 0.1

    degradation = (trial_ppl - baseline_ppl) / baseline_ppl

    if degradation <= 0:
        return 1.0
    if degradation <= PPL_DEGRADATION_WARN:
        return 1.0 - 0.15 * (degradation / PPL_DEGRADATION_WARN)
    if degradation <= PPL_DEGRADATION_FAIL:
        t = (degradation - PPL_DEGRADATION_WARN) / (PPL_DEGRADATION_FAIL - PPL_DEGRADATION_WARN)
        return 0.85 - t * 0.75
    return 0.1


# ---------------------------------------------------------------------------
# kl_quality_factor — copied from evals.py
# ---------------------------------------------------------------------------
def kl_quality_factor(kl_div):
    if kl_div is None or kl_div <= 0:
        return 1.0
    if kl_div <= KL_DIV_THRESHOLD:
        return 1.0 - 0.15 * (kl_div / KL_DIV_THRESHOLD)
    if kl_div <= KL_DIV_HARD_FAIL:
        t = (kl_div - KL_DIV_THRESHOLD) / (KL_DIV_HARD_FAIL - KL_DIV_THRESHOLD)
        return 0.85 - t * 0.75
    return 0.1


# ===================================================================
# Tests
# ===================================================================

class TestExtractAnswerLetter(unittest.TestCase):
    """Tests for _extract_answer_letter()."""

    def test_answer_is_A_in_parens(self):
        self.assertEqual(_extract_answer_letter("The answer is (A)"), "A")

    def test_bare_letter_B(self):
        self.assertEqual(_extract_answer_letter("B"), "B")

    def test_letter_in_parens_C(self):
        self.assertEqual(_extract_answer_letter("(C)"), "C")

    def test_answer_is_D_with_period(self):
        self.assertEqual(_extract_answer_letter("I think the answer is D."), "D")

    def test_no_answer_returns_none(self):
        self.assertIsNone(_extract_answer_letter("no answer here"))

    def test_choice_colon_format(self):
        self.assertEqual(_extract_answer_letter("choice: B"), "B")

    def test_lowercase_answer(self):
        """Case-insensitive matching should work."""
        self.assertEqual(_extract_answer_letter("the answer is a"), "A")

    def test_answer_at_end(self):
        self.assertEqual(_extract_answer_letter("After analysis, A"), "A")

    def test_paren_letter_in_middle(self):
        self.assertEqual(_extract_answer_letter("I believe (B) is correct because..."), "B")

    def test_empty_string(self):
        self.assertIsNone(_extract_answer_letter(""))

    def test_whitespace_only(self):
        self.assertIsNone(_extract_answer_letter("   \n  "))


class TestPplQualityFactor(unittest.TestCase):
    """Tests for ppl_quality_factor()."""

    def test_no_degradation_returns_1(self):
        """Trial PPL same as baseline -> factor = 1.0."""
        self.assertAlmostEqual(ppl_quality_factor(10.0, 10.0), 1.0)

    def test_improvement_returns_1(self):
        """Trial PPL lower than baseline (improvement) -> factor = 1.0."""
        self.assertAlmostEqual(ppl_quality_factor(10.0, 9.0), 1.0)

    def test_10pct_degradation(self):
        """10% degradation = PPL_DEGRADATION_WARN boundary.

        degradation = (11.0 - 10.0) / 10.0 = 0.10
        factor = 1.0 - 0.15 * (0.10 / 0.10) = 1.0 - 0.15 = 0.85
        """
        self.assertAlmostEqual(ppl_quality_factor(10.0, 11.0), 0.85, places=2)

    def test_5pct_degradation(self):
        """5% degradation (half of warn threshold).

        degradation = 0.05
        factor = 1.0 - 0.15 * (0.05 / 0.10) = 1.0 - 0.075 = 0.925
        """
        self.assertAlmostEqual(ppl_quality_factor(10.0, 10.5), 0.925, places=3)

    def test_30pct_degradation(self):
        """30% degradation = PPL_DEGRADATION_FAIL boundary.

        degradation = 0.30
        t = (0.30 - 0.10) / (0.30 - 0.10) = 1.0
        factor = 0.85 - 1.0 * 0.75 = 0.10
        """
        self.assertAlmostEqual(ppl_quality_factor(10.0, 13.0), 0.10, places=2)

    def test_above_30pct_returns_floor(self):
        """50% degradation -> floor at 0.1."""
        self.assertAlmostEqual(ppl_quality_factor(10.0, 15.0), 0.1)

    def test_baseline_zero_returns_1(self):
        """baseline=0 -> no valid baseline, skip penalty."""
        self.assertAlmostEqual(ppl_quality_factor(0.0, 15.0), 1.0)

    def test_baseline_inf_returns_1(self):
        """baseline=inf -> no valid baseline, skip penalty."""
        self.assertAlmostEqual(ppl_quality_factor(float('inf'), 15.0), 1.0)

    def test_trial_inf_returns_01(self):
        """trial=inf -> measurement failed = 0.1."""
        self.assertAlmostEqual(ppl_quality_factor(10.0, float('inf')), 0.1)

    def test_20pct_degradation(self):
        """20% degradation (midway between warn and fail).

        degradation = 0.20
        t = (0.20 - 0.10) / (0.30 - 0.10) = 0.5
        factor = 0.85 - 0.5 * 0.75 = 0.85 - 0.375 = 0.475
        """
        self.assertAlmostEqual(ppl_quality_factor(10.0, 12.0), 0.475, places=3)


class TestKlQualityFactor(unittest.TestCase):
    """Tests for kl_quality_factor()."""

    def test_kl_zero_returns_1(self):
        self.assertAlmostEqual(kl_quality_factor(0.0), 1.0)

    def test_kl_none_returns_1(self):
        self.assertAlmostEqual(kl_quality_factor(None), 1.0)

    def test_kl_negative_returns_1(self):
        self.assertAlmostEqual(kl_quality_factor(-0.5), 1.0)

    def test_kl_at_threshold(self):
        """kl=0.5 (KL_DIV_THRESHOLD) -> factor = 1.0 - 0.15 * 1.0 = 0.85."""
        self.assertAlmostEqual(kl_quality_factor(KL_DIV_THRESHOLD), 0.85, places=2)

    def test_kl_half_threshold(self):
        """kl=0.25 -> factor = 1.0 - 0.15 * (0.25/0.5) = 1.0 - 0.075 = 0.925."""
        self.assertAlmostEqual(kl_quality_factor(0.25), 0.925, places=3)

    def test_kl_at_hard_fail(self):
        """kl=1.5 (KL_DIV_HARD_FAIL).

        t = (1.5 - 0.5) / (1.5 - 0.5) = 1.0
        factor = 0.85 - 1.0 * 0.75 = 0.10
        """
        self.assertAlmostEqual(kl_quality_factor(KL_DIV_HARD_FAIL), 0.10, places=2)

    def test_kl_above_hard_fail(self):
        """kl > KL_DIV_HARD_FAIL -> floor at 0.1."""
        self.assertAlmostEqual(kl_quality_factor(3.0), 0.1)

    def test_kl_midpoint_between_threshold_and_fail(self):
        """kl=1.0 (midpoint).

        t = (1.0 - 0.5) / (1.5 - 0.5) = 0.5
        factor = 0.85 - 0.5 * 0.75 = 0.475
        """
        self.assertAlmostEqual(kl_quality_factor(1.0), 0.475, places=3)

    def test_kl_monotonically_decreasing(self):
        """Quality factor should monotonically decrease as KL increases."""
        values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        factors = [kl_quality_factor(v) for v in values]
        for i in range(len(factors) - 1):
            # Use assertGreaterEqual with rounding to handle float precision
            self.assertGreaterEqual(round(factors[i], 10), round(factors[i + 1], 10),
                                    f"kl_quality_factor not monotonic at kl={values[i+1]}")

    def test_ppl_and_kl_symmetry(self):
        """PPL and KL quality factors should produce the same result at corresponding
        degradation fractions (both use the same piecewise-linear curve shape)."""
        # At their respective thresholds, both give 0.85
        self.assertAlmostEqual(ppl_quality_factor(10.0, 11.0), 0.85, places=2)  # 10% PPL
        self.assertAlmostEqual(kl_quality_factor(0.5), 0.85, places=2)           # threshold KL
        # At their respective fail points, both give 0.1
        self.assertAlmostEqual(ppl_quality_factor(10.0, 13.0), 0.10, places=2)  # 30% PPL
        self.assertAlmostEqual(kl_quality_factor(1.5), 0.10, places=2)           # hard fail KL


# ---------------------------------------------------------------------------
# _compute_kl_divergence -- copied from evals.py
# ---------------------------------------------------------------------------
def _compute_kl_divergence(baseline_dists, trial_dists):
    """Compute mean KL-divergence between baseline and trial logprob distributions."""
    n = min(len(baseline_dists), len(trial_dists))
    if n == 0:
        return 0.0

    kl_values = []
    for i in range(n):
        p_dist = baseline_dists[i]
        q_dist = trial_dists[i]
        all_tokens = p_dist.keys()
        kl = 0.0
        for token in all_tokens:
            p_logprob = p_dist.get(token, -20.0)
            q_logprob = q_dist.get(token, -20.0)
            p_prob = math.exp(p_logprob)
            q_prob = math.exp(q_logprob)
            if p_prob > 1e-10 and q_prob > 1e-10:
                kl += p_prob * math.log(p_prob / q_prob)
        kl_values.append(max(0.0, kl))

    return sum(kl_values) / len(kl_values) if kl_values else 0.0


# ---------------------------------------------------------------------------
# _score_quality_results -- copied from evals.py
# ---------------------------------------------------------------------------
QUALITY_WEIGHT_CORRECTNESS = 0.40
QUALITY_WEIGHT_CONFIDENCE = 0.40
QUALITY_WEIGHT_EFFICIENCY = 0.20
QUALITY_TTFT_BASELINE = 500


def _score_quality_results(task_results):
    """Compute composite quality score from individual task results."""
    if not task_results:
        return 0.0

    n = len(task_results)
    correctness = sum(1.0 for r in task_results if r["correct"]) / n

    logprobs = [r["logprob"] for r in task_results if r["logprob"] is not None]
    if logprobs:
        avg_logprob = sum(logprobs) / len(logprobs)
        confidence = min(1.0, max(0.0, math.exp(avg_logprob)))
    else:
        confidence = correctness

    ttfts = [r["ttft_ms"] for r in task_results if r["ttft_ms"] is not None and r["ttft_ms"] > 0]
    if ttfts:
        avg_ttft = sum(ttfts) / len(ttfts)
        efficiency = min(1.0, QUALITY_TTFT_BASELINE / avg_ttft) if avg_ttft > 0 else 0.0
    else:
        efficiency = 0.5

    score = (QUALITY_WEIGHT_CORRECTNESS * correctness +
             QUALITY_WEIGHT_CONFIDENCE * confidence +
             QUALITY_WEIGHT_EFFICIENCY * efficiency)
    return score * 100


class TestComputeKlDivergence(unittest.TestCase):
    """Tests for _compute_kl_divergence()."""

    def test_identical_distributions(self):
        """Identical distributions should give KL=0."""
        dists = [{"a": -1.0, "b": -2.0}]
        kl = _compute_kl_divergence(dists, dists)
        self.assertAlmostEqual(kl, 0.0, places=5)

    def test_empty_distributions(self):
        self.assertEqual(_compute_kl_divergence([], []), 0.0)

    def test_different_distributions_positive_kl(self):
        """Different distributions should give positive KL."""
        baseline = [{"a": -0.1, "b": -3.0}]  # P strongly favors "a"
        trial = [{"a": -3.0, "b": -0.1}]     # Q strongly favors "b"
        kl = _compute_kl_divergence(baseline, trial)
        self.assertGreater(kl, 0.0)

    def test_kl_nonnegative(self):
        """KL-divergence should always be non-negative."""
        baseline = [{"x": -0.5, "y": -1.5, "z": -3.0}]
        trial = [{"x": -1.0, "y": -1.0, "z": -2.0}]
        kl = _compute_kl_divergence(baseline, trial)
        self.assertGreaterEqual(kl, 0.0)

    def test_uses_minimum_length(self):
        """Should use the shorter of the two lists."""
        baseline = [{"a": -1.0}, {"b": -1.0}, {"c": -1.0}]
        trial = [{"a": -1.0}]
        kl = _compute_kl_divergence(baseline, trial)
        self.assertIsInstance(kl, float)

    def test_missing_tokens_use_floor(self):
        """Tokens missing from trial should use the -20.0 floor logprob."""
        baseline = [{"a": -0.1, "b": -3.0}]
        trial = [{"a": -0.1}]  # "b" missing from trial -> uses -20.0
        kl = _compute_kl_divergence(baseline, trial)
        self.assertGreaterEqual(kl, 0.0)


class TestScoreQualityResults(unittest.TestCase):
    """Tests for _score_quality_results()."""

    def test_empty_results(self):
        self.assertEqual(_score_quality_results([]), 0.0)

    def test_perfect_score(self):
        """All correct, high confidence (logprob=0), fast TTFT."""
        results = [
            {"correct": True, "logprob": 0.0, "ttft_ms": 100.0},
            {"correct": True, "logprob": 0.0, "ttft_ms": 100.0},
        ]
        score = _score_quality_results(results)
        # correctness=1.0, confidence=exp(0)=1.0, efficiency=min(1,500/100)=1.0
        # score = (0.4*1 + 0.4*1 + 0.2*1)*100 = 100
        self.assertAlmostEqual(score, 100.0, places=1)

    def test_all_wrong_no_logprobs(self):
        """All wrong, no logprobs -> correctness=0, confidence=0 (falls back)."""
        results = [
            {"correct": False, "logprob": None, "ttft_ms": None},
            {"correct": False, "logprob": None, "ttft_ms": None},
        ]
        score = _score_quality_results(results)
        # correctness=0, confidence=correctness=0, efficiency=0.5 (no timing)
        # score = (0.4*0 + 0.4*0 + 0.2*0.5)*100 = 10
        self.assertAlmostEqual(score, 10.0, places=1)

    def test_half_correct_moderate_confidence(self):
        """Half correct, moderate logprobs."""
        results = [
            {"correct": True, "logprob": -0.5, "ttft_ms": 500.0},
            {"correct": False, "logprob": -2.0, "ttft_ms": 500.0},
        ]
        score = _score_quality_results(results)
        expected = (0.4 * 0.5 + 0.4 * math.exp(-1.25) + 0.2 * 1.0) * 100
        self.assertAlmostEqual(score, expected, places=1)

    def test_slow_ttft_reduces_efficiency(self):
        """Slow TTFT should reduce the overall score."""
        fast = [{"correct": True, "logprob": 0.0, "ttft_ms": 100.0}]
        slow = [{"correct": True, "logprob": 0.0, "ttft_ms": 2000.0}]
        score_fast = _score_quality_results(fast)
        score_slow = _score_quality_results(slow)
        self.assertGreater(score_fast, score_slow)

    def test_no_timing_data_uses_neutral_efficiency(self):
        """When no TTFT data, efficiency defaults to 0.5."""
        results = [{"correct": True, "logprob": 0.0, "ttft_ms": None}]
        score = _score_quality_results(results)
        # correctness=1, confidence=1, efficiency=0.5
        expected = (0.4 * 1.0 + 0.4 * 1.0 + 0.2 * 0.5) * 100
        self.assertAlmostEqual(score, expected, places=1)


if __name__ == "__main__":
    unittest.main()
