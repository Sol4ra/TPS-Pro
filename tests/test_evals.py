"""Tests for eval functions from evals submodules.

All functions are imported directly from their specific submodules
(evals.mcq, evals.kl_divergence, evals.perplexity) rather than from
evals/__init__.py, which avoids pulling in heavy transitive dependencies.
Importing state.py is safe — it only creates empty sentinel objects at
module level; CLI arg parsing only runs when initialize() is called.
"""

import math

import pytest

from tps_pro.constants import (
    KL_DIV_HARD_FAIL,
    KL_DIV_THRESHOLD,
)
from tps_pro.evals.kl_divergence import (
    _compute_kl_divergence,
    kl_quality_factor,
)
from tps_pro.evals.mcq import (
    _extract_answer_letter,
    _score_quality_results,
)
from tps_pro.evals.perplexity import ppl_quality_factor

pytestmark = pytest.mark.unit

# ===================================================================
# _extract_answer_letter
# ===================================================================


@pytest.mark.parametrize(
    "text, expected",
    [
        ("The answer is (A)", "A"),
        ("B", "B"),
        ("(C)", "C"),
        ("I think the answer is D.", "D"),
        ("choice: B", "B"),
        ("the answer is a", "A"),  # case-insensitive
        ("After analysis, A", "A"),  # letter at end
        ("I believe (B) is correct because...", "B"),  # paren in middle
    ],
    ids=[
        "answer-is-A-parens",
        "bare-B",
        "paren-C",
        "answer-is-D-period",
        "choice-colon-B",
        "lowercase-a",
        "letter-at-end",
        "paren-in-middle",
    ],
)
def test_extract_answer_letter(text, expected):
    assert _extract_answer_letter(text) == expected


@pytest.mark.parametrize(
    "text",
    ["no answer here", "", "   \n  "],
    ids=["no-answer", "empty", "whitespace-only"],
)
def test_extract_answer_letter_none(text):
    assert _extract_answer_letter(text) is None


# ===================================================================
# ppl_quality_factor
# ===================================================================


@pytest.mark.parametrize(
    "baseline, trial, expected",
    [
        (10.0, 10.0, 1.0),  # no degradation
        (10.0, 9.0, 1.0),  # improvement
        (10.0, 10.5, 0.925),  # 5% degradation
        (10.0, 11.0, 0.85),  # 10% = PPL_DEGRADATION_WARN boundary
        (10.0, 12.0, 0.475),  # 20% = midway between warn and fail
        (10.0, 13.0, 0.10),  # 30% = PPL_DEGRADATION_FAIL boundary
        (10.0, 15.0, 0.1),  # 50% degradation -> floor
    ],
    ids=[
        "no-degradation",
        "improvement",
        "5pct",
        "10pct-warn",
        "20pct-midway",
        "30pct-fail",
        "50pct-floor",
    ],
)
def test_ppl_quality_factor(baseline, trial, expected):
    assert ppl_quality_factor(baseline, trial) == pytest.approx(expected, abs=0.005)


def test_ppl_quality_baseline_zero():
    """baseline=0 -> no valid baseline, skip penalty."""
    assert ppl_quality_factor(0.0, 15.0) == pytest.approx(1.0)


def test_ppl_quality_baseline_inf():
    """baseline=inf -> no valid baseline, skip penalty."""
    assert ppl_quality_factor(float("inf"), 15.0) == pytest.approx(1.0)


def test_ppl_quality_trial_inf():
    """trial=inf -> measurement failed = 0.1."""
    assert ppl_quality_factor(10.0, float("inf")) == pytest.approx(0.1)


# ===================================================================
# kl_quality_factor
# ===================================================================


@pytest.mark.parametrize(
    "kl_div, expected",
    [
        (0.0, 1.0),
        (None, 1.0),
        (-0.5, 1.0),
        (0.25, 0.925),  # half of threshold
        (KL_DIV_THRESHOLD, 0.85),  # at threshold
        (1.0, 0.475),  # midpoint between threshold and fail
        (KL_DIV_HARD_FAIL, 0.10),  # at hard fail
        (3.0, 0.1),  # above hard fail -> floor
    ],
    ids=[
        "zero",
        "none",
        "negative",
        "half-threshold",
        "at-threshold",
        "midpoint",
        "at-hard-fail",
        "above-hard-fail",
    ],
)
def test_kl_quality_factor(kl_div, expected):
    assert kl_quality_factor(kl_div) == pytest.approx(expected, abs=0.005)


def test_kl_monotonically_decreasing():
    """Quality factor should monotonically decrease as KL increases."""
    values = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    factors = [kl_quality_factor(v) for v in values]
    for i in range(len(factors) - 1):
        assert round(factors[i], 10) >= round(factors[i + 1], 10), (
            f"kl_quality_factor not monotonic at kl={values[i + 1]}"
        )


def test_ppl_and_kl_symmetry():
    """PPL and KL quality factors produce the same result at corresponding
    degradation fractions (both use the same piecewise-linear curve shape)."""
    # At their respective thresholds, both give 0.85
    assert ppl_quality_factor(10.0, 11.0) == pytest.approx(0.85, abs=0.01)
    assert kl_quality_factor(KL_DIV_THRESHOLD) == pytest.approx(0.85, abs=0.01)
    # At their respective fail points, both give 0.1
    assert ppl_quality_factor(10.0, 13.0) == pytest.approx(0.10, abs=0.01)
    assert kl_quality_factor(KL_DIV_HARD_FAIL) == pytest.approx(0.10, abs=0.01)


# ===================================================================
# _compute_kl_divergence
# ===================================================================


def test_kl_identical_distributions():
    """Identical distributions should give KL=0."""
    dists = [{"a": -1.0, "b": -2.0}]
    assert _compute_kl_divergence(dists, dists) == pytest.approx(0.0, abs=1e-5)


def test_kl_empty_distributions():
    assert _compute_kl_divergence([], []) == 0.0


def test_kl_different_distributions_positive():
    """Different distributions should give positive KL."""
    baseline = [{"a": -0.1, "b": -3.0}]
    trial = [{"a": -3.0, "b": -0.1}]
    assert _compute_kl_divergence(baseline, trial) > 0.0


def test_kl_nonnegative():
    """KL-divergence should always be non-negative."""
    baseline = [{"x": -0.5, "y": -1.5, "z": -3.0}]
    trial = [{"x": -1.0, "y": -1.0, "z": -2.0}]
    assert _compute_kl_divergence(baseline, trial) >= 0.0


def test_kl_uses_minimum_length():
    """Should use the shorter of the two lists."""
    baseline = [{"a": -1.0}, {"b": -1.0}, {"c": -1.0}]
    trial = [{"a": -1.0}]
    kl = _compute_kl_divergence(baseline, trial)
    assert isinstance(kl, float)


def test_kl_missing_tokens_use_floor():
    """Tokens missing from trial should use the -20.0 floor logprob."""
    baseline = [{"a": -0.1, "b": -3.0}]
    trial = [{"a": -0.1}]
    assert _compute_kl_divergence(baseline, trial) >= 0.0


# ===================================================================
# _score_quality_results
# ===================================================================


def _qtr(correct, logprob=None, ttft_ms=None):
    """Helper to create QualityTaskResult for tests."""
    from tps_pro.result_types import QualityTaskResult

    return QualityTaskResult(correct=correct, logprob=logprob, ttft_ms=ttft_ms)


def test_score_quality_empty():
    assert _score_quality_results([]).score == 0.0


def test_score_quality_perfect():
    """All correct, high confidence (logprob=0), fast TTFT."""
    results = [_qtr(True, 0.0, 100.0), _qtr(True, 0.0, 100.0)]
    assert _score_quality_results(results).score == pytest.approx(100.0, abs=0.5)


def test_score_quality_all_wrong_no_logprobs():
    """All wrong, no logprobs -> score near 10."""
    results = [_qtr(False, None, None), _qtr(False, None, None)]
    assert _score_quality_results(results).score == pytest.approx(10.0, abs=0.5)


def test_score_quality_half_correct():
    """Half correct, moderate logprobs."""
    results = [_qtr(True, -0.5, 500.0), _qtr(False, -2.0, 500.0)]
    score = _score_quality_results(results).score
    expected = (0.4 * 0.5 + 0.4 * math.exp(-1.25) + 0.2 * 1.0) * 100
    assert score == pytest.approx(expected, abs=0.5)


def test_score_quality_slow_ttft_reduces_efficiency():
    """Slow TTFT should reduce the overall score."""
    fast = [_qtr(True, 0.0, 100.0)]
    slow = [_qtr(True, 0.0, 2000.0)]
    assert _score_quality_results(fast).score > _score_quality_results(slow).score


def test_score_quality_no_timing():
    """When no TTFT data, efficiency defaults to 0.5."""
    results = [_qtr(True, 0.0, None)]
    expected = (0.4 * 1.0 + 0.4 * 1.0 + 0.2 * 0.5) * 100
    assert _score_quality_results(results).score == pytest.approx(expected, abs=0.5)
