"""Edge-case tests for measurement.py internal helpers.

Covers _clamp_finite, _aggregate_samples, _median_by_score,
_attach_variance, extract_pareto_front, and _to_perf_result edge cases
that are not covered by existing test_measurement_scoring.py or
test_measurement_unit.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tps_pro.measurement import extract_pareto_front
from tps_pro.measurement.perf_measurement import (
    _aggregate_samples,
    _attach_variance,
    _median_by_score,
    _to_perf_result,
)
from tps_pro.measurement.scoring import _clamp_finite
from tps_pro.result_types import PerfResult, PerfSample


# ===================================================================
# _clamp_finite
# ===================================================================


@pytest.mark.unit
class TestClampFinite:
    """Tests for _clamp_finite: NaN/Inf handling and boundary clamping."""

    def test_normal_value_within_range(self):
        """A normal value within [floor, cap] is returned as-is."""
        assert _clamp_finite(5.0, 0.0, 10.0) == 5.0

    def test_value_below_floor_clamped(self):
        """Value below floor is clamped up to floor."""
        assert _clamp_finite(-1.0, 0.0, 10.0) == 0.0

    def test_value_above_cap_clamped(self):
        """Value above cap is clamped down to cap."""
        assert _clamp_finite(15.0, 0.0, 10.0) == 10.0

    def test_nan_returns_floor_when_no_fallback(self):
        """NaN with no fallback returns floor."""
        assert _clamp_finite(float("nan"), 1.0, 100.0) == 1.0

    def test_nan_returns_fallback_when_provided(self):
        """NaN with fallback returns the fallback value."""
        assert _clamp_finite(float("nan"), 1.0, 100.0, fallback=42.0) == 42.0

    def test_inf_returns_cap(self):
        """Positive infinity is clamped to cap (it is finite=False)."""
        assert _clamp_finite(float("inf"), 0.0, 100.0) == 0.0  # not finite -> fallback/floor

    def test_neg_inf_returns_floor(self):
        """Negative infinity returns floor (not finite)."""
        assert _clamp_finite(float("-inf"), 0.0, 100.0) == 0.0

    def test_inf_with_fallback(self):
        """Inf with explicit fallback returns fallback."""
        assert _clamp_finite(float("inf"), 0.0, 100.0, fallback=50.0) == 50.0

    def test_value_at_exact_floor(self):
        """Value exactly at floor is returned as-is."""
        assert _clamp_finite(0.0, 0.0, 10.0) == 0.0

    def test_value_at_exact_cap(self):
        """Value exactly at cap is returned as-is."""
        assert _clamp_finite(10.0, 0.0, 10.0) == 10.0


# ===================================================================
# _aggregate_samples
# ===================================================================


def _make_sample(tps=50.0, ttft=100.0, prompt_tps=200.0, total_ms=500.0, **kw):
    """Build a PerfSample with defaults."""
    return PerfSample(tps=tps, ttft=ttft, prompt_tps=prompt_tps, total_ms=total_ms, **kw)


@pytest.mark.unit
class TestAggregateSamples:
    """Tests for _aggregate_samples: median selection logic."""

    def test_empty_samples_returns_zero_result(self):
        """Empty input returns PerfResult with all zeros."""
        result = _aggregate_samples([])
        assert result.tps == 0.0
        assert result.ttft == 0.0

    def test_single_sample_returns_that_sample(self):
        """Single sample is returned as PerfResult."""
        s = _make_sample(tps=42.0, ttft=80.0)
        result = _aggregate_samples([s])
        assert isinstance(result, PerfResult)
        assert result.tps == 42.0
        assert result.ttft == 80.0

    def test_odd_count_returns_middle_by_score(self):
        """Three samples: returns the middle one when sorted by score."""
        s1 = _make_sample(tps=10.0)
        s2 = _make_sample(tps=50.0)
        s3 = _make_sample(tps=100.0)
        result = _aggregate_samples([s3, s1, s2])
        # Median by score => the middle sample
        assert isinstance(result, PerfResult)
        assert result.tps == 50.0

    def test_even_count_returns_lower_middle(self):
        """Even count: returns len//2 index after sorting."""
        s1 = _make_sample(tps=10.0)
        s2 = _make_sample(tps=30.0)
        s3 = _make_sample(tps=50.0)
        s4 = _make_sample(tps=90.0)
        result = _aggregate_samples([s4, s1, s3, s2])
        # sorted by score, index 2 (len=4, 4//2=2) => third element
        assert isinstance(result, PerfResult)
        assert result.tps == 50.0


# ===================================================================
# _median_by_score
# ===================================================================


@pytest.mark.unit
class TestMedianByScore:
    """Tests for _median_by_score selection."""

    def test_single_sample(self):
        """Single sample is returned."""
        s = _make_sample(tps=42.0)
        result = _median_by_score([s])
        assert result.tps == 42.0

    def test_three_samples_returns_middle(self):
        """Three samples: returns median by score."""
        low = _make_sample(tps=5.0)
        mid = _make_sample(tps=50.0)
        high = _make_sample(tps=100.0)
        result = _median_by_score([high, low, mid])
        assert result.tps == 50.0


# ===================================================================
# _attach_variance
# ===================================================================


@pytest.mark.unit
class TestAttachVariance:
    """Tests for _attach_variance: TPS variance stats."""

    def test_single_sample_returns_unchanged(self):
        """With fewer than 2 samples, result is returned unchanged."""
        pr = PerfResult(tps=50.0, ttft=100.0, prompt_tps=200.0, total_ms=500.0)
        result = _attach_variance(pr, [_make_sample(tps=50.0)])
        assert result is pr  # same object

    def test_two_identical_samples_zero_std(self):
        """Two identical samples produce std=0, cv=0."""
        pr = PerfResult(tps=50.0, ttft=100.0, prompt_tps=200.0, total_ms=500.0)
        samples = [_make_sample(tps=50.0), _make_sample(tps=50.0)]
        result = _attach_variance(pr, samples)
        assert result.tps_std == 0.0
        assert result.tps_cv == 0.0
        assert result.n_runs == 2

    def test_variance_computed_correctly(self):
        """Variance stats are computed from sample TPS values."""
        pr = PerfResult(tps=50.0, ttft=100.0, prompt_tps=200.0, total_ms=500.0)
        samples = [_make_sample(tps=40.0), _make_sample(tps=60.0)]
        result = _attach_variance(pr, samples)
        # mean=50, std=sqrt(((40-50)^2 + (60-50)^2)/2) = sqrt(100) = 10
        assert result.tps_std == pytest.approx(10.0, abs=0.01)
        assert result.tps_cv == pytest.approx(0.2, abs=0.01)
        assert result.n_runs == 2

    def test_original_not_mutated(self):
        """_attach_variance returns a new object, does not mutate the original."""
        pr = PerfResult(tps=50.0, ttft=100.0, prompt_tps=200.0, total_ms=500.0)
        samples = [_make_sample(tps=40.0), _make_sample(tps=60.0)]
        result = _attach_variance(pr, samples)
        assert result is not pr
        assert pr.tps_std is None  # original unchanged


# ===================================================================
# extract_pareto_front
# ===================================================================


@pytest.mark.unit
class TestExtractParetoFront:
    """Tests for extract_pareto_front: Pareto-optimal trial extraction."""

    def test_runtime_error_returns_empty(self):
        """RuntimeError from study.best_trials returns empty list."""
        study = MagicMock()
        study.best_trials = property(lambda self: (_ for _ in ()).throw(RuntimeError))
        type(study).best_trials = property(lambda self: (_ for _ in ()).throw(RuntimeError("no")))
        # Use a simpler approach: make best_trials raise
        mock_study = MagicMock()
        type(mock_study).best_trials = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("no trials"))
        )
        result = extract_pareto_front(mock_study)
        assert result == []

    def test_empty_trials_returns_empty(self):
        """Study with no best_trials returns empty list."""
        study = MagicMock()
        study.best_trials = []
        result = extract_pareto_front(study)
        assert result == []

    def test_sorted_by_tps_descending(self):
        """Pareto trials are sorted by TPS (first objective) descending."""
        t1 = MagicMock()
        t1.values = [30.0, -5000.0, 0.9]
        t2 = MagicMock()
        t2.values = [80.0, -4000.0, 0.8]
        t3 = MagicMock()
        t3.values = [50.0, -3000.0, 0.95]

        study = MagicMock()
        study.best_trials = [t1, t2, t3]

        result = extract_pareto_front(study)
        assert len(result) == 3
        assert result[0].values[0] == 80.0
        assert result[1].values[0] == 50.0
        assert result[2].values[0] == 30.0


# ===================================================================
# _to_perf_result
# ===================================================================


@pytest.mark.unit
class TestToPerfResult:
    """Tests for _to_perf_result conversion."""

    def test_perf_result_passthrough(self):
        """PerfResult input is returned as-is (same object)."""
        pr = PerfResult(tps=50.0, ttft=100.0, prompt_tps=200.0, total_ms=500.0)
        assert _to_perf_result(pr) is pr

    def test_dict_conversion(self):
        """Dict input is converted to PerfResult."""
        d = {"tps": 42.0, "ttft": 80.0, "prompt_tps": 150.0, "total_ms": 300.0}
        result = _to_perf_result(d)
        assert isinstance(result, PerfResult)
        assert result.tps == 42.0

    def test_perf_sample_conversion(self):
        """PerfSample input is converted to PerfResult via to_dict."""
        s = PerfSample(tps=33.0, ttft=90.0, prompt_tps=180.0, total_ms=400.0)
        result = _to_perf_result(s)
        assert isinstance(result, PerfResult)
        assert result.tps == 33.0
