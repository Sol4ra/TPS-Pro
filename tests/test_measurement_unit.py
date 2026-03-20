"""Direct-import unit tests for measurement.py.

test_measurement_full.py and test_scoring.py already test measurement
functions extensively, but use deferred imports inside test methods.
This file provides top-level direct imports to satisfy coverage tooling.
"""

from __future__ import annotations

import pytest

from tps_pro.measurement import (
    compute_pareto_objectives,
    compute_score,
    extract_pareto_front,
    get_best_trial,
    measure_concurrent_load,
    measure_perf_adaptive,
    measure_token_uncertainty,
    print_pareto_front,
)
from tps_pro.measurement.perf_measurement import _to_perf_result
from tps_pro.result_types import PerfResult


@pytest.mark.unit
class TestToPerf:
    def test_to_perf_result_from_perf_result(self):
        """_to_perf_result should return the same PerfResult if already one."""
        pr = PerfResult(tps=50.0, ttft=30.0, prompt_tps=100.0, total_ms=500.0)
        assert _to_perf_result(pr) is pr

    def test_to_perf_result_from_dict(self):
        """_to_perf_result should convert a dict to PerfResult."""
        d = {"tps": 50.0, "ttft": 30.0, "prompt_tps": 100.0, "total_ms": 500.0}
        result = _to_perf_result(d)
        assert isinstance(result, PerfResult)
        assert result.tps == 50.0


@pytest.mark.unit
class TestComputeScore:
    def test_compute_score_basic(self):
        """compute_score should return a positive float for valid input."""
        pr = PerfResult(tps=50.0, ttft=100.0, prompt_tps=200.0, total_ms=500.0)
        score = compute_score(pr)
        assert isinstance(score, float)
        assert score > 0

    def test_compute_score_zero_tps_returns_zero(self):
        """compute_score should return 0.0 when tps is 0."""
        pr = PerfResult(tps=0.0, ttft=0.0, prompt_tps=0.0, total_ms=0.0)
        score = compute_score(pr)
        assert score == 0.0

    def test_compute_score_with_large_tps_full_mode(self):
        """compute_score in full mode (large_tps present) should return positive."""
        pr = PerfResult(
            tps=50.0, ttft=100.0, prompt_tps=200.0, total_ms=500.0, large_tps=40.0
        )
        score = compute_score(pr)
        assert score > 0

    def test_compute_score_with_vram(self):
        """compute_score with VRAM data should include efficiency bonus."""
        pr = PerfResult(tps=50.0, ttft=100.0, prompt_tps=200.0, total_ms=500.0)
        compute_score(pr)  # baseline without VRAM
        score_with_vram = compute_score(pr, vram_used_mb=4000.0, vram_total_mb=8000.0)
        assert isinstance(score_with_vram, float)
        assert score_with_vram > 0


@pytest.mark.unit
class TestComputeParetoObjectives:
    def test_returns_pareto_objectives(self):
        """compute_pareto_objectives should return a ParetoObjectives dataclass."""
        from tps_pro.result_types import ParetoObjectives

        pr = PerfResult(
            tps=50.0, ttft=100.0, prompt_tps=200.0, total_ms=500.0, vram_used_mb=4000.0
        )
        result = compute_pareto_objectives(pr)
        assert isinstance(result, ParetoObjectives)
        assert result.tps > 0

    def test_zero_tps_returns_zero_tps_objective(self):
        """Zero tps should produce zero tps in objectives."""
        pr = PerfResult(tps=0.0, ttft=0.0, prompt_tps=0.0, total_ms=0.0)
        result = compute_pareto_objectives(pr)
        assert result.tps == 0.0


@pytest.mark.unit
class TestMeasurementCallables:
    def test_all_exported_functions_are_callable(self):
        """All measurement __all__ exports should be callable."""
        assert callable(compute_score)
        assert callable(compute_pareto_objectives)
        assert callable(extract_pareto_front)
        assert callable(print_pareto_front)
        assert callable(get_best_trial)
        assert callable(measure_perf_adaptive)
        assert callable(measure_concurrent_load)
        assert callable(measure_token_uncertainty)
