"""Tests for evals/quality_gate.py — quality gate using token uncertainty.

Direct imports from the target module to satisfy coverage detection.
Tests measure_quality_gate with mocked measurement dependency.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tps_pro.evals.quality_gate import measure_quality_gate
from tps_pro.result_types import TokenUncertaintyResult


def _make_ctx(**overrides):
    """Build a minimal ctx for quality gate tests."""
    from _ctx_factory import make_ctx_from_defaults

    return make_ctx_from_defaults(**overrides)


def _make_uncertainty(uncertain_count=5, total_tokens=100, tail_avg=-1.5):
    """Build a TokenUncertaintyResult."""
    return TokenUncertaintyResult(
        uncertain_count=uncertain_count,
        total_tokens=total_tokens,
        tail_avg=tail_avg,
    )


@pytest.mark.unit
class TestMeasureQualityGate:
    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_baseline_run_returns_one(self, mock_measure):
        """Baseline run should store metrics and return 1.0."""
        mock_measure.return_value = _make_uncertainty()
        ctx = _make_ctx()
        result = measure_quality_gate(ctx, is_baseline=True)
        assert result == 1.0
        assert ctx.quality_baseline is not None

    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_failed_measurement_baseline_returns_one(self, mock_measure):
        """Failed measurement on baseline should return 1.0."""
        mock_measure.return_value = None
        ctx = _make_ctx()
        result = measure_quality_gate(ctx, is_baseline=True)
        assert result == 1.0

    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_failed_measurement_trial_returns_penalty(self, mock_measure):
        """Failed measurement on trial should return cliff penalty."""
        mock_measure.return_value = None
        baseline = _make_uncertainty()
        ctx = _make_ctx(quality_baseline=baseline)
        result = measure_quality_gate(ctx, is_baseline=False)
        assert result < 1.0

    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_no_degradation_returns_one(self, mock_measure):
        """Same metrics as baseline should return 1.0."""
        baseline = _make_uncertainty(uncertain_count=5, total_tokens=100, tail_avg=-1.5)
        mock_measure.return_value = _make_uncertainty(
            uncertain_count=5, total_tokens=100, tail_avg=-1.5
        )
        ctx = _make_ctx(quality_baseline=baseline)
        result = measure_quality_gate(ctx, is_baseline=False)
        assert result == pytest.approx(1.0)

    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_severe_degradation_low_factor(self, mock_measure):
        """Severe quality degradation should return a low factor."""
        baseline = _make_uncertainty(uncertain_count=5, total_tokens=100, tail_avg=-1.5)
        mock_measure.return_value = _make_uncertainty(
            uncertain_count=50, total_tokens=100, tail_avg=-5.0
        )
        ctx = _make_ctx(quality_baseline=baseline)
        result = measure_quality_gate(ctx, is_baseline=False)
        assert result < 0.5

    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_dict_baseline_converted(self, mock_measure):
        """Dict baseline should be auto-converted to TokenUncertaintyResult."""
        baseline_dict = {"uncertain_count": 5, "total_tokens": 100, "tail_avg": -1.5}
        mock_measure.return_value = _make_uncertainty(
            uncertain_count=5, total_tokens=100, tail_avg=-1.5
        )
        ctx = _make_ctx(quality_baseline=baseline_dict)
        result = measure_quality_gate(ctx, is_baseline=False)
        assert isinstance(result, float)
