"""Behavioral tests for phases/quality.py: phase_quality.

Covers:
    - n_trials=0 returns None immediately
    - Negative n_trials returns None
    - Server boot failure returns None
    - Completed study (remaining=0) returns cached best trial params
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tps_pro.phases.quality import phase_quality

_Q = "tps_pro.phases.quality"


@pytest.mark.unit
class TestPhaseQualityBehavior:
    """Behavioral tests for phase_quality."""

    def test_zero_trials_returns_none(self):
        """phase_quality with n_trials=0 returns None immediately."""
        ctx = MagicMock()
        result = phase_quality(ctx, n_trials=0)
        assert result is None

    def test_negative_trials_returns_none(self):
        """phase_quality with negative n_trials returns None."""
        ctx = MagicMock()
        result = phase_quality(ctx, n_trials=-5)
        assert result is None

    @patch(f"{_Q}.load_phase_results", return_value=None)
    @patch(f"{_Q}.build_phase_config", return_value={"context": 4096})
    @patch(f"{_Q}.boot_server_with_jinja_recovery")
    def test_server_boot_failure_returns_none(
        self, mock_boot, mock_build, mock_load, make_ctx
    ):
        """When server fails to start, returns None."""
        ctx = make_ctx()
        mock_boot.return_value = (None, "error")

        result = phase_quality(ctx, n_trials=60)

        assert result is None

    @patch(f"{_Q}.clear_param_cache")
    @patch(f"{_Q}.load_phase_results", return_value=None)
    @patch(f"{_Q}.build_phase_config", return_value={"context": 4096})
    @patch(f"{_Q}.measure_quality")
    @patch(f"{_Q}.boot_server_with_jinja_recovery")
    @patch(f"{_Q}.setup_study")
    def test_completed_study_returns_best_params(
        self,
        mock_setup,
        mock_boot,
        mock_quality,
        mock_build,
        mock_load,
        mock_clear,
        make_ctx,
    ):
        """When study has already completed all trials, returns best_trial params."""
        ctx = make_ctx()
        proc = MagicMock()
        proc.load_time_ms = 500.0
        mock_boot.return_value = (proc, "ok")

        quality_result = MagicMock()
        quality_result.score = 80.0
        mock_quality.return_value = quality_result

        mock_study = MagicMock()
        mock_study.best_trial.params = {"temperature": 0.7, "top_p": 0.9}
        mock_study.study_name = "quality_test"
        # remaining=0 means all trials already completed
        mock_setup.return_value = (mock_study, 0, 60)

        result = phase_quality(ctx, n_trials=60)

        assert result is not None
        assert result["best_params"]["temperature"] == 0.7
        assert result["best_params"]["top_p"] == 0.9
