"""Tests for phases/quality.py — quality/sampling phase."""

from __future__ import annotations

import pytest

from tps_pro.phases.quality import phase_quality


@pytest.mark.unit
class TestPhaseQuality:
    def test_phase_quality_is_callable(self):
        """phase_quality should be a callable function."""
        assert callable(phase_quality)

    def test_phase_quality_zero_trials_returns_none(self):
        """phase_quality with n_trials=0 should return None immediately."""
        from unittest.mock import MagicMock

        ctx = MagicMock()
        result = phase_quality(ctx, n_trials=0)
        assert result is None

    def test_phase_quality_negative_trials_returns_none(self):
        """phase_quality with negative n_trials should return None."""
        from unittest.mock import MagicMock

        ctx = MagicMock()
        result = phase_quality(ctx, n_trials=-5)
        assert result is None
