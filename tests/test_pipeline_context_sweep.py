"""Tests for pipeline.py: _select_context_size helper.

Covers viable context selection, peak TPS thresholds, empty sweep results,
and ctx.naked_engine mutation via update_naked_engine.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tps_pro.pipeline import _select_context_size


@pytest.mark.unit
class TestProcessContextSweep:
    """Tests for _select_context_size: context selection from sweep results."""

    def _make_ctx(self):
        """Build a minimal ctx mock with naked_engine."""
        ctx = SimpleNamespace(naked_engine={"threads": 8, "context": 4096})
        return ctx

    def test_selects_largest_viable_context(self):
        """Picks the largest context achieving >= 80% of peak TPS."""
        ctx = self._make_ctx()
        sweep = {
            "2048": {"tps": 80.0, "fits": True},
            "4096": {"tps": 100.0, "fits": True},
            "8192": {"tps": 85.0, "fits": True},
            "16384": {"tps": 50.0, "fits": True},
        }
        with patch("tps_pro.pipeline.update_naked_engine"):
            result = _select_context_size(ctx, sweep, {"threads": 8})
        # Peak is 100 t/s, 80% threshold = 80 t/s
        # 8192 (85 t/s) and 4096 (100 t/s) and 2048 (80 t/s) all qualify
        # Largest qualifying is 8192
        assert result["context"] == 8192

    def test_no_viable_contexts_returns_base(self):
        """When no context fits, returns base_config unchanged."""
        ctx = self._make_ctx()
        sweep = {
            "2048": {"tps": 0, "fits": False},
            "4096": {"tps": 0, "fits": False},
        }
        base = {"threads": 8}
        result = _select_context_size(ctx, sweep, base)
        assert result is base  # same object, no changes

    def test_all_fits_false_returns_base(self):
        """When everything fails to fit, returns base unchanged."""
        ctx = self._make_ctx()
        sweep = {
            "4096": {"tps": 50.0, "fits": False},
            "8192": {"tps": 30.0, "fits": False},
        }
        base = {"threads": 8, "context": 4096}
        result = _select_context_size(ctx, sweep, base)
        assert result is base

    def test_zero_peak_tps_returns_base(self):
        """When all fitting contexts have 0 TPS, returns base."""
        ctx = self._make_ctx()
        sweep = {
            "4096": {"tps": 0.0, "fits": True},
        }
        base = {"threads": 8}
        result = _select_context_size(ctx, sweep, base)
        assert result is base

    def test_single_viable_context_selected(self):
        """Single viable context is selected."""
        ctx = self._make_ctx()
        sweep = {
            "4096": {"tps": 100.0, "fits": True},
            "8192": {"tps": 10.0, "fits": True},  # below 80% threshold
        }
        with patch("tps_pro.pipeline.update_naked_engine"):
            result = _select_context_size(ctx, sweep, {"threads": 8})
        assert result["context"] == 4096

    def test_base_config_not_mutated(self):
        """Original base_config dict is not mutated."""
        ctx = self._make_ctx()
        sweep = {
            "4096": {"tps": 100.0, "fits": True},
        }
        base = {"threads": 8}
        original = dict(base)
        with patch("tps_pro.pipeline.update_naked_engine"):
            _select_context_size(ctx, sweep, base)
        assert base == original  # not mutated

    def test_calls_update_naked_engine(self):
        """update_naked_engine is called with the selected context."""
        ctx = self._make_ctx()
        sweep = {
            "4096": {"tps": 100.0, "fits": True},
        }
        with patch("tps_pro.pipeline.update_naked_engine") as mock_update:
            _select_context_size(ctx, sweep, {"threads": 8})
        mock_update.assert_called_once_with(ctx, context=4096)
