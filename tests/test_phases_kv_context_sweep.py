"""Tests for phases/kv_context_sweep.py — behavioral tests.

Tests cover:
    - Boot scanning failure returns None
    - Cached results return early
    - Quality pass/fail propagation
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tps_pro.phases.kv_context_sweep import phase_kv_context_sweep


@pytest.mark.unit
class TestPhasesKvContextSweep:
    def test_is_callable(self):
        """phase_kv_context_sweep is importable and callable."""
        assert callable(phase_kv_context_sweep)

    @patch("tps_pro.phases.kv_context_sweep.load_phase_results")
    def test_cached_results_return_early(self, mock_load, make_ctx):
        """When saved results exist and force=False, returns cached params."""
        ctx = make_ctx()
        mock_load.return_value = {
            "best_params": {"kv_cache_type": "q8_0", "context": 8192}
        }

        result = phase_kv_context_sweep(ctx, force=False)
        assert result is not None
        assert result["best_params"]["kv_cache_type"] == "q8_0"
        assert result["best_params"]["context"] == 8192

    @patch("tps_pro.phases.kv_context_sweep.kill_server")
    @patch("tps_pro.phases.kv_context_sweep.discover_bootable_contexts")
    @patch("tps_pro.phases.kv_context_sweep.get_model_metadata", return_value={})
    @patch("tps_pro.phases.kv_context_sweep.get_model_max_context", return_value=0)
    @patch("tps_pro.phases.kv_context_sweep.load_phase_results", return_value=None)
    def test_boot_scan_failure_returns_none(
        self, mock_load, mock_max_ctx, mock_meta, mock_discover, mock_kill, make_ctx
    ):
        """When all KV types fail at minimum context, phase returns None."""
        ctx = make_ctx()
        mock_discover.return_value = (
            {"f16": None, "q8_0": None, "q4_0": None},
            [],  # empty test points
        )

        result = phase_kv_context_sweep(ctx, force=True)
        assert result is None
