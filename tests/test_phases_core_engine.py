"""Tests for phases/core_engine.py — behavioral tests beyond smoke-check.

Tests cover:
    - Baseline failure returns None
    - Layer 1 A/B produces winners dict
    - _build_ab_flags respects skip_flags
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tps_pro.phases.core_engine import (
    _build_ab_flags,
    _layer1_ab_sweeps,
    phase_core_engine,
)


@pytest.mark.unit
class TestPhasesCoreEngine:
    def test_phase_core_engine_is_callable(self):
        """phase_core_engine is importable and callable."""
        assert callable(phase_core_engine)

    @patch("tps_pro.phases.core_engine.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.core_engine.kill_server")
    def test_baseline_failure_returns_none(self, mock_kill, mock_boot, make_ctx):
        """When baseline server fails to start, phase returns None."""
        ctx = make_ctx(fail_fast=False, skip_flags=set())
        mock_boot.return_value = (None, "error")

        result = phase_core_engine(ctx, n_trials=10, base_config={"context": 4096})
        assert result is None

    def test_build_ab_flags_respects_skip_flags(self):
        """_build_ab_flags should pre-determine winners for skipped flags."""
        skip_flags = {"op_offload", "repack"}
        winners, ab_flags = _build_ab_flags(skip_flags)
        assert winners["op_offload"] is False  # MoE default
        assert winners["repack"] is True  # ON by default
        # Skipped flags should NOT appear in ab_flags
        ab_names = [name for name, _, _ in ab_flags]
        assert "op_offload" not in ab_names
        assert "repack" not in ab_names
        # prio should still be there
        assert "prio" in ab_names

    @patch("tps_pro.phases.core_engine.measure_perf_adaptive")
    @patch("tps_pro.phases.core_engine.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.core_engine.kill_server")
    def test_layer1_ab_produces_winners_dict(
        self, mock_kill, mock_boot, mock_measure, make_ctx
    ):
        """Layer 1 A/B should return a dict of flag winners."""
        ctx = make_ctx(skip_flags=set())
        perf = SimpleNamespace(
            tps=50.0,
            ttft=500.0,
            prompt_tps=300.0,
            total_ms=1000.0,
            vram_used_mb=4096.0,
            vram_total_mb=8192.0,
            large_tps=None,
            concurrent_total_tps=None,
            concurrent_users=None,
            quality_factor=None,
            tps_std=1.0,
            tps_cv=0.02,
            n_runs=3,
            concurrent_avg_tps=None,
            concurrent_avg_ttft=None,
            concurrent_avg_wall_ms=None,
            concurrent_max_wall_ms=None,
            concurrent_success_rate=None,
            load_time_ms=None,
        )
        mock_boot.return_value = (MagicMock(), "ok")
        mock_measure.return_value = (perf, False)

        def score_fn(p):
            return p.tps

        winners = _layer1_ab_sweeps(
            ctx, {"context": 4096}, score_fn, baseline=perf, baseline_score=50.0
        )
        assert isinstance(winners, dict)
        # Should always include no_mmap and mlock (hardcoded in _build_ab_flags)
        assert "no_mmap" in winners
        assert "mlock" in winners
