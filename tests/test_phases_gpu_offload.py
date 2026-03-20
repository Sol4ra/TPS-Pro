"""Tests for phases/gpu_offload.py — behavioral tests.

Tests cover:
    - Single-layer skip (max_ngl <= 1 returns early)
    - Cached result return (existing results loaded from disk)
    - MoE skip path
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tps_pro.phases.gpu_offload import phase_gpu_offload


@pytest.mark.unit
class TestPhasesGpuOffload:
    def test_phase_gpu_offload_is_callable(self):
        """phase_gpu_offload is importable and callable."""
        assert callable(phase_gpu_offload)

    @patch("tps_pro.phases.gpu_offload.save_phase_results")
    @patch("tps_pro.phases.gpu_offload.load_phase_results", return_value=None)
    def test_single_layer_skip(self, mock_load, mock_save, make_ctx):
        """When max_gpu_layers <= 1, phase returns immediately without sweep."""
        ctx = make_ctx(max_gpu_layers=1, is_moe=False)

        result = phase_gpu_offload(ctx)
        assert result is not None
        assert result["best_params"]["n_gpu_layers"] == 1
        mock_save.assert_called_once()

    @patch("tps_pro.phases.gpu_offload.load_phase_results")
    def test_cached_result_return(self, mock_load, make_ctx):
        """When existing results exist, phase returns cached value."""
        ctx = make_ctx(max_gpu_layers=99, is_moe=False)
        mock_load.return_value = {"best_ngl": 42}

        result = phase_gpu_offload(ctx)
        assert result is not None
        assert result["best_params"]["n_gpu_layers"] == 42
        assert ctx.default_gpu_layers == 42

    @patch("tps_pro.phases.gpu_offload.save_phase_results")
    @patch("tps_pro.phases.gpu_offload.update_naked_engine")
    @patch("tps_pro.phases.gpu_offload.load_phase_results", return_value=None)
    def test_moe_skips_sweep(self, mock_load, mock_update, mock_save, make_ctx):
        """MoE models skip GPU sweep and use all layers on GPU."""
        ctx = make_ctx(max_gpu_layers=99, is_moe=True)

        result = phase_gpu_offload(ctx)
        assert result is not None
        assert result["best_params"]["n_gpu_layers"] == 99
