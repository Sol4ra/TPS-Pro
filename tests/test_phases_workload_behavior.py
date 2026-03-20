"""Behavioral tests for phases/workload.py: phase_workload_sim.

Covers:
    - Server boot failure returns None
    - Hot-cache TTFT measurement with successful HTTP responses
    - Fallback to naked_engine when base_config is None
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tps_pro.phases.workload import phase_workload_sim


_WL = "tps_pro.phases.workload"


@pytest.mark.unit
class TestPhaseWorkloadSimBehavior:
    """Behavioral tests for phase_workload_sim."""

    @patch(f"{_WL}.kill_server")
    @patch(f"{_WL}.boot_server_with_jinja_recovery")
    def test_server_boot_failure_returns_none(self, mock_boot, mock_kill, make_ctx):
        """When server fails to start, returns None."""
        ctx = make_ctx()
        mock_boot.return_value = (None, "error")

        result = phase_workload_sim(ctx, base_config={"context": 4096})

        assert result is None
        mock_kill.assert_called_once_with(ctx)

    @patch(f"{_WL}.save_phase_results")
    @patch(f"{_WL}.measure_concurrent_load", return_value=None)
    @patch(f"{_WL}.kill_server")
    @patch(f"{_WL}.boot_server_with_jinja_recovery")
    def test_successful_hot_cache_measurement(
        self, mock_boot, mock_kill, mock_concurrent, mock_save, make_ctx
    ):
        """Successful server boot and HTTP responses produce valid results."""
        ctx = make_ctx()
        proc = MagicMock()
        proc.load_time_ms = 1500.0
        mock_boot.return_value = (proc, "ok")

        # Mock HTTP responses for cold + hot cache TTFTs
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.json.return_value = {
            "timings": {"prompt_ms": 50.0, "predicted_per_second": 80.0}
        }
        ctx.http.post.return_value = response_mock

        result = phase_workload_sim(ctx, base_config={"context": 4096})

        assert result is not None
        assert "best_params" in result
        # HTTP should have been called for cold + 3 hot requests = 4 total
        assert ctx.http.post.call_count >= 4

    @patch(f"{_WL}.kill_server")
    @patch(f"{_WL}.boot_server_with_jinja_recovery")
    def test_none_base_config_uses_naked_engine(self, mock_boot, mock_kill, make_ctx):
        """When base_config is None, falls back to ctx.naked_engine."""
        ctx = make_ctx(naked_engine={"threads": 4, "context": 2048})
        mock_boot.return_value = (None, "error")

        phase_workload_sim(ctx, base_config=None)

        # Verify boot was called (it uses the fallback config internally)
        mock_boot.assert_called_once()
        call_args = mock_boot.call_args
        config = call_args[0][1]  # second positional arg is config
        assert config["context"] == 2048
        assert config["cache_reuse"] == 256

    @patch(f"{_WL}.save_phase_results")
    @patch(f"{_WL}.measure_concurrent_load", return_value=None)
    @patch(f"{_WL}.kill_server")
    @patch(f"{_WL}.boot_server_with_jinja_recovery")
    def test_http_failure_graceful(
        self, mock_boot, mock_kill, mock_concurrent, mock_save, make_ctx
    ):
        """HTTP errors during measurement are handled gracefully."""
        import requests

        ctx = make_ctx()
        proc = MagicMock()
        proc.load_time_ms = None
        mock_boot.return_value = (proc, "ok")
        ctx.http.post.side_effect = requests.RequestException("connection refused")

        result = phase_workload_sim(ctx, base_config={"context": 4096})

        # Should still return a result (with no TTFT data)
        assert result is not None
