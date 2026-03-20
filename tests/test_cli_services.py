"""Tests for CLI services modules — services_config, services_pipeline, services_command.

Tests cover:
    - cycle_preset cycles through presets
    - get_toggle_states returns correct shape
    - get_pipeline_progress returns list with mock ctx
    - generate_optimized_command returns string or None
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.mark.unit
class TestServicesConfig:
    @patch("tps_pro.cli.services_config.set_config")
    def test_cycle_preset(self, mock_set_config):
        """cycle_preset should cycle through quick/normal/thorough."""
        from tps_pro.cli.services_config import cycle_preset

        config = {"preset": "normal"}
        result = cycle_preset(config)
        assert result == "thorough"
        mock_set_config.assert_called_once_with("preset", "thorough")

    @patch("tps_pro.cli.services_config.set_config")
    def test_cycle_preset_wraps_around(self, mock_set_config):
        """cycle_preset wraps from thorough back to quick."""
        from tps_pro.cli.services_config import cycle_preset

        config = {"preset": "thorough"}
        result = cycle_preset(config)
        assert result == "quick"

    @patch("tps_pro.cli.services_config.get_config", return_value=False)
    def test_get_toggle_states(self, mock_get_config, make_ctx):
        """get_toggle_states returns a dict of boolean values."""
        from tps_pro.cli.services_config import get_toggle_states

        ctx = make_ctx()
        config = {}
        states = get_toggle_states(ctx, config)

        assert isinstance(states, dict)
        expected_keys = {
            "pareto",
            "debug",
            "no_jinja",
            "no_bench",
            "fail_fast",
            "skip_quality",
            "interactive",
        }
        assert set(states.keys()) == expected_keys
        assert all(isinstance(v, bool) for v in states.values())


@pytest.mark.unit
class TestServicesPipeline:
    @patch("tps_pro.cli.services_pipeline.load_phase_results", return_value=None)
    @patch("tps_pro.cli.services_pipeline.get_config", return_value=False)
    def test_get_pipeline_progress_returns_list(self, mock_config, mock_load, make_ctx):
        """get_pipeline_progress should return a list of PhaseProgress objects."""
        from tps_pro.cli.services_pipeline import get_pipeline_progress

        ctx = make_ctx(optuna_db="sqlite:///nonexistent.db")
        progress = get_pipeline_progress(ctx)

        assert isinstance(progress, list)
        assert len(progress) > 0
        # Each item should have status attribute
        for p in progress:
            assert hasattr(p, "status")
            assert p.status in ("done", "partial", "pending")


@pytest.mark.unit
class TestServicesCommand:
    @patch("tps_pro.cli.services_command._merge_phase_results", return_value={})
    def test_generate_optimized_command_no_results(self, mock_merge, make_ctx):
        """generate_optimized_command returns None when no results exist."""
        from tps_pro.cli.services_command import generate_optimized_command

        ctx = make_ctx(results_dir=Path("/tmp/nonexistent"))
        result = generate_optimized_command(ctx)
        assert result is None

    @patch("tps_pro.cli.services_command.ensure_results_dir")
    @patch("tps_pro.cli.services_command._merge_phase_results")
    def test_generate_optimized_command_returns_string(
        self, mock_merge, mock_ensure, make_ctx, tmp_path
    ):
        """generate_optimized_command returns a string when results exist."""
        from tps_pro.cli.services_command import generate_optimized_command

        mock_merge.return_value = {"n_gpu_layers": 99, "threads": 8, "context": 4096}
        ctx = make_ctx(
            results_dir=tmp_path,
            server_path=Path("/tmp/llama-server"),
            model_path=SimpleNamespace(name="test.gguf"),
            chat_template_path=Path(""),
            port=8090,
            no_jinja=False,
            expert_override_key="",
            default_experts=8,
        )

        result = generate_optimized_command(ctx)
        assert isinstance(result, str)
        assert "llama-server" in result
