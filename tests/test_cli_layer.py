"""Unit tests for the CLI layer modules: report, wizard, display, args, dashboard.

Covers the previously zero-coverage CLI surface with focused unit tests.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# ===================================================================
# cli/args.py
# ===================================================================


@pytest.mark.unit
class TestParseCliArgs:
    """Tests for cli.args.parse_cli_args."""

    def test_defaults_return_namespace(self):
        """parse_cli_args returns a Namespace with expected defaults."""
        with patch("sys.argv", ["prog"]):
            from tps_pro.cli.args import parse_cli_args

            args = parse_cli_args()
        assert hasattr(args, "server")
        assert hasattr(args, "model")
        assert args.preset == "normal"
        assert args.dry_run is False
        assert args.debug is False
        assert args.port is None

    def test_valid_port(self):
        """A valid --port argument is parsed correctly."""
        with patch("sys.argv", ["prog", "--port", "9090"]):
            from tps_pro.cli.args import parse_cli_args

            args = parse_cli_args()
        assert args.port == 9090

    def test_invalid_port_out_of_range(self):
        """An out-of-range port raises SystemExit via argparse."""
        import argparse

        from tps_pro.cli.args import _valid_port

        with pytest.raises(argparse.ArgumentTypeError):
            _valid_port("99999")

    def test_invalid_port_not_integer(self):
        """A non-integer port raises ArgumentTypeError."""
        import argparse

        from tps_pro.cli.args import _valid_port

        with pytest.raises(argparse.ArgumentTypeError):
            _valid_port("abc")

    def test_quick_preset(self):
        """--quick sets preset to 'quick'."""
        with patch("sys.argv", ["prog", "--quick"]):
            from tps_pro.cli.args import parse_cli_args

            args = parse_cli_args()
        assert args.preset == "quick"

    def test_thorough_preset(self):
        """--thorough sets preset to 'thorough'."""
        with patch("sys.argv", ["prog", "--thorough"]):
            from tps_pro.cli.args import parse_cli_args

            args = parse_cli_args()
        assert args.preset == "thorough"

    def test_dense_flag(self):
        """--dense sets the dense flag."""
        with patch("sys.argv", ["prog", "--dense"]):
            from tps_pro.cli.args import parse_cli_args

            args = parse_cli_args()
        assert args.dense is True

    def test_unknown_args_ignored(self):
        """Unknown arguments are silently ignored (forward compatibility)."""
        with patch("sys.argv", ["prog", "--unknown-flag", "value"]):
            from tps_pro.cli.args import parse_cli_args

            args = parse_cli_args()
        # Should not raise; unknown args are discarded
        assert args.server is None


# ===================================================================
# cli/wizard.py
# ===================================================================


@pytest.mark.unit
class TestNeedsSetup:
    """Tests for wizard.needs_setup."""

    def test_needs_setup_missing_server(self, tmp_path):
        """Returns True when server path is empty."""
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"\x00" * 10)
        fake_cfg = {"server": "", "model": str(model_file)}
        with patch("tps_pro.cli.wizard.get_config", side_effect=lambda k, d="": fake_cfg.get(k, d)):
            from tps_pro.cli.wizard import needs_setup

            assert needs_setup() is True

    def test_needs_setup_missing_model(self, tmp_path):
        """Returns True when model path is empty."""
        server_file = tmp_path / "llama-server"
        server_file.write_bytes(b"\x00" * 10)
        fake_cfg = {"server": str(server_file), "model": ""}
        with patch("tps_pro.cli.wizard.get_config", side_effect=lambda k, d="": fake_cfg.get(k, d)):
            from tps_pro.cli.wizard import needs_setup

            assert needs_setup() is True

    def test_needs_setup_both_exist(self, tmp_path):
        """Returns False when both server and model files exist."""
        server_file = tmp_path / "llama-server"
        server_file.write_bytes(b"\x00" * 10)
        model_file = tmp_path / "model.gguf"
        model_file.write_bytes(b"\x00" * 10)
        fake_cfg = {"server": str(server_file), "model": str(model_file)}
        with patch("tps_pro.cli.wizard.get_config", side_effect=lambda k, d="": fake_cfg.get(k, d)):
            from tps_pro.cli.wizard import needs_setup

            assert needs_setup() is False

    def test_needs_setup_nonexistent_files(self):
        """Returns True when paths are set but files don't exist."""
        fake_cfg = {
            "server": "/nonexistent/llama-server",
            "model": "/nonexistent/model.gguf",
        }
        with patch("tps_pro.cli.wizard.get_config", side_effect=lambda k, d="": fake_cfg.get(k, d)):
            from tps_pro.cli.wizard import needs_setup

            assert needs_setup() is True


# ===================================================================
# cli/wizard.py — _safe_int and _resolve_architecture helpers
# ===================================================================


@pytest.mark.unit
class TestWizardHelpers:
    """Tests for wizard helper functions."""

    def test_safe_int_valid(self):
        from tps_pro.cli.wizard import _safe_int

        assert _safe_int("42", 10) == 42

    def test_safe_int_empty(self):
        from tps_pro.cli.wizard import _safe_int

        assert _safe_int("", 10) == 10

    def test_safe_int_invalid(self):
        from tps_pro.cli.wizard import _safe_int

        assert _safe_int("abc", 10) == 10


# ===================================================================
# cli/display.py
# ===================================================================


@pytest.mark.unit
class TestViewResults:
    """Tests for display.view_results with empty directory."""

    @patch("tps_pro.cli.display.get_model_results", return_value=[])
    @patch("tps_pro.cli.display._migrate_legacy_results")
    def test_view_results_empty_dir(self, mock_migrate, mock_get_models, capsys):
        """view_results prints 'No optimization results' when no models found."""
        from tps_pro.cli.display import view_results

        view_results()
        captured = capsys.readouterr()
        assert "No optimization results found" in captured.out

    @patch("tps_pro.cli.display.get_model_results", side_effect=EOFError)
    @patch("tps_pro.cli.display._migrate_legacy_results")
    def test_view_results_eof(self, mock_migrate, mock_get_models):
        """view_results handles EOFError gracefully."""
        from tps_pro.cli.display import view_results

        # Should not raise
        view_results()


# ===================================================================
# cli/report.py
# ===================================================================


@pytest.mark.unit
class TestGenerateOptimizedCommand:
    """Tests for services_command.generate_optimized_command."""

    @patch("tps_pro.cli.services_command._format_command", return_value="/usr/bin/server -m model.gguf")
    @patch("tps_pro.cli.services_command._build_command_parts", return_value=["/usr/bin/server"])
    @patch(
        "tps_pro.cli.services_command._merge_phase_results",
        return_value={"n_gpu_layers": 99},
    )
    @patch("tps_pro.cli.services_command.ensure_results_dir")
    def test_generate_optimized_command_returns_string(
        self, mock_ensure, mock_merge, mock_build, mock_format, tmp_path
    ):
        """generate_optimized_command returns a string when results exist."""
        from tps_pro.cli.services_command import generate_optimized_command

        mock_ctx = SimpleNamespace(
            results_dir=tmp_path,
            server_path=Path("/usr/bin/server"),
            model_path=Path("/models/model.gguf"),
            chat_template_path=Path(""),
            port=8090,
            no_jinja=False,
            expert_override_key="",
            default_experts=8,
        )

        result = generate_optimized_command(mock_ctx)
        assert isinstance(result, str)
        assert len(result) > 0

    @patch("tps_pro.cli.services_command._merge_phase_results", return_value={})
    def test_generate_optimized_command_no_results(self, mock_merge, tmp_path):
        """generate_optimized_command returns None when no results found."""
        from tps_pro.cli.services_command import generate_optimized_command

        mock_ctx = SimpleNamespace(
            results_dir=tmp_path,
            server_path=Path("/usr/bin/server"),
            model_path=Path("/models/model.gguf"),
            chat_template_path=Path(""),
            port=8090,
        )

        result = generate_optimized_command(mock_ctx)
        assert result is None


@pytest.mark.unit
class TestBuildPhaseTableRows:
    """Tests for report._build_phase_table_rows."""

    def test_escapes_html_values(self):
        """Phase names and scores are HTML-escaped."""
        from tps_pro.cli.report import _build_phase_table_rows

        phases = {
            "<script>alert(1)</script>": {
                "best_tps": 42.0,
                "all_trials": [1, 2],
                "duration_minutes": 5.0,
                "beat_baseline": True,
            }
        }
        html_out = _build_phase_table_rows(phases)
        # The script tag should be escaped
        assert "&lt;script&gt;" in html_out
        assert "<script>" not in html_out

    def test_skips_context_sweep(self):
        """context_sweep phase is not included in table rows."""
        from tps_pro.cli.report import _build_phase_table_rows

        phases = {
            "context_sweep": {
                "best_tps": 10.0,
                "all_trials": [],
                "duration_minutes": 1.0,
            },
            "gpu": {
                "best_tps": 50.0,
                "all_trials": [1],
                "duration_minutes": 2.0,
                "beat_baseline": True,
            },
        }
        html_out = _build_phase_table_rows(phases)
        assert "gpu" in html_out
        # context_sweep row is skipped
        assert "context_sweep" not in html_out


@pytest.mark.unit
class TestGenerateHtmlReport:
    """Tests for report.generate_html_report."""

    @patch("tps_pro.cli.report.ctx")
    @patch("tps_pro.cli.report._load_report_phases")
    def test_returns_html_with_escaped_model_name(
        self, mock_load, mock_ctx, tmp_path
    ):
        """generate_html_report returns path and escapes model name."""
        from tps_pro.cli.report import generate_html_report

        mock_ctx.results_dir = tmp_path
        mock_ctx.model_path = SimpleNamespace(name="<evil>.gguf")
        mock_load.return_value = {
            "gpu": {
                "best_tps": 50.0,
                "all_trials": [],
                "duration_minutes": 1.0,
                "beat_baseline": True,
            }
        }

        result = generate_html_report(
            results_dir=tmp_path,
            model_name="<evil>",
            gpus=[{"name": "RTX 4090", "vram_total_gb": 24}],
        )
        assert result is not None
        report_path = Path(result)
        assert report_path.exists()

        content = report_path.read_text(encoding="utf-8")
        assert "&lt;evil&gt;" in content
        assert "<evil>" not in content.split("<style>")[0]  # Not in title area

    @patch("tps_pro.cli.report.ctx")
    @patch("tps_pro.cli.report._load_report_phases", return_value={})
    def test_returns_none_when_no_phases(self, mock_load, mock_ctx, tmp_path):
        """generate_html_report returns None when no phases found."""
        from tps_pro.cli.report import generate_html_report

        mock_ctx.results_dir = tmp_path
        mock_ctx.model_path = SimpleNamespace(name="model.gguf")

        result = generate_html_report(results_dir=tmp_path, gpus=[])
        assert result is None


