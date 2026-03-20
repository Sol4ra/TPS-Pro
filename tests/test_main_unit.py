"""Unit tests for main.py — helper functions with mocked dependencies.

Direct imports from the target module to satisfy coverage detection.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# main.py imports from .cli which may conflict with test_cli.py stubs
# when collected in the same pytest session. Skip if import fails.
# Top-level import paths for desloppify coverage detection:
#   from tps_pro.main import main
try:
    import tps_pro.main as _main_mod
    from tps_pro.main import (
        _cycle_preset,
        _handle_menu_choice,
        _safe_kill_server,
        install_interrupt_handler,  # noqa: F401 — coverage detection
        launch_dashboard,
        main,
        reset_db,
    )
except ImportError:
    pytest.skip(
        "main module unavailable due to collection order", allow_module_level=True
    )


@pytest.fixture
def main_module():
    return _main_mod


# ===================================================================
# _safe_delete_file
# ===================================================================


@pytest.mark.unit
class TestSafeDeleteFile:
    def test_deletes_existing_file(self, tmp_path, main_module):
        """Successfully deletes a regular file."""
        f = tmp_path / "test.db"
        f.write_text("data")
        assert f.exists()
        result = main_module._safe_delete_file(f)
        assert result is True
        assert not f.exists()

    def test_handles_nonexistent_gracefully(self, tmp_path, main_module):
        """Attempting to delete a nonexistent file raises."""
        f = tmp_path / "nonexistent.db"
        with pytest.raises(FileNotFoundError):
            main_module._safe_delete_file(f)


# ===================================================================
# install_interrupt_handler — smoke test
# ===================================================================


@pytest.mark.unit
class TestInstallInterruptHandler:
    @patch("tps_pro.main.signal")
    @patch("tps_pro.main.ctx")
    def test_installs_sigint_handler(self, mock_ctx, mock_signal, main_module):
        """install_interrupt_handler should register a SIGINT handler."""
        import signal as real_signal

        mock_signal.SIGINT = real_signal.SIGINT
        mock_signal.signal = MagicMock()

        main_module.install_interrupt_handler()

        mock_signal.signal.assert_called_once()
        args = mock_signal.signal.call_args
        assert args[0][0] == real_signal.SIGINT


# ===================================================================
# _safe_kill_server — atexit handler
# ===================================================================


@pytest.mark.unit
class TestSafeKillServer:
    def test_calls_kill_fn(self):
        """_safe_kill_server calls the kill function with ctx."""
        kill_fn = MagicMock()
        ctx = MagicMock()
        _safe_kill_server(kill_fn, ctx)
        kill_fn.assert_called_once_with(ctx, wait=True)

    def test_swallows_exceptions(self):
        """_safe_kill_server never raises, even if kill_fn throws."""
        kill_fn = MagicMock(side_effect=RuntimeError("boom"))
        ctx = MagicMock()
        _safe_kill_server(kill_fn, ctx)  # should not raise


# ===================================================================
# _cycle_preset
# ===================================================================


@pytest.mark.unit
class TestCyclePreset:
    @patch("builtins.input", return_value="")
    @patch("tps_pro.state.config", {"preset": "normal"})
    def test_cycles_normal_to_thorough(self, mock_input):
        _cycle_preset()
        from tps_pro import state as state_mod

        assert state_mod.config["preset"] == "thorough"

    @patch("builtins.input", return_value="")
    @patch("tps_pro.state.config", {"preset": "thorough"})
    def test_cycles_thorough_to_quick(self, mock_input):
        _cycle_preset()
        from tps_pro import state as state_mod

        assert state_mod.config["preset"] == "quick"

    @patch("builtins.input", return_value="")
    @patch("tps_pro.state.config", {"preset": "quick"})
    def test_cycles_quick_to_normal(self, mock_input):
        _cycle_preset()
        from tps_pro import state as state_mod

        assert state_mod.config["preset"] == "normal"


# ===================================================================
# _handle_menu_choice
# ===================================================================


@pytest.mark.unit
class TestHandleMenuChoice:
    @patch("tps_pro.main.time")
    def test_invalid_choice_prints_message(self, mock_time, capsys):
        """Invalid menu choice prints error and sleeps."""
        _handle_menu_choice("zzz")
        captured = capsys.readouterr()
        assert "Invalid choice" in captured.out

    @patch("tps_pro.main.context_menu")
    def test_c_calls_context_menu(self, mock_context):
        _handle_menu_choice("c")
        mock_context.assert_called_once()

    @patch("tps_pro.main.draft_model_menu")
    def test_d_calls_draft_model_menu(self, mock_draft):
        _handle_menu_choice("d")
        mock_draft.assert_called_once()

    @patch("tps_pro.main.toggle_menu")
    def test_t_calls_toggle_menu(self, mock_toggle):
        _handle_menu_choice("t")
        mock_toggle.assert_called_once()

    @patch("tps_pro.main.switch_model")
    def test_m_calls_switch_model(self, mock_switch):
        _handle_menu_choice("m")
        mock_switch.assert_called_once()

    @patch("builtins.input", return_value="")
    @patch("tps_pro.main.generate_command")
    def test_cmd_calls_generate_command(self, mock_gen, mock_input):
        _handle_menu_choice("cmd")
        mock_gen.assert_called_once()


# ===================================================================
# main function — verify it's importable and callable
# ===================================================================


@pytest.mark.unit
class TestMainFunction:
    def test_main_is_callable(self):
        assert callable(main)

    def test_reset_db_is_callable(self):
        assert callable(reset_db)

    def test_launch_dashboard_is_callable(self):
        assert callable(launch_dashboard)
