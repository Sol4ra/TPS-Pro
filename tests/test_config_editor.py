"""Unit tests for the config_editor TUI module.

Tests the pure logic and display functions by mocking input/output.
No subprocess needed — functions are tested directly.
"""

from __future__ import annotations

import json
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tps_pro.pipeline_config import PipelineConfig, PhaseConfig

# ── Helpers ──────────────────────────────────────────────────────

_MODULE = "tps_pro.cli.config_editor"


def _make_test_config() -> PipelineConfig:
    """Build a small config for testing."""
    return PipelineConfig(
        global_flags={"no_mmap": True},
        phases=[
            PhaseConfig(
                phase="gpu_offload",
                display_name="GPU Offload",
                enabled=True,
            ),
            PhaseConfig(
                phase="core_engine",
                display_name="Core Engine",
                enabled=True,
                trials=100,
                search_params=["threads", "batch_size", "ubatch_size", "flash_attn"],
                lock={},
            ),
            PhaseConfig(
                phase="quality",
                display_name="Quality/Sampling",
                enabled=False,
                trials=60,
            ),
        ],
    )


def _make_ctx(tmpdir: Path) -> SimpleNamespace:
    results = tmpdir / "results" / "test-model"
    results.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        results_dir=str(results),
        is_moe=False,
    )


# ── Display tests ────────────────────────────────────────────────


@pytest.mark.unit
class TestFormatPhaseLine:
    """Tests for _format_phase_line."""

    def test_enabled_phase_basic(self):
        from tps_pro.cli.config_editor import _format_phase_line

        phase = PhaseConfig(phase="gpu", display_name="GPU Offload", enabled=True)
        line = _format_phase_line(0, phase)
        assert "[1]" in line
        assert "\u2713" in line
        assert "GPU Offload" in line

    def test_disabled_phase(self):
        from tps_pro.cli.config_editor import _format_phase_line

        phase = PhaseConfig(phase="q", display_name="Quality", enabled=False)
        line = _format_phase_line(2, phase)
        assert "[3]" in line
        assert "\u2717" in line
        assert "disabled" in line

    def test_phase_with_trials(self):
        from tps_pro.cli.config_editor import _format_phase_line

        phase = PhaseConfig(
            phase="core", display_name="Core Engine", enabled=True, trials=100
        )
        line = _format_phase_line(0, phase)
        assert "100 trials" in line

    def test_phase_with_range(self):
        from tps_pro.cli.config_editor import _format_phase_line

        phase = PhaseConfig(
            phase="moe",
            display_name="MoE Threads",
            enabled=True,
            range=[8, 32],
            step=2,
        )
        line = _format_phase_line(0, phase)
        assert "range: 8-32" in line
        assert "step: 2" in line

    def test_phase_with_test_flags(self):
        from tps_pro.cli.config_editor import _format_phase_line

        phase = PhaseConfig(
            phase="ab",
            display_name="A/B Toggles",
            enabled=True,
            test_flags=["op_offload", "prio"],
        )
        line = _format_phase_line(0, phase)
        assert "flags:" in line
        assert "op_offload" in line

    def test_phase_with_search_params_truncated(self):
        from tps_pro.cli.config_editor import _format_phase_line

        phase = PhaseConfig(
            phase="core",
            display_name="Core Engine",
            enabled=True,
            search_params=["threads", "batch_size", "ubatch_size", "flash_attn"],
        )
        line = _format_phase_line(0, phase)
        assert "searching:" in line
        assert "..." in line

    def test_phase_with_kv_types(self):
        from tps_pro.cli.config_editor import _format_phase_line

        phase = PhaseConfig(
            phase="kv",
            display_name="KV Sweep",
            enabled=True,
            kv_types=["f16", "q8_0"],
        )
        line = _format_phase_line(0, phase)
        assert "kv:" in line
        assert "f16" in line


@pytest.mark.unit
class TestPrintConfigView:
    """Tests for _print_config_view output."""

    def test_prints_global_flags(self, capsys):
        from tps_pro.cli.config_editor import _print_config_view

        config = _make_test_config()
        _print_config_view(config)
        output = capsys.readouterr().out
        assert "Global Flags:" in output
        assert "no_mmap=True" in output

    def test_prints_no_flags(self, capsys):
        from tps_pro.cli.config_editor import _print_config_view

        config = PipelineConfig(global_flags={}, phases=[])
        _print_config_view(config)
        output = capsys.readouterr().out
        assert "(none)" in output

    def test_prints_pipeline_order(self, capsys):
        from tps_pro.cli.config_editor import _print_config_view

        config = _make_test_config()
        _print_config_view(config)
        output = capsys.readouterr().out
        assert "Pipeline Order:" in output
        assert "GPU Offload" in output
        assert "Core Engine" in output

    def test_prints_menu_keys(self, capsys):
        from tps_pro.cli.config_editor import _print_config_view

        config = _make_test_config()
        _print_config_view(config)
        output = capsys.readouterr().out
        for key in ("[g]", "[r]", "[t]", "[m]", "[e]", "[d]", "[s]", "[b]"):
            assert key in output


# ── Parse flag value ─────────────────────────────────────────────


@pytest.mark.unit
class TestParseFlagValue:
    """Tests for _parse_flag_value."""

    def test_true_variants(self):
        from tps_pro.cli.config_editor import _parse_flag_value

        for s in ("true", "True", "on", "yes", "1"):
            assert _parse_flag_value(s) is True

    def test_false_variants(self):
        from tps_pro.cli.config_editor import _parse_flag_value

        for s in ("false", "False", "off", "no", "0"):
            assert _parse_flag_value(s) is False

    def test_integer(self):
        from tps_pro.cli.config_editor import _parse_flag_value

        assert _parse_flag_value("42") == 42

    def test_float(self):
        from tps_pro.cli.config_editor import _parse_flag_value

        assert _parse_flag_value("3.14") == 3.14

    def test_string_fallback(self):
        from tps_pro.cli.config_editor import _parse_flag_value

        assert _parse_flag_value("some_value") == "some_value"


# ── Set global flag ──────────────────────────────────────────────


@pytest.mark.unit
class TestSetGlobalFlag:
    """Tests for _do_set_global_flag."""

    @patch(f"{_MODULE}._input", side_effect=["no_mmap true", ""])
    def test_set_flag(self, mock_input):
        from tps_pro.cli.config_editor import _do_set_global_flag

        config = PipelineConfig(global_flags={}, phases=[])
        result = _do_set_global_flag(config)
        assert result.global_flags["no_mmap"] is True
        # Original is not mutated
        assert config.global_flags == {}

    @patch(f"{_MODULE}._input", side_effect=["flash_attn on", ""])
    def test_set_flag_on(self, mock_input):
        from tps_pro.cli.config_editor import _do_set_global_flag

        config = PipelineConfig(global_flags={}, phases=[])
        result = _do_set_global_flag(config)
        assert result.global_flags["flash_attn"] is True

    @patch(f"{_MODULE}._input", return_value="")
    def test_empty_input_returns_unchanged(self, mock_input):
        from tps_pro.cli.config_editor import _do_set_global_flag

        config = _make_test_config()
        result = _do_set_global_flag(config)
        assert result is config

    @patch(f"{_MODULE}._input", side_effect=["badformat", ""])
    def test_missing_value_shows_error(self, mock_input):
        from tps_pro.cli.config_editor import _do_set_global_flag

        config = PipelineConfig(global_flags={}, phases=[])
        result = _do_set_global_flag(config)
        assert result is config


# ── Remove global flag ───────────────────────────────────────────


@pytest.mark.unit
class TestRemoveGlobalFlag:
    """Tests for _do_remove_global_flag."""

    @patch(f"{_MODULE}._input", side_effect=["no_mmap", ""])
    def test_remove_existing_flag(self, mock_input):
        from tps_pro.cli.config_editor import _do_remove_global_flag

        config = PipelineConfig(global_flags={"no_mmap": True, "foo": 1}, phases=[])
        result = _do_remove_global_flag(config)
        assert "no_mmap" not in result.global_flags
        assert result.global_flags["foo"] == 1

    @patch(f"{_MODULE}._input", side_effect=["nonexistent", ""])
    def test_remove_nonexistent_flag(self, mock_input):
        from tps_pro.cli.config_editor import _do_remove_global_flag

        config = PipelineConfig(global_flags={"no_mmap": True}, phases=[])
        result = _do_remove_global_flag(config)
        assert result is config

    @patch(f"{_MODULE}._input")
    def test_no_flags_shows_error(self, mock_input):
        from tps_pro.cli.config_editor import _do_remove_global_flag

        mock_input.return_value = ""
        config = PipelineConfig(global_flags={}, phases=[])
        result = _do_remove_global_flag(config)
        assert result is config


# ── Toggle phase ─────────────────────────────────────────────────


@pytest.mark.unit
class TestTogglePhase:
    """Tests for _do_toggle_phase."""

    @patch(f"{_MODULE}._input", side_effect=["3", ""])
    def test_toggle_disabled_to_enabled(self, mock_input):
        from tps_pro.cli.config_editor import _do_toggle_phase

        config = _make_test_config()
        assert not config.phases[2].enabled
        result = _do_toggle_phase(config)
        assert result.phases[2].enabled
        # Original not mutated
        assert not config.phases[2].enabled

    @patch(f"{_MODULE}._input", side_effect=["1", ""])
    def test_toggle_enabled_to_disabled(self, mock_input):
        from tps_pro.cli.config_editor import _do_toggle_phase

        config = _make_test_config()
        assert config.phases[0].enabled
        result = _do_toggle_phase(config)
        assert not result.phases[0].enabled

    @patch(f"{_MODULE}._input", return_value="")
    def test_empty_input_unchanged(self, mock_input):
        from tps_pro.cli.config_editor import _do_toggle_phase

        config = _make_test_config()
        result = _do_toggle_phase(config)
        assert result is config

    @patch(f"{_MODULE}._input", side_effect=["99", ""])
    def test_invalid_number(self, mock_input):
        from tps_pro.cli.config_editor import _do_toggle_phase

        config = _make_test_config()
        result = _do_toggle_phase(config)
        assert result is config


# ── Move phase ───────────────────────────────────────────────────


@pytest.mark.unit
class TestMovePhase:
    """Tests for _do_move_phase."""

    @patch(f"{_MODULE}._input", side_effect=["2", "u", ""])
    def test_move_up(self, mock_input):
        from tps_pro.cli.config_editor import _do_move_phase

        config = _make_test_config()
        result = _do_move_phase(config)
        assert result.phases[0].phase == "core_engine"
        assert result.phases[1].phase == "gpu_offload"

    @patch(f"{_MODULE}._input", side_effect=["1", "d", ""])
    def test_move_down(self, mock_input):
        from tps_pro.cli.config_editor import _do_move_phase

        config = _make_test_config()
        result = _do_move_phase(config)
        assert result.phases[0].phase == "core_engine"
        assert result.phases[1].phase == "gpu_offload"

    @patch(f"{_MODULE}._input", side_effect=["1", "u", ""])
    def test_move_up_at_top(self, mock_input):
        from tps_pro.cli.config_editor import _do_move_phase

        config = _make_test_config()
        result = _do_move_phase(config)
        assert result is config

    @patch(f"{_MODULE}._input", side_effect=["3", "d", ""])
    def test_move_down_at_bottom(self, mock_input):
        from tps_pro.cli.config_editor import _do_move_phase

        config = _make_test_config()
        result = _do_move_phase(config)
        assert result is config

    @patch(f"{_MODULE}._input", return_value="")
    def test_empty_input(self, mock_input):
        from tps_pro.cli.config_editor import _do_move_phase

        config = _make_test_config()
        result = _do_move_phase(config)
        assert result is config


# ── Edit phase ───────────────────────────────────────────────────


@pytest.mark.unit
class TestEditPhase:
    """Tests for _do_edit_phase."""

    @patch(f"{_MODULE}._input", side_effect=["2", "trials 50", "done"])
    def test_set_trials(self, mock_input):
        from tps_pro.cli.config_editor import _do_edit_phase

        config = _make_test_config()
        result = _do_edit_phase(config)
        assert result.phases[1].trials == 50

    @patch(f"{_MODULE}._input", side_effect=["2", "+param poll", "done"])
    def test_add_search_param(self, mock_input):
        from tps_pro.cli.config_editor import _do_edit_phase

        config = _make_test_config()
        result = _do_edit_phase(config)
        assert "poll" in result.phases[1].search_params

    @patch(f"{_MODULE}._input", side_effect=["2", "-param threads", "done"])
    def test_remove_search_param(self, mock_input):
        from tps_pro.cli.config_editor import _do_edit_phase

        config = _make_test_config()
        result = _do_edit_phase(config)
        assert "threads" not in result.phases[1].search_params

    @patch(f"{_MODULE}._input", side_effect=["2", "+flag mlock", "done"])
    def test_add_flag(self, mock_input):
        from tps_pro.cli.config_editor import _do_edit_phase

        config = _make_test_config()
        result = _do_edit_phase(config)
        assert "mlock" in result.phases[1].test_flags

    @patch(f"{_MODULE}._input", side_effect=["2", "lock threads 8", "done"])
    def test_lock_param(self, mock_input):
        from tps_pro.cli.config_editor import _do_edit_phase

        config = _make_test_config()
        result = _do_edit_phase(config)
        assert result.phases[1].lock["threads"] == 8

    @patch(f"{_MODULE}._input", side_effect=["2", "lock threads 8", "unlock threads", "done"])
    def test_unlock_param(self, mock_input):
        from tps_pro.cli.config_editor import _do_edit_phase

        config = _make_test_config()
        result = _do_edit_phase(config)
        assert "threads" not in result.phases[1].lock

    @patch(f"{_MODULE}._input", return_value="")
    def test_empty_input(self, mock_input):
        from tps_pro.cli.config_editor import _do_edit_phase

        config = _make_test_config()
        result = _do_edit_phase(config)
        assert result is config


# ── Reset defaults ───────────────────────────────────────────────


@pytest.mark.unit
class TestResetDefaults:
    """Tests for _do_reset_defaults."""

    @patch(f"{_MODULE}.ctx", SimpleNamespace(is_moe=False))
    @patch(f"{_MODULE}._input", return_value="y")
    def test_reset_confirms(self, mock_input):
        from tps_pro.cli.config_editor import _do_reset_defaults

        config = _make_test_config()
        result = _do_reset_defaults(config)
        # Should have all default phases
        assert len(result.phases) == 8
        assert result.global_flags == {}

    @patch(f"{_MODULE}._input", side_effect=["n", ""])
    def test_reset_cancelled(self, mock_input):
        from tps_pro.cli.config_editor import _do_reset_defaults

        config = _make_test_config()
        result = _do_reset_defaults(config)
        assert result is config


# ── Save ─────────────────────────────────────────────────────────


@pytest.mark.unit
class TestSave:
    """Tests for _do_save."""

    def test_save_writes_json(self, tmp_path):
        from tps_pro.cli.config_editor import _do_save

        results = tmp_path / "results"
        results.mkdir()

        with patch(f"{_MODULE}.ctx", SimpleNamespace(results_dir=str(results))):
            with patch(f"{_MODULE}._input", return_value=""):
                config = _make_test_config()
                _do_save(config)

        saved = results / "pipeline-config.json"
        assert saved.exists()
        data = json.loads(saved.read_text(encoding="utf-8"))
        assert "global_flags" in data
        assert "pipeline" in data
        assert data["global_flags"]["no_mmap"] is True


# ── Menu loop ────────────────────────────────────────────────────


@pytest.mark.unit
class TestConfigEditorMenu:
    """Tests for config_editor_menu main loop."""

    def test_back_exits(self, tmp_path):
        """Pressing 'b' exits the menu."""
        from tps_pro.cli.config_editor import config_editor_menu

        results = tmp_path / "results"
        results.mkdir()
        mock_ctx = SimpleNamespace(results_dir=str(results), is_moe=False)

        with (
            patch(f"{_MODULE}.ctx", mock_ctx),
            patch(f"{_MODULE}._input", return_value="b"),
            patch("tps_pro.cli.menu.clear_screen"),
        ):
            config_editor_menu()  # Should return without error

    def test_empty_exits(self, tmp_path):
        """Empty input exits the menu."""
        from tps_pro.cli.config_editor import config_editor_menu

        results = tmp_path / "results"
        results.mkdir()
        mock_ctx = SimpleNamespace(results_dir=str(results), is_moe=False)

        with (
            patch(f"{_MODULE}.ctx", mock_ctx),
            patch(f"{_MODULE}._input", return_value=""),
            patch("tps_pro.cli.menu.clear_screen"),
        ):
            config_editor_menu()

    def test_invalid_choice_loops_then_exits(self, tmp_path):
        """Invalid choice shows message, then 'b' exits."""
        from tps_pro.cli.config_editor import config_editor_menu

        results = tmp_path / "results"
        results.mkdir()
        mock_ctx = SimpleNamespace(results_dir=str(results), is_moe=False)

        with (
            patch(f"{_MODULE}.ctx", mock_ctx),
            patch(f"{_MODULE}._input", side_effect=["z", "", "b"]),
            patch("tps_pro.cli.menu.clear_screen"),
        ):
            config_editor_menu()


# ── Advanced menu integration ────────────────────────────────────


@pytest.mark.unit
class TestAdvancedMenuCfgOption:
    """Verify [cfg] is wired into the advanced menu."""

    def test_cfg_in_valid_keys(self):
        from tps_pro.cli.menu import _print_advanced_menu

        with patch("tps_pro.cli.menu.ctx", SimpleNamespace(is_moe=False)):
            valid = _print_advanced_menu()
        assert "cfg" in valid

    def test_dispatch_calls_config_editor(self):
        from tps_pro.cli.menu import _dispatch_advanced

        with patch("tps_pro.cli.menu._do_config_editor") as mock_editor:
            _dispatch_advanced("cfg")
        mock_editor.assert_called_once()


# ── Immutability checks ─────────────────────────────────────────


@pytest.mark.unit
class TestImmutability:
    """Ensure actions return new objects, never mutate the original."""

    @patch(f"{_MODULE}._input", side_effect=["foo bar", ""])
    def test_set_flag_immutable(self, mock_input):
        from tps_pro.cli.config_editor import _do_set_global_flag

        original = _make_test_config()
        original_flags = dict(original.global_flags)
        _do_set_global_flag(original)
        assert original.global_flags == original_flags

    @patch(f"{_MODULE}._input", side_effect=["1", ""])
    def test_toggle_immutable(self, mock_input):
        from tps_pro.cli.config_editor import _do_toggle_phase

        original = _make_test_config()
        original_enabled = original.phases[0].enabled
        _do_toggle_phase(original)
        assert original.phases[0].enabled == original_enabled

    @patch(f"{_MODULE}._input", side_effect=["2", "d", ""])
    def test_move_immutable(self, mock_input):
        from tps_pro.cli.config_editor import _do_move_phase

        original = _make_test_config()
        original_order = [p.phase for p in original.phases]
        _do_move_phase(original)
        assert [p.phase for p in original.phases] == original_order
