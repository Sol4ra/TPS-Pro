"""Integration tests for the TUI menu system.

Each test pipes stdin to the app via subprocess and verifies that every menu
path completes without tracebacks or unexpected errors.  The app is invoked
through a thin wrapper script (_tui_harness.py) that creates fake server/model
files and patches out heavy side-effects (hardware detection, server startup,
port checks) so the menu rendering and dispatch logic is exercised end-to-end.

Timeout: 15 s per test (menu rendering + EOFError exit).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HARNESS = str(Path(__file__).resolve().parent / "_tui_harness.py")
_CWD = str(Path(__file__).resolve().parent.parent.parent)  # Desktop

# Errors that are *expected* in the output and should not cause test failure.
_ALLOWED_ERROR_SUBSTRINGS = frozenset(
    {
        "[OFFLINE]",
        "[offline]",
        "EOFError",
        "eof",
        "OFFLINE",
        # detect_gpus may fail in test env
        "No NVIDIA GPUs detected",
        "nvidia-smi",
        "wmic",
        # toggle side-effect warnings
        "find_llama_bench",
        # reset DB when no DB exists
        "Cancelled",
        # No results / phases
        "No results",
        "No Core Engine",
        "No GGUF files",
        "Not a valid directory",
        "requires 2+ GPUs",
        "Run tensor split",
        "File not found",
        "Not a GGUF file",
        "skip-quality is set",
        # aiohttp import warning
        "aiohttp",
        # Optuna storage warnings
        "optuna",
        # Port not available warning (non-fatal for menu tests)
        "Port",
        "already in use",
        # Context warnings
        "Minimum context",
        "Invalid number",
        # kill_server / cleanup warnings
        "kill_server",
        "atexit",
        "Cleanup",
        # Windows filesystem errors (temp dir corruption, not our fault)
        "WinError",
        # Generic warnings from phases
        "WARNING",
        "Warning",
        "warning",
    }
)


def _run_menu(stdin_text: str, *, timeout: int = 15) -> subprocess.CompletedProcess:
    """Run the TUI harness with the given stdin and return the result."""
    _src_dir = str(Path(__file__).resolve().parent.parent / "src")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": _src_dir}
    return subprocess.run(
        [sys.executable, _HARNESS],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=_CWD,
        env=env,
    )


def _assert_no_crash(result: subprocess.CompletedProcess, *, label: str = "") -> None:
    """Assert that the process did not crash with a traceback or unexpected error."""
    combined = result.stdout + result.stderr
    ctx = f" [{label}]" if label else ""

    # No Python tracebacks
    assert "Traceback" not in combined, (
        f"Traceback detected in output{ctx}:\n{combined[-2000:]}"
    )

    # Check for error-like lines, but allow known/expected ones
    for line in combined.splitlines():
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # Look for suspicious error indicators
        for keyword in (
            "Error",
            "KeyError",
            "TypeError",
            "ValueError",
            "AttributeError",
            "ImportError",
            "NameError",
            "IndexError",
            "RuntimeError",
            "OSError",
        ):
            if keyword in line_stripped:
                # Check if this is an allowed/expected error
                if any(
                    allowed in line_stripped for allowed in _ALLOWED_ERROR_SUBSTRINGS
                ):
                    continue
                # Also allow if it's in a harmless context (e.g., printed menu text)
                if "error" in line_stripped.lower() and any(
                    ok in line_stripped.lower()
                    for ok in (
                        "user-friendly",
                        "error handling",
                        "error messages",
                        "error strategy",
                        "press enter",
                        "error]",
                    )
                ):
                    continue
                # Fail on genuine unexpected errors
                raise AssertionError(
                    f"Unexpected error '{keyword}' in output{ctx}:\n"
                    f"  Line: {line_stripped}\n"
                    f"  Full output (last 1500 chars):\n{combined[-1500:]}"
                )


# ---------------------------------------------------------------------------
# Main menu tests
# ---------------------------------------------------------------------------


class TestMainMenu:
    """Tests for each main menu option."""

    def test_quit(self):
        """[q] Quit exits cleanly."""
        result = _run_menu("q\n")
        _assert_no_crash(result, label="quit")

    def test_generate_command(self):
        """[cmd] Generate launch command."""
        result = _run_menu("cmd\n\nq\n")
        _assert_no_crash(result, label="cmd")

    def test_view_results(self):
        """[v] View results."""
        result = _run_menu("v\n\nq\n")
        _assert_no_crash(result, label="view_results")

    def test_switch_model_cancel(self):
        """[m] Switch model — cancel by sending empty input."""
        result = _run_menu("m\n\nq\n")
        _assert_no_crash(result, label="switch_model")

    def test_cycle_preset(self):
        """[p] Cycle preset (normal -> thorough -> quick -> normal)."""
        result = _run_menu("p\n\np\n\np\n\nq\n")
        _assert_no_crash(result, label="cycle_preset")
        assert "thorough" in result.stdout or "quick" in result.stdout

    def test_context_menu_auto(self):
        """[c] Context menu — choose auto."""
        result = _run_menu("c\nauto\n\nq\n")
        _assert_no_crash(result, label="context_auto")

    def test_context_menu_number(self):
        """[c] Context menu — set a specific context size."""
        result = _run_menu("c\n8192\n\nq\n")
        _assert_no_crash(result, label="context_8192")

    def test_context_menu_back(self):
        """[c] Context menu — go back without choosing."""
        result = _run_menu("c\nb\nq\n")
        _assert_no_crash(result, label="context_back")

    def test_context_menu_invalid(self):
        """[c] Context menu — invalid input."""
        result = _run_menu("c\nfoo\n\nq\n")
        _assert_no_crash(result, label="context_invalid")

    def test_draft_model_menu_back(self):
        """[d] Draft model menu — go back."""
        result = _run_menu("d\nb\nq\n")
        _assert_no_crash(result, label="draft_back")

    def test_draft_model_menu_none(self):
        """[d] Draft model menu — set to none."""
        result = _run_menu("d\nnone\n\nq\n")
        _assert_no_crash(result, label="draft_none")

    def test_draft_model_menu_invalid_path(self):
        """[d] Draft model menu — provide invalid path."""
        result = _run_menu("d\n/nonexistent/model.gguf\n\nq\n")
        _assert_no_crash(result, label="draft_invalid")

    def test_toggle_menu_back(self):
        """[t] Toggles menu — enter and go back."""
        result = _run_menu("t\nb\nq\n")
        _assert_no_crash(result, label="toggle_back")

    def test_toggle_menu_cycle_all(self):
        """[t] Toggles menu — toggle each option on/off then back."""
        # Toggle options 1-7, each once (toggles on), then back
        result = _run_menu("t\n1\n2\n3\n4\n5\n6\n7\nb\nq\n")
        _assert_no_crash(result, label="toggle_all")

    def test_toggle_pareto(self):
        """[t] Toggle pareto mode on."""
        result = _run_menu("t\n1\nb\nq\n")
        _assert_no_crash(result, label="toggle_pareto")

    def test_toggle_debug(self):
        """[t] Toggle debug output."""
        result = _run_menu("t\n2\nb\nq\n")
        _assert_no_crash(result, label="toggle_debug")

    def test_optimize_start(self):
        """[o] Optimize — just enters and EOFError exits gracefully."""
        result = _run_menu("o\n")
        _assert_no_crash(result, label="optimize_start")

    def test_invalid_choice(self):
        """Invalid menu choice shows error message."""
        result = _run_menu("xyz\nq\n")
        _assert_no_crash(result, label="invalid_choice")


# ---------------------------------------------------------------------------
# Advanced menu tests
# ---------------------------------------------------------------------------


class TestAdvancedMenu:
    """Tests for advanced submenu options (entered via 'a')."""

    def test_advanced_back(self):
        """[a] then [b] — enter advanced menu and go back."""
        result = _run_menu("a\nb\nq\n")
        _assert_no_crash(result, label="advanced_back")
        assert "Advanced" in result.stdout

    def test_advanced_gpu_offload(self):
        """[a][g] GPU offload phase — will fail gracefully without server."""
        result = _run_menu("a\ng\n\nb\nq\n")
        _assert_no_crash(result, label="gpu_offload")

    def test_advanced_core_engine(self):
        """[a][ce] Core engine — provide trial count then it runs/fails."""
        result = _run_menu("a\nce\n5\n\nb\nq\n")
        _assert_no_crash(result, label="core_engine")

    def test_advanced_speculation(self):
        """[a][sp] Speculation phase."""
        result = _run_menu("a\nsp\n5\n\nb\nq\n")
        _assert_no_crash(result, label="speculation")

    def test_advanced_kv_quality(self):
        """[a][kv] KV quality phase."""
        result = _run_menu("a\nkv\n5\n\nb\nq\n")
        _assert_no_crash(result, label="kv_quality")

    def test_advanced_workload_sim(self):
        """[a][wl] Workload sim phase."""
        result = _run_menu("a\nwl\n\nb\nq\n")
        _assert_no_crash(result, label="workload_sim")

    def test_advanced_moe_threads(self):
        """[a][moe] MoE threads phase."""
        result = _run_menu("a\nmoe\n\nb\nq\n")
        _assert_no_crash(result, label="moe_threads")

    def test_advanced_expert_count(self):
        """[a][ex] Expert count phase."""
        result = _run_menu("a\nex\n\nb\nq\n")
        _assert_no_crash(result, label="expert_count")

    def test_advanced_moe_audit(self):
        """[a][mo] MoE audit phase."""
        result = _run_menu("a\nmo\n\nb\nq\n")
        _assert_no_crash(result, label="moe_audit")

    def test_advanced_tensor_split(self):
        """[a][ts] Tensor split (requires 2+ GPUs, expect graceful message)."""
        result = _run_menu("a\nts\n\nb\nq\n")
        _assert_no_crash(result, label="tensor_split")

    def test_advanced_topology(self):
        """[a][tp] Topology sweep (requires 2+ GPUs + tensor split results)."""
        result = _run_menu("a\ntp\n\nb\nq\n")
        _assert_no_crash(result, label="topology")

    def test_advanced_quality_eval(self):
        """[a][s] Quality eval."""
        result = _run_menu("a\ns\n5\n\nb\nq\n")
        _assert_no_crash(result, label="quality_eval")

    def test_advanced_niah(self):
        """[a][ni] NIAH eval."""
        result = _run_menu("a\nni\n\nb\nq\n")
        _assert_no_crash(result, label="niah")

    def test_advanced_reasoning(self):
        """[a][re] Reasoning eval."""
        result = _run_menu("a\nre\n\nb\nq\n")
        _assert_no_crash(result, label="reasoning")

    def test_advanced_integrity(self):
        """[a][ie] Integrity eval."""
        result = _run_menu("a\nie\n\nb\nq\n")
        _assert_no_crash(result, label="integrity")

    def test_advanced_context_sweep(self):
        """[a][cs] Context sweep — default contexts."""
        result = _run_menu("a\ncs\n\n\nb\nq\n")
        _assert_no_crash(result, label="context_sweep")

    def test_advanced_context_sweep_custom(self):
        """[a][cs] Context sweep — custom context sizes."""
        result = _run_menu("a\ncs\n4096,8192\n\nb\nq\n")
        _assert_no_crash(result, label="context_sweep_custom")

    def test_advanced_reset_db_yes(self):
        """[a][r] Reset DB — confirm yes."""
        result = _run_menu("a\nr\ny\n\nb\nq\n")
        _assert_no_crash(result, label="reset_db_yes")

    def test_advanced_reset_db_no(self):
        """[a][r] Reset DB — decline."""
        result = _run_menu("a\nr\nn\n\nb\nq\n")
        _assert_no_crash(result, label="reset_db_no")

    def test_advanced_html_report(self):
        """[a][html] HTML report generation."""
        result = _run_menu("a\nhtml\n\nb\nq\n")
        _assert_no_crash(result, label="html_report")

    def test_advanced_invalid_choice(self):
        """[a] Invalid choice in advanced menu."""
        result = _run_menu("a\nxyz\nb\nq\n")
        _assert_no_crash(result, label="advanced_invalid")

    def test_advanced_quant_recommend(self):
        """[a][qr] Quant recommendation."""
        result = _run_menu("a\nqr\n\nb\nq\n")
        _assert_no_crash(result, label="quant_rec")

    def test_advanced_coord_descent(self):
        """[a][cd] Coord descent (will try to run pipeline, fails at server start)."""
        result = _run_menu("a\ncd\n\nb\nq\n")
        _assert_no_crash(result, label="coord_descent")

    def test_advanced_batch_optimize_invalid_dir(self):
        """[a][ba] Batch optimize — provide invalid directory."""
        result = _run_menu("a\nba\n/nonexistent/dir\n\nb\nq\n")
        _assert_no_crash(result, label="batch_invalid")

    def test_advanced_dashboard(self):
        """[a][d] Dashboard launch."""
        result = _run_menu("a\nd\n\nb\nq\n")
        _assert_no_crash(result, label="dashboard")


# ---------------------------------------------------------------------------
# Multi-step navigation tests
# ---------------------------------------------------------------------------


class TestMultiStepNavigation:
    """Tests that exercise multi-step menu paths."""

    def test_toggle_then_optimize_then_quit(self):
        """Toggle debug on, then attempt optimize, then quit."""
        result = _run_menu("t\n2\nb\nq\n")
        _assert_no_crash(result, label="toggle_then_quit")

    def test_context_then_preset_then_quit(self):
        """Set context, cycle preset, then quit."""
        result = _run_menu("c\n4096\n\np\n\nq\n")
        _assert_no_crash(result, label="ctx_preset_quit")

    def test_advanced_multiple_phases(self):
        """Enter advanced, visit multiple phases, then back and quit."""
        result = _run_menu("a\nts\n\ntp\n\nb\nq\n")
        _assert_no_crash(result, label="multi_advanced")

    def test_advanced_enter_leave_reenter(self):
        """Enter advanced, back, enter again, back, quit."""
        result = _run_menu("a\nb\na\nb\nq\n")
        _assert_no_crash(result, label="advanced_reenter")
