"""Comprehensive TUI flow integration tests.

Each test drives a specific menu path via subprocess with piped stdin,
verifying the flow completes without tracebacks or unexpected errors.

Uses the shared _tui_harness.py to stub out heavy operations (server,
hardware detection, phase execution) so only the menu rendering and
dispatch logic is exercised end-to-end.

Timeout: 15 s per test (menu rendering + EOFError exit).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Harness plumbing
# ---------------------------------------------------------------------------

_HARNESS = str(Path(__file__).resolve().parent / "_tui_harness.py")
_CWD = str(Path(__file__).resolve().parent.parent.parent)  # Desktop
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")

# Errors that are *expected* in the output and should not cause test failure.
_ALLOWED_ERROR_SUBSTRINGS = frozenset(
    {
        "[OFFLINE]",
        "[offline]",
        "EOFError",
        "eof",
        "OFFLINE",
        "No NVIDIA GPUs detected",
        "nvidia-smi",
        "wmic",
        "find_llama_bench",
        "Cancelled",
        "No results",
        "No Core Engine",
        "No GGUF files",
        "Not a valid directory",
        "requires 2+ GPUs",
        "Run tensor split",
        "File not found",
        "Not a GGUF file",
        "skip-quality is set",
        "aiohttp",
        "optuna",
        "Port",
        "already in use",
        "Minimum context",
        "Invalid number",
        "kill_server",
        "atexit",
        "Cleanup",
        "WinError",
        "WARNING",
        "Warning",
        "warning",
        # Phase stubs / dense model messages
        "stub",
        "dense",
        "Dense",
        "MoE",
        "skipped",
        "not applicable",
        "No expert",
    }
)

_ERROR_KEYWORDS = (
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
)

# Benign contexts where the word "error" may appear in normal output.
_BENIGN_ERROR_CONTEXTS = (
    "user-friendly",
    "error handling",
    "error messages",
    "error strategy",
    "press enter",
    "error]",
)


def _run_tui(stdin_text: str, *, timeout: int = 15) -> subprocess.CompletedProcess:
    """Run the TUI harness with the given piped stdin and return the result."""
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": _SRC_DIR,
    }
    return subprocess.run(
        [sys.executable, _HARNESS],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=_CWD,
        env=env,
    )


def _assert_no_traceback(result: subprocess.CompletedProcess, *, label: str) -> None:
    """Fail if a Python traceback appears in either stdout or stderr."""
    combined = result.stdout + result.stderr
    assert "Traceback" not in combined, (
        f"Traceback detected [{label}]:\n{combined[-2000:]}"
    )


def _assert_no_unexpected_errors(
    result: subprocess.CompletedProcess, *, label: str
) -> None:
    """Fail if unexpected error-class strings appear in output."""
    combined = result.stdout + result.stderr

    for line in combined.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        for keyword in _ERROR_KEYWORDS:
            if keyword not in stripped:
                continue

            # Allowed / expected?
            if any(allowed in stripped for allowed in _ALLOWED_ERROR_SUBSTRINGS):
                continue

            # Benign context?
            if "error" in stripped.lower() and any(
                ok in stripped.lower() for ok in _BENIGN_ERROR_CONTEXTS
            ):
                continue

            raise AssertionError(
                f"Unexpected error '{keyword}' [{label}]:\n"
                f"  Line: {stripped}\n"
                f"  Tail:\n{combined[-1500:]}"
            )


def _assert_clean(result: subprocess.CompletedProcess, *, label: str) -> None:
    """Composite assertion: no traceback and no unexpected errors."""
    _assert_no_traceback(result, label=label)
    _assert_no_unexpected_errors(result, label=label)


# ---------------------------------------------------------------------------
# 1. Main menu: q quits cleanly
# ---------------------------------------------------------------------------


class TestMainMenuQuit:
    """Pressing 'q' at the main menu exits without error."""

    def test_quit_exits_cleanly(self):
        result = _run_tui("q\n")
        _assert_clean(result, label="quit")


# ---------------------------------------------------------------------------
# 2. Main menu: invalid choice shows error
# ---------------------------------------------------------------------------


class TestMainMenuInvalidChoice:
    """An unrecognised key shows an error message and re-draws the menu."""

    def test_invalid_then_quit(self):
        result = _run_tui("xyz\nq\n")
        _assert_clean(result, label="invalid_choice")
        assert "Invalid" in result.stdout, (
            "Expected 'Invalid' message in output after bad choice"
        )


# ---------------------------------------------------------------------------
# 3. Preset cycle: p cycles normal -> thorough -> quick -> normal
# ---------------------------------------------------------------------------


class TestPresetCycle:
    """Pressing 'p' three times should cycle through all presets."""

    def test_cycle_three_times(self):
        # Each 'p' press cycles the preset; pause with Enter between.
        result = _run_tui("p\n\np\n\np\n\nq\n")
        _assert_clean(result, label="preset_cycle")
        out = result.stdout
        # At least two of the three preset names should appear.
        hits = sum(1 for name in ("thorough", "quick", "normal") if name in out)
        assert hits >= 2, (
            f"Expected at least 2 preset names in output, got: {out[-800:]}"
        )


# ---------------------------------------------------------------------------
# 4. Context menu: c -> auto, c -> number, c -> back
# ---------------------------------------------------------------------------


class TestContextMenu:
    """The context sub-menu supports auto, numeric, and back inputs."""

    def test_context_auto(self):
        result = _run_tui("c\nauto\n\nq\n")
        _assert_clean(result, label="context_auto")
        assert "auto" in result.stdout.lower(), "Expected 'auto' confirmation in output"

    def test_context_number(self):
        result = _run_tui("c\n8192\n\nq\n")
        _assert_clean(result, label="context_number")
        assert "8192" in result.stdout or "8,192" in result.stdout, (
            "Expected context size confirmation in output"
        )

    def test_context_back(self):
        result = _run_tui("c\nb\nq\n")
        _assert_clean(result, label="context_back")


# ---------------------------------------------------------------------------
# 5. Switch model: m -> shows model list -> b backs out
# ---------------------------------------------------------------------------


class TestSwitchModel:
    """Entering 'm' shows the model list; 'b' (or empty) returns."""

    def test_switch_model_back(self):
        # Empty input or 'b' backs out of the model picker.
        result = _run_tui("m\nb\nq\n")
        _assert_clean(result, label="switch_model_back")

    def test_switch_model_empty_backs_out(self):
        result = _run_tui("m\n\nq\n")
        _assert_clean(result, label="switch_model_empty")


# ---------------------------------------------------------------------------
# 6. View results: v -> shows results or empty message -> back
# ---------------------------------------------------------------------------


class TestViewResults:
    """'v' shows results (or "No … results" for a fresh env) and returns."""

    def test_view_results(self):
        result = _run_tui("v\n\nq\n")
        _assert_clean(result, label="view_results")
        out = result.stdout.lower()
        # Should see either result data or the "no results" message.
        assert "result" in out or "no " in out, (
            "Expected results or 'no results' message in output"
        )


# ---------------------------------------------------------------------------
# 7. Advanced menu: a -> shows advanced options -> b backs out
# ---------------------------------------------------------------------------


class TestAdvancedMenuBack:
    """'a' opens the advanced sub-menu; 'b' returns to main."""

    def test_advanced_back(self):
        result = _run_tui("a\nb\nq\n")
        _assert_clean(result, label="advanced_back")
        assert "Advanced" in result.stdout, "Expected 'Advanced' header in output"


# ---------------------------------------------------------------------------
# 8. Advanced GPU offload: a -> g -> starts phase (mock server)
# ---------------------------------------------------------------------------


class TestAdvancedGPUOffload:
    """GPU offload phase dispatches without crash (server is stubbed)."""

    def test_gpu_offload(self):
        result = _run_tui("a\ng\n\nb\nq\n")
        _assert_clean(result, label="gpu_offload")


# ---------------------------------------------------------------------------
# 9. Advanced Core Engine: a -> ce -> starts phase
# ---------------------------------------------------------------------------


class TestAdvancedCoreEngine:
    """Core-engine phase asks for trials and dispatches."""

    def test_core_engine(self):
        # Provide trial count '5', then pause, back, quit.
        result = _run_tui("a\nce\n5\n\nb\nq\n")
        _assert_clean(result, label="core_engine")


# ---------------------------------------------------------------------------
# 10. Advanced KV sweep: a -> kv -> starts phase
# ---------------------------------------------------------------------------


class TestAdvancedKVSweep:
    """KV + context sweep phase dispatches without crash."""

    def test_kv_sweep(self):
        result = _run_tui("a\nkv\n5\n\nb\nq\n")
        _assert_clean(result, label="kv_sweep")


# ---------------------------------------------------------------------------
# 11. Advanced MoE: a -> moe -> starts/skips for dense models
# ---------------------------------------------------------------------------


class TestAdvancedMoE:
    """MoE phase either runs or reports not-applicable for dense models."""

    def test_moe_phase(self):
        # The harness uses --dense, so MoE should be skipped or invalid.
        result = _run_tui("a\nmoe\n\nb\nq\n")
        _assert_clean(result, label="moe_phase")


# ---------------------------------------------------------------------------
# 12. Advanced Reset DB: a -> r -> y confirms reset
# ---------------------------------------------------------------------------


class TestAdvancedResetDBConfirm:
    """Reset DB with 'y' confirmation deletes trial data."""

    def test_reset_db_yes(self):
        result = _run_tui("a\nr\ny\n\nb\nq\n")
        _assert_clean(result, label="reset_db_yes")
        out = result.stdout.lower()
        assert (
            "delete" in out or "reset" in out or "fresh" in out or "cancelled" in out
        ), "Expected reset confirmation or cancellation message"


# ---------------------------------------------------------------------------
# 13. Advanced Reset DB: a -> r -> n cancels reset
# ---------------------------------------------------------------------------


class TestAdvancedResetDBCancel:
    """Reset DB with 'n' cancels the operation."""

    def test_reset_db_no(self):
        result = _run_tui("a\nr\nn\n\nb\nq\n")
        _assert_clean(result, label="reset_db_no")
        assert "Cancelled" in result.stdout or "cancel" in result.stdout.lower(), (
            "Expected cancellation message after declining reset"
        )


# ---------------------------------------------------------------------------
# 14. Optimize: o -> shows resume prompt if results exist
# ---------------------------------------------------------------------------


class TestOptimizeResume:
    """Entering 'o' either shows the resume prompt or starts fresh."""

    def test_optimize_entry(self):
        # In the test harness there are no prior results, so the pipeline
        # should start directly.  The stub prints a message and returns.
        result = _run_tui("o\n")
        _assert_clean(result, label="optimize_entry")


# ---------------------------------------------------------------------------
# 15. Optimize fresh: o -> 2 starts fresh
# ---------------------------------------------------------------------------


class TestOptimizeFresh:
    """If progress exists, choosing '2' resets and starts fresh."""

    def test_optimize_fresh(self):
        # Without prior results the "2" input is consumed harmlessly
        # (the pipeline starts immediately).  This verifies no crash.
        result = _run_tui("o\n2\n")
        _assert_clean(result, label="optimize_fresh")


# ---------------------------------------------------------------------------
# Bonus: multi-step navigation smoke tests
# ---------------------------------------------------------------------------


class TestMultiStepFlows:
    """Compound navigation sequences that touch several menus."""

    def test_preset_then_context_then_quit(self):
        result = _run_tui("p\n\nc\n4096\n\nq\n")
        _assert_clean(result, label="preset_ctx_quit")

    def test_advanced_two_phases_then_quit(self):
        result = _run_tui("a\ng\n\nce\n5\n\nb\nq\n")
        _assert_clean(result, label="adv_two_phases")

    def test_view_then_advanced_then_quit(self):
        result = _run_tui("v\n\na\nb\nq\n")
        _assert_clean(result, label="view_adv_quit")

    def test_reenter_advanced(self):
        """Enter advanced, back, re-enter, back, quit."""
        result = _run_tui("a\nb\na\nb\nq\n")
        _assert_clean(result, label="reenter_advanced")
