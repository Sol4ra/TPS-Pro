"""Tests for engine/util.py — LogTee, PhaseTimer, check_dry_run, and BaselineFailure.

Uses tmp_path for file-system tests. No external dependencies needed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from tps_pro.engine.util import (
    LogTee,
    PhaseTimer,
    check_dry_run,
)
from tps_pro.errors import BaselineFailure

# ===================================================================
# Helpers
# ===================================================================


def _make_ctx(**overrides):
    """Build a minimal mock ctx for util tests."""
    from _ctx_factory import make_ctx_from_defaults

    return make_ctx_from_defaults(**overrides)


# ===================================================================
# BaselineFailure
# ===================================================================


@pytest.mark.unit
class TestBaselineFailure:
    def test_is_exception(self):
        assert issubclass(BaselineFailure, Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(BaselineFailure, match="test message"):
            raise BaselineFailure("test message")


# ===================================================================
# check_dry_run
# ===================================================================


@pytest.mark.unit
class TestCheckDryRun:
    def test_returns_false_when_not_dry_run(self):
        ctx = _make_ctx(dry_run=False)
        assert check_dry_run(ctx, "phase1", {"key": "val"}, 10) is False

    def test_returns_true_when_dry_run(self):
        ctx = _make_ctx(dry_run=True)
        assert check_dry_run(ctx, "phase1", {"key": "val"}, 10) is True

    def test_handles_full_sweep(self):
        ctx = _make_ctx(dry_run=True)
        assert check_dry_run(ctx, "phase1", None, "full") is True

    def test_handles_none_config(self):
        ctx = _make_ctx(dry_run=True)
        assert check_dry_run(ctx, "phase1", None, 5) is True


# ===================================================================
# PhaseTimer
# ===================================================================


@pytest.mark.unit
class TestPhaseTimer:
    def test_start_and_end_phase(self):
        timer = PhaseTimer()
        timer.start_phase("phase_a")
        time.sleep(0.01)
        timer.end_phase("phase_a")
        assert timer._phases["phase_a"]["duration"] is not None
        assert timer._phases["phase_a"]["duration"] > 0

    def test_end_nonexistent_phase_is_safe(self):
        timer = PhaseTimer()
        timer.end_phase("nonexistent")  # Should not raise

    def test_record_trial_and_eta(self):
        timer = PhaseTimer()
        timer.start_phase("p")
        timer.record_trial(10.0)
        timer.record_trial(20.0)
        eta = timer.eta(remaining_trials=5)
        # avg = 15.0, remaining = 5 -> 75s -> "1.2m"
        assert "m" in eta or "s" in eta

    def test_eta_unknown_when_no_trials(self):
        timer = PhaseTimer()
        assert timer.eta(remaining_trials=5) == "unknown"

    def test_eta_unknown_when_zero_remaining(self):
        timer = PhaseTimer()
        timer.record_trial(10.0)
        assert timer.eta(remaining_trials=0) == "unknown"

    def test_eta_seconds_format(self):
        timer = PhaseTimer()
        timer.start_phase("p")
        timer.record_trial(2.0)
        eta = timer.eta(remaining_trials=3)
        # avg = 2.0, remaining = 3 -> 6s
        assert eta == "6s"

    def test_eta_hours_format(self):
        timer = PhaseTimer()
        timer.start_phase("p")
        timer.record_trial(600.0)
        eta = timer.eta(remaining_trials=10)
        # avg = 600, remaining = 10 -> 6000s -> "1.7h"
        assert "h" in eta

    def test_summary_output(self):
        timer = PhaseTimer()
        timer.start_phase("alpha")
        timer._phases["alpha"]["end"] = timer._phases["alpha"]["start"] + 30
        timer._phases["alpha"]["duration"] = 30
        timer.start_phase("beta")
        # beta has no end => skipped
        summary = timer.summary()
        assert "alpha" in summary
        assert "beta" in summary
        assert "skipped" in summary
        assert "Total" in summary

    def test_summary_with_long_duration(self):
        timer = PhaseTimer()
        timer.start_phase("long")
        timer._phases["long"]["end"] = timer._phases["long"]["start"] + 7200
        timer._phases["long"]["duration"] = 7200
        summary = timer.summary()
        assert "h" in summary


# ===================================================================
# LogTee
# ===================================================================


@pytest.mark.unit
class TestLogTee:
    def test_writes_to_file(self, tmp_path):
        tee = LogTee(str(tmp_path))
        try:
            assert tee._file is not None
            tee.write("hello\n")
            tee.flush()
            content = Path(tee.log_path).read_text(encoding="utf-8")
            assert "hello" in content
        finally:
            tee.close()

    def test_restores_stdout(self, tmp_path):
        original = sys.stdout
        tee = LogTee(str(tmp_path))
        try:
            # stdout should now be the tee
            assert sys.stdout is tee
        finally:
            tee.close()
        assert sys.stdout is original

    def test_close_is_idempotent(self, tmp_path):
        tee = LogTee(str(tmp_path))
        tee.close()
        tee.close()  # Should not raise

    def test_context_manager(self, tmp_path):
        original = sys.stdout
        with LogTee(str(tmp_path)) as tee:
            tee.write("ctx test\n")
        assert sys.stdout is original

    def test_isatty_delegates(self, tmp_path):
        tee = LogTee(str(tmp_path))
        try:
            # Should not raise, returns bool
            result = tee.isatty()
            assert isinstance(result, bool)
        finally:
            tee.close()

    def test_encoding_property(self, tmp_path):
        tee = LogTee(str(tmp_path))
        try:
            enc = tee.encoding
            assert isinstance(enc, str)
        finally:
            tee.close()

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        tee = LogTee(str(nested))
        try:
            assert nested.exists()
        finally:
            tee.close()
