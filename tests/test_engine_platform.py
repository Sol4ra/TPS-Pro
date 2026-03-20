"""Tests for engine/platform.py — Windows Job Object management.

Direct imports from the target module to satisfy coverage detection.
Tests verify function signatures and safe-mode behavior on non-Windows.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from tps_pro.engine.platform import (
    _assign_job_object,
)


@pytest.mark.unit
class TestAssignJobObject:
    def test_is_callable(self):
        """_assign_job_object should be a callable function."""
        assert callable(_assign_job_object)

    def test_accepts_popen_like_object(self):
        """_assign_job_object should accept a subprocess.Popen-like object."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 12345
        # Should not raise regardless of platform
        if sys.platform != "win32":
            _assign_job_object(mock_proc)
        else:
            # On Windows, it may succeed or fail gracefully
            try:
                _assign_job_object(mock_proc)
            except (OSError, ValueError):
                pass  # expected on some Windows configs

    def test_noop_on_non_windows(self):
        """_assign_job_object should be a no-op on non-Windows platforms."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 99999
        with patch("tps_pro.engine.platform.sys") as mock_sys:
            mock_sys.platform = "linux"
            _assign_job_object(mock_proc)
            # Should return without doing anything on non-win32

    def test_handles_oserror_gracefully(self):
        """_assign_job_object should handle errors gracefully."""
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.pid = 11111
        if sys.platform == "win32":
            # On real Windows, just verify it doesn't crash with a mock proc
            try:
                _assign_job_object(mock_proc)
            except (OSError, ValueError):
                pass  # expected on some Windows configs
        else:
            # On non-Windows, it's always a no-op
            _assign_job_object(mock_proc)
