"""Comprehensive tests for hardware.py — GPU detection, VRAM caching, thermal monitoring, process cleanup.

All hardware dependencies (pynvml, psutil, subprocess) are mocked so tests run
on any machine without a GPU.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODULE = "tps_pro.hardware"


def _make_mem(total_gb: float, free_gb: float):
    """Return a fake pynvml memory-info object."""
    return SimpleNamespace(
        total=int(total_gb * 1024**3),
        free=int(free_gb * 1024**3),
        used=int((total_gb - free_gb) * 1024**3),
    )


def _make_proc(pid: int, used_bytes: int):
    """Return a fake pynvml process-info object."""
    return SimpleNamespace(pid=pid, usedGpuMemory=used_bytes)


def _reset_nvml_state():
    """Reset the module-level NVML initialisation flag and VRAM cache."""
    import tps_pro.hardware as hw

    hw._nvml_state.initialized = False
    hw._vram_cache.time = 0.0
    hw._vram_cache.value = None


@pytest.fixture(autouse=True)
def _clean_nvml_state():
    """Ensure each test starts with a clean NVML / cache state."""
    _reset_nvml_state()
    yield
    _reset_nvml_state()


# ===================================================================
# 1. detect_gpus()
# ===================================================================


class TestDetectGpus:
    """Tests for detect_gpus()."""

    @pytest.mark.unit
    def test_single_gpu(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "handle0"
        mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA RTX 4090"
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = _make_mem(24.0, 20.0)
        mock_pynvml.NVMLError = Exception

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import detect_gpus

            gpus = detect_gpus()

        assert len(gpus) == 1
        assert gpus[0]["index"] == 0
        assert gpus[0]["name"] == "NVIDIA RTX 4090"
        assert gpus[0]["vram_total_gb"] == 24.0
        assert gpus[0]["vram_free_gb"] == 20.0

    @pytest.mark.unit
    def test_multiple_gpus(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        handles = ["handle0", "handle1"]
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = handles
        mock_pynvml.nvmlDeviceGetName.side_effect = ["GPU A", "GPU B"]
        mock_pynvml.nvmlDeviceGetMemoryInfo.side_effect = [
            _make_mem(16.0, 12.0),
            _make_mem(8.0, 6.0),
        ]
        mock_pynvml.NVMLError = Exception

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import detect_gpus

            gpus = detect_gpus()

        assert len(gpus) == 2
        assert gpus[0]["name"] == "GPU A"
        assert gpus[1]["name"] == "GPU B"
        assert gpus[1]["vram_total_gb"] == 8.0

    @pytest.mark.unit
    def test_name_returned_as_bytes(self):
        """pynvml sometimes returns GPU name as bytes — must decode."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA A100"
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = _make_mem(40.0, 38.0)
        mock_pynvml.NVMLError = Exception

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import detect_gpus

            gpus = detect_gpus()

        assert gpus[0]["name"] == "NVIDIA A100"

    @pytest.mark.unit
    def test_no_gpus_returns_empty(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 0
        mock_pynvml.NVMLError = Exception

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import detect_gpus

            gpus = detect_gpus()

        assert gpus == []

    @pytest.mark.unit
    def test_pynvml_not_installed(self):
        """If pynvml is not importable, detect_gpus returns []."""
        with patch.dict("sys.modules", {"pynvml": None}):
            from tps_pro.hardware import detect_gpus

            gpus = detect_gpus()

        assert gpus == []

    @pytest.mark.unit
    def test_nvml_error_during_enumeration(self):
        """NVMLError mid-enumeration falls back to empty list."""
        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = type("NVMLError", (Exception,), {})  # noqa: N806
        mock_pynvml.nvmlDeviceGetCount.side_effect = mock_pynvml.NVMLError(
            "driver gone"
        )

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import detect_gpus

            gpus = detect_gpus()

        assert gpus == []


# ===================================================================
# 2. _ensure_nvml() / _ensure_nvml_with_retry()
# ===================================================================


class TestEnsureNvml:
    """Tests for the persistent NVML session helpers."""

    @pytest.mark.unit
    def test_first_call_initialises(self):
        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = Exception

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import _ensure_nvml

            _ensure_nvml()

        mock_pynvml.nvmlInit.assert_called_once()

    @pytest.mark.unit
    def test_second_call_is_noop(self):
        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = Exception

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import _ensure_nvml

            _ensure_nvml()
            _ensure_nvml()

        # nvmlInit only called once even though _ensure_nvml called twice
        mock_pynvml.nvmlInit.assert_called_once()

    @pytest.mark.unit
    def test_retry_on_driver_failure(self):
        """_ensure_nvml_with_retry resets flag and retries once on NVMLError."""
        NVMLError = type("NVMLError", (Exception,), {})  # noqa: N806
        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = NVMLError

        # First nvmlInit fails, second succeeds
        mock_pynvml.nvmlInit.side_effect = [NVMLError("driver died"), None]

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import _ensure_nvml_with_retry

            _ensure_nvml_with_retry()

        assert mock_pynvml.nvmlInit.call_count == 2

    @pytest.mark.unit
    def test_retry_exhausted_raises(self):
        """If driver is truly gone, NVMLError propagates after retry."""
        NVMLError = type("NVMLError", (Exception,), {})  # noqa: N806
        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = NVMLError

        # Both attempts fail
        mock_pynvml.nvmlInit.side_effect = NVMLError("permanent failure")

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import _ensure_nvml_with_retry

            with pytest.raises(NVMLError):
                _ensure_nvml_with_retry()


# ===================================================================
# 3. get_vram_used_mb()
# ===================================================================


class TestGetVramUsedMb:
    """Tests for VRAM usage snapshot with TTL cache."""

    @pytest.mark.unit
    def test_returns_used_vram(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        mock_pynvml.nvmlDeviceGetName.return_value = "GPU"
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = _make_mem(24.0, 20.0)
        mock_pynvml.NVMLError = Exception

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import get_vram_used_mb

            result = get_vram_used_mb(cache_seconds=0)

        # Used = (24 - 20) * 1024 = 4096 MB
        assert result == pytest.approx(4096.0, abs=1.0)

    @pytest.mark.unit
    def test_returns_cached_value_within_ttl(self):
        """Second call within TTL returns cached value without calling detect_gpus again."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        mock_pynvml.nvmlDeviceGetName.return_value = "GPU"
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = _make_mem(24.0, 20.0)
        mock_pynvml.NVMLError = Exception

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import get_vram_used_mb

            first = get_vram_used_mb(cache_seconds=60)
            # Change the return to something different — should NOT be picked up
            mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = _make_mem(24.0, 10.0)
            second = get_vram_used_mb(cache_seconds=60)

        assert first == second
        # detect_gpus (via nvmlDeviceGetCount) should have been called only once
        assert mock_pynvml.nvmlDeviceGetCount.call_count == 1

    @pytest.mark.unit
    def test_refreshes_after_ttl_expires(self):
        """After TTL expires, the next call queries hardware again."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        mock_pynvml.nvmlDeviceGetName.return_value = "GPU"
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = _make_mem(24.0, 20.0)
        mock_pynvml.NVMLError = Exception

        time_values = [100.0, 200.0]  # second call well past TTL

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(f"{MODULE}.time") as mock_time,
        ):
            mock_time.time.side_effect = time_values
            mock_time.sleep = MagicMock()
            from tps_pro.hardware import get_vram_used_mb

            get_vram_used_mb(cache_seconds=2.0)
            # Now change what hardware reports
            mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = _make_mem(24.0, 10.0)
            result = get_vram_used_mb(cache_seconds=2.0)

        # Should reflect the new reading: (24 - 10) * 1024 = 14336 MB
        assert result == pytest.approx(14336.0, abs=1.0)
        assert mock_pynvml.nvmlDeviceGetCount.call_count == 2

    @pytest.mark.unit
    def test_no_gpus_returns_none(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 0
        mock_pynvml.NVMLError = Exception

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import get_vram_used_mb

            result = get_vram_used_mb(cache_seconds=0)

        assert result is None

    @pytest.mark.unit
    def test_thread_safety(self):
        """Multiple threads calling get_vram_used_mb concurrently don't corrupt state."""
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        mock_pynvml.nvmlDeviceGetName.return_value = "GPU"
        mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = _make_mem(24.0, 20.0)
        mock_pynvml.NVMLError = Exception

        results = []
        errors = []

        def worker():
            try:
                from tps_pro.hardware import get_vram_used_mb

                val = get_vram_used_mb(cache_seconds=0)
                results.append(val)
            except Exception as exc:
                errors.append(exc)

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            threads = [threading.Thread(target=worker) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert not errors
        assert all(r == pytest.approx(4096.0, abs=1.0) for r in results)


# ===================================================================
# 4. kill_competing_processes()
# ===================================================================


class TestKillCompetingProcesses:
    """Tests for competing-process cleanup."""

    @pytest.mark.unit
    def test_kills_process_above_threshold(self):
        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = Exception
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        big_proc = _make_proc(pid=1234, used_bytes=600 * 1024 * 1024)
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [big_proc]
        mock_pynvml.nvmlDeviceGetGraphicsRunningProcesses.return_value = []
        mock_pynvml.nvmlSystemGetProcessName.return_value = b"some_app.exe"

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(f"{MODULE}.subprocess") as mock_subprocess,
            patch(f"{MODULE}.time") as mock_time,
            patch(f"{MODULE}.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            mock_time.sleep = MagicMock()
            from tps_pro.hardware import kill_competing_processes

            killed = kill_competing_processes()

        assert len(killed) == 1
        assert killed[0]["pid"] == 1234
        mock_subprocess.run.assert_called_once()

    @pytest.mark.unit
    def test_whitelisted_process_not_killed(self):
        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = Exception
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        big_proc = _make_proc(pid=999, used_bytes=800 * 1024 * 1024)
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [big_proc]
        mock_pynvml.nvmlDeviceGetGraphicsRunningProcesses.return_value = []
        mock_pynvml.nvmlSystemGetProcessName.return_value = (
            b"C:\\Windows\\System32\\dwm.exe"
        )

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(f"{MODULE}.time") as mock_time,
        ):
            mock_time.sleep = MagicMock()
            from tps_pro.hardware import kill_competing_processes

            killed = kill_competing_processes()

        assert killed == []

    @pytest.mark.unit
    def test_below_threshold_not_killed(self):
        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = Exception
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        small_proc = _make_proc(pid=5555, used_bytes=100 * 1024 * 1024)  # 100 MB
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [small_proc]
        mock_pynvml.nvmlDeviceGetGraphicsRunningProcesses.return_value = []

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(f"{MODULE}.time") as mock_time,
        ):
            mock_time.sleep = MagicMock()
            from tps_pro.hardware import kill_competing_processes

            killed = kill_competing_processes()

        assert killed == []

    @pytest.mark.unit
    def test_handles_no_such_process(self):
        """psutil.NoSuchProcess / os.kill failing gracefully."""

        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = Exception
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        proc = _make_proc(pid=7777, used_bytes=600 * 1024 * 1024)
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [proc]
        mock_pynvml.nvmlDeviceGetGraphicsRunningProcesses.return_value = []
        mock_pynvml.nvmlSystemGetProcessName.return_value = b"gone_app"

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(f"{MODULE}.time") as mock_time,
            patch(f"{MODULE}.sys") as mock_sys,
            patch(f"{MODULE}.os") as mock_os,
        ):
            mock_sys.platform = "linux"
            mock_os.kill.side_effect = ProcessLookupError("No such process")
            mock_time.sleep = MagicMock()
            from tps_pro.hardware import kill_competing_processes

            killed = kill_competing_processes()

        # Process kill failed — should NOT appear in the killed list
        assert killed == []

    @pytest.mark.unit
    def test_handles_access_denied(self):
        """PermissionError (AccessDenied equivalent) during kill is handled."""
        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = Exception
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        proc = _make_proc(pid=8888, used_bytes=600 * 1024 * 1024)
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [proc]
        mock_pynvml.nvmlDeviceGetGraphicsRunningProcesses.return_value = []
        mock_pynvml.nvmlSystemGetProcessName.return_value = b"protected_app"

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(f"{MODULE}.time") as mock_time,
            patch(f"{MODULE}.sys") as mock_sys,
            patch(f"{MODULE}.os") as mock_os,
        ):
            mock_sys.platform = "linux"
            mock_os.kill.side_effect = PermissionError("Access denied")
            mock_time.sleep = MagicMock()
            from tps_pro.hardware import kill_competing_processes

            killed = kill_competing_processes()

        assert killed == []

    @pytest.mark.unit
    def test_no_competing_processes(self):
        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = Exception
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = []
        mock_pynvml.nvmlDeviceGetGraphicsRunningProcesses.return_value = []

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(f"{MODULE}.time") as mock_time,
        ):
            mock_time.sleep = MagicMock()
            from tps_pro.hardware import kill_competing_processes

            killed = kill_competing_processes()

        assert killed == []
        mock_time.sleep.assert_not_called()

    @pytest.mark.unit
    def test_pynvml_not_installed_returns_empty(self):
        with patch.dict("sys.modules", {"pynvml": None}):
            from tps_pro.hardware import kill_competing_processes

            killed = kill_competing_processes()

        assert killed == []

    @pytest.mark.unit
    def test_windows_uses_taskkill(self):
        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = Exception
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        proc = _make_proc(pid=4444, used_bytes=600 * 1024 * 1024)
        mock_pynvml.nvmlDeviceGetComputeRunningProcesses.return_value = [proc]
        mock_pynvml.nvmlDeviceGetGraphicsRunningProcesses.return_value = []
        mock_pynvml.nvmlSystemGetProcessName.return_value = b"some_app.exe"

        with (
            patch.dict("sys.modules", {"pynvml": mock_pynvml}),
            patch(f"{MODULE}.subprocess") as mock_subprocess,
            patch(f"{MODULE}.time") as mock_time,
            patch(f"{MODULE}.sys") as mock_sys,
        ):
            mock_sys.platform = "win32"
            mock_time.sleep = MagicMock()
            from tps_pro.hardware import kill_competing_processes

            killed = kill_competing_processes()

        assert len(killed) == 1
        mock_subprocess.run.assert_called_once()
        args = mock_subprocess.run.call_args[0][0]
        assert args[0] == "taskkill"
        assert "/PID" in args


# ===================================================================
# 5. check_thermal_throttle()
# ===================================================================


class TestCheckThermalThrottle:
    """Tests for thermal throttle detection."""

    @pytest.mark.unit
    def test_below_threshold_returns_false(self):
        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = Exception
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 70
        mock_pynvml.NVML_TEMPERATURE_GPU = 0

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import check_thermal_throttle

            is_throttling, temp = check_thermal_throttle(threshold=85)

        assert is_throttling is False
        assert temp == 70

    @pytest.mark.unit
    def test_at_threshold_returns_true(self):
        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = Exception
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 85
        mock_pynvml.NVML_TEMPERATURE_GPU = 0

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import check_thermal_throttle

            is_throttling, temp = check_thermal_throttle(threshold=85)

        assert is_throttling is True
        assert temp == 85

    @pytest.mark.unit
    def test_above_threshold_returns_true(self):
        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = Exception
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        mock_pynvml.nvmlDeviceGetTemperature.return_value = 95
        mock_pynvml.NVML_TEMPERATURE_GPU = 0

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import check_thermal_throttle

            is_throttling, temp = check_thermal_throttle(threshold=85)

        assert is_throttling is True
        assert temp == 95

    @pytest.mark.unit
    def test_multi_gpu_returns_max_temp(self):
        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = Exception
        mock_pynvml.nvmlDeviceGetCount.return_value = 2
        mock_pynvml.nvmlDeviceGetHandleByIndex.side_effect = ["h0", "h1"]
        mock_pynvml.nvmlDeviceGetTemperature.side_effect = [60, 90]
        mock_pynvml.NVML_TEMPERATURE_GPU = 0

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import check_thermal_throttle

            is_throttling, temp = check_thermal_throttle(threshold=85)

        assert is_throttling is True
        assert temp == 90

    @pytest.mark.unit
    def test_pynvml_error_returns_safe_default(self):
        mock_pynvml = MagicMock()
        NVMLError = type("NVMLError", (Exception,), {})  # noqa: N806
        mock_pynvml.NVMLError = NVMLError
        mock_pynvml.nvmlInit.side_effect = NVMLError("driver error")

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            from tps_pro.hardware import check_thermal_throttle

            is_throttling, temp = check_thermal_throttle()

        assert is_throttling is False
        assert temp == 0

    @pytest.mark.unit
    def test_pynvml_not_installed_returns_safe_default(self):
        with patch.dict("sys.modules", {"pynvml": None}):
            from tps_pro.hardware import check_thermal_throttle

            is_throttling, temp = check_thermal_throttle()

        assert is_throttling is False
        assert temp == 0


# ===================================================================
# 6. wait_for_cooldown()
# ===================================================================


class TestWaitForCooldown:
    """Tests for the GPU cooldown loop."""

    @pytest.mark.unit
    def test_already_cool_returns_immediately(self):
        with patch(f"{MODULE}.check_thermal_throttle") as mock_check:
            mock_check.return_value = (False, 60)
            from tps_pro.hardware import wait_for_cooldown

            result = wait_for_cooldown(target_temp=75, timeout=120)

        assert result is True
        # Should only call check once (the initial check)
        mock_check.assert_called_once()

    @pytest.mark.unit
    def test_cools_down_returns_true(self):
        """GPU starts hot, cools below target within timeout."""
        call_count = 0

        def fake_check(threshold=85):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (True, 80)  # initial: hot
            elif call_count == 2:
                return (True, 78)  # still hot in loop
            else:
                return (False, 70)  # cooled down

        time_values = iter([0.0, 5.0, 10.0, 15.0])

        with (
            patch(f"{MODULE}.check_thermal_throttle", side_effect=fake_check),
            patch(f"{MODULE}.time") as mock_time,
        ):
            mock_time.time.side_effect = lambda: next(time_values)
            mock_time.sleep = MagicMock()
            from tps_pro.hardware import wait_for_cooldown

            result = wait_for_cooldown(target_temp=75, timeout=120)

        assert result is True

    @pytest.mark.unit
    def test_timeout_returns_false(self):
        """GPU never cools — timeout reached."""
        with (
            patch(f"{MODULE}.check_thermal_throttle", return_value=(True, 90)),
            patch(f"{MODULE}.time") as mock_time,
        ):
            # Simulate time progressing past timeout
            time_counter = iter(range(0, 200, 5))
            mock_time.time.side_effect = lambda: float(next(time_counter))
            mock_time.sleep = MagicMock()
            from tps_pro.hardware import wait_for_cooldown

            result = wait_for_cooldown(target_temp=75, timeout=30)

        assert result is False

    @pytest.mark.unit
    def test_sleep_called_between_checks(self):
        """Verifies time.sleep(5) is called between temperature polls."""
        call_count = 0

        def fake_check(threshold=85):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return (True, 80)
            return (False, 70)

        time_values = iter([0.0, 5.0, 10.0, 15.0])

        with (
            patch(f"{MODULE}.check_thermal_throttle", side_effect=fake_check),
            patch(f"{MODULE}.time") as mock_time,
        ):
            mock_time.time.side_effect = lambda: next(time_values)
            mock_time.sleep = MagicMock()
            from tps_pro.hardware import wait_for_cooldown

            wait_for_cooldown(target_temp=75, timeout=120)

        mock_time.sleep.assert_called_with(5)
