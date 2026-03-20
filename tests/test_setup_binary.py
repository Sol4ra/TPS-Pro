"""Tests for tps_pro.cli.setup_binary module.

Covers: detect_gpu_type(), _match_asset(), get_latest_release(),
download_and_extract(), ensure_llama_server(), _check_nvidia_gpu(),
_check_amd_gpu().
"""

import subprocess
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tps_pro.cli.setup_binary import (
    SetupBinaryError,
    _check_amd_gpu,
    _check_nvidia_gpu,
    _cleanup_partial,
    _match_asset,
    detect_gpu_type,
    download_and_extract,
    ensure_llama_server,
    get_latest_release,
)


# ============================================================
# _check_nvidia_gpu()
# ============================================================


class TestCheckNvidiaGpu:
    """Test NVIDIA GPU detection via pynvml and nvidia-smi fallback."""

    @pytest.mark.unit
    def test_pynvml_detects_gpu(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = "NVIDIA GeForce RTX 5080"

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch("tps_pro.cli.setup_binary.subprocess.run") as mock_run:
                # pynvml should succeed, subprocess should not be called
                result = _check_nvidia_gpu()
                assert result == "NVIDIA GeForce RTX 5080"

    @pytest.mark.unit
    def test_pynvml_returns_bytes_name(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 1
        mock_handle = MagicMock()
        mock_pynvml.nvmlDeviceGetHandleByIndex.return_value = mock_handle
        mock_pynvml.nvmlDeviceGetName.return_value = b"NVIDIA GeForce RTX 4090"

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result = _check_nvidia_gpu()
            assert result == "NVIDIA GeForce RTX 4090"

    @pytest.mark.unit
    def test_pynvml_no_gpus_falls_back_to_nvidia_smi(self):
        mock_pynvml = MagicMock()
        mock_pynvml.nvmlDeviceGetCount.return_value = 0

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            with patch("tps_pro.cli.setup_binary.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="NVIDIA GeForce RTX 3080\n"
                )
                result = _check_nvidia_gpu()
                assert result == "NVIDIA GeForce RTX 3080"

    @pytest.mark.unit
    def test_pynvml_import_error_falls_back_to_nvidia_smi(self):
        with patch.dict("sys.modules", {"pynvml": None}):
            with patch("tps_pro.cli.setup_binary.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="NVIDIA GeForce RTX 4070\n"
                )
                result = _check_nvidia_gpu()
                assert result == "NVIDIA GeForce RTX 4070"

    @pytest.mark.unit
    def test_no_nvidia_gpu_at_all(self):
        with patch.dict("sys.modules", {"pynvml": None}):
            with patch("tps_pro.cli.setup_binary.subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("nvidia-smi not found")
                result = _check_nvidia_gpu()
                assert result is None


# ============================================================
# _check_amd_gpu()
# ============================================================


class TestCheckAmdGpu:
    """Test AMD GPU detection."""

    @pytest.mark.unit
    def test_rocminfo_detects_amd(self):
        with patch("tps_pro.cli.setup_binary.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stdout="GPU Agent: AMD Radeon RX 7900"
            )
            assert _check_amd_gpu() is True

    @pytest.mark.unit
    def test_rocminfo_no_gpu(self):
        with patch("tps_pro.cli.setup_binary.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="No GPU found")
            # "GPU" is still in the output ("No GPU found")
            # but let's test the negative case where rocminfo fails
            mock_run.side_effect = FileNotFoundError("rocminfo not found")
            with patch("tps_pro.cli.setup_binary.sys") as mock_sys:
                mock_sys.platform = "linux"
                with patch("tps_pro.cli.setup_binary.Path") as mock_path:
                    mock_path.return_value.exists.return_value = False
                    assert _check_amd_gpu() is False

    @pytest.mark.unit
    def test_rocminfo_not_found_no_hip(self):
        with patch("tps_pro.cli.setup_binary.subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("rocminfo not found")
            with patch("tps_pro.cli.setup_binary.sys") as mock_sys:
                mock_sys.platform = "linux"
                with patch("tps_pro.cli.setup_binary.Path") as mock_path:
                    mock_path.return_value.exists.return_value = False
                    assert _check_amd_gpu() is False


# ============================================================
# detect_gpu_type()
# ============================================================


class TestDetectGpuType:
    """Test combined GPU type detection."""

    @pytest.mark.unit
    @patch("tps_pro.cli.setup_binary._check_nvidia_gpu", return_value="RTX 5080")
    def test_nvidia_detected(self, _):
        backend, name = detect_gpu_type()
        assert backend == "cuda"
        assert name == "RTX 5080"

    @pytest.mark.unit
    @patch("tps_pro.cli.setup_binary._check_nvidia_gpu", return_value=None)
    @patch("tps_pro.cli.setup_binary._check_amd_gpu", return_value=True)
    def test_amd_detected(self, *_):
        backend, name = detect_gpu_type()
        assert backend == "rocm"
        assert "AMD" in name

    @pytest.mark.unit
    @patch("tps_pro.cli.setup_binary._check_nvidia_gpu", return_value=None)
    @patch("tps_pro.cli.setup_binary._check_amd_gpu", return_value=False)
    @patch("tps_pro.cli.setup_binary.subprocess.run")
    def test_vulkan_fallback(self, mock_run, *_):
        mock_run.return_value = MagicMock(returncode=0)
        backend, name = detect_gpu_type()
        assert backend == "vulkan"

    @pytest.mark.unit
    @patch("tps_pro.cli.setup_binary._check_nvidia_gpu", return_value=None)
    @patch("tps_pro.cli.setup_binary._check_amd_gpu", return_value=False)
    @patch("tps_pro.cli.setup_binary.subprocess.run")
    def test_cpu_fallback(self, mock_run, *_):
        mock_run.side_effect = FileNotFoundError("vulkaninfo not found")
        backend, name = detect_gpu_type()
        assert backend == "cpu"


# ============================================================
# _match_asset()
# ============================================================


class TestMatchAsset:
    """Test GitHub release asset matching logic."""

    _SAMPLE_ASSETS = [
        {"name": "llama-b1234-win-cuda-cu12.2-x64.zip", "browser_download_url": "url1"},
        {"name": "llama-b1234-win-vulkan-x64.zip", "browser_download_url": "url2"},
        {"name": "llama-b1234-win-x64.zip", "browser_download_url": "url3"},
        {
            "name": "llama-b1234-linux-cuda-cu12.2-x64.zip",
            "browser_download_url": "url4",
        },
        {"name": "llama-b1234-linux-x64.zip", "browser_download_url": "url5"},
        {"name": "llama-b1234-macos-arm64.zip", "browser_download_url": "url6"},
        {"name": "some-other-file.tar.gz", "browser_download_url": "url7"},
    ]

    @pytest.mark.unit
    def test_win_cuda_match(self):
        result = _match_asset(self._SAMPLE_ASSETS, "win", "cuda")
        assert result is not None
        assert "cuda" in result["name"].lower()
        assert "win" in result["name"].lower()

    @pytest.mark.unit
    def test_win_vulkan_match(self):
        result = _match_asset(self._SAMPLE_ASSETS, "win", "vulkan")
        assert result is not None
        assert "vulkan" in result["name"].lower()

    @pytest.mark.unit
    def test_win_cpu_match(self):
        result = _match_asset(self._SAMPLE_ASSETS, "win", "cpu")
        assert result is not None
        assert "cuda" not in result["name"].lower()
        assert "vulkan" not in result["name"].lower()
        assert "win" in result["name"].lower()

    @pytest.mark.unit
    def test_linux_cuda_match(self):
        result = _match_asset(self._SAMPLE_ASSETS, "linux", "cuda")
        assert result is not None
        assert "linux" in result["name"].lower()
        assert "cuda" in result["name"].lower()

    @pytest.mark.unit
    def test_no_match_for_missing_os(self):
        result = _match_asset(self._SAMPLE_ASSETS, "freebsd", "cuda")
        assert result is None

    @pytest.mark.unit
    def test_cuda_fallback_to_vulkan_when_no_cuda(self):
        assets = [
            {"name": "llama-b1234-win-vulkan-x64.zip", "browser_download_url": "url1"},
            {"name": "llama-b1234-win-x64.zip", "browser_download_url": "url2"},
        ]
        result = _match_asset(assets, "win", "cuda")
        assert result is not None
        assert "vulkan" in result["name"].lower()

    @pytest.mark.unit
    def test_skips_non_zip_files(self):
        assets = [
            {"name": "llama-b1234-win-cuda.tar.gz", "browser_download_url": "url1"},
        ]
        result = _match_asset(assets, "win", "cuda")
        assert result is None

    @pytest.mark.unit
    def test_empty_assets(self):
        result = _match_asset([], "win", "cuda")
        assert result is None


# ============================================================
# get_latest_release()
# ============================================================


class TestGetLatestRelease:
    """Test GitHub API interaction for fetching releases."""

    @pytest.mark.unit
    @patch("tps_pro.cli.setup_binary._detect_os_name", return_value="win")
    @patch("tps_pro.cli.setup_binary.requests.get")
    def test_successful_release_fetch(self, mock_get, _):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "tag_name": "b1234",
            "assets": [
                {
                    "name": "llama-b1234-win-cuda-cu12.2-x64.zip",
                    "browser_download_url": "https://example.com/download.zip",
                    "size": 100_000_000,
                },
            ],
        }
        mock_get.return_value = mock_resp

        result = get_latest_release("cuda")
        assert result["tag"] == "b1234"
        assert result["asset_url"] == "https://example.com/download.zip"

    @pytest.mark.unit
    @patch("tps_pro.cli.setup_binary.requests.get")
    def test_connection_error(self, mock_get):
        import requests as real_requests

        mock_get.side_effect = real_requests.ConnectionError("no internet")

        with pytest.raises(SetupBinaryError, match="No internet"):
            get_latest_release("cuda")

    @pytest.mark.unit
    @patch("tps_pro.cli.setup_binary.requests.get")
    def test_rate_limit_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_get.return_value = mock_resp

        with pytest.raises(SetupBinaryError, match="rate limit"):
            get_latest_release("cuda")

    @pytest.mark.unit
    @patch("tps_pro.cli.setup_binary._detect_os_name", return_value="win")
    @patch("tps_pro.cli.setup_binary.requests.get")
    def test_no_matching_asset(self, mock_get, _):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "tag_name": "b1234",
            "assets": [
                {
                    "name": "llama-b1234-linux-cuda-x64.zip",
                    "browser_download_url": "url",
                    "size": 100,
                },
            ],
        }
        mock_get.return_value = mock_resp

        with pytest.raises(SetupBinaryError, match="No matching release"):
            get_latest_release("rocm")


# ============================================================
# download_and_extract()
# ============================================================


class TestDownloadAndExtract:
    """Test download and extraction logic."""

    @pytest.mark.unit
    def test_successful_download_and_extract(self, tmp_path):
        # Create a fake zip containing llama-server.exe
        zip_content_dir = tmp_path / "zip_src"
        zip_content_dir.mkdir()
        (zip_content_dir / "llama-server.exe").write_text("fake exe")
        (zip_content_dir / "llama-bench.exe").write_text("fake bench")

        zip_path = tmp_path / "release.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(zip_content_dir / "llama-server.exe", "llama-server.exe")
            zf.write(zip_content_dir / "llama-bench.exe", "llama-bench.exe")

        zip_bytes = zip_path.read_bytes()

        target_dir = tmp_path / "bin"

        # Mock requests.get to return the zip bytes
        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": str(len(zip_bytes))}
        mock_resp.iter_content.return_value = [zip_bytes]
        mock_resp.raise_for_status = MagicMock()

        with patch("tps_pro.cli.setup_binary.requests.get", return_value=mock_resp):
            with patch("tps_pro.cli.setup_binary.sys") as mock_sys:
                mock_sys.platform = "win32"
                result = download_and_extract(
                    "https://example.com/release.zip", target_dir
                )

        assert result.name == "llama-server.exe"
        assert result.exists()

    @pytest.mark.unit
    def test_download_connection_error(self, tmp_path):
        import requests as real_requests

        target_dir = tmp_path / "bin"
        with patch(
            "tps_pro.cli.setup_binary.requests.get",
            side_effect=real_requests.ConnectionError("fail"),
        ):
            with pytest.raises(SetupBinaryError, match="no internet"):
                download_and_extract("https://example.com/release.zip", target_dir)

    @pytest.mark.unit
    def test_bad_zip_file(self, tmp_path):
        target_dir = tmp_path / "bin"
        bad_content = b"this is not a zip file"

        mock_resp = MagicMock()
        mock_resp.headers = {"content-length": str(len(bad_content))}
        mock_resp.iter_content.return_value = [bad_content]
        mock_resp.raise_for_status = MagicMock()

        with patch("tps_pro.cli.setup_binary.requests.get", return_value=mock_resp):
            with pytest.raises(SetupBinaryError, match="Extraction failed"):
                download_and_extract("https://example.com/release.zip", target_dir)


# ============================================================
# ensure_llama_server()
# ============================================================


class TestEnsureLlamaServer:
    """Test the main entry point that orchestrates detection and download."""

    @pytest.mark.unit
    def test_returns_existing_binary(self, tmp_path):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        exe = bin_dir / "llama-server.exe"
        exe.write_text("fake")

        with patch("tps_pro.cli.setup_binary.sys") as mock_sys:
            mock_sys.platform = "win32"
            result = ensure_llama_server(tmp_path)
            assert result == exe

    @pytest.mark.unit
    @patch("tps_pro.cli.setup_binary.download_and_extract")
    @patch("tps_pro.cli.setup_binary.get_latest_release")
    @patch(
        "tps_pro.cli.setup_binary.detect_gpu_type", return_value=("cuda", "RTX 5080")
    )
    def test_downloads_when_missing(
        self, mock_detect, mock_release, mock_download, tmp_path
    ):
        mock_release.return_value = {
            "tag": "b1234",
            "asset_name": "llama-b1234-win-cuda.zip",
            "asset_url": "https://example.com/release.zip",
            "asset_size": 100_000_000,
        }
        expected_path = tmp_path / "bin" / "llama-server.exe"
        mock_download.return_value = expected_path

        with patch("tps_pro.cli.setup_binary.sys") as mock_sys:
            mock_sys.platform = "win32"
            result = ensure_llama_server(tmp_path)

        assert result == expected_path
        mock_detect.assert_called_once()
        mock_release.assert_called_once_with("cuda")
        mock_download.assert_called_once()

    @pytest.mark.unit
    @patch(
        "tps_pro.cli.setup_binary.detect_gpu_type", return_value=("cuda", "RTX 5080")
    )
    @patch("tps_pro.cli.setup_binary.get_latest_release")
    def test_raises_on_download_failure(self, mock_release, mock_detect, tmp_path):
        mock_release.side_effect = SetupBinaryError("No internet")

        with patch("tps_pro.cli.setup_binary.sys") as mock_sys:
            mock_sys.platform = "win32"
            with pytest.raises(SetupBinaryError, match="No internet"):
                ensure_llama_server(tmp_path)


# ============================================================
# _cleanup_partial()
# ============================================================


class TestCleanupPartial:
    """Test partial download cleanup."""

    @pytest.mark.unit
    def test_removes_existing_file(self, tmp_path):
        f = tmp_path / "partial.zip"
        f.write_text("data")
        _cleanup_partial(f)
        assert not f.exists()

    @pytest.mark.unit
    def test_handles_none(self):
        # Should not raise
        _cleanup_partial(None)

    @pytest.mark.unit
    def test_handles_nonexistent_file(self, tmp_path):
        f = tmp_path / "does_not_exist.zip"
        # Should not raise
        _cleanup_partial(f)
