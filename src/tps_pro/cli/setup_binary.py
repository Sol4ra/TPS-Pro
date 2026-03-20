"""Auto-download llama-server binary based on detected GPU type.

On first run, detects the user's GPU and downloads the correct
llama.cpp release binary from GitHub. No user interaction needed.

Error strategy:
    - No internet -> clear error message telling user to download manually
    - GitHub rate limit -> raise SetupBinaryError so caller can fall back
    - Extraction fails -> clean up partial download, raise SetupBinaryError
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import requests

from ..constants import HTTP_FORBIDDEN, HTTP_OK

logger = logging.getLogger(__name__)

__all__ = [
    "detect_gpu_type",
    "download_and_extract",
    "ensure_llama_server",
    "get_latest_release",
    "SetupBinaryError",
    "_verify_checksum",
    "_download_checksum",
]


class SetupBinaryError(Exception):
    """Raised when binary auto-download fails and manual intervention is needed."""


# ============================================================
# GPU Detection
# ============================================================


def _check_nvidia_gpu() -> str | None:
    """Check for NVIDIA GPU via pynvml or nvidia-smi. Returns GPU name or None."""
    # Try pynvml first (already a dependency of this project)
    try:
        import pynvml as _pynvml_mod
    except ImportError:
        _pynvml_mod = None  # type: ignore[assignment]

    if _pynvml_mod is not None:
        try:
            _pynvml_mod.nvmlInit()
            try:
                count = _pynvml_mod.nvmlDeviceGetCount()
                if count > 0:
                    handle = _pynvml_mod.nvmlDeviceGetHandleByIndex(0)
                    name = _pynvml_mod.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")
                    return name
            finally:
                _pynvml_mod.nvmlShutdown()
        except _pynvml_mod.NVMLError:
            pass

    # Fall back to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass

    return None


def _check_amd_gpu() -> bool:
    """Check for AMD GPU via rocminfo or HIP runtime."""
    try:
        result = subprocess.run(
            ["rocminfo"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and "GPU" in result.stdout:
            return True
    except (OSError, subprocess.SubprocessError):
        pass

    # Check for HIP runtime DLL on Windows
    if sys.platform == "win32":
        hip_paths = [
            Path("C:/Program Files/AMD/ROCm"),
            Path("C:/Windows/System32/amdhip64.dll"),
        ]
        for p in hip_paths:
            if p.exists():
                return True
    else:
        # Check for HIP on Linux
        hip_path = Path("/opt/rocm/bin/hipconfig")
        if hip_path.exists():
            return True

    return False


def detect_gpu_type() -> tuple[str, str]:
    """Detect the user's GPU type and return (backend, gpu_name).

    Returns:
        Tuple of (backend, gpu_name) where backend is one of:
        "cuda", "rocm", "vulkan", "cpu" and gpu_name is a
        human-readable description.
    """
    # Check NVIDIA first (most common for llama.cpp users)
    nvidia_name = _check_nvidia_gpu()
    if nvidia_name is not None:
        return "cuda", nvidia_name

    # Check AMD
    if _check_amd_gpu():
        return "rocm", "AMD GPU (ROCm)"

    # Vulkan fallback (most modern GPUs support Vulkan)
    # Check if vulkaninfo is available as a heuristic
    try:
        result = subprocess.run(
            ["vulkaninfo", "--summary"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return "vulkan", "Vulkan-compatible GPU"
    except (OSError, subprocess.SubprocessError):
        pass

    return "cpu", "No GPU detected"


# ============================================================
# GitHub Release Query
# ============================================================


def _match_asset(  # noqa: C901, PLR0912
    assets: list[dict[str, Any]],
    os_name: str,
    backend: str,
) -> dict[str, Any] | None:
    """Find the best matching asset for the given OS and GPU backend.

    Args:
        assets: List of GitHub release asset dicts.
        os_name: "win", "linux", or "macos".
        backend: "cuda", "rocm", "vulkan", or "cpu".

    Returns:
        The matching asset dict, or None if no match found.
    """
    # Normalize asset names for matching
    candidates = []
    for asset in assets:
        name_lower = asset["name"].lower()
        # Skip non-zip files
        if not name_lower.endswith(".zip"):
            continue
        candidates.append((name_lower, asset))

    # Filter by OS
    os_candidates = [(n, a) for n, a in candidates if os_name in n]
    if not os_candidates:
        return None

    # Match by backend preference
    if backend == "cuda":
        # Prefer CUDA builds — exclude cudart (runtime DLLs only, no binaries)
        for name, asset in os_candidates:
            if "cuda" in name and "cudart" not in name and "cu12" in name:
                return asset
        for name, asset in os_candidates:
            if "cuda" in name and "cudart" not in name:
                return asset
    elif backend == "rocm":
        # ROCm builds (Linux only typically)
        for name, asset in os_candidates:
            if "rocm" in name or "hip" in name:
                return asset
    elif backend == "vulkan":
        for name, asset in os_candidates:
            if "vulkan" in name:
                return asset

    # CPU fallback: find an asset without cuda/vulkan/rocm
    if backend == "cpu":
        for name, asset in os_candidates:
            if "cuda" not in name and "vulkan" not in name and "rocm" not in name:
                return asset

    # If specific backend not found, try vulkan, then cpu-only
    if backend not in ("vulkan", "cpu"):
        for name, asset in os_candidates:
            if "vulkan" in name:
                return asset
    for name, asset in os_candidates:
        if "cuda" not in name and "vulkan" not in name and "rocm" not in name:
            return asset

    return None


def _detect_os_name() -> str:
    """Return the OS identifier used in llama.cpp release asset names."""
    system = platform.system().lower()
    if system == "windows":
        return "win"
    if system == "darwin":
        return "macos"
    return "linux"


def get_latest_release(backend: str) -> dict[str, Any]:
    """Query GitHub API for the latest llama.cpp release and find the matching asset.

    Args:
        backend: GPU backend type ("cuda", "rocm", "vulkan", "cpu").

    Returns:
        Dict with keys: "tag", "asset_name", "asset_url", "asset_size".

    Raises:
        SetupBinaryError: If the release cannot be fetched or no matching asset found.
    """
    url = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    try:
        resp = requests.get(
            url,
            timeout=30,
            headers={"Accept": "application/vnd.github.v3+json"},
        )
    except requests.ConnectionError as err:
        raise SetupBinaryError(
            "No internet connection. Please download llama-server manually from:\n"
            "  https://github.com/ggml-org/llama.cpp/releases/latest"
        ) from err
    except requests.Timeout as err:
        raise SetupBinaryError(
            "GitHub request timed out. Please download llama-server manually from:\n"
            "  https://github.com/ggml-org/llama.cpp/releases/latest"
        ) from err

    if resp.status_code == HTTP_FORBIDDEN:
        raise SetupBinaryError(
            "GitHub API rate limit exceeded."
            " Please download llama-server manually"
            " from:\n"
            "  https://github.com/ggml-org/llama.cpp/releases/latest"
        )

    if resp.status_code != HTTP_OK:
        raise SetupBinaryError(
            f"GitHub API returned status {resp.status_code}. "
            "Please download llama-server manually from:\n"
            "  https://github.com/ggml-org/llama.cpp/releases/latest"
        )

    data = resp.json()
    tag = data.get("tag_name", "unknown")
    assets = data.get("assets", [])

    os_name = _detect_os_name()
    asset = _match_asset(assets, os_name, backend)
    if asset is None:
        raise SetupBinaryError(
            f"No matching release asset found for {os_name}/{backend}.\n"
            "Please download llama-server manually from:\n"
            f"  https://github.com/ggml-org/llama.cpp/releases/tag/{tag}"
        )

    return {
        "tag": tag,
        "asset_name": asset["name"],
        "asset_url": asset["browser_download_url"],
        "asset_size": asset.get("size", 0),
    }


# ============================================================
# Download & Extract
# ============================================================


def _verify_checksum(zip_path: Path, expected_hex: str) -> None:
    """Verify SHA256 checksum of downloaded file."""
    sha256 = hashlib.sha256()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    actual = sha256.hexdigest()
    if not hmac.compare_digest(actual, expected_hex.lower()):
        raise SetupBinaryError(
            f"Checksum mismatch: expected {expected_hex}, got {actual}. "
            "Download may be corrupted or tampered with."
        )


def _download_checksum(asset_url: str, asset_name: str) -> str | None:
    """Try to download SHA256 checksum for a release asset.

    Looks for a matching .sha256 file at the same base URL level.
    Returns the expected hex digest string, or None if not found.
    """
    # llama.cpp publishes SHA256 files alongside release assets
    checksum_url = asset_url + ".sha256"
    try:
        resp = requests.get(checksum_url, timeout=30)
        if resp.status_code == HTTP_OK:
            # Format is typically: "<hex>  <filename>" or just "<hex>"
            content = resp.text.strip()
            hex_digest = content.split()[0]
            # Validate it looks like a SHA256 hex string
            hex_chars = "0123456789abcdefABCDEF"
            is_valid = (
                len(hex_digest) == 64  # noqa: PLR2004
                and all(c in hex_chars for c in hex_digest)
            )
            if is_valid:
                return hex_digest
            logger.warning(
                "SHA256 file found but content format"
                " unexpected: %s",
                content[:100],
            )
            return None
        logger.debug(
            "No SHA256 checksum file at %s (status %d)",
            checksum_url,
            resp.status_code,
        )
    except (requests.ConnectionError, requests.Timeout):
        logger.debug(
            "Could not download SHA256 checksum from %s",
            checksum_url,
        )
    return None


def download_and_extract(asset_url: str, target_dir: Path) -> Path:  # noqa: C901, PLR0912, PLR0915
    """Download a release zip and extract llama-server from it.

    Args:
        asset_url: Direct download URL for the release asset.
        target_dir: Directory to extract into (e.g. <project_root>/bin/).

    Returns:
        Path to the extracted llama-server executable.

    Raises:
        SetupBinaryError: On download failure, extraction failure, or
            if llama-server is not found in the archive.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    tmp_zip = None

    try:
        # Download with progress indication
        resp = requests.get(asset_url, stream=True, timeout=300)
        resp.raise_for_status()

        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0

        tmp_fd, tmp_zip_str = tempfile.mkstemp(suffix=".zip")
        tmp_zip = Path(tmp_zip_str)
        with open(tmp_fd, "wb", closefd=True) as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded * 100 // total_size
                    mb_done = downloaded / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    msg = (
                        f"\r        Downloading:"
                        f" {mb_done:.0f}/{mb_total:.0f}"
                        f" MB ({pct}%)"
                    )
                    print(msg, end="", flush=True)
        print()  # newline after progress

    except requests.ConnectionError as err:
        _cleanup_partial(tmp_zip)
        raise SetupBinaryError(
            "Download failed (no internet). Please download llama-server manually."
        ) from err
    except requests.Timeout as err:
        _cleanup_partial(tmp_zip)
        raise SetupBinaryError(
            "Download timed out. Please download llama-server manually."
        ) from err
    except requests.HTTPError as err:
        _cleanup_partial(tmp_zip)
        raise SetupBinaryError(f"Download failed: {err}") from err
    except OSError as err:
        _cleanup_partial(tmp_zip)
        raise SetupBinaryError(f"Failed to write download: {err}") from err

    # Verify checksum if available
    expected_checksum = _download_checksum(asset_url, Path(asset_url).name)
    if expected_checksum is not None:
        try:
            _verify_checksum(tmp_zip, expected_checksum)
            logger.info("SHA256 checksum verified successfully.")
        except SetupBinaryError:
            _cleanup_partial(tmp_zip)
            raise
    else:
        logger.warning(
            "No SHA256 checksum available for this release asset. "
            "Skipping integrity verification."
        )

    # Extract (with Zip Slip protection)
    try:
        with zipfile.ZipFile(tmp_zip, "r") as zf:
            # Validate all member paths before extraction to prevent path traversal
            resolved_root = target_dir.resolve()
            for member in zf.namelist():
                member_path = (target_dir / member).resolve()
                try:
                    member_path.relative_to(resolved_root)
                except ValueError as err:
                    raise SetupBinaryError(
                        f"Zip contains path traversal entry: {member!r}. "
                        "Please download llama-server manually."
                    ) from err
            zf.extractall(target_dir)
    except (zipfile.BadZipFile, OSError) as err:
        _cleanup_partial(tmp_zip)
        # Clean up partial extraction
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        raise SetupBinaryError(
            f"Extraction failed: {err}\n"
            "Please download llama-server manually."
        ) from err
    finally:
        _cleanup_partial(tmp_zip)

    # Find llama-server executable in extracted files
    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    found = list(target_dir.rglob(exe_name))
    if not found:
        raise SetupBinaryError(
            f"{exe_name} not found in the downloaded archive.\n"
            "Please download llama-server manually."
        )

    # Prefer the shallowest path (fewest directory components)
    server_path = min(found, key=lambda p: len(p.parts))

    # On Unix, ensure executable permission
    if sys.platform != "win32":
        server_path.chmod(server_path.stat().st_mode | 0o755)

    return server_path


def _cleanup_partial(path: Path | None) -> None:
    """Remove a partial download file if it exists."""
    if path is not None:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


# ============================================================
# Main Entry Point
# ============================================================


def ensure_llama_server(project_root: Path) -> Path:
    """Ensure llama-server is available, downloading if necessary.

    Checks for an existing binary in <project_root>/bin/. If not found,
    auto-detects the GPU, fetches the latest release from GitHub, and
    extracts it.

    Args:
        project_root: Root directory of the tps_pro project.

    Returns:
        Path to the llama-server executable.

    Raises:
        SetupBinaryError: If auto-download fails and manual intervention
            is needed.
    """
    bin_dir = project_root / "bin"
    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"

    # Check if already downloaded
    if bin_dir.exists():
        found = list(bin_dir.rglob(exe_name))
        if found:
            best = min(found, key=lambda p: len(p.parts))
            if best.is_file():
                return best

    # Auto-detect GPU
    backend, gpu_name = detect_gpu_type()
    print(f"  [*] Detecting GPU... {gpu_name} ({backend.upper()})")

    # Get latest release
    release = get_latest_release(backend)
    tag = release["tag"]
    print(f"  [*] Downloading llama-server {tag}...", end=" ", flush=True)

    # Download and extract
    server_path = download_and_extract(release["asset_url"], bin_dir)
    print("done")

    return server_path
