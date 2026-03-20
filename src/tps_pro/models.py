"""
Model classification, GGUF metadata reading, architecture
detection, and quantization recommendations.

Error strategy (see errors.py for full documentation):
    - read_gguf_metadata(): returns empty dict on any read/parse error
      (logged at debug).  GGUF files may be incomplete, corrupted, or
      use unsupported format versions -- this must not crash the optimizer.
    - classify_model(): returns ("unknown", 0.0) if the file cannot be
      stat'd.
    - detect_model_layers(): returns None if metadata unavailable.
    - detect_gguf_architecture(): returns {"type": "dense"} default on error.
    - detect_skippable_flags(): returns empty set on error (logged at debug).
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any, Callable

from .result_types import ArchConfig

logger = logging.getLogger(__name__)

_MAX_ARRAY_DEPTH = 3
_MAX_ARRAY_COUNT = 500_000  # Qwen 3.5 tokenizer has 248K entries
_MAX_METADATA_BYTES = 256 * 1024 * 1024  # 256 MiB safety limit for metadata parsing
_MAX_METADATA_KV = 5_000
_MAX_GGUF_TYPE = 9  # maximum GGUF value type id (nested array)

# Model size classification thresholds (GB)
_VRAM_SMALL = 2
_VRAM_MEDIUM = 8
_VRAM_LARGE = 20

__all__ = [
    "classify_model",
    "read_gguf_metadata",
    "detect_model_layers",
    "detect_gguf_architecture",
    "detect_skippable_flags",
]


def classify_model(model_path: str | Path) -> tuple[str, float]:
    """Classify model by file size.

    Returns (class, size_gb) where class is
    tiny/small/medium/large.
    """
    try:
        size_gb = round(Path(model_path).stat().st_size / (1024**3), 2)
        if size_gb < _VRAM_SMALL:
            return ("tiny", size_gb)
        elif size_gb < _VRAM_MEDIUM:
            return ("small", size_gb)
        elif size_gb < _VRAM_LARGE:
            return ("medium", size_gb)
        else:
            return ("large", size_gb)
    except (OSError, ValueError):
        return ("unknown", 0.0)


# ============================================================
# Unified GGUF Metadata Reader
# ============================================================


def read_gguf_metadata(model_path: str | Path) -> dict[str, Any]:  # noqa: C901
    """Read all key-value metadata from a GGUF file.

    Returns dict of {key: value} or empty dict on failure.
    This is the single source of truth for GGUF parsing — all other
    functions that need GGUF metadata should call this.
    """
    try:
        p = Path(model_path)
        if not p.is_file():
            return {}
        with open(p, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                return {}
            _version = int.from_bytes(f.read(4), "little")
            _tensor_count = int.from_bytes(f.read(8), "little")
            metadata_kv_count = int.from_bytes(f.read(8), "little")
            if metadata_kv_count > _MAX_METADATA_KV:
                return {}

            def read_string():
                length = int.from_bytes(f.read(8), "little")
                if length > 1_000_000:  # noqa: PLR2004
                    raise ValueError("GGUF string too long")
                return f.read(length).decode("utf-8", errors="replace")

            def _read_array(_depth: int = 0):
                if _depth >= _MAX_ARRAY_DEPTH:
                    raise ValueError("GGUF array recursion depth exceeded")
                elem_type = int.from_bytes(f.read(4), "little")
                count = int.from_bytes(f.read(8), "little")
                if count > _MAX_ARRAY_COUNT:
                    raise ValueError("GGUF array too large")
                if elem_type == _MAX_GGUF_TYPE:  # nested array
                    return [_read_array(_depth + 1) for _ in range(count)]
                return [read_value(elem_type) for _ in range(count)]

            # GGUF value type -> reader function.  Each reader consumes the
            # correct number of bytes from the open file handle ``f``.
            vtype_readers: dict[int, Callable[[], Any]] = {
                0: lambda: int.from_bytes(f.read(1), "little"),
                1: lambda: int.from_bytes(f.read(1), "little", signed=True),
                2: lambda: int.from_bytes(f.read(2), "little"),
                3: lambda: int.from_bytes(f.read(2), "little", signed=True),
                4: lambda: int.from_bytes(f.read(4), "little"),
                5: lambda: int.from_bytes(f.read(4), "little", signed=True),
                6: lambda: struct.unpack("<f", f.read(4))[0],
                7: lambda: bool(f.read(1)[0]),
                8: read_string,
                9: _read_array,
                10: lambda: int.from_bytes(f.read(8), "little"),
                11: lambda: int.from_bytes(f.read(8), "little", signed=True),
                12: lambda: struct.unpack("<d", f.read(8))[0],
            }

            def read_value(vtype):
                reader = vtype_readers.get(vtype)
                return reader() if reader is not None else None

            metadata_start = f.tell()
            metadata = {}
            for _ in range(metadata_kv_count):
                if f.tell() - metadata_start > _MAX_METADATA_BYTES:
                    logger.debug(
                        "GGUF metadata exceeds %d bytes limit — truncating",
                        _MAX_METADATA_BYTES,
                    )
                    break
                key = read_string()
                vtype = int.from_bytes(f.read(4), "little")
                val = read_value(vtype)
                metadata[key] = val
            return metadata
    except (OSError, struct.error, UnicodeDecodeError, ValueError) as e:
        logger.debug("GGUF metadata read failed for %s: %s", model_path, e)
        return {}


def detect_model_layers(model_path: str | Path) -> int | None:
    """Read layer count from GGUF metadata. Returns int or None."""
    metadata = read_gguf_metadata(model_path)
    for key, val in metadata.items():
        if key.endswith(".block_count"):
            return int(val)
    return None


def detect_gguf_architecture(model_path: str | Path) -> ArchConfig:
    """Read GGUF metadata to detect model architecture (moe vs dense)."""
    arch_info = {"type": "dense"}
    try:
        metadata = read_gguf_metadata(model_path)
        if not metadata:
            return arch_info

        general_arch = metadata.get("general.architecture", "")
        num_experts = metadata.get(f"{general_arch}.expert_count")
        num_used = metadata.get(f"{general_arch}.expert_used_count")
        if num_experts and num_experts > 1:
            arch_info = {
                "type": "moe",
                "expert_override_key": f"{general_arch}.expert_used_count",
                "default_experts": int(num_used) if num_used else int(num_experts),
                "max_experts": int(num_experts),
            }
    except (OSError, struct.error, ValueError) as e:
        logger.warning("Could not read GGUF metadata for architecture detection: %s", e)
    return arch_info


def detect_skippable_flags(model_path: str | Path, n_gpu_layers: int) -> set[str]:
    """Detect engine flags that are irrelevant for this model + GPU config.

    Reads GGUF metadata and compares against the active GPU layer count to
    determine which flags can be fixed to their defaults instead of searched.

    Returns a set of flag names to skip (e.g. {"swa_full", "mlock"}).
    Never skips flash_attn (always beneficial to test).
    """
    import sys

    skip = set()
    try:
        metadata = read_gguf_metadata(model_path)
        if not metadata:
            return skip

        general_arch = metadata.get("general.architecture", "")
        block_count = None
        for key, val in metadata.items():
            if key.endswith(".block_count"):
                block_count = int(val)
                break

        # SWA: skip if model has no sliding window architecture
        has_swa = any(
            "sliding_window" in k or "attention.window" in k for k in metadata
        )
        if not has_swa:
            skip.add("swa_full")
            logger.debug("GGUF: no sliding window attention — skipping swa_full")

        # Detect MoE
        num_experts = metadata.get(f"{general_arch}.expert_count")
        is_moe = num_experts is not None and int(num_experts) > 1
        partially_offloaded = block_count is not None and n_gpu_layers < block_count

        # MoE with partial offload: op_offload causes 10x slowdown (issue #13241)
        # Copies all CPU expert weights to GPU over PCIe every prompt batch
        if is_moe and partially_offloaded:
            skip.add("op_offload")
            logger.debug(
                "GGUF: MoE with partial offload — skipping op_offload (default OFF)"
            )

        # MoE: no_mmap can cause memory errors (issue #14999)
        if is_moe:
            skip.add("no_mmap")
            logger.debug(
                "GGUF: MoE model — skipping no_mmap (default OFF, can cause errors)"
            )

        # Fully on GPU: skip CPU-only flags
        if block_count is not None and n_gpu_layers >= block_count:
            skip.add("mlock")
            skip.add("no_mmap")
            skip.add("cpu_strict")
            skip.add("cpu_strict_batch")
            skip.add("repack")  # CPU-only optimization, irrelevant when fully on GPU
            logger.debug(
                "GGUF: all %d layers on GPU (ngl=%d)"
                " — skipping mlock, no_mmap,"
                " cpu_strict, repack",
                block_count,
                n_gpu_layers,
            )

        # NUMA: useless on Windows (non-functional) and single-NUMA systems
        if sys.platform == "win32":
            skip.add("numa")
            logger.debug("GGUF: Windows detected — skipping numa (non-functional)")

    except (OSError, struct.error, ValueError) as e:
        logger.debug("Could not detect skippable flags: %s", e)

    return skip
