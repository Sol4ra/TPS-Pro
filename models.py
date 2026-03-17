"""
Model classification, GGUF metadata reading, architecture detection, and quantization recommendations.
"""

import logging
import struct
from pathlib import Path

logger = logging.getLogger(__name__)


def classify_model(model_path):
    """Classify model by file size: tiny/small/medium/large. Returns (class, size_gb)."""
    try:
        size_gb = round(Path(model_path).stat().st_size / (1024**3), 2)
        if size_gb < 2: return ("tiny", size_gb)
        elif size_gb < 8: return ("small", size_gb)
        elif size_gb < 20: return ("medium", size_gb)
        else: return ("large", size_gb)
    except (OSError, ValueError):
        return ("unknown", 0.0)


# ============================================================
# Unified GGUF Metadata Reader
# ============================================================

def read_gguf_metadata(model_path):
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

            def read_string():
                length = int.from_bytes(f.read(8), "little")
                return f.read(length).decode("utf-8", errors="replace")

            def read_value(vtype):
                if vtype == 0:    return int.from_bytes(f.read(1), "little")
                elif vtype == 1:  return int.from_bytes(f.read(1), "little", signed=True)
                elif vtype == 2:  return int.from_bytes(f.read(2), "little")
                elif vtype == 3:  return int.from_bytes(f.read(2), "little", signed=True)
                elif vtype == 4:  return int.from_bytes(f.read(4), "little")
                elif vtype == 5:  return int.from_bytes(f.read(4), "little", signed=True)
                elif vtype == 6:  return struct.unpack("<f", f.read(4))[0]
                elif vtype == 7:  return bool(f.read(1)[0])
                elif vtype == 8:  return read_string()
                elif vtype == 9:
                    elem_type = int.from_bytes(f.read(4), "little")
                    count = int.from_bytes(f.read(8), "little")
                    return [read_value(elem_type) for _ in range(count)]
                elif vtype == 10: return int.from_bytes(f.read(8), "little")
                elif vtype == 11: return int.from_bytes(f.read(8), "little", signed=True)
                elif vtype == 12: return struct.unpack("<d", f.read(8))[0]
                else: return None

            metadata = {}
            for _ in range(metadata_kv_count):
                key = read_string()
                vtype = int.from_bytes(f.read(4), "little")
                val = read_value(vtype)
                metadata[key] = val
            return metadata
    except (OSError, struct.error, UnicodeDecodeError, ValueError) as e:
        logger.debug("GGUF metadata read failed for %s: %s", model_path, e)
        return {}


def detect_model_layers(model_path):
    """Read layer count from GGUF metadata. Returns int or None."""
    metadata = read_gguf_metadata(model_path)
    for key, val in metadata.items():
        if key.endswith(".block_count"):
            return int(val)
    return None


def _detect_gguf_architecture(model_path):
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


def recommend_quant(model_path, gpus, classify_fn=None):
    """Recommend quantization based on model size and available VRAM."""
    if classify_fn is None:
        classify_fn = classify_model
    _, size_gb = classify_fn(model_path)
    total_vram = sum(g["vram_total_gb"] for g in gpus) if gpus else 0

    if total_vram == 0:
        return {"recommended": "unknown", "reasoning": "No GPU detected", "estimated_vram_gb": size_gb}

    headroom = total_vram - size_gb
    if headroom > 4:
        return {"recommended": "Q6_K", "reasoning": f"{headroom:.1f}GB headroom — Q6_K fits with room to spare",
                "estimated_vram_gb": size_gb * 1.1}
    elif headroom > 2:
        return {"recommended": "Q4_K_M", "reasoning": f"{headroom:.1f}GB headroom — Q4_K_M is the sweet spot",
                "estimated_vram_gb": size_gb * 0.85}
    elif headroom > 0:
        return {"recommended": "Q3_K_S", "reasoning": f"Tight fit ({headroom:.1f}GB) — Q3_K_S for minimum VRAM",
                "estimated_vram_gb": size_gb * 0.7}
    else:
        return {"recommended": "Q2_K or offload", "reasoning": f"Model ({size_gb:.1f}GB) exceeds VRAM ({total_vram:.1f}GB)",
                "estimated_vram_gb": size_gb * 0.6}
