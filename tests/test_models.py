"""Tests for model classification, quantization recommendation, and GGUF parsing from models.py.

Functions are copied directly to avoid importing from the package
(which would trigger state.py's module-level CLI arg parsing).
"""
import os
import struct
import tempfile
import unittest
from pathlib import Path


# ---------------------------------------------------------------------------
# classify_model -- copied from models.py (no internal imports)
# ---------------------------------------------------------------------------
def classify_model(model_path):
    """Classify model by file size: tiny/small/medium/large. Returns (class, size_gb)."""
    try:
        size_gb = round(Path(model_path).stat().st_size / (1024**3), 2)
        if size_gb < 2:
            return ("tiny", size_gb)
        elif size_gb < 8:
            return ("small", size_gb)
        elif size_gb < 20:
            return ("medium", size_gb)
        else:
            return ("large", size_gb)
    except (OSError, ValueError):
        return ("unknown", 0.0)


# ---------------------------------------------------------------------------
# recommend_quant -- copied from models.py (uses classify_model above)
# ---------------------------------------------------------------------------
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
        return {"recommended": "Q2_K or offload",
                "reasoning": f"Model ({size_gb:.1f}GB) exceeds VRAM ({total_vram:.1f}GB)",
                "estimated_vram_gb": size_gb * 0.6}


# ---------------------------------------------------------------------------
# read_gguf_metadata + detect_model_layers -- copied from models.py
# ---------------------------------------------------------------------------
def read_gguf_metadata(model_path):
    """Read all key-value metadata from a GGUF file."""
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
    except Exception:
        return {}


def detect_model_layers(model_path):
    """Read layer count from GGUF metadata. Returns int or None."""
    metadata = read_gguf_metadata(model_path)
    for key, val in metadata.items():
        if key.endswith(".block_count"):
            return int(val)
    return None


# ===================================================================
# Helper: build a minimal GGUF binary with metadata
# ===================================================================
def _build_gguf_bytes(metadata_kvs):
    """Build a minimal GGUF file (v3) in memory.

    Args:
        metadata_kvs: list of (key_str, vtype_int, value_bytes) tuples.

    Returns:
        bytes: A valid GGUF header with the given metadata.
    """
    buf = bytearray()
    buf += b"GGUF"                                       # magic
    buf += (3).to_bytes(4, "little")                     # version 3
    buf += (0).to_bytes(8, "little")                     # tensor_count = 0
    buf += len(metadata_kvs).to_bytes(8, "little")       # metadata_kv_count

    for key, vtype, val_bytes in metadata_kvs:
        key_enc = key.encode("utf-8")
        buf += len(key_enc).to_bytes(8, "little")
        buf += key_enc
        buf += vtype.to_bytes(4, "little")
        buf += val_bytes
    return bytes(buf)


def _gguf_string_value(s):
    """Encode a string value for GGUF (vtype=8)."""
    enc = s.encode("utf-8")
    return len(enc).to_bytes(8, "little") + enc


def _gguf_uint32_value(n):
    """Encode a uint32 value for GGUF (vtype=4)."""
    return n.to_bytes(4, "little")


# ===================================================================
# Tests
# ===================================================================

class TestClassifyModel(unittest.TestCase):
    """Tests for classify_model()."""

    def _make_temp_file(self, size_bytes):
        """Create a sparse temporary file of a given size. Returns path."""
        fd, path = tempfile.mkstemp(suffix=".gguf")
        os.close(fd)
        with open(path, "wb") as f:
            f.seek(size_bytes - 1)
            f.write(b"\0")
        self.addCleanup(os.unlink, path)
        return path

    def test_tiny_model(self):
        """File < 2GB is classified as tiny."""
        path = self._make_temp_file(1 * 1024**3)  # 1 GB
        cls, size = classify_model(path)
        self.assertEqual(cls, "tiny")
        self.assertAlmostEqual(size, 1.0, places=1)

    def test_small_model(self):
        """File >= 2GB and < 8GB is classified as small."""
        path = self._make_temp_file(5 * 1024**3)  # 5 GB
        cls, size = classify_model(path)
        self.assertEqual(cls, "small")
        self.assertAlmostEqual(size, 5.0, places=1)

    def test_medium_model(self):
        """File >= 8GB and < 20GB is classified as medium."""
        path = self._make_temp_file(10 * 1024**3)  # 10 GB
        cls, size = classify_model(path)
        self.assertEqual(cls, "medium")
        self.assertAlmostEqual(size, 10.0, places=1)

    def test_large_model(self):
        """File >= 20GB is classified as large."""
        path = self._make_temp_file(25 * 1024**3)  # 25 GB
        cls, size = classify_model(path)
        self.assertEqual(cls, "large")
        self.assertAlmostEqual(size, 25.0, places=1)

    def test_nonexistent_file(self):
        """Nonexistent file returns ('unknown', 0.0)."""
        cls, size = classify_model("/nonexistent/path/model.gguf")
        self.assertEqual(cls, "unknown")
        self.assertEqual(size, 0.0)

    def test_boundary_just_below_2gb(self):
        """File just below 2GB boundary stays tiny.
        Need to be far enough below 2GB that round(size, 2) < 2.0.
        1.99 GB = 1.99 * 1024**3 bytes.
        """
        path = self._make_temp_file(int(1.99 * 1024**3))
        cls, _ = classify_model(path)
        self.assertEqual(cls, "tiny")

    def test_boundary_exactly_2gb(self):
        """File exactly 2GB rounds to 2.0 and is classified as small."""
        path = self._make_temp_file(2 * 1024**3)
        cls, _ = classify_model(path)
        self.assertEqual(cls, "small")


class TestRecommendQuant(unittest.TestCase):
    """Tests for recommend_quant()."""

    def _fake_classify(self, size_gb):
        """Return a classify_fn that always reports a fixed size."""
        def _cls(path):
            if size_gb < 2:
                return ("tiny", size_gb)
            elif size_gb < 8:
                return ("small", size_gb)
            elif size_gb < 20:
                return ("medium", size_gb)
            else:
                return ("large", size_gb)
        return _cls

    def test_no_gpus_returns_unknown(self):
        result = recommend_quant("dummy.gguf", [], classify_fn=self._fake_classify(5.0))
        self.assertEqual(result["recommended"], "unknown")

    def test_plenty_of_headroom_returns_Q6_K(self):
        """24GB VRAM - 5GB model = 19GB headroom > 4 -> Q6_K."""
        gpus = [{"vram_total_gb": 24}]
        result = recommend_quant("dummy.gguf", gpus, classify_fn=self._fake_classify(5.0))
        self.assertEqual(result["recommended"], "Q6_K")

    def test_moderate_headroom_returns_Q4_K_M(self):
        """10GB VRAM - 7GB model = 3GB headroom (2 < 3 <= 4) -> Q4_K_M."""
        gpus = [{"vram_total_gb": 10}]
        result = recommend_quant("dummy.gguf", gpus, classify_fn=self._fake_classify(7.0))
        self.assertEqual(result["recommended"], "Q4_K_M")

    def test_tight_fit_returns_Q3_K_S(self):
        """9GB VRAM - 8GB model = 1GB headroom (0 < 1 <= 2) -> Q3_K_S."""
        gpus = [{"vram_total_gb": 9}]
        result = recommend_quant("dummy.gguf", gpus, classify_fn=self._fake_classify(8.0))
        self.assertEqual(result["recommended"], "Q3_K_S")

    def test_exceeds_vram_returns_offload(self):
        """6GB VRAM - 10GB model = -4GB headroom -> Q2_K or offload."""
        gpus = [{"vram_total_gb": 6}]
        result = recommend_quant("dummy.gguf", gpus, classify_fn=self._fake_classify(10.0))
        self.assertEqual(result["recommended"], "Q2_K or offload")

    def test_multi_gpu_vram_summed(self):
        """Two 12GB GPUs = 24GB total; 5GB model -> 19GB headroom -> Q6_K."""
        gpus = [{"vram_total_gb": 12}, {"vram_total_gb": 12}]
        result = recommend_quant("dummy.gguf", gpus, classify_fn=self._fake_classify(5.0))
        self.assertEqual(result["recommended"], "Q6_K")

    def test_estimated_vram_scales_with_quant(self):
        """Estimated VRAM should be scaled by the quant-specific factor."""
        gpus = [{"vram_total_gb": 24}]
        result = recommend_quant("dummy.gguf", gpus, classify_fn=self._fake_classify(5.0))
        # Q6_K: estimated = size * 1.1
        self.assertAlmostEqual(result["estimated_vram_gb"], 5.0 * 1.1, places=2)

    def test_headroom_exactly_4_returns_Q4_K_M(self):
        """Headroom == 4 is NOT > 4, so falls to the >2 bracket -> Q4_K_M."""
        gpus = [{"vram_total_gb": 9}]
        result = recommend_quant("dummy.gguf", gpus, classify_fn=self._fake_classify(5.0))
        # headroom = 9 - 5 = 4, not > 4, so Q4_K_M
        self.assertEqual(result["recommended"], "Q4_K_M")


class TestReadGgufMetadata(unittest.TestCase):
    """Tests for read_gguf_metadata()."""

    def test_nonexistent_file(self):
        self.assertEqual(read_gguf_metadata("/no/such/file.gguf"), {})

    def test_non_gguf_file(self):
        """File that doesn't start with GGUF magic returns empty dict."""
        fd, path = tempfile.mkstemp(suffix=".bin")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with open(path, "wb") as f:
            f.write(b"NOT_GGUF_DATA_HERE_1234567890")
        self.assertEqual(read_gguf_metadata(path), {})

    def test_reads_string_metadata(self):
        """Parse a GGUF file with one string metadata entry."""
        kvs = [
            ("general.architecture", 8, _gguf_string_value("llama")),
        ]
        data = _build_gguf_bytes(kvs)
        fd, path = tempfile.mkstemp(suffix=".gguf")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with open(path, "wb") as f:
            f.write(data)

        metadata = read_gguf_metadata(path)
        self.assertEqual(metadata.get("general.architecture"), "llama")

    def test_reads_uint32_metadata(self):
        """Parse a GGUF file with a uint32 metadata entry."""
        kvs = [
            ("llama.block_count", 4, _gguf_uint32_value(32)),
        ]
        data = _build_gguf_bytes(kvs)
        fd, path = tempfile.mkstemp(suffix=".gguf")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with open(path, "wb") as f:
            f.write(data)

        metadata = read_gguf_metadata(path)
        self.assertEqual(metadata.get("llama.block_count"), 32)

    def test_reads_multiple_entries(self):
        """Parse a GGUF file with multiple metadata entries of different types."""
        kvs = [
            ("general.architecture", 8, _gguf_string_value("llama")),
            ("llama.block_count", 4, _gguf_uint32_value(40)),
        ]
        data = _build_gguf_bytes(kvs)
        fd, path = tempfile.mkstemp(suffix=".gguf")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with open(path, "wb") as f:
            f.write(data)

        metadata = read_gguf_metadata(path)
        self.assertEqual(len(metadata), 2)
        self.assertEqual(metadata["general.architecture"], "llama")
        self.assertEqual(metadata["llama.block_count"], 40)


class TestDetectModelLayers(unittest.TestCase):
    """Tests for detect_model_layers()."""

    def test_returns_layer_count_from_gguf(self):
        """Should return integer layer count when block_count key exists."""
        kvs = [
            ("llama.block_count", 4, _gguf_uint32_value(32)),
        ]
        data = _build_gguf_bytes(kvs)
        fd, path = tempfile.mkstemp(suffix=".gguf")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with open(path, "wb") as f:
            f.write(data)

        self.assertEqual(detect_model_layers(path), 32)

    def test_returns_none_when_no_block_count(self):
        """Should return None when no block_count key exists."""
        kvs = [
            ("general.architecture", 8, _gguf_string_value("llama")),
        ]
        data = _build_gguf_bytes(kvs)
        fd, path = tempfile.mkstemp(suffix=".gguf")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with open(path, "wb") as f:
            f.write(data)

        self.assertIsNone(detect_model_layers(path))

    def test_returns_none_for_nonexistent_file(self):
        self.assertIsNone(detect_model_layers("/no/such/file.gguf"))

    def test_works_with_different_architectures(self):
        """block_count key may have any prefix (e.g., qwen2.block_count)."""
        kvs = [
            ("qwen2.block_count", 4, _gguf_uint32_value(64)),
        ]
        data = _build_gguf_bytes(kvs)
        fd, path = tempfile.mkstemp(suffix=".gguf")
        os.close(fd)
        self.addCleanup(os.unlink, path)
        with open(path, "wb") as f:
            f.write(data)

        self.assertEqual(detect_model_layers(path), 64)


if __name__ == "__main__":
    unittest.main()
