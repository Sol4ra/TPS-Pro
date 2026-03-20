"""Tests for model classification, quantization recommendation, and GGUF parsing
from models.py.

models.py is a leaf module (no internal package imports), so all functions
are imported directly without triggering state.py side effects.
"""

import pytest
from conftest import (
    gguf_string_value,
    gguf_uint32_value,
)

from tps_pro.models import (
    classify_model,
    detect_model_layers,
    read_gguf_metadata,
)

pytestmark = pytest.mark.unit

# ===================================================================
# classify_model
# ===================================================================


@pytest.mark.parametrize(
    "size_gb, expected_class",
    [
        (1, "tiny"),  # < 2 GB
        (5, "small"),  # 2..8 GB
        (10, "medium"),  # 8..20 GB
        (25, "large"),  # >= 20 GB
    ],
    ids=["tiny-1GB", "small-5GB", "medium-10GB", "large-25GB"],
)
def test_classify_model_size_brackets(tmp_sized_file, size_gb, expected_class):
    """Each size bracket maps to the correct class label."""
    path = tmp_sized_file(size_gb * 1024**3)
    cls, size = classify_model(str(path))
    assert cls == expected_class
    assert abs(size - size_gb) < 0.5


def test_classify_model_nonexistent():
    """Nonexistent file returns ('unknown', 0.0)."""
    cls, size = classify_model("/nonexistent/path/model.gguf")
    assert cls == "unknown"
    assert size == 0.0


@pytest.mark.parametrize(
    "size_bytes, expected_class",
    [
        (int(1.99 * 1024**3), "tiny"),  # just below 2 GB boundary
        (2 * 1024**3, "small"),  # exactly 2 GB
    ],
    ids=["just-below-2GB", "exactly-2GB"],
)
def test_classify_model_boundary(tmp_sized_file, size_bytes, expected_class):
    """Boundary values around the 2 GB threshold."""
    path = tmp_sized_file(size_bytes)
    cls, _ = classify_model(str(path))
    assert cls == expected_class


# ===================================================================
# read_gguf_metadata
# ===================================================================


def test_read_gguf_nonexistent():
    assert read_gguf_metadata("/no/such/file.gguf") == {}


def test_read_gguf_non_gguf_file(tmp_path):
    """File that doesn't start with GGUF magic returns empty dict."""
    path = tmp_path / "bad.bin"
    path.write_bytes(b"NOT_GGUF_DATA_HERE_1234567890")
    assert read_gguf_metadata(str(path)) == {}


def test_read_gguf_string_metadata(tmp_gguf_file):
    """Parse a GGUF file with one string metadata entry."""
    path = tmp_gguf_file([("general.architecture", 8, gguf_string_value("llama"))])
    metadata = read_gguf_metadata(str(path))
    assert metadata.get("general.architecture") == "llama"


def test_read_gguf_uint32_metadata(tmp_gguf_file):
    """Parse a GGUF file with a uint32 metadata entry."""
    path = tmp_gguf_file([("llama.block_count", 4, gguf_uint32_value(32))])
    metadata = read_gguf_metadata(str(path))
    assert metadata.get("llama.block_count") == 32


def test_read_gguf_multiple_entries(tmp_gguf_file):
    """Parse a GGUF file with multiple metadata entries of different types."""
    path = tmp_gguf_file(
        [
            ("general.architecture", 8, gguf_string_value("llama")),
            ("llama.block_count", 4, gguf_uint32_value(40)),
        ]
    )
    metadata = read_gguf_metadata(str(path))
    assert len(metadata) == 2
    assert metadata["general.architecture"] == "llama"
    assert metadata["llama.block_count"] == 40


# ===================================================================
# detect_model_layers
# ===================================================================


def test_detect_layers_from_gguf(tmp_gguf_file):
    """Should return integer layer count when block_count key exists."""
    path = tmp_gguf_file([("llama.block_count", 4, gguf_uint32_value(32))])
    assert detect_model_layers(str(path)) == 32


def test_detect_layers_no_block_count(tmp_gguf_file):
    """Should return None when no block_count key exists."""
    path = tmp_gguf_file([("general.architecture", 8, gguf_string_value("llama"))])
    assert detect_model_layers(str(path)) is None


def test_detect_layers_nonexistent():
    assert detect_model_layers("/no/such/file.gguf") is None


def test_detect_layers_different_architecture(tmp_gguf_file):
    """block_count key may have any prefix (e.g., qwen2.block_count)."""
    path = tmp_gguf_file([("qwen2.block_count", 4, gguf_uint32_value(64))])
    assert detect_model_layers(str(path)) == 64
