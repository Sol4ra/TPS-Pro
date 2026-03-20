"""Shared test fixtures and helpers for the tps_pro test suite.

NOTE on bare MagicMock() usage (213 instances across 26 test files)
====================================================================
The codebase has many ``MagicMock()`` calls without ``spec=``.  Bare mocks
silently accept any attribute access, which can hide real AttributeError
bugs.  The highest-value mocks to spec are:

  - ``MagicMock(spec=AppContext)`` — for ctx mocks that represent the
    application context.  Catches typos in field names.
  - ``MagicMock(spec=ServerProcess)`` — for server process mocks.
    Catches typos in proc/stderr_lines access.
  - ``MagicMock(spec=requests.Session)`` — for HTTP session mocks.
    Already applied in _ctx_factory.py and several test files.

The canonical ``make_ctx_from_defaults()`` factory in ``_ctx_factory.py``
uses ``SimpleNamespace`` (not MagicMock) for ctx, which provides similar
protection since SimpleNamespace raises AttributeError on missing attrs.
New tests should prefer the factory over bare MagicMock for ctx objects.
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

# Ensure tests/ directory is on sys.path so `from _ctx_factory import ...` works
# inside test files that import it at function scope.
_tests_dir = str(Path(__file__).resolve().parent)
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

# Force-import the real pipeline module before test_cli.py has a chance to
# install a stub into sys.modules.  test_cli.py's _ensure_stub() checks
# ``if name in sys.modules`` and skips the stub when the real module is
# already loaded.  Without this early import, alphabetical collection order
# (test_bench -> test_cli -> test_e2e) means test_cli.py replaces pipeline
# with a MagicMock-based stub that lacks attributes like kill_server.
import tps_pro.pipeline  # noqa: F401, E402

# ===================================================================
# GGUF binary helpers — used by test_models.py (and any future tests
# that need synthetic GGUF files).
# ===================================================================


def build_gguf_bytes(metadata_kvs):
    """Build a minimal GGUF file (v3) in memory.

    Args:
        metadata_kvs: list of (key_str, vtype_int, value_bytes) tuples.

    Returns:
        bytes: A valid GGUF header with the given metadata.
    """
    buf = bytearray()
    buf += b"GGUF"  # magic
    buf += (3).to_bytes(4, "little")  # version 3
    buf += (0).to_bytes(8, "little")  # tensor_count = 0
    buf += len(metadata_kvs).to_bytes(8, "little")  # metadata_kv_count

    for key, vtype, val_bytes in metadata_kvs:
        key_enc = key.encode("utf-8")
        buf += len(key_enc).to_bytes(8, "little")
        buf += key_enc
        buf += vtype.to_bytes(4, "little")
        buf += val_bytes
    return bytes(buf)


def gguf_string_value(s):
    """Encode a string value for GGUF (vtype=8)."""
    enc = s.encode("utf-8")
    return len(enc).to_bytes(8, "little") + enc


def gguf_uint32_value(n):
    """Encode a uint32 value for GGUF (vtype=4)."""
    return n.to_bytes(4, "little")


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def tmp_gguf_file(tmp_path):
    """Factory fixture: returns a callable that writes GGUF bytes to a temp file.

    Usage::

        def test_something(tmp_gguf_file):
            path = tmp_gguf_file([("key", 4, gguf_uint32_value(32))])
            assert path.exists()
    """

    def _write(metadata_kvs):
        data = build_gguf_bytes(metadata_kvs)
        path = tmp_path / "test_model.gguf"
        path.write_bytes(data)
        return path

    return _write


@pytest.fixture
def tmp_sized_file(tmp_path):
    """Factory fixture: returns a callable that creates a sparse file of a given size.

    Usage::

        def test_classify(tmp_sized_file):
            path = tmp_sized_file(5 * 1024**3)  # 5 GB sparse file
    """

    def _make(size_bytes):
        path = tmp_path / "sized_model.gguf"
        with open(path, "wb") as f:
            f.seek(size_bytes - 1)
            f.write(b"\0")
        return path

    return _make


# ===================================================================
# Shared make_ctx factory fixture
# ===================================================================

_CTX_DEFAULTS = dict(
    naked_engine={"threads": 8, "context": 4096, "n_gpu_layers": 99},
    is_moe=False,
    moe_sweep_center=4,
    default_experts=2,
    max_experts=4,
    max_gpu_layers=99,
    default_gpu_layers=99,
    max_threads=8,
    numa_nodes=1,
    model_path=SimpleNamespace(
        name="test.gguf", parent=SimpleNamespace(parent="/tmp"), stem="test"
    ),
    server_path="/tmp/llama-server",
    chat_template_path="",
    server_url="http://localhost:8090",
    port=8090,
    results_dir=SimpleNamespace(
        **{"__truediv__": lambda self, x: SimpleNamespace(exists=lambda: False)}
    ),
    optuna_db="sqlite:///test.db",
    config={"pareto": False},
    server_proc=None,
    skip_flags=set(),
    debug=False,
    bench_path=None,
    kl_baseline_cache=None,
    fail_fast=False,
)


def _make_ctx_from_defaults(**overrides):
    """Build a minimal mock ctx (SimpleNamespace) with sensible defaults.

    This is the canonical implementation used by the ``make_ctx`` fixture.
    It is also available as a plain function for test files that cannot
    easily use the fixture (e.g., module-level helpers).
    """
    merged = dict(_CTX_DEFAULTS)
    # Always supply a fresh MagicMock for http unless caller overrides
    if "http" not in overrides:
        merged["http"] = MagicMock(spec=requests.Session)
    merged.update(overrides)
    return SimpleNamespace(**merged)


@pytest.fixture
def make_ctx():
    """Factory fixture that builds a mock ctx SimpleNamespace.

    Usage::

        def test_something(make_ctx):
            ctx = make_ctx(is_moe=True, max_threads=16)
    """
    return _make_ctx_from_defaults
