"""Shared make_ctx / make_perf / make_mock_ctx factories for the test suite.

This module is importable from any test file. The conftest.py fixture
delegates to make_ctx_from_defaults.

Consolidated helpers:
    - make_ctx_from_defaults() -- canonical mock AppContext builder
    - make_perf()             -- canonical PerfResult builder
    - make_mock_ctx()         -- legacy mock ctx builder (HTTP mock pre-wired)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import requests

# Comprehensive default values covering all fields used across the test suite.
_CTX_DEFAULTS = dict(
    naked_engine={"threads": 8, "context": 4096, "n_gpu_layers": 99},
    is_moe=False,
    moe_sweep_center=4,
    default_experts=2,
    max_experts=4,
    max_gpu_layers=99,
    default_gpu_layers=99,
    max_threads=8,
    moe_sweep_max=16,
    numa_nodes=1,
    model_path=SimpleNamespace(
        name="test.gguf", parent=SimpleNamespace(parent="/tmp"), stem="test"
    ),
    server_path=Path("/tmp/llama-server"),
    chat_template_path=Path(""),
    server_url="http://localhost:8090",
    port=8090,
    _port_alt=8091,
    results_dir=SimpleNamespace(
        **{"__truediv__": lambda self, x: SimpleNamespace(exists=lambda: False)}
    ),
    optuna_db="sqlite:///test.db",
    config={},
    http=None,  # replaced per-call with a fresh MagicMock
    server_proc=None,
    active_server_proc=None,
    _dying_server_proc=None,
    skip_flags=set(),
    debug=False,
    fail_fast=False,
    dry_run=False,
    bench_path=None,
    kl_baseline_cache=None,
    no_jinja=False,
    model_size_class="medium",
    model_size_gb=2.0,
    vram_total_mb=8192.0,
    expert_override_key="",
    quality_baseline=None,
    fresh_run=False,
    skip_quality=False,
    lookup_cache_file="",
    arch={"type": "dense"},
    hw={},
)


def make_ctx_from_defaults(**overrides):
    """Build a minimal mock ctx (SimpleNamespace) with sensible defaults.

    This is the canonical implementation used by the ``make_ctx`` fixture
    in conftest.py. It is also importable directly for test files that
    cannot easily use the fixture (e.g., module-level helpers).
    """
    merged = dict(_CTX_DEFAULTS)
    # Always supply a fresh MagicMock for http unless caller overrides
    if "http" not in overrides:
        merged["http"] = MagicMock(spec=requests.Session)
    merged.update(overrides)
    return SimpleNamespace(**merged)


def make_perf(**overrides) -> SimpleNamespace:
    """Build a PerfResult-like SimpleNamespace with sensible defaults.

    Canonical factory -- import from here instead of duplicating in each test file.
    """
    defaults = dict(
        tps=50.0,
        ttft=500.0,
        prompt_tps=300.0,
        total_ms=1000.0,
        vram_used_mb=4096.0,
        vram_total_mb=8192.0,
        tps_std=1.0,
        tps_cv=0.02,
        n_runs=3,
        large_tps=None,
        concurrent_total_tps=None,
        concurrent_avg_tps=None,
        concurrent_avg_ttft=None,
        concurrent_avg_wall_ms=None,
        concurrent_max_wall_ms=None,
        concurrent_success_rate=None,
        concurrent_users=None,
        quality_factor=None,
        load_time_ms=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_mock_ctx(**overrides) -> SimpleNamespace:
    """Build a mock ctx with an HTTP mock pre-configured for measurement tests.

    Canonical factory -- import from here instead of duplicating in each test file.
    """
    return make_ctx_from_defaults(
        config={"pareto": False},
        **overrides,
    )
