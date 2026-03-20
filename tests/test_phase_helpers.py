"""Tests for phases/_helpers.py — build_phase_config, bench_score, get_moe_config,
_get_val.

Uses SimpleNamespace for mock ctx and patch for load_phase_results.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tps_pro.phases._helpers import (
    _get_val,
    _phase_config_cache,
    bench_score,
    build_phase_config,
    get_moe_config,
)


@pytest.fixture(autouse=True)
def _clear_phase_config_cache():
    """Clear the module-level phase config cache between tests."""
    _phase_config_cache.clear()
    yield
    _phase_config_cache.clear()


def _make_ctx(**overrides):
    """Build a minimal ctx namespace for phase helper tests."""
    from _ctx_factory import make_ctx_from_defaults

    return make_ctx_from_defaults(
        naked_engine={"threads": 8, "context": 4096},
        **overrides,
    )


# ===================================================================
# build_phase_config
# ===================================================================


@pytest.mark.unit
def test_build_phase_config_no_completed_phases():
    """When no phases have results, returns a copy of naked_engine."""
    ctx = _make_ctx()

    with patch(
        "tps_pro.phases._helpers.load_phase_results",
        return_value=None,
    ):
        config = build_phase_config(ctx)

    assert config == {"threads": 8, "context": 4096}
    # Must be a new dict, not the same object
    assert config is not ctx.naked_engine


@pytest.mark.unit
def test_build_phase_config_merges_completed_phases():
    """Completed phases' best_params are merged onto naked_engine."""
    ctx = _make_ctx()

    def mock_load(ctx_, phase_name):
        if phase_name == "gpu":
            return {"best_params": {"n_gpu_layers": 99}}
        if phase_name == "core_engine":
            return {"best_params": {"threads": 16, "batch_size": 512}}
        return None

    with patch(
        "tps_pro.phases._helpers.load_phase_results",
        side_effect=mock_load,
    ):
        config = build_phase_config(ctx)

    assert config["n_gpu_layers"] == 99
    assert config["threads"] == 16  # overridden by core_engine
    assert config["batch_size"] == 512
    assert config["context"] == 4096  # untouched from naked_engine


@pytest.mark.unit
def test_build_phase_config_include_phases_filter():
    """Only phases in include_phases are merged."""
    ctx = _make_ctx()

    def mock_load(ctx_, phase_name):
        if phase_name == "gpu":
            return {"best_params": {"n_gpu_layers": 99}}
        if phase_name == "core_engine":
            return {"best_params": {"threads": 16}}
        return None

    with patch(
        "tps_pro.phases._helpers.load_phase_results",
        side_effect=mock_load,
    ):
        config = build_phase_config(ctx, include_phases=["gpu"])

    assert config["n_gpu_layers"] == 99
    # core_engine was not included, so threads stays at 8
    assert config["threads"] == 8


@pytest.mark.unit
def test_build_phase_config_does_not_mutate_naked_engine():
    """build_phase_config must never mutate ctx.naked_engine."""
    ctx = _make_ctx()
    original_engine = dict(ctx.naked_engine)

    def mock_load(ctx_, phase_name):
        if phase_name == "gpu":
            return {"best_params": {"n_gpu_layers": 99}}
        return None

    with patch(
        "tps_pro.phases._helpers.load_phase_results",
        side_effect=mock_load,
    ):
        build_phase_config(ctx)

    assert ctx.naked_engine == original_engine


# ===================================================================
# bench_score
# ===================================================================


@pytest.mark.unit
def testbench_score_with_dict():
    """bench_score works with plain dicts."""
    p = {"tps": 100.0, "prompt_tps": 500.0}
    score = bench_score(p)
    # No baseline -> pp_ratio=1.0 -> score = 100 * (0.85 + 0.15*1.0) = 100
    assert score == pytest.approx(100.0)


@pytest.mark.unit
def testbench_score_with_baseline():
    """bench_score applies prompt_tps ratio relative to baseline."""
    p = {"tps": 100.0, "prompt_tps": 1000.0}
    baseline = {"tps": 80.0, "prompt_tps": 500.0}
    score = bench_score(p, baseline=baseline)
    # pp_ratio = min(1000/500, 2.0) = 2.0
    # score = 100 * (0.85 + 0.15 * 2.0) = 100 * 1.15 = 115
    assert score == pytest.approx(115.0)


@pytest.mark.unit
def testbench_score_with_dataclass():
    """bench_score works with BenchResult dataclass."""
    from tps_pro.result_types import BenchResult

    p = BenchResult(tps=50.0, prompt_tps=200.0, ttft=100.0, total_ms=500.0)
    score = bench_score(p)
    assert score == pytest.approx(50.0)


@pytest.mark.unit
def testbench_score_zero_tps():
    """Zero tps returns 0.0 score."""
    p = {"tps": 0.0, "prompt_tps": 500.0}
    assert bench_score(p) == 0.0


@pytest.mark.unit
def testbench_score_pp_ratio_capped_at_2x():
    """Prompt_tps ratio is capped at 2.0x baseline."""
    p = {"tps": 100.0, "prompt_tps": 10000.0}
    baseline = {"tps": 80.0, "prompt_tps": 100.0}
    score = bench_score(p, baseline=baseline)
    # pp_ratio = min(10000/100, 2.0) = 2.0
    assert score == pytest.approx(100.0 * (0.85 + 0.15 * 2.0))


# ===================================================================
# get_moe_config
# ===================================================================


@pytest.mark.unit
def testget_moe_config_not_moe():
    """Non-MoE model returns empty dict."""
    ctx = _make_ctx(is_moe=False)
    assert get_moe_config(ctx) == {}


@pytest.mark.unit
def testget_moe_config_with_p1a_results():
    """When p1a_results has best_params, use those."""
    ctx = _make_ctx(is_moe=True, moe_sweep_center=4, default_experts=2)
    p1a = {"best_params": {"n_cpu_moe": 8, "expert_used_count": 4}}

    result = get_moe_config(ctx, p1a_results=p1a)
    assert result == {"n_cpu_moe": 8, "expert_used_count": 4}


@pytest.mark.unit
def testget_moe_config_defaults_from_ctx():
    """When p1a_results missing, falls back to ctx defaults."""
    ctx = _make_ctx(is_moe=True, moe_sweep_center=6, default_experts=3)
    p1a = {"best_params": {}}  # no specific values

    result = get_moe_config(ctx, p1a_results=p1a)
    assert result["n_cpu_moe"] == 6
    assert result["expert_used_count"] == 3


@pytest.mark.unit
def testget_moe_config_from_saved_phase():
    """When no p1a_results, loads from saved 'moe' phase results."""
    ctx = _make_ctx(is_moe=True, moe_sweep_center=4, default_experts=2)

    with patch("tps_pro.phases._helpers.load_phase_results") as mock_load:
        mock_load.return_value = {"best_params": {"n_cpu_moe": 12}}
        result = get_moe_config(ctx)

    assert result["n_cpu_moe"] == 12
    assert result["expert_used_count"] == 2  # default


@pytest.mark.unit
def testget_moe_config_no_saved_results():
    """When no p1a_results and no saved results, returns ctx defaults."""
    ctx = _make_ctx(is_moe=True, moe_sweep_center=4, default_experts=2)

    with patch(
        "tps_pro.phases._helpers.load_phase_results",
        return_value=None,
    ):
        result = get_moe_config(ctx)

    assert result == {"n_cpu_moe": 4, "expert_used_count": 2}


# ===================================================================
# _get_val
# ===================================================================


@pytest.mark.unit
def test_get_val_from_dict():
    """_get_val extracts values from dicts."""
    d = {"tps": 100.0, "prompt_tps": 500.0}
    assert _get_val(d, "tps") == 100.0
    assert _get_val(d, "prompt_tps") == 500.0


@pytest.mark.unit
def test_get_val_from_dataclass():
    """_get_val extracts values from dataclass-like objects."""
    obj = SimpleNamespace(tps=42.0, prompt_tps=200.0)
    assert _get_val(obj, "tps") == 42.0
    assert _get_val(obj, "prompt_tps") == 200.0


@pytest.mark.unit
def test_get_val_missing_key_returns_default():
    """_get_val returns default when key is missing."""
    assert _get_val({"tps": 10}, "missing_key", 99) == 99
    assert _get_val(SimpleNamespace(tps=10), "missing_key", 99) == 99


@pytest.mark.unit
def test_get_val_default_is_zero():
    """_get_val default is 0 when not specified."""
    assert _get_val({}, "tps") == 0
    assert _get_val(SimpleNamespace(), "tps") == 0


