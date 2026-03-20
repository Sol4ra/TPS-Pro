"""Tests for result_types.py — frozen dataclasses with _DictAccessMixin.

Covers construction, from_dict, to_dict roundtrip, dict-style access,
immutability guards, NamedTuple behavior, and nested deserialization.
"""

from __future__ import annotations

import dataclasses

import pytest

from tps_pro.result_types import (
    BenchResult,
    ConcurrentLoadResult,
    ConcurrentUserResult,
    KLResult,
    NIAHPhaseResult,
    NIAHResult,
    NIAHTestResult,
    ParetoObjectives,
    PerfResult,
    PerfSample,
    PhaseResult,
    PPLResult,
    QualityResult,
    QualityTaskResult,
    TokenUncertaintyResult,
    TrialSummary,
)

# ===================================================================
# PerfSample
# ===================================================================


@pytest.mark.unit
class TestPerfSample:
    def test_construction_with_defaults(self):
        s = PerfSample(tps=10.0, ttft=50.0, prompt_tps=100.0, total_ms=200.0)
        assert s.tps == 10.0
        assert s.vram_used_mb is None
        assert s.vram_total_mb is None

    def test_from_dict_missing_keys(self):
        s = PerfSample.from_dict({})
        assert s.tps == 0.0
        assert s.ttft == 0.0
        assert s.vram_used_mb is None

    def test_from_dict_all_keys(self):
        data = {
            "tps": 5.0,
            "ttft": 10.0,
            "prompt_tps": 20.0,
            "total_ms": 30.0,
            "vram_used_mb": 1024.0,
            "vram_total_mb": 2048.0,
        }
        s = PerfSample.from_dict(data)
        assert s.vram_used_mb == 1024.0
        assert s.vram_total_mb == 2048.0

    def test_to_dict_roundtrip(self):
        original = PerfSample(tps=1.0, ttft=2.0, prompt_tps=3.0, total_ms=4.0)
        d = original.to_dict()
        restored = PerfSample.from_dict(d)
        assert restored == original

    def test_getitem(self):
        s = PerfSample(tps=7.5, ttft=1.0, prompt_tps=2.0, total_ms=3.0)
        assert s["tps"] == 7.5

    def test_getitem_missing_raises_key_error(self):
        s = PerfSample(tps=1.0, ttft=2.0, prompt_tps=3.0, total_ms=4.0)
        with pytest.raises(KeyError):
            s["nonexistent"]

    def test_get_existing(self):
        s = PerfSample(tps=5.0, ttft=1.0, prompt_tps=2.0, total_ms=3.0)
        assert s.get("tps") == 5.0

    def test_get_missing_returns_default(self):
        s = PerfSample(tps=1.0, ttft=2.0, prompt_tps=3.0, total_ms=4.0)
        assert s.get("nonexistent", 42) == 42

    def test_contains(self):
        s = PerfSample(tps=1.0, ttft=2.0, prompt_tps=3.0, total_ms=4.0)
        assert "tps" in s
        assert "nonexistent" not in s

    def test_setitem_raises_type_error(self):
        s = PerfSample(tps=1.0, ttft=2.0, prompt_tps=3.0, total_ms=4.0)
        with pytest.raises(TypeError, match="frozen"):
            s["tps"] = 99.0

    def test_update_raises_type_error(self):
        s = PerfSample(tps=1.0, ttft=2.0, prompt_tps=3.0, total_ms=4.0)
        with pytest.raises(TypeError, match="frozen"):
            s.update({"tps": 99.0})

    def test_frozen_attribute_assignment(self):
        s = PerfSample(tps=1.0, ttft=2.0, prompt_tps=3.0, total_ms=4.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.tps = 99.0


# ===================================================================
# PerfResult
# ===================================================================


@pytest.mark.unit
class TestPerfResult:
    def test_construction_defaults(self):
        r = PerfResult(tps=10.0, ttft=5.0, prompt_tps=20.0, total_ms=100.0)
        assert r.tps_std is None
        assert r.concurrent_users is None
        assert r.quality_factor is None

    def test_from_dict_missing_keys(self):
        r = PerfResult.from_dict({})
        assert r.tps == 0.0
        assert r.n_runs is None

    def test_from_dict_all_keys(self):
        data = {
            "tps": 50.0,
            "ttft": 10.0,
            "prompt_tps": 100.0,
            "total_ms": 200.0,
            "vram_used_mb": 4096.0,
            "tps_std": 1.5,
            "tps_cv": 0.03,
            "n_runs": 5,
            "large_tps": 40.0,
            "concurrent_total_tps": 80.0,
            "concurrent_avg_tps": 40.0,
            "concurrent_avg_ttft": 15.0,
            "concurrent_avg_wall_ms": 300.0,
            "concurrent_max_wall_ms": 500.0,
            "concurrent_success_rate": 0.95,
            "concurrent_users": 2,
            "quality_factor": 0.9,
            "load_time_ms": 1200.0,
        }
        r = PerfResult.from_dict(data)
        assert r.concurrent_users == 2
        assert r.quality_factor == pytest.approx(0.9)

    def test_to_dict_roundtrip(self):
        original = PerfResult(
            tps=30.0, ttft=5.0, prompt_tps=60.0, total_ms=80.0, tps_std=1.0, n_runs=3
        )
        d = original.to_dict()
        restored = PerfResult.from_dict(d)
        assert restored == original


# ===================================================================
# BenchResult
# ===================================================================


@pytest.mark.unit
class TestBenchResult:
    def test_from_dict_defaults(self):
        r = BenchResult.from_dict({})
        assert r.tps == 0.0
        assert r.total_ms == 0.0

    def test_to_dict_roundtrip(self):
        original = BenchResult(tps=50.0, prompt_tps=200.0, ttft=12.0, total_ms=100.0)
        assert BenchResult.from_dict(original.to_dict()) == original


# ===================================================================
# ConcurrentLoadResult
# ===================================================================


@pytest.mark.unit
class TestConcurrentLoadResult:
    def test_from_dict_defaults(self):
        r = ConcurrentLoadResult.from_dict({})
        assert r.concurrent_total_tps == 0.0
        assert r.concurrent_users == 0

    def test_to_dict_roundtrip(self):
        original = ConcurrentLoadResult(
            concurrent_total_tps=100.0,
            concurrent_avg_tps=50.0,
            concurrent_avg_ttft=10.0,
            concurrent_avg_wall_ms=200.0,
            concurrent_max_wall_ms=400.0,
            concurrent_success_rate=1.0,
            concurrent_users=2,
        )
        assert ConcurrentLoadResult.from_dict(original.to_dict()) == original


# ===================================================================
# TokenUncertaintyResult
# ===================================================================


@pytest.mark.unit
class TestTokenUncertaintyResult:
    def test_from_dict_defaults(self):
        r = TokenUncertaintyResult.from_dict({})
        assert r.uncertain_count == 0
        assert r.tail_avg == 0.0

    def test_to_dict_roundtrip(self):
        original = TokenUncertaintyResult(
            uncertain_count=3, tail_avg=0.5, total_tokens=100
        )
        assert TokenUncertaintyResult.from_dict(original.to_dict()) == original


# ===================================================================
# ParetoObjectives (NamedTuple)
# ===================================================================


@pytest.mark.unit
class TestParetoObjectives:
    def test_construction(self):
        p = ParetoObjectives(tps=50.0, neg_vram=-4096.0, quality_factor=0.9)
        assert p.tps == 50.0
        assert p.neg_vram == -4096.0
        assert p.quality_factor == pytest.approx(0.9)

    def test_tuple_unpacking(self):
        p = ParetoObjectives(tps=10.0, neg_vram=-2000.0, quality_factor=0.8)
        tps, neg_vram, qf = p
        assert tps == 10.0
        assert neg_vram == -2000.0
        assert qf == pytest.approx(0.8)

    def test_indexing(self):
        p = ParetoObjectives(tps=1.0, neg_vram=2.0, quality_factor=3.0)
        assert p[0] == 1.0
        assert p[1] == 2.0
        assert p[2] == 3.0


# ===================================================================
# KLResult — __iter__ for tuple unpacking
# ===================================================================


@pytest.mark.unit
class TestKLResult:
    def test_defaults(self):
        r = KLResult()
        assert r.distributions is None
        assert r.kl_divergence is None

    def test_iter_tuple_unpacking(self):
        dists = [{"a": 0.5, "b": 0.5}]
        r = KLResult(distributions=dists, kl_divergence=0.01)
        unpacked_dists, unpacked_kl = r
        assert unpacked_dists is dists
        assert unpacked_kl == pytest.approx(0.01)

    def test_from_dict_roundtrip(self):
        original = KLResult(distributions=[{"x": 1.0}], kl_divergence=0.05)
        assert KLResult.from_dict(original.to_dict()) == original


# ===================================================================
# PPLResult
# ===================================================================


@pytest.mark.unit
class TestPPLResult:
    def test_from_dict_defaults(self):
        r = PPLResult.from_dict({})
        assert r.perplexity == float("inf")
        assert r.baseline_ppl is None

    def test_to_dict_roundtrip(self):
        original = PPLResult(perplexity=5.5, baseline_ppl=4.0, quality_factor=0.95)
        assert PPLResult.from_dict(original.to_dict()) == original


# ===================================================================
# TrialSummary
# ===================================================================


@pytest.mark.unit
class TestTrialSummary:
    def test_from_dict_defaults(self):
        t = TrialSummary.from_dict({})
        assert t.number == 0
        assert t.tps is None
        assert t.metrics == {}
        assert t.params == {}

    def test_from_dict_full(self):
        data = {"number": 5, "tps": 42.0, "metrics": {"a": 1}, "params": {"b": 2}}
        t = TrialSummary.from_dict(data)
        assert t.number == 5
        assert t.tps == 42.0


# ===================================================================
# Nested from_dict — PhaseResult.all_trials
# ===================================================================


@pytest.mark.unit
class TestPhaseResult:
    def test_from_dict_defaults(self):
        r = PhaseResult.from_dict({})
        assert r.phase == ""
        assert r.all_trials == []

    def test_nested_all_trials_from_dict(self):
        data = {
            "phase": "test_phase",
            "all_trials": [
                {"number": 1, "tps": 10.0},
                {"number": 2, "tps": 20.0},
            ],
        }
        r = PhaseResult.from_dict(data)
        assert len(r.all_trials) == 2
        assert isinstance(r.all_trials[0], TrialSummary)
        assert r.all_trials[0].tps == 10.0

    def test_nested_all_trials_preserves_existing_objects(self):
        ts = TrialSummary(number=1, tps=5.0)
        data = {"phase": "p", "all_trials": [ts]}
        r = PhaseResult.from_dict(data)
        assert r.all_trials[0] is ts

    def test_to_dict_roundtrip(self):
        original = PhaseResult(
            phase="x",
            baseline_score=10.0,
            best_tps=20.0,
            all_trials=[TrialSummary(number=1, tps=15.0)],
        )
        d = original.to_dict()
        restored = PhaseResult.from_dict(d)
        assert restored.phase == original.phase
        assert len(restored.all_trials) == 1


# ===================================================================
# Nested from_dict — QualityResult.task_results
# ===================================================================


@pytest.mark.unit
class TestQualityResult:
    def test_nested_task_results(self):
        data = {
            "score": 85.0,
            "task_results": [
                {"correct": True, "answer": "A"},
                {"correct": False, "logprob": -1.5},
            ],
        }
        r = QualityResult.from_dict(data)
        assert len(r.task_results) == 2
        assert isinstance(r.task_results[0], QualityTaskResult)
        assert r.task_results[0].correct is True
        assert r.task_results[1].logprob == pytest.approx(-1.5)

    def test_preserves_existing_objects(self):
        tr = QualityTaskResult(correct=True)
        r = QualityResult.from_dict({"score": 50.0, "task_results": [tr]})
        assert r.task_results[0] is tr


# ===================================================================
# Nested from_dict — NIAHResult.results
# ===================================================================


@pytest.mark.unit
class TestNIAHResult:
    def test_nested_results(self):
        data = {
            "kv_type": "q4_0",
            "results": [
                {"context": 2048, "depth": 0.5, "passed": True},
                {"context": 4096, "depth": 0.8, "passed": False, "error": "timeout"},
            ],
            "pass_rate": 50.0,
        }
        r = NIAHResult.from_dict(data)
        assert len(r.results) == 2
        assert isinstance(r.results[0], NIAHTestResult)
        assert r.results[1].error == "timeout"

    def test_preserves_existing_objects(self):
        tr = NIAHTestResult(context=1024, depth=0.3, passed=True)
        r = NIAHResult.from_dict({"kv_type": "f16", "results": [tr]})
        assert r.results[0] is tr


# ===================================================================
# NIAHPhaseResult nested deserialization
# ===================================================================


@pytest.mark.unit
class TestNIAHPhaseResult:
    def test_nested_kv_results(self):
        data = {
            "kv_results": [
                {
                    "kv_type": "f16",
                    "results": [{"context": 512, "depth": 0.1, "passed": True}],
                },
            ],
        }
        r = NIAHPhaseResult.from_dict(data)
        assert len(r.kv_results) == 1
        assert isinstance(r.kv_results[0], NIAHResult)
        assert isinstance(r.kv_results[0].results[0], NIAHTestResult)


# ===================================================================
# ConcurrentUserResult
# ===================================================================


@pytest.mark.unit
class TestConcurrentUserResult:
    def test_from_dict_defaults(self):
        r = ConcurrentUserResult.from_dict({})
        assert r.user_id == 0
        assert r.success is False
        assert r.error is None

    def test_to_dict_roundtrip(self):
        original = ConcurrentUserResult(
            user_id=1, success=True, tps=50.0, wall_time=200.0
        )
        assert ConcurrentUserResult.from_dict(original.to_dict()) == original


# ===================================================================
# QualityTaskResult
# ===================================================================


@pytest.mark.unit
class TestQualityTaskResult:
    def test_from_dict_defaults(self):
        r = QualityTaskResult.from_dict({})
        assert r.correct is False
        assert r.logprob is None

    def test_to_dict_roundtrip(self):
        original = QualityTaskResult(
            correct=True, logprob=-0.5, answer="B", category="math"
        )
        assert QualityTaskResult.from_dict(original.to_dict()) == original
