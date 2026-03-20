"""Tests for phase objective functions across Core Engine, Speculation, MoE Sweep,
and GPU Offload.

Covers:
    Core Engine:
        1. _build_ab_flags returns correct flags for dense model
        2. _build_ab_flags skips flags in skip_flags set
        3. _run_single_ab_test picks higher scoring value
        4. _run_single_ab_test keeps default when neither beats baseline
        5. Layer 1 A/B sweep produces winners dict with all flags
        6. _layer2_objective returns score for valid trial
        7. _layer2_objective handles server boot failure (prunes trial)
        8. Core Engine locks n_cpu_moe when moe_sweep already ran

    Speculation:
        9. _suggest_spec_params returns valid param dict
        10. _build_spec_config merges base_config with spec params
        11. _clear_lookup_cache_if_needed handles missing file
        12. Speculation uses base_config (not naked_engine)

    MoE Sweep:
        13. phase_moe_sweep skips for dense models
        14. phase_moe_sweep sweeps n_cpu_moe range
        15. phase_moe_sweep picks highest scoring value

    GPU Offload:
        16. _find_oom_boundary binary search works correctly
        17. _score_sweep stops early when score drops below 50%
        18. GPU offload results show Winner not identical Baseline/Optimal
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import optuna
import pytest

from tps_pro.phases.core_engine import (
    _build_ab_flags,
    _layer1_ab_sweeps,
    _layer2_objective,
    _ObjectiveParams,
    _run_single_ab_test,
)
from tps_pro.phases.gpu_offload import (
    _find_oom_boundary,
    _score_sweep,
    phase_gpu_offload,
)
from tps_pro.phases.moe_sweep import phase_moe_sweep
from tps_pro.phases.speculation import (
    _build_spec_config,
    _clear_lookup_cache_if_needed,
    _suggest_spec_params,
    phase_speculation,
)
from tps_pro.phases.trial_helpers import BestScoreTracker
from tps_pro.result_types import PerfResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_perf(tps=30.0, prompt_tps=200.0, ttft=50.0, total_ms=1000.0):
    """Build a minimal PerfResult for testing."""
    return PerfResult(tps=tps, prompt_tps=prompt_tps, ttft=ttft, total_ms=total_ms)


def _score_fn(perf):
    """Simple scoring: just return tps as the score."""
    return perf.tps


# ---------------------------------------------------------------------------
# Core Engine Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBuildAbFlags:
    """Tests 1-2: _build_ab_flags."""

    def test_returns_correct_flags_for_dense_model(self):
        """Dense model (no skip flags) should include op_offload, repack, prio,
        prio_batch."""
        winners, ab_flags = _build_ab_flags(skip_flags=set())
        ab_names = [name for name, _, _ in ab_flags]

        # op_offload and repack should be in the A/B test list
        assert "op_offload" in ab_names
        assert "repack" in ab_names
        assert "prio" in ab_names
        assert "prio_batch" in ab_names

        # Hardcoded winners: no_mmap=True, mlock=False
        assert winners["no_mmap"] is True
        assert winners["mlock"] is False

        # op_offload and repack should NOT be in pre-determined winners
        assert "op_offload" not in winners
        assert "repack" not in winners

    def test_skips_flags_in_skip_flags_set(self):
        """Flags in skip_flags should become pre-determined winners, not A/B tested."""
        winners, ab_flags = _build_ab_flags(skip_flags={"op_offload", "repack"})
        ab_names = [name for name, _, _ in ab_flags]

        assert "op_offload" not in ab_names
        assert "repack" not in ab_names
        assert winners["op_offload"] is False
        assert winners["repack"] is True
        # prio should still be testable
        assert "prio" in ab_names


@pytest.mark.unit
class TestRunSingleAbTest:
    """Tests 3-4: _run_single_ab_test."""

    @patch("tps_pro.phases.core_engine.measure_perf_adaptive")
    @patch("tps_pro.phases.core_engine.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.core_engine.kill_server")
    def test_picks_higher_scoring_value(self, _kill, mock_boot, mock_measure, make_ctx):
        """Should return the candidate with the higher score."""
        ctx = make_ctx(skip_flags=set())
        mock_boot.return_value = (MagicMock(), "ok")

        perf_low = _make_perf(tps=10.0)
        perf_high = _make_perf(tps=50.0)
        mock_measure.side_effect = [(perf_low, False), (perf_high, False)]

        base_config = {"context": 4096, "n_gpu_layers": 99}
        flag_name, best_val, best_perf, best_score = _run_single_ab_test(
            ctx,
            base_config,
            winners={},
            flag_name="repack",
            candidates=[False, True],
            default_val=False,
            score_fn=_score_fn,
            baseline_score=0.0,
        )

        assert flag_name == "repack"
        assert best_val is True
        assert best_score == 50.0

    @patch("tps_pro.phases.core_engine.measure_perf_adaptive")
    @patch("tps_pro.phases.core_engine.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.core_engine.kill_server")
    def test_keeps_default_when_neither_beats_baseline(
        self, _kill, mock_boot, mock_measure, make_ctx
    ):
        """When no candidate beats baseline_score, default_val should be kept."""
        ctx = make_ctx(skip_flags=set())
        mock_boot.return_value = (MagicMock(), "ok")

        # Both candidates score below baseline of 100
        perf_a = _make_perf(tps=20.0)
        perf_b = _make_perf(tps=30.0)
        mock_measure.side_effect = [(perf_a, False), (perf_b, False)]

        flag_name, best_val, best_perf, best_score = _run_single_ab_test(
            ctx,
            base_config={"context": 4096},
            winners={},
            flag_name="prio",
            candidates=[0, 2],
            default_val=0,
            score_fn=_score_fn,
            baseline_score=100.0,
        )

        assert flag_name == "prio"
        assert best_val == 0  # kept default because neither beats baseline


@pytest.mark.unit
class TestLayer1AbSweep:
    """Test 5: Layer 1 A/B sweep produces winners dict with all flags."""

    @patch("tps_pro.phases.core_engine.measure_perf_adaptive")
    @patch("tps_pro.phases.core_engine.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.core_engine.kill_server")
    def test_produces_winners_dict_with_all_flags(
        self, _kill, mock_boot, mock_measure, make_ctx
    ):
        """Winners dict should contain entries for every flag (A/B tested +
        hardcoded)."""
        ctx = make_ctx(skip_flags=set())
        mock_boot.return_value = (MagicMock(), "ok")

        perf = _make_perf(tps=40.0)
        mock_measure.return_value = (perf, False)

        baseline = _make_perf(tps=30.0)
        winners = _layer1_ab_sweeps(
            ctx,
            base_config={"context": 4096, "n_gpu_layers": 99},
            score_fn=_score_fn,
            baseline=baseline,
            baseline_score=30.0,
        )

        # Should have all standard Layer 1 flags
        expected_keys = {
            "no_mmap",
            "mlock",
            "op_offload",
            "repack",
            "prio",
            "prio_batch",
        }
        assert expected_keys.issubset(set(winners.keys()))


@pytest.mark.unit
class TestLayer2Objective:
    """Tests 6-8: _layer2_objective."""

    def _make_obj_params(self, ctx, layer2_base=None, skip_flags=None):
        """Build _ObjectiveParams for testing."""
        return _ObjectiveParams(
            ctx=ctx,
            layer2_base=layer2_base or {"context": 4096, "n_gpu_layers": 99},
            score_fn=_score_fn,
            is_pareto=False,
            total_trials=10,
            thread_opts=[2, 4, 8],
            batch_opts=[256, 512],
            ubatch_opts=[128, 256],
            skip_flags=skip_flags or set(),
        )

    @patch("tps_pro.phases.core_engine.update_param_cache")
    @patch("tps_pro.phases.core_engine.finalize_trial")
    @patch("tps_pro.phases.core_engine.record_trial_attrs")
    @patch("tps_pro.phases.core_engine.measure_perf_adaptive")
    @patch("tps_pro.phases.core_engine.measure_perf_once", return_value=None)
    @patch("tps_pro.phases.core_engine.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.core_engine.kill_server")
    @patch(
        "tps_pro.phases.core_engine.check_and_mark_duplicate_trial", return_value=None
    )
    @patch("tps_pro.phases.core_engine.thermal_gate")
    def test_returns_score_for_valid_trial(
        self,
        _thermal,
        _dup,
        _kill,
        mock_boot,
        _gate,
        mock_measure,
        mock_record,
        mock_finalize,
        _cache,
        make_ctx,
    ):
        """Valid trial should boot server, measure, and return a score."""
        ctx = make_ctx(is_moe=False, numa_nodes=1, skip_flags=set())
        proc = MagicMock()
        proc.load_time_ms = 100.0
        mock_boot.return_value = (proc, "ok")

        perf = _make_perf(tps=45.0)
        mock_measure.return_value = (perf, True)
        mock_finalize.return_value = (45.0, 45.0)

        params = self._make_obj_params(ctx)
        best = BestScoreTracker(0.0)

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        result = _layer2_objective(trial, params, best)

        assert result == 45.0
        mock_boot.assert_called_once()
        mock_measure.assert_called_once()

    @patch("tps_pro.phases.core_engine.server_start_failed")
    @patch("tps_pro.phases.core_engine.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.core_engine.kill_server")
    @patch(
        "tps_pro.phases.core_engine.check_and_mark_duplicate_trial", return_value=None
    )
    @patch("tps_pro.phases.core_engine.thermal_gate")
    def test_handles_server_boot_failure(
        self, _thermal, _dup, _kill, mock_boot, mock_failed, make_ctx
    ):
        """OOM during boot should raise TrialPruned."""
        ctx = make_ctx(is_moe=False, numa_nodes=1, skip_flags=set())
        mock_boot.return_value = (None, "oom")

        params = self._make_obj_params(ctx)
        best = BestScoreTracker(0.0)

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        with pytest.raises(optuna.exceptions.TrialPruned):
            _layer2_objective(trial, params, best)

    @patch("tps_pro.phases.core_engine.update_param_cache")
    @patch("tps_pro.phases.core_engine.finalize_trial")
    @patch("tps_pro.phases.core_engine.record_trial_attrs")
    @patch("tps_pro.phases.core_engine.measure_perf_adaptive")
    @patch("tps_pro.phases.core_engine.measure_perf_once", return_value=None)
    @patch("tps_pro.phases.core_engine.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.core_engine.kill_server")
    @patch(
        "tps_pro.phases.core_engine.check_and_mark_duplicate_trial", return_value=None
    )
    @patch("tps_pro.phases.core_engine.thermal_gate")
    def test_locks_n_cpu_moe_when_moe_sweep_already_ran(
        self,
        _thermal,
        _dup,
        _kill,
        mock_boot,
        _gate,
        mock_measure,
        mock_record,
        mock_finalize,
        _cache,
        make_ctx,
    ):
        """When n_cpu_moe is in layer2_base (from moe_sweep), it should be locked."""
        ctx = make_ctx(
            is_moe=True,
            numa_nodes=1,
            skip_flags=set(),
            default_experts=2,
            moe_sweep_max=24,
            moe_sweep_center=12,
        )
        proc = MagicMock()
        proc.load_time_ms = 50.0
        mock_boot.return_value = (proc, "ok")

        perf = _make_perf(tps=35.0)
        mock_measure.return_value = (perf, True)
        mock_finalize.return_value = (35.0, 35.0)

        # n_cpu_moe in layer2_base means moe_sweep already ran and locked the value
        layer2_base = {"context": 4096, "n_gpu_layers": 99, "n_cpu_moe": 16}
        params = self._make_obj_params(ctx, layer2_base=layer2_base)
        best = BestScoreTracker(0.0)

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        _layer2_objective(trial, params, best)

        # The config passed to boot should have n_cpu_moe=16 (locked), not suggested
        boot_call_config = mock_boot.call_args[0][1]
        assert boot_call_config["n_cpu_moe"] == 16


# ---------------------------------------------------------------------------
# Speculation Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSuggestSpecParams:
    """Test 9: _suggest_spec_params returns valid param dict."""

    def test_returns_valid_param_dict(self):
        """Should return dict with all expected keys."""
        study = optuna.create_study()
        trial = study.ask()

        params = _suggest_spec_params(trial, draft_model=None)

        expected_keys = {
            "spec_type",
            "spec_ngram_n",
            "spec_ngram_m",
            "spec_ngram_min_hits",
            "draft_max",
            "draft_min",
            "draft_p_min",
            "use_lookup_cache",
        }
        assert expected_keys == set(params.keys())

        # spec_type should NOT include "draft" when no draft model
        assert params["spec_type"] in {
            "ngram-simple",
            "ngram-cache",
            "ngram-map-k",
            "ngram-map-k4v",
            "ngram-mod",
        }

    def test_includes_draft_option_when_draft_model_set(self):
        """When draft_model is provided, 'draft' should be a possible spec_type."""
        # Run enough trials to check draft is in the options
        study = optuna.create_study()
        spec_types_seen = set()
        for _ in range(50):
            trial = study.ask()
            try:
                params = _suggest_spec_params(trial, draft_model="/path/to/draft.gguf")
                spec_types_seen.add(params["spec_type"])
            except optuna.exceptions.TrialPruned:
                pass
            study.tell(trial, 1.0)

        assert "draft" in spec_types_seen


@pytest.mark.unit
class TestBuildSpecConfig:
    """Test 10: _build_spec_config merges base_config with spec params."""

    def test_merges_base_config_with_spec_params(self):
        """Resulting config should contain base keys plus spec params."""
        base = {"context": 4096, "threads": 8, "n_gpu_layers": 99}
        params = {
            "spec_type": "ngram-cache",
            "spec_ngram_n": 14,
            "spec_ngram_m": 64,
            "spec_ngram_min_hits": 4,
            "draft_max": 47,
            "draft_min": 4,
            "draft_p_min": 0.52,
            "use_lookup_cache": False,
        }

        config, params_short = _build_spec_config(
            base, params, lookup_cache_file="", draft_model=None
        )

        # Base keys preserved
        assert config["context"] == 4096
        assert config["threads"] == 8
        # Spec params merged
        assert config["spec_type"] == "ngram-cache"
        assert config["spec_ngram_n"] == 14
        assert config["draft_max"] == 47
        # lookup_cache_dynamic should NOT be set (use_lookup_cache=False)
        assert "lookup_cache_dynamic" not in config
        # params_short should be a non-empty string summary
        assert "ngram-cache" in params_short

    def test_lookup_cache_added_when_enabled(self):
        """When use_lookup_cache=True with a file, lookup_cache_dynamic should be
        set."""
        base = {"context": 4096}
        params = {
            "spec_type": "ngram-simple",
            "spec_ngram_n": 5,
            "spec_ngram_m": 10,
            "spec_ngram_min_hits": 1,
            "draft_max": 10,
            "draft_min": 2,
            "draft_p_min": 0.5,
            "use_lookup_cache": True,
        }

        config, _ = _build_spec_config(
            base, params, lookup_cache_file="/tmp/cache.bin", draft_model=None
        )
        assert config["lookup_cache_dynamic"] == "/tmp/cache.bin"


@pytest.mark.unit
class TestClearLookupCache:
    """Test 11: _clear_lookup_cache_if_needed handles missing file."""

    def test_handles_missing_file(self, tmp_path):
        """Should not raise when the file does not exist."""
        missing = str(tmp_path / "nonexistent_cache.bin")
        # Should not raise
        _clear_lookup_cache_if_needed(use_lookup_cache=True, lookup_cache_file=missing)

    def test_removes_existing_file(self, tmp_path):
        """Should remove the file when it exists."""
        cache_file = tmp_path / "cache.bin"
        cache_file.write_bytes(b"data")
        assert cache_file.exists()

        _clear_lookup_cache_if_needed(
            use_lookup_cache=True, lookup_cache_file=str(cache_file)
        )
        assert not cache_file.exists()

    def test_noop_when_disabled(self, tmp_path):
        """Should not remove the file when use_lookup_cache is False."""
        cache_file = tmp_path / "cache.bin"
        cache_file.write_bytes(b"data")

        _clear_lookup_cache_if_needed(
            use_lookup_cache=False, lookup_cache_file=str(cache_file)
        )
        assert cache_file.exists()


@pytest.mark.unit
class TestSpeculationUsesBaseConfig:
    """Test 12: Speculation uses base_config (not naked_engine)."""

    @patch("tps_pro.phases.speculation.run_study_with_callbacks")
    @patch("tps_pro.phases.speculation.print_phase_summary")
    @patch("tps_pro.phases.speculation.clear_param_cache")
    @patch("tps_pro.phases.speculation.setup_baseline_server")
    @patch("tps_pro.phases.speculation.setup_study")
    def test_uses_base_config_not_naked_engine(
        self, mock_study, mock_baseline, _cache, mock_summary, _run, make_ctx
    ):
        """When base_config is provided, speculation should use it, not
        ctx.naked_engine."""
        ctx = make_ctx(
            config={"pareto": False},
            naked_engine={"context": 2048, "threads": 4},
            lookup_cache_file="",
        )

        mock_optuna_study = MagicMock()
        mock_study.return_value = (mock_optuna_study, 5, 0)

        baseline_perf = _make_perf(tps=20.0)
        mock_baseline.return_value = (baseline_perf, 20.0)
        mock_summary.return_value = ({"spec_type": "ngram-cache"}, None)

        custom_base = {"context": 8192, "threads": 16, "n_gpu_layers": 99}
        phase_speculation(ctx, n_trials=5, base_config=custom_base)

        # The baseline server should be called with our custom base config
        baseline_call_config = mock_baseline.call_args[0][1]
        assert baseline_call_config["context"] == 8192
        assert baseline_call_config["threads"] == 16


# ---------------------------------------------------------------------------
# MoE Sweep Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhaseMoeSweep:
    """Tests 13-15: phase_moe_sweep."""

    def test_skips_for_dense_models(self, make_ctx):
        """Dense (non-MoE) models should return None immediately."""
        ctx = make_ctx(is_moe=False)
        result = phase_moe_sweep(ctx)
        assert result is None

    @patch("tps_pro.phases.moe_sweep.save_phase_results")
    @patch("tps_pro.phases.moe_sweep.compute_score")
    @patch("tps_pro.phases.moe_sweep.measure_perf_adaptive")
    @patch("tps_pro.phases.moe_sweep.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.moe_sweep.kill_server")
    @patch("tps_pro.phases.moe_sweep.load_phase_results", return_value=None)
    def test_sweeps_n_cpu_moe_range(
        self, _load, _kill, mock_boot, mock_measure, mock_score, _save, make_ctx
    ):
        """Should test multiple n_cpu_moe values from 8 to moe_sweep_max."""
        ctx = make_ctx(
            is_moe=True,
            naked_engine={"context": 4096, "n_gpu_layers": 99},
            fail_fast=False,
            moe_sweep_max=14,
        )
        mock_boot.return_value = (MagicMock(), "ok")

        perf = _make_perf(tps=30.0)
        mock_measure.return_value = (perf, False)
        mock_score.return_value = 30.0

        result = phase_moe_sweep(ctx, force=True)

        assert result is not None
        assert "n_cpu_moe" in result["best_params"]
        # Should have called boot for baseline + each sweep value (8, 10, 12, 14)
        assert mock_boot.call_count >= 5  # 1 baseline + 4 sweep values

    @patch("tps_pro.phases.moe_sweep.save_phase_results")
    @patch("tps_pro.phases.moe_sweep.compute_score")
    @patch("tps_pro.phases.moe_sweep.measure_perf_adaptive")
    @patch("tps_pro.phases.moe_sweep.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.moe_sweep.kill_server")
    @patch("tps_pro.phases.moe_sweep.load_phase_results", return_value=None)
    def test_picks_highest_scoring_value(
        self, _load, _kill, mock_boot, mock_measure, mock_score, _save, make_ctx
    ):
        """Should return the n_cpu_moe value with the highest score."""
        ctx = make_ctx(
            is_moe=True,
            naked_engine={"context": 4096, "n_gpu_layers": 99},
            fail_fast=False,
            moe_sweep_max=12,
        )
        mock_boot.return_value = (MagicMock(), "ok")

        perf = _make_perf(tps=30.0)
        mock_measure.return_value = (perf, False)

        # Scores: baseline=10, moe=8->20, moe=10->50(best), moe=12->30
        # Plus neighbor recheck: moe=9->25, moe=11->35
        scores = iter([10.0, 20.0, 50.0, 30.0, 25.0, 35.0])
        mock_score.side_effect = lambda _p: next(scores)

        result = phase_moe_sweep(ctx, force=True)

        assert result is not None
        assert result["best_params"]["n_cpu_moe"] == 10


# ---------------------------------------------------------------------------
# GPU Offload Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindOomBoundary:
    """Test 16: _find_oom_boundary binary search works correctly."""

    @patch("tps_pro.phases.gpu_offload._test_ngl")
    def test_binary_search_finds_boundary(self, mock_test, make_ctx):
        """Should find the highest ngl that fits via binary search."""
        ctx = make_ctx()

        # ngl 0-7 work, 8+ fail. Max is 10.
        def side_effect(ctx, ngl, use_bench):
            return ngl <= 7

        mock_test.side_effect = side_effect

        boundary = _find_oom_boundary(ctx, max_ngl=10, use_bench=False)

        assert boundary == 7

    @patch("tps_pro.phases.gpu_offload._test_ngl")
    def test_returns_max_when_all_fit(self, mock_test, make_ctx):
        """When max_ngl fits, should return max_ngl immediately."""
        ctx = make_ctx()
        mock_test.return_value = True

        boundary = _find_oom_boundary(ctx, max_ngl=99, use_bench=False)

        assert boundary == 99
        # Should only have tested max_ngl once
        mock_test.assert_called_once_with(ctx, 99, False)

    @patch("tps_pro.phases.gpu_offload._test_ngl")
    def test_returns_none_when_zero_fails(self, mock_test, make_ctx):
        """When even ngl=0 fails, should return None."""
        ctx = make_ctx()
        mock_test.return_value = False

        boundary = _find_oom_boundary(ctx, max_ngl=10, use_bench=False)

        assert boundary is None


@pytest.mark.unit
class TestScoreSweep:
    """Test 17: _score_sweep stops early when score drops below 50%."""

    @patch("tps_pro.phases.gpu_offload.check_thermal_throttle", return_value=(False, 0))
    @patch("tps_pro.phases.gpu_offload._score_ngl")
    @patch("tps_pro.phases.gpu_offload._full_range_checkpoints")
    def test_stops_early_when_score_drops(
        self, mock_checkpoints, mock_score_ngl, _thermal, make_ctx
    ):
        """Full-range sweep should stop early when score < 50% of best."""
        ctx = make_ctx()

        # Simulate: oom_boundary == max_ngl (full range sweep path)
        mock_checkpoints.return_value = [99, 80, 60, 40, 20, 0]

        # Scores: 99->100, 80->90, 60->80, 40->10 (below 50% of 100)
        perfs_and_scores = [
            (_make_perf(tps=100.0), 100.0),
            (_make_perf(tps=90.0), 90.0),
            (_make_perf(tps=80.0), 80.0),
            (_make_perf(tps=10.0), 10.0),  # < 50% of 100 -> should stop
            (_make_perf(tps=5.0), 5.0),  # should NOT be reached
            (_make_perf(tps=1.0), 1.0),  # should NOT be reached
        ]
        mock_score_ngl.side_effect = perfs_and_scores

        results, best_score, best_ngl = _score_sweep(
            ctx, oom_boundary=99, max_ngl=99, use_bench=False
        )

        # Should have stopped after 4 points (3 good + 1 that triggered early stop)
        assert len(results) == 4
        assert best_score == 100.0
        assert best_ngl == 99


@pytest.mark.unit
class TestGpuOffloadResults:
    """Test 18: GPU offload results show Winner not identical Baseline/Optimal."""

    @patch("tps_pro.phases.gpu_offload.save_phase_results")
    @patch("tps_pro.phases.gpu_offload._score_sweep")
    @patch("tps_pro.phases.gpu_offload._find_oom_boundary")
    @patch("tps_pro.phases.gpu_offload.load_phase_results", return_value=None)
    @patch("tps_pro.phases.gpu_offload.update_naked_engine")
    def test_winner_differs_from_max(
        self, _update, mock_load, mock_oom, mock_sweep, _save, make_ctx
    ):
        """When the best ngl is not max_ngl, result should reflect the actual winner."""
        ctx = make_ctx(max_gpu_layers=99, is_moe=False, bench_path=None)
        mock_oom.return_value = 80

        # Sweep finds that ngl=75 is best
        results = [
            {"ngl": 80, "perf": _make_perf(tps=40.0), "score": 40.0, "promoted": True},
            {"ngl": 79, "perf": _make_perf(tps=50.0), "score": 50.0, "promoted": True},
            {"ngl": 78, "perf": _make_perf(tps=60.0), "score": 60.0, "promoted": True},
            {"ngl": 77, "perf": _make_perf(tps=55.0), "score": 55.0, "promoted": True},
            {"ngl": 76, "perf": _make_perf(tps=45.0), "score": 45.0, "promoted": True},
            {"ngl": 75, "perf": _make_perf(tps=70.0), "score": 70.0, "promoted": True},
        ]
        mock_sweep.return_value = (results, 70.0, 75)

        result = phase_gpu_offload(ctx)

        assert result is not None
        assert result["best_params"]["n_gpu_layers"] == 75
        assert ctx.default_gpu_layers == 75
