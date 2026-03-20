"""Integration tests verifying component interactions across the
llama_optimizer pipeline.

Mocks llama-server HTTP endpoints but lets everything else run for real.
Tests cross-module boundaries: server lifecycle -> measurement -> scoring ->
study management.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import optuna
import pytest
import requests

from tps_pro.constants import SCORE_VERSION
from tps_pro.engine.server import warmup_server
from tps_pro.evals.quality_gate import measure_quality_gate
from tps_pro.measurement import (
    compute_pareto_objectives,
    compute_score,
    measure_perf_once,
)
from tps_pro.phases.trial_helpers import (
    recover_best_score,
    run_server_trial,
)
from tps_pro.result_types import (
    PerfResult,
    PerfSample,
    ServerProcess,
)
from tps_pro.search import (
    load_phase_results,
    save_phase_results,
    setup_study,
)

# ===================================================================
# Helpers
# ===================================================================


def _make_ctx(tmp_path, **overrides):
    """Build a mock ctx with filesystem-backed results_dir for integration tests."""
    from _ctx_factory import make_ctx_from_defaults

    results_dir = tmp_path / "results"
    results_dir.mkdir(exist_ok=True)

    return make_ctx_from_defaults(
        server_path=Path("/usr/bin/llama-server"),
        model_path=Path("/models/test.gguf"),
        port=18090,
        _port_alt=18091,
        server_url="http://127.0.0.1:18090",
        model_size_class="small",
        http=MagicMock(spec=requests.Session),
        default_experts=8,
        max_gpu_layers=33,
        default_gpu_layers=33,
        naked_engine={"context": 4096, "mlock": True, "n_gpu_layers": 33},
        results_dir=results_dir,
        optuna_db="sqlite:///" + str(results_dir / "optuna.db").replace("\\", "/"),
        fresh_run=True,
        lookup_cache_file=str(results_dir / "lookup-cache.bin"),
        **overrides,
    )


def _make_server_proc(poll_return=None, stderr_lines=None):
    """Build a ServerProcess with a mocked subprocess.Popen."""
    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = poll_return
    mock_proc.pid = 12345
    mock_proc.stderr = MagicMock()
    sp = ServerProcess(proc=mock_proc)
    if stderr_lines:
        sp.stderr_lines = list(stderr_lines)
    return sp


def _mock_http_completions_response(
    tps=25.0, prompt_tps=300.0, ttft=50.0, predicted_ms=2000.0
):
    """Create a mock HTTP response mimicking /v1/chat/completions."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "timings": {
            "predicted_per_second": tps,
            "prompt_per_second": prompt_tps,
            "prompt_ms": ttft,
            "predicted_ms": predicted_ms,
        }
    }
    return resp


def _mock_health_ok():
    """Create a mock HTTP response for /health returning ok."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"status": "ok"}
    return resp


# ===================================================================
# 1. Server -> Measurement flow
# ===================================================================


@pytest.mark.integration
class TestServerMeasurementFlow:
    """Mock HTTP responses. Test that start_server + warmup_server +
    measure_perf_once works end-to-end with a mock ctx."""

    @patch("tps_pro.engine.server._assign_job_object")
    @patch("tps_pro.engine.server.subprocess.Popen")
    @patch("tps_pro.hardware.init_vram_info")
    @patch("tps_pro.engine.server.time")
    def test_start_warmup_measure_flow(
        self, mock_time, mock_vram, mock_popen, mock_job, tmp_path
    ):
        ctx = _make_ctx(tmp_path)
        mock_time.time.side_effect = [0.0, 0.5]

        # Mock subprocess
        mock_proc = MagicMock()
        mock_proc.pid = 42
        mock_proc.stderr = iter([])
        mock_popen.return_value = mock_proc

        # Warmup: two successful POST requests
        warmup_resp = MagicMock()
        warmup_resp.status_code = 200
        ctx.http.post.return_value = warmup_resp

        result = warmup_server(ctx)
        assert result is True

        # Now measure: measure_perf_once uses ctx.http.post
        measure_resp = _mock_http_completions_response(
            tps=30.0, prompt_tps=400.0, ttft=45.0
        )
        ctx.http.post.return_value = measure_resp

        with patch("tps_pro.hardware.get_vram_used_mb", return_value=4000.0):
            sample = measure_perf_once(ctx, n_predict=50)

        assert sample is not None
        assert sample.tps == 30.0
        assert sample.prompt_tps == 400.0
        assert sample.ttft == 45.0

        # Verify score is computed correctly from this sample
        score = compute_score(sample)
        assert score > 0


# ===================================================================
# 2. Trial lifecycle
# ===================================================================


@pytest.mark.integration
class TestTrialLifecycle:
    """Test run_server_trial() end-to-end: boot -> gate -> adaptive measurement ->
    score."""

    @patch(
        "tps_pro.phases.trial_helpers.get_vram_used_mb",
        return_value=5000.0,
    )
    @patch("tps_pro.phases.trial_helpers.measure_perf_adaptive")
    @patch("tps_pro.phases.trial_helpers.measure_perf_once")
    @patch("tps_pro.phases.trial_helpers.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.trial_helpers.kill_server")
    def test_run_server_trial_success(
        self, mock_kill, mock_boot, mock_gate, mock_adaptive, mock_vram, tmp_path
    ):
        ctx = _make_ctx(tmp_path)

        # Boot succeeds
        sp = _make_server_proc()
        mock_boot.return_value = (sp, "ok")

        # Gate measurement returns a quick sample
        gate_sample = PerfSample(tps=20.0, ttft=50.0, prompt_tps=300.0, total_ms=2050.0)
        mock_gate.return_value = gate_sample

        # Adaptive measurement returns promoted result
        perf = PerfResult(tps=25.0, ttft=45.0, prompt_tps=350.0, total_ms=2045.0)
        mock_adaptive.return_value = (perf, True)

        # Create a real trial via optuna
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        config = {"threads": 8, "batch_size": 512}
        result_perf, score = run_server_trial(
            ctx, trial, config, "t=8 b=512", best_score=0.0
        )

        assert result_perf is not None
        assert result_perf.tps == 25.0
        assert score > 0
        # Verify gate was called
        mock_gate.assert_called_once()
        # Verify adaptive was called
        mock_adaptive.assert_called_once()

    @patch("tps_pro.phases.trial_helpers.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.trial_helpers.kill_server")
    def test_run_server_trial_oom_prunes(self, mock_kill, mock_boot, tmp_path):
        ctx = _make_ctx(tmp_path)
        sp = _make_server_proc()
        mock_boot.return_value = (sp, "oom")

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        with pytest.raises(optuna.exceptions.TrialPruned):
            run_server_trial(ctx, trial, {}, "oom-config", best_score=10.0)


# ===================================================================
# 3. Phase result round-trip
# ===================================================================


@pytest.mark.integration
class TestPhaseResultRoundTrip:
    """Test save_phase_results() -> load_phase_results() -> use in next phase."""

    def test_save_load_preserves_data(self, tmp_path):
        ctx = _make_ctx(tmp_path)

        results = {
            "phase": "core_engine",
            "best_params": {"threads": 12, "batch_size": 1024, "flash_attn": "on"},
            "best_tps": 42.5,
            "baseline_score": 35.0,
            "beat_baseline": True,
            "duration_minutes": 5.2,
        }
        save_phase_results(ctx, "core_engine", results)

        loaded = load_phase_results(ctx, "core_engine")
        assert loaded is not None
        assert loaded["best_params"]["threads"] == 12
        assert loaded["best_tps"] == 42.5
        assert loaded["beat_baseline"] is True
        assert loaded["score_version"] == SCORE_VERSION

    def test_stale_version_returns_none(self, tmp_path):
        ctx = _make_ctx(tmp_path)

        # Save with a fake old version
        results = {"phase": "gpu", "best_ngl": 33, "score_version": "v0_stale"}
        path = ctx.results_dir / "gpu_results.json"
        with open(path, "w") as f:
            json.dump(results, f)

        loaded = load_phase_results(ctx, "gpu")
        assert loaded is None  # stale version rejected

    def test_results_chain_across_phases(self, tmp_path):
        """Simulate GPU offload results feeding into core engine phase."""
        ctx = _make_ctx(tmp_path)

        # Save GPU phase results
        gpu_results = {"best_ngl": 28, "best_tps": 30.0}
        save_phase_results(ctx, "gpu", gpu_results)

        # Load and apply to ctx (as pipeline would)
        gpu_data = load_phase_results(ctx, "gpu")
        assert gpu_data is not None
        ctx.default_gpu_layers = gpu_data["best_ngl"]
        ctx.naked_engine["n_gpu_layers"] = gpu_data["best_ngl"]

        # Save core engine results that build on GPU results
        core_results = {
            "best_params": {"threads": 16, "batch_size": 512},
            "best_tps": 45.0,
            "baseline_score": 30.0,
            "beat_baseline": True,
        }
        save_phase_results(ctx, "core_engine", core_results)

        # Verify both are loadable
        assert load_phase_results(ctx, "gpu") is not None
        assert load_phase_results(ctx, "core_engine") is not None
        assert ctx.naked_engine["n_gpu_layers"] == 28


# ===================================================================
# 4. Baseline -> Trial comparison
# ===================================================================


@pytest.mark.integration
class TestBaselineTrialComparison:
    """Test setup_baseline_server() -> run_server_trial() with a trial that beats
    baseline."""

    @patch(
        "tps_pro.phases.trial_helpers.get_vram_used_mb",
        return_value=5000.0,
    )
    @patch("tps_pro.phases.trial_helpers.measure_perf_adaptive")
    @patch("tps_pro.phases.trial_helpers.measure_perf_once")
    @patch("tps_pro.phases.trial_helpers.boot_server_with_jinja_recovery")
    @patch("tps_pro.phases.trial_helpers.kill_server")
    def test_trial_beats_baseline(
        self, mock_kill, mock_boot, mock_gate, mock_adaptive, mock_vram, tmp_path
    ):
        ctx = _make_ctx(tmp_path)

        # Baseline measurement
        baseline_perf = PerfResult(
            tps=20.0, ttft=60.0, prompt_tps=250.0, total_ms=2060.0
        )
        baseline_score = compute_score(baseline_perf)
        assert baseline_score > 0

        # Now run a trial that should beat baseline
        sp = _make_server_proc()
        mock_boot.return_value = (sp, "ok")

        # Gate returns quick sample
        gate_sample = PerfSample(tps=28.0, ttft=40.0, prompt_tps=400.0, total_ms=2040.0)
        mock_gate.return_value = gate_sample

        # Adaptive measurement returns better perf
        trial_perf = PerfResult(tps=30.0, ttft=35.0, prompt_tps=450.0, total_ms=2035.0)
        mock_adaptive.return_value = (trial_perf, True)

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        result_perf, trial_score = run_server_trial(
            ctx, trial, {"threads": 16}, "t=16", best_score=baseline_score
        )

        assert result_perf is not None
        assert trial_score > baseline_score

    def test_score_ordering_is_consistent(self):
        """Better perf metrics must always produce higher scores."""
        slow = PerfResult(tps=10.0, ttft=100.0, prompt_tps=100.0, total_ms=2100.0)
        medium = PerfResult(tps=20.0, ttft=50.0, prompt_tps=300.0, total_ms=2050.0)
        fast = PerfResult(tps=35.0, ttft=30.0, prompt_tps=500.0, total_ms=2030.0)

        slow_score = compute_score(slow)
        medium_score = compute_score(medium)
        fast_score = compute_score(fast)

        assert slow_score < medium_score < fast_score


# ===================================================================
# 5. Quality gate flow
# ===================================================================


@pytest.mark.integration
class TestQualityGateFlow:
    """Test measure_quality_gate() baseline + trial measurement -> decision."""

    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_baseline_then_good_trial(self, mock_uncertainty, tmp_path):
        from tps_pro.result_types import TokenUncertaintyResult

        ctx = _make_ctx(tmp_path)

        # Baseline measurement — use large enough counts that small changes
        # stay under the 1.5% ceiling (QUALITY_GATE_CEILING=0.015)
        baseline_metrics = TokenUncertaintyResult(
            uncertain_count=1000, tail_avg=-1.5, total_tokens=50000
        )
        mock_uncertainty.return_value = baseline_metrics
        baseline_qf = measure_quality_gate(ctx, is_baseline=True)
        assert baseline_qf == 1.0
        assert ctx.quality_baseline is not None

        # Good trial: uncertain_count increases by < 1.5% (1000 -> 1010 = +1%)
        trial_metrics = TokenUncertaintyResult(
            uncertain_count=1010, tail_avg=-1.51, total_tokens=50000
        )
        mock_uncertainty.return_value = trial_metrics
        trial_qf = measure_quality_gate(ctx, is_baseline=False)
        assert trial_qf >= 0.85  # small degradation, still high quality

    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_baseline_then_bad_trial(self, mock_uncertainty, tmp_path):
        from tps_pro.result_types import TokenUncertaintyResult

        ctx = _make_ctx(tmp_path)

        # Baseline
        baseline_metrics = TokenUncertaintyResult(
            uncertain_count=100, tail_avg=-1.5, total_tokens=5000
        )
        mock_uncertainty.return_value = baseline_metrics
        measure_quality_gate(ctx, is_baseline=True)

        # Bad trial: uncertain_count increases by >3% (100 -> 110 = +10%)
        # which is well past the cliff (QUALITY_GATE_CLIFF=0.03)
        bad_metrics = TokenUncertaintyResult(
            uncertain_count=110, tail_avg=-2.5, total_tokens=5000
        )
        mock_uncertainty.return_value = bad_metrics
        bad_qf = measure_quality_gate(ctx, is_baseline=False)
        assert bad_qf < 0.85  # noticeable quality penalty

    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_measurement_failure_returns_penalty(self, mock_uncertainty, tmp_path):
        ctx = _make_ctx(tmp_path)

        # Baseline succeeds
        from tps_pro.result_types import TokenUncertaintyResult

        mock_uncertainty.return_value = TokenUncertaintyResult(
            uncertain_count=10, tail_avg=-1.5, total_tokens=500
        )
        measure_quality_gate(ctx, is_baseline=True)

        # Trial measurement fails
        mock_uncertainty.return_value = None
        qf = measure_quality_gate(ctx, is_baseline=False)
        assert qf < 1.0  # penalty applied


# ===================================================================
# 6. Multi-phase pipeline segment (result propagation)
# ===================================================================


@pytest.mark.integration
class TestMultiPhasePipelineSegment:
    """Test that GPU offload results feed correctly into core engine phase
    via save/load/apply cycle."""

    def test_gpu_results_propagate_to_core_engine(self, tmp_path):
        ctx = _make_ctx(tmp_path)

        # Phase 1: GPU offload produces best_ngl
        gpu_results = {
            "best_ngl": 25,
            "best_tps": 28.0,
            "all_configs": [
                {"ngl": 20, "tps": 22.0},
                {"ngl": 25, "tps": 28.0},
                {"ngl": 30, "tps": 26.0},
            ],
        }
        save_phase_results(ctx, "gpu", gpu_results)

        # Simulate pipeline: load GPU results and update ctx
        gpu_data = load_phase_results(ctx, "gpu")
        assert gpu_data is not None
        ctx.default_gpu_layers = gpu_data["best_ngl"]
        ctx.naked_engine["n_gpu_layers"] = gpu_data["best_ngl"]

        # Phase 2: Core engine uses the updated ctx
        base_config = {**ctx.naked_engine}
        assert base_config["n_gpu_layers"] == 25

        # Save core engine results
        core_results = {
            "best_params": {"threads": 12, "batch_size": 768},
            "best_tps": 35.0,
            "baseline_score": 28.0,
            "beat_baseline": True,
        }
        save_phase_results(ctx, "core_engine", core_results)

        # Phase 3: Speculation uses accumulated config
        core_data = load_phase_results(ctx, "core_engine")
        assert core_data is not None
        best_config = {**base_config}
        best_config.update(core_data["best_params"])
        assert best_config["n_gpu_layers"] == 25
        assert best_config["threads"] == 12
        assert best_config["batch_size"] == 768


# ===================================================================
# 7. Scoring consistency
# ===================================================================


@pytest.mark.integration
class TestScoringConsistency:
    """Feed same perf data through compute_score() -> compute_pareto_objectives()
    and verify objectives are consistent with the score."""

    def test_score_and_pareto_agree_on_direction(self):
        slow = PerfResult(tps=10.0, ttft=100.0, prompt_tps=100.0, total_ms=2100.0)
        fast = PerfResult(tps=30.0, ttft=30.0, prompt_tps=400.0, total_ms=2030.0)

        slow_score = compute_score(slow)
        fast_score = compute_score(fast)

        slow_pareto = compute_pareto_objectives(slow)
        fast_pareto = compute_pareto_objectives(fast)

        # Higher TPS -> higher score and higher pareto TPS objective
        assert fast_score > slow_score
        assert fast_pareto.tps > slow_pareto.tps

    def test_pareto_vram_direction(self):
        """Lower VRAM usage -> higher (less negative) neg_vram objective."""
        low_vram = PerfResult(
            tps=25.0,
            ttft=40.0,
            prompt_tps=300.0,
            total_ms=2040.0,
            vram_used_mb=3000.0,
        )
        high_vram = PerfResult(
            tps=25.0,
            ttft=40.0,
            prompt_tps=300.0,
            total_ms=2040.0,
            vram_used_mb=7000.0,
        )

        low_obj = compute_pareto_objectives(low_vram)
        high_obj = compute_pareto_objectives(high_vram)

        # Less VRAM -> higher neg_vram (less negative)
        assert low_obj.neg_vram > high_obj.neg_vram

    def test_quality_factor_propagates(self):
        perf = PerfResult(tps=20.0, ttft=50.0, prompt_tps=300.0, total_ms=2050.0)

        good = compute_pareto_objectives(perf, quality_factor=1.0)
        bad = compute_pareto_objectives(perf, quality_factor=0.5)

        assert good.quality_factor > bad.quality_factor

    def test_zero_tps_gives_zero_score(self):
        perf = PerfResult(tps=0.0, ttft=50.0, prompt_tps=300.0, total_ms=2050.0)
        assert compute_score(perf) == 0.0

    def test_vram_affects_full_mode_score(self):
        """In full mode (with large_tps), VRAM efficiency matters."""
        base = PerfResult(
            tps=25.0,
            ttft=40.0,
            prompt_tps=300.0,
            total_ms=2040.0,
            large_tps=20.0,
            vram_used_mb=2000.0,
            vram_total_mb=8000.0,
        )
        heavy = PerfResult(
            tps=25.0,
            ttft=40.0,
            prompt_tps=300.0,
            total_ms=2040.0,
            large_tps=20.0,
            vram_used_mb=7500.0,
            vram_total_mb=8000.0,
        )

        assert compute_score(base) > compute_score(heavy)


# ===================================================================
# 8. Study resumption
# ===================================================================


@pytest.mark.integration
class TestStudyResumption:
    """Create a study with setup_study(), add trials, save results,
    resume and verify state is preserved."""

    def test_study_resume_preserves_trials(self, tmp_path):
        ctx = _make_ctx(tmp_path)

        # Create initial study
        study, remaining, completed = setup_study(
            ctx, "test_phase", n_trials=20, seed=42
        )
        assert remaining == 20
        assert completed == 0

        # Add some trials
        def objective(trial):
            x = trial.suggest_float("x", 0.0, 10.0)
            return x * 2

        study.optimize(objective, n_trials=5)
        assert len(study.trials) == 5

        # Resume: setup_study with same name should find existing trials
        ctx.fresh_run = False  # don't delete existing study
        study2, remaining2, completed2 = setup_study(
            ctx, "test_phase", n_trials=20, seed=42
        )
        assert completed2 == 5
        assert remaining2 == 15

    def test_recover_best_score_from_resumed_study(self, tmp_path):
        ctx = _make_ctx(tmp_path)

        study, _, _ = setup_study(ctx, "test_recover", n_trials=10, seed=42)

        # Add trials with known perf data
        for i in range(5):
            trial = study.ask()
            trial.set_user_attr("tps", 20.0 + i * 5)
            trial.set_user_attr("ttft", 50.0)
            trial.set_user_attr("prompt_tps", 300.0)
            trial.set_user_attr("total_ms", 2050.0)
            study.tell(trial, 20.0 + i * 5)

        # Recover best score
        best = recover_best_score(study, compute_score)
        assert best > 0

        # The highest TPS trial (i=4, tps=40) should produce the highest score
        perf_best = PerfResult(tps=40.0, ttft=50.0, prompt_tps=300.0, total_ms=2050.0)
        expected_best = compute_score(perf_best)
        assert best == pytest.approx(expected_best, rel=0.01)

    def test_fresh_run_deletes_existing_study(self, tmp_path):
        ctx = _make_ctx(tmp_path)

        # Create and populate a study
        study, _, _ = setup_study(ctx, "test_fresh", n_trials=10, seed=42)

        def objective(trial):
            x = trial.suggest_float("x", 0.0, 10.0)
            return x

        study.optimize(objective, n_trials=3)
        assert len(study.trials) == 3

        # Fresh run should start clean
        ctx.fresh_run = True
        study2, remaining2, completed2 = setup_study(
            ctx, "test_fresh", n_trials=10, seed=42
        )
        assert completed2 == 0
        assert remaining2 == 10
