"""Tests for KV + Context Sweep logic — boot scanning, quality,
scoring, results logging, and phase return shape.

Tests cover:
    1.  _find_max_bootable doubles context until OOM then binary searches
    2.  _find_max_bootable returns None when 4096 fails
    3.  _find_max_bootable caps at model max context
    7.  Quality test passes when model returns correct answer
    8.  Quality test fails when model returns wrong answer
    9.  PP timeout stops testing higher contexts
    10. Scoring formula weights TPS, context bonus, and PP speed correctly
    11. Results section shows Baseline/Optimal/Params format
    12. Phase returns PhaseReturnDict with kv_cache_type and context
"""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import requests
from _ctx_factory import make_ctx_from_defaults

from tps_pro.phases.kv_context_sweep import phase_kv_context_sweep
from tps_pro.phases.kv_sweep_boot import (
    _find_max_bootable,
)
from tps_pro.phases.kv_sweep_measure import (
    _run_quality_test,
    log_sweep_results,
    measure_single_kv_type,
    score_measurements,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_boot_side_effect(pass_set: set[int]):
    """Return a side effect for boot_server_with_jinja_recovery.

    Contexts in *pass_set* succeed ("ok"), everything else OOMs.
    """

    def _boot(ctx, config):
        ctx_size = config.get("context", 4096)
        status = "ok" if ctx_size in pass_set else "oom"
        proc = SimpleNamespace(load_time_ms=100)
        return proc, status

    return _boot


def _quality_http_response(content: str, prompt_ms: int = 500, predicted_ms: int = 200):
    """Build a mock requests.Response for the /v1/chat/completions endpoint."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
        "timings": {
            "prompt_ms": prompt_ms,
            "prompt_n": 1000,
            "predicted_ms": predicted_ms,
            "predicted_n": 20,
        },
    }
    return resp


# ---------------------------------------------------------------------------
# 1. _find_max_bootable doubles then binary-searches
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFindMaxBootable:
    @patch("tps_pro.phases.kv_sweep_boot.kill_server")
    @patch("tps_pro.phases.kv_sweep_boot.boot_server_with_jinja_recovery")
    def test_doubles_then_binary_searches(self, mock_boot, mock_kill):
        """Doubling 4096 -> 8192 -> 16384 passes; 32768 fails.
        Binary search between 16384 and 32768 narrows to ~16384.
        """
        pass_set = {4096, 8192, 16384}
        mock_boot.side_effect = _make_boot_side_effect(pass_set)
        ctx = make_ctx_from_defaults()

        result = _find_max_bootable(ctx, "f16")

        assert result is not None
        assert result >= 16384
        # The binary search converges within _BISECT_THRESHOLD (2048) of
        # the boundary. Because only exact sizes in pass_set boot, the
        # search should settle at 16384 (the last passing size, rounded
        # down to 1024 alignment).
        assert result % 1024 == 0

    # -------------------------------------------------------------------
    # 2. Returns None when 4096 fails
    # -------------------------------------------------------------------

    @patch("tps_pro.phases.kv_sweep_boot.kill_server")
    @patch("tps_pro.phases.kv_sweep_boot.boot_server_with_jinja_recovery")
    def test_returns_none_when_4096_fails(self, mock_boot, mock_kill):
        """Even the minimum 4096 context OOMs — returns None."""
        mock_boot.side_effect = _make_boot_side_effect(set())
        ctx = make_ctx_from_defaults()

        result = _find_max_bootable(ctx, "f16")

        assert result is None

    # -------------------------------------------------------------------
    # 3. Caps at model max context
    # -------------------------------------------------------------------

    @patch("tps_pro.phases.kv_sweep_boot.kill_server")
    @patch("tps_pro.phases.kv_sweep_boot.boot_server_with_jinja_recovery")
    def test_caps_at_model_max_context(self, mock_boot, mock_kill):
        """When model_max_ctx is 8192, result should not exceed 8192."""
        # Everything boots fine — unlimited VRAM.
        mock_boot.side_effect = _make_boot_side_effect(
            {4096, 8192, 16384, 32768, 65536}
        )
        ctx = make_ctx_from_defaults()

        result = _find_max_bootable(ctx, "f16", model_max_ctx=8192)

        assert result is not None
        assert result <= 8192


# ---------------------------------------------------------------------------
# 4–6. (estimate_kv_cache_mb and estimate_safe_max_context removed as dead code)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7–8. Quality test pass / fail
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestQualityTest:
    def test_quality_passes_with_correct_answer(self):
        """Model returns the expected needle — quality_pass is True."""
        ctx = make_ctx_from_defaults()
        ctx.http.post.return_value = _quality_http_response("ALPHA-7749")

        result = _run_quality_test(ctx, "some long prompt", "ALPHA-7749")

        assert result is not None
        assert result["quality_pass"] is True
        assert result["tps"] > 0
        assert result["pp"] > 0

    def test_quality_fails_with_wrong_answer(self):
        """Model returns the wrong value — quality_pass is False."""
        ctx = make_ctx_from_defaults()
        ctx.http.post.return_value = _quality_http_response("BETA-3182")

        result = _run_quality_test(ctx, "some long prompt", "ALPHA-7749")

        assert result is not None
        assert result["quality_pass"] is False


# ---------------------------------------------------------------------------
# 9. PP timeout stops testing higher contexts
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPPTimeout:
    @patch("tps_pro.phases.kv_sweep_measure._read_vram_mb", return_value=4000.0)
    @patch("tps_pro.phases.kv_sweep_measure.kill_server")
    @patch("tps_pro.phases.kv_sweep_measure.boot_server_with_jinja_recovery")
    def test_pp_timeout_stops_testing(self, mock_boot, mock_kill, mock_vram):
        """When _run_quality_test returns None (PP timeout), measure_single_kv_type
        stops testing higher contexts."""
        ctx = make_ctx_from_defaults()

        # Boot always succeeds.
        mock_boot.return_value = (SimpleNamespace(load_time_ms=100), "ok")

        # Quality test: first call succeeds, second call times out (ReadTimeout).
        ctx.http.post.side_effect = [
            _quality_http_response("ALPHA-7749"),
            requests.exceptions.ReadTimeout("timeout"),
        ]

        test_prompts = {
            4096: ("prompt4k", "ALPHA-7749"),
            8192: ("prompt8k", "ALPHA-7749"),
            16384: ("prompt16k", "ALPHA-7749"),
        }

        measurements, max_practical = measure_single_kv_type(
            ctx, "f16", 16384, test_prompts, {"threads": 8}
        )

        # Should have only one measurement (4096 passed, 8192 timed out).
        assert len(measurements) == 1
        assert measurements[0]["context"] == 4096
        assert max_practical == 4096


# ---------------------------------------------------------------------------
# 10. Scoring formula
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestScoringFormula:
    def test_scoring_weights(self):
        """Score = (tps * 0.5) + (context_bonus * 0.3) + (pp_speed * 0.2)."""
        measurements = [
            {
                "kv_type": "f16",
                "context": 4096,
                "tps": 40.0,
                "pp": 200,
                "ttft": 500,
                "vram_mb": 4000,
                "quality_pass": True,
            },
            {
                "kv_type": "q8_0",
                "context": 8192,
                "tps": 45.0,
                "pp": 300,
                "ttft": 300,
                "vram_mb": 3500,
                "quality_pass": True,
            },
        ]

        best_score, best_kv, best_ctx, best_m = score_measurements(measurements)

        # Verify the formula for the second measurement (should win):
        context_bonus = math.log2(8192 / 4096) * 10  # log2(2) * 10 = 10
        pp_speed = min(1000 / 300, 100)  # ~3.33
        expected_score = (45.0 * 0.5) + (context_bonus * 0.3) + (pp_speed * 0.2)

        assert best_kv == "q8_0"
        assert best_ctx == 8192
        assert abs(best_score - expected_score) < 0.01

    def test_quality_fail_excluded_from_scoring(self):
        """Measurements with quality_pass=False are not scored."""
        measurements = [
            {
                "kv_type": "f16",
                "context": 4096,
                "tps": 40.0,
                "pp": 200,
                "ttft": 500,
                "vram_mb": 4000,
                "quality_pass": False,
            },
        ]

        best_score, best_kv, best_ctx, best_m = score_measurements(measurements)

        assert best_kv is None
        assert best_ctx is None

    def test_empty_measurements(self):
        """No measurements at all returns None."""
        best_score, best_kv, best_ctx, best_m = score_measurements([])
        assert best_kv is None
        assert best_ctx is None


# ---------------------------------------------------------------------------
# 11. Results log format — Baseline / Optimal / Params
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestResultsLog:
    def test_results_section_format(self, caplog):
        """log_sweep_results emits Baseline, Optimal, and Params lines."""
        baseline_m = {
            "kv_type": "f16",
            "context": 4096,
            "tps": 35.0,
            "pp": 200,
            "ttft": 600,
            "vram_mb": 4096,
            "quality_pass": True,
        }
        best_m = {
            "kv_type": "q8_0",
            "context": 8192,
            "tps": 42.0,
            "pp": 300,
            "ttft": 400,
            "vram_mb": 3500,
            "quality_pass": True,
        }

        with caplog.at_level(logging.INFO, logger="tps_pro.phases.kv_sweep_measure"):
            log_sweep_results(baseline_m, best_m, "q8_0", 8192)

        log_text = caplog.text
        assert "RESULTS" in log_text
        assert "Baseline" in log_text
        assert "Optimal" in log_text
        assert "Params" in log_text
        assert "kv_cache_type=q8_0" in log_text
        assert "context=8192" in log_text


# ---------------------------------------------------------------------------
# 12. Phase returns PhaseReturnDict with kv_cache_type and context
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhaseReturnDict:
    @patch("tps_pro.phases.kv_context_sweep.save_phase_results")
    @patch("tps_pro.phases.kv_context_sweep.log_sweep_results")
    @patch("tps_pro.phases.kv_context_sweep.score_measurements")
    @patch("tps_pro.phases.kv_context_sweep.measure_single_kv_type")
    @patch("tps_pro.phases.kv_context_sweep.prepare_test_prompts")
    @patch("tps_pro.phases.kv_context_sweep.discover_bootable_contexts")
    @patch("tps_pro.phases.kv_context_sweep.get_model_max_context", return_value=0)
    @patch("tps_pro.phases.kv_context_sweep.get_model_metadata", return_value={})
    @patch("tps_pro.phases.kv_context_sweep.load_phase_results", return_value=None)
    @patch("tps_pro.phases.kv_context_sweep.kill_server")
    def test_returns_phase_return_dict(
        self,
        mock_kill,
        mock_load,
        mock_meta,
        mock_max_ctx,
        mock_discover,
        mock_prepare,
        mock_measure,
        mock_score,
        mock_log,
        mock_save,
        make_ctx,
    ):
        """Full phase returns PhaseReturnDict with kv_cache_type and context."""
        ctx = make_ctx()

        # Discover: f16 boots to 8192, others fail.
        mock_discover.return_value = (
            {"f16": 8192, "q8_0": None, "q4_0": None},
            {4096, 8192},
        )
        mock_prepare.return_value = {
            4096: ("p4k", "ALPHA-7749"),
            8192: ("p8k", "ALPHA-7749"),
        }

        # Measurements for f16.
        f16_measurements = [
            {
                "kv_type": "f16",
                "context": 4096,
                "tps": 40.0,
                "pp": 200,
                "ttft": 500,
                "vram_mb": 4000,
                "quality_pass": True,
            },
            {
                "kv_type": "f16",
                "context": 8192,
                "tps": 38.0,
                "pp": 150,
                "ttft": 800,
                "vram_mb": 5000,
                "quality_pass": True,
            },
        ]
        mock_measure.return_value = (f16_measurements, 8192)

        best_m = f16_measurements[1]
        mock_score.return_value = (25.0, "f16", 8192, best_m)

        result = phase_kv_context_sweep(ctx, force=True)

        assert result is not None
        assert "best_params" in result
        assert result["best_params"]["kv_cache_type"] == "f16"
        assert result["best_params"]["context"] == 8192
        assert result["phase_name"] == "kv_context_sweep"
