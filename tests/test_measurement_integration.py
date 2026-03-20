"""Integration tests for measurement and scoring in tps_pro.

Tests mock HTTP responses (requests.Session), not the measurement
functions themselves, to verify end-to-end behavior of:
  - measure_perf_once
  - measure_perf_adaptive
  - compute_score
  - measure_concurrent_load
  - measure_token_uncertainty
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest
import requests

from tps_pro.constants import (
    ADAPTIVE_WARMUP_RUNS,
    QUALITY_GATE_PROMPTS,
    SCORE_PP_BASELINE,
    TTFT_BASELINE_MS,
)
from tps_pro.measurement import (
    compute_pareto_objectives,
    compute_score,
    measure_concurrent_load,
    measure_perf_adaptive,
    measure_perf_once,
    measure_token_uncertainty,
)
from tps_pro.result_types import (
    ConcurrentLoadResult,
    ConcurrentUserResult,
    PerfResult,
    PerfSample,
)

# ===================================================================
# Helpers
# ===================================================================


def _make_ctx(*, vram_total_mb=None, config=None):
    """Build a minimal mock AppContext with a spec'd HTTP session."""
    ctx = MagicMock()
    ctx.http = MagicMock(spec=requests.Session)
    ctx.server_url = "http://127.0.0.1:8090"
    ctx.vram_total_mb = vram_total_mb
    ctx.config = config or {}
    return ctx


def _ok_response(timings):
    """Build a mock 200 response with timings payload."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"timings": timings}
    return resp


def _error_response(status_code=500):
    """Build a mock error response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {}
    return resp


_GOOD_TIMINGS = {
    "predicted_per_second": 55.0,
    "prompt_ms": 120.0,
    "prompt_per_second": 400.0,
    "predicted_ms": 800.0,
}


# ===================================================================
# 1-4: measure_perf_once
# ===================================================================


class TestMeasurePerfOnceIntegration:
    """Integration tests for measure_perf_once via mocked HTTP."""

    @pytest.mark.unit
    def test_returns_perf_result_with_valid_tps(self):
        """1. Returns PerfSample with valid TPS from mocked HTTP response."""
        ctx = _make_ctx()
        ctx.http.post.return_value = _ok_response(_GOOD_TIMINGS)

        with patch("tps_pro.hardware.get_vram_used_mb", return_value=None):
            result = measure_perf_once(ctx, n_predict=50)

        assert result is not None
        assert isinstance(result, PerfSample)
        assert result.tps == 55.0
        assert result.ttft == 120.0
        assert result.prompt_tps == 400.0
        assert result.total_ms == 920.0  # prompt_ms + predicted_ms

    @pytest.mark.unit
    def test_returns_none_on_server_error(self):
        """2. Returns None when server returns error."""
        ctx = _make_ctx()
        ctx.http.post.return_value = _error_response(500)

        with patch("tps_pro.hardware.get_vram_used_mb", return_value=None):
            result = measure_perf_once(ctx)

        assert result is None

    @pytest.mark.unit
    def test_returns_none_on_timeout(self):
        """3. Returns None when server times out."""
        ctx = _make_ctx()
        ctx.http.post.side_effect = requests.Timeout("connection timed out")

        with patch("tps_pro.hardware.get_vram_used_mb", return_value=None):
            result = measure_perf_once(ctx)

        assert result is None

    @pytest.mark.unit
    def test_captures_vram_from_hardware(self):
        """4. Captures VRAM usage from hardware module."""
        ctx = _make_ctx(vram_total_mb=8192.0)
        ctx.http.post.return_value = _ok_response(_GOOD_TIMINGS)

        with patch("tps_pro.hardware.get_vram_used_mb", return_value=5120.0):
            result = measure_perf_once(ctx)

        assert result is not None
        assert result.vram_used_mb == 5120.0
        assert result.vram_total_mb == 8192.0


# ===================================================================
# 5-8: measure_perf_adaptive
# ===================================================================


class TestMeasurePerfAdaptiveIntegration:
    """Integration tests for measure_perf_adaptive via mocked HTTP."""

    @pytest.mark.unit
    def test_returns_result_after_cv_stabilizes(self):
        """5. Returns result after CV stabilizes (identical TPS = zero CV)."""
        ctx = _make_ctx()
        # All calls return identical TPS -> CV = 0 -> stabilizes quickly
        ctx.http.post.return_value = _ok_response(_GOOD_TIMINGS)

        with (
            patch("tps_pro.hardware.get_vram_used_mb", return_value=None),
            patch("tps_pro.state.config", {}),
        ):
            result, promoted = measure_perf_adaptive(ctx, best_score=0.0)

        assert promoted is True
        assert result.tps == 55.0
        assert result.n_runs is not None
        assert result.tps_cv is not None

    @pytest.mark.unit
    def test_returns_warmup_result_when_subsequent_fail(self):
        """6. Returns warmup result when all subsequent attempts fail."""
        ctx = _make_ctx()
        call_count = 0

        def _post_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # First ADAPTIVE_WARMUP_RUNS calls succeed, rest fail
            if call_count <= ADAPTIVE_WARMUP_RUNS:
                return _ok_response(_GOOD_TIMINGS)
            return _error_response(500)

        ctx.http.post.side_effect = _post_side_effect

        with (
            patch("tps_pro.hardware.get_vram_used_mb", return_value=None),
            patch("tps_pro.state.config", {}),
        ):
            result, promoted = measure_perf_adaptive(ctx, best_score=0.0)

        # Still gets promoted with warmup data; result has valid TPS
        assert result.tps == 55.0

    @pytest.mark.unit
    def test_single_run_mode(self):
        """7. Handles single-run mode (runs=1)."""
        ctx = _make_ctx()
        ctx.http.post.return_value = _ok_response(_GOOD_TIMINGS)

        with patch("tps_pro.hardware.get_vram_used_mb", return_value=None):
            result, promoted = measure_perf_adaptive(ctx, runs=1)

        assert promoted is True
        assert result.tps == 55.0

    @pytest.mark.unit
    def test_promotes_best_sample(self):
        """8. Promotes best sample as result (median by score)."""
        ctx = _make_ctx()
        call_count = 0

        # Return varying TPS values so median selection is meaningful
        tps_sequence = [30.0, 50.0, 70.0, 50.0, 50.0]

        def _post_side_effect(*args, **kwargs):
            nonlocal call_count
            idx = min(call_count, len(tps_sequence) - 1)
            tps = tps_sequence[idx]
            call_count += 1
            timings = {
                "predicted_per_second": tps,
                "prompt_ms": 120.0,
                "prompt_per_second": 400.0,
                "predicted_ms": 800.0,
            }
            return _ok_response(timings)

        ctx.http.post.side_effect = _post_side_effect

        with (
            patch("tps_pro.hardware.get_vram_used_mb", return_value=None),
            patch("tps_pro.state.config", {}),
        ):
            result, promoted = measure_perf_adaptive(ctx, best_score=0.0)

        assert promoted is True
        # Median of the collected samples by composite score
        assert result.tps == 50.0


# ===================================================================
# 9-14: compute_score
# ===================================================================


class TestComputeScoreIntegration:
    """Integration tests for compute_score."""

    @pytest.mark.unit
    def test_zero_tps_returns_zero(self):
        """9. Returns 0 for zero TPS."""
        perf = PerfResult(tps=0.0, ttft=100.0, prompt_tps=300.0, total_ms=500.0)
        assert compute_score(perf) == 0.0

    @pytest.mark.unit
    def test_higher_tps_higher_score(self):
        """10. Higher TPS = higher score."""
        low = PerfResult(
            tps=20.0,
            ttft=TTFT_BASELINE_MS,
            prompt_tps=SCORE_PP_BASELINE,
            total_ms=1000.0,
        )
        high = PerfResult(
            tps=80.0,
            ttft=TTFT_BASELINE_MS,
            prompt_tps=SCORE_PP_BASELINE,
            total_ms=1000.0,
        )
        assert compute_score(high) > compute_score(low)

    @pytest.mark.unit
    def test_pp_and_ttft_contribute_to_score(self):
        """11. PP and TTFT contribute to score."""
        baseline = PerfResult(
            tps=50.0,
            ttft=TTFT_BASELINE_MS,
            prompt_tps=SCORE_PP_BASELINE,
            total_ms=1000.0,
        )
        # Better PP and TTFT -> higher score
        better = PerfResult(
            tps=50.0,
            ttft=TTFT_BASELINE_MS / 2,
            prompt_tps=SCORE_PP_BASELINE * 2,
            total_ms=1000.0,
        )
        assert compute_score(better) > compute_score(baseline)

    @pytest.mark.unit
    def test_nan_inf_inputs_clamped(self):
        """12. NaN/Inf inputs get clamped to produce finite output."""
        nan_perf = PerfResult(
            tps=50.0,
            ttft=float("nan"),
            prompt_tps=float("inf"),
            total_ms=1000.0,
        )
        score = compute_score(nan_perf)
        assert math.isfinite(score)
        assert score > 0

        # NaN TPS -> 0 score
        nan_tps = PerfResult(
            tps=float("nan"),
            ttft=100.0,
            prompt_tps=300.0,
            total_ms=500.0,
        )
        assert compute_score(nan_tps) == 0.0

    @pytest.mark.unit
    def test_score_is_deterministic(self):
        """13. Score is deterministic (same input = same output)."""
        perf = PerfResult(
            tps=50.0,
            ttft=TTFT_BASELINE_MS,
            prompt_tps=SCORE_PP_BASELINE,
            total_ms=1000.0,
            vram_used_mb=4096.0,
            vram_total_mb=8192.0,
        )
        scores = [compute_score(perf) for _ in range(100)]
        assert all(s == scores[0] for s in scores)

    @pytest.mark.unit
    def test_pareto_mode_returns_tuple(self):
        """14. Pareto mode returns tuple of objectives."""
        perf = PerfResult(
            tps=60.0,
            ttft=200.0,
            prompt_tps=500.0,
            total_ms=800.0,
            vram_used_mb=5000.0,
        )
        result = compute_pareto_objectives(perf, quality_factor=0.85)

        tps, neg_vram, qf = result
        assert tps == 60.0
        assert neg_vram == -5000.0
        assert qf == 0.85
        assert len(result) == 3


# ===================================================================
# 15-17: measure_concurrent_load
# ===================================================================


class TestMeasureConcurrentLoadIntegration:
    """Integration tests for measure_concurrent_load via mocked async."""

    @pytest.mark.unit
    def test_returns_correct_user_count(self):
        """15. Returns ConcurrentLoadResult with correct user count."""
        ctx = _make_ctx()
        user_results = [
            ConcurrentUserResult(
                user_id=i,
                tps=40.0,
                ttft=100.0,
                prompt_tps=300.0,
                wall_time=2000.0,
                success=True,
            )
            for i in range(6)
        ]

        with (
            patch("tps_pro.measurement.concurrent.HAS_AIOHTTP", True),
            patch("tps_pro.measurement.concurrent.asyncio") as mock_asyncio,
        ):
            mock_asyncio.get_running_loop.side_effect = RuntimeError
            mock_asyncio.run.return_value = user_results

            result = measure_concurrent_load(ctx, n_users=6, n_predict=50)

        assert result is not None
        assert isinstance(result, ConcurrentLoadResult)
        assert result.concurrent_users == 6
        assert result.concurrent_total_tps == 240.0  # 6 * 40
        assert result.concurrent_avg_tps == 40.0
        assert result.concurrent_success_rate == 1.0

    @pytest.mark.unit
    def test_handles_server_failure_gracefully(self):
        """16. Handles server failure gracefully (all users fail)."""
        ctx = _make_ctx()
        user_results = [
            ConcurrentUserResult(
                user_id=i,
                success=False,
                error="status 500",
            )
            for i in range(4)
        ]

        with (
            patch("tps_pro.measurement.concurrent.HAS_AIOHTTP", True),
            patch("tps_pro.measurement.concurrent.asyncio") as mock_asyncio,
        ):
            mock_asyncio.get_running_loop.side_effect = RuntimeError
            mock_asyncio.run.return_value = user_results

            result = measure_concurrent_load(ctx, n_users=4)

        assert result is None

    @pytest.mark.unit
    def test_returns_none_on_timeout(self):
        """17. Returns None on timeout (RuntimeError from async loop)."""
        ctx = _make_ctx()

        with (
            patch("tps_pro.measurement.concurrent.HAS_AIOHTTP", True),
            patch("tps_pro.measurement.concurrent.asyncio") as mock_asyncio,
        ):
            mock_asyncio.get_running_loop.side_effect = RuntimeError
            mock_asyncio.run.side_effect = RuntimeError("event loop failed")

            result = measure_concurrent_load(ctx, n_users=4)

        assert result is None


# ===================================================================
# 18-19: measure_token_uncertainty
# ===================================================================


class TestMeasureTokenUncertaintyIntegration:
    """Integration tests for measure_token_uncertainty via mocked HTTP."""

    @pytest.mark.unit
    def test_returns_result_from_valid_response(self):
        """18. Returns TokenUncertaintyResult from valid response."""
        ctx = _make_ctx()
        logprobs_content = [
            {"logprob": -0.1},
            {"logprob": -0.3},
            {"logprob": -0.8},
            {"logprob": -2.5},
        ]
        json_data = {"choices": [{"logprobs": {"content": logprobs_content}}]}
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = json_data
        ctx.http.post.return_value = resp

        result = measure_token_uncertainty(ctx)

        assert result is not None
        n_prompts = len(QUALITY_GATE_PROMPTS)
        # 4 logprobs per prompt, all negative -> all counted
        assert result.total_tokens == 4 * n_prompts
        # Only logprob < -0.5 threshold counts as uncertain
        # -0.8 and -2.5 are below threshold -> 2 per prompt
        assert result.uncertain_count == 2 * n_prompts
        assert result.tail_avg < 0

    @pytest.mark.unit
    def test_handles_empty_logprobs(self):
        """19. Handles empty logprobs -> returns None."""
        ctx = _make_ctx()
        json_data = {"choices": [{"logprobs": {"content": []}}]}
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = json_data
        ctx.http.post.return_value = resp

        result = measure_token_uncertainty(ctx)

        assert result is None
