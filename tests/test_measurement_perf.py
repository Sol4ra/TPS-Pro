"""Tests for measurement.py: measure_perf_once, measure_perf_adaptive,
_aggregate_samples, measure_concurrent_load, and measure_token_uncertainty.

All HTTP and external dependencies are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tps_pro.constants import (
    ADAPTIVE_WARMUP_RUNS,
    CV_MAX_RUNS,
    QUALITY_GATE_PROMPTS,
    SCORE_PP_BASELINE,
    TTFT_BASELINE_MS,
)
from tps_pro.measurement.perf_measurement import _aggregate_samples
from tps_pro.result_types import ConcurrentUserResult, PerfResult


def _make_perf(
    tps=50.0,
    prompt_tps=SCORE_PP_BASELINE,
    ttft=TTFT_BASELINE_MS,
    total_ms=1000.0,
    **extra,
) -> PerfResult:
    """Build a PerfResult with sensible defaults."""
    return PerfResult(
        tps=tps, prompt_tps=prompt_tps, ttft=ttft, total_ms=total_ms, **extra
    )


def _mock_response(status_code=200, json_data=None):
    """Create a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


def _make_mock_ctx():
    """Create a mock AppContext with the attributes used by measurement functions."""
    mock_ctx = MagicMock()
    mock_ctx.http = MagicMock()
    mock_ctx.server_url = "http://127.0.0.1:8090"
    mock_ctx.vram_total_mb = None
    mock_ctx.config = {}
    return mock_ctx


# ===================================================================
# measure_perf_once
# ===================================================================


class TestMeasurePerfOnce:
    """Tests for measure_perf_once — mocks ctx.http.post."""

    @pytest.mark.unit
    def test_normal_response_parsing(self):
        json_data = {
            "timings": {
                "predicted_per_second": 55.0,
                "prompt_ms": 120.0,
                "prompt_per_second": 400.0,
                "predicted_ms": 800.0,
            }
        }
        mock_resp = _mock_response(200, json_data)
        mock_ctx = _make_mock_ctx()
        with patch("tps_pro.hardware.get_vram_used_mb", return_value=None):
            mock_ctx.http.post.return_value = mock_resp
            from tps_pro.measurement import measure_perf_once

            result = measure_perf_once(mock_ctx, n_predict=50)
        assert result is not None
        assert result.tps == 55.0
        assert result.ttft == 120.0
        assert result.prompt_tps == 400.0
        assert result.total_ms == 920.0

    @pytest.mark.unit
    def test_normal_response_with_vram(self):
        json_data = {
            "timings": {
                "predicted_per_second": 55.0,
                "prompt_ms": 120.0,
                "prompt_per_second": 400.0,
                "predicted_ms": 800.0,
            }
        }
        mock_resp = _mock_response(200, json_data)
        mock_ctx = _make_mock_ctx()
        with patch(
            "tps_pro.hardware.get_vram_used_mb",
            return_value=5000.0,
        ):
            mock_ctx.http.post.return_value = mock_resp
            mock_ctx.vram_total_mb = 8000.0
            from tps_pro.measurement import measure_perf_once

            result = measure_perf_once(mock_ctx)
        assert result is not None
        assert result.vram_used_mb == 5000.0
        assert result.vram_total_mb == 8000.0

    @pytest.mark.unit
    def test_zero_tps_returns_none(self):
        json_data = {
            "timings": {
                "predicted_per_second": 0,
                "prompt_ms": 120.0,
                "prompt_per_second": 400.0,
                "predicted_ms": 800.0,
            }
        }
        mock_resp = _mock_response(200, json_data)
        mock_ctx = _make_mock_ctx()
        with patch("tps_pro.hardware.get_vram_used_mb"):
            mock_ctx.http.post.return_value = mock_resp
            from tps_pro.measurement import measure_perf_once

            result = measure_perf_once(mock_ctx)
        assert result is None

    @pytest.mark.unit
    def test_http_error_status_returns_none(self):
        mock_resp = _mock_response(500)
        mock_ctx = _make_mock_ctx()
        with patch("tps_pro.hardware.get_vram_used_mb"):
            mock_ctx.http.post.return_value = mock_resp
            from tps_pro.measurement import measure_perf_once

            result = measure_perf_once(mock_ctx)
        assert result is None

    @pytest.mark.unit
    def test_timeout_returns_none(self):
        import requests

        mock_ctx = _make_mock_ctx()
        with patch("tps_pro.hardware.get_vram_used_mb"):
            mock_ctx.http.post.side_effect = requests.Timeout("timeout")
            from tps_pro.measurement import measure_perf_once

            result = measure_perf_once(mock_ctx)
        assert result is None

    @pytest.mark.unit
    def test_connection_error_returns_none(self):
        import requests

        mock_ctx = _make_mock_ctx()
        with patch("tps_pro.hardware.get_vram_used_mb"):
            mock_ctx.http.post.side_effect = requests.ConnectionError("refused")
            from tps_pro.measurement import measure_perf_once

            result = measure_perf_once(mock_ctx)
        assert result is None

    @pytest.mark.unit
    def test_malformed_json_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("bad json")
        mock_ctx = _make_mock_ctx()
        with patch("tps_pro.hardware.get_vram_used_mb"):
            mock_ctx.http.post.return_value = mock_resp
            from tps_pro.measurement import measure_perf_once

            result = measure_perf_once(mock_ctx)
        assert result is None

    @pytest.mark.unit
    def test_missing_timings_key_returns_none(self):
        mock_resp = _mock_response(200, {"choices": []})
        mock_ctx = _make_mock_ctx()
        with patch("tps_pro.hardware.get_vram_used_mb"):
            mock_ctx.http.post.return_value = mock_resp
            from tps_pro.measurement import measure_perf_once

            result = measure_perf_once(mock_ctx)
        assert result is None

    @pytest.mark.unit
    def test_custom_prompt_forwarded(self):
        json_data = {
            "timings": {
                "predicted_per_second": 55.0,
                "prompt_ms": 100.0,
                "prompt_per_second": 300.0,
                "predicted_ms": 500.0,
            }
        }
        mock_resp = _mock_response(200, json_data)
        mock_ctx = _make_mock_ctx()
        with patch("tps_pro.hardware.get_vram_used_mb", return_value=None):
            mock_ctx.http.post.return_value = mock_resp
            from tps_pro.measurement import measure_perf_once

            measure_perf_once(mock_ctx, prompt="Custom test prompt")
        call_args = mock_ctx.http.post.call_args
        payload = call_args[1]["json"]
        assert payload["messages"][0]["content"] == "Custom test prompt"

    @pytest.mark.unit
    def test_spec_params_forwarded(self):
        json_data = {
            "timings": {
                "predicted_per_second": 55.0,
                "prompt_ms": 100.0,
                "prompt_per_second": 300.0,
                "predicted_ms": 500.0,
            }
        }
        mock_resp = _mock_response(200, json_data)
        mock_ctx = _make_mock_ctx()
        with patch("tps_pro.hardware.get_vram_used_mb", return_value=None):
            mock_ctx.http.post.return_value = mock_resp
            from tps_pro.measurement import measure_perf_once

            measure_perf_once(mock_ctx, spec_params={"draft_max": 5})
        call_args = mock_ctx.http.post.call_args
        payload = call_args[1]["json"]
        assert payload["speculative"] == {"draft_max": 5}


# ===================================================================
# measure_perf_adaptive
# ===================================================================


class TestMeasurePerfAdaptive:
    """Tests for measure_perf_adaptive — mocks measure_perf_once."""

    def _make_sample(self, tps=50.0):
        from tps_pro.result_types import PerfSample

        return PerfSample(
            tps=tps,
            prompt_tps=SCORE_PP_BASELINE,
            ttft=TTFT_BASELINE_MS,
            total_ms=1000.0,
        )

    @pytest.mark.unit
    def test_fixed_run_mode(self):
        sample = self._make_sample()
        mock_ctx = _make_mock_ctx()
        with patch(
            "tps_pro.measurement.perf_measurement.measure_perf_once",
            return_value=sample,
        ):
            from tps_pro.measurement import measure_perf_adaptive

            result, promoted = measure_perf_adaptive(mock_ctx, runs=3)
        assert promoted is True
        assert result.tps == 50.0

    @pytest.mark.unit
    def test_fixed_run_all_fail_returns_zeros(self):
        mock_ctx = _make_mock_ctx()
        with patch(
            "tps_pro.measurement.perf_measurement.measure_perf_once",
            return_value=None,
        ):
            from tps_pro.measurement import measure_perf_adaptive

            result, promoted = measure_perf_adaptive(mock_ctx, runs=3)
        assert promoted is True
        assert result.tps == 0.0

    @pytest.mark.unit
    def test_warmup_gate_all_fail_returns_not_promoted(self):
        mock_ctx = _make_mock_ctx()
        with patch(
            "tps_pro.measurement.perf_measurement.measure_perf_once",
            return_value=None,
        ):
            from tps_pro.measurement import measure_perf_adaptive

            result, promoted = measure_perf_adaptive(mock_ctx, best_score=100.0)
        assert promoted is False
        assert result.tps == 0.0

    @pytest.mark.unit
    def test_warmup_gate_prunes_bad_config(self):
        weak_sample = self._make_sample(tps=5.0)
        call_count = 0

        def _mock_once(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return weak_sample

        mock_ctx = _make_mock_ctx()
        with (
            patch(
                "tps_pro.measurement.perf_measurement.measure_perf_once",
                side_effect=_mock_once,
            ),
            patch("tps_pro.state.config", {}),
        ):
            from tps_pro.measurement import measure_perf_adaptive

            result, promoted = measure_perf_adaptive(mock_ctx, best_score=100.0)
        assert promoted is False
        assert call_count == ADAPTIVE_WARMUP_RUNS

    @pytest.mark.unit
    def test_promoted_when_competitive(self):
        good_sample = self._make_sample(tps=60.0)
        mock_ctx = _make_mock_ctx()
        with (
            patch(
                "tps_pro.measurement.perf_measurement.measure_perf_once",
                return_value=good_sample,
            ),
            patch("tps_pro.state.config", {}),
        ):
            from tps_pro.measurement import measure_perf_adaptive

            result, promoted = measure_perf_adaptive(mock_ctx, best_score=50.0)
        assert promoted is True
        assert result.tps == 60.0

    @pytest.mark.unit
    def test_cv_stabilization_stops_early(self):
        identical_sample = self._make_sample(tps=50.0)
        call_count = 0

        def _mock_once(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return identical_sample

        mock_ctx = _make_mock_ctx()
        with (
            patch(
                "tps_pro.measurement.perf_measurement.measure_perf_once",
                side_effect=_mock_once,
            ),
            patch("tps_pro.state.config", {}),
        ):
            from tps_pro.measurement import measure_perf_adaptive

            result, promoted = measure_perf_adaptive(mock_ctx, best_score=0.0)
        assert promoted is True
        assert call_count <= ADAPTIVE_WARMUP_RUNS + (CV_MAX_RUNS - ADAPTIVE_WARMUP_RUNS)

    @pytest.mark.unit
    def test_max_runs_limit(self):
        call_count = 0
        tps_val = 10.0

        def _mock_once(*args, **kwargs):
            nonlocal call_count, tps_val
            call_count += 1
            tps_val += 20.0
            from tps_pro.result_types import PerfSample

            return PerfSample(
                tps=tps_val,
                prompt_tps=SCORE_PP_BASELINE,
                ttft=TTFT_BASELINE_MS,
                total_ms=1000.0,
            )

        mock_ctx = _make_mock_ctx()
        with (
            patch(
                "tps_pro.measurement.perf_measurement.measure_perf_once",
                side_effect=_mock_once,
            ),
            patch("tps_pro.state.config", {}),
        ):
            from tps_pro.measurement import measure_perf_adaptive

            result, promoted = measure_perf_adaptive(mock_ctx, best_score=0.0)
        assert promoted is True
        assert call_count == CV_MAX_RUNS

    @pytest.mark.unit
    def test_best_score_zero_always_promotes(self):
        sample = self._make_sample(tps=1.0)
        mock_ctx = _make_mock_ctx()
        with (
            patch(
                "tps_pro.measurement.perf_measurement.measure_perf_once",
                return_value=sample,
            ),
            patch("tps_pro.state.config", {}),
        ):
            from tps_pro.measurement import measure_perf_adaptive

            result, promoted = measure_perf_adaptive(mock_ctx, best_score=0.0)
        assert promoted is True

    @pytest.mark.unit
    def test_concurrent_load_integrated_when_configured(self):
        sample = self._make_sample(tps=50.0)
        from tps_pro.result_types import ConcurrentLoadResult

        load_data = ConcurrentLoadResult(
            concurrent_total_tps=180.0,
            concurrent_avg_tps=45.0,
            concurrent_avg_ttft=100.0,
            concurrent_avg_wall_ms=2000.0,
            concurrent_max_wall_ms=2500.0,
            concurrent_success_rate=1.0,
            concurrent_users=4,
        )
        mock_ctx = _make_mock_ctx()
        with (
            patch(
                "tps_pro.measurement.perf_measurement.measure_perf_once",
                return_value=sample,
            ),
            patch(
                "tps_pro.measurement.concurrent.measure_concurrent_load",
                return_value=load_data,
            ),
            patch(
                "tps_pro.state.config",
                {"simulate_users": 4},
            ),
        ):
            from tps_pro.measurement import measure_perf_adaptive

            result, promoted = measure_perf_adaptive(mock_ctx, best_score=0.0)
        assert promoted is True
        assert result.concurrent_total_tps == 180.0

    @pytest.mark.unit
    def test_single_sample_only_in_fixed_mode(self):
        sample = self._make_sample(tps=42.0)
        mock_ctx = _make_mock_ctx()
        with patch(
            "tps_pro.measurement.perf_measurement.measure_perf_once",
            return_value=sample,
        ):
            from tps_pro.measurement import measure_perf_adaptive

            result, promoted = measure_perf_adaptive(mock_ctx, runs=1)
        assert promoted is True
        assert result.tps == 42.0

    @pytest.mark.unit
    def test_adaptive_single_warmup_success_rest_fail(self):
        sample = self._make_sample(tps=30.0)
        call_count = 0

        def _mock_once(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return sample
            return None

        mock_ctx = _make_mock_ctx()
        with (
            patch(
                "tps_pro.measurement.perf_measurement.measure_perf_once",
                side_effect=_mock_once,
            ),
            patch("tps_pro.state.config", {}),
        ):
            from tps_pro.measurement import measure_perf_adaptive

            result, promoted = measure_perf_adaptive(mock_ctx, best_score=0.0)
        assert result.tps == 30.0
        assert promoted is True


# ===================================================================
# _aggregate_samples
# ===================================================================


class TestAggregateSamples:
    """Tests for _aggregate_samples."""

    @pytest.mark.unit
    def test_empty_returns_zeros(self):
        result = _aggregate_samples([])
        assert result.tps == 0.0
        assert result.ttft == 0.0
        assert result.prompt_tps == 0.0
        assert result.total_ms == 0.0

    @pytest.mark.unit
    def test_single_sample_returns_same_values(self):
        sample = PerfResult(tps=42.0, ttft=100, prompt_tps=500, total_ms=200)
        result = _aggregate_samples([sample])
        assert result.tps == sample.tps
        assert result.ttft == sample.ttft

    @pytest.mark.unit
    def test_returns_median_by_composite_score(self):
        low = _make_perf(tps=10, prompt_tps=100, ttft=1000)
        mid = _make_perf(tps=50)
        high = _make_perf(tps=90, prompt_tps=SCORE_PP_BASELINE * 2, ttft=200)
        result = _aggregate_samples([high, low, mid])
        assert result.tps == mid.tps

    @pytest.mark.unit
    def test_even_count_returns_upper_median(self):
        s1 = _make_perf(tps=10, prompt_tps=100, ttft=1000)
        s2 = _make_perf(tps=30, prompt_tps=200, ttft=700)
        s3 = _make_perf(tps=50)
        s4 = _make_perf(tps=70, prompt_tps=400, ttft=300)
        result = _aggregate_samples([s4, s1, s3, s2])
        assert result.tps == s3.tps

    @pytest.mark.unit
    def test_two_samples_returns_higher(self):
        low = _make_perf(tps=10, prompt_tps=100, ttft=1000)
        high = _make_perf(tps=80)
        result = _aggregate_samples([low, high])
        assert result.tps == high.tps

    @pytest.mark.unit
    def test_identical_samples_returns_one(self):
        s = _make_perf(tps=50)
        samples = [PerfResult.from_dict(s.to_dict()) for _ in range(5)]
        result = _aggregate_samples(samples)
        assert result.tps == 50.0


# ===================================================================
# measure_concurrent_load
# ===================================================================


class TestMeasureConcurrentLoad:
    """Tests for measure_concurrent_load — mocks aiohttp.ClientSession."""

    @pytest.mark.unit
    def test_no_aiohttp_returns_none(self):
        mock_ctx = _make_mock_ctx()
        with patch("tps_pro.measurement.concurrent.HAS_AIOHTTP", False):
            from tps_pro.measurement import measure_concurrent_load

            result = measure_concurrent_load(mock_ctx, n_users=4)
        assert result is None

    @pytest.mark.unit
    def test_successful_concurrent_requests(self):
        mock_ctx = _make_mock_ctx()
        with (
            patch("tps_pro.measurement.concurrent.HAS_AIOHTTP", True),
            patch("tps_pro.measurement.concurrent.asyncio") as mock_asyncio,
        ):
            mock_asyncio.get_running_loop.side_effect = RuntimeError
            mock_asyncio.run.return_value = [
                ConcurrentUserResult(
                    user_id=i,
                    tps=40.0,
                    ttft=100.0,
                    prompt_tps=300.0,
                    wall_time=2000.0,
                    success=True,
                )
                for i in range(4)
            ]
            from tps_pro.measurement import measure_concurrent_load

            result = measure_concurrent_load(mock_ctx, n_users=4, n_predict=50)
        assert result is not None
        assert result.concurrent_total_tps == 160.0
        assert result.concurrent_avg_tps == 40.0
        assert result.concurrent_success_rate == 1.0
        assert result.concurrent_users == 4

    @pytest.mark.unit
    def test_some_requests_fail(self):
        results = [
            ConcurrentUserResult(
                user_id=0,
                tps=40.0,
                ttft=100.0,
                prompt_tps=300.0,
                wall_time=2000.0,
                success=True,
            ),
            ConcurrentUserResult(user_id=1, success=False, error="timeout"),
            ConcurrentUserResult(
                user_id=2,
                tps=50.0,
                ttft=80.0,
                prompt_tps=350.0,
                wall_time=1800.0,
                success=True,
            ),
            ConcurrentUserResult(user_id=3, success=False, error="status 500"),
        ]
        mock_ctx = _make_mock_ctx()
        with (
            patch("tps_pro.measurement.concurrent.HAS_AIOHTTP", True),
            patch("tps_pro.measurement.concurrent.asyncio") as mock_asyncio,
        ):
            mock_asyncio.get_running_loop.side_effect = RuntimeError
            mock_asyncio.run.return_value = results
            from tps_pro.measurement import measure_concurrent_load

            result = measure_concurrent_load(mock_ctx, n_users=4)
        assert result is not None
        assert result.concurrent_total_tps == 90.0
        assert result.concurrent_success_rate == 0.5

    @pytest.mark.unit
    def test_all_requests_fail_returns_none(self):
        results = [
            ConcurrentUserResult(user_id=0, success=False, error="timeout"),
            ConcurrentUserResult(user_id=1, success=False, error="timeout"),
        ]
        mock_ctx = _make_mock_ctx()
        with (
            patch("tps_pro.measurement.concurrent.HAS_AIOHTTP", True),
            patch("tps_pro.measurement.concurrent.asyncio") as mock_asyncio,
        ):
            mock_asyncio.get_running_loop.side_effect = RuntimeError
            mock_asyncio.run.return_value = results
            from tps_pro.measurement import measure_concurrent_load

            result = measure_concurrent_load(mock_ctx, n_users=2)
        assert result is None

    @pytest.mark.unit
    def test_scaling_efficiency_calculation(self):
        results = [
            ConcurrentUserResult(
                user_id=i,
                tps=30.0,
                ttft=150.0,
                prompt_tps=250.0,
                wall_time=2500.0,
                success=True,
            )
            for i in range(4)
        ]
        mock_ctx = _make_mock_ctx()
        with (
            patch("tps_pro.measurement.concurrent.HAS_AIOHTTP", True),
            patch("tps_pro.measurement.concurrent.asyncio") as mock_asyncio,
        ):
            mock_asyncio.get_running_loop.side_effect = RuntimeError
            mock_asyncio.run.return_value = results
            from tps_pro.measurement import measure_concurrent_load

            result = measure_concurrent_load(mock_ctx, n_users=4, n_predict=50)
        assert result.concurrent_total_tps == 120.0
        assert result.concurrent_avg_tps == 30.0
        assert result.concurrent_avg_ttft == 150.0
        assert result.concurrent_max_wall_ms == 2500.0

    @pytest.mark.unit
    def test_runtime_error_returns_none(self):
        mock_ctx = _make_mock_ctx()
        with (
            patch("tps_pro.measurement.concurrent.HAS_AIOHTTP", True),
            patch("tps_pro.measurement.concurrent.asyncio") as mock_asyncio,
        ):
            mock_asyncio.get_running_loop.side_effect = RuntimeError
            mock_asyncio.run.side_effect = RuntimeError("event loop error")
            from tps_pro.measurement import measure_concurrent_load

            result = measure_concurrent_load(mock_ctx, n_users=4)
        assert result is None


# ===================================================================
# measure_token_uncertainty
# ===================================================================


class TestMeasureTokenUncertainty:
    """Tests for measure_token_uncertainty — mocks ctx.http.post."""

    @pytest.mark.unit
    def test_normal_logprob_extraction(self):
        logprobs_content = [
            {"logprob": -0.1},
            {"logprob": -0.5},
            {"logprob": -1.0},
            {"logprob": -2.5},
            {"logprob": -0.3},
        ]
        json_data = {"choices": [{"logprobs": {"content": logprobs_content}}]}
        mock_resp = _mock_response(200, json_data)
        n_prompts = len(QUALITY_GATE_PROMPTS)
        mock_ctx = _make_mock_ctx()
        mock_ctx.http.post.return_value = mock_resp
        from tps_pro.measurement import measure_token_uncertainty

        result = measure_token_uncertainty(mock_ctx)
        assert result is not None
        assert result["total_tokens"] == 5 * n_prompts
        assert result["uncertain_count"] == 2 * n_prompts
        assert "tail_avg" in result

    @pytest.mark.unit
    def test_missing_logprobs_in_response(self):
        json_data = {"choices": [{"message": {"content": "Hello"}}]}
        mock_resp = _mock_response(200, json_data)
        mock_ctx = _make_mock_ctx()
        mock_ctx.http.post.return_value = mock_resp
        from tps_pro.measurement import measure_token_uncertainty

        result = measure_token_uncertainty(mock_ctx)
        assert result is None

    @pytest.mark.unit
    def test_http_failure_returns_none(self):
        import requests

        mock_ctx = _make_mock_ctx()
        mock_ctx.http.post.side_effect = requests.RequestException("fail")
        from tps_pro.measurement import measure_token_uncertainty

        result = measure_token_uncertainty(mock_ctx)
        assert result is None

    @pytest.mark.unit
    def test_non_200_status_returns_none(self):
        mock_resp = _mock_response(500)
        mock_ctx = _make_mock_ctx()
        mock_ctx.http.post.return_value = mock_resp
        from tps_pro.measurement import measure_token_uncertainty

        result = measure_token_uncertainty(mock_ctx)
        assert result is None

    @pytest.mark.unit
    def test_positive_logprobs_filtered_out(self):
        logprobs_content = [
            {"logprob": 0.5},
            {"logprob": 0.0},
            {"logprob": -0.2},
        ]
        json_data = {"choices": [{"logprobs": {"content": logprobs_content}}]}
        mock_resp = _mock_response(200, json_data)
        n_prompts = len(QUALITY_GATE_PROMPTS)
        mock_ctx = _make_mock_ctx()
        mock_ctx.http.post.return_value = mock_resp
        from tps_pro.measurement import measure_token_uncertainty

        result = measure_token_uncertainty(mock_ctx)
        assert result is not None
        assert result["total_tokens"] == 1 * n_prompts

    @pytest.mark.unit
    def test_tail_avg_calculation(self):
        logprobs_content = [{"logprob": -0.1 * (i + 1)} for i in range(10)]
        json_data = {"choices": [{"logprobs": {"content": logprobs_content}}]}
        mock_resp = _mock_response(200, json_data)
        mock_ctx = _make_mock_ctx()
        mock_ctx.http.post.return_value = mock_resp
        from tps_pro.measurement import measure_token_uncertainty

        result = measure_token_uncertainty(mock_ctx)
        assert result is not None
        assert result["tail_avg"] < 0

    @pytest.mark.unit
    def test_logprob_none_filtered(self):
        logprobs_content = [{"logprob": None}, {"logprob": -0.5}]
        json_data = {"choices": [{"logprobs": {"content": logprobs_content}}]}
        mock_resp = _mock_response(200, json_data)
        mock_ctx = _make_mock_ctx()
        mock_ctx.http.post.return_value = mock_resp
        from tps_pro.measurement import measure_token_uncertainty

        result = measure_token_uncertainty(mock_ctx)
        assert result is not None
        assert result["total_tokens"] == 2

    @pytest.mark.unit
    def test_empty_content_logprobs(self):
        json_data = {"choices": [{"logprobs": {"content": []}}]}
        mock_resp = _mock_response(200, json_data)
        mock_ctx = _make_mock_ctx()
        mock_ctx.http.post.return_value = mock_resp
        from tps_pro.measurement import measure_token_uncertainty

        result = measure_token_uncertainty(mock_ctx)
        assert result is None

    @pytest.mark.unit
    def test_mixed_success_and_failure(self):
        logprobs_content = [{"logprob": -0.3}, {"logprob": -1.5}]
        json_data = {"choices": [{"logprobs": {"content": logprobs_content}}]}
        import requests

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_response(200, json_data)
            raise requests.RequestException("fail")

        mock_ctx = _make_mock_ctx()
        mock_ctx.http.post.side_effect = side_effect
        from tps_pro.measurement import measure_token_uncertainty

        result = measure_token_uncertainty(mock_ctx)
        assert result is not None
        assert result["total_tokens"] == 2
