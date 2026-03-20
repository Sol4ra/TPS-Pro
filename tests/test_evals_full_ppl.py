"""Tests for evals submodules: perplexity, quality_gate, mcq helpers, kl_divergence helpers.

Covers TestMeasureTruePerplexity, TestMeasureQualityGate, TestExtractAnswerLogprob,
TestEvalSingleTask, and TestCollectLogprobDistribution.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from _evals_full_helpers import (
    _chat_completion_response,
    _logprob_token,
    _mock_response,
    _patch_ctx,
)


# ===================================================================
# perplexity.py — measure_true_perplexity
# ===================================================================


class TestMeasureTruePerplexity:
    """Tests for measure_true_perplexity."""

    @pytest.mark.unit
    def test_normal_perplexity_from_chat(self, _patch_ctx):
        """Normal PPL calculation from chat completions logprobs."""
        from tps_pro.evals.perplexity import measure_true_perplexity

        logprobs_content = [_logprob_token(f"t{i}", -2.0) for i in range(50)]
        _patch_ctx.http.post.return_value = _mock_response(
            200,
            {
                "choices": [
                    {
                        "message": {"content": "generated text"},
                        "logprobs": {"content": logprobs_content},
                    }
                ]
            },
        )

        ppl = measure_true_perplexity(_patch_ctx, "A" * 500)
        assert ppl == pytest.approx(math.exp(2.0), rel=0.01)

    @pytest.mark.unit
    def test_fallback_to_completions_endpoint(self, _patch_ctx):
        """When chat gives no logprobs, falls back to /v1/completions."""
        from tps_pro.evals.perplexity import measure_true_perplexity

        chat_response = _mock_response(
            200,
            {
                "choices": [
                    {
                        "message": {"content": "text"},
                        "logprobs": {"content": []},
                    }
                ]
            },
        )
        completions_response = _mock_response(
            200,
            {
                "choices": [
                    {
                        "logprobs": {"token_logprobs": [-1.5] * 50},
                    }
                ]
            },
        )
        _patch_ctx.http.post.side_effect = [chat_response, completions_response]

        ppl = measure_true_perplexity(_patch_ctx, "A" * 500)
        assert ppl == pytest.approx(math.exp(1.5), rel=0.01)

    @pytest.mark.unit
    def test_server_error_returns_inf(self, _patch_ctx):
        """Server error -> inf."""
        from tps_pro.evals.perplexity import measure_true_perplexity

        _patch_ctx.http.post.return_value = _mock_response(500)
        ppl = measure_true_perplexity(_patch_ctx, "A" * 500)
        assert ppl == float("inf")

    @pytest.mark.unit
    def test_connection_error_returns_inf(self, _patch_ctx):
        """Network error -> inf."""
        import requests as req

        from tps_pro.evals.perplexity import measure_true_perplexity

        _patch_ctx.http.post.side_effect = req.ConnectionError("refused")
        ppl = measure_true_perplexity(_patch_ctx, "A" * 500)
        assert ppl == float("inf")

    @pytest.mark.unit
    def test_too_few_logprobs_returns_inf(self, _patch_ctx):
        """Fewer than 10 logprobs -> inf."""
        from tps_pro.evals.perplexity import measure_true_perplexity

        logprobs_content = [_logprob_token(f"t{i}", -1.0) for i in range(5)]
        _patch_ctx.http.post.return_value = _mock_response(
            200,
            {
                "choices": [
                    {
                        "message": {"content": "text"},
                        "logprobs": {"content": logprobs_content},
                    }
                ]
            },
        )

        ppl = measure_true_perplexity(_patch_ctx, "A" * 500)
        assert ppl == float("inf")

    @pytest.mark.unit
    def test_uses_reference_text_when_none(self, _patch_ctx):
        """When text_chunk is None, loads reference text."""
        from tps_pro.evals.perplexity import measure_true_perplexity

        logprobs_content = [_logprob_token(f"t{i}", -2.0) for i in range(50)]
        _patch_ctx.http.post.return_value = _mock_response(
            200,
            {
                "choices": [
                    {
                        "message": {"content": "text"},
                        "logprobs": {"content": logprobs_content},
                    }
                ]
            },
        )

        ref_text = "A" * 1000
        with patch(
            "tps_pro.evals.perplexity.get_ppl_reference_text",
            return_value=ref_text,
        ):
            ppl = measure_true_perplexity(_patch_ctx, None)
        assert ppl != float("inf")
        assert ppl > 0

    @pytest.mark.unit
    def test_low_ppl_good_model(self, _patch_ctx):
        """Very confident model -> low PPL."""
        from tps_pro.evals.perplexity import measure_true_perplexity

        logprobs_content = [_logprob_token(f"t{i}", -0.1) for i in range(50)]
        _patch_ctx.http.post.return_value = _mock_response(
            200,
            {
                "choices": [
                    {
                        "message": {"content": "text"},
                        "logprobs": {"content": logprobs_content},
                    }
                ]
            },
        )

        ppl = measure_true_perplexity(_patch_ctx, "A" * 500)
        assert ppl < 2.0

    @pytest.mark.unit
    def test_high_ppl_uncertain_model(self, _patch_ctx):
        """Very uncertain model -> high PPL."""
        from tps_pro.evals.perplexity import measure_true_perplexity

        logprobs_content = [_logprob_token(f"t{i}", -5.0) for i in range(50)]
        _patch_ctx.http.post.return_value = _mock_response(
            200,
            {
                "choices": [
                    {
                        "message": {"content": "text"},
                        "logprobs": {"content": logprobs_content},
                    }
                ]
            },
        )

        ppl = measure_true_perplexity(_patch_ctx, "A" * 500)
        assert ppl > 100.0

    @pytest.mark.unit
    def test_both_endpoints_fail(self, _patch_ctx):
        """Both chat and completions return non-200 -> inf."""
        from tps_pro.evals.perplexity import measure_true_perplexity

        _patch_ctx.http.post.return_value = _mock_response(503)
        ppl = measure_true_perplexity(_patch_ctx, "A" * 500)
        assert ppl == float("inf")


# ===================================================================
# quality_gate.py — measure_quality_gate
# ===================================================================


class TestMeasureQualityGate:
    """Tests for measure_quality_gate."""

    @pytest.mark.unit
    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_baseline_run_stores_metrics(self, mock_mtu, _patch_ctx):
        """Baseline run stores metrics and returns 1.0."""
        from tps_pro.evals.quality_gate import measure_quality_gate

        mock_mtu.return_value = {
            "uncertain_count": 10,
            "tail_avg": -1.5,
            "total_tokens": 500,
        }

        factor = measure_quality_gate(_patch_ctx, is_baseline=True)
        assert factor == 1.0
        assert _patch_ctx.quality_baseline is not None

    @pytest.mark.unit
    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_measurement_failure_on_baseline(self, mock_mtu, _patch_ctx):
        """Measurement failure on baseline -> 1.0 (no penalty)."""
        from tps_pro.evals.quality_gate import measure_quality_gate

        mock_mtu.return_value = None
        factor = measure_quality_gate(_patch_ctx, is_baseline=True)
        assert factor == 1.0

    @pytest.mark.unit
    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_measurement_failure_on_trial(self, mock_mtu, _patch_ctx):
        """Measurement failure on trial -> cliff penalty."""
        from tps_pro.constants import QUALITY_GATE_CLIFF_PENALTY
        from tps_pro.evals.quality_gate import measure_quality_gate

        _patch_ctx.quality_baseline = {
            "uncertain_count": 10,
            "tail_avg": -1.5,
            "total_tokens": 500,
        }
        mock_mtu.return_value = None
        factor = measure_quality_gate(_patch_ctx, is_baseline=False)
        assert factor == pytest.approx(QUALITY_GATE_CLIFF_PENALTY)

    @pytest.mark.unit
    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_no_degradation(self, mock_mtu, _patch_ctx):
        """Same metrics as baseline -> factor 1.0."""
        from tps_pro.evals.quality_gate import measure_quality_gate

        baseline = {"uncertain_count": 10, "tail_avg": -1.5, "total_tokens": 500}
        _patch_ctx.quality_baseline = baseline
        mock_mtu.return_value = {
            "uncertain_count": 10,
            "tail_avg": -1.5,
            "total_tokens": 500,
        }

        factor = measure_quality_gate(_patch_ctx, is_baseline=False)
        assert factor == pytest.approx(1.0, abs=0.05)

    @pytest.mark.unit
    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_mild_degradation_soft_penalty(self, mock_mtu, _patch_ctx):
        """Small degradation -> factor between 0.85 and 1.0."""
        from tps_pro.evals.quality_gate import measure_quality_gate

        baseline = {"uncertain_count": 100, "tail_avg": -2.0, "total_tokens": 1000}
        _patch_ctx.quality_baseline = baseline
        mock_mtu.return_value = {
            "uncertain_count": 101,
            "tail_avg": -2.0,
            "total_tokens": 1000,
        }

        factor = measure_quality_gate(_patch_ctx, is_baseline=False)
        assert 0.85 <= factor <= 1.0

    @pytest.mark.unit
    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_severe_degradation_cliff_penalty(self, mock_mtu, _patch_ctx):
        """Severe degradation -> cliff penalty (0.1)."""
        from tps_pro.constants import QUALITY_GATE_CLIFF_PENALTY
        from tps_pro.evals.quality_gate import measure_quality_gate

        baseline = {"uncertain_count": 100, "tail_avg": -2.0, "total_tokens": 1000}
        _patch_ctx.quality_baseline = baseline
        mock_mtu.return_value = {
            "uncertain_count": 150,
            "tail_avg": -2.0,
            "total_tokens": 1000,
        }

        factor = measure_quality_gate(_patch_ctx, is_baseline=False)
        assert factor == pytest.approx(QUALITY_GATE_CLIFF_PENALTY)

    @pytest.mark.unit
    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_improvement_over_baseline(self, mock_mtu, _patch_ctx):
        """Better than baseline -> factor 1.0."""
        from tps_pro.evals.quality_gate import measure_quality_gate

        baseline = {"uncertain_count": 100, "tail_avg": -2.0, "total_tokens": 1000}
        _patch_ctx.quality_baseline = baseline
        mock_mtu.return_value = {
            "uncertain_count": 50,
            "tail_avg": -1.5,
            "total_tokens": 1000,
        }

        factor = measure_quality_gate(_patch_ctx, is_baseline=False)
        assert factor == pytest.approx(1.0)

    @pytest.mark.unit
    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_tail_degradation_signal(self, mock_mtu, _patch_ctx):
        """Tail avg degradation (worse logprobs) -> penalty."""
        from tps_pro.evals.quality_gate import measure_quality_gate

        baseline = {"uncertain_count": 10, "tail_avg": -2.0, "total_tokens": 1000}
        _patch_ctx.quality_baseline = baseline
        mock_mtu.return_value = {
            "uncertain_count": 10,
            "tail_avg": -4.0,
            "total_tokens": 1000,
        }

        factor = measure_quality_gate(_patch_ctx, is_baseline=False)
        assert factor < 0.85

    @pytest.mark.unit
    @patch("tps_pro.evals.quality_gate.measure_token_uncertainty")
    def test_baseline_floor_prevents_extreme_sensitivity(self, mock_mtu, _patch_ctx):
        """Low baseline uncertain count uses floor to prevent oversensitivity."""
        from tps_pro.evals.quality_gate import measure_quality_gate

        baseline = {"uncertain_count": 0, "tail_avg": -1.0, "total_tokens": 1000}
        _patch_ctx.quality_baseline = baseline
        mock_mtu.return_value = {
            "uncertain_count": 3,
            "tail_avg": -1.0,
            "total_tokens": 1000,
        }

        factor = measure_quality_gate(_patch_ctx, is_baseline=False)
        assert factor == pytest.approx(1.0, abs=0.05)


# ===================================================================
# mcq.py — _extract_answer_logprob
# ===================================================================


class TestExtractAnswerLogprob:
    """Tests for _extract_answer_logprob."""

    @pytest.mark.unit
    def test_finds_correct_letter_logprob(self):
        from tps_pro.evals.mcq import _extract_answer_logprob

        data = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": "B",
                                "logprob": -0.5,
                                "top_logprobs": [
                                    {"token": "B", "logprob": -0.5},
                                    {"token": "A", "logprob": -2.0},
                                ],
                            },
                        ]
                    }
                }
            ]
        }
        result = _extract_answer_logprob(data, "B")
        assert result == pytest.approx(-0.5)

    @pytest.mark.unit
    def test_finds_in_top_logprobs(self):
        from tps_pro.evals.mcq import _extract_answer_logprob

        data = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": "The",
                                "logprob": -0.1,
                                "top_logprobs": [
                                    {"token": "The", "logprob": -0.1},
                                    {"token": "A", "logprob": -1.5},
                                ],
                            },
                        ]
                    }
                }
            ]
        }
        result = _extract_answer_logprob(data, "A")
        assert result == pytest.approx(-1.5)

    @pytest.mark.unit
    def test_not_found_returns_none(self):
        from tps_pro.evals.mcq import _extract_answer_logprob

        data = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"token": "hello", "logprob": -0.1, "top_logprobs": []},
                        ]
                    }
                }
            ]
        }
        result = _extract_answer_logprob(data, "C")
        assert result is None

    @pytest.mark.unit
    def test_no_logprobs_returns_none(self):
        from tps_pro.evals.mcq import _extract_answer_logprob

        data = {"choices": [{"message": {"content": "A"}}]}
        result = _extract_answer_logprob(data, "A")
        assert result is None

    @pytest.mark.unit
    def test_parenthesized_token(self):
        """Token like '(A)' should match letter 'A'."""
        from tps_pro.evals.mcq import _extract_answer_logprob

        data = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"token": "(A)", "logprob": -0.3, "top_logprobs": []},
                        ]
                    }
                }
            ]
        }
        result = _extract_answer_logprob(data, "A")
        assert result == pytest.approx(-0.3)


# ===================================================================
# mcq.py — _eval_single_task
# ===================================================================


class TestEvalSingleTask:
    """Tests for _eval_single_task."""

    @pytest.mark.unit
    def test_correct_answer(self, _patch_ctx):
        from tps_pro.evals.mcq import _eval_single_task

        logprobs_content = [
            {
                "token": "B",
                "logprob": -0.2,
                "top_logprobs": [{"token": "B", "logprob": -0.2}],
            }
        ]
        _patch_ctx.http.post.return_value = _mock_response(
            200,
            {
                "choices": [
                    {
                        "message": {"content": "The answer is B"},
                        "logprobs": {"content": logprobs_content},
                    }
                ]
            },
        )

        result = _eval_single_task(_patch_ctx, "question", "B", "math", 1024, {})
        assert result["correct"] is True
        assert result["logprob"] is not None
        assert result["ttft_ms"] is not None
        assert result["category"] == "math"

    @pytest.mark.unit
    def test_wrong_answer(self, _patch_ctx):
        from tps_pro.evals.mcq import _eval_single_task

        _patch_ctx.http.post.return_value = _chat_completion_response("The answer is C")

        result = _eval_single_task(_patch_ctx, "question", "B", "math", 1024, {})
        assert result["correct"] is False

    @pytest.mark.unit
    def test_server_error(self, _patch_ctx):
        import requests as req

        from tps_pro.evals.mcq import _eval_single_task

        _patch_ctx.http.post.side_effect = req.ConnectionError("timeout")

        result = _eval_single_task(_patch_ctx, "question", "A", "code", 1024, {})
        assert result["correct"] is False
        assert result["logprob"] is None
        assert result["ttft_ms"] is None


# ===================================================================
# kl_divergence.py — _collect_logprob_distribution
# ===================================================================


class TestCollectLogprobDistribution:
    """Tests for _collect_logprob_distribution."""

    @pytest.mark.unit
    def test_collects_distributions(self, _patch_ctx):
        from tps_pro.evals.kl_divergence import (
            _collect_logprob_distribution,
        )

        content_logprobs = [
            {
                "top_logprobs": [
                    {"token": "a", "logprob": -0.5},
                    {"token": "b", "logprob": -1.0},
                ]
            },
            {"top_logprobs": [{"token": "c", "logprob": -0.3}]},
        ]
        _patch_ctx.http.post.return_value = _mock_response(
            200,
            {
                "choices": [
                    {
                        "logprobs": {"content": content_logprobs},
                        "message": {"content": "x"},
                    }
                ]
            },
        )

        dists = _collect_logprob_distribution(_patch_ctx, ["prompt1"])
        assert dists is not None
        assert len(dists) == 2
        assert dists[0] == {"a": -0.5, "b": -1.0}

    @pytest.mark.unit
    def test_all_prompts_fail_returns_none(self, _patch_ctx):
        from tps_pro.evals.kl_divergence import (
            _collect_logprob_distribution,
        )

        _patch_ctx.http.post.return_value = _mock_response(500)
        dists = _collect_logprob_distribution(_patch_ctx, ["p1", "p2"])
        assert dists is None

    @pytest.mark.unit
    def test_partial_failure_still_returns(self, _patch_ctx):
        from tps_pro.evals.kl_divergence import (
            _collect_logprob_distribution,
        )

        good_response = _mock_response(
            200,
            {
                "choices": [
                    {
                        "logprobs": {
                            "content": [
                                {"top_logprobs": [{"token": "x", "logprob": -1.0}]}
                            ]
                        },
                        "message": {"content": "y"},
                    }
                ]
            },
        )
        bad_response = _mock_response(500)
        _patch_ctx.http.post.side_effect = [good_response, bad_response]

        dists = _collect_logprob_distribution(_patch_ctx, ["p1", "p2"])
        assert dists is not None
        assert len(dists) == 1

    @pytest.mark.unit
    def test_missing_top_logprobs_skipped(self, _patch_ctx):
        from tps_pro.evals.kl_divergence import (
            _collect_logprob_distribution,
        )

        content_logprobs = [
            {"top_logprobs": []},  # empty top_logprobs
            {"top_logprobs": [{"token": "a", "logprob": -0.5}]},
        ]
        _patch_ctx.http.post.return_value = _mock_response(
            200,
            {
                "choices": [
                    {
                        "logprobs": {"content": content_logprobs},
                        "message": {"content": "x"},
                    }
                ]
            },
        )

        dists = _collect_logprob_distribution(_patch_ctx, ["prompt1"])
        assert dists is not None
        assert len(dists) == 1  # only the non-empty one
