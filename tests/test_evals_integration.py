"""Integration tests for evaluation modules.

Covers MCQ/Quality, NIAH, Perplexity, KL-Divergence, Integrity,
and Variable Tracking (KV sweep quality). All HTTP calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from tps_pro.evals.mcq import (
    _extract_answer_letter,
    _extract_answer_logprob,
    _score_quality_results,
    measure_quality,
)
from tps_pro.evals.niah import TokenizeCache, build_niah_prompt, niah_test
from tps_pro.evals.perplexity import measure_true_perplexity, ppl_quality_factor
from tps_pro.evals.kl_divergence import _compute_kl_divergence
from tps_pro.evals.integrity import phase_reasoning_eval, phase_integrity_eval
from tps_pro.phases.kv_sweep_measure import (
    _build_variable_tracking_prompt,
)
from tps_pro.result_types import QualityResult, QualityTaskResult

from _ctx_factory import make_ctx_from_defaults


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int = 200, json_data: dict | None = None):
    """Build a mock requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.text = ""
    return resp


def _chat_response(content: str, logprobs: list | None = None):
    """Build a minimal OpenAI chat-completions JSON body."""
    choice: dict = {"message": {"content": content}}
    if logprobs is not None:
        choice["logprobs"] = {"content": logprobs}
    return {"choices": [choice]}


def _make_ctx(**overrides):
    """Shorthand that ensures http mock is present."""
    return make_ctx_from_defaults(**overrides)


# ===================================================================
# 1. _extract_answer_letter extracts A/B/C/D from various formats
# ===================================================================


class TestExtractAnswerLetter:
    @pytest.mark.parametrize(
        "text, expected",
        [
            ("The answer is A", "A"),
            ("(B)", "B"),
            ("C.", "C"),
            ("choice: D", "D"),
            ("Answer: (A)", "A"),
            ("the answer is B", "B"),
            ("I think (C) is correct", "C"),
            ("D", "D"),
            ("B)", "B"),
            ("Some long explanation and then A", "A"),
        ],
    )
    def test_extracts_from_various_formats(self, text: str, expected: str):
        assert _extract_answer_letter(text) == expected

    def test_returns_none_for_garbage(self):
        assert _extract_answer_letter("no answer here") is None

    def test_returns_none_for_empty(self):
        assert _extract_answer_letter("") is None


# ===================================================================
# 2. _extract_answer_logprob picks highest probability answer
# ===================================================================


class TestExtractAnswerLogprob:
    def test_direct_token_match(self):
        data = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"token": "A", "logprob": -0.1, "top_logprobs": []},
                        ]
                    }
                }
            ]
        }
        assert _extract_answer_logprob(data, "A") == pytest.approx(-0.1)

    def test_parenthesized_token_match(self):
        data = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"token": "(B)", "logprob": -0.5, "top_logprobs": []},
                        ]
                    }
                }
            ]
        }
        assert _extract_answer_logprob(data, "B") == pytest.approx(-0.5)

    def test_finds_in_top_logprobs(self):
        data = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": "X",
                                "logprob": -2.0,
                                "top_logprobs": [
                                    {"token": "C", "logprob": -0.3},
                                ],
                            },
                        ]
                    }
                }
            ]
        }
        assert _extract_answer_logprob(data, "C") == pytest.approx(-0.3)

    def test_returns_none_when_missing(self):
        data = {"choices": [{"logprobs": {"content": []}}]}
        assert _extract_answer_logprob(data, "D") is None


# ===================================================================
# 3. _score_quality_results calculates correct percentage
# ===================================================================


class TestScoreQualityResults:
    def test_all_correct(self):
        tasks = [
            QualityTaskResult(correct=True, logprob=-0.1, ttft_ms=100.0, category="c"),
            QualityTaskResult(correct=True, logprob=-0.2, ttft_ms=120.0, category="c"),
        ]
        result = _score_quality_results(tasks)
        assert isinstance(result, QualityResult)
        assert result.score > 0

    def test_none_correct(self):
        tasks = [
            QualityTaskResult(correct=False, logprob=None, ttft_ms=None, category="c"),
            QualityTaskResult(correct=False, logprob=None, ttft_ms=None, category="c"),
        ]
        result = _score_quality_results(tasks)
        # Correctness = 0, confidence falls back to correctness = 0
        assert result.score < 50

    def test_empty_returns_zero(self):
        result = _score_quality_results([])
        assert result.score == 0.0

    def test_half_correct(self):
        tasks = [
            QualityTaskResult(correct=True, logprob=-0.1, ttft_ms=200.0, category="c"),
            QualityTaskResult(correct=False, logprob=None, ttft_ms=200.0, category="c"),
        ]
        result = _score_quality_results(tasks)
        assert 0 < result.score < 100


# ===================================================================
# 4. measure_quality returns QualityResult
# ===================================================================


class TestMeasureQuality:
    @patch("tps_pro.evals.mcq.HAS_AIOHTTP", False)
    def test_returns_quality_result(self):
        ctx = _make_ctx()
        chat_json = _chat_response(
            "The answer is A",
            logprobs=[{"token": "A", "logprob": -0.05, "top_logprobs": []}],
        )
        ctx.http.post.return_value = _mock_response(200, chat_json)

        tasks = [("Question?", "A", "math")]
        result = measure_quality(ctx, {}, tasks=tasks)

        assert isinstance(result, QualityResult)
        assert result.score > 0
        assert len(result.task_results) == 1

    # ===============================================================
    # 5. measure_quality handles server errors gracefully
    # ===============================================================

    @patch("tps_pro.evals.mcq.HAS_AIOHTTP", False)
    def test_server_error_returns_zero_score_task(self):
        ctx = _make_ctx()
        ctx.http.post.return_value = _mock_response(500, {})

        tasks = [("Question?", "A", "math")]
        result = measure_quality(ctx, {}, tasks=tasks)

        assert isinstance(result, QualityResult)
        assert len(result.task_results) == 1
        assert result.task_results[0].correct is False

    @patch("tps_pro.evals.mcq.HAS_AIOHTTP", False)
    def test_connection_error_returns_failed_task(self):
        ctx = _make_ctx()
        ctx.http.post.side_effect = requests.ConnectionError("refused")

        tasks = [("Question?", "B", "logic")]
        result = measure_quality(ctx, {}, tasks=tasks)

        assert isinstance(result, QualityResult)
        assert result.task_results[0].correct is False


# ===================================================================
# 6. build_niah_prompt fills context to target token count
# ===================================================================


class TestBuildNiahPrompt:
    def test_prompt_contains_needle(self):
        ctx = _make_ctx()
        # Mock tokenize endpoint to return proportional token list
        def _tokenize_side_effect(*args, **kwargs):
            content = kwargs.get("json", {}).get("content", args[1] if len(args) > 1 else "")
            if isinstance(content, dict):
                content = content.get("content", "")
            token_count = max(1, len(content) // 3)
            return _mock_response(200, {"tokens": list(range(token_count))})

        ctx.http.post.side_effect = _tokenize_side_effect

        prompt = build_niah_prompt(ctx, target_tokens=500, needle_fact="The sky is blue.")
        assert "The sky is blue." in prompt

    # ===============================================================
    # 7. build_niah_prompt injects needle at correct depth
    # ===============================================================

    def test_needle_depth_positioning(self):
        ctx = _make_ctx()

        def _tokenize_side_effect(*args, **kwargs):
            body = kwargs.get("json", {})
            content = body.get("content", "")
            return _mock_response(200, {"tokens": list(range(max(1, len(content) // 3)))})

        ctx.http.post.side_effect = _tokenize_side_effect

        # Needle at 10% depth should appear in the first part of the prompt
        prompt_early = build_niah_prompt(
            ctx, target_tokens=1000, needle_fact="NEEDLE_EARLY", needle_depth_pct=0.10
        )
        pos_early = prompt_early.find("NEEDLE_EARLY")
        assert pos_early >= 0
        assert pos_early < len(prompt_early) * 0.4  # should be in first 40%

        # Needle at 90% depth should appear near the end
        prompt_late = build_niah_prompt(
            ctx, target_tokens=1000, needle_fact="NEEDLE_LATE", needle_depth_pct=0.90
        )
        pos_late = prompt_late.find("NEEDLE_LATE")
        assert pos_late >= 0
        assert pos_late > len(prompt_late) * 0.5  # should be past halfway


# ===================================================================
# 8. niah_test passes when model finds the needle
# ===================================================================


class TestNiahTest:
    @patch("tps_pro.evals.niah.kill_server")
    @patch("tps_pro.evals.niah.wait_for_server", return_value="ok")
    @patch("tps_pro.evals.niah.start_server", return_value=MagicMock())
    def test_pass_when_model_finds_needle(self, mock_start, mock_wait, mock_kill):
        from tps_pro.constants import NIAH_NEEDLES

        ctx = _make_ctx()
        needle = NIAH_NEEDLES[0]

        # Mock tokenize for build_niah_prompt
        def _http_post(url, **kwargs):
            if "/tokenize" in url:
                content = kwargs.get("json", {}).get("content", "")
                return _mock_response(200, {"tokens": list(range(max(1, len(content) // 3)))})
            # Chat completion: return the expected answer
            return _mock_response(
                200,
                _chat_response(needle["expected"]),
            )

        ctx.http.post.side_effect = _http_post

        result = niah_test(
            ctx,
            kv_cache_type="f16",
            base_config={"n_gpu_layers": 99, "context": 4096},
            depths=[0.5],
            context_sizes=[1024],
        )
        assert result.pass_rate > 0
        assert any(r.passed for r in result.results)

    # ===============================================================
    # 9. niah_test fails when model returns wrong answer
    # ===============================================================

    @patch("tps_pro.evals.niah.kill_server")
    @patch("tps_pro.evals.niah.wait_for_server", return_value="ok")
    @patch("tps_pro.evals.niah.start_server", return_value=MagicMock())
    def test_fail_when_model_returns_wrong_answer(self, mock_start, mock_wait, mock_kill):
        ctx = _make_ctx()

        def _http_post(url, **kwargs):
            if "/tokenize" in url:
                content = kwargs.get("json", {}).get("content", "")
                return _mock_response(200, {"tokens": list(range(max(1, len(content) // 3)))})
            return _mock_response(200, _chat_response("I have no idea"))

        ctx.http.post.side_effect = _http_post

        result = niah_test(
            ctx,
            kv_cache_type="f16",
            base_config={"n_gpu_layers": 99, "context": 4096},
            depths=[0.5],
            context_sizes=[1024],
        )
        assert result.pass_rate == 0.0
        assert all(not r.passed for r in result.results)


# ===================================================================
# 10. TokenizeCache caches tokenization ratios
# ===================================================================


class TestTokenizeCache:
    def test_initial_state_is_none(self):
        cache = TokenizeCache()
        assert cache.get() is None

    def test_set_and_get(self):
        cache = TokenizeCache()
        cache.set(3.5)
        assert cache.get() == pytest.approx(3.5)

    def test_subsequent_calls_use_ratio(self):
        from tps_pro.evals.niah import tokenize_count

        ctx = _make_ctx()
        cache = TokenizeCache()

        # First call hits server
        ctx.http.post.return_value = _mock_response(
            200, {"tokens": list(range(100))}
        )
        count1 = tokenize_count(ctx, "x" * 300, cache)
        assert count1 == 100  # exact from server
        assert cache.get() is not None

        # Second call should NOT hit server (uses cached ratio)
        ctx.http.post.reset_mock()
        count2 = tokenize_count(ctx, "y" * 300, cache)
        ctx.http.post.assert_not_called()
        assert count2 > 0


# ===================================================================
# 11. measure_true_perplexity returns float (PPL)
# ===================================================================


class TestMeasureTruePerplexity:
    def test_returns_finite_ppl(self):
        ctx = _make_ctx()
        logprobs = [{"logprob": -2.0} for _ in range(20)]
        chat_json = {
            "choices": [
                {
                    "logprobs": {"content": logprobs},
                    "message": {"content": "some text"},
                }
            ]
        }
        ctx.http.post.return_value = _mock_response(200, chat_json)

        ppl = measure_true_perplexity(ctx, text_chunk="A" * 500)
        assert ppl != float("inf")
        assert ppl > 0

    # ===============================================================
    # 12. measure_true_perplexity falls back between endpoints
    # ===============================================================

    def test_falls_back_to_completions(self):
        ctx = _make_ctx()
        call_count = 0

        def _post_side_effect(url, **kwargs):
            nonlocal call_count
            call_count += 1
            if "/v1/chat/completions" in url:
                # Chat endpoint returns too few logprobs
                return _mock_response(200, {
                    "choices": [{"logprobs": {"content": []}, "message": {"content": ""}}]
                })
            if "/v1/completions" in url:
                # Completions endpoint succeeds
                token_logprobs = [-2.0] * 20
                return _mock_response(200, {
                    "choices": [{"logprobs": {"token_logprobs": token_logprobs}}]
                })
            return _mock_response(404, {})

        ctx.http.post.side_effect = _post_side_effect
        ppl = measure_true_perplexity(ctx, text_chunk="A" * 500)
        assert ppl != float("inf")
        assert call_count >= 2  # both endpoints called


# ===================================================================
# 13. PPL quality factor penalizes high perplexity
# ===================================================================


class TestPPLQualityFactor:
    def test_no_degradation(self):
        assert ppl_quality_factor(5.0, 5.0) == 1.0

    def test_improvement_is_perfect(self):
        assert ppl_quality_factor(5.0, 4.0) == 1.0

    def test_high_degradation_penalizes(self):
        # 50% degradation (above 30% fail threshold)
        factor = ppl_quality_factor(5.0, 7.5)
        assert factor <= 0.1

    def test_mild_degradation_gentle_penalty(self):
        # 5% degradation (within warn threshold)
        factor = ppl_quality_factor(10.0, 10.5)
        assert 0.85 < factor < 1.0

    def test_inf_trial_returns_low(self):
        factor = ppl_quality_factor(5.0, float("inf"))
        assert factor == 0.1


# ===================================================================
# 14. _compute_kl_divergence returns float for valid distributions
# ===================================================================


class TestComputeKLDivergence:
    def test_returns_float(self):
        baseline = [{"a": -0.1, "b": -2.0}]
        trial = [{"a": -0.5, "b": -1.5}]
        kl = _compute_kl_divergence(baseline, trial)
        assert isinstance(kl, float)
        assert kl >= 0

    def test_empty_distributions(self):
        kl = _compute_kl_divergence([], [])
        assert kl == 0.0

    # ===============================================================
    # 15. KL divergence of identical distributions is ~0
    # ===============================================================

    def test_identical_distributions_near_zero(self):
        dist = [{"a": -0.1, "b": -2.0, "c": -5.0}]
        kl = _compute_kl_divergence(dist, dist)
        assert kl == pytest.approx(0.0, abs=1e-8)

    def test_different_distributions_positive(self):
        baseline = [{"a": -0.1, "b": -5.0}]
        trial = [{"a": -3.0, "b": -0.1}]
        kl = _compute_kl_divergence(baseline, trial)
        assert kl > 0.01


# ===================================================================
# 16. phase_reasoning_eval returns score 0-100
# ===================================================================


class TestPhaseReasoningEval:
    def test_returns_score_in_range(self):
        ctx = _make_ctx()
        # All correct
        ctx.http.post.return_value = _mock_response(
            200, _chat_response("391")
        )
        score = phase_reasoning_eval(ctx, n_tasks=1)
        assert 0 <= score <= 100

    def test_all_wrong_returns_zero(self):
        ctx = _make_ctx()
        ctx.http.post.return_value = _mock_response(
            200, _chat_response("completely wrong answer xyz")
        )
        score = phase_reasoning_eval(ctx, n_tasks=5)
        assert score == 0.0


# ===================================================================
# 17. phase_integrity_eval handles HTTP failures
# ===================================================================


class TestPhaseIntegrityEval:
    def test_http_error_does_not_crash(self):
        ctx = _make_ctx()
        ctx.http.post.side_effect = requests.ConnectionError("refused")

        score = phase_integrity_eval(ctx, n_tasks=3)
        assert score == 0.0

    def test_500_status_does_not_crash(self):
        ctx = _make_ctx()
        ctx.http.post.return_value = _mock_response(500, {})

        score = phase_integrity_eval(ctx, n_tasks=3)
        assert score == 0.0


# ===================================================================
# 18. _build_variable_tracking_prompt includes confusable codes
# ===================================================================


class TestBuildVariableTrackingPrompt:
    def test_includes_confusable_codes(self):
        prompt, expected = _build_variable_tracking_prompt(
            ctx=None, target_tokens=2000, seed=42
        )
        # Should contain many ALPHA-77XX codes as filler
        assert "ALPHA-77" in prompt
        # Should contain the needle assignment
        assert "ALPHA-7749" in prompt
        assert expected == "ALPHA-7749"

    def test_includes_project_cipher_assignments(self):
        prompt, _ = _build_variable_tracking_prompt(
            ctx=None, target_tokens=2000, seed=42
        )
        assert "PROJECT_CIPHER" in prompt
        # Should include multiple reassignments
        assert "BETA-3182" in prompt
        assert "GAMMA-5501" in prompt

    # ===============================================================
    # 19. Answer extraction finds correct ALPHA code
    # ===============================================================

    def test_expected_answer_is_first_value(self):
        """The expected answer should be the FIRST assigned value (ALPHA-7749)."""
        _, expected = _build_variable_tracking_prompt(
            ctx=None, target_tokens=1000, seed=99
        )
        assert expected == "ALPHA-7749"

    def test_prompt_has_question_at_end(self):
        prompt, _ = _build_variable_tracking_prompt(
            ctx=None, target_tokens=1000, seed=42
        )
        # Question should be at the end
        assert prompt.rstrip().endswith("nothing else.")
