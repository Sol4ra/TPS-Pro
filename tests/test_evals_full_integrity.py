"""Tests for evals submodules: integrity, kl_divergence, mcq flow.

Covers TestPhaseReasoningEval, TestPhaseIntegrityEval,
TestMeasureKlDivergence, and TestMeasureQualityFlow.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from _evals_full_helpers import (
    _chat_completion_response,
    _logprob_token,
    _mock_response,
    _patch_ctx,
)


# ===================================================================
# integrity.py — phase_reasoning_eval, phase_integrity_eval
# ===================================================================


class TestPhaseReasoningEval:
    """Tests for phase_reasoning_eval."""

    @pytest.mark.unit
    def test_all_correct(self, _patch_ctx):
        """All reasoning tasks answered correctly -> 100%."""
        from tps_pro.evals.integrity import phase_reasoning_eval

        def _side_effect(*args, **kwargs):
            prompt = kwargs.get("json", {}).get("messages", [{}])[0].get("content", "")
            answer_map = {
                "derivative": "x^2(3ln(x)+1)",
                "train travels": "70",
                "roses": "no, we cannot conclude",
                "17 * 23": "391",
                "strawberry": "3",
            }
            for key, ans in answer_map.items():
                if key.lower() in prompt.lower():
                    return _chat_completion_response(ans)
            return _chat_completion_response("unknown")

        _patch_ctx.http.post.side_effect = _side_effect
        score = phase_reasoning_eval(_patch_ctx, n_tasks=5)
        assert score == pytest.approx(100.0)

    @pytest.mark.unit
    def test_all_wrong(self, _patch_ctx):
        """All wrong answers -> 0%."""
        from tps_pro.evals.integrity import phase_reasoning_eval

        _patch_ctx.http.post.return_value = _chat_completion_response(
            "completely wrong answer xyz"
        )
        score = phase_reasoning_eval(_patch_ctx, n_tasks=5)
        assert score == pytest.approx(0.0)

    @pytest.mark.unit
    def test_server_error_graceful(self, _patch_ctx):
        """Server returning 500 should not crash, score 0."""
        from tps_pro.evals.integrity import phase_reasoning_eval

        _patch_ctx.http.post.return_value = _mock_response(500)
        score = phase_reasoning_eval(_patch_ctx, n_tasks=5)
        assert score == pytest.approx(0.0)

    @pytest.mark.unit
    def test_request_exception_graceful(self, _patch_ctx):
        """Network error should be caught, score 0."""
        import requests as req

        from tps_pro.evals.integrity import phase_reasoning_eval

        _patch_ctx.http.post.side_effect = req.ConnectionError("refused")
        score = phase_reasoning_eval(_patch_ctx, n_tasks=3)
        assert score == pytest.approx(0.0)

    @pytest.mark.unit
    def test_partial_correct(self, _patch_ctx):
        """Some correct, some wrong -> partial score."""
        from tps_pro.evals.integrity import phase_reasoning_eval

        responses = [
            _chat_completion_response("x^2(3ln(x)+1)"),  # correct
            _chat_completion_response("wrong"),  # wrong
        ]
        _patch_ctx.http.post.side_effect = responses
        score = phase_reasoning_eval(_patch_ctx, n_tasks=2)
        assert score == pytest.approx(50.0)

    @pytest.mark.unit
    def test_n_tasks_clamped(self, _patch_ctx):
        """n_tasks > len(tasks) should be clamped."""
        from tps_pro.evals.integrity import phase_reasoning_eval

        _patch_ctx.http.post.return_value = _chat_completion_response("wrong")
        score = phase_reasoning_eval(_patch_ctx, n_tasks=100)
        assert score == pytest.approx(0.0)
        # Only 5 tasks exist, so only 5 HTTP calls
        assert _patch_ctx.http.post.call_count == 5


class TestPhaseIntegrityEval:
    """Tests for phase_integrity_eval."""

    @pytest.mark.unit
    def test_all_pass(self, _patch_ctx):
        """All integrity checks pass -> 100%."""
        from tps_pro.evals.integrity import phase_integrity_eval

        responses = [
            _chat_completion_response(
                "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"
            ),
            _chat_completion_response("1024"),
            _chat_completion_response("red\nblue\ngreen\nyellow\npurple"),
            _chat_completion_response("hello"),
            _chat_completion_response("The capital of France is Paris."),
        ]
        _patch_ctx.http.post.side_effect = responses
        score = phase_integrity_eval(_patch_ctx, n_tasks=5)
        assert score == pytest.approx(100.0)

    @pytest.mark.unit
    def test_gibberish_response_fails_checks(self, _patch_ctx):
        """Gibberish output fails integrity checks -> 0%."""
        from tps_pro.evals.integrity import phase_integrity_eval

        _patch_ctx.http.post.return_value = _chat_completion_response(
            "asdfghjkl qwerty zxcvbn repetitive gibberish garbage"
        )
        score = phase_integrity_eval(_patch_ctx, n_tasks=5)
        assert score == pytest.approx(0.0)

    @pytest.mark.unit
    def test_empty_response_fails(self, _patch_ctx):
        """Empty model output fails all checks."""
        from tps_pro.evals.integrity import phase_integrity_eval

        _patch_ctx.http.post.return_value = _chat_completion_response("")
        score = phase_integrity_eval(_patch_ctx, n_tasks=5)
        assert score == pytest.approx(0.0)

    @pytest.mark.unit
    def test_server_error(self, _patch_ctx):
        """Server error on all requests -> 0%."""
        from tps_pro.evals.integrity import phase_integrity_eval

        _patch_ctx.http.post.return_value = _mock_response(503)
        score = phase_integrity_eval(_patch_ctx, n_tasks=5)
        assert score == pytest.approx(0.0)


# ===================================================================
# kl_divergence.py — measure_kl_divergence (flow test)
# ===================================================================


class TestMeasureKlDivergence:
    """Tests for the full measure_kl_divergence flow."""

    def _make_logprob_response(self, token_dists: list[dict[str, float]]):
        """Build a chat completion response with top_logprobs data."""
        content_logprobs = []
        for dist in token_dists:
            top_lps = [{"token": tok, "logprob": lp} for tok, lp in dist.items()]
            content_logprobs.append({"top_logprobs": top_lps})
        return _mock_response(
            200,
            {
                "choices": [
                    {
                        "logprobs": {"content": content_logprobs},
                        "message": {"content": "test"},
                    }
                ]
            },
        )

    @pytest.mark.unit
    def test_baseline_collection(self, _patch_ctx):
        """First call (no baseline) collects and returns distributions."""
        from tps_pro.evals.kl_divergence import measure_kl_divergence

        dist = {"hello": -0.5, "world": -2.0}
        _patch_ctx.http.post.return_value = self._make_logprob_response([dist])

        dists, kl_div = measure_kl_divergence(_patch_ctx, baseline_cache=None)
        assert dists is not None
        assert len(dists) > 0
        assert kl_div is None  # no comparison on first call

    @pytest.mark.unit
    def test_comparison_identical(self, _patch_ctx):
        """Comparing identical distributions gives KL near 0."""
        from tps_pro.evals.kl_divergence import measure_kl_divergence

        dist = {"the": -0.1, "a": -3.0, "an": -4.0}
        _patch_ctx.http.post.return_value = self._make_logprob_response([dist, dist])

        baseline = [dist, dist]
        dists, kl_div = measure_kl_divergence(_patch_ctx, baseline_cache=baseline)
        assert dists is not None
        assert kl_div is not None
        assert kl_div == pytest.approx(0.0, abs=0.01)

    @pytest.mark.unit
    def test_comparison_divergent(self, _patch_ctx):
        """Divergent distributions give positive KL."""
        from tps_pro.evals.kl_divergence import measure_kl_divergence

        baseline_dist = {"the": -0.1, "a": -5.0}
        trial_dist = {"the": -5.0, "a": -0.1}
        _patch_ctx.http.post.return_value = self._make_logprob_response([trial_dist])

        dists, kl_div = measure_kl_divergence(
            _patch_ctx, baseline_cache=[baseline_dist]
        )
        assert kl_div is not None
        assert kl_div > 0.0

    @pytest.mark.unit
    def test_server_error_returns_none(self, _patch_ctx):
        """Server failure -> (None, None)."""
        from tps_pro.evals.kl_divergence import measure_kl_divergence

        _patch_ctx.http.post.return_value = _mock_response(500)
        dists, kl_div = measure_kl_divergence(_patch_ctx, baseline_cache=None)
        assert dists is None
        assert kl_div is None

    @pytest.mark.unit
    def test_missing_logprobs_in_response(self, _patch_ctx):
        """Response without logprobs data -> None distributions."""
        from tps_pro.evals.kl_divergence import measure_kl_divergence

        _patch_ctx.http.post.return_value = _mock_response(
            200, {"choices": [{"message": {"content": "hello"}}]}
        )
        dists, kl_div = measure_kl_divergence(_patch_ctx, baseline_cache=None)
        assert dists is None

    @pytest.mark.unit
    def test_server_error_during_comparison(self, _patch_ctx):
        """Server failure during comparison returns (None, None)."""
        from tps_pro.evals.kl_divergence import measure_kl_divergence

        _patch_ctx.http.post.return_value = _mock_response(500)
        baseline = [{"a": -0.5}]
        dists, kl_div = measure_kl_divergence(_patch_ctx, baseline_cache=baseline)
        assert dists is None
        assert kl_div is None


# ===================================================================
# mcq.py — measure_quality (full flow)
# ===================================================================


class TestMeasureQualityFlow:
    """Tests for the full measure_quality sequential flow."""

    @pytest.fixture(autouse=True)
    def _force_sequential(self):
        """Force sequential path by disabling aiohttp."""
        with patch("tps_pro.evals.mcq.HAS_AIOHTTP", False):
            yield

    def _make_mc_response(self, answer_letter: str, logprob: float = -0.1):
        """Build a MC answer response with logprobs."""
        logprobs_content = [
            _logprob_token(
                answer_letter, logprob, [{"token": answer_letter, "logprob": logprob}]
            )
        ]
        return _chat_completion_response(
            f"The answer is ({answer_letter})",
            logprobs_content=logprobs_content,
        )

    @pytest.mark.unit
    def test_all_correct_high_score(self, _patch_ctx):
        """All correct answers with high confidence -> high score."""
        from tps_pro.constants import QUALITY_TASKS
        from tps_pro.evals.mcq import measure_quality

        correct_letters = [task[1] for task in QUALITY_TASKS]
        responses = [
            self._make_mc_response(letter, -0.05) for letter in correct_letters
        ]
        _patch_ctx.http.post.side_effect = responses

        result = measure_quality(_patch_ctx, sampling_params={"temperature": 0.0})
        assert result.score > 70.0

    @pytest.mark.unit
    def test_all_wrong_low_score(self, _patch_ctx):
        """All wrong answers -> low score."""
        from tps_pro.constants import QUALITY_TASKS
        from tps_pro.evals.mcq import measure_quality

        wrong = {"A": "B", "B": "C", "C": "D", "D": "A"}
        responses = [
            self._make_mc_response(wrong.get(task[1], "A"), -5.0)
            for task in QUALITY_TASKS
        ]
        _patch_ctx.http.post.side_effect = responses

        result = measure_quality(_patch_ctx, sampling_params={"temperature": 0.0})
        assert result.score < 30.0

    @pytest.mark.unit
    def test_server_failure_mid_evaluation(self, _patch_ctx):
        """Mix of successes and failures -> partial score."""
        import requests as req

        from tps_pro.evals.mcq import measure_quality

        side_effects = [
            self._make_mc_response("C", -0.1),
            req.ConnectionError("connection lost"),
            self._make_mc_response("C", -0.1),
        ]
        from tps_pro.constants import QUALITY_TASKS

        while len(side_effects) < len(QUALITY_TASKS):
            side_effects.append(req.ConnectionError("timeout"))
        _patch_ctx.http.post.side_effect = side_effects

        result = measure_quality(_patch_ctx, sampling_params={"temperature": 0.0})
        assert result.score > 0.0

    @pytest.mark.unit
    def test_n_predict_mapped_to_max_tokens(self, _patch_ctx):
        """sampling_params with n_predict should be mapped to max_tokens."""
        from tps_pro.evals.mcq import measure_quality

        _patch_ctx.http.post.return_value = _chat_completion_response("A")
        measure_quality(
            _patch_ctx, sampling_params={"n_predict": 512, "temperature": 0.0}
        )

        call_kwargs = _patch_ctx.http.post.call_args
        payload = (
            call_kwargs[1]["json"] if "json" in call_kwargs[1] else call_kwargs[0][1]
        )
        assert payload.get("max_tokens") == 512

    @pytest.mark.unit
    def test_early_exit_on_hopeless_score(self, _patch_ctx):
        """With target_to_beat, should short-circuit if score is hopeless."""
        from tps_pro.constants import QUALITY_TASKS
        from tps_pro.evals.mcq import measure_quality

        _patch_ctx.debug = True
        _patch_ctx.http.post.return_value = _chat_completion_response("wrong answer Z")

        result = measure_quality(
            _patch_ctx,
            sampling_params={"temperature": 0.0},
            target_to_beat=95.0,
        )
        assert _patch_ctx.http.post.call_count <= len(QUALITY_TASKS)
        assert result.score < 95.0

    @pytest.mark.unit
    def test_custom_tasks(self, _patch_ctx):
        """Passing custom tasks list works."""
        from tps_pro.evals.mcq import measure_quality

        custom_tasks = [
            ("What is 2+2?\n(A) 3\n(B) 4\n(C) 5\n(D) 6\nAnswer:", "B", "math"),
        ]
        _patch_ctx.http.post.return_value = self._make_mc_response("B", -0.01)

        result = measure_quality(
            _patch_ctx,
            sampling_params={"temperature": 0.0},
            tasks=custom_tasks,
        )
        assert result.score > 50.0
        assert _patch_ctx.http.post.call_count == 1
