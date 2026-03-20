"""Unit tests for evals/mcq.py — answer extraction, scoring, and quality measurement.

Consolidated from test_evals_mcq.py and test_evals_mcq_unit.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tps_pro.evals.mcq import (
    _eval_single_task,
    _extract_answer_letter,
    _extract_answer_logprob,
    _measure_quality_sequential,
    _score_quality_results,
    measure_quality,
)
from tps_pro.result_types import QualityTaskResult

# ===================================================================
# _extract_answer_letter
# ===================================================================


@pytest.mark.unit
class TestExtractAnswerLetter:
    def test_simple_letter(self):
        assert _extract_answer_letter("A") == "A"

    def test_parenthesized(self):
        assert _extract_answer_letter("(B)") == "B"

    def test_answer_is_pattern(self):
        assert _extract_answer_letter("The answer is C") == "C"

    def test_choice_colon_pattern(self):
        assert _extract_answer_letter("choice: D") == "D"

    def test_trailing_letter(self):
        assert _extract_answer_letter("I think the best option is (A)") == "A"

    def test_no_match_returns_none(self):
        assert _extract_answer_letter("I don't know") is None

    def test_empty_string(self):
        assert _extract_answer_letter("") is None

    def test_lowercase_normalized(self):
        assert _extract_answer_letter("answer is b") == "B"


# ===================================================================
# _extract_answer_logprob
# ===================================================================


@pytest.mark.unit
class TestExtractAnswerLogprob:
    def test_finds_logprob_for_correct_letter(self):
        data = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"token": "A", "logprob": -0.5, "top_logprobs": []},
                        ]
                    }
                }
            ]
        }
        result = _extract_answer_logprob(data, "A")
        assert result == -0.5

    def test_finds_in_top_logprobs(self):
        data = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {
                                "token": "X",
                                "logprob": -1.0,
                                "top_logprobs": [
                                    {"token": "B", "logprob": -0.3},
                                ],
                            },
                        ]
                    }
                }
            ]
        }
        result = _extract_answer_logprob(data, "B")
        assert result == -0.3

    def test_returns_none_when_not_found(self):
        data = {"choices": [{"logprobs": {"content": []}}]}
        assert _extract_answer_logprob(data, "C") is None

    def test_handles_missing_logprobs(self):
        data = {"choices": [{}]}
        assert _extract_answer_logprob(data, "A") is None

    def test_handles_empty_choices(self):
        """Empty choices list triggers IndexError (expected — caller handles it)."""
        data = {"choices": []}
        with pytest.raises(IndexError):
            _extract_answer_logprob(data, "A")


# ===================================================================
# _score_quality_results
# ===================================================================


@pytest.mark.unit
class TestScoreQualityResults:
    def test_empty_results_returns_zero(self):
        result = _score_quality_results([])
        assert result.score == 0.0

    def test_all_correct_high_score(self):
        tasks = [
            QualityTaskResult(correct=True, logprob=-0.1, ttft_ms=50.0),
            QualityTaskResult(correct=True, logprob=-0.1, ttft_ms=50.0),
        ]
        result = _score_quality_results(tasks)
        assert result.score > 50.0  # should be a high score
        assert len(result.task_results) == 2

    def test_all_wrong_low_score(self):
        tasks = [
            QualityTaskResult(correct=False, logprob=-5.0, ttft_ms=5000.0),
            QualityTaskResult(correct=False, logprob=-5.0, ttft_ms=5000.0),
        ]
        result = _score_quality_results(tasks)
        assert result.score < 50.0

    def test_no_logprobs_uses_correctness(self):
        tasks = [
            QualityTaskResult(correct=True, logprob=None, ttft_ms=100.0),
        ]
        result = _score_quality_results(tasks)
        assert result.score > 0.0

    def test_no_ttft_uses_neutral_efficiency(self):
        tasks = [
            QualityTaskResult(correct=True, logprob=-0.5, ttft_ms=None),
        ]
        result = _score_quality_results(tasks)
        assert result.score > 0.0


# ===================================================================
# measure_quality — mocked HTTP
# ===================================================================


@pytest.mark.unit
class TestEvalSingleTask:
    def test_correct_answer_returns_correct_result(self):
        """When the model returns the correct letter, result.correct is True."""
        from types import SimpleNamespace

        ctx = SimpleNamespace(
            server_url="http://localhost:8090",
            http=MagicMock(),
            debug=False,
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {"content": "The answer is A"},
                    "logprobs": {
                        "content": [{"token": "A", "logprob": -0.1, "top_logprobs": []}]
                    },
                }
            ],
        }
        ctx.http.post.return_value = mock_resp

        result = _eval_single_task(ctx, "What is 1+1?", "A", "math", 128, {})
        assert result.correct is True
        assert result.logprob == -0.1
        assert result.ttft_ms is not None

    def test_wrong_answer_returns_incorrect(self):
        """When the model returns wrong letter, result.correct is False."""
        from types import SimpleNamespace

        ctx = SimpleNamespace(
            server_url="http://localhost:8090",
            http=MagicMock(),
            debug=False,
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {"message": {"content": "I think B"}, "logprobs": {"content": []}}
            ],
        }
        ctx.http.post.return_value = mock_resp

        result = _eval_single_task(ctx, "What is 1+1?", "A", "math", 128, {})
        assert result.correct is False

    def test_http_error_returns_failed_result(self):
        """When HTTP fails, returns a failure result."""
        from types import SimpleNamespace

        import requests

        ctx = SimpleNamespace(
            server_url="http://localhost:8090",
            http=MagicMock(),
            debug=False,
        )
        ctx.http.post.side_effect = requests.RequestException("timeout")

        result = _eval_single_task(ctx, "What?", "A", "test", 128, {})
        assert result.correct is False
        assert result.logprob is None


@pytest.mark.unit
class TestMeasureQualitySequential:
    def test_sequential_scores_all_tasks(self):
        """_measure_quality_sequential runs all tasks and returns a QualityResult."""
        from types import SimpleNamespace

        ctx = SimpleNamespace(
            server_url="http://localhost:8090",
            http=MagicMock(),
            debug=False,
        )
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {"content": "A"},
                    "logprobs": {
                        "content": [{"token": "A", "logprob": -0.2, "top_logprobs": []}]
                    },
                }
            ],
        }
        ctx.http.post.return_value = mock_resp

        tasks = [("Q1", "A", "cat1"), ("Q2", "A", "cat2")]
        result = _measure_quality_sequential(ctx, tasks, max_tokens=128, oai_params={})
        assert result.score > 0.0
        assert len(result.task_results) == 2

    def test_measure_quality_is_callable(self):
        assert callable(measure_quality)
