"""Tests for evals/integrity.py — reasoning and integrity evaluation phases.

Uses SimpleNamespace for mock ctx and MagicMock for HTTP sessions.
Consolidated from test_evals_integrity.py and test_evals_integrity2.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tps_pro.evals.integrity import (
    phase_integrity_eval,
    phase_reasoning_eval,
)


def _make_ctx(server_url="http://localhost:8080"):
    """Build a minimal ctx with a mocked http session."""
    from _ctx_factory import make_ctx_from_defaults

    return make_ctx_from_defaults(server_url=server_url)


def _ok_response(content: str):
    """Build a mock HTTP response with status 200 and given content."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}],
    }
    return resp


def _error_response(status_code: int = 500):
    """Build a mock HTTP response with a non-200 status."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {}
    return resp


# ===================================================================
# phase_reasoning_eval
# ===================================================================


@pytest.mark.unit
def test_reasoning_eval_all_correct():
    """All tasks return the accepted answer -> 100% score."""
    ctx = _make_ctx()
    # Answers matching the accept lists for each of the 5 tasks
    answers = [
        "x^2(3ln(x)+1)",  # task 0
        "The answer is 70 mph.",  # task 1
        "No, we cannot conclude.",  # task 2
        "391",  # task 3
        "There are 3 r's.",  # task 4
    ]
    ctx.http.post.side_effect = [_ok_response(a) for a in answers]

    score = phase_reasoning_eval(ctx, n_tasks=5)
    assert score == pytest.approx(100.0)


@pytest.mark.unit
def test_reasoning_eval_partial():
    """Only 2 out of 3 tasks correct -> ~66.7% score."""
    ctx = _make_ctx()
    # tasks[:3] are: derivative, train speed, logic question
    ctx.http.post.side_effect = [
        _ok_response("x^2(3ln(x)+1)"),  # task 0: correct
        _ok_response("wrong answer"),  # task 1: incorrect (not "70")
        _ok_response("no, we cannot conclude"),  # task 2: correct
    ]

    score = phase_reasoning_eval(ctx, n_tasks=3)
    assert score == pytest.approx(2 / 3 * 100, abs=0.1)


@pytest.mark.unit
def test_reasoning_eval_http_failure():
    """Non-200 responses are treated as failures, not exceptions."""
    ctx = _make_ctx()
    ctx.http.post.side_effect = [_error_response(500)] * 5

    score = phase_reasoning_eval(ctx, n_tasks=5)
    assert score == pytest.approx(0.0)


@pytest.mark.unit
def test_reasoning_eval_request_exception():
    """RequestException is caught gracefully -> score 0."""
    import requests

    ctx = _make_ctx()
    ctx.http.post.side_effect = requests.RequestException("connection refused")

    score = phase_reasoning_eval(ctx, n_tasks=1)
    assert score == pytest.approx(0.0)


@pytest.mark.unit
def test_reasoning_eval_returns_float():
    """phase_reasoning_eval returns a numeric score in [0, 100]."""
    ctx = _make_ctx()
    ctx.http.post.side_effect = [_ok_response("391")] * 5
    score = phase_reasoning_eval(ctx, n_tasks=5)
    assert isinstance(score, (int, float))
    assert 0 <= score <= 100


# ===================================================================
# phase_integrity_eval
# ===================================================================


@pytest.mark.unit
def test_integrity_eval_all_pass():
    """All tasks pass their check lambdas -> 100% score."""
    ctx = _make_ctx()
    answers = [
        "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
        "1024",
        "Red\nBlue\nGreen\nYellow\nPurple",
        "hello",
        "The capital of France is Paris.",
    ]
    ctx.http.post.side_effect = [_ok_response(a) for a in answers]

    score = phase_integrity_eval(ctx, n_tasks=5)
    assert score == pytest.approx(100.0)


@pytest.mark.unit
def test_integrity_eval_some_fail():
    """2 out of 3 pass -> ~66.7%."""
    ctx = _make_ctx()
    ctx.http.post.side_effect = [
        _ok_response("def foo():\n    return 42"),  # pass: has 'def' and 'return'
        _ok_response("999"),  # fail: not '1024'
        _ok_response("A\nB\nC\nD\nE"),  # pass: 5 lines
    ]

    score = phase_integrity_eval(ctx, n_tasks=3)
    assert score == pytest.approx(2 / 3 * 100, abs=0.1)


@pytest.mark.unit
def test_integrity_eval_http_failure():
    """Non-200 responses don't increment passed count."""
    ctx = _make_ctx()
    ctx.http.post.side_effect = [_error_response(503)] * 5

    score = phase_integrity_eval(ctx, n_tasks=5)
    assert score == pytest.approx(0.0)


@pytest.mark.unit
def test_integrity_eval_request_exception():
    """RequestException is caught gracefully -> score 0."""
    import requests

    ctx = _make_ctx()
    ctx.http.post.side_effect = requests.RequestException("timeout")

    score = phase_integrity_eval(ctx, n_tasks=1)
    assert score == pytest.approx(0.0)


@pytest.mark.unit
def test_integrity_eval_n_tasks_clipped():
    """n_tasks > len(tasks) is clamped to len(tasks)=5."""
    ctx = _make_ctx()
    answers = [
        "def f():\n    return 1",
        "1024",
        "A\nB\nC\nD\nE",
        "hello",
        "Paris",
    ]
    ctx.http.post.side_effect = [_ok_response(a) for a in answers]

    score = phase_integrity_eval(ctx, n_tasks=100)
    # Only 5 tasks exist, so denominator is min(100, 5) = 5
    assert score == pytest.approx(100.0)
    assert ctx.http.post.call_count == 5
