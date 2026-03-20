"""Shared helpers and fixtures for test_evals_full_* split files."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# Pre-import all eval submodules so patch targets resolve correctly.
import tps_pro.evals.integrity  # noqa: F401
import tps_pro.evals.kl_divergence  # noqa: F401
import tps_pro.evals.mcq  # noqa: F401
import tps_pro.evals.niah  # noqa: F401
import tps_pro.evals.perplexity  # noqa: F401
import tps_pro.evals.quality_gate  # noqa: F401


def _mock_response(status_code: int = 200, json_data: dict | None = None):
    """Build a requests.Response-like mock."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    return resp


def _chat_completion_response(content: str, logprobs_content: list | None = None):
    """Build a standard chat/completions 200 response."""
    choice = {"message": {"content": content}}
    if logprobs_content is not None:
        choice["logprobs"] = {"content": logprobs_content}
    return _mock_response(200, {"choices": [choice]})


def _logprob_token(token: str, logprob: float, top_logprobs: list | None = None):
    """Build a single token logprob entry."""
    entry = {"token": token, "logprob": logprob}
    if top_logprobs is not None:
        entry["top_logprobs"] = top_logprobs
    return entry


@pytest.fixture(autouse=True)
def _patch_ctx():
    """Patch evals.state.ctx with a minimal mock for every test."""
    mock_http = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.http = mock_http
    mock_ctx.server_url = "http://127.0.0.1:9999"
    mock_ctx.debug = False
    mock_ctx.quality_baseline = None

    yield mock_ctx
