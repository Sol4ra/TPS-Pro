"""Unit tests for evals/niah.py — tokenize_count, build_niah_prompt, niah_test.

Consolidated from test_evals_niah.py and test_evals_niah_unit.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tps_pro.evals.niah import (
    TokenizeCache,
    build_niah_prompt,
    niah_test,
    phase_niah,
    tokenize_count,
)

# ===================================================================
# Helpers
# ===================================================================


def _make_ctx(**overrides):
    from _ctx_factory import make_ctx_from_defaults

    return make_ctx_from_defaults(**overrides)


# ===================================================================
# TokenizeCache
# ===================================================================


@pytest.mark.unit
class TestTokenizeCache:
    def test_initial_state(self):
        cache = TokenizeCache()
        assert cache.get() is None

    def test_set_and_get(self):
        cache = TokenizeCache()
        cache.set(4.0)
        assert cache.get() == 4.0


# ===================================================================
# tokenize_count
# ===================================================================


@pytest.mark.unit
class TestTokenizeCount:
    def test_uses_server_tokenize(self):
        """When server responds, uses exact token count."""
        ctx = _make_ctx()
        cache = TokenizeCache()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"tokens": list(range(10))}
        ctx.http.post.return_value = mock_response

        count = tokenize_count(ctx, "hello world test", cache)
        assert count == 10

    def test_caches_ratio_after_first_call(self):
        """After first call, ratio is set for fast estimation."""
        ctx = _make_ctx()
        cache = TokenizeCache()
        mock_response = MagicMock()
        mock_response.status_code = 200
        # 20 chars -> 5 tokens => ratio = 4.0
        mock_response.json.return_value = {"tokens": list(range(5))}
        ctx.http.post.return_value = mock_response

        tokenize_count(ctx, "12345678901234567890", cache)  # 20 chars
        assert cache.get() is not None
        assert cache.get() == 4.0

    def test_uses_cached_ratio(self):
        """When ratio is cached, estimates without HTTP call."""
        ctx = _make_ctx()
        cache = TokenizeCache()
        cache.set(4.0)

        count = tokenize_count(ctx, "a" * 40, cache)  # 40 chars / 4.0 = 10 tokens
        assert count == 10
        ctx.http.post.assert_not_called()

    def test_fallback_on_request_error(self):
        """When server fails, falls back to 3.0 chars/token."""
        import requests

        ctx = _make_ctx()
        cache = TokenizeCache()
        ctx.http.post.side_effect = requests.RequestException("connection refused")

        count = tokenize_count(ctx, "a" * 30, cache)  # 30 chars / 3.0 = 10
        assert count == 10


# ===================================================================
# build_niah_prompt
# ===================================================================


@pytest.mark.unit
class TestBuildNiahPrompt:
    def test_returns_string(self):
        """build_niah_prompt returns a string containing the needle."""
        ctx = _make_ctx()
        cache = TokenizeCache()
        cache.set(3.0)

        prompt = build_niah_prompt(
            ctx,
            target_tokens=500,
            needle_fact="The capital of Mars is Olympus.",
            cache=cache,
        )
        assert isinstance(prompt, str)
        assert "The capital of Mars is Olympus." in prompt

    def test_needle_at_different_depths(self):
        """Needle should appear regardless of depth setting."""
        ctx = _make_ctx()
        cache = TokenizeCache()
        cache.set(3.0)

        for depth in [0.0, 0.25, 0.5, 0.75, 1.0]:
            prompt = build_niah_prompt(
                ctx,
                target_tokens=300,
                needle_fact="SECRET_FACT",
                needle_depth_pct=depth,
                cache=cache,
            )
            assert "SECRET_FACT" in prompt


# ===================================================================
# niah_test — mocked
# ===================================================================


@pytest.mark.unit
class TestNiahTest:
    def test_niah_test_is_callable(self):
        """niah_test is a callable function."""
        assert callable(niah_test)
        # Verify it accepts expected params
        import inspect

        sig = inspect.signature(niah_test)
        assert "ctx" in sig.parameters
        assert "kv_cache_type" in sig.parameters
        assert "base_config" in sig.parameters

    def test_phase_niah_is_callable(self):
        """phase_niah is a callable function."""
        assert callable(phase_niah)
        import inspect

        sig = inspect.signature(phase_niah)
        assert "ctx" in sig.parameters

    @patch("tps_pro.evals.niah.load_phase_results")
    def test_phase_niah_cached_result(self, mock_load):
        """When cached NIAH results exist, returns them without running tests."""
        mock_load.return_value = {
            "niah_results": [{"kv_type": "f16", "pass_rate": 1.0}],
        }
        ctx = _make_ctx()
        # phase_niah checks for existing results
        result = phase_niah(ctx)
        # It should return the cached results
        assert result is not None
