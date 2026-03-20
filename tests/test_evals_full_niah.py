"""Tests for evals submodules: niah.py — build_niah_prompt, tokenize_count,
niah_test."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from _evals_full_helpers import (
    _chat_completion_response,
    _mock_response,
    _patch_ctx,  # noqa: F401 — pytest fixture
)

# ===================================================================
# niah.py — build_niah_prompt, tokenize_count, niah_test
# ===================================================================


class TestBuildNiahPrompt:
    """Tests for build_niah_prompt."""

    @pytest.fixture(autouse=True)
    def _reset_tokenize_cache(self):
        pass  # TokenizeCache is now per-call; no module state to reset
        yield

    @pytest.mark.unit
    def test_needle_present_in_output(self, _patch_ctx):
        """The needle fact should appear in the built prompt."""
        from tps_pro.evals.niah import build_niah_prompt

        _patch_ctx.http.post.return_value = _mock_response(
            200, {"tokens": list(range(500))}
        )

        needle = "The secret code is ALPHA-7"
        prompt = build_niah_prompt(_patch_ctx, target_tokens=2048, needle_fact=needle)
        assert "ALPHA-7" in prompt
        assert "IMPORTANT NOTE:" in prompt

    @pytest.mark.unit
    def test_needle_at_start(self, _patch_ctx):
        """depth_pct=0.0 should place needle near the start."""
        from tps_pro.evals.niah import build_niah_prompt

        _patch_ctx.http.post.return_value = _mock_response(
            200, {"tokens": list(range(500))}
        )

        needle = "NEEDLE_START_TEST"
        prompt = build_niah_prompt(
            _patch_ctx, target_tokens=2048, needle_fact=needle, needle_depth_pct=0.0
        )
        pos = prompt.find("NEEDLE_START_TEST")
        assert pos >= 0
        assert pos < len(prompt) * 0.3

    @pytest.mark.unit
    def test_needle_at_end(self, _patch_ctx):
        """depth_pct=1.0 should place needle near the end."""
        from tps_pro.evals.niah import build_niah_prompt

        _patch_ctx.http.post.return_value = _mock_response(
            200, {"tokens": list(range(500))}
        )

        needle = "NEEDLE_END_TEST"
        prompt = build_niah_prompt(
            _patch_ctx, target_tokens=2048, needle_fact=needle, needle_depth_pct=1.0
        )
        pos = prompt.find("NEEDLE_END_TEST")
        assert pos >= 0
        assert pos > len(prompt) * 0.7

    @pytest.mark.unit
    def test_prompt_has_filler(self, _patch_ctx):
        """Prompt should contain filler text sections."""
        from tps_pro.evals.niah import build_niah_prompt

        _patch_ctx.http.post.return_value = _mock_response(
            200, {"tokens": list(range(500))}
        )

        prompt = build_niah_prompt(
            _patch_ctx, target_tokens=2048, needle_fact="test fact"
        )
        assert "Section" in prompt


class TestTokenizeCount:
    """Tests for tokenize_count."""

    @pytest.mark.unit
    def test_exact_count_from_server(self, _patch_ctx):
        """First call uses server /tokenize for exact count."""
        from tps_pro.evals.niah import TokenizeCache, tokenize_count

        cache = TokenizeCache()
        tokens = list(range(42))
        _patch_ctx.http.post.return_value = _mock_response(200, {"tokens": tokens})

        count = tokenize_count(_patch_ctx, "some text here for testing", cache)
        assert count == 42

    @pytest.mark.unit
    def test_cached_ratio_used_on_subsequent_calls(self, _patch_ctx):
        """After calibration, subsequent calls use the cached ratio."""
        from tps_pro.evals.niah import TokenizeCache, tokenize_count

        cache = TokenizeCache()

        text = "a" * 100
        _patch_ctx.http.post.return_value = _mock_response(
            200, {"tokens": list(range(25))}
        )
        tokenize_count(_patch_ctx, text, cache)

        assert cache.get() == pytest.approx(4.0)

        _patch_ctx.http.post.reset_mock()
        count = tokenize_count(_patch_ctx, "b" * 200, cache)
        _patch_ctx.http.post.assert_not_called()
        assert count == 50  # 200 / 4.0

    @pytest.mark.unit
    def test_fallback_on_server_failure(self, _patch_ctx):
        """Server failure -> falls back to 3.0 chars/token estimate."""
        import requests as req

        from tps_pro.evals.niah import TokenizeCache, tokenize_count

        cache = TokenizeCache()
        _patch_ctx.http.post.side_effect = req.ConnectionError("refused")

        count = tokenize_count(_patch_ctx, "a" * 90, cache)  # 90 / 3.0 = 30
        assert count == 30

    @pytest.mark.unit
    def test_empty_tokens_response(self, _patch_ctx):
        """Server returns empty tokens list -> falls back."""
        from tps_pro.evals.niah import TokenizeCache, tokenize_count

        cache = TokenizeCache()
        _patch_ctx.http.post.return_value = _mock_response(200, {"tokens": []})
        count = tokenize_count(_patch_ctx, "a" * 90, cache)
        assert count == 0  # len([]) = 0, returned directly


class TestNiahTest:
    """Tests for niah_test (full flow, heavily mocked)."""

    @pytest.mark.unit
    @patch("tps_pro.evals.niah.kill_server")
    @patch("tps_pro.evals.niah.wait_for_server", return_value="ok")
    @patch("tps_pro.evals.niah.start_server")
    @patch(
        "tps_pro.evals.niah.measure_true_perplexity",
        return_value=5.0,
    )
    @patch("tps_pro.evals.niah.tokenize_count", return_value=500)
    def test_all_needles_found(
        self, mock_tok, mock_ppl, mock_start, mock_wait, mock_kill, _patch_ctx
    ):
        """All needles found -> 100% pass rate."""
        from tps_pro.constants import NIAH_NEEDLES
        from tps_pro.evals.niah import niah_test

        def _respond(*args, **kwargs):
            payload = kwargs.get("json", {})
            msgs = payload.get("messages", [])
            user_msg = msgs[-1].get("content", "") if msgs else ""
            for needle in NIAH_NEEDLES:
                if needle["query"] in user_msg:
                    return _chat_completion_response(needle["expected"])
            return _chat_completion_response("sapphire falcon")

        _patch_ctx.http.post.side_effect = _respond

        result = niah_test(
            _patch_ctx,
            kv_cache_type="f16",
            base_config={"context": 4096, "n_gpu_layers": 99},
            depths=[0.25],
            context_sizes=[2048],
        )
        assert result["pass_rate"] == pytest.approx(100.0)
        assert result["kv_type"] == "f16"
        assert len(result["results"]) == 1
        assert result["results"][0]["passed"] is True

    @pytest.mark.unit
    @patch("tps_pro.evals.niah.kill_server")
    @patch("tps_pro.evals.niah.wait_for_server", return_value="oom")
    @patch("tps_pro.evals.niah.start_server")
    def test_oom_returns_zero_pass_rate(
        self, mock_start, mock_wait, mock_kill, _patch_ctx
    ):
        """OOM during server start -> 0% pass rate with oom flag."""
        from tps_pro.evals.niah import niah_test

        result = niah_test(
            _patch_ctx,
            kv_cache_type="q4_0",
            base_config={"context": 4096},
        )
        assert result["pass_rate"] == pytest.approx(0.0)
        assert result.get("oom") is True
        mock_kill.assert_called()

    @pytest.mark.unit
    @patch("tps_pro.evals.niah.kill_server")
    @patch("tps_pro.evals.niah.wait_for_server", return_value="ok")
    @patch("tps_pro.evals.niah.start_server")
    @patch(
        "tps_pro.evals.niah.measure_true_perplexity",
        return_value=5.0,
    )
    @patch("tps_pro.evals.niah.tokenize_count", return_value=500)
    def test_needle_not_found(
        self, mock_tok, mock_ppl, mock_start, mock_wait, mock_kill, _patch_ctx
    ):
        """Needle not found in response -> failed test."""
        from tps_pro.evals.niah import niah_test

        _patch_ctx.http.post.return_value = _chat_completion_response(
            "I don't know the answer"
        )

        result = niah_test(
            _patch_ctx,
            kv_cache_type="q4_0",
            base_config={"context": 4096},
            depths=[0.5],
            context_sizes=[2048],
        )
        assert result["pass_rate"] == pytest.approx(0.0)
        assert result["results"][0]["passed"] is False

    @pytest.mark.unit
    @patch("tps_pro.evals.niah.kill_server")
    @patch("tps_pro.evals.niah.wait_for_server", return_value="ok")
    @patch("tps_pro.evals.niah.start_server")
    @patch(
        "tps_pro.evals.niah.measure_true_perplexity",
        return_value=5.0,
    )
    @patch("tps_pro.evals.niah.tokenize_count", return_value=500)
    def test_http_error_during_test(
        self, mock_tok, mock_ppl, mock_start, mock_wait, mock_kill, _patch_ctx
    ):
        """HTTP error during NIAH request -> recorded as failure."""
        import requests as req

        from tps_pro.evals.niah import niah_test

        _patch_ctx.http.post.side_effect = req.ConnectionError("timeout")

        result = niah_test(
            _patch_ctx,
            kv_cache_type="f16",
            base_config={"context": 4096},
            depths=[0.25],
            context_sizes=[2048],
        )
        assert result["pass_rate"] == pytest.approx(0.0)
        assert "error" in result["results"][0]

    @pytest.mark.unit
    @patch("tps_pro.evals.niah.kill_server")
    @patch("tps_pro.evals.niah.wait_for_server", return_value="ok")
    @patch("tps_pro.evals.niah.start_server")
    @patch(
        "tps_pro.evals.niah.measure_true_perplexity",
        return_value=5.0,
    )
    @patch("tps_pro.evals.niah.tokenize_count", return_value=500)
    def test_strips_speculative_params(
        self, mock_tok, mock_ppl, mock_start, mock_wait, mock_kill, _patch_ctx
    ):
        """Speculative decoding params should be stripped from config."""
        from tps_pro.evals.niah import niah_test

        _patch_ctx.http.post.return_value = _chat_completion_response("sapphire falcon")

        base_config = {
            "context": 4096,
            "n_gpu_layers": 99,
            "spec_draft_model": "some_model",
            "draft_n": 5,
            "lookup_cache_dynamic": True,
        }

        niah_test(
            _patch_ctx,
            kv_cache_type="f16",
            base_config=base_config,
            depths=[0.25],
            context_sizes=[2048],
        )

        config_arg = mock_start.call_args[0][1]
        assert "spec_draft_model" not in config_arg
        assert "draft_n" not in config_arg
        assert "lookup_cache_dynamic" not in config_arg
        assert config_arg["kv_cache_type"] == "f16"
