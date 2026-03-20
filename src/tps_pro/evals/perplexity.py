"""Perplexity (PPL) measurement and quality scoring."""

from __future__ import annotations

import logging
import math

import requests

from ..constants import (
    HTTP_OK,
    LARGE_REQUEST_TIMEOUT,
    PPL_DEGRADATION_FAIL,
    PPL_DEGRADATION_WARN,
    get_ppl_reference_text,
)
from ..state import AppContext

logger = logging.getLogger(__name__)

__all__ = ["measure_true_perplexity", "ppl_quality_factor"]


def _split_text_for_ppl(text_chunk: str) -> tuple[str, str, int]:
    """Split reference text into prompt and continuation for PPL measurement.

    Returns:
        (prompt_part, continuation, estimated_continuation_tokens)
    """
    split_point = int(len(text_chunk) * 0.8)
    period_pos = text_chunk.rfind(". ", 0, split_point)
    if period_pos > split_point * 0.5:
        split_point = period_pos + 2

    prompt_part = text_chunk[:split_point]
    continuation = text_chunk[split_point:]
    est_tokens = max(50, len(continuation) // 3)
    return prompt_part, continuation, est_tokens


def _extract_chat_logprobs(data: dict) -> list[float]:
    """Extract valid logprob values from a chat completions response."""
    choice = data.get("choices", [{}])[0]
    logprobs_data = choice.get("logprobs", {})
    content_logprobs = logprobs_data.get("content", []) if logprobs_data else []
    return [t.get("logprob") for t in content_logprobs if t.get("logprob") is not None]


def _extract_completions_logprobs(data: dict) -> list[float]:
    """Extract valid logprob values from a completions response."""
    logprobs_data = data.get("choices", [{}])[0].get("logprobs", {})
    token_logprobs = logprobs_data.get("token_logprobs", []) if logprobs_data else []
    return [lp for lp in token_logprobs if lp is not None]


def _fetch_logprobs_via_completions(
    ctx: AppContext, payload: dict
) -> list[float] | None:
    """Attempt to get logprobs via the /v1/completions endpoint.

    Returns:
        List of logprob floats, or None on failure.
    """
    r = ctx.http.post(
        f"{ctx.server_url}/v1/completions",
        json=payload,
        timeout=LARGE_REQUEST_TIMEOUT,
    )
    if r.status_code != HTTP_OK:
        return None
    return _extract_completions_logprobs(r.json())


def _fetch_logprobs_with_fallback(
    ctx: AppContext,
    chat_payload: dict[str, object],
    completions_payload: dict[str, object],
) -> list[float] | None:
    """Try chat endpoint first, fall back to completions endpoint.

    Returns:
        List of valid logprob floats (>= 10 entries), or None on failure.
    """
    r = ctx.http.post(
        f"{ctx.server_url}/v1/chat/completions",
        json=chat_payload,
        timeout=LARGE_REQUEST_TIMEOUT,
    )

    if r.status_code == HTTP_OK:
        logprobs = _extract_chat_logprobs(r.json())
        if len(logprobs) >= 10:  # noqa: PLR2004
            return logprobs
        logger.debug(
            "Chat endpoint returned too few logprobs (%d), trying completions",
            len(logprobs),
        )
    else:
        logger.debug(
            "Chat endpoint returned status %s, trying completions", r.status_code
        )

    fallback = _fetch_logprobs_via_completions(ctx, completions_payload)
    if fallback is not None and len(fallback) >= 10:  # noqa: PLR2004
        return fallback

    logger.warning("Both endpoints returned insufficient logprobs")
    return None


def measure_true_perplexity(ctx: AppContext, text_chunk: str | None = None) -> float:
    """Measure the exact Perplexity (PPL) of the model over standardized reference text.

    Uses the /v1/completions endpoint with logprobs to measure how well the model
    predicts each token in the reference text. Lower PPL = better.

    Returns:
        float: Perplexity value, or float('inf') on failure.
    """
    if text_chunk is None:
        text_chunk = get_ppl_reference_text()

    prompt_part, _continuation, est_tokens = _split_text_for_ppl(text_chunk)

    completions_payload = {
        "prompt": prompt_part,
        "max_tokens": est_tokens + 50,
        "temperature": 0.0,
        "logprobs": 1,
        "n_probs": 1,
    }
    chat_payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Continue the following text exactly"
                    " as written. Do not add commentary."
                ),
            },
            {"role": "user", "content": prompt_part},
        ],
        "max_tokens": est_tokens + 50,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 1,
    }

    try:
        valid_logprobs = _fetch_logprobs_with_fallback(
            ctx, chat_payload, completions_payload
        )
        if valid_logprobs is None:
            return float("inf")

        avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
        return math.exp(-avg_logprob)

    except (requests.RequestException, ValueError, KeyError, ZeroDivisionError) as e:
        logger.warning("PPL measurement failed: %s", e)
        return float("inf")


def ppl_quality_factor(baseline_ppl: float, trial_ppl: float) -> float:
    """Convert PPL degradation to a quality factor (0.0 to 1.0).

    Args:
        baseline_ppl: PPL from the baseline/reference config.
        trial_ppl: PPL from the trial config.

    Returns:
        float: 1.0 if no degradation, scaled down based on PPL increase.
    """
    from ._helpers import quality_factor_curve

    if baseline_ppl <= 0 or baseline_ppl == float("inf"):
        return 1.0  # no valid baseline, skip penalty
    if trial_ppl == float("inf"):
        return 0.1  # measurement failed = assume bad

    degradation = (trial_ppl - baseline_ppl) / baseline_ppl  # fraction increase
    return quality_factor_curve(degradation, PPL_DEGRADATION_WARN, PPL_DEGRADATION_FAIL)
