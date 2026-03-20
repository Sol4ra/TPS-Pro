"""Multiple-choice question quality evaluation."""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time

import requests

from ..constants import (
    HAS_AIOHTTP,
    HTTP_OK,
    LARGE_REQUEST_TIMEOUT,
    QUALITY_EVAL_SEED,
    QUALITY_TASKS,
    QUALITY_TTFT_BASELINE,
    QUALITY_WEIGHT_CONFIDENCE,
    QUALITY_WEIGHT_CORRECTNESS,
    QUALITY_WEIGHT_EFFICIENCY,
)
from ..result_types import QualityResult, QualityTaskResult, SamplingParams
from ..state import AppContext

if HAS_AIOHTTP:
    import aiohttp

logger = logging.getLogger(__name__)

__all__ = [
    "_extract_answer_letter",
    "_extract_answer_logprob",
    "measure_quality",
    "_score_quality_results",
    "_eval_single_task",
    "_measure_quality_sequential",
    "_eval_one_async",
    "_run_async_quality_tasks",
    "_measure_quality_async",
]


def _extract_answer_letter(content: str) -> str | None:
    """Extract the MC answer letter (A/B/C/D) from model response.

    Handles various formats: "(A)", "A)", "A.", "Answer: A", "the answer is A", etc.
    Returns the letter or None if no clear answer found.
    """
    content = content.strip()
    # Try common patterns, most specific first
    patterns = [
        # "answer is (A)" / "choice: B"
        r"(?:answer|choice)\s*(?:is|:)\s*\(?([A-D])\)?",
        r"\(?([A-D])\)?\s*$",  # ends with "(A)" or "A"
        r"^([A-D])\b",  # starts with "A"
        r"\(([A-D])\)",  # "(A)" anywhere
    ]
    for pattern in patterns:
        m = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def _extract_answer_logprob(data: dict, correct_letter: str) -> float | None:
    """Extract the logprob for the answer token from OpenAI-format response.

    Scans the logprobs for the token matching the correct letter.
    Returns float logprob or None if not found.
    """
    logprobs_data = data.get("choices", [{}])[0].get("logprobs", {})
    content_logprobs = logprobs_data.get("content", []) if logprobs_data else []

    for token_info in content_logprobs:
        token = token_info.get("token", "").strip()
        # Match the answer letter token (could be "A", "(A)", etc.)
        if (
            token.upper() == correct_letter
            or token.strip("()").upper() == correct_letter
        ):
            return token_info.get("logprob", None)
        # Also check top_logprobs for the correct answer
        for alt in token_info.get("top_logprobs", []):
            alt_token = alt.get("token", "").strip()
            if (
                alt_token.upper() == correct_letter
                or alt_token.strip("()").upper() == correct_letter
            ):
                return alt.get("logprob", None)

    return None


def measure_quality(
    ctx: AppContext,
    sampling_params: SamplingParams,
    tasks: list[tuple[str, str, str]] = QUALITY_TASKS,
    target_to_beat: float | None = None,
) -> QualityResult:
    """3-signal quality eval: Correctness (40%) + Confidence (40%) + Efficiency (20%).

    Each task is a hard multiple-choice question. The model's response is scored on:
      1. Correctness: did it pick the right letter? (binary 0/1)
      2. Confidence: logprob of the correct answer token (higher = more confident)
      3. Efficiency: TTFT in ms (lower = faster, normalized against baseline)

    Uses seed=0 and temperature=0 for deterministic grading so any score change
    is strictly from parameter changes, not model randomness.

    Returns QualityResult with composite score 0-100 and individual task results.
    """
    oai_params = {}
    for k, v in sampling_params.items():
        if k == "n_predict":
            oai_params["max_tokens"] = v
        else:
            oai_params[k] = v
    max_tokens = oai_params.pop("max_tokens", 1024)

    if HAS_AIOHTTP:
        return _measure_quality_async(
            ctx, tasks, max_tokens, oai_params
        )

    # Sequential path — pass target_to_beat for early short-circuit
    return _measure_quality_sequential(
        ctx, tasks, max_tokens, oai_params, target_to_beat=target_to_beat
    )


def _score_quality_results(task_results: list[QualityTaskResult]) -> QualityResult:
    """Compute composite quality score from individual task results.

    Args:
        task_results: list of QualityTaskResult objects.

    Returns:
        QualityResult with composite score 0-100 and the task_results.
    """
    if not task_results:
        return QualityResult(score=0.0)

    n = len(task_results)

    # Signal 1: Correctness (0-1 per task, averaged)
    correctness = sum(1.0 for r in task_results if r.correct) / n

    # Signal 2: Confidence (logprob of correct answer, normalized to 0-1)
    # logprob ranges: 0.0 (100% confident) to -inf (no confidence)
    # Map: 0.0 → 1.0, -1.0 → 0.5, -3.0 → 0.1, worse → 0.0
    logprobs = [r.logprob for r in task_results if r.logprob is not None]
    if logprobs:
        avg_logprob = sum(logprobs) / len(logprobs)
        # Sigmoid-like mapping: confidence = exp(logprob) clamped to [0, 1]
        confidence = min(1.0, max(0.0, math.exp(avg_logprob)))
    else:
        # No logprobs available — fall back to correctness as proxy
        confidence = correctness

    # Signal 3: Efficiency (TTFT normalized against baseline)
    ttfts = [r.ttft_ms for r in task_results if r.ttft_ms is not None and r.ttft_ms > 0]
    if ttfts:
        avg_ttft = sum(ttfts) / len(ttfts)
        # Lower TTFT = better. Normalize: baseline/actual, capped at [0, 1]
        efficiency = min(1.0, QUALITY_TTFT_BASELINE / avg_ttft) if avg_ttft > 0 else 0.0
    else:
        efficiency = 0.5  # neutral if no timing data

    # Weighted composite
    score = (
        QUALITY_WEIGHT_CORRECTNESS * correctness
        + QUALITY_WEIGHT_CONFIDENCE * confidence
        + QUALITY_WEIGHT_EFFICIENCY * efficiency
    )

    return QualityResult(score=score * 100, task_results=list(task_results))


def _eval_single_task(  # noqa: PLR0913
    ctx: AppContext,
    prompt: str,
    correct_letter: str,
    category: str,
    max_tokens: int,
    oai_params: dict,
) -> QualityTaskResult:
    """Evaluate a single MC task. Returns QualityTaskResult."""
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "seed": QUALITY_EVAL_SEED,
        "logprobs": True,
        "top_logprobs": 5,
        **oai_params,
    }

    try:
        start_ms = time.time() * 1000
        r = ctx.http.post(
            f"{ctx.server_url}/v1/chat/completions",
            json=payload,
            timeout=LARGE_REQUEST_TIMEOUT,
        )
        ttft_ms = time.time() * 1000 - start_ms

        if r.status_code == HTTP_OK:
            data = r.json()
            choices = data.get("choices", [])
            content = (
                choices[0].get("message", {}).get("content", "") if choices else ""
            )

            # Extract answer letter
            answer = _extract_answer_letter(content)
            correct = answer == correct_letter

            # Extract logprob for the correct answer
            logprob = _extract_answer_logprob(data, correct_letter)

            return QualityTaskResult(
                correct=correct,
                logprob=logprob,
                ttft_ms=ttft_ms,
                answer=answer,
                category=category,
            )
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.debug("Quality eval request failed: %s", e)

    return QualityTaskResult(
        correct=False, logprob=None, ttft_ms=None, answer=None, category=category
    )


def _measure_quality_sequential(
    ctx: AppContext,
    tasks: list[tuple[str, str, str]],
    max_tokens: int,
    oai_params: dict,
    target_to_beat: float | None = None,
) -> QualityResult:
    """Sequential quality eval with early exit.

    If target_to_beat is set: after each question, check if the maximum possible
    score (assuming all remaining questions are perfect) can still beat the target.
    If not, bail out early to save time on obviously bad configs.
    """
    results: list[QualityTaskResult] = []
    total_tasks = len(tasks)

    for i, (prompt, correct_letter, category) in enumerate(tasks):
        result = _eval_single_task(
            ctx, prompt, correct_letter, category, max_tokens, oai_params
        )
        results.append(result)

        # Short-circuit: can we still mathematically beat the target?
        if target_to_beat is not None and i < total_tasks - 1:
            current_correct = sum(1.0 for r in results if r.correct)
            remaining = total_tasks - (i + 1)
            max_possible_correct = (current_correct + remaining) / total_tasks
            # Even with perfect confidence + efficiency, correctness is 40% of score
            # Use 20% margin to account for confidence/efficiency contributions
            if max_possible_correct * 100 < target_to_beat - 20:
                if ctx.debug:
                    logger.debug(
                        "[short-circuit] %d/%d correct,"
                        " max possible %.0f%%"
                        " — bailing (%d skipped)",
                        int(current_correct),
                        i + 1,
                        max_possible_correct * 100,
                        total_tasks - i - 1,
                    )
                return _score_quality_results(results)

    return _score_quality_results(results)


async def _eval_one_async(  # noqa: PLR0913
    session: aiohttp.ClientSession,
    server_url: str,
    prompt: str,
    correct_letter: str,
    category: str,
    max_tokens: int,
    oai_params: dict,
) -> QualityTaskResult:
    """Evaluate a single MC task asynchronously via aiohttp.

    Args:
        session: aiohttp ClientSession.
        server_url: Base URL for the llama server.
        prompt: The question prompt.
        correct_letter: Expected answer letter (A/B/C/D).
        category: Task category string.
        max_tokens: Maximum tokens for completion.
        oai_params: Additional OpenAI-format parameters.

    Returns:
        QualityTaskResult with evaluation outcome.
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "seed": QUALITY_EVAL_SEED,
        "logprobs": True,
        "top_logprobs": 5,
        **oai_params,
    }
    try:
        start_ms = time.time() * 1000
        async with session.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=LARGE_REQUEST_TIMEOUT),
        ) as r:
            ttft_ms = time.time() * 1000 - start_ms
            if r.status == HTTP_OK:
                data = await r.json()
                choices = data.get("choices", [])
                content = (
                    choices[0].get("message", {}).get("content", "") if choices else ""
                )
                answer = _extract_answer_letter(content)
                correct = answer == correct_letter
                logprob = _extract_answer_logprob(data, correct_letter)
                return QualityTaskResult(
                    correct=correct,
                    logprob=logprob,
                    ttft_ms=ttft_ms,
                    answer=answer,
                    category=category,
                )
    except (aiohttp.ClientError, ValueError, KeyError) as e:
        logger.debug("Async quality eval failed: %s", e)
    return QualityTaskResult(
        correct=False, logprob=None, ttft_ms=None, answer=None, category=category
    )


async def _run_async_quality_tasks(
    server_url: str,
    tasks: list[tuple[str, str, str]],
    max_tokens: int,
    oai_params: dict,
) -> QualityResult:
    """Gather all quality tasks concurrently via aiohttp.

    Extracted to module level to reduce nesting depth in _measure_quality_async.

    Args:
        server_url: Base URL for the llama server.
        tasks: List of (prompt, correct_letter, category) tuples.
        max_tokens: Maximum tokens for completion.
        oai_params: Additional OpenAI-format parameters.

    Returns:
        QualityResult with composite score and individual task results.
    """
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *[
                _eval_one_async(
                    session,
                    server_url,
                    prompt,
                    correct_letter,
                    category,
                    max_tokens,
                    oai_params,
                )
                for prompt, correct_letter, category in tasks
            ]
        )
    return _score_quality_results(list(results))


def _measure_quality_async(
    ctx: AppContext,
    tasks: list[tuple[str, str, str]],
    max_tokens: int,
    oai_params: dict,
) -> QualityResult:
    """Fire all quality tasks concurrently via aiohttp and return QualityResult.

    Note: async path does not implement early-exit optimization.
    All tasks run to completion.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(
                asyncio.run,
                _run_async_quality_tasks(ctx.server_url, tasks, max_tokens, oai_params),
            ).result()
    return asyncio.run(
        _run_async_quality_tasks(ctx.server_url, tasks, max_tokens, oai_params)
    )
