"""Concurrent load testing and token uncertainty measurement.

Error strategy (see errors.py for full documentation):
    - measure_concurrent_load(): returns None on failure (logged at debug).
    - measure_token_uncertainty(): returns None on failure.  Individual
      prompt failures are logged at warning but do not abort the batch.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aiohttp as _aiohttp_type

import requests  # type: ignore[import-untyped]

from ..constants import (
    CONCURRENT_REQUEST_TIMEOUT,
    DEFAULT_TEMPERATURE,
    HAS_AIOHTTP,
    HTTP_OK,
    LARGE_REQUEST_TIMEOUT,
)
from ..result_types import (
    ConcurrentLoadResult,
    ConcurrentUserResult,
    TokenUncertaintyResult,
)
from ..state import AppContext

if HAS_AIOHTTP:
    import aiohttp

logger = logging.getLogger(__name__)

__all__ = [
    "measure_concurrent_load",
    "measure_token_uncertainty",
]


_CONCURRENT_PROMPTS: list[str] = [
    "Explain the concept of entropy in information theory.",
    "Write a Python function that implements binary search on a sorted array.",
    "What are the key differences between TCP and UDP protocols?",
    "Describe the process of photosynthesis in plants.",
    "Explain how a hash table works and its time complexity.",
    "What is the difference between a stack and a queue?",
    "Describe the CAP theorem in distributed systems.",
    "Explain the concept of polymorphism in OOP.",
]


def _run_async_load_test(
    coro_factory: Callable[[], Coroutine[Any, Any, list[ConcurrentUserResult]]],
) -> list[ConcurrentUserResult]:
    """Execute an async load test, handling existing event loops.

    Returns:
        List of ConcurrentUserResult from the async load test.

    Raises:
        RuntimeError, OSError: If the event loop cannot be created or run.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro_factory()).result()
    return asyncio.run(coro_factory())


def _aggregate_concurrent_results(
    results: list[ConcurrentUserResult], n_users: int
) -> ConcurrentLoadResult | None:
    """Aggregate individual user results into a ConcurrentLoadResult."""
    successful = [r for r in results if r.success]
    if not successful:
        return None

    total_tps = sum(r.tps for r in successful)
    avg_ttft = sum(r.ttft for r in successful) / len(successful)
    avg_wall = sum(r.wall_time for r in successful) / len(successful)
    max_wall = max(r.wall_time for r in successful)

    return ConcurrentLoadResult(
        concurrent_total_tps=total_tps,
        concurrent_avg_tps=total_tps / len(successful),
        concurrent_avg_ttft=avg_ttft,
        concurrent_avg_wall_ms=avg_wall,
        concurrent_max_wall_ms=max_wall,
        concurrent_success_rate=len(successful) / n_users,
        concurrent_users=n_users,
    )


def measure_concurrent_load(
    ctx: AppContext, n_users: int = 4, n_predict: int = 50
) -> ConcurrentLoadResult | None:
    """Concurrent load test: send N simultaneous requests and measure system throughput.

    Scores the config on:
      - Total system throughput (aggregate tokens/sec across all users)
      - Queue latency (time from request sent to first token received per user)
      - Crash resilience (did all requests succeed?)

    Requires aiohttp. Returns ConcurrentLoadResult or None on failure.
    """
    if not HAS_AIOHTTP:
        return None

    async def _single_request(
        session: _aiohttp_type.ClientSession, prompt: str, user_id: int
    ) -> ConcurrentUserResult:
        payload = {
            "model": "test",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict,
            "temperature": DEFAULT_TEMPERATURE,
        }
        start_time = time.time()
        try:
            async with session.post(
                f"{ctx.server_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=CONCURRENT_REQUEST_TIMEOUT),
            ) as resp:
                if resp.status == HTTP_OK:
                    data = await resp.json()
                    elapsed = time.time() - start_time
                    timings = data.get("timings", {})
                    return ConcurrentUserResult(
                        user_id=user_id,
                        tps=timings.get("predicted_per_second", 0),
                        ttft=timings.get("prompt_ms", 0),
                        prompt_tps=timings.get("prompt_per_second", 0),
                        wall_time=elapsed * 1000,
                        success=True,
                    )
                return ConcurrentUserResult(
                    user_id=user_id, success=False, error=f"status {resp.status}"
                )
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
        ) as e:
            return ConcurrentUserResult(user_id=user_id, success=False, error=str(e))

    async def _run_load_test() -> list[ConcurrentUserResult]:
        async with aiohttp.ClientSession() as session:
            tasks = [
                _single_request(
                    session, _CONCURRENT_PROMPTS[i % len(_CONCURRENT_PROMPTS)], i
                )
                for i in range(n_users)
            ]
            return await asyncio.gather(*tasks)

    try:
        results = _run_async_load_test(_run_load_test)
    except (RuntimeError, OSError) as e:
        logger.debug("Concurrent load test failed: %s", e)
        return None

    return _aggregate_concurrent_results(results, n_users)


def _collect_prompt_logprobs(
    ctx: AppContext,
    prompt_text: str,
    n_predict: int,
    seed: int,
) -> list[float]:
    """Fetch logprobs for a single quality-gate prompt.

    Returns:
        List of negative logprob values from the response (may be empty).
    """
    payload = {
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": n_predict,
        "temperature": 0.0,
        "seed": seed,
        "logprobs": True,
        "top_logprobs": 1,
    }
    try:
        r = ctx.http.post(
            f"{ctx.server_url}/v1/chat/completions",
            json=payload,
            timeout=LARGE_REQUEST_TIMEOUT,
        )
        if r.status_code != HTTP_OK:
            logger.warning("Quality request returned status %s", r.status_code)
            return []
        data = r.json()
        logprobs_data = data.get("choices", [{}])[0].get("logprobs", {})
        content_logprobs = logprobs_data.get("content", []) if logprobs_data else []
        if not content_logprobs:
            logger.warning("No logprobs in response. Keys: %s", list(data.keys()))
            return []
        return [
            t.get("logprob")
            for t in content_logprobs
            if t.get("logprob") is not None and t.get("logprob") < 0
        ]
    except (
        requests.RequestException,
        json.JSONDecodeError,
        ValueError,
        KeyError,
    ) as e:
        logger.warning("Quality measurement failed: %s", e)
        return []


def measure_token_uncertainty(ctx: AppContext) -> TokenUncertaintyResult | None:
    """Measure token-level uncertainty on quality gate prompts.

    Returns TokenUncertaintyResult with:
      - uncertain_count: number of tokens with logprob < threshold (-2.0)
      - tail_avg: average logprob of the worst 20% of tokens
      - total_tokens: total tokens measured
    Or None on failure.
    """
    from ..constants import (
        QUALITY_GATE_N_PREDICT,
        QUALITY_GATE_PROMPTS,
        QUALITY_GATE_SEED,
        QUALITY_GATE_TAIL_PCT,
        QUALITY_GATE_UNCERTAIN_THRESHOLD,
    )

    all_logprobs: list[float] = []
    for prompt_text in QUALITY_GATE_PROMPTS:
        all_logprobs.extend(
            _collect_prompt_logprobs(
                ctx, prompt_text, QUALITY_GATE_N_PREDICT, QUALITY_GATE_SEED
            )
        )

    if not all_logprobs:
        return None

    # Count uncertain tokens (logprob < -2.0 = less than ~13% confidence)
    uncertain_count = sum(
        1 for lp in all_logprobs if lp < QUALITY_GATE_UNCERTAIN_THRESHOLD
    )

    # Tail-20% average: sort logprobs ascending, average the worst 20%
    sorted_lps = sorted(all_logprobs)
    tail_n = max(1, int(len(sorted_lps) * QUALITY_GATE_TAIL_PCT))
    tail_avg = sum(sorted_lps[:tail_n]) / tail_n

    return TokenUncertaintyResult(
        uncertain_count=uncertain_count,
        tail_avg=tail_avg,
        total_tokens=len(all_logprobs),
    )
