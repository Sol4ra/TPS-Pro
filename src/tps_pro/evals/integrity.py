"""Reasoning and integrity evaluation phases.

Error strategy (see errors.py for full documentation):
    Individual eval tasks catch HTTP errors and log at warning level.
    A failed task counts as a non-pass (reduces score) rather than
    aborting the entire evaluation.  This is intentional: eval phases
    are informational and should be resilient to transient server hiccups.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

import requests  # type: ignore[import-untyped]

from ..constants import CONCURRENT_REQUEST_TIMEOUT, HTTP_OK
from ..state import AppContext

logger = logging.getLogger(__name__)

__all__ = ["phase_reasoning_eval", "phase_integrity_eval"]


# ---------------------------------------------------------------------------
# Task type: each task has a prompt and a checker that returns bool given
# the model's response content string.
# ---------------------------------------------------------------------------

_EvalTask = tuple[str, Callable[[str], bool]]


def _run_eval(
    ctx: AppContext,
    label: str,
    tasks: Sequence[_EvalTask],
    n_tasks: int = 5,
) -> float:
    """Run a generic evaluation phase over a list of tasks.

    Each task is a ``(prompt, check_fn)`` pair.  ``check_fn`` receives the
    model's response content string and returns ``True`` if the answer passes.

    Args:
        ctx: Application context with ``http`` session and ``server_url``.
        label: Human-readable phase name for log output.
        tasks: Sequence of ``(prompt, check_fn)`` pairs.
        n_tasks: Maximum number of tasks to evaluate.

    Returns:
        Score as a percentage (0.0 -- 100.0).
    """
    logger.info("=" * 60)
    logger.info("%s", label)
    logger.info("=" * 60)

    passed = 0
    evaluated = min(n_tasks, len(tasks))
    for prompt, check in tasks[:n_tasks]:
        try:
            r = ctx.http.post(
                f"{ctx.server_url}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.0,
                },
                timeout=CONCURRENT_REQUEST_TIMEOUT,
            )
            if r.status_code == HTTP_OK:
                content = (
                    r.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if check(content):
                    passed += 1
                    logger.info("[PASS] %s...", prompt[:60])
                else:
                    logger.info("[FAIL] %s... -> %s", prompt[:60], content[:80])
            else:
                logger.warning("Request failed: %s", r.status_code)
        except (requests.RequestException, ValueError, KeyError) as e:
            logger.warning("Error: %s", e)

    score = (passed / evaluated) * 100 if evaluated > 0 else 0.0
    logger.info("%s score: %.0f%% (%d/%d)", label, score, passed, evaluated)
    return score


def _accept_checker(accept: Sequence[str]) -> Callable[[str], bool]:
    """Return a checker: passes when any accept string found."""
    return lambda content: any(a.lower() in content.lower() for a in accept)


def phase_reasoning_eval(ctx: AppContext, n_tasks: int = 5) -> float:
    """Evaluate model reasoning quality. Purely informational."""
    tasks: list[_EvalTask] = [
        (
            "What is the derivative of x^3 * ln(x)?",
            _accept_checker(
                [
                    "3x^2 ln(x) + x^2",
                    "x^2(3ln(x)+1)",
                    "x^2(3 ln(x) + 1)",
                    "3x^2ln(x) + x^2",
                ]
            ),
        ),
        (
            "A train travels 60mph for 2 hours, then 90mph"
            " for 1 hour. What is the average speed?",
            _accept_checker(["70"]),
        ),
        (
            "If all roses are flowers and some flowers fade"
            " quickly, can we conclude all roses fade"
            " quickly?",
            _accept_checker(["no", "cannot conclude", "not necessarily"]),
        ),
        ("What is 17 * 23?", _accept_checker(["391"])),
        ("How many r's are in 'strawberry'?", _accept_checker(["3", "three"])),
    ]
    return _run_eval(ctx, "Reasoning Evaluation", tasks, n_tasks)


def phase_integrity_eval(ctx: AppContext, n_tasks: int = 5) -> float:
    """Evaluate output integrity under optimized settings."""
    tasks: list[_EvalTask] = [
        (
            "Write a Python function that returns the"
            " factorial of n. Only output the code,"
            " no explanation.",
            lambda c: "def " in c and "return" in c,
        ),
        ("What is 2^10?", lambda c: "1024" in c),
        (
            "List exactly 5 colors, one per line, no numbering.",
            lambda c: (
                len([line for line in c.strip().split("\n") if line.strip()]) >= 4  # noqa: PLR2004
            ),
        ),
        (
            "Respond with ONLY the word 'hello' and nothing else.",
            lambda c: "hello" in c.lower() and len(c.strip()) < 20,  # noqa: PLR2004
        ),
        ("What is the capital of France?", lambda c: "paris" in c.lower()),
    ]
    return _run_eval(ctx, "Integrity Evaluation", tasks, n_tasks)
