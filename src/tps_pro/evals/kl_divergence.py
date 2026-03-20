"""KL-divergence measurement between baseline and trial logprob distributions."""

from __future__ import annotations

import logging
import math

import requests

from ..constants import (
    CONCURRENT_REQUEST_TIMEOUT,
    HTTP_OK,
    KL_DIV_HARD_FAIL,
    KL_DIV_PROMPTS,
    KL_DIV_THRESHOLD,
    KL_DIV_TOP_K,
)
from ..result_types import KLResult
from ..state import AppContext

# Default logprob assigned when a token is absent from the API's top-K list.
# Very negative so exp() ~ 0, effectively treating missing tokens as unseen.
_MISSING_TOKEN_LOGPROB = -100.0

# Floor logprob used in KL computation for tokens present in one distribution
# but absent in the other.  Less extreme than _MISSING_TOKEN_LOGPROB because
# KL only sums over the baseline's support, so the floor just needs to be low
# enough to represent "negligible probability" without causing numerical issues.
_KL_FLOOR_LOGPROB = -20.0

logger = logging.getLogger(__name__)

__all__ = [
    "_collect_logprob_distribution",
    "_compute_kl_divergence",
    "measure_kl_divergence",
    "kl_quality_factor",
]


def _collect_logprob_distribution(
    ctx: AppContext, prompts: list[str], top_k: int = KL_DIV_TOP_K
) -> list[dict[str, float]] | None:
    """Collect top-K logprob distributions for each token across prompts.

    Returns list of lists: for each token position, a dict {token: logprob}.
    Returns None on failure.
    """
    all_distributions = []
    for prompt_text in prompts:
        payload = {
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": 128,
            "temperature": 0.0,
            "seed": 42,
            "logprobs": True,
            "top_logprobs": top_k,
        }
        try:
            r = ctx.http.post(
                f"{ctx.server_url}/v1/chat/completions",
                json=payload,
                timeout=CONCURRENT_REQUEST_TIMEOUT,
            )
            if r.status_code != HTTP_OK:
                continue
            data = r.json()
            logprobs_data = data.get("choices", [{}])[0].get("logprobs", {})
            content_logprobs = logprobs_data.get("content", []) if logprobs_data else []
            for token_info in content_logprobs:
                top_lps = token_info.get("top_logprobs", [])
                if top_lps:
                    dist = {}
                    for entry in top_lps:
                        tok = entry.get("token", "")
                        lp = entry.get("logprob", _MISSING_TOKEN_LOGPROB)
                        dist[tok] = lp
                    all_distributions.append(dist)
        except (requests.RequestException, ValueError, KeyError) as e:
            logger.debug("KL distribution collection failed for prompt: %s", e)
            continue
    return all_distributions if all_distributions else None


def _compute_kl_divergence(
    baseline_dists: list[dict[str, float]],
    trial_dists: list[dict[str, float]],
) -> float:
    """Compute mean KL-divergence between baseline and trial logprob distributions.

    KL(P||Q) = sum_x P(x) * log(P(x) / Q(x))
    where P = baseline distribution, Q = trial distribution.
    Uses the minimum number of tokens available from both.
    """
    n = min(len(baseline_dists), len(trial_dists))
    if n == 0:
        return 0.0

    kl_values = []
    for i in range(n):
        p_dist = baseline_dists[i]  # baseline
        q_dist = trial_dists[i]  # trial

        # Iterate only over baseline's support — KL(P||Q) sums over x where P(x) > 0
        all_tokens = p_dist.keys()
        kl = 0.0
        for token in all_tokens:
            p_logprob = p_dist.get(token, _KL_FLOOR_LOGPROB)
            q_logprob = q_dist.get(token, _KL_FLOOR_LOGPROB)
            p_prob = math.exp(p_logprob)
            q_prob = math.exp(q_logprob)
            if p_prob > 1e-10 and q_prob > 1e-10:  # noqa: PLR2004
                kl += p_prob * math.log(p_prob / q_prob)
        kl_values.append(max(0.0, kl))  # KL is non-negative

    return sum(kl_values) / len(kl_values) if kl_values else 0.0


def measure_kl_divergence(
    ctx: AppContext, baseline_cache: list[dict[str, float] | None] = None
) -> KLResult:
    """Measure KL-divergence against a cached baseline.

    Args:
        baseline_cache: list of logprob distributions from the baseline run.
                        If None, collects and returns the baseline (no comparison).

    Returns:
        KLResult with distributions and optional kl_divergence score.
        If baseline_cache is None: distributions are populated, kl_divergence is None.
        If baseline_cache provided: both distributions and kl_divergence are populated.
    """
    dists = _collect_logprob_distribution(ctx, KL_DIV_PROMPTS)
    if dists is None:
        return KLResult(distributions=None, kl_divergence=None)

    if baseline_cache is None:
        return KLResult(distributions=dists, kl_divergence=None)

    kl_div = _compute_kl_divergence(baseline_cache, dists)
    return KLResult(distributions=dists, kl_divergence=kl_div)


def kl_quality_factor(kl_div: float | None) -> float:
    """Convert KL-divergence score to a quality factor (0.0 to 1.0).

    KL-div = 0.0 -> 1.0 (identical to baseline)
    KL-div = KL_DIV_THRESHOLD -> 0.85 (soft penalty)
    KL-div = KL_DIV_HARD_FAIL -> 0.1 (hard fail)
    KL-div > KL_DIV_HARD_FAIL -> 0.1 (floor)
    """
    from ._helpers import quality_factor_curve

    return quality_factor_curve(kl_div, KL_DIV_THRESHOLD, KL_DIV_HARD_FAIL)
