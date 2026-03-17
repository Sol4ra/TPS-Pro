"""Quality evaluation: quality gate, KL-divergence, perplexity, NIAH, MCQ."""
import logging
import math
import random
import re
import time

import requests

from .state import ctx

logger = logging.getLogger(__name__)
from .constants import (
    QUALITY_TASKS, QUALITY_GATE_PROMPTS, QUALITY_GATE_SEED,
    QUALITY_GATE_N_PREDICT, QUALITY_GATE_UNCERTAIN_THRESHOLD,
    QUALITY_GATE_TAIL_PCT, QUALITY_GATE_CEILING, QUALITY_GATE_SOFT_PENALTY,
    QUALITY_GATE_CLIFF, QUALITY_GATE_CLIFF_PENALTY,
    QUALITY_WEIGHT_CORRECTNESS, QUALITY_WEIGHT_CONFIDENCE, QUALITY_WEIGHT_EFFICIENCY,
    QUALITY_EVAL_SEED, QUALITY_TTFT_BASELINE,
    KL_DIV_TOP_K, KL_DIV_THRESHOLD, KL_DIV_HARD_FAIL, KL_DIV_PROMPTS,
    _PPL_REFERENCE_TEXT, PPL_DEGRADATION_WARN, PPL_DEGRADATION_FAIL,
    _NIAH_NEEDLES, _NIAH_FILLER_BLOCKS,
)

try:
    import aiohttp
    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False

from .engine import start_server, kill_server, wait_for_server
from .search import save_phase_results, load_phase_results
from .measurement import measure_token_uncertainty

import asyncio


def measure_quality_gate(is_baseline=False):
    """Quality gate using token-level uncertainty comparison against baseline.

    Measures two signals:
      1. Uncertain token count increase (tokens with logprob < -2.0)
      2. Tail-20% logprob degradation (worst 20% of tokens)
    Uses the worse of the two signals to determine the quality factor.

    On baseline run (is_baseline=True): measures and stores baseline metrics.
    On subsequent runs: returns quality_factor (0.1-1.0) based on degradation.
    """

    metrics = measure_token_uncertainty()
    if metrics is None:
        if is_baseline:
            return 1.0
        print(f"  [Q] Quality measurement failed/timed out — applying max penalty")
        return QUALITY_GATE_CLIFF_PENALTY

    if is_baseline or ctx.quality_baseline is None:
        ctx.quality_baseline = metrics
        print(f"  [Q] Baseline: {metrics['uncertain_count']} uncertain tokens "
              f"(of {metrics['total_tokens']}), tail-20% avg: {metrics['tail_avg']:.3f}")
        return 1.0

    # Signal 1: uncertain token count increase
    # When baseline has very few uncertain tokens, use a floor based on total token count
    # to avoid extreme sensitivity (e.g., going from 0→3 out of 1698 shouldn't be a cliff)
    base_uc = ctx.quality_baseline["uncertain_count"]
    uc_floor = max(1, int(ctx.quality_baseline["total_tokens"] * 0.01))  # 1% floor
    base_uc = max(base_uc, uc_floor)
    uc_increase = (metrics["uncertain_count"] - base_uc) / base_uc

    # Signal 2: tail-20% logprob degradation (more negative = worse)
    base_tail = ctx.quality_baseline["tail_avg"]
    if base_tail < 0:
        tail_increase = (base_tail - metrics["tail_avg"]) / abs(base_tail)  # positive = degraded
    else:
        tail_increase = 0.0

    # Use the worse signal
    degradation = max(uc_increase, tail_increase)

    if degradation <= 0:
        quality_factor = 1.0
    elif degradation <= QUALITY_GATE_CEILING:
        # 0% to 15%: gentle slope from 1.0 → 0.85
        penalty_range = 1.0 - QUALITY_GATE_SOFT_PENALTY
        quality_factor = 1.0 - (degradation / QUALITY_GATE_CEILING) * penalty_range
    elif degradation <= QUALITY_GATE_CLIFF:
        # 15% to 30%: steep cliff from 0.85 → 0.1
        t = (degradation - QUALITY_GATE_CEILING) / (QUALITY_GATE_CLIFF - QUALITY_GATE_CEILING)
        quality_factor = QUALITY_GATE_SOFT_PENALTY - t * (QUALITY_GATE_SOFT_PENALTY - QUALITY_GATE_CLIFF_PENALTY)
    else:
        quality_factor = QUALITY_GATE_CLIFF_PENALTY

    print(f"  [Q] Uncertain: {metrics['uncertain_count']} (baseline: {ctx.quality_baseline['uncertain_count']}, "
          f"{uc_increase:+.0%}) | Tail: {metrics['tail_avg']:.3f} (baseline: {ctx.quality_baseline['tail_avg']:.3f}, "
          f"{tail_increase:+.0%}) | factor: {quality_factor:.2f}")
    return quality_factor


def _collect_logprob_distribution(prompts, top_k=KL_DIV_TOP_K):
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
            r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json=payload, timeout=120)
            if r.status_code != 200:
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
                        lp = entry.get("logprob", -100)
                        dist[tok] = lp
                    all_distributions.append(dist)
        except (requests.RequestException, ValueError, KeyError) as e:
            logger.debug("KL distribution collection failed for prompt: %s", e)
            continue
    return all_distributions if all_distributions else None


def _compute_kl_divergence(baseline_dists, trial_dists):
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
        q_dist = trial_dists[i]     # trial

        # Iterate only over baseline's support — KL(P||Q) sums over x where P(x) > 0
        all_tokens = p_dist.keys()
        kl = 0.0
        for token in all_tokens:
            p_logprob = p_dist.get(token, -20.0)  # small floor for missing tokens
            q_logprob = q_dist.get(token, -20.0)
            p_prob = math.exp(p_logprob)
            q_prob = math.exp(q_logprob)
            if p_prob > 1e-10 and q_prob > 1e-10:
                kl += p_prob * math.log(p_prob / q_prob)
        kl_values.append(max(0.0, kl))  # KL is non-negative

    return sum(kl_values) / len(kl_values) if kl_values else 0.0


def measure_kl_divergence(baseline_cache=None):
    """Measure KL-divergence against a cached baseline.

    Args:
        baseline_cache: list of logprob distributions from the baseline run.
                        If None, collects and returns the baseline (no comparison).

    Returns:
        If baseline_cache is None: (distributions, None) — save these as baseline
        If baseline_cache provided: (distributions, kl_div_score)
    """
    dists = _collect_logprob_distribution(KL_DIV_PROMPTS)
    if dists is None:
        return None, None

    if baseline_cache is None:
        return dists, None

    kl_div = _compute_kl_divergence(baseline_cache, dists)
    return dists, kl_div


def kl_quality_factor(kl_div):
    """Convert KL-divergence score to a quality factor (0.0 to 1.0).

    KL-div = 0.0 → 1.0 (identical to baseline)
    KL-div = KL_DIV_THRESHOLD → 0.85 (soft penalty)
    KL-div = KL_DIV_HARD_FAIL → 0.1 (hard fail)
    KL-div > KL_DIV_HARD_FAIL → 0.1 (floor)
    """
    if kl_div is None or kl_div <= 0:
        return 1.0
    if kl_div <= KL_DIV_THRESHOLD:
        return 1.0 - 0.15 * (kl_div / KL_DIV_THRESHOLD)
    if kl_div <= KL_DIV_HARD_FAIL:
        t = (kl_div - KL_DIV_THRESHOLD) / (KL_DIV_HARD_FAIL - KL_DIV_THRESHOLD)
        return 0.85 - t * 0.75  # 0.85 → 0.1
    return 0.1


def measure_true_perplexity(text_chunk=None):
    """Measure the exact Perplexity (PPL) of the model over standardized reference text.

    Uses the /v1/completions endpoint with logprobs to measure how well the model
    predicts each token in the reference text. Lower PPL = better.

    Returns:
        float: Perplexity value, or float('inf') on failure.
    """
    if text_chunk is None:
        text_chunk = _PPL_REFERENCE_TEXT

    # llama-server's completions endpoint supports logprobs on generated tokens.
    # Strategy: feed most of the text as prompt, generate the rest, measure logprobs.
    # Split at ~80% — use first part as context, predict the last ~20%.
    split_point = int(len(text_chunk) * 0.8)
    # Find a clean sentence boundary near the split
    period_pos = text_chunk.rfind(". ", 0, split_point)
    if period_pos > split_point * 0.5:
        split_point = period_pos + 2  # after the period and space

    prompt_part = text_chunk[:split_point]
    continuation = text_chunk[split_point:]

    # Estimate tokens in continuation (~3 chars/token for dense English)
    est_continuation_tokens = max(50, len(continuation) // 3)

    payload = {
        "prompt": prompt_part,
        "max_tokens": est_continuation_tokens + 50,
        "temperature": 0.0,
        "logprobs": 1,
        "n_probs": 1,  # llama-server specific logprobs parameter
    }

    try:
        # Try chat completions first — more reliable with thinking models
        chat_payload = {
            "messages": [
                {"role": "system", "content": "Continue the following text exactly as written. Do not add commentary."},
                {"role": "user", "content": prompt_part},
            ],
            "max_tokens": est_continuation_tokens + 50,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 1,
        }
        r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json=chat_payload, timeout=300)
        if r.status_code == 200:
            data = r.json()
            choice = data.get("choices", [{}])[0]
            logprobs_data = choice.get("logprobs", {})
            content_logprobs = logprobs_data.get("content", []) if logprobs_data else []
            valid_logprobs = [t.get("logprob") for t in content_logprobs if t.get("logprob") is not None]

            if not valid_logprobs or len(valid_logprobs) < 10:
                # Content logprobs empty — model may be thinking. Try completions endpoint.
                r = ctx.http.post(f"{ctx.server_url}/v1/completions", json=payload, timeout=300)
                if r.status_code != 200:
                    print(f"  [PPL] Both endpoints returned no logprobs")
                    return float('inf')
                data = r.json()
                logprobs_data = data.get("choices", [{}])[0].get("logprobs", {})
                token_logprobs = logprobs_data.get("token_logprobs", []) if logprobs_data else []
                valid_logprobs = [lp for lp in token_logprobs if lp is not None]

        else:
            # Chat failed, try completions
            r = ctx.http.post(f"{ctx.server_url}/v1/completions", json=payload, timeout=300)
            if r.status_code != 200:
                print(f"  [PPL] Completions endpoint returned {r.status_code}")
                return float('inf')
            data = r.json()
            logprobs_data = data.get("choices", [{}])[0].get("logprobs", {})
            token_logprobs = logprobs_data.get("token_logprobs", []) if logprobs_data else []
            valid_logprobs = [lp for lp in token_logprobs if lp is not None]

        if not valid_logprobs or len(valid_logprobs) < 10:
            print(f"  [PPL] Too few logprobs ({len(valid_logprobs)})")
            return float('inf')

        # PPL = exp(-1/N * sum(logprobs))
        avg_logprob = sum(valid_logprobs) / len(valid_logprobs)
        perplexity = math.exp(-avg_logprob)
        return perplexity

    except Exception as e:
        print(f"  [PPL] PPL measurement failed: {e}")
        return float('inf')


def ppl_quality_factor(baseline_ppl, trial_ppl):
    """Convert PPL degradation to a quality factor (0.0 to 1.0).

    Args:
        baseline_ppl: PPL from the baseline/reference config.
        trial_ppl: PPL from the trial config.

    Returns:
        float: 1.0 if no degradation, scaled down based on PPL increase.
    """
    if baseline_ppl <= 0 or baseline_ppl == float('inf'):
        return 1.0  # no valid baseline, skip penalty
    if trial_ppl == float('inf'):
        return 0.1  # measurement failed = assume bad

    degradation = (trial_ppl - baseline_ppl) / baseline_ppl  # fraction increase

    if degradation <= 0:
        return 1.0  # trial is same or better
    if degradation <= PPL_DEGRADATION_WARN:
        # 0-10%: gentle slope 1.0 → 0.85
        return 1.0 - 0.15 * (degradation / PPL_DEGRADATION_WARN)
    if degradation <= PPL_DEGRADATION_FAIL:
        # 10-30%: steep cliff 0.85 → 0.1
        t = (degradation - PPL_DEGRADATION_WARN) / (PPL_DEGRADATION_FAIL - PPL_DEGRADATION_WARN)
        return 0.85 - t * 0.75
    return 0.1  # >30% = severely damaged


def _extract_answer_letter(content):
    """Extract the MC answer letter (A/B/C/D) from model response.

    Handles various formats: "(A)", "A)", "A.", "Answer: A", "the answer is A", etc.
    Returns the letter or None if no clear answer found.
    """
    content = content.strip()
    # Try common patterns, most specific first
    patterns = [
        r'(?:answer|choice)\s*(?:is|:)\s*\(?([A-D])\)?',  # "answer is (A)" / "choice: B"
        r'\(?([A-D])\)?\s*$',                               # ends with "(A)" or "A"
        r'^([A-D])\b',                                       # starts with "A"
        r'\(([A-D])\)',                                      # "(A)" anywhere
    ]
    for pattern in patterns:
        m = re.search(pattern, content, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper()
    return None


def _extract_answer_logprob(data, correct_letter):
    """Extract the logprob for the answer token from OpenAI-format response.

    Scans the logprobs for the token matching the correct letter.
    Returns float logprob or None if not found.
    """
    logprobs_data = data.get("choices", [{}])[0].get("logprobs", {})
    content_logprobs = logprobs_data.get("content", []) if logprobs_data else []

    for token_info in content_logprobs:
        token = token_info.get("token", "").strip()
        # Match the answer letter token (could be "A", "(A)", etc.)
        if token.upper() == correct_letter or token.strip("()").upper() == correct_letter:
            return token_info.get("logprob", None)
        # Also check top_logprobs for the correct answer
        for alt in token_info.get("top_logprobs", []):
            alt_token = alt.get("token", "").strip()
            if alt_token.upper() == correct_letter or alt_token.strip("()").upper() == correct_letter:
                return alt.get("logprob", None)

    return None


def measure_quality(sampling_params, tasks=QUALITY_TASKS, target_to_beat=None):
    """3-signal quality eval: Correctness (40%) + Confidence (40%) + Efficiency (20%).

    Each task is a hard multiple-choice question. The model's response is scored on:
      1. Correctness: did it pick the right letter? (binary 0/1)
      2. Confidence: logprob of the correct answer token (higher = more confident)
      3. Efficiency: TTFT in ms (lower = faster, normalized against baseline)

    Uses seed=0 and temperature=0 for deterministic grading so any score change
    is strictly from parameter changes, not model randomness.

    Returns composite score 0-100.
    """
    oai_params = {}
    for k, v in sampling_params.items():
        if k == "n_predict":
            oai_params["max_tokens"] = v
        else:
            oai_params[k] = v
    max_tokens = oai_params.pop("max_tokens", 1024)

    if _HAS_AIOHTTP:
        return _measure_quality_async(tasks, max_tokens, oai_params)

    # Sequential path — pass target_to_beat for early short-circuit
    return _measure_quality_sequential(tasks, max_tokens, oai_params, target_to_beat=target_to_beat)


def _score_quality_results(task_results):
    """Compute composite quality score from individual task results.

    Args:
        task_results: list of dicts with keys: correct (bool), logprob (float|None), ttft_ms (float|None)

    Returns:
        float: composite score 0-100
    """
    if not task_results:
        return 0.0

    n = len(task_results)

    # Signal 1: Correctness (0-1 per task, averaged)
    correctness = sum(1.0 for r in task_results if r["correct"]) / n

    # Signal 2: Confidence (logprob of correct answer, normalized to 0-1)
    # logprob ranges: 0.0 (100% confident) to -inf (no confidence)
    # Map: 0.0 → 1.0, -1.0 → 0.5, -3.0 → 0.1, worse → 0.0
    logprobs = [r["logprob"] for r in task_results if r["logprob"] is not None]
    if logprobs:
        avg_logprob = sum(logprobs) / len(logprobs)
        # Sigmoid-like mapping: confidence = exp(logprob) clamped to [0, 1]
        confidence = min(1.0, max(0.0, math.exp(avg_logprob)))
    else:
        # No logprobs available — fall back to correctness as proxy
        confidence = correctness

    # Signal 3: Efficiency (TTFT normalized against baseline)
    ttfts = [r["ttft_ms"] for r in task_results if r["ttft_ms"] is not None and r["ttft_ms"] > 0]
    if ttfts:
        avg_ttft = sum(ttfts) / len(ttfts)
        # Lower TTFT = better. Normalize: baseline/actual, capped at [0, 1]
        efficiency = min(1.0, QUALITY_TTFT_BASELINE / avg_ttft) if avg_ttft > 0 else 0.0
    else:
        efficiency = 0.5  # neutral if no timing data

    # Weighted composite
    score = (QUALITY_WEIGHT_CORRECTNESS * correctness +
             QUALITY_WEIGHT_CONFIDENCE * confidence +
             QUALITY_WEIGHT_EFFICIENCY * efficiency)

    return score * 100


def _eval_single_task(prompt, correct_letter, category, max_tokens, oai_params):
    """Evaluate a single MC task. Returns dict with correct, logprob, ttft_ms."""
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
        r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json=payload, timeout=300)
        ttft_ms = time.time() * 1000 - start_ms

        if r.status_code == 200:
            data = r.json()
            choices = data.get("choices", [])
            content = choices[0].get("message", {}).get("content", "") if choices else ""

            # Extract answer letter
            answer = _extract_answer_letter(content)
            correct = answer == correct_letter

            # Extract logprob for the correct answer
            logprob = _extract_answer_logprob(data, correct_letter)

            return {"correct": correct, "logprob": logprob, "ttft_ms": ttft_ms,
                    "answer": answer, "category": category}
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.debug("Quality eval request failed: %s", e)

    return {"correct": False, "logprob": None, "ttft_ms": None,
            "answer": None, "category": category}


def _measure_quality_sequential(tasks, max_tokens, oai_params, target_to_beat=None):
    """Sequential quality eval with early exit.

    If target_to_beat is set: after each question, check if the maximum possible
    score (assuming all remaining questions are perfect) can still beat the target.
    If not, bail out early to save time on obviously bad configs.
    """
    results = []
    total_tasks = len(tasks)

    for i, (prompt, correct_letter, category) in enumerate(tasks):
        result = _eval_single_task(prompt, correct_letter, category, max_tokens, oai_params)
        results.append(result)

        # Short-circuit: can we still mathematically beat the target?
        if target_to_beat is not None and i < total_tasks - 1:
            current_correct = sum(1.0 for r in results if r["correct"])
            remaining = total_tasks - (i + 1)
            max_possible_correct = (current_correct + remaining) / total_tasks
            # Even with perfect confidence + efficiency, correctness is 40% of score
            # Use 20% margin to account for confidence/efficiency contributions
            if max_possible_correct * 100 < target_to_beat - 20:
                if ctx.debug:
                    print(f"    [short-circuit] {int(current_correct)}/{i+1} correct, "
                          f"max possible {max_possible_correct:.0%} — bailing ({total_tasks - i - 1} skipped)")
                return _score_quality_results(results)

    return _score_quality_results(results)


def _measure_quality_async(tasks, max_tokens, oai_params):
    """Fire all quality tasks concurrently via aiohttp and return composite score."""

    async def _run():
        async def _eval_one(session, prompt, correct_letter, category):
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
                    f"{ctx.server_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as r:
                    ttft_ms = time.time() * 1000 - start_ms
                    if r.status == 200:
                        data = await r.json()
                        choices = data.get("choices", [])
                        content = choices[0].get("message", {}).get("content", "") if choices else ""
                        answer = _extract_answer_letter(content)
                        correct = answer == correct_letter
                        logprob = _extract_answer_logprob(data, correct_letter)
                        return {"correct": correct, "logprob": logprob, "ttft_ms": ttft_ms,
                                "answer": answer, "category": category}
            except (aiohttp.ClientError, ValueError, KeyError) as e:
                logger.debug("Async quality eval failed: %s", e)
            return {"correct": False, "logprob": None, "ttft_ms": None,
                    "answer": None, "category": category}

        async with aiohttp.ClientSession() as session:
            results = await asyncio.gather(
                *[_eval_one(session, prompt, correct_letter, category)
                  for prompt, correct_letter, category in tasks]
            )
        return _score_quality_results(list(results))

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, _run()).result()
    return asyncio.run(_run())


def _tokenize_count(text):
    """Get exact token count from the server's /tokenize endpoint.

    Falls back to the 3.0 chars/token estimate if the server is unreachable
    or the endpoint is unavailable.

    Returns:
        int: Exact token count, or estimated count as fallback.
    """
    try:
        r = ctx.http.post(f"{ctx.server_url}/tokenize", json={"content": text}, timeout=10)
        if r.status_code == 200:
            tokens = r.json().get("tokens", [])
            return len(tokens)
    except (requests.RequestException, ValueError) as e:
        logger.debug("Tokenization endpoint failed, using estimate: %s", e)
    # Fallback: estimate at 3.0 chars/token
    return max(1, int(len(text) / 3.0))


def _build_niah_prompt(target_tokens, needle_fact, needle_depth_pct=0.25):
    """Build a prompt with a needle (fact) injected at a specific depth.

    Uses the /tokenize endpoint for exact token counting when available,
    falling back to character estimation otherwise.

    Args:
        target_tokens: Approximate total token count for the prompt.
        needle_fact: The fact string to inject.
        needle_depth_pct: Where to inject (0.0=start, 0.5=middle, 1.0=end).

    Returns:
        str: The constructed prompt with needle embedded in filler.
    """
    # Generate ~20% more filler than needed, then trim to exact token count.
    # Overshoot ensures we have enough material to reach the target.
    overshoot_chars = int(target_tokens * 3.6)  # ~20% over at 3 chars/token

    # Dynamic filler: seeded PRNG shuffles blocks with unique prefixes/suffixes
    # so no two sections are identical, preventing attention collapse on repetitive patterns.
    rng = random.Random(target_tokens ^ int(needle_depth_pct * 1000))
    _section_prefixes = [
        "In this section, we discuss", "The following covers", "This part examines",
        "Consider the following", "Next, we explore", "An important aspect is",
        "Here we analyze", "The topic below addresses",
    ]
    _section_suffixes = [
        "This requires careful consideration of edge cases.",
        "Performance benchmarks should validate the approach.",
        "Error handling must be robust and well-tested.",
        "Documentation should cover all public interfaces.",
        "Security implications must be reviewed thoroughly.",
        "Scalability should be evaluated under production load.",
        "Integration tests should cover the critical paths.",
        "Code review should focus on maintainability.",
    ]
    shuffled_blocks = list(_NIAH_FILLER_BLOCKS)
    filler = ""
    block_idx = 0
    while len(filler) < overshoot_chars:
        if block_idx % len(shuffled_blocks) == 0:
            rng.shuffle(shuffled_blocks)
        block = shuffled_blocks[block_idx % len(shuffled_blocks)]
        prefix = _section_prefixes[rng.randint(0, len(_section_prefixes) - 1)]
        suffix = _section_suffixes[rng.randint(0, len(_section_suffixes) - 1)]
        filler += f"\n\nSection {block_idx + 1}: {prefix}:\n{block}\n{suffix}\n"
        block_idx += 1

    # FIX 2: Truncate BEFORE injecting the needle so it never gets sliced off.
    # Use /tokenize for exact sizing — binary trim to hit target_tokens precisely.
    actual_tokens = _tokenize_count(filler)
    if actual_tokens > target_tokens:
        # Binary search for the right character cutoff
        lo, hi = 0, len(filler)
        while lo < hi - 50:  # 50-char granularity is fine
            mid = (lo + hi) // 2
            count = _tokenize_count(filler[:mid])
            if count <= target_tokens:
                lo = mid
            else:
                hi = mid
        filler = filler[:lo]
    elif actual_tokens < target_tokens * 0.9:
        # Fallback: if /tokenize returned a low count, pad more
        target_chars = int(target_tokens * 3.0)
        while len(filler) < target_chars:
            if block_idx % len(shuffled_blocks) == 0:
                rng.shuffle(shuffled_blocks)
            block = shuffled_blocks[block_idx % len(shuffled_blocks)]
            prefix = _section_prefixes[rng.randint(0, len(_section_prefixes) - 1)]
            suffix = _section_suffixes[rng.randint(0, len(_section_suffixes) - 1)]
            filler += f"\n\nSection {block_idx + 1}: {prefix}:\n{block}\n{suffix}\n"
            block_idx += 1
        filler = filler[:target_chars]

    # Inject needle at the specified depth
    inject_pos = int(len(filler) * needle_depth_pct)
    # Find a clean paragraph break near the injection point
    newline_pos = filler.rfind("\n\n", 0, inject_pos)
    if newline_pos == -1:
        newline_pos = inject_pos
    filler = filler[:newline_pos] + f"\n\nIMPORTANT NOTE: {needle_fact}\n\n" + filler[newline_pos:]

    return filler


def niah_test(kv_cache_type, base_config, depths=None, context_sizes=None):
    """Run Needle-in-a-Haystack test for a given KV cache quantization level.

    Tests fact retrieval at multiple context depths to find the breaking point
    where KV cache quantization destroys the model's attention mechanism.

    Args:
        kv_cache_type: KV cache type to test (e.g., "f16", "q8_0", "q4_0").
        base_config: Base server config dict (compute + memory params).
        depths: List of depth percentages to test (default: [0.1, 0.25, 0.5, 0.75, 0.9]).
        context_sizes: List of context sizes to test (default: [2048, 8192, 16384, 32768]).

    Returns:
        dict: {"kv_type": str, "results": [{context, depth, passed, needle_idx}], "pass_rate": float}
    """
    if depths is None:
        depths = [0.10, 0.25, 0.50, 0.75, 0.90]
    if context_sizes is None:
        context_sizes = [16384, 65536]

    # FIX 3: Strip speculative decoding params to prevent N-gram cache poisoning.
    # The repetitive filler text poisons the N-gram cache, causing the draft model
    # to hallucinate repetitive blocks instead of attending to the needle.
    config = {k: v for k, v in base_config.items()
              if not k.startswith("spec_") and not k.startswith("draft_")
              and k != "lookup_cache_dynamic"}
    config["kv_cache_type"] = kv_cache_type
    config["flash_attn"] = "on"

    # FIX 4: Give 50% extra context padding. With 3 chars/token ratio the prompt
    # is close to target_tokens but the query + template add overhead.
    max_ctx = max(context_sizes)
    config["context"] = int(max_ctx * 1.5)

    print(f"\n  [NIAH] Testing KV cache type: {kv_cache_type}")
    print(f"         Context sizes: {context_sizes}")
    print(f"         Depths: {[f'{d:.0%}' for d in depths]}")
    print(f"         Server context: {config['context']} (1.5x padding)")

    kill_server()
    proc = start_server(config)
    status = wait_for_server(proc=proc)
    if status == "oom":
        print(f"  [NIAH] OOM with {kv_cache_type} at ctx={config['context']} — skipping")
        kill_server()
        return {"kv_type": kv_cache_type, "results": [], "pass_rate": 0.0, "oom": True}
    elif status != "ok":
        print(f"  [NIAH] Server failed to start — skipping")
        kill_server()
        return {"kv_type": kv_cache_type, "results": [], "pass_rate": 0.0, "error": status}

    results = []
    total_tests = 0
    total_passed = 0

    for ctx_size in context_sizes:
        for depth in depths:
            needle = _NIAH_NEEDLES[total_tests % len(_NIAH_NEEDLES)]
            prompt = _build_niah_prompt(
                target_tokens=ctx_size,
                needle_fact=needle["fact"],
                needle_depth_pct=depth,
            )

            full_prompt = (
                f"{prompt}\n\n"
                f"Question: {needle['query']}\n"
                f"Answer with ONLY the exact answer, nothing else."
            )

            try:
                r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json={
                    "messages": [
                        {"role": "system", "content": "You are a fact extraction assistant. Reply with only the answer."},
                        {"role": "user", "content": full_prompt},
                    ],
                    "max_tokens": 2048,
                    "temperature": 0.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repeat_penalty": 1.0,
                    "top_p": 1.0,
                }, timeout=600)

                passed = False
                if r.status_code == 200:
                    choices = r.json().get("choices", [])
                    msg = choices[0].get("message", {}) if choices else {}
                    content = msg.get("content", "") or ""
                    reasoning = msg.get("reasoning_content", "") or ""
                    full_text = content + " " + reasoning
                    content_clean = re.sub(r"<think>.*?(</think>|$)", "", full_text, flags=re.DOTALL).strip()
                    passed = needle["expected"].lower() in content_clean.lower()
                else:
                    print(f"      [!] HTTP {r.status_code}")

                results.append({
                    "context": ctx_size,
                    "depth": depth,
                    "passed": passed,
                    "needle_idx": total_tests % len(_NIAH_NEEDLES),
                })
                total_tests += 1
                if passed:
                    total_passed += 1

                status_icon = "PASS" if passed else "FAIL"
                print(f"    ctx={ctx_size:>6} depth={depth:.0%}: {status_icon}")

            except Exception as e:
                results.append({"context": ctx_size, "depth": depth, "passed": False, "error": str(e)})
                total_tests += 1
                print(f"    ctx={ctx_size:>6} depth={depth:.0%}: ERROR ({e})")

    pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0.0
    print(f"\n  [NIAH] {kv_cache_type}: {total_passed}/{total_tests} passed ({pass_rate:.0f}%)")

    return {"kv_type": kv_cache_type, "results": results, "pass_rate": pass_rate}


def phase_niah(base_config=None):
    """NIAH Phase: Test KV cache quantization levels for long-context recall.

    Runs after Memory phase. Tests each KV cache type (f16, q8_0, q5_1, q4_0)
    with needle-in-a-haystack at increasing context depths to find the
    breaking point where quantization destroys attention recall.

    Results are saved and used to warn users about unsafe KV quant levels.
    """
    print("\n" + "=" * 60)
    print("  Needle-in-a-Haystack — KV Cache Quality Validation")
    print("=" * 60)

    # Check for existing results
    existing = load_phase_results("niah")
    if existing:
        print(f"\n[*] NIAH already complete. Results:")
        for r in existing.get("kv_results", []):
            print(f"    {r['kv_type']:>6}: {r['pass_rate']:.0f}% recall")
        return existing

    if base_config is None:
        base_config = dict(ctx.naked_engine)
        # Load best compute config
        compute_src = load_phase_results("compute_audit") or load_phase_results("compute")
        if compute_src:
            base_config.update(compute_src["best_params"])
        # Load best MoE config
        moe_src = load_phase_results("moe_combined") or load_phase_results("moe")
        if moe_src and "best_params" in moe_src:
            base_config.update(moe_src["best_params"])

    # Test all KV cache types — f16 is the reference (should be 100%)
    kv_types = ["f16", "q8_0", "q5_1", "q4_0"]
    kv_results = []

    # Measure PPL for each KV type alongside NIAH recall.
    # The NIAH server stays up after testing, so we measure PPL before killing it.
    for kv_type in kv_types:
        result = niah_test(kv_type, base_config)
        # Measure PPL while server is still running with this KV type
        if not result.get("oom") and not result.get("error"):
            ppl = measure_true_perplexity()
            result["ppl"] = round(ppl, 2) if ppl != float('inf') else None
            if ppl != float('inf'):
                print(f"  [PPL] {kv_type}: {ppl:.2f}")
        kv_results.append(result)

    kill_server()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  NIAH + PERPLEXITY — RESULTS")
    print(f"{'=' * 60}")

    f16_rate = next((r["pass_rate"] for r in kv_results if r["kv_type"] == "f16"), 100.0)
    f16_ppl = next((r.get("ppl") for r in kv_results if r["kv_type"] == "f16" and r.get("ppl")), None)

    for r in kv_results:
        kv = r["kv_type"]
        rate = r["pass_rate"]
        delta = rate - f16_rate

        if r.get("oom"):
            status = "OOM"
        elif rate >= f16_rate - 5:
            status = "SAFE"
        elif rate >= f16_rate - 20:
            status = "DEGRADED"
        else:
            status = "BROKEN"

        # PPL info
        ppl_str = ""
        if r.get("ppl") is not None and f16_ppl is not None:
            ppl_pct = (r["ppl"] - f16_ppl) / f16_ppl * 100 if f16_ppl > 0 else 0
            ppl_str = f"  PPL: {r['ppl']:.2f} ({ppl_pct:+.1f}%)"
            # Override status if PPL shows severe degradation
            if ppl_pct > PPL_DEGRADATION_FAIL * 100 and status == "SAFE":
                status = "DEGRADED (PPL)"
        elif r.get("ppl") is not None and kv == "f16":
            ppl_str = f"  PPL: {r['ppl']:.2f} (ref)"

        print(f"  {kv:>6}: {rate:5.0f}% recall  [{status}]" +
              (f"  ({delta:+.0f}% vs f16)" if kv != "f16" else "  (reference)") +
              ppl_str)

    # Save results
    results = {
        "phase": "niah",
        "kv_results": kv_results,
        "reference_kv": "f16",
        "reference_pass_rate": f16_rate,
        "reference_ppl": f16_ppl,
    }
    save_phase_results("niah", results)

    return results


def phase_reasoning_eval(n_tasks=5):
    """Evaluate model reasoning quality. Purely informational."""
    print("\n" + "=" * 60)
    print("  Reasoning Evaluation")
    print("=" * 60)

    tasks = [
        ("What is the derivative of x^3 * ln(x)?", "3x^2", ["3x^2 ln(x)", "x^2(3ln(x)+1)", "3x²"]),
        ("A train travels 60mph for 2 hours, then 90mph for 1 hour. What is the average speed?", "70", ["70"]),
        ("If all roses are flowers and some flowers fade quickly, can we conclude all roses fade quickly?", "no", ["no", "cannot conclude", "not necessarily"]),
        ("What is 17 * 23?", "391", ["391"]),
        ("How many r's are in 'strawberry'?", "3", ["3", "three"]),
    ]

    correct = 0
    for prompt, _, accept in tasks[:n_tasks]:
        try:
            r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256, "temperature": 0.0,
            }, timeout=120)
            if r.status_code == 200:
                content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                if any(a.lower() in content.lower() for a in accept):
                    correct += 1
                    print(f"  [\u2713] {prompt[:60]}...")
                else:
                    print(f"  [\u2717] {prompt[:60]}... \u2192 {content[:80]}")
            else:
                print(f"  [!] Request failed: {r.status_code}")
        except Exception as e:
            print(f"  [!] Error: {e}")

    score = (correct / min(n_tasks, len(tasks))) * 100
    print(f"\n  Reasoning score: {score:.0f}% ({correct}/{min(n_tasks, len(tasks))})")
    return score


def phase_integrity_eval(n_tasks=5):
    """Evaluate output integrity under optimized settings."""
    print("\n" + "=" * 60)
    print("  Integrity Evaluation")
    print("=" * 60)

    tasks = [
        {"prompt": "Write a Python function that returns the factorial of n. Only output the code, no explanation.",
         "check": lambda c: "def " in c and "return" in c},
        {"prompt": "What is 2^10?", "check": lambda c: "1024" in c},
        {"prompt": "List exactly 5 colors, one per line, no numbering.",
         "check": lambda c: len([l for l in c.strip().split("\n") if l.strip()]) >= 4},
        {"prompt": "Respond with ONLY the word 'hello' and nothing else.",
         "check": lambda c: "hello" in c.lower() and len(c.strip()) < 20},
        {"prompt": "What is the capital of France?", "check": lambda c: "paris" in c.lower()},
    ]

    passed = 0
    for task in tasks[:n_tasks]:
        try:
            r = ctx.http.post(f"{ctx.server_url}/v1/chat/completions", json={
                "messages": [{"role": "user", "content": task["prompt"]}],
                "max_tokens": 256, "temperature": 0.0,
            }, timeout=120)
            if r.status_code == 200:
                content = r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                if task["check"](content):
                    passed += 1
                    print(f"  [\u2713] {task['prompt'][:60]}...")
                else:
                    print(f"  [\u2717] {task['prompt'][:60]}... \u2192 {content[:80]}")
        except Exception as e:
            print(f"  [!] Error: {e}")

    score = (passed / min(n_tasks, len(tasks))) * 100
    print(f"\n  Integrity score: {score:.0f}% ({passed}/{min(n_tasks, len(tasks))})")
    return score
