"""Needle-in-a-Haystack (NIAH) testing for KV cache quality validation."""

from __future__ import annotations

import dataclasses
import logging
import random
import re
import threading
from typing import cast

import requests  # type: ignore[import-untyped]

from ..constants import (
    HTTP_OK,
    NIAH_FILLER_BLOCKS,
    NIAH_NEEDLES,
    NIAH_REQUEST_TIMEOUT,
    PPL_DEGRADATION_FAIL,
    SERVER_PROBE_TIMEOUT,
)
from ..engine import kill_server, start_server, wait_for_server
from ..result_types import EngineConfig, NIAHResult, NIAHTestResult, PhaseReturnDict
from ..search import load_phase_results, save_phase_results
from ..state import AppContext
from .perplexity import measure_true_perplexity

logger = logging.getLogger(__name__)

__all__ = [
    "TokenizeCache",
    "tokenize_count",
    "build_niah_prompt",
    "niah_test",
    "phase_niah",
]


# ---------------------------------------------------------------------------
# Tokenize cache (replaces mutable module-level _tokenize_state)
# ---------------------------------------------------------------------------


class TokenizeCache:
    """Thread-safe cache for the chars-per-token ratio.

    Passed explicitly to ``tokenize_count`` so there is no mutable
    module-level state.
    """

    __slots__ = ("ratio", "_lock")

    def __init__(self) -> None:
        self.ratio: float | None = None
        self._lock = threading.Lock()

    def get(self) -> float | None:
        with self._lock:
            return self.ratio

    def set(self, value: float) -> None:
        with self._lock:
            self.ratio = value


def tokenize_count(ctx: AppContext, text: str, cache: TokenizeCache) -> int:
    """Get token count from the server's /tokenize endpoint.

    On the first call, measures the exact char/token ratio via /tokenize.
    Subsequent calls use that ratio as a fast heuristic instead of hitting
    the HTTP endpoint every time (saves ~15 round trips during NIAH binary search).

    Falls back to the 3.0 chars/token estimate if the server is unreachable.
    """
    cached_ratio = cache.get()
    if cached_ratio is not None:
        return max(1, int(len(text) / cached_ratio))

    # First call: get exact count and calibrate the ratio
    try:
        r = ctx.http.post(
            f"{ctx.server_url}/tokenize",
            json={"content": text},
            timeout=SERVER_PROBE_TIMEOUT,
        )
        if r.status_code == HTTP_OK:
            tokens = r.json().get("tokens", [])
            count = len(tokens)
            if count > 0:
                cache.set(len(text) / count)
            return count
    except (requests.RequestException, ValueError) as e:
        logger.debug("Tokenization endpoint failed, using estimate: %s", e)
    # Fallback: estimate at 3.0 chars/token
    cache.set(3.0)
    return max(1, int(len(text) / 3.0))


# ---------------------------------------------------------------------------
# Filler generation helpers for build_niah_prompt
# ---------------------------------------------------------------------------

_SECTION_PREFIXES = [
    "In this section, we discuss",
    "The following covers",
    "This part examines",
    "Consider the following",
    "Next, we explore",
    "An important aspect is",
    "Here we analyze",
    "The topic below addresses",
]

_SECTION_SUFFIXES = [
    "This requires careful consideration of edge cases.",
    "Performance benchmarks should validate the approach.",
    "Error handling must be robust and well-tested.",
    "Documentation should cover all public interfaces.",
    "Security implications must be reviewed thoroughly.",
    "Scalability should be evaluated under production load.",
    "Integration tests should cover the critical paths.",
    "Code review should focus on maintainability.",
]


def _generate_filler(
    rng: random.Random,
    min_chars: int,
) -> tuple[str, int]:
    """Generate shuffled filler text of at least *min_chars* characters.

    Returns (filler_text, next_block_idx) for potential further padding.
    """
    shuffled_blocks = list(NIAH_FILLER_BLOCKS)
    parts: list[str] = []
    block_idx = 0
    while sum(len(p) for p in parts) < min_chars:
        if block_idx % len(shuffled_blocks) == 0:
            rng.shuffle(shuffled_blocks)
        block = shuffled_blocks[block_idx % len(shuffled_blocks)]
        prefix = _SECTION_PREFIXES[rng.randint(0, len(_SECTION_PREFIXES) - 1)]
        suffix = _SECTION_SUFFIXES[rng.randint(0, len(_SECTION_SUFFIXES) - 1)]
        parts.append(f"\n\nSection {block_idx + 1}: {prefix}:\n{block}\n{suffix}\n")
        block_idx += 1
    return "".join(parts), block_idx


def _trim_filler_to_tokens(  # noqa: PLR0913
    ctx: AppContext,
    filler: str,
    target_tokens: int,
    cache: TokenizeCache,
    block_idx: int,
    rng: random.Random,
) -> str:
    """Trim or pad *filler* so it is close to *target_tokens* tokens."""
    actual_tokens = tokenize_count(ctx, filler, cache)
    if actual_tokens > target_tokens:
        lo, hi = 0, len(filler)
        while lo < hi - 50:
            mid = (lo + hi) // 2
            count = tokenize_count(ctx, filler[:mid], cache)
            if count <= target_tokens:
                lo = mid
            else:
                hi = mid
        return filler[:lo]

    if actual_tokens < target_tokens * 0.9:
        target_chars = int(target_tokens * 3.0)
        shuffled_blocks = list(NIAH_FILLER_BLOCKS)
        extra_parts = [filler]
        while sum(len(p) for p in extra_parts) < target_chars:
            if block_idx % len(shuffled_blocks) == 0:
                rng.shuffle(shuffled_blocks)
            block = shuffled_blocks[block_idx % len(shuffled_blocks)]
            prefix = _SECTION_PREFIXES[rng.randint(0, len(_SECTION_PREFIXES) - 1)]
            suffix = _SECTION_SUFFIXES[rng.randint(0, len(_SECTION_SUFFIXES) - 1)]
            extra_parts.append(
                f"\n\nSection {block_idx + 1}: {prefix}:\n{block}\n{suffix}\n"
            )
            block_idx += 1
        return "".join(extra_parts)[:target_chars]

    return filler


def _inject_needle(filler: str, needle_fact: str, depth_pct: float) -> str:
    """Insert *needle_fact* at a paragraph break near *depth_pct* of *filler*."""
    inject_pos = int(len(filler) * depth_pct)
    newline_pos = filler.rfind("\n\n", 0, inject_pos)
    if newline_pos == -1:
        newline_pos = inject_pos
    return (
        filler[:newline_pos]
        + f"\n\nIMPORTANT NOTE: {needle_fact}\n\n"
        + filler[newline_pos:]
    )


# ---------------------------------------------------------------------------
# Public: build_niah_prompt
# ---------------------------------------------------------------------------


def build_niah_prompt(
    ctx: AppContext,
    target_tokens: int,
    needle_fact: str,
    needle_depth_pct: float = 0.25,
    cache: TokenizeCache | None = None,
) -> str:
    """Build a prompt with a needle (fact) injected at a specific depth.

    Args:
        target_tokens: Approximate total token count for the prompt.
        needle_fact: The fact string to inject.
        needle_depth_pct: Where to inject (0.0=start, 0.5=middle, 1.0=end).
        cache: Optional TokenizeCache; a fresh one is created if omitted.

    Returns:
        The constructed prompt with needle embedded in filler.
    """
    if cache is None:
        cache = TokenizeCache()

    overshoot_chars = int(target_tokens * 3.6)
    rng = random.Random(target_tokens ^ int(needle_depth_pct * 1000))
    filler, block_idx = _generate_filler(rng, overshoot_chars)
    filler = _trim_filler_to_tokens(ctx, filler, target_tokens, cache, block_idx, rng)
    return _inject_needle(filler, needle_fact, needle_depth_pct)


# ---------------------------------------------------------------------------
# HTTP request + result parsing helpers for niah_test
# ---------------------------------------------------------------------------


def _send_niah_request(ctx: AppContext, full_prompt: str) -> requests.Response | None:
    """Send a chat completion request for NIAH evaluation.

    Returns Response object, or None on request failure.
    """
    try:
        return ctx.http.post(
            f"{ctx.server_url}/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a fact extraction"
                            " assistant. Reply with only"
                            " the answer."
                        ),
                    },
                    {"role": "user", "content": full_prompt},
                ],
                "max_tokens": 2048,
                "temperature": 0.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repeat_penalty": 1.0,
                "top_p": 1.0,
            },
            timeout=NIAH_REQUEST_TIMEOUT,
        )
    except (requests.RequestException, ValueError) as e:
        logger.error("NIAH request failed: %s", e)
        return None


def _parse_niah_response(response: requests.Response, expected: str) -> bool:
    """Check whether the model's response contains the expected answer."""
    if response.status_code != HTTP_OK:
        logger.warning("NIAH HTTP %s", response.status_code)
        return False
    choices = response.json().get("choices", [])
    msg = choices[0].get("message", {}) if choices else {}
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    full_text = content + " " + reasoning
    content_clean = re.sub(
        r"<think>.*?(</think>|$)", "", full_text, flags=re.DOTALL
    ).strip()
    return expected.lower() in content_clean.lower()


def _run_single_niah_probe(  # noqa: PLR0913
    ctx: AppContext,
    ctx_size: int,
    depth: float,
    needle: dict,
    test_index: int,
    cache: TokenizeCache,
) -> tuple[NIAHTestResult, bool]:
    """Execute one needle probe at a given context size and depth."""
    prompt = build_niah_prompt(
        ctx,
        target_tokens=ctx_size,
        needle_fact=needle["fact"],
        needle_depth_pct=depth,
        cache=cache,
    )
    full_prompt = (
        f"{prompt}\n\n"
        f"Question: {needle['query']}\n"
        f"Answer with ONLY the exact answer, nothing else."
    )

    response = _send_niah_request(ctx, full_prompt)
    if response is None:
        result = NIAHTestResult(
            context=ctx_size, depth=depth, passed=False, error="request_failed"
        )
        logger.error(
            "  ctx=%6d depth=%.0f%%: ERROR (request failed)", ctx_size, depth * 100
        )
        return result, False

    try:
        passed = _parse_niah_response(response, needle["expected"])
    except (ValueError, KeyError) as e:
        result = NIAHTestResult(
            context=ctx_size, depth=depth, passed=False, error=str(e)
        )
        logger.error("  ctx=%6d depth=%.0f%%: ERROR (%s)", ctx_size, depth * 100, e)
        return result, False

    status_icon = "PASS" if passed else "FAIL"
    logger.info("  ctx=%6d depth=%.0f%%: %s", ctx_size, depth * 100, status_icon)
    result = NIAHTestResult(
        context=ctx_size,
        depth=depth,
        passed=passed,
        needle_idx=test_index % len(NIAH_NEEDLES),
    )
    return result, passed


# ---------------------------------------------------------------------------
# Public: niah_test
# ---------------------------------------------------------------------------


_NIAH_STRIP_PREFIXES = ("spec_", "draft_")
_NIAH_STRIP_KEYS = frozenset({"lookup_cache_dynamic"})


def _build_niah_config(
    base_config: EngineConfig,
    kv_cache_type: str,
    context_sizes: list[int],
) -> EngineConfig:
    """Build a server config for NIAH testing by stripping speculative params.

    Speculative decoding params are stripped because repetitive filler text
    poisons the N-gram cache, causing the draft model to hallucinate.
    """
    config: EngineConfig = cast(
        EngineConfig,
        {
            k: v
            for k, v in base_config.items()
            if not any(k.startswith(p) for p in _NIAH_STRIP_PREFIXES)
            and k not in _NIAH_STRIP_KEYS
        },
    )
    config["kv_cache_type"] = kv_cache_type
    config["flash_attn"] = "on"
    config["context"] = int(max(context_sizes) * 1.5)
    return config


def _run_niah_probes(
    ctx: AppContext,
    context_sizes: list[int],
    depths: list[float],
) -> tuple[list[NIAHTestResult], int, int]:
    """Execute all needle probes across context sizes and depths.

    Returns:
        (results, total_tests, total_passed)
    """
    cache = TokenizeCache()
    results: list[NIAHTestResult] = []
    total_tests = 0
    total_passed = 0

    for ctx_size in context_sizes:
        for depth in depths:
            needle = NIAH_NEEDLES[total_tests % len(NIAH_NEEDLES)]
            result, passed = _run_single_niah_probe(
                ctx, ctx_size, depth, needle, total_tests, cache
            )
            results.append(result)
            total_tests += 1
            if passed:
                total_passed += 1

    return results, total_tests, total_passed


def niah_test(
    ctx: AppContext,
    kv_cache_type: str,
    base_config: EngineConfig,
    depths: list[float] | None = None,
    context_sizes: list[int] | None = None,
) -> NIAHResult:
    """Run Needle-in-a-Haystack test for a given KV cache quantization level.

    Tests fact retrieval at multiple context depths to find the breaking point
    where KV cache quantization destroys the model's attention mechanism.

    Args:
        kv_cache_type: KV cache type to test (e.g., "f16", "q8_0", "q4_0").
        base_config: Base server config dict (compute + memory params).
        depths: List of depth percentages to test.
        context_sizes: List of context sizes to test.

    Returns:
        NIAHResult with kv_type, results list, pass_rate, and optional oom/error fields.
    """
    if depths is None:
        depths = [0.10, 0.25, 0.50, 0.75, 0.90]
    if context_sizes is None:
        context_sizes = [16384, 65536]

    config = _build_niah_config(base_config, kv_cache_type, context_sizes)

    logger.info("Testing KV cache type: %s", kv_cache_type)
    logger.info("Context sizes: %s", context_sizes)
    logger.info("Depths: %s", ["{:.0%}".format(d) for d in depths])
    logger.info("Server context: %s (1.5x padding)", config["context"])

    kill_server(ctx)
    proc = start_server(ctx, config)
    status = wait_for_server(ctx, proc=proc)
    if status == "oom":
        logger.warning(
            "NIAH OOM with %s at ctx=%s — skipping", kv_cache_type, config["context"]
        )
        kill_server(ctx)
        return NIAHResult(kv_type=kv_cache_type, results=[], pass_rate=0.0, oom=True)
    if status != "ok":
        logger.warning("NIAH server failed to start — skipping")
        kill_server(ctx)
        return NIAHResult(
            kv_type=kv_cache_type, results=[], pass_rate=0.0, error=status
        )

    results, total_tests, total_passed = _run_niah_probes(ctx, context_sizes, depths)

    pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0.0
    logger.info(
        "NIAH %s: %d/%d passed (%.0f%%)",
        kv_cache_type,
        total_passed,
        total_tests,
        pass_rate,
    )

    return NIAHResult(kv_type=kv_cache_type, results=results, pass_rate=pass_rate)


# ---------------------------------------------------------------------------
# phase_niah helpers
# ---------------------------------------------------------------------------


def _run_kv_sweep(
    ctx: AppContext,
    kv_types: list[str],
    base_config: EngineConfig,
) -> list[NIAHResult]:
    """Run NIAH + PPL measurement for each KV type."""
    kv_results: list[NIAHResult] = []
    for kv_type in kv_types:
        result = niah_test(ctx, kv_type, base_config)
        if not result.oom and not result.error:
            ppl = measure_true_perplexity(ctx)
            ppl_rounded = round(ppl, 2) if ppl != float("inf") else None
            result = dataclasses.replace(result, ppl=ppl_rounded)
            if ppl != float("inf"):
                logger.info("PPL %s: %.2f", kv_type, ppl)
        kv_results.append(result)
    return kv_results


def _classify_kv_status(result: NIAHResult, f16_rate: float) -> str:
    """Classify a KV result's quality status relative to f16 baseline."""
    if result.oom:
        return "OOM"
    if result.pass_rate >= f16_rate - 5:
        return "SAFE"
    if result.pass_rate >= f16_rate - 20:
        return "DEGRADED"
    return "BROKEN"


def _format_ppl_info(
    result: NIAHResult, f16_ppl: float | None, status: str
) -> tuple[str, str]:
    """Format PPL string and potentially update status based on PPL degradation.

    Returns:
        (ppl_str, possibly_updated_status)
    """
    if result.ppl is not None and f16_ppl is not None:
        ppl_pct = (result.ppl - f16_ppl) / f16_ppl * 100 if f16_ppl > 0 else 0
        ppl_str = f"  PPL: {result.ppl:.2f} ({ppl_pct:+.1f}%)"
        if ppl_pct > PPL_DEGRADATION_FAIL * 100 and status == "SAFE":
            return ppl_str, "DEGRADED (PPL)"
        return ppl_str, status
    if result.ppl is not None and result.kv_type == "f16":
        return f"  PPL: {result.ppl:.2f} (ref)", status
    return "", status


def _log_niah_summary(kv_results: list[NIAHResult]) -> tuple[float, float | None]:
    """Log the NIAH + PPL results summary.

    Returns (f16_rate, f16_ppl) reference values.
    """
    logger.info("=" * 60)
    logger.info("NIAH + PERPLEXITY — RESULTS")
    logger.info("=" * 60)

    f16_rate = next((r.pass_rate for r in kv_results if r.kv_type == "f16"), 100.0)
    f16_ppl = next((r.ppl for r in kv_results if r.kv_type == "f16" and r.ppl), None)

    for r in kv_results:
        status = _classify_kv_status(r, f16_rate)
        ppl_str, status = _format_ppl_info(r, f16_ppl, status)
        delta = r.pass_rate - f16_rate
        delta_str = (
            "  (%+.0f%% vs f16)" % delta if r.kv_type != "f16" else "  (reference)"
        )
        logger.info(
            "  %6s: %5.0f%% recall  [%s]%s%s",
            r.kv_type,
            r.pass_rate,
            status,
            delta_str,
            ppl_str,
        )

    return f16_rate, f16_ppl


# ---------------------------------------------------------------------------
# Public: phase_niah
# ---------------------------------------------------------------------------


def phase_niah(
    ctx: AppContext, base_config: EngineConfig | None = None
) -> PhaseReturnDict | None:
    """NIAH Phase: Test KV cache quantization levels for long-context recall.

    Runs after Memory phase. Tests each KV cache type (f16, q8_0, q5_1, q4_0)
    with needle-in-a-haystack at increasing context depths to find the
    breaking point where quantization destroys attention recall.

    Results are saved and used to warn users about unsafe KV quant levels.
    """
    logger.info("=" * 60)
    logger.info("Needle-in-a-Haystack — KV Cache Quality Validation")
    logger.info("=" * 60)

    # Check for existing results
    existing = load_phase_results(ctx, "niah")
    if existing:
        logger.info("NIAH already complete. Results:")
        for r in existing.get("kv_results", []):
            logger.info("  %6s: %.0f%% recall", r["kv_type"], r["pass_rate"])
        return PhaseReturnDict(
            best_params=existing.get("best_params", {}), phase_name="niah"
        )

    if base_config is None:
        base_config = cast(EngineConfig, dict(ctx.naked_engine))
        compute_src = load_phase_results(ctx, "compute_audit") or load_phase_results(
            ctx, "compute"
        )
        if compute_src:
            cast(dict, base_config).update(compute_src["best_params"])
        moe_src = load_phase_results(ctx, "moe_combined") or load_phase_results(
            ctx, "moe"
        )
        if moe_src and "best_params" in moe_src:
            cast(dict, base_config).update(moe_src["best_params"])

    kv_types = ["f16", "q8_0", "q5_1", "q4_0"]
    kv_results = _run_kv_sweep(ctx, kv_types, base_config)
    kill_server(ctx)

    f16_rate, f16_ppl = _log_niah_summary(kv_results)

    results = {
        "phase": "niah",
        "kv_results": [r.to_dict() for r in kv_results],
        "reference_kv": "f16",
        "reference_pass_rate": f16_rate,
        "reference_ppl": f16_ppl,
    }
    save_phase_results(ctx, "niah", results)

    return PhaseReturnDict(best_params={}, phase_name="niah")
