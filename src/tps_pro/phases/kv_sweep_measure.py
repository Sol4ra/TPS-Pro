"""KV Sweep quality testing and measurement.

Extracted from kv_context_sweep.py for module size management.
Contains prompt generation, quality testing, VRAM reading, and scoring.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import re
from typing import Any

import requests

from ..constants import DEFAULT_CONTEXT_SIZE, HTTP_OK
from ..engine import (
    boot_server_with_jinja_recovery,
    kill_server,
)
from ..state import AppContext
from .kv_sweep_boot import generate_test_points

logger = logging.getLogger(__name__)

__all__ = [
    "prepare_test_prompts",
    "measure_single_kv_type",
    "score_measurements",
    "log_sweep_results",
]

# Maximum prompt-processing time (TTFT) in seconds before declaring PP timeout.
_PP_TIMEOUT_S = 120

# Unique values that are unambiguous and won't appear in filler text
_VAR_VALUES = [
    ("ALPHA-7749", "7749"),
    ("BETA-3182", "3182"),
    ("GAMMA-5501", "5501"),
    ("DELTA-9267", "9267"),
    ("EPSILON-4038", "4038"),
]

_VAR_NAME = "PROJECT_CIPHER"


def _count_tokens(ctx: AppContext, text: str) -> int:
    """Count tokens via /tokenize endpoint.

    Falls back to char estimate.
    """
    try:
        r = ctx.http.post(
            f"{ctx.server_url}/tokenize",
            json={"content": text},
            timeout=10,
        )
        if r.status_code == HTTP_OK:
            return len(r.json().get("tokens", []))
    except (requests.RequestException, ValueError, KeyError) as e:
        logger.debug("Tokenize fallback (char estimate): %s", e)
    return len(text) // 4


def _generate_confusable_filler(rng: random.Random, n_codes: int) -> str:
    """Generate a wall of ALPHA-XXXX codes as filler.

    All codes are in the 7700-7799 range to be maximally confusable with
    the needle value ALPHA-7749.
    """
    codes = [f"ALPHA-77{i:02d}" for i in range(100)]
    parts = []
    for _ in range(n_codes):
        parts.append(rng.choice(codes))
    return " ".join(parts)


def _build_variable_tracking_prompt(
    ctx: AppContext | None,
    target_tokens: int,
    seed: int = 42,
) -> tuple[str, str]:
    """Build a variable-tracking prompt filled with confusable ALPHA-XXXX codes.

    Returns (prompt_text, expected_answer).
    """
    rng = random.Random(seed)
    needle = _VAR_VALUES[0][0]  # ALPHA-7749

    question = (
        f"The variable {_VAR_NAME} was assigned multiple"
        f" values throughout the text above. "
        f"What was the VERY FIRST value it was set to? "
        f"Reply with ONLY the value code"
        f" (e.g. ALPHA-7749), nothing else."
    )

    assignments = [
        (
            0.05,
            f"IMPORTANT NOTICE: The value of {_VAR_NAME}"
            f" has been set to {needle}."
            f" Record this carefully.",
        ),
        (
            0.30,
            f"UPDATE: {_VAR_NAME} is now changed to"
            f" {_VAR_VALUES[1][0]}. The previous value"
            f" is no longer current.",
        ),
        (
            0.55,
            f"REVISION: {_VAR_NAME} has been updated to"
            f" {_VAR_VALUES[2][0]}. All prior values"
            f" are superseded.",
        ),
        (
            0.80,
            f"FINAL UPDATE: {_VAR_NAME} is reassigned to"
            f" {_VAR_VALUES[3][0]}. This is the most"
            f" recent value.",
        ),
    ]

    usable_tokens = int(target_tokens * 0.90)
    estimated_codes = int(usable_tokens * 0.8 / 5)
    filler_text = _generate_confusable_filler(rng, estimated_codes)

    def _assemble(filler: str) -> str:
        parts = []
        last_pos = 0
        for depth, statement in assignments:
            insert_pos = int(len(filler) * depth)
            parts.append(filler[last_pos:insert_pos])
            parts.append(f"\n\n{statement}\n\n")
            last_pos = insert_pos
        parts.append(filler[last_pos:])
        return "".join(parts) + f"\n\n{question}"

    prompt = _assemble(filler_text)

    if ctx is not None:
        for _attempt in range(3):
            actual = _count_tokens(ctx, prompt)
            if actual <= 0:
                break
            if abs(actual - usable_tokens) <= usable_tokens * 0.05:
                break
            ratio = usable_tokens / max(actual, 1)
            estimated_codes = max(10, int(estimated_codes * ratio))
            filler_text = _generate_confusable_filler(
                random.Random(seed), estimated_codes
            )
            prompt = _assemble(filler_text)

        final_count = _count_tokens(ctx, prompt) if ctx else 0
        if final_count > 0:
            logger.info(
                "  Prompt target: %d tokens, actual: %d tokens (%.0f%% fill)",
                target_tokens,
                final_count,
                (final_count / target_tokens * 100) if target_tokens else 0,
            )

    return prompt, needle


def _pregenerate_prompts(
    ctx: AppContext | None,
    context_sizes: list[int],
) -> dict[int, tuple[str, str]]:
    """Pre-generate variable tracking prompts for all context sizes."""
    prompts: dict[int, tuple[str, str]] = {}
    for ctx_size in context_sizes:
        seed = int(
            hashlib.md5(str(ctx_size).encode(), usedforsecurity=False).hexdigest()[:8],
            16,
        )
        prompts[ctx_size] = _build_variable_tracking_prompt(ctx, ctx_size, seed)
    return prompts


def _run_quality_test(
    ctx: AppContext,
    prompt: str,
    expected: str,
) -> dict | None:
    """Send the pre-generated prompt and check the answer.

    Returns dict with keys: quality_pass, tps, pp, ttft, prompt_tokens.
    Returns None if PP times out.
    """
    try:
        r = ctx.http.post(
            f"{ctx.server_url}/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a precise fact-extraction assistant. "
                            "Answer with ONLY the exact value requested, nothing else. "
                            "Do not explain, do not add context."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": 128,
                "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=_PP_TIMEOUT_S,
        )
    except requests.exceptions.ReadTimeout:
        logger.info("  HTTP read timeout after %ds", _PP_TIMEOUT_S)
        return None
    except requests.exceptions.ConnectionError as e:
        logger.warning("  Server connection failed (crashed?): %s", e)
        return None
    except (requests.RequestException, ValueError) as e:
        logger.warning("  Quality test failed: %s: %s", type(e).__name__, e)
        return None

    if r.status_code != HTTP_OK:
        logger.warning("  Server returned HTTP %d: %s", r.status_code, r.text[:200])
        return None

    data = r.json()

    timings = data.get("timings", data.get("usage", {}))
    prompt_ms = timings.get("prompt_ms", 0)
    prompt_n = timings.get("prompt_n", 0)
    predicted_ms = timings.get("predicted_ms", 0)
    predicted_n = timings.get("predicted_n", 1)

    pp = (prompt_n / (prompt_ms / 1000)) if prompt_ms > 0 else 0
    tps = (predicted_n / (predicted_ms / 1000)) if predicted_ms > 0 else 0
    ttft = prompt_ms

    choices = data.get("choices", [])
    msg = choices[0].get("message", {}) if choices else {}
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    expected_lower = expected.lower()

    content_no_think = re.sub(
        r"<think>.*?(</think>|$)", "", content, flags=re.DOTALL
    ).strip()
    found = (
        expected_lower in content_no_think.lower()
        or expected_lower in content.lower()
        or expected_lower in reasoning.lower()
    )
    logger.debug(
        "  VarTrack: expected=%r found=%s content=%r",
        expected,
        found,
        content_no_think[:200],
    )

    return {
        "quality_pass": found,
        "tps": tps,
        "pp": pp,
        "ttft": ttft,
        "prompt_tokens": prompt_n,
    }


def _read_vram_mb() -> float:
    """Read current VRAM usage, returning 0.0 on failure."""
    try:
        from ..hardware import get_vram_used_mb

        return get_vram_used_mb() or 0.0
    except (ImportError, OSError, RuntimeError) as e:
        logger.debug("VRAM read failed: %s", e)
        return 0.0


def _build_measurement(
    kv_type: str,
    ctx_size: int,
    result: dict[str, Any],
    vram_mb: float,
) -> dict[str, Any]:
    """Build a measurement dict and log the result line."""
    quality_pass = result["quality_pass"]
    quality_status = "PASS" if quality_pass else "FAIL"
    vram_gb = vram_mb / 1024 if vram_mb else 0

    logger.info(
        "  KV Sweep: %s @ %d -- %.1f t/s | pp: %.0f"
        " | TTFT: %.0fms | VRAM: %.1fGB | quality: %s",
        kv_type,
        ctx_size,
        result["tps"],
        result["pp"],
        result["ttft"],
        vram_gb,
        quality_status,
    )

    return {
        "kv_type": kv_type,
        "context": ctx_size,
        "tps": result["tps"],
        "pp": result["pp"],
        "ttft": result["ttft"],
        "vram_mb": vram_mb,
        "quality_pass": quality_pass,
    }


def prepare_test_prompts(
    ctx: AppContext,
    all_test_points: set[int],
    base_config: dict,
) -> dict[int, tuple[str, str]]:
    """Boot a small server and pre-generate quality test prompts."""
    sorted_points = sorted(all_test_points)
    logger.info(
        "  Pre-generating variable tracking prompts for %d context sizes...",
        len(sorted_points),
    )
    token_config = {**base_config, "context": DEFAULT_CONTEXT_SIZE}
    _proc, status = boot_server_with_jinja_recovery(ctx, token_config)
    if status == "ok":
        test_prompts = _pregenerate_prompts(ctx, sorted_points)
        kill_server(ctx)
    else:
        kill_server(ctx)
        test_prompts = _pregenerate_prompts(None, sorted_points)
    return test_prompts


def measure_single_kv_type(
    ctx: AppContext,
    kv_type: str,
    max_bootable: int,
    test_prompts: dict[int, tuple[str, str]],
    base_config: dict,
) -> tuple[list[dict], int]:
    """Measure TPS + quality for one KV type across context sizes.

    Returns (measurements list, max_practical context).
    """
    test_points = generate_test_points(max_bootable)
    measurements: list[dict] = []
    max_practical = 0

    for ctx_size in test_points:
        config = {
            **base_config,
            "kv_cache_type": kv_type,
            "flash_attn": "on",
            "context": ctx_size,
        }

        _proc, status = boot_server_with_jinja_recovery(ctx, config)
        if status != "ok":
            logger.warning(
                "  KV Sweep: %s @ %d -- boot failed (%s)",
                kv_type,
                ctx_size,
                status,
            )
            kill_server(ctx)
            break

        vram_mb = _read_vram_mb()

        prompt, expected = test_prompts[ctx_size]
        ctx_label = ctx_size // 1024 if ctx_size >= 1024 else ctx_size  # noqa: PLR2004
        logger.info(
            "  Sending %dk quality test (%d chars, timeout=%ds)...",
            ctx_label,
            len(prompt),
            _PP_TIMEOUT_S,
        )
        result = _run_quality_test(ctx, prompt, expected)
        kill_server(ctx)

        if result is None:
            logger.info(
                "  KV Sweep: %s @ %d -- PP timeout (>%ds), stopping",
                kv_type,
                ctx_size,
                _PP_TIMEOUT_S,
            )
            break

        measurement = _build_measurement(kv_type, ctx_size, result, vram_mb)
        measurements.append(measurement)

        if measurement["quality_pass"]:
            max_practical = ctx_size
        else:
            logger.info(
                "  KV Sweep: %s @ %d -- quality failed, stopping",
                kv_type,
                ctx_size,
            )
            break

    return measurements, max_practical


def score_measurements(
    all_measurements: list[dict],
    scoring_weights: tuple[float, float, float] | None = None,
) -> tuple[float, str | None, int | None, dict | None]:
    """Score all measurements, return (best_score, kv, ctx, measurement).

    Args:
        all_measurements: List of measurement dicts from measure_single_kv_type.
        scoring_weights: Optional ``(tps_weight, context_weight, pp_speed_weight)``
            tuple.  Defaults to ``(0.5, 0.3, 0.2)`` when not provided.
    """
    w_tps, w_ctx, w_pp = (
        scoring_weights if scoring_weights is not None else (0.5, 0.3, 0.2)
    )

    best_score = -1.0
    best_kv: str | None = None
    best_ctx: int | None = None
    best_measurement: dict | None = None

    for m in all_measurements:
        if not m.get("quality_pass", False):
            continue
        if m["ttft"] > _PP_TIMEOUT_S * 1000:
            continue

        context_bonus = math.log2(max(m["context"] / DEFAULT_CONTEXT_SIZE, 1)) * 10
        pp_speed = min(1000 / max(m["ttft"], 1), 100)
        score = (m["tps"] * w_tps) + (context_bonus * w_ctx) + (pp_speed * w_pp)

        if score > best_score:
            best_score = score
            best_kv = m["kv_type"]
            best_ctx = m["context"]
            best_measurement = m

    return best_score, best_kv, best_ctx, best_measurement


def log_sweep_results(
    baseline_m: dict | None,
    best_measurement: dict | None,
    best_kv: str,
    best_ctx: int,
) -> None:
    """Log the final KV + Context Sweep results block."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("  KV + Context Sweep -- RESULTS")
    logger.info("=" * 60)
    logger.info("")

    if baseline_m:
        quality = "PASS" if baseline_m.get("quality_pass") else "FAIL"
        baseline_vram = baseline_m.get("vram_mb", 0) / 1024
        logger.info(
            "  Baseline: %.1f t/s | pp: %.0f"
            " | TTFT: %.0fms | VRAM: %.1fGB | quality: %s",
            baseline_m["tps"],
            baseline_m["pp"],
            baseline_m["ttft"],
            baseline_vram,
            quality,
        )
    if best_measurement:
        ctx_multiplier = best_ctx / DEFAULT_CONTEXT_SIZE if best_ctx else 1
        quality = "PASS" if best_measurement.get("quality_pass") else "FAIL"
        best_vram = best_measurement.get("vram_mb", 0) / 1024
        logger.info(
            "  Optimal:  %.1f t/s | pp: %.0f"
            " | TTFT: %.0fms | VRAM: %.1fGB"
            " | quality: %s | %s @ %d (%.0fx context)",
            best_measurement["tps"],
            best_measurement["pp"],
            best_measurement["ttft"],
            best_vram,
            quality,
            best_kv,
            best_ctx,
            ctx_multiplier,
        )
    logger.info("")
    logger.info(
        "  Params:   kv_cache_type=%s, context=%s",
        best_kv,
        best_ctx,
    )
