"""Quality gate prompts, NIAH tasks, quality thresholds, and KL/PPL checks."""

from __future__ import annotations

import functools
import json
from pathlib import Path

from ._lazy import _LazyJsonList
from .scoring import TTFT_BASELINE_MS

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ============================================================
# TPS Measurement
# ============================================================

TPS_TEST_PROMPT = (
    "Write a Python function that implements binary search"
    " on a sorted list. Include docstring and type hints."
)

# ============================================================
# Quality Scoring Weights (3-signal formula)
# ============================================================

QUALITY_WEIGHT_CORRECTNESS = 0.40  # did it get the MC answer right
QUALITY_WEIGHT_CONFIDENCE = 0.40  # logprob of the answer token (how sure is it)
QUALITY_WEIGHT_EFFICIENCY = 0.20  # TTFT (lower = better, breaks ties)
# seed passed to API (only affects non-greedy; temp=0 is
# deterministic regardless)
QUALITY_EVAL_SEED = 0
# Canonical definition is TTFT_BASELINE_MS in scoring.py; alias for compatibility.
QUALITY_TTFT_BASELINE = TTFT_BASELINE_MS

# ============================================================
# Quality Gate: Token-Level Uncertainty
# ============================================================

QUALITY_GATE_SEED = 42
QUALITY_GATE_N_PREDICT = 1024
QUALITY_GATE_UNCERTAIN_THRESHOLD = (
    -0.5
)  # logprob < this = uncertain token (~60% confidence)
QUALITY_GATE_TAIL_PCT = 0.20  # average the worst 20% of logprobs
QUALITY_GATE_CEILING = 0.015  # 1.5% more uncertain tokens = still acceptable
QUALITY_GATE_SOFT_PENALTY = 0.85
QUALITY_GATE_CLIFF = 0.03  # 3% more uncertain tokens = hard disqualify
QUALITY_GATE_CLIFF_PENALTY = 0.1

# ============================================================
# KL-Divergence Quality Check
# ============================================================

KL_DIV_TOP_K = 10  # number of top logprobs to request
KL_DIV_THRESHOLD = 0.5  # mean KL-div above this = significant degradation
KL_DIV_HARD_FAIL = 1.5  # mean KL-div above this = config is provably broken
KL_DIV_PROMPTS = [
    (
        "Explain the second law of thermodynamics in terms"
        " of entropy and spontaneous processes. Be precise."
    ),
    (
        "What is the difference between a monad and a"
        " functor in functional programming?"
        " Give concrete examples."
    ),
]

# ============================================================
# Perplexity Quality Check
# ============================================================

PPL_DEGRADATION_WARN = 0.10  # 10% PPL increase = warning (soft penalty)
PPL_DEGRADATION_FAIL = 0.30  # 30% PPL increase = hard fail (severe penalty)


@functools.lru_cache(maxsize=1)
def get_ppl_reference_text() -> str:
    """Load and return the perplexity reference text."""
    ppl_path = _DATA_DIR / "ppl_reference.txt"
    if ppl_path.exists():
        return ppl_path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"PPL reference text not found: {ppl_path}")


# ============================================================
# Quality Gate Prompts (GPQA Diamond -- graduate-level)
# ============================================================


@functools.lru_cache(maxsize=1)
def _load_quality_gate_prompts() -> tuple[str, ...]:
    """Load quality gate prompts from data/quality_gate_prompts.json.

    Each entry has 'source', 'domain', and 'prompt' keys.
    Returns a tuple of prompt strings (tuple for lru_cache hashability).
    """
    path = _DATA_DIR / "quality_gate_prompts.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(item["prompt"] for item in raw)


QUALITY_GATE_PROMPTS: _LazyJsonList[str] = _LazyJsonList(_load_quality_gate_prompts)

# ============================================================
# Quality Eval Tasks -- Hard Multiple Choice
# ============================================================


@functools.lru_cache(maxsize=1)
def _load_quality_tasks() -> tuple[tuple[str, str, str], ...]:
    """Load quality evaluation tasks from data/quality_tasks.json.

    Each entry is a dict with 'prompt', 'answer', and 'category' keys.
    Returns a tuple of (prompt, correct_letter, category) tuples.
    """
    path = _DATA_DIR / "quality_tasks.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple((item["prompt"], item["answer"], item["category"]) for item in raw)


QUALITY_TASKS: _LazyJsonList[tuple[str, str, str]] = _LazyJsonList(_load_quality_tasks)

# ============================================================
# Needle-in-a-Haystack (NIAH) Test Data
# ============================================================


@functools.lru_cache(maxsize=1)
def _load_niah_needles() -> tuple[dict[str, str], ...]:
    """Load NIAH needle data lazily on first access."""
    path = _DATA_DIR / "niah_needles.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(raw)


@functools.lru_cache(maxsize=1)
def _load_niah_filler_blocks() -> tuple[str, ...]:
    """Load NIAH filler blocks lazily on first access."""
    path = _DATA_DIR / "niah_filler_blocks.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(raw)


NIAH_NEEDLES: _LazyJsonList[dict[str, str]] = _LazyJsonList(_load_niah_needles)
NIAH_FILLER_BLOCKS: _LazyJsonList[str] = _LazyJsonList(_load_niah_filler_blocks)
