"""
Scoring, quality evaluation, and measurement constants.

This module is a leaf — no internal package imports.
"""

from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"

# ============================================================
# TPS Measurement
# ============================================================

TPS_TEST_PROMPT = "Write a Python function that implements binary search on a sorted list. Include docstring and type hints."

# ============================================================
# Composite Score Formula
# ============================================================

SCORE_VERSION = "v3"        # bump this when changing score formula to isolate Optuna studies
SCORE_TTFT_BASELINE = 500   # ms — reference TTFT for normalization
SCORE_PP_BASELINE = 300     # t/s — reference prompt processing speed for normalization

# ============================================================
# Quality Scoring Weights (3-signal formula)
# ============================================================

QUALITY_WEIGHT_CORRECTNESS = 0.40  # did it get the MC answer right
QUALITY_WEIGHT_CONFIDENCE = 0.40   # logprob of the answer token (how sure is it)
QUALITY_WEIGHT_EFFICIENCY = 0.20   # TTFT (lower = better, breaks ties)
QUALITY_EVAL_SEED = 0              # seed passed to API (only affects non-greedy; temp=0 is deterministic regardless)
QUALITY_TTFT_BASELINE = 500        # ms — reference TTFT for efficiency normalization

# ============================================================
# Quality Gate: Token-Level Uncertainty
# ============================================================

QUALITY_GATE_SEED = 42
QUALITY_GATE_N_PREDICT = 1024
QUALITY_GATE_UNCERTAIN_THRESHOLD = -0.5  # logprob < this = uncertain token (~60% confidence)
QUALITY_GATE_TAIL_PCT = 0.20             # average the worst 20% of logprobs
QUALITY_GATE_CEILING = 0.015    # 1.5% more uncertain tokens = still acceptable
QUALITY_GATE_SOFT_PENALTY = 0.85
QUALITY_GATE_CLIFF = 0.03       # 3% more uncertain tokens = hard disqualify
QUALITY_GATE_CLIFF_PENALTY = 0.1

# ============================================================
# Adaptive Measurement
# ============================================================

ADAPTIVE_THRESHOLD = 0.50  # score must be >= 50% of best to warrant full measurement
ADAPTIVE_WARMUP_RUNS = 2   # number of warmup runs before deciding promotion (reduces false kills)
CV_TARGET = 0.05           # target coefficient of variation for stable measurement
CV_MIN_RUNS = 3            # minimum runs before checking CV
CV_MAX_RUNS = 5            # maximum runs for stability (keep tight to avoid 35s/trial)

# ============================================================
# KL-Divergence Quality Check
# ============================================================

KL_DIV_TOP_K = 10          # number of top logprobs to request
KL_DIV_THRESHOLD = 0.5     # mean KL-div above this = significant degradation
KL_DIV_HARD_FAIL = 1.5     # mean KL-div above this = config is provably broken
KL_DIV_PROMPTS = [
    "Explain the second law of thermodynamics in terms of entropy and spontaneous processes. Be precise.",
    "What is the difference between a monad and a functor in functional programming? Give concrete examples.",
]

# ============================================================
# Perplexity Quality Check
# ============================================================

PPL_DEGRADATION_WARN = 0.10   # 10% PPL increase = warning (soft penalty)
PPL_DEGRADATION_FAIL = 0.30   # 30% PPL increase = hard fail (severe penalty)

_PPL_REFERENCE_TEXT = None  # lazy-loaded on first use to avoid import-time FileNotFoundError

def get_ppl_reference_text():
    """Lazy-load the perplexity reference text on first use."""
    global _PPL_REFERENCE_TEXT
    if _PPL_REFERENCE_TEXT is None:
        ppl_path = _DATA_DIR / "ppl_reference.txt"
        if ppl_path.exists():
            _PPL_REFERENCE_TEXT = ppl_path.read_text(encoding="utf-8")
        else:
            raise FileNotFoundError(f"PPL reference text not found: {ppl_path}")
    return _PPL_REFERENCE_TEXT

# ============================================================
# Quality Gate Prompts (GPQA Diamond — graduate-level)
# ============================================================

QUALITY_GATE_PROMPTS = [
    # GPQA Diamond — graduate-level physics (energy-time uncertainty principle)
    """Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved?

(A) 10^-8 eV
(B) 10^-9 eV
(C) 10^-4 eV
(D) 10^-11 eV

Explain your reasoning step by step before giving your final answer.""",
    # GPQA Diamond — graduate-level biology (mitochondrial genetics)
    """Mitochondria are semi-autonomous cellular organelles in charge of energy production. They encode for a part of their own translational machinery and respiratory complexes. Mitochondrial function is governed by over a thousand proteins imported from the cell, contributing to processes like the transport of proteins, ribosome biogenesis and translation regulation, respiratory oxidation, metabolism, and apoptotic signaling cascade. Mutations in the code for mitochondrial protein networks can cause numerous diseases in humans that are inherited through generations. Mutations of which of the mitochondrial proteins listed below are least likely to be genetically transmitted from a father to his children?

(A) Translocase of inner mitochondrial membrane 17B
(B) ATP binding cassette subfamily B member 8
(C) NADH dehydrogenase 2
(D) Tu translation elongation factor, mitochondrial

Explain your reasoning step by step before giving your final answer.""",
]

# ============================================================
# Quality Eval Tasks — Hard Multiple Choice
# ============================================================

# Format: (prompt, correct_letter, category)
# Categories: math (GSM8K-style), reasoning, stem (MMLU-style), code
QUALITY_TASKS = [
    # GSM8K-style multi-step math
    ("""A store sells notebooks for $4 each. They offer a 15% discount on orders of 10 or more. Sarah buys 12 notebooks, and her friend buys 7 notebooks in a separate order. What is the total amount they pay combined?

(A) $73.60
(B) $76.00
(C) $68.80
(D) $72.40

Think step by step, then answer with just the letter.""", "A", "math"),

    ("""A train leaves Station A at 9:00 AM traveling at 80 km/h. Another train leaves Station B (which is 420 km from Station A) at 10:00 AM traveling toward Station A at 100 km/h. At what time do they meet?

(A) 11:20 AM
(B) 12:00 PM
(C) 11:53 AM
(D) 12:20 PM

Think step by step, then answer with just the letter.""", "C", "math"),

    ("""A rectangular pool is 25m long, 10m wide, and 2m deep. Water fills at 500 liters per minute. If the pool is currently 40% full, how many more hours are needed to fill it completely?

(A) 6 hours
(B) 10 hours
(C) 8 hours
(D) 5 hours

Think step by step, then answer with just the letter.""", "B", "math"),

    # Reasoning (logic, deduction)
    ("""Five people (A, B, C, D, E) sit in a row. B sits immediately to the right of A. C does not sit next to D. E sits at one of the ends. D sits in the middle (position 3). Which of these arrangements is valid?

(A) E, A, D, B, C
(B) A, B, D, C, E
(C) E, A, D, C, B
(D) A, B, D, E, C

Think step by step, then answer with just the letter.""", "B", "reasoning"),

    ("""In a tournament, every team plays every other team exactly once. There are 8 teams. If each game takes 45 minutes and 2 games can be played simultaneously on 2 fields, what is the minimum total time needed to complete the tournament?

(A) 10 hours 30 minutes
(B) 10 hours 15 minutes
(C) 5 hours 15 minutes
(D) 21 hours

Think step by step, then answer with just the letter.""", "A", "reasoning"),

    # STEM / MMLU-style (graduate-level)
    ("""In a double-slit experiment, the distance between the slits is halved while the wavelength of light used is doubled. What happens to the fringe spacing on the screen?

(A) It remains the same
(B) It doubles
(C) It quadruples
(D) It halves

Think step by step, then answer with just the letter.""", "C", "stem"),

    ("""A protein has a mutation that changes a hydrophobic leucine residue in the core to a charged glutamic acid residue. Which of the following is the most likely primary consequence?

(A) Enhanced enzymatic activity due to new charge interactions
(B) Protein misfolding due to disruption of the hydrophobic core
(C) Increased protein-protein binding specificity
(D) No significant effect since single mutations rarely matter

Think step by step, then answer with just the letter.""", "B", "stem"),

    # Code reasoning
    ("""What does this Python code print?

```python
def f(n, memo={}):
    if n in memo: return memo[n]
    if n <= 1: return n
    memo[n] = f(n-1, memo) + f(n-2, memo)
    return memo[n]

result = [f(i) % 10 for i in range(10, 15)]
print(sum(result))
```

(A) 25
(B) 27
(C) 23
(D) 21

Think step by step, then answer with just the letter.""", "B", "code"),

    ("""What is the output of this code?

```python
x = [1, 2, 3, 4, 5]
y = x[1::2]
z = [a * b for a, b in zip(x[::2], y)]
print(z)
```

(A) [2, 12]
(B) [2, 6, 20]
(C) [2, 12, 25]
(D) [1, 6, 15]

Think step by step, then answer with just the letter.""", "A", "code"),

    # Hard reasoning
    ("""A researcher tests a drug on 1000 patients. The drug has an actual effectiveness rate of 95%. The test has a 2% false positive rate and a 5% false negative rate. If a patient tests positive, what is the approximate probability they actually benefit from the drug?

(A) 95%
(B) 97%
(C) 99%
(D) 93%

Think step by step, then answer with just the letter.""", "C", "reasoning"),
]

# ============================================================
# Needle-in-a-Haystack (NIAH) Test Data
# ============================================================

_NIAH_NEEDLES = [
    {
        "fact": "The secret project codename is Operation Sapphire Falcon.",
        "query": "What is the secret project codename?",
        "expected": "sapphire falcon",
    },
    {
        "fact": "The database migration is scheduled for March 27th at 3:00 AM UTC.",
        "query": "When is the database migration scheduled?",
        "expected": "march 27",
    },
    {
        "fact": "The API rate limit for premium users is 4,500 requests per minute.",
        "query": "What is the API rate limit for premium users?",
        "expected": "4,500",
    },
]

_NIAH_FILLER_BLOCKS = [
    "Implement a Python class for a thread-safe connection pool with configurable max connections, idle timeout, and health checking. The pool should support both sync and async contexts.",
    "Design a database schema for a multi-tenant SaaS application with row-level security, audit logging, and soft deletes. Consider indexing strategies for common query patterns.",
    "Write a comprehensive test suite for a REST API that handles user authentication, including unit tests for token validation, integration tests for login flow, and load tests.",
    "Create a CI/CD pipeline configuration that supports multi-environment deployments with canary releases, automatic rollbacks on error rate thresholds, and Slack notifications.",
    "Implement a distributed cache invalidation system using pub/sub messaging. Handle network partitions gracefully and ensure eventual consistency across all cache nodes.",
    "Design a logging and monitoring architecture for a microservices system with structured logging, distributed tracing, metric aggregation, and alerting rules.",
    "Write a data pipeline that ingests CSV files, validates schema, deduplicates records, applies business transformations, and loads into a star schema warehouse.",
    "Implement rate limiting middleware with sliding window algorithm, per-user quotas, burst allowance, and graceful degradation under high load conditions.",
]
