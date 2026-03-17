"""
Scoring, quality evaluation, and measurement constants.

This module is a leaf — no internal package imports.
"""

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
QUALITY_EVAL_SEED = 0              # fixed seed for deterministic grading
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

ADAPTIVE_THRESHOLD = 0.70  # score must be >= 70% of best to warrant full measurement
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

_PPL_REFERENCE_TEXT = """The development of the transistor at Bell Labs in 1947 fundamentally transformed \
the landscape of electronic engineering. William Shockley, John Bardeen, and Walter Brattain demonstrated \
that a solid-state device could amplify electrical signals, replacing the fragile and power-hungry vacuum \
tubes that had dominated electronic circuits since the early twentieth century. The point-contact transistor, \
while crude by modern standards, proved that semiconductor junctions could control current flow with \
remarkable precision.

The implications rippled through every branch of technology. By the mid-1950s, transistor radios had \
become consumer products, shrinking what had once required furniture-sized cabinets into pocket-sized \
devices. Military applications followed rapidly: guidance computers for intercontinental ballistic missiles \
demanded the reliability and compactness that only solid-state electronics could provide. The TRADIC \
computer, built by Bell Labs for the United States Air Force in 1954, was among the first fully \
transistorized computers, consuming a fraction of the power of its vacuum-tube predecessors.

The invention of the integrated circuit by Jack Kilby at Texas Instruments and Robert Noyce at Fairchild \
Semiconductor in 1958 and 1959 respectively marked the next evolutionary leap. Rather than connecting \
discrete transistors with hand-soldered wires, entire circuits could be fabricated on a single silicon \
wafer. This innovation made possible the exponential scaling predicted by Gordon Moore in 1965: the \
number of transistors on an integrated circuit would double approximately every two years, a trend that \
has held with remarkable consistency for over five decades.

Modern processors contain billions of transistors etched at scales measured in nanometers. The AMD Ryzen \
and Intel Core families pack extraordinary computational density into chips smaller than a postage stamp. \
Graphics processing units from NVIDIA and AMD contain even more transistors, organized into thousands of \
parallel execution units optimized for the matrix multiplications that underpin both computer graphics and \
artificial intelligence workloads. The transition from general-purpose computing to specialized accelerators \
represents a philosophical shift in computer architecture: rather than building one fast processor, engineers \
now build thousands of simpler processors that work in concert.

Quantum computing represents a fundamentally different approach to computation. Instead of classical bits \
that exist in states of zero or one, quantum bits or qubits exploit superposition to exist in multiple \
states simultaneously. Entanglement allows qubits to be correlated in ways that have no classical analog, \
enabling certain algorithms to explore solution spaces exponentially faster than any classical computer. \
Peter Shor's algorithm for integer factorization and Lov Grover's search algorithm demonstrated that \
quantum computers could solve specific problems with dramatic speedups, though building stable quantum \
hardware remains one of the greatest engineering challenges of the twenty-first century.

The field of machine learning, particularly deep neural networks, has emerged as perhaps the most \
transformative application of modern computing hardware. Convolutional neural networks revolutionized \
computer vision, recurrent networks advanced natural language processing, and transformer architectures \
unified both domains under a single attention-based framework. The training of large language models \
requires distributing computation across thousands of accelerators, consuming megawatts of electrical \
power and generating proportional waste heat that must be dissipated through sophisticated cooling \
systems. The environmental implications of this computational appetite have sparked vigorous debate about \
the sustainability of continued scaling.

Despite these concerns, the practical applications are compelling. Medical imaging systems powered by deep \
learning detect tumors with accuracy rivaling experienced radiologists. Protein structure prediction, \
exemplified by AlphaFold, has accelerated biological research by decades. Autonomous vehicles, though \
still imperfect, demonstrate that neural networks can process complex sensory environments in real time. \
Natural language models assist with programming, writing, analysis, and creative tasks, augmenting human \
capabilities in ways that were science fiction a decade ago. The trajectory suggests that artificial \
intelligence will continue to reshape industry, science, and daily life for generations to come."""

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

Think step by step, then answer with just the letter.""", "A", "math"),

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
