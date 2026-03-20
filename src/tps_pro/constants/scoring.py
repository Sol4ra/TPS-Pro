"""Scoring weights, thresholds, baselines, and Pareto constants."""

from __future__ import annotations

# ============================================================
# Composite Score Formula
# ============================================================

SCORE_VERSION = "v3"  # bump this when changing score formula to isolate Optuna studies
TTFT_BASELINE_MS = 500  # ms -- single source of truth for TTFT baseline normalization
SCORE_PP_BASELINE = 300  # t/s -- reference prompt processing speed for normalization

# ============================================================
# Scoring weights -- Full mode (large-prompt data present)
# ============================================================

WEIGHT_GEN_TPS = 0.35  # generation TPS (dominant signal)
WEIGHT_LARGE_TPS = 0.25  # TPS under large-context pressure
WEIGHT_PP_COMPONENT = 0.15  # prompt-processing component (as gen_tps multiplier)
WEIGHT_TTFT_COMPONENT = 0.15  # time-to-first-token component (as gen_tps multiplier)
WEIGHT_VRAM = 0.10  # VRAM efficiency bonus (or fallback gen_tps weight)

# ============================================================
# Scoring weights -- Lightweight / quick-gate mode
# ============================================================

LITE_BASE_MULTIPLIER = 0.60  # base multiplier applied to gen_tps
LITE_WEIGHT_PP = 0.25  # pp_factor contribution to multiplier
LITE_WEIGHT_TTFT = 0.15  # ttft_factor contribution to multiplier
LITE_MULTIPLIER_CAP = 1.50  # cap so gen_tps stays the dominant signal
LITE_VRAM_BONUS_CAP = 0.05  # maximum fractional VRAM headroom bonus
LITE_VRAM_BONUS_SCALE = 0.10  # scale factor for raw headroom -> bonus

# ============================================================
# Concurrent-load blending
# ============================================================

CONCURRENT_BASE_FACTOR = 0.85  # base weight when blending in concurrent score
CONCURRENT_BONUS_WEIGHT = 0.15  # max bonus for perfect scaling efficiency

# ============================================================
# Bench scoring weights (bench_score in phases/_helpers.py)
# ============================================================

BENCH_WEIGHT_GEN_TPS = 0.85  # base weight on generation TPS
BENCH_WEIGHT_PP_TIEBREAK = 0.15  # weight on prompt-processing tiebreaker

# ============================================================
# Sanity-clamp limits applied before scoring
# ============================================================

PROMPT_TPS_CLAMP_MAX = 50000.0  # ceiling for prompt TPS (guards against server junk)
TTFT_FLOOR_MS = 1.0  # minimum TTFT in ms (prevents division by zero)
NORM_CAP_MULTIPLIER = 3.0  # max multiplier for pp_norm / ttft_norm factors

# ============================================================
# Adaptive Measurement
# ============================================================

ADAPTIVE_THRESHOLD = 0.50  # score must be >= 50% of best to warrant full measurement
ADAPTIVE_WARMUP_RUNS = (
    2  # number of warmup runs before deciding promotion (reduces false kills)
)
CV_TARGET = 0.05  # target coefficient of variation for stable measurement
CV_MIN_RUNS = 3  # minimum runs before checking CV
CV_MAX_RUNS = 5  # maximum runs for stability (keep tight to avoid 35s/trial)

# ============================================================
# Failure / Sentinel Values
# ============================================================

VRAM_FAILURE_PENALTY = -99999.0  # sentinel score for unknown/failed VRAM
EARLY_STOP_RATIO = 0.50  # score must drop below 50% of best to trigger early stop
DEFAULT_TRIAL_COUNT = 60  # default number of trials for quality/sampling phase
