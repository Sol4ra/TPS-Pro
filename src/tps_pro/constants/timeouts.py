"""Timeout values, retry counts, polling intervals, and thermal thresholds."""

from __future__ import annotations

# ============================================================
# Network Binding
# ============================================================

BIND_HOST = "127.0.0.1"

# ============================================================
# Timeout Values (seconds)
# ============================================================

SERVER_HEALTH_POLL_TIMEOUT = 0.5  # per-request timeout for /health polling
SERVER_HEALTH_POLL_SLEEP = 0.1  # sleep between /health polls
SERVER_KILL_WAIT_TIMEOUT = 2  # psutil wait after killing the server process (was 5)
SERVER_PROBE_TIMEOUT = (
    10  # timeout for quick informational server probes (e.g. /props, /tokenize)
)
SERVER_PORT_RELEASE_INITIAL_DELAY = 0.05  # initial backoff delay for port release check
SERVER_PORT_RELEASE_MAX_DELAY = 0.5  # maximum backoff delay for port release check
SERVER_PORT_RELEASE_RETRIES = (
    4  # number of retries when waiting for port to close (was 12)
)
WARMUP_REQUEST_TIMEOUT = 60  # timeout for each warmup HTTP request
BENCH_SUBPROCESS_TIMEOUT = 300  # timeout for llama-bench subprocess
LARGE_REQUEST_TIMEOUT = 300  # timeout for large-prompt and quality HTTP requests
QUICK_GATE_TIMEOUT = 30  # timeout for quick-gate (5-token) measurement requests
CONCURRENT_REQUEST_TIMEOUT = 120  # timeout for concurrent load test requests

# ============================================================
# Warmup Token Counts
# ============================================================

WARMUP_TOKENS_STAGE1 = 5  # max_tokens for first warmup request (CUDA pipeline warm)
WARMUP_TOKENS_STAGE2 = (
    10  # max_tokens for second warmup request (speculation cache prime)
)

# ============================================================
# Thermal Thresholds (Celsius)
# ============================================================

THERMAL_THROTTLE_THRESHOLD = (
    85  # GPU temperature at/above which we consider it throttling
)
THERMAL_COOLDOWN_TARGET = 75  # GPU temperature to cool down to before resuming
THERMAL_COOLDOWN_TIMEOUT = 120  # maximum seconds to wait for GPU to cool down

# ============================================================
# Study / Optimization Limits
# ============================================================

STUDY_SAFETY_TIMEOUT_SEC = 90 * 60  # 90-minute hard timeout per Optuna study run

# ============================================================
# NIAH / Long-Context Timeouts
# ============================================================

NIAH_REQUEST_TIMEOUT = 600  # timeout for NIAH full-context inference requests (seconds)

# ============================================================
# HTTP Status Codes
# ============================================================

HTTP_OK = 200
HTTP_FORBIDDEN = 403
HTTP_SERVER_ERROR = 500

# ============================================================
# Server Boot Timeouts (by model size class)
# ============================================================

SERVER_BOOT_TIMEOUTS = {"tiny": 60, "small": 120, "medium": 300, "large": 600}
MOE_TIMEOUT_MULTIPLIER = 1.5  # boot timeout multiplier for MoE models
