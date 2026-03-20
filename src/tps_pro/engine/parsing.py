"""OOM detection, error-line classification, and load-time parsing
from server stderr."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import ServerProcess

logger = logging.getLogger(__name__)

_MIN_LOAD_TIME_MS = 50  # filter tiny irrelevant matches from load-time parsing

# Shared OOM keyword list -- used by wait_for_server,
# run_bench_trial, server_start_failed
_OOM_KEYWORDS = [
    "out of memory",
    "oom",
    "alloc failed",
    "cuda error",
    "not enough memory",
    "failed to allocate",
    "ggml_cuda_op_mul_mat",
    "cudamalloc failed",
    "insufficient memory",
]

_ERROR_KEYWORDS = [
    "error:",
    "error",
    "failed to",
    "out of memory",
    "cuda error",
    "cannot allocate",
    "segfault",
    "abort",
    "fatal",
    "panic",
    "exception",
    "oom",
    "alloc failed",
]

_POSITIVE_INDICATORS = [
    "success",
    "initialized",
    "loaded",
    "ready",
]


def is_oom(text: str) -> bool:
    """Check if text contains OOM-related keywords."""
    lower = text.lower()
    return any(kw in lower for kw in _OOM_KEYWORDS)


def _is_error_line(line: str) -> bool:
    """Check if a stderr line looks like an error."""
    if not line.strip():
        return False
    lower = line.lower()
    if any(pos in lower for pos in _POSITIVE_INDICATORS):
        return False
    return any(kw in lower for kw in _ERROR_KEYWORDS)


# Patterns for model load timing from llama-server stderr
# Ordered by specificity -- first match wins
_LOAD_TIME_PATTERNS = [
    # Classic: "llm_load_tensors: loaded in 1234.56 ms"
    re.compile(r"llm_load_tensors:\s*loaded\s+in\s+([\d.]+)\s*ms", re.IGNORECASE),
    # Newer: "llama_model_load: loaded ... in 1234 ms"
    re.compile(r"llama_model_load.*?(\d+(?:\.\d+)?)\s*ms", re.IGNORECASE),
    # "load_time = 1234.56 ms"
    re.compile(r"load\s*_?\s*time\s*=\s*([\d.]+)\s*ms", re.IGNORECASE),
    # "model load time = 1234 ms"
    re.compile(r"model\s+load\s+time\s*=\s*([\d.]+)\s*ms", re.IGNORECASE),
    # "total time = 1234.56 ms"
    re.compile(r"total\s+time\s*=\s*([\d.]+)\s*ms", re.IGNORECASE),
    # Generic: any line with "loaded" and "ms" with a number > 100
    re.compile(r"loaded\b.*?(\d{3,}(?:\.\d+)?)\s*ms", re.IGNORECASE),
]
_load_time_debug_done_set: set[str] = set()

# Fallback regex for load-time parsing -- compiled once at module level
# instead of on every call to _parse_load_time.
_MS_FALLBACK_RE = re.compile(r"(\d+(?:\.\d+)?)\s*ms")

# Keywords indicating a load-time line in fallback parsing.
_LOAD_KEYWORDS = ("loaded", "load_time", "init", "warmup")


def reset_load_time_debug():
    """Reset load-time debug state for a new model (batch mode)."""
    _load_time_debug_done_set.clear()


def _try_pattern_match(lines: list[str]) -> float | None:
    """Try to match load time using known patterns.

    Returns:
        Parsed load time in milliseconds, or None if no match.
    """
    for line in lines:
        for pattern in _LOAD_TIME_PATTERNS:
            m = pattern.search(line)
            if not m:
                continue
            try:
                val = float(m.group(1))
                if val > _MIN_LOAD_TIME_MS:
                    return val
            except (ValueError, IndexError) as e:
                logger.debug(
                    "Load time parse error for pattern %s: %s", pattern.pattern, e
                )
    return None


def _try_fallback_match(lines: list[str]) -> float | None:
    """Fallback: scan for keyword lines containing a ms-value.

    Returns:
        Parsed load time in milliseconds, or None if no match.
    """
    for line in reversed(lines):
        lower = line.lower()
        if not any(kw in lower for kw in _LOAD_KEYWORDS):
            continue
        m = _MS_FALLBACK_RE.search(line)
        if m:
            val = float(m.group(1))
            if val > _MIN_LOAD_TIME_MS:
                return val
    return None


def _log_parse_failure(lines: list[str]) -> None:
    """Log debug info on first parse failure to help fix regexes."""
    if "done" in _load_time_debug_done_set:
        return
    _load_time_debug_done_set.add("done")
    candidates = [
        line for line in lines if "ms" in line.lower() or "load" in line.lower()
    ]
    if candidates:
        logger.debug("Load time parse failed. Candidate stderr lines:")
        for c in candidates[:10]:
            logger.debug("  >> %s", c)
    else:
        logger.debug("Load time parse: no stderr lines contain 'ms' or 'load'")


def _parse_load_time(server_proc: ServerProcess) -> float | None:
    """Extract model load time from server stderr lines.

    Looks for llama.cpp's load timing lines (format varies by version).
    Returns the parsed load time in milliseconds, or None if not found.
    """
    with server_proc.lock:
        lines = list(server_proc.stderr_lines)

    result = _try_pattern_match(lines)
    if result is not None:
        _load_time_debug_done_set.add("done")
        return result

    result = _try_fallback_match(lines)
    if result is not None:
        _load_time_debug_done_set.add("done")
        return result

    _log_parse_failure(lines)
    return None
