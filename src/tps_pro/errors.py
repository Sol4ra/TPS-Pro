"""Project error hierarchy and error handling strategy.

Error Strategy
==============

This module defines the canonical error hierarchy and documents the
per-layer error handling strategy used throughout the project.  Every
``except`` block should follow one of the patterns below.

Layer strategies
----------------

**Phase functions** (``phases/*.py``):
    Catch and return ``None`` (or a sentinel).  The caller (pipeline
    orchestrator) decides whether to skip, retry, or abort.

**Server lifecycle** (``engine/server.py``):
    Returns status strings (``"ok"``, ``"oom"``, ``"timeout"``, ``"died"``,
    ``"jinja_error"``) from ``wait_for_server`` -- never raises for
    expected failure modes.  Cleanup code (process kill, stderr close)
    may swallow errors with a ``# Cleanup: safe to ignore`` comment.

**Measurement** (``measurement.py``):
    Return empty/zero results on transient HTTP failures (the adaptive
    measurement loop handles retries).  Log at ``warning`` level so
    failures are visible.

**File I/O** (``engine/util.py``):
    Use ``read_json_safe()`` for all JSON file reads -- returns ``None``
    on error with a ``warning`` log.  Write errors propagate.

**Hardware detection** (``hardware.py``):
    Return safe defaults (empty list, ``None``, ``False``) because GPU
    detection is optional -- the optimizer must work on CPU-only systems.
    Log at ``warning`` or ``debug`` depending on severity.

**CLI** (``main.py``, ``cli/*.py``):
    Catch and display user-friendly messages.  Never silently swallow.

**Search / Optuna** (``search/_study.py``, ``search/_callbacks.py``):
    Catch ``RuntimeError`` / ``ValueError`` from Optuna internals and
    log at ``warning``.  These are recoverable (study continues).

Cleanup / atexit code
---------------------
Cleanup handlers (``atexit``, signal handlers, ``kill_process_tree``) catch
broad exceptions to guarantee the process exits cleanly.  Each such
block is annotated with ``# Cleanup: safe to ignore``.
"""

from __future__ import annotations


class OptimizerError(Exception):
    """Base exception for all project errors."""


class ServerError(OptimizerError):
    """Server lifecycle error (base class)."""


class BenchOOMError(OptimizerError):
    """Raised when a llama-bench trial fails due to out-of-memory.

    This is the single canonical definition.  ``engine.bench`` re-imports
    this class for backward compatibility.
    """


class BaselineFailure(OptimizerError):  # noqa: N818
    """Raised when a baseline server fails to start with --fail-fast enabled.

    This is the canonical definition. ``engine.util`` re-exports this class
    for backward compatibility.
    """
