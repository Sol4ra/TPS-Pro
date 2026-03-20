"""
LogTee, PhaseTimer, check_dry_run, generate_tensor_splits, read_json_safe.

Error strategy (see errors.py for full documentation):
    - read_json_safe(): canonical File I/O pattern -- returns None on any
      read/parse error, logs at warning level.  All JSON file reads in
      the project should use this function.
    - LogTee: if the log file cannot be opened, _file is set to None and
      all writes silently skip the file (degraded but functional).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from ..constants.engine import MIN_SPLIT_FRACTION as _MIN_SPLIT_FRACTION
from ..result_types import EngineConfig
from ..state import AppContext

logger = logging.getLogger(__name__)

_SECS_PER_MIN = 60
_SECS_PER_HOUR = 3600

__all__ = [
    "LogTee",
    "PhaseTimer",
    "check_dry_run",
    "generate_tensor_splits",
    "read_json_safe",
]


def read_json_safe(
    path: str | Path, logger_instance: logging.Logger | None = None
) -> dict | None:
    """Read and parse a JSON file, returning None on any error."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, ValueError) as e:
        if logger_instance:
            logger_instance.warning("Failed to read %s: %s", path, e)
        return None


# ============================================================
# LogTee — duplicate stdout to log file
# ============================================================


class LogTee:
    """Tees stdout to both the console and a timestamped log file."""

    def __init__(self, results_dir):
        import atexit

        self._closed = False
        self._original_stdout = sys.stdout
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(results_dir) / f"optimize_{ts}.log"
        self.log_path = str(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._file = open(log_path, "w", encoding="utf-8", errors="replace")
        except OSError:
            # Degraded mode: log file unavailable, stdout-only operation.
            # This is intentional -- the optimizer should not abort just
            # because the log file could not be created.
            self._file = None
            return
        sys.stdout = self
        atexit.register(self.close)

    def write(self, data: str) -> None:
        self._original_stdout.write(data)
        if self._file is not None:
            self._file.write(data)

    def flush(self) -> None:
        self._original_stdout.flush()
        if self._file is not None:
            self._file.flush()

    def isatty(self) -> bool:
        return self._original_stdout.isatty()

    @property
    def encoding(self) -> str:
        return getattr(self._original_stdout, "encoding", None) or "utf-8"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.flush()
        sys.stdout = self._original_stdout
        if self._file is not None:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()
        return False


# ============================================================
# Dry-run mode
# ============================================================


def check_dry_run(
    ctx: AppContext,
    phase_name: str,
    config: EngineConfig | None,
    n_trials: int | str,
) -> bool:
    """If dry-run mode is active, print what WOULD happen and return True."""
    if not ctx.dry_run:
        return False
    trial_str = "full sweep" if n_trials == "full" else f"{n_trials} trials"
    logger.info("=" * 50)
    logger.info("[DRY RUN] Phase: %s", phase_name)
    logger.info("Trials: %s", trial_str)
    if config:
        logger.info("Config keys: %s", sorted(config.keys()))
    logger.info("=" * 50)
    return True


# ============================================================
# Multi-GPU Tensor Split Generation
# ============================================================


def _splits_for_2_gpus(step: float) -> list[tuple[float, ...]]:
    """Generate split candidates for 2 GPUs."""
    splits: list[tuple[float, ...]] = []
    ratio = step
    while ratio <= 1.0 - step:
        splits.append((round(ratio, 2), round(1.0 - ratio, 2)))
        ratio += step
    return splits


def _splits_for_3_gpus(step: float) -> list[tuple[float, ...]]:
    """Generate split candidates for 3 GPUs."""
    splits: list[tuple[float, ...]] = []
    for a_int in range(1, 19):
        a = round(a_int * step, 2)
        for b_int in range(1, 19):
            b = round(b_int * step, 2)
            c = round(1.0 - a - b, 2)
            if c >= step:
                splits.append((a, b, c))
    return splits


def _splits_for_n_gpus(gpu_count: int) -> list[tuple[float, ...]]:
    """Generate split candidates for 4+ GPUs using boost/deficit strategy."""
    even = round(1.0 / gpu_count, 2)
    splits: list[tuple[float, ...]] = [tuple([even] * gpu_count)]
    for primary in range(gpu_count):
        for boost in (0.10, 0.20, 0.30):
            deficit = boost / (gpu_count - 1)
            split_up = [
                round(even + boost, 2) if i == primary else round(even - deficit, 2)
                for i in range(gpu_count)
            ]
            if all(v > _MIN_SPLIT_FRACTION for v in split_up):
                total = sum(split_up)
                splits.append(tuple(round(v / total, 2) for v in split_up))
            split_down = [
                round(even - boost, 2) if i == primary else round(even + deficit, 2)
                for i in range(gpu_count)
            ]
            if all(v > _MIN_SPLIT_FRACTION for v in split_down):
                total = sum(split_down)
                splits.append(tuple(round(v / total, 2) for v in split_down))
    return list(set(splits))


def generate_tensor_splits(gpu_count: int) -> list[tuple[float, ...]]:
    """Generate candidate tensor split ratio tuples for 2-4 GPUs."""
    step = 0.05
    if gpu_count == 2:  # noqa: PLR2004
        return _splits_for_2_gpus(step)
    if gpu_count == 3:  # noqa: PLR2004
        return _splits_for_3_gpus(step)
    if gpu_count >= 4:  # noqa: PLR2004
        return _splits_for_n_gpus(gpu_count)
    return []


# ============================================================
# PhaseTimer — ETA and timing tracking
# ============================================================


class PhaseTimer:
    """Track timing for phases and individual trials to provide ETAs."""

    def __init__(self):
        from collections import deque

        self._phases = {}
        self._trial_durations: deque[float] = deque(maxlen=20)

    def start_phase(self, name: str) -> None:
        self._phases[name] = {"start": time.time(), "end": None, "duration": None}
        self._trial_durations.clear()

    def end_phase(self, name: str) -> None:
        if name in self._phases and self._phases[name]["start"]:
            self._phases[name]["end"] = time.time()
            self._phases[name]["duration"] = (
                self._phases[name]["end"] - self._phases[name]["start"]
            )

    def record_trial(self, duration: float) -> None:
        self._trial_durations.append(duration)

    def eta(self, remaining_trials: int) -> str:
        if not self._trial_durations or remaining_trials <= 0:
            return "unknown"
        avg = sum(self._trial_durations) / len(self._trial_durations)
        remaining_sec = avg * remaining_trials
        if remaining_sec < _SECS_PER_MIN:
            return f"{remaining_sec:.0f}s"
        elif remaining_sec < _SECS_PER_HOUR:
            return f"{remaining_sec / _SECS_PER_MIN:.1f}m"
        else:
            return f"{remaining_sec / _SECS_PER_HOUR:.1f}h"

    def summary(self) -> str:
        lines = ["\n  Phase Timing Summary", "  " + "-" * 45]
        total = 0.0
        for name, info in self._phases.items():
            dur = info.get("duration")
            if dur is not None:
                total += dur
                dur_str = (
                    f"{dur:.1f}s"
                    if dur < _SECS_PER_MIN
                    else (
                        f"{dur / _SECS_PER_MIN:.1f}m"
                        if dur < _SECS_PER_HOUR
                        else f"{dur / _SECS_PER_HOUR:.1f}h"
                    )
                )
                lines.append(f"  {name:20s}  {dur_str:>8s}")
            else:
                lines.append(f"  {name:20s}  {'skipped':>8s}")
        lines.append("  " + "-" * 45)
        t_str = (
            f"{total / _SECS_PER_MIN:.1f}m"
            if total < _SECS_PER_HOUR
            else f"{total / _SECS_PER_HOUR:.1f}h"
        )
        lines.append(f"  {'Total':20s}  {t_str}")
        return "\n".join(lines)
