"""Windows-only module: Job Object management for child-process lifecycle.

Assigns server processes to a Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
so that child processes are automatically terminated when the parent dies.

Error strategy (see errors.py for full documentation):
    Job Object assignment is best-effort -- if it fails the optimizer
    continues normally; the only consequence is that orphan server
    processes may survive if the parent is killed.  Errors are logged
    at warning level.
"""

from __future__ import annotations

import logging
import subprocess
import sys

logger = logging.getLogger(__name__)

# Windows Job Object constants
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFO = 9
_PROCESS_ALL_ACCESS = 0x1F0FFF

# Windows Job Object -- auto-kills child processes when parent dies
# Use a mutable container to avoid the `global` keyword.
_job_state: dict[str, object] = {"handle": None}


def _assign_job_object(proc: subprocess.Popen) -> None:
    """Assign process to a Windows Job Object.

    Uses JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE so when the parent
    process dies (even if killed), all children are terminated.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32

        if _job_state["handle"] is None:
            _job_state["handle"] = kernel32.CreateJobObjectW(None, None)
            if not _job_state["handle"]:
                return

            # Set JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", ctypes.c_byte * 48),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            kernel32.SetInformationJobObject(
                _job_state["handle"],
                _JOB_OBJECT_EXTENDED_LIMIT_INFO,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )

        # Assign process to job
        handle = kernel32.OpenProcess(_PROCESS_ALL_ACCESS, False, proc.pid)
        if handle:
            kernel32.AssignProcessToJobObject(_job_state["handle"], handle)
            kernel32.CloseHandle(handle)
    except (OSError, ValueError) as e:
        logger.warning("Job Object setup failed -- children may outlive parent: %s", e)
