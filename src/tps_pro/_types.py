"""Internal type protocols for optional third-party libraries.

Provides structural typing for libraries that lack type stubs (e.g. pynvml)
so the rest of the codebase can avoid bare ``type: ignore`` comments.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# pynvml Protocol
# ---------------------------------------------------------------------------


class NVMLError(Exception):
    """Stand-in for pynvml.NVMLError used in type annotations."""


class NVMLProcessInfo(Protocol):
    """Minimal protocol for pynvml process info objects."""

    @property
    def pid(self) -> int: ...

    @property
    def usedGpuMemory(self) -> int | None: ...  # noqa: N802


@runtime_checkable
class PynvmlModule(Protocol):
    """Structural type for the ``pynvml`` module.

    Covers only the subset of the pynvml API used by this project.
    Because pynvml ships without type stubs, functions that accept the
    module as a parameter annotate it as ``PynvmlModule`` instead of
    ``object``, eliminating the need for ``# type: ignore[attr-defined]``.
    """

    NVMLError: type[Exception]

    def nvmlInit(self) -> None: ...  # noqa: N802
    def nvmlShutdown(self) -> None: ...  # noqa: N802
    def nvmlDeviceGetCount(self) -> int: ...  # noqa: N802
    def nvmlDeviceGetHandleByIndex(self, index: int) -> Any: ...  # noqa: N802
    def nvmlDeviceGetName(self, handle: Any) -> str | bytes: ...  # noqa: N802
    def nvmlDeviceGetMemoryInfo(self, handle: Any) -> Any: ...  # noqa: N802
    def nvmlDeviceGetTemperature(self, handle: Any, sensor_type: int) -> int: ...  # noqa: N802
    def nvmlDeviceGetComputeRunningProcesses(  # noqa: N802
        self, handle: Any,
    ) -> list[NVMLProcessInfo]: ...
    def nvmlDeviceGetGraphicsRunningProcesses(  # noqa: N802
        self, handle: Any,
    ) -> list[NVMLProcessInfo]: ...
    def nvmlSystemGetProcessName(self, pid: int) -> bytes: ...  # noqa: N802

    # Constants
    NVML_TEMPERATURE_GPU: int
