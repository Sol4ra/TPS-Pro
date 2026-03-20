"""Generic lazy-loading list wrapper for JSON-backed data."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Callable, Generic, TypeVar, overload

T = TypeVar("T")


class _LazyJsonList(Generic[T]):
    """Lazy-loading list-like wrapper backed by an lru_cache loader.

    Replaces the former ``_LazyQualityGatePrompts``, ``_LazyQualityTasks``,
    and ``_LazyJsonList`` classes with a single generic implementation.

    Generic over ``T`` so that ``_LazyJsonList[str]`` communicates the
    element type to type checkers even though the runtime object is not
    a true ``list``.
    """

    def __init__(self, loader: Callable[[], tuple[T, ...]]) -> None:
        self._loader = loader

    def __repr__(self) -> str:
        return repr(list(self._loader()))

    def __iter__(self) -> Iterator[T]:
        return iter(self._loader())

    def __len__(self) -> int:
        return len(self._loader())

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[T, ...]: ...

    def __getitem__(self, index: int | slice) -> T | tuple[T, ...]:
        return self._loader()[index]

    def __bool__(self) -> bool:
        return bool(self._loader())
