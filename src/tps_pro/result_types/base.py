"""Base mixin for frozen dataclasses with dict-style backward compatibility."""

from __future__ import annotations

import dataclasses
import typing
import warnings
from typing import Any, cast


class _DictAccessMixin:
    """Mixin that provides dict-style [] and .get() access on frozen dataclasses.

    This enables gradual migration: callers can use obj["field"] or obj.field
    interchangeably until all bracket access is replaced with attribute access.

    Deprecation timeline:
        v3.1 (current) -- deprecation warnings emitted once per class
        v4.0           -- bracket/get access removed; callers must use attributes
    """

    # Cache for typing.get_type_hints() results, keyed by class.
    # Avoids re-evaluating string annotations on every from_dict() call.
    _type_hints_cache: dict[type, dict[str, Any]] = {}

    # Per-class deprecation warning flags (overridden per subclass).
    _bracket_warned: bool = False
    _get_warned: bool = False

    def __getitem__(self, key: str) -> Any:
        # Fast path: skip warning machinery entirely for classes that already warned.
        cls = type(self)
        if not getattr(cls, "_bracket_warned", False):
            cls._bracket_warned = True
            warnings.warn(
                f"Bracket access on {cls.__name__} is deprecated. "
                "Use attribute access instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def get(self, key: str, default: Any = None) -> Any:
        # Fast path: skip warning machinery entirely for classes that already warned.
        cls = type(self)
        if not getattr(cls, "_get_warned", False):
            cls._get_warned = True
            warnings.warn(
                f".get() access on {cls.__name__} is deprecated. "
                "Use attribute access instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return getattr(self, key, default)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        raise TypeError(
            f"{type(self).__name__} is frozen."
            " Use dataclasses.replace() instead of"
            " item assignment."
        )

    def update(self, other: dict[str, Any]) -> None:
        """No-op guard: raises TypeError since frozen dataclasses can't be mutated.

        Use dataclasses.replace() instead.
        """
        raise TypeError(
            f"{type(self).__name__} is frozen."
            " Use dataclasses.replace() instead of"
            " .update()."
        )

    # Zero-value fallbacks for required fields whose type is known.
    # Used by the generic from_dict when a field has no dataclass default
    # and the key is absent from the input dict.
    _ZERO_VALUES: dict = {
        int: 0,
        float: 0.0,
        str: "",
        bool: False,
    }

    @classmethod
    def _get_type_hints(cls) -> dict[str, Any]:
        """Return cached type hints for this class."""
        if cls not in _DictAccessMixin._type_hints_cache:
            _DictAccessMixin._type_hints_cache[cls] = typing.get_type_hints(cls)
        return _DictAccessMixin._type_hints_cache[cls]

    def to_dict(self) -> dict[str, Any]:
        """Serialize this frozen dataclass to a plain dict via dataclasses.asdict."""
        return dataclasses.asdict(cast(Any, self))

    @classmethod
    def from_dict(cls, data: dict) -> _DictAccessMixin:
        """Generic constructor from dict, using field names and defaults.

        For required fields missing from *data*, a type-appropriate zero value
        is used (0 for int, 0.0 for float, ``""`` for str, ``False`` for bool).
        ``T | None`` fields without an explicit default fall back to
        ``None``.

        Subclasses with nested dataclass fields (e.g. lists of other
        dataclasses) should override this method to deserialize those fields.
        """
        kwargs: dict = {}
        type_hints = cls._get_type_hints()
        for f in dataclasses.fields(cast(Any, cls)):
            if f.name in data:
                kwargs[f.name] = data[f.name]
            elif f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:
                kwargs[f.name] = f.default_factory()
            else:
                # Required field missing from data -- try type-based zero value
                hint = type_hints.get(f.name)
                origin = getattr(hint, "__origin__", None)
                if origin is typing.Union:
                    # X | None is Union[X, None] -- use None
                    kwargs[f.name] = None
                elif hint in cls._ZERO_VALUES:
                    kwargs[f.name] = cls._ZERO_VALUES[hint]
                # else: omit and let cls() raise TypeError for truly unknown types
        return cls(**kwargs)
