"""Benchmark Earth system models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from climateeval._variable import (
        NO_LEVEL,
        Coordinate,
        Variable,
        VerticalLevel,
    )

__all__ = [
    "NO_LEVEL",
    "Coordinate",
    "Variable",
    "VerticalLevel",
]


def __getattr__(name: str) -> Any:
    """Lazily re-export the public names (PEP 562).

    ``climateeval._variable`` reads ``variables.yml``/``coordinates.yml`` (and
    logs) at import time; deferring it keeps a bare ``import climateeval``
    side-effect free so entry points can configure logging first.
    """
    if name in __all__:
        from climateeval import _variable  # noqa: PLC0415

        return getattr(_variable, name)
    msg = f"module 'climateeval' has no attribute {name!r}"
    raise AttributeError(msg)
