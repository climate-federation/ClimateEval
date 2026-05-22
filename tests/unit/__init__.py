from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from iris.cube import Cube


def raise_value_error(_cube: Cube, **__: Any) -> None:
    """Raise :class:`ValueError`."""
    msg = "value error"
    raise ValueError(msg)
