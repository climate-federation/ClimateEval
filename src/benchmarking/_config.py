"""Configure."""

from dataclasses import dataclass, field

import numpy as np
from esmvalcore.config import CFG


@dataclass(frozen=True, order=True)
class VerticalLevel:
    """Manage vertical levels."""

    coordinate_name: str
    units: str
    value: float


@dataclass(frozen=True, order=True)
class _NoLevel(VerticalLevel):
    """Represent no vertical levels."""

    coordinate_name: str = field(default="", init=False)
    units: str = field(default="", init=False)
    value: float = field(default=np.nan, init=False)

    def __repr__(self) -> str:
        """Get string representation."""
        return "NO_LEVEL"


NO_LEVEL = _NoLevel()
"""Represent no vertical level."""

# See https://docs.python.org/3/library/dataclasses.html#frozen-instances
object.__setattr__(NO_LEVEL, "__repr__", lambda _: "NO_LEVEL")
object.__setattr__(NO_LEVEL, "__str__", lambda _: "NO_LEVEL")


def reset_esmvaltool_config() -> None:
    """Rest ESMValTool configuration to default values."""
    CFG.load_from_dirs([])
