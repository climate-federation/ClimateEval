"""Configure."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import resources
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable
    from pathlib import Path


@dataclass(frozen=True, order=True)
class Coordinate:
    """Manage coordinate."""

    var_name: str
    standard_name: str = field(init=False, repr=False, compare=False)
    units: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize class instance."""
        if self.var_name not in COORDINATES:
            msg = f"Coordinate '{self.var_name}' not supported yet"
            raise NotImplementedError(msg)

        # See https://docs.python.org/3/library/dataclasses.html#frozen-instances
        for key, val in COORDINATES[self.var_name].items():
            object.__setattr__(self, key, val)


@dataclass(frozen=True, order=True)
class Variable:
    """Manage variables."""

    id: str
    var_name: str = field(repr=False)
    frequency: str = field(repr=False)
    level: str = field(default="*", repr=False, kw_only=True)
    min_lat: float = field(default=-90.0, repr=False, kw_only=True)
    max_lat: float = field(default=90.0, repr=False, kw_only=True)
    min_lon: float = field(default=0.0, repr=False, kw_only=True)
    max_lon: float = field(default=360.0, repr=False, kw_only=True)
    timerange: str = field(default="*", repr=False, kw_only=True)
    coordinates: list[Coordinate] = field(init=False, repr=False, compare=False)
    realm: str = field(init=False, repr=False, compare=False)
    standard_name: str = field(init=False, repr=False, compare=False)
    units: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize class instance."""
        if self.var_name not in VARIABLES:
            msg = f"Variable '{self.var_name}' not supported yet"
            raise NotImplementedError(msg)
        var_info = VARIABLES[self.var_name]

        # See https://docs.python.org/3/library/dataclasses.html#frozen-instances
        object.__setattr__(
            self,
            "coordinates",
            [Coordinate(c) for c in var_info["coordinates"]],
        )
        for key, val in var_info.items():
            if key == "coordinates":
                continue
            object.__setattr__(self, key, val)


def _read_yml(yml_path: Path | Traversable) -> dict[str, dict[str, str | list[str]]]:
    """Read YAML files."""
    with yml_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


_VARIABLES_YML_PATH = resources.files("benchmarking") / "variables.yml"
_COORDINATES_YML_PATH = resources.files("benchmarking") / "coordinates.yml"

VARIABLES = _read_yml(_VARIABLES_YML_PATH)
COORDINATES = _read_yml(_COORDINATES_YML_PATH)
