"""Configure."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from importlib import resources
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from esmvalcore.preprocessor._derive import get_required  # not part of public API yet

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable
    from pathlib import Path


@dataclass(frozen=True, order=True)
class VerticalLevel:
    """Manage vertical levels."""

    coordinate_name: str
    units: str
    value: float

    def __str__(self) -> str:
        """Get human-readable string."""
        return f"{self.coordinate_name} = {self.value} {self.units}"


@dataclass(frozen=True, order=True)
class _NoLevel(VerticalLevel):
    """Represent no vertical levels."""

    coordinate_name: str = field(default="", init=False)
    units: str = field(default="", init=False)
    value: float = field(default=np.nan, init=False)

    def __repr__(self) -> str:
        """Get string representation."""
        return "NO_LEVEL"

    def __str__(self) -> str:
        """Get human-readable string."""
        return ""


NO_LEVEL = _NoLevel()
"""Represent no vertical level."""


@dataclass(frozen=True, order=True)
class Coordinate:
    """Manage coordinates."""

    var_name: str
    long_name: str = field(default="", init=False, repr=False, compare=False)
    standard_name: str = field(default="", init=False, repr=False, compare=False)
    units: str = field(default="", init=False, repr=False, compare=False)
    allowed_coordinates: tuple[Coordinate, ...] = field(
        default_factory=tuple,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Initialize class instance."""
        if self.var_name not in COORDINATES:
            msg = f"Coordinate '{self.var_name}' not supported yet"
            raise NotImplementedError(msg)

        # See https://docs.python.org/3/library/dataclasses.html#frozen-instances
        coord_info = COORDINATES[self.var_name]
        for key, val in coord_info.items():
            if key == "allowed_coordinates":
                continue
            object.__setattr__(self, key, val)
        if "allowed_coordinates" in coord_info:
            object.__setattr__(
                self,
                "allowed_coordinates",
                tuple(Coordinate(c) for c in coord_info["allowed_coordinates"]),
            )

    def name(self) -> str:
        """Get human-readable name of coordinate."""
        if self.standard_name:
            return self.standard_name
        if self.long_name:
            return self.long_name
        return self.var_name


@dataclass(frozen=True, order=True)
class Variable:
    """Manage variables."""

    id: str
    var_name: str = field(repr=False)
    frequency: str = field(repr=False)

    # Settings
    settings: dict[str, Any] = field(init=False, repr=False, compare=False)
    vertical_level: VerticalLevel = field(default=NO_LEVEL, repr=False, kw_only=True)
    min_lat: float = field(default=-90.0, repr=False, kw_only=True)
    max_lat: float = field(default=90.0, repr=False, kw_only=True)
    min_lon: float = field(default=0.0, repr=False, kw_only=True)
    max_lon: float = field(default=360.0, repr=False, kw_only=True)
    timerange: str = field(default="*", repr=False, kw_only=True)
    units: str = field(default="", repr=False, kw_only=True)

    # Automatically filled fields
    coordinates: list[Coordinate] = field(init=False, repr=False, compare=False)
    derived: bool = field(default=False, init=False, repr=False, compare=False)
    long_name: str = field(init=False, repr=False, compare=False)
    realm: str = field(init=False, repr=False, compare=False)
    standard_name: str = field(init=False, repr=False, compare=False)

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
            if key == "units" and self.units != "":
                continue
            object.__setattr__(self, key, val)
        object.__setattr__(self, "settings", self._get_settings())

    def _get_settings(self) -> dict[str, Any]:
        """Get settings."""
        settings: dict[str, Any] = {}
        for field_ in fields(self):
            if field_.init and field_.kw_only:
                settings[field_.name] = getattr(self, field_.name)
        return settings

    def get_required_variables(self) -> tuple[Variable, ...]:
        """Get variables necessary for derivation."""
        if not self.derived:
            return (self,)
        required_facets = get_required(self.var_name, "CMIP6")
        return tuple(
            Variable(
                v["short_name"],
                v["short_name"],
                self.frequency,
                **self.settings,
            )
            for v in required_facets
        )


def _read_yml(yml_path: Path | Traversable) -> dict[str, dict[str, str | list[str]]]:
    """Read YAML files."""
    with yml_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


_VARIABLES_YML_PATH = resources.files("benchmarking") / "variables.yml"
_COORDINATES_YML_PATH = resources.files("benchmarking") / "coordinates.yml"

VARIABLES = _read_yml(_VARIABLES_YML_PATH)
COORDINATES = _read_yml(_COORDINATES_YML_PATH)
