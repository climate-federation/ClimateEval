"""Configure."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml
from esmvalcore.preprocessor._derive import get_required  # not part of public API yet
from loguru import logger

from climateeval._config import setup_esmvaltool_config_and_logging

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable


logger = logger.opt(colors=True)


@dataclass(frozen=True, order=True)
class VerticalLevel:
    """Manage vertical levels."""

    coordinate_name: str
    units: str
    value: float
    extrapolate: bool = field(default=False, repr=False, compare=False)

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


NO_LEVEL = _NoLevel()
"""Represent no vertical level."""


@dataclass(frozen=True, order=True)
class Coordinate:
    """Manage coordinates."""

    var_name: str

    # Fields that can be filled automatically if not set (additional metadata)
    long_name: str = field(default="", repr=False, compare=False)
    standard_name: str = field(default="", repr=False, compare=False)
    units: str = field(default="", repr=False, compare=False)
    allowed_coordinates: tuple[Coordinate, ...] = field(
        default_factory=tuple,
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
            if getattr(self, key):  # allow manually settings these
                continue
            if key == "allowed_coordinates":
                continue
            object.__setattr__(self, key, val)
        if "allowed_coordinates" in coord_info and not self.allowed_coordinates:
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

    # Custom preprocessing settings
    vertical_level: VerticalLevel = field(default=NO_LEVEL, repr=False, kw_only=True)
    min_lat: float = field(default=-90.0, repr=False, kw_only=True)
    max_lat: float = field(default=90.0, repr=False, kw_only=True)
    min_lon: float = field(default=0.0, repr=False, kw_only=True)
    max_lon: float = field(default=360.0, repr=False, kw_only=True)
    landsea_mask: str = field(default="*", repr=False, kw_only=True)
    ar6_regions_mask: tuple[str, ...] = field(
        default_factory=tuple,
        repr=False,
        kw_only=True,
    )
    timerange: str = field(default="*", repr=False, kw_only=True)
    units: str = field(default="", repr=False, kw_only=True)
    diagnostic_kwargs: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        kw_only=True,
        compare=False,
    )
    additional_preprocessors: dict[str, dict[str, Any]] = field(
        default_factory=dict,
        repr=False,
        kw_only=True,
        compare=False,
    )

    # Reasonable values that can be used downstream code (like plotting)
    reasonable_min: float = field(
        default=np.nan,
        repr=False,
        kw_only=True,
        compare=False,
    )
    reasonable_max: float = field(
        default=np.nan,
        repr=False,
        kw_only=True,
        compare=False,
    )

    # Fields that can be filled automatically if not set (additional metadata)
    coordinates: tuple[Coordinate] = field(  # type: ignore[assignment]
        default_factory=tuple,
        repr=False,
        compare=False,
    )
    derived: bool = field(default=False, repr=False, compare=False)
    long_name: str = field(default="", repr=False, compare=False)
    realm: str = field(default="", repr=False, compare=False)
    standard_name: str = field(default="", repr=False, compare=False)
    cmip6_table_id: str = field(default="", repr=False, compare=False)
    ndim: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize class instance."""
        if self.var_name not in VARIABLES:
            msg = f"Variable '{self.var_name}' not supported yet"
            raise NotImplementedError(msg)
        var_info = VARIABLES[self.var_name]

        # See https://docs.python.org/3/library/dataclasses.html#frozen-instances
        if not self.coordinates:
            object.__setattr__(
                self,
                "coordinates",
                tuple(Coordinate(c) for c in var_info["coordinates"]),
            )
        for key, val in var_info.items():
            if getattr(self, key):  # allow manually settings these
                continue
            if key == "coordinates":
                continue
            object.__setattr__(self, key, val)
        if not self.cmip6_table_id:
            object.__setattr__(self, "cmip6_table_id", self._get_cmip6_table_id())
        object.__setattr__(self, "ndim", len(self.coordinates))

    def get_required_variables(self) -> tuple[Variable, ...]:
        """Get variables necessary for derivation."""
        if not self.derived:
            return (self,)
        with setup_esmvaltool_config_and_logging():
            required_facets = get_required(self.var_name, "CMIP6")
        return tuple(
            Variable(
                v["short_name"],
                v["short_name"],
                self.frequency,
                **self.get_preprocessing_settings(),
            )
            for v in required_facets
        )

    def _get_cmip6_table_id(self) -> str:  # noqa: C901, PLR0911
        """Get CMIP6 table ID.

        Note
        ----
        Note all Table IDs are supported at the moment. See
        https://github.com/WCRP-CMIP/CMIP6_CVs/blob/main/CMIP6_table_id.json
        for a list of all available table IDs.

        """
        if self.realm == "":
            return ""
        if self.realm == "atmos":
            if self.frequency == "1hr":
                if self.var_name in ("pr", "prc"):
                    return "E1hr"
                if self.var_name in ("ps", "tas"):
                    return "AERhr"
            if self.frequency in ("3hr", "day", "fx"):
                return self.frequency
            if self.frequency == "mon":
                return "Amon"
        if self.realm in ("ocean", "ocnBgchem") and self.frequency in (
            "day",
            "fx",
            "mon",
            "yr",
        ):
            return f"O{self.frequency}"
        if self.realm == "land" and self.frequency == "mon":
            return "Lmon"
        if self.realm == "sea_ice" and self.frequency == "mon":
            return "SImon"
        msg = (
            f"Cannot determine CMIP6 table ID of {self}, this might be a problem "
            f"for some data sources"
        )
        logger.warning(msg)
        return ""

    def get_preprocessing_settings(self) -> dict[str, Any]:
        """Get preprocessing settings."""
        preprocessing_attributes = (
            "vertical_level",
            "min_lat",
            "max_lat",
            "min_lon",
            "max_lon",
            "landsea_mask",
            "ar6_regions_mask",
            "timerange",
            "units",
            "diagnostic_kwargs",
            "additional_preprocessors",
        )
        preprocessing_settings: dict[str, Any] = {}
        for attribute in preprocessing_attributes:
            preprocessing_settings[attribute] = getattr(self, attribute)
        return preprocessing_settings


def _read_coordinates() -> dict[str, dict[str, str | list[str]]]:
    """Read coordinates YAML file."""
    env_variable = "CLIMATEEVAL_COORDINATES_FILE"
    yml_path: Path | Traversable
    if env_variable in os.environ:
        yml_path = Path(os.environ[env_variable])
        logger.debug(
            f"Using custom coordinates file {yml_path} (defined by environment "
            f"variable '{env_variable}')",
        )
    else:
        yml_path = resources.files("climateeval") / "coordinates.yml"
        logger.debug(f"Using default coordinates file {yml_path}")
    return _read_yml(yml_path)


def _read_variables() -> dict[str, dict[str, str | list[str]]]:
    """Read variables YAML file."""
    env_variable = "CLIMATEEVAL_VARIABLES_FILE"
    yml_path: Path | Traversable
    if env_variable in os.environ:
        yml_path = Path(os.environ[env_variable])
        logger.debug(
            f"Using custom variables file {yml_path} (defined by environment "
            f"variable '{env_variable}')",
        )
    else:
        yml_path = resources.files("climateeval") / "variables.yml"
        logger.debug(f"Using default variables file {yml_path}")
    return _read_yml(yml_path)


def _read_yml(yml_path: Path | Traversable) -> dict[str, dict[str, str | list[str]]]:
    """Read YAML files."""
    with yml_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


COORDINATES = _read_coordinates()
VARIABLES = _read_variables()
