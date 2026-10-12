"""Provide map diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cf_units import Unit
from esmvalcore.preprocessor import (
    annual_statistics,
    climate_statistics,
    convert_units,
    linear_trend,
    regrid,
    regrid_time,
)

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._variable import COORDINATES
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


class Map(SimpleDiagnostic):
    """Manage map diagnostics."""

    _final_coordinates = (Coordinate("lat"), Coordinate("lon"))

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
            cube = climate_statistics(cube, "mean", "full")
        return cube  # noqa: RET504

class TrendMap(SimpleDiagnostic):
    """Manage trend map diagnostics."""

    _final_coordinates = (Coordinate("lat"), Coordinate("lon"))

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
            cube = annual_statistics(cube, "mean")
            cube = regrid_time(
                cube,
                frequency="yr",
                calendar="standard",
                units=COORDINATES["time"]["units"],
            )
            units = cube.units
            cube = linear_trend(cube)
            cube = convert_units(cube, units / Unit("yr"))
        return cube  # noqa: RET504
