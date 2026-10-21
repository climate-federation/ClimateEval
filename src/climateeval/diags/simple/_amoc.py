"""Provide AMOC diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from esmvalcore.preprocessor import (
    annual_statistics,
    climate_statistics,
    convert_units,
    extract_levels,
    regrid_time,
)

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._variable import COORDINATES
from climateeval.diags._utils import (
    ATLANTIC_BASIN,
    DEPTH_LEVELS,
    RAPID_LATITUDE,
    extract_basin_at_latitude,
)
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


class AMOCTimeSeries(SimpleDiagnostic):
    """Manage AMOC time series diagnostics.

    Computes the annual-mean Atlantic Meridional Overturning Circulation at the
    RAPID array (26.5N). The ``amoc`` variable is derived (by ESMValTool) from
    the meridional overturning mass streamfunction (``msftmz``/``msftyz``) and
    is already reduced to a time series, so this diagnostic only takes the
    annual mean and converts to ``Tg s-1``.
    """

    _final_coordinates = (Coordinate("time"),)

    _units: ClassVar[str] = "Tg s-1"

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = annual_statistics(cube, "mean")
            cube = convert_units(cube, self._units)
            cube = regrid_time(
                cube,
                frequency="yr",
                calendar="standard",
                units=COORDINATES["time"]["units"],
            )
        return cube  # noqa: RET504


class AMOCDepthProfile(SimpleDiagnostic):
    """Manage AMOC depth profile diagnostics.

    Computes the climatological mean of the Atlantic meridional overturning
    mass streamfunction (``msftmz``) at the RAPID array (26.5N) as a function
    of depth, interpolated onto the common
    :data:`~climateeval.diags._utils.DEPTH_LEVELS` and converted to
    ``Tg s-1``.
    """

    _final_coordinates = (Coordinate("lev"),)

    _units: ClassVar[str] = "Tg s-1"

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = extract_basin_at_latitude(cube, ATLANTIC_BASIN, RAPID_LATITUDE)
            cube = climate_statistics(cube, "mean", "full")
            cube = extract_levels(cube, DEPTH_LEVELS, "linear", coordinate="depth")
            cube = convert_units(cube, self._units)
        return cube  # noqa: RET504
