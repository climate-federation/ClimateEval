"""Provide AMOC diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import iris
import numpy as np
from esmvalcore.preprocessor import (
    annual_statistics,
    climate_statistics,
    extract_levels,
    regrid_time,
    convert_units,
)

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._variable import COORDINATES
from climateeval.diags._utils import DEPTH_LEVELS
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
    """Manage AMOC time series diagnostics.

    Computes the annual-mean Atlantic Meridional Overturning Circulation at the
    RAPID array (26.5N). The ``amoc`` variable is derived (by ESMValTool) from
    the meridional overturning mass streamfunction (``msftmz``/``msftyz``) and
    is already reduced to a time series, so this diagnostic only takes the
    annual mean and converts to ``Tg s-1``.
    """

    _final_coordinates = (Coordinate("depth"),)

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():

            cube = extract_levels(cube, DEPTH_LEVELS, "linear", coordinate="depth")
            cube = convert_units(cube, self._units)
            cube = cube.extract(iris.Constraint(region="atlantic_arctic_ocean"))

            """Preprocess 3D streamfunction data and slice at 26.5N."""
            target_lat = 26.5

            lat_coord = cube.coord("latitude")
            lat_dims_tuple = cube.coord_dims(lat_coord)

            if lat_dims_tuple:
                closest_lat_idx = np.argmin(np.abs(lat_coord.points - target_lat))
                lat_dim = int(lat_dims_tuple[0])

                slice_structure = [slice(None)] * cube.ndim
                slice_structure[lat_dim] = int(closest_lat_idx)
                cube = cube[tuple(slice_structure)]

            return climate_statistics(cube, "mean", "full")
