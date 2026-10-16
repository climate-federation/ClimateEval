"""Provide Ocean Heat Transport spatial profile diagnostics using hfbasin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris
import numpy as np
from esmvalcore.preprocessor import annual_statistics, climate_statistics, regrid_time

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._variable import COORDINATES
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


class OceanHeatTransportZonal(SimpleDiagnostic):
    """Ocean heat transport (hfbasin) diagnostics.

    Calculate the climatological OHT in the Atlantic and Arctic Oceans.
    """

    _final_coordinates = (Coordinate("lat"),)

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Calculate climate statistics over time while keeping all latitude points.

        Expects coordinates: [time, basin, lat] and units in Watt.
        """
        with setup_esmvaltool_config_and_logging():
            cube = cube.extract(iris.Constraint(region="atlantic_arctic_ocean"))
            cube.data = cube.core_data() / 1e15
            cube.units = "PW"
            cube.long_name = "Climatological_Northward_Ocean_Heat_Transport_Basin"

            if cube.coords("basin"):
                cube.remove_coord("basin")
            cube = climate_statistics(cube, "mean", "full")
        return cube  # noqa: RET504

class AtlanticHeatTransportTimeseries(SimpleDiagnostic):
    """Ocean heat transport (hfbasin) diagnostics.

    Calculate the climatological OHT in the Atlantic and Arctic Oceans.
    """

    _final_coordinates = (Coordinate("time"),)

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Calculate climate statistics over time while keeping all latitude points.

        Expects coordinates: [time, basin, lat] and units in Watt.
        """
        with setup_esmvaltool_config_and_logging():
            cube = cube.extract(iris.Constraint(region="atlantic_arctic_ocean"))
            cube.data = cube.core_data() / 1e15
            cube.units = "PW"

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

            cube = annual_statistics(cube, "mean")
            cube = regrid_time(
                cube,
                frequency="yr",
                calendar="standard",
                units=COORDINATES["time"]["units"],
            )

        return cube # noqa: RET504
