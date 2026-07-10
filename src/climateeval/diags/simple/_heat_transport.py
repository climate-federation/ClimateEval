"""Provide Ocean Heat Transport spatial profile diagnostics using hfbasin."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris
import numpy as np
from esmvalcore.preprocessor import climate_statistics

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


class OceanHeatTransportProfile(SimpleDiagnostic):
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
            cube.long_name = (
                "Climatological_Northward_Ocean_Heat_Transport_Basin"
            )

            if cube.coords("basin"):
                cube.remove_coord("basin")
            cube = climate_statistics(cube, "mean", "full")
        return cube  # noqa: RET504
