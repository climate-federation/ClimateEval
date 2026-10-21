"""Provide ocean heat transport diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import iris
from esmvalcore.preprocessor import (
    annual_statistics,
    climate_statistics,
    convert_units,
    regrid_time,
)

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._variable import COORDINATES
from climateeval.diags._utils import (
    ATLANTIC_BASIN,
    RAPID_LATITUDE,
    extract_basin_at_latitude,
)
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


class OceanHeatTransportZonal(SimpleDiagnostic):
    """Ocean heat transport (``hfbasin``) zonal profile diagnostics.

    Calculate the climatological northward ocean heat transport in the
    Atlantic and Arctic Oceans as a function of latitude (in PW).
    """

    _final_coordinates = (Coordinate("lat"),)

    _units: ClassVar[str] = "PW"

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Calculate climate statistics over time while keeping all latitude points.

        Expects coordinates: [time, basin, lat] and units in Watt.
        """
        with setup_esmvaltool_config_and_logging():
            cube = cube.extract(iris.Constraint(region=ATLANTIC_BASIN))
            cube = convert_units(cube, self._units)
            cube.long_name = "Climatological_Northward_Ocean_Heat_Transport_Basin"

            if cube.coords("basin"):
                cube.remove_coord("basin")
            cube = climate_statistics(cube, "mean", "full")
        return cube  # noqa: RET504


class AtlanticHeatTransportTimeseries(SimpleDiagnostic):
    """Atlantic ocean heat transport (``hfbasin``) time series at 26.5N.

    Extracts the Atlantic-Arctic basin at the grid latitude closest to the
    RAPID array (26.5N) and computes the annual-mean northward ocean heat
    transport in PW.
    """

    _final_coordinates = (Coordinate("time"),)

    _units: ClassVar[str] = "PW"

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = extract_basin_at_latitude(cube, ATLANTIC_BASIN, RAPID_LATITUDE)
            cube = convert_units(cube, self._units)
            cube = annual_statistics(cube, "mean")
            cube = regrid_time(
                cube,
                frequency="yr",
                calendar="standard",
                units=COORDINATES["time"]["units"],
            )
        return cube  # noqa: RET504
