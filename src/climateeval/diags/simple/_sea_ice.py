"""Provide sea-ice diagnostics.

These diagnostics evaluate total sea-ice area over a hemisphere (the region is
selected per variable via ``min_lat``/``max_lat`` in the suite). Unlike the
generic area-mean diagnostics, sea-ice area is obtained by an area-weighted
*sum* of the sea-ice concentration and converted to ``1e6 km2``.

To evaluate a single calendar month (e.g. the September minimum in the Northern
Hemisphere or the February minimum in the Southern Hemisphere), add an
``esmvalcore.preprocessor.extract_month`` step to the variable's
``additional_preprocessors`` in the suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from esmvalcore.preprocessor import (
    annual_statistics,
    area_statistics,
    climate_statistics,
    convert_units,
    regrid,
    regrid_time,
)

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._variable import COORDINATES
from climateeval.diags._utils import SEA_ICE_GRID
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


#: Target units of the integrated sea-ice area.
_SEA_ICE_AREA_UNITS = "1e6 km2"


class SeaIceAreaAnnualCycle(SimpleDiagnostic):
    """Manage sea-ice area annual cycle diagnostics.

    Computes the monthly climatology of the total sea-ice area over the region
    selected for the variable (e.g. a hemisphere), in ``1e6 km2``.
    """

    _final_coordinates = (Coordinate("month_number"),)

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(
                cube,
                SEA_ICE_GRID,
                "linear",
                cache_weights=True,
            )  # to ensure we are on a regular grid
            cube = area_statistics(cube, "sum")
            cube = convert_units(cube, _SEA_ICE_AREA_UNITS)
            cube = climate_statistics(cube, "mean", "month")
        return cube  # noqa: RET504


class SeaIceAreaTimeSeries(SimpleDiagnostic):
    """Manage sea-ice area annual mean time series diagnostics.

    Computes an annual time series of the total sea-ice area over the region
    selected for the variable, in ``1e6 km2``. Combine with an
    ``extract_month`` additional preprocessor in the suite to obtain the time
    series of a single calendar month (e.g. the seasonal sea-ice minimum).
    """

    _final_coordinates = (Coordinate("time"),)

    _annual_operator: ClassVar[str] = "mean"

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(
                cube,
                SEA_ICE_GRID,
                "linear",
                cache_weights=True,
            )  # to ensure we are on a regular grid
            cube = area_statistics(cube, "sum")
            cube = convert_units(cube, _SEA_ICE_AREA_UNITS)
            cube = annual_statistics(cube, self._annual_operator)
            cube = regrid_time(
                cube,
                frequency="yr",
                calendar="standard",
                units=COORDINATES["time"]["units"],
            )
        return cube  # noqa: RET504
