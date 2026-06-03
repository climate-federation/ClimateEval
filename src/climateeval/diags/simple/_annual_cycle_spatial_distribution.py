"""Provide annual cycle spatial distribution diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.preprocessor import climate_statistics, regrid

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


class AnnualCycleSpatialDistribution(SimpleDiagnostic):
    """Manage annual cycle spatial distribution diagnostics."""

    _final_coordinates = (
        Coordinate("month_number"),
        Coordinate("lat"),
        Coordinate("lon"),
    )

    _output_data_table = False

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
            cube = climate_statistics(cube, "mean", "month")
        return cube  # noqa: RET504
