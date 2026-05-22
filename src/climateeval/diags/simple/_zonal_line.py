"""Provide zonal line diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.preprocessor import climate_statistics, regrid, zonal_statistics

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


class ZonalLine(SimpleDiagnostic):
    """Manage zonal line diagnostics."""

    _FINAL_COORDINATES = (Coordinate("lat"),)

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
            cube = zonal_statistics(cube, "mean")
            cube = climate_statistics(cube, "mean", "full")
        return cube  # noqa: RET504
