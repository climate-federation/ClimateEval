"""Provide annual cycle spatial distribution diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.preprocessor import (
    climate_statistics,
    regrid,
)

from benchmarking import Coordinate, setup_esmvaltool_config_and_logging
from benchmarking.diags._utils import DEFAULT_GRID
from benchmarking.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube


class AnnualCycleSpatialDistribution(SimpleDiagnostic):
    """Manage annual cycle spatial distribution diagnostics."""

    FINAL_COORDINATES = (
        Coordinate("month_number"),
        Coordinate("lat"),
        Coordinate("lon"),
    )

    OUTPUT_DATA_TABLE = False

    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(
                cube,
                DEFAULT_GRID,
                "linear",
                cache_weights=True,
            )
            cube = climate_statistics(cube, operator="mean", period="month")
        return cube  # noqa: RET504
