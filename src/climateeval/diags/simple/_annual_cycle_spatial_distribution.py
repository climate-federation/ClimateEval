"""Provide annual cycle spatial distribution diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from climateeval import Coordinate, setup_esmvaltool_config_and_logging
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.simple._base import SimpleDiagnostic
from esmvalcore.preprocessor import (
    climate_statistics,
    regrid,
)

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
            cube = climate_statistics(cube, "mean", "month")
        return cube  # noqa: RET504
