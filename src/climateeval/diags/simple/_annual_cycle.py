"""Provide annual cycle diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.preprocessor import (
    area_statistics,
    climate_statistics,
    regrid,
)

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube


class AnnualCycle(SimpleDiagnostic):
    """Manage annual cycle diagnostics."""

    FINAL_COORDINATES = (Coordinate("month_number"),)

    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(
                cube,
                DEFAULT_GRID,
                "linear",
                cache_weights=True,
            )  # to ensure we are on a regular grid
            cube = area_statistics(cube, "mean")
            cube = climate_statistics(cube, "mean", "month")
        return cube  # noqa: RET504
