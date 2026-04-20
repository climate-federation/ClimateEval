"""Provide annual cycle diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.preprocessor import (
    area_statistics,
    climate_statistics,
    regrid,
)

from benchmarking import Coordinate, setup_esmvaltool_config_and_logging
from benchmarking.diags._base import Diagnostic
from benchmarking.diags._utils import DEFAULT_GRID

if TYPE_CHECKING:
    from iris.cube import Cube


class AnnualCycle(Diagnostic):
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
            cube = climate_statistics(cube, operator="mean", period="month")
        return cube  # noqa: RET504
