"""Provide diurnal cycle diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.preprocessor import (
    area_statistics,
    climate_statistics,
    local_solar_time,
    regrid,
)

from benchmarking import Coordinate, setup_esmvaltool_config_and_logging
from benchmarking.diags._utils import DEFAULT_GRID
from benchmarking.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube


class DiurnalCycle(SimpleDiagnostic):
    """Manage diurnal cycle diagnostics."""

    FINAL_COORDINATES = (Coordinate("hour"),)

    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(
                cube,
                DEFAULT_GRID,
                "linear",
                cache_weights=True,
            )  # to ensure we are on a regular grid
            cube = local_solar_time(cube)
            cube = area_statistics(cube, "mean")
            cube = climate_statistics(cube, "mean", "hour")
        return cube  # noqa: RET504
