"""Provide timeseries diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.preprocessor import (
    area_statistics,
    climate_statistics,
    local_solar_time,
    regrid,
)

from benchmarking.diags._base import DEFAULT_GRID, Diagnostic

if TYPE_CHECKING:
    from iris.cube import Cube


class DiurnalCycle(Diagnostic):
    """Manage diurnal cycle diagnostics."""

    FINAL_COORDINATES = ("hour",)

    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess data."""
        cube = regrid(
            cube,
            DEFAULT_GRID,
            "linear",
            cache_weights=True,
        )  # to ensure we are on a regular grid
        cube = local_solar_time(cube)
        cube = area_statistics(cube, "mean")
        cube = climate_statistics(cube, operator="mean", period="hour")
        return cube  # noqa: RET504
