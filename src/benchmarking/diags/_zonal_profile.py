"""Provide timeseries diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.preprocessor import (
    climate_statistics,
    extract_levels,
    regrid,
    zonal_statistics,
)

from benchmarking.diags._base import DEFAULT_GRID, PLEV19_LEVELS, Diagnostic

if TYPE_CHECKING:
    from iris.cube import Cube


class ZonalProfileAirPressure(Diagnostic):
    """Manage zonal profile diagnostics."""

    FINAL_COORDINATES = ("air_pressure", "latitude")

    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess data."""
        cube = extract_levels(
            cube,
            PLEV19_LEVELS,
            "linear",
            coordinate="air_pressure",
        )
        cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
        cube = zonal_statistics(cube, "mean")
        cube = climate_statistics(cube, operator="mean", period="full")
        return cube  # noqa: RET504
