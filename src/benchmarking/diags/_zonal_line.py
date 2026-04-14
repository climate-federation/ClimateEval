"""Provide timeseries diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.preprocessor import (
    climate_statistics,
    regrid,
    zonal_statistics,
)

from benchmarking import Coordinate
from benchmarking.diags._base import DEFAULT_GRID, Diagnostic

if TYPE_CHECKING:
    from iris.cube import Cube


class ZonalLine(Diagnostic):
    """Manage zonal line diagnostics."""

    FINAL_COORDINATES = (Coordinate("lat"),)

    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess data."""
        cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
        cube = zonal_statistics(cube, "mean")
        cube = climate_statistics(cube, operator="mean", period="full")
        return cube  # noqa: RET504
