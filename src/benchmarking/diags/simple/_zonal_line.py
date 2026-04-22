"""Provide zonal line diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.preprocessor import (
    climate_statistics,
    regrid,
    zonal_statistics,
)

from benchmarking import Coordinate, setup_esmvaltool_config_and_logging
from benchmarking.diags._utils import DEFAULT_GRID
from benchmarking.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube


class ZonalLine(SimpleDiagnostic):
    """Manage zonal line diagnostics."""

    FINAL_COORDINATES = (Coordinate("lat"),)

    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
            cube = zonal_statistics(cube, "mean")
            cube = climate_statistics(cube, operator="mean", period="full")
        return cube  # noqa: RET504
