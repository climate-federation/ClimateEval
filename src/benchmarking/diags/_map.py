"""Provide map diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.preprocessor import (
    climate_statistics,
    regrid,
)

from benchmarking import Coordinate, setup_esmvaltool_config_and_logging
from benchmarking.diags._base import Diagnostic
from benchmarking.diags._utils import DEFAULT_GRID

if TYPE_CHECKING:
    from iris.cube import Cube


class Map(Diagnostic):
    """Manage map diagnostics."""

    FINAL_COORDINATES = (Coordinate("lat"), Coordinate("lon"))

    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
            cube = climate_statistics(cube, "mean", "full")
        return cube  # noqa: RET504
