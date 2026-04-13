"""Provide map diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.preprocessor import (
    climate_statistics,
    regrid,
)

from benchmarking.diags._base import DEFAULT_GRID, Diagnostic

if TYPE_CHECKING:
    from iris.cube import Cube


class Map(Diagnostic):
    """Manage map diagnostics."""

    FINAL_COORDINATES = ("latitude", "longitude")

    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess data."""
        cube = regrid(cube, DEFAULT_GRID, "linear")
        cube = climate_statistics(cube, "mean", "full")
        return cube  # noqa: RET504
