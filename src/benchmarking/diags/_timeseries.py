"""Benchmarking Diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.preprocessor import (
    annual_statistics,
    area_statistics,
    regrid,
    regrid_time,
)

from benchmarking import COORDINATES
from benchmarking.diags import Diagnostic

if TYPE_CHECKING:
    from iris.cube import Cube


class TimeSeries(Diagnostic):
    """Manage atmosphere timeseries diagnostics."""

    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess data."""
        cube = regrid(cube, "2x2", "linear")  # to ensure we are on a regular grid
        cube = area_statistics(cube, "mean")
        cube = annual_statistics(cube, "mean")
        cube = regrid_time(
            cube,
            frequency="yr",
            calendar="standard",
            units=COORDINATES["time"]["units"],
        )
        return cube  # noqa: RET504
