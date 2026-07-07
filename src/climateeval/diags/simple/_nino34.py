"""Provide Nino 3.4 diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from esmvalcore.preprocessor import (
    anomalies,
    area_statistics,
    extract_region,
    regrid,
    regrid_time,
    rolling_window_statistics,
)

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._variable import COORDINATES
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


class Nino34(SimpleDiagnostic):
    """Manage Nino 3.4 (Oceanic Nino Index) diagnostics.

    Computes the Oceanic Nino Index time series: monthly sea surface temperature
    anomalies averaged over the Nino 3.4 region (170W-120W, 5S-5N) and smoothed
    with a 3-month running mean.
    """

    _final_coordinates = (Coordinate("time"),)

    # Nino 3.4 region (longitudes given in [0, 360], i.e. 170W-120W, 5S-5N).
    _start_longitude: ClassVar[float] = 190.0
    _end_longitude: ClassVar[float] = 240.0
    _start_latitude: ClassVar[float] = -5.0
    _end_latitude: ClassVar[float] = 5.0
    _rolling_window_length: ClassVar[int] = 3

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(
                cube,
                DEFAULT_GRID,
                "linear",
                cache_weights=True,
            )  # to ensure we are on a regular grid
            cube = extract_region(
                cube,
                start_longitude=self._start_longitude,
                end_longitude=self._end_longitude,
                start_latitude=self._start_latitude,
                end_latitude=self._end_latitude,
            )
            cube = anomalies(cube, period="month")
            cube = rolling_window_statistics(
                cube,
                coordinate="time",
                operator="mean",
                window_length=self._rolling_window_length,
            )
            cube = area_statistics(cube, "mean")
            cube = regrid_time(
                cube,
                frequency="mon",
                calendar="standard",
                units=COORDINATES["time"]["units"],
            )
        return cube  # noqa: RET504
