"""Provide timeseries diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.preprocessor import (
    annual_statistics,
    area_statistics,
    regrid,
    regrid_time,
)

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._variable import COORDINATES
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


class AnnualMeanTimeSeries(SimpleDiagnostic):
    """Manage annual mean timeseries diagnostics."""

    _FINAL_COORDINATES = (Coordinate("time"),)

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(
                cube,
                DEFAULT_GRID,
                "linear",
                cache_weights=True,
            )  # to ensure we are on a regular grid
            cube = area_statistics(cube, "mean")
            cube = annual_statistics(cube, "mean")
            cube = regrid_time(
                cube,
                frequency="yr",
                calendar="standard",
                units=COORDINATES["time"]["units"],
            )
        return cube  # noqa: RET504


class TimeSeriesNoPreprocessing(SimpleDiagnostic):
    """Manage timeseries diagnostics without preprocessing."""

    _FINAL_COORDINATES = (Coordinate("time"),)

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        return cube
