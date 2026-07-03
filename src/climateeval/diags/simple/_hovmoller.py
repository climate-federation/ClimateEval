"""Provide Hovmöller (longitude × time) diagnostics.

A Hovmöller diagram shows the temporal evolution of a zonally-varying field
averaged over a chosen latitude band. Typical use cases:

* MJO propagation: OLR or precipitation, 5-day running mean, 15°S–15°N
* Kelvin waves: precipitation, 3-day running mean, 5°S–5°N
* ENSO: SST, 30-day running mean, 5°S–5°N

Processing pipeline per variable:
  1. Regrid to a regular 2×2 grid.
  2. Extract the latitude band [lat_min, lat_max].
  3. Meridional mean (area-weighted collapse of latitude).
  4. Remove the seasonal cycle: subtract per-day-of-year climatology.
  5. Optionally apply a rolling mean along the time axis.

Output cube coordinates: (time, longitude).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris.analysis
import numpy as np
from esmvalcore.preprocessor import (
    extract_region,
    meridional_statistics,
    regrid,
)

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


def _remove_seasonal_cycle(cube: Cube) -> Cube:
    """Subtract per-day-of-year climatology from each time step."""
    time_coord = cube.coord("time")
    dates = time_coord.units.num2date(time_coord.points)
    doys = np.array([d.timetuple().tm_yday for d in dates])

    data = cube.data.copy().astype(float)
    for doy in np.unique(doys):
        mask = doys == doy
        data[mask] -= np.nanmean(data[mask], axis=0)

    return cube.copy(data=data)


def _apply_rolling_mean(cube: Cube, window: int) -> Cube:
    """Apply a rolling mean of `window` steps along the time axis.

    Returns the cube unchanged when window is 0 or 1.
    """
    if window < 2:
        return cube
    return cube.rolling_window("time", iris.analysis.MEAN, window)


class Hovmoller(SimpleDiagnostic):
    """Hovmöller diagnostic: longitude × time heatmap of meridional-mean anomalies."""

    _final_coordinates = (Coordinate("time"), Coordinate("lon"))

    def __init__(
        self,
        name,
        variables_and_data,
        *,
        lat_min: float = -15.0,
        lat_max: float = 15.0,
        rolling_window_days: int = 5,
        **kwargs,
    ) -> None:
        """Initialize with optional latitude band and rolling-mean window.

        Parameters
        ----------
        lat_min, lat_max:
            Latitude bounds for the meridional average (degrees N).
        rolling_window_days:
            Number of days for the rolling mean. 0 = no smoothing.
        """
        self._lat_min = lat_min
        self._lat_max = lat_max
        self._rolling_window_days = rolling_window_days
        super().__init__(name, variables_and_data, **kwargs)

    def _preprocess(self, cube: Cube, variable: Variable) -> Cube:  # noqa: ARG002
        """Preprocess to (time, longitude) Hovmöller field."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
            cube = extract_region(
                cube,
                start_longitude=0.0,
                end_longitude=360.0,
                start_latitude=self._lat_min,
                end_latitude=self._lat_max,
            )
            cube = meridional_statistics(cube, "mean")

        cube = _remove_seasonal_cycle(cube)
        cube = _apply_rolling_mean(cube, self._rolling_window_days)
        return cube
