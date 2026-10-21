"""Provide ocean depth profile diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import iris.analysis
from esmvalcore.preprocessor import (
    climate_statistics,
    extract_levels,
    extract_trajectory,
)

from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval.diags._utils import DEPTH_LEVELS
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iris.cube import Cube

    from climateeval import Variable
    from climateeval.diags.simple._base import SimpleDiagnosticInputData


class DepthProfile(SimpleDiagnostic):
    """Manage depth profile diagnostics.

    Computes the climatological mean of a 3-D ocean variable along a
    horizontal section given by two way points (``lats``/``lons``, sampled at
    ``n_points`` locations), averages over the section and interpolates the
    resulting profile onto the common
    :data:`~climateeval.diags._utils.DEPTH_LEVELS`. The defaults give the mean
    profile of the zonal section across the Atlantic at 26.5N (the RAPID
    array).

    Longitudes may be given in ``[-180, 180]`` or ``[0, 360]``; they are
    normalized to ``[0, 360)`` (the convention of the prepared cubes), so the
    section must not cross the Greenwich meridian.
    """

    _final_coordinates = (Coordinate("lev"),)

    def __init__(
        self,
        name: str,
        variables_and_data: dict[Variable, SimpleDiagnosticInputData],
        *,
        lats: Sequence[float] = (26.5, 26.5),
        lons: Sequence[float] = (-80.0, -15.0),
        n_points: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize class instance."""
        if len(lats) != 2 or len(lons) != 2:  # noqa: PLR2004
            msg = (
                f"Expected exactly two way points for 'lats' and 'lons', got "
                f"lats={list(lats)} and lons={list(lons)}"
            )
            raise ValueError(msg)
        self._lats = [float(lat) for lat in lats]
        self._lons = [float(lon) % 360.0 for lon in lons]
        self._n_points = int(n_points)
        super().__init__(name, variables_and_data, **kwargs)

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = climate_statistics(cube, "mean", "full")
            cube = extract_trajectory(cube, self._lats, self._lons, self._n_points)
            # The section points share a single dimension (with latitude and
            # longitude as auxiliary coordinates on it); average over it.
            cube = cube.collapsed(["latitude", "longitude"], iris.analysis.MEAN)
            cube = extract_levels(cube, DEPTH_LEVELS, "linear", coordinate="depth")
        return cube  # noqa: RET504
