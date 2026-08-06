"""Provide AMOC diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from esmvalcore.preprocessor import (
    climate_statistics,
    extract_levels,
    extract_point,
    extract_trajectory,
)
from iris.cube import Cube
from iris.util import squeeze
from climateeval import Coordinate
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval.diags._utils import DEPTH_LEVELS
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


class DepthProfile(SimpleDiagnostic):
    """Manage AMOC time series diagnostics."""

    _final_coordinates = (
        Coordinate("depth"),
    )

    def __init__(self, *args, **kwargs):
        # Hier werden 'lats' und 'lons' direkt abgefangen
        self.lats = kwargs.pop("lats", [26.5])
        self.lons = kwargs.pop("lons", [-30.0])
        self.n_points = kwargs.pop("n_points", 10)
        # Reicht alle restlichen Framework-Argumente (wie Name und Variablen) weiter
        super().__init__(*args, **kwargs)

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = climate_statistics(cube, "mean", "full")
            cube = extract_trajectory(cube, self.lats, self.lons, self.n_points)
            new_cube = cube.copy()[..., 0]
            new_cube = squeeze(new_cube)
            new_cube.data = np.mean(cube.data, axis=cube.coord_dims("longitude")[0])
            cube = new_cube

            cube = extract_levels(cube, DEPTH_LEVELS, "linear", coordinate="depth")
            return cube
