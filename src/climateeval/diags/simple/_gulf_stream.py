"""Provide Gulf Stream separation diagnostics for SSH and SST gradients."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris.analysis.calculus
import iris.analysis.cartography
import numpy as np
from esmvalcore.preprocessor import (
    climate_statistics,
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


class GulfStreamSshGradient(SimpleDiagnostic):
    """Manage Gulf Stream separation diagnostics using SSH gradients.

    Computes the horizontal gradient magnitude of Sea Surface Height (SSH)
    to allow bias evaluation against reference datasets.
    """

    _final_coordinates = (
        Coordinate("lat"),
        Coordinate("lon"),
    )

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess SSH data to compute coherent spatial gradient magnitude."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(
                cube,
                DEFAULT_GRID,
                "linear",
                cache_weights=True,
            )  # to ensure we are on a regular grid

            grad_y = iris.analysis.calculus.differentiate(cube, "latitude")
            grad_x = iris.analysis.calculus.differentiate(cube, "longitude")

            gx_data = grad_x.data[..., :-1, :]
            gy_data = grad_y.data[..., :, :-1]
            ssh_grad = cube[..., :-1, :-1]

            radius = 6371000.0
            deg_to_rad = np.pi / 180.0
            dy = radius * deg_to_rad
            cos_lat = iris.analysis.cartography.cosine_latitude_weights(ssh_grad)
            dx = radius * cos_lat * deg_to_rad

            ssh_grad.data = np.sqrt((gx_data / dx) ** 2 + (gy_data / dy) ** 2)
            ssh_grad.var_name = "ssh_grad_mag"
            ssh_grad.long_name = "SSH Gradient Magnitude"
            ssh_grad.units = "m m-1"

            ssh_grad = regrid_time(
                ssh_grad,
                frequency="mon",
                calendar="standard",
                units=COORDINATES["time"]["units"],
            )
            ssh_grad = climate_statistics(ssh_grad, "mean", "full")
        return ssh_grad  # noqa: RET504
