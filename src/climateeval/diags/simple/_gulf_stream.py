"""Provide Gulf Stream separation diagnostics for SSH and SST gradients."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris.util
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

            # Centred differences on the grid points themselves (one-sided at
            # the outermost latitudes); longitude is periodic on the global
            # regular target grid.
            lat = cube.coord("latitude")
            lon = cube.coord("longitude")
            lat_dim = cube.coord_dims(lat)[0]
            lon_dim = cube.coord_dims(lon)[0]
            data = cube.data
            dssh_dlat = np.gradient(data, lat.points, axis=lat_dim)
            dlon = lon.points[1] - lon.points[0]
            dssh_dlon = (
                np.roll(data, -1, axis=lon_dim) - np.roll(data, 1, axis=lon_dim)
            ) / (2.0 * dlon)

            radius = 6371000.0
            deg_to_rad = np.pi / 180.0
            dy = radius * deg_to_rad
            cos_lat = iris.util.broadcast_to_shape(
                np.cos(np.deg2rad(lat.points)),
                cube.shape,
                (lat_dim,),
            )
            dx = radius * cos_lat * deg_to_rad

            # Centred differences skip the centre point, so masked (land)
            # cells keep their mask explicitly
            grad_mag = np.ma.masked_where(
                np.ma.getmaskarray(data),
                np.sqrt((dssh_dlon / dx) ** 2 + (dssh_dlat / dy) ** 2),
            )
            ssh_grad = cube.copy(grad_mag)
            # No CF standard name exists for the SSH gradient magnitude
            ssh_grad.standard_name = None
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
