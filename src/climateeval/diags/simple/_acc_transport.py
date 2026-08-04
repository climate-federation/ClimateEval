"""Provide ACC transport diagnostics on regular and native ocean grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris
import iris.analysis
import numpy as np

from climateeval import Coordinate
from climateeval.diags.simple._base import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


class ACCTransportZonalTransect(SimpleDiagnostic):
    """Calculate ACC transport through a constant longitude transect using uo."""

    _final_coordinates = (Coordinate("time"),)

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess uo data and integrate transport along the transect."""
        lon_target = -67.0
        lat_min = -70.0
        lat_max = -54.0

        lon_coord = cube.coord("longitude")
        lat_coord = cube.coord("latitude")
        target_lon_normalized = lon_target % 360

        if lon_coord.ndim == 1:
            lon_points_normalized = lon_coord.points % 360
            closest_x_idx = int(
                np.argmin(np.abs(lon_points_normalized - target_lon_normalized)),
            )

            y_indices = np.where(
                (lat_coord.points >= lat_min) & (lat_coord.points <= lat_max),
            )[0]

            x_dim_index = int(cube.coord_dims(lon_coord)[0])
            y_dim_index = int(cube.coord_dims(lat_coord)[0])

        else:
            lon_points = lon_coord.points
            lat_points = lat_coord.points
            lon_points_normalized = lon_points % 360
            lat_mid = (lat_min + lat_max) / 2.0

            distance_matrix = (lon_points_normalized - target_lon_normalized) ** 2 + (
                lat_points - lat_mid
            ) ** 2
            _, closest_x = np.unravel_index(
                np.argmin(distance_matrix),
                lon_points.shape,
            )
            closest_x_idx = int(closest_x)

            lats_along_transect = lat_points[:, closest_x_idx]

            y_indices = np.where(
                (lats_along_transect >= lat_min) & (lats_along_transect <= lat_max),
            )[0]

            if y_indices.size == 0:
                msg = f"No latitude between {lat_min} and {lat_max} found."
                raise ValueError(
                    msg,
                )

            y_dim_index = int(cube.coord_dims(lat_coord)[0])
            x_dim_index = int(cube.coord_dims(lon_coord)[1])

        slice_structure: list[slice | int | np.ndarray] = [slice(None)] * cube.ndim

        slice_structure[x_dim_index] = closest_x_idx
        slice_structure[y_dim_index] = y_indices

        cube = cube[tuple(slice_structure)]

        transect_lats = cube.coord("latitude").points
        if transect_lats.ndim > 1:
            transect_lats = transect_lats.squeeze()

        lat_rad = np.deg2rad(transect_lats)
        dlat = np.gradient(lat_rad) if len(lat_rad) > 1 else np.array([0.0])
        earth_radius = 6371000.0
        dy_array = np.abs(earth_radius * dlat)

        dy_coord = iris.coords.AuxCoord(
            dy_array,
            long_name="grid_cell_width_y",
            units="m",
        )
        cube.add_aux_coord(dy_coord, cube.coord_dims("latitude"))

        z_coord = cube.coord("depth")
        if z_coord.has_bounds():
            dz_array = np.abs(np.diff(z_coord.bounds, axis=1)).squeeze()
        else:
            dz_array = np.gradient(z_coord.points)

        cube = self._compute_transport(cube, dy_array, dz_array)

        return cube  # noqa: RET504

    def _compute_transport(
        self,
        cube: Cube,
        dy_array: np.ndarray,
        dz_array: np.ndarray,
    ) -> Cube:
        """Compute the transport through the transect."""
        dz_coord = iris.coords.AuxCoord(
            dz_array,
            long_name="grid_cell_height_z",
            units="m",
        )
        cube.add_aux_coord(dz_coord, cube.coord_dims("depth"))

        dy_cube = iris.util.broadcast_to_shape(
            dy_array,
            cube.shape,
            cube.coord_dims("latitude"),
        )
        dz_cube = iris.util.broadcast_to_shape(
            dz_array,
            cube.shape,
            cube.coord_dims("depth"),
        )

        transport_cube = cube.copy()
        transport_cube.data = transport_cube.core_data() * dy_cube * dz_cube / 1e6

        transport_cube = transport_cube.collapsed(
            ["latitude", "depth"],
            iris.analysis.SUM,
        )

        actual_lon = np.mean(cube.coord("longitude").points)
        transport_cube.units = "Sv"
        transport_cube.long_name = f"ACC_Transport_at_{actual_lon:.2f}_degrees_east"

        if transport_cube.coords("longitude"):
            transport_cube.remove_coord("longitude")

        return transport_cube
