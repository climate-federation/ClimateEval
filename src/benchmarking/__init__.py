"""Benchmark Earth system models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris
from esmvalcore.preprocessor import clip_timerange
from iris import Constraint, NameConstraint
from iris.coords import CoordExtent
from ncdata.iris_xarray import cubes_from_xarray
from xarray import Dataset

if TYPE_CHECKING:
    from iris.cube import Cube, CubeList

    from benchmarking._config import Config
    from benchmarking._variable import Variable


# Make Iris as flexible as possible when combining cubes (e.g., during loading)
iris.COMBINE_POLICY.equalise_cubes_kwargs = {
    "equalise_attributes": True,
    "unify_time_units": True,
}
iris.COMBINE_POLICY.merge_concat_sequence = "mc"
iris.COMBINE_POLICY.repeat_until_unchanged = True

# Use new behavior of iris.pandas.as_data_frame
iris.FUTURE.pandas_ndim = True


def get_prepared_cube(
    data: CubeList | Dataset,
    variable: Variable,
    config: Config,
) -> Cube:
    """Get prepared Iris cube."""
    if isinstance(data, Dataset):
        data = cubes_from_xarray(data)
    cube = data.extract_cube(NameConstraint(var_name=variable.var_name))

    # Make sure units are correct (this will potentiall raise an error)
    cube.convert_units(variable.units)

    # Make sure longitude is in [0, 360]
    if cube.coords("longitude"):
        cube = _set_lon_to_0_360(cube)

    # Extract desired time range and bounding box
    if cube.coords("time") and config.timerange != "*":
        cube = clip_timerange(cube, config.timerange)
    if cube.coords("latitude"):
        cube = cube.extract(
            Constraint(latitude=lambda cell: config.min_lat <= cell <= config.max_lat),
        )
    if cube.coords("longitude"):
        cube = cube.extract(
            Constraint(longitude=lambda cell: config.min_lon <= cell <= config.max_lon),
        )

    return cube


def _set_lon_to_0_360(cube: Cube) -> Cube:
    """Put longitude into [0, 360]."""
    cube = cube.copy()
    coord = cube.coord("longitude")

    # cube.intersection only works for cells with 0 or 2 bounds
    # Note: nbounds==0 means there are no bounds given, nbounds==2 implies a
    # regular grid with bounds in the grid direction, nbounds>2 implies an
    # irregular grid with bounds given as vertices of the cell polygon.
    if coord.ndim == 1 and coord.nbounds in (0, 2):
        lon_extent = CoordExtent(
            coord,
            0.0,
            360.0,
            min_inclusive=True,
            max_inclusive=False,
        )
        cube = cube.intersection(lon_extent)
    else:
        new_points = (coord.core_points().copy() + 360.0) % 360.0
        if coord.has_bounds():
            new_bounds = (coord.core_bounds().copy() + 360.0) % 360.0
        else:
            new_bounds = None
        new_coord = coord.copy(points=new_points, bounds=new_bounds)
        coord_dims = cube.coord_dims(coord)
        cube.remove_coord(coord)
        cube.add_aux_coord(new_coord, coord_dims)

    return cube
