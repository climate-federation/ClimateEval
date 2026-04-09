"""Manage utility functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris
from esmvalcore.preprocessor import clip_timerange, derive, extract_levels
from iris import Constraint, NameConstraint
from iris.coords import CoordExtent
from iris.cube import CubeList
from loguru import logger
from ncdata.iris_xarray import cubes_from_xarray
from xarray import Dataset

from benchmarking import NO_LEVEL

if TYPE_CHECKING:
    from collections.abc import Iterable

    from iris.cube import Cube

    from benchmarking import Variable, VerticalLevel

logger = logger.opt(colors=True)


def check_coordinates(cube: Cube, expected_coordinates: Iterable[str]) -> None:
    """Check coordinates of data."""
    covered_dimensions: list[int] = []
    for expected_coordinate in expected_coordinates:
        if not cube.coords(expected_coordinate):
            msg = (
                f"Expected coordinate '{expected_coordinate}' (standard_name) "
                f"missing in cube\n{cube}"
            )
            raise ValueError(msg)
        covered_dimensions.extend(cube.coord_dims(expected_coordinate))
    if set(covered_dimensions) != set(range(cube.ndim)):
        msg = (
            f"Expected coordinates {expected_coordinates} do not cover all "
            f"dimensions of cube\n{cube}"
        )
        raise ValueError(msg)


def get_prepared_cube(data: CubeList | Dataset, variable: Variable) -> Cube:
    """Get prepared data.

    This performs preprocessing according to `variable.settings`.

    Parameters
    ----------
    data:
        Input data.
    variable:
        Variable.

    Returns
    -------
    :
        Prepared data.

    """
    if isinstance(data, Dataset):
        data = cubes_from_xarray(data)

    try:
        cube = data.extract_cube(NameConstraint(var_name=variable.var_name))
    except iris.exceptions.ConstraintMismatchError:
        if not variable.derived:
            raise
        cube = _get_derived_cube(data, variable)

    _check_data(cube, variable)

    # Make sure units are correct (this will potentiall raise an error)
    cube.convert_units(variable.units)

    # Make sure longitude is in [0, 360]
    if cube.coords("longitude"):
        cube = _set_lon_to_0_360(cube)

    # Extract desired time range and bounding box
    if cube.coords("time") and variable.timerange != "*":
        cube = clip_timerange(cube, variable.timerange)
    if cube.coords("latitude"):
        cube = cube.extract(
            Constraint(
                latitude=lambda cell: variable.min_lat <= cell <= variable.max_lat,
            ),
        )
    if cube.coords("longitude"):
        cube = cube.extract(
            Constraint(
                longitude=lambda cell: variable.min_lon <= cell <= variable.max_lon,
            ),
        )

    # Extract level
    if variable.vertical_level is not NO_LEVEL:
        cube = _extract_level(cube, variable.vertical_level)

    return cube


def _check_data(cube: Cube, variable: Variable) -> None:
    """Check data."""
    expected_coordinates = {c.standard_name for c in variable.coordinates}
    check_coordinates(cube, expected_coordinates)

    # Time
    if not cube.coords("time"):
        return
    time_coord = cube.coord("time")
    datetimes = time_coord.units.num2date(time_coord.points)
    actual_min_year = datetimes[0].year
    actual_max_year = datetimes[-1].year
    expected_min_year = int(variable.timerange.split("/")[0][:4])
    expected_max_year = int(variable.timerange.split("/")[1][:4])
    if expected_min_year < actual_min_year:
        msg = (
            f"Selected timerange is {variable.timerange}, but data starts in "
            f"year {actual_min_year} for cube \n{cube}"
        )
        raise ValueError(msg)
    if expected_max_year > actual_max_year:
        msg = (
            f"Selected timerange is {variable.timerange}, but data ends in "
            f"year {actual_max_year} for cube \n{cube}"
        )
        raise ValueError(msg)


def _extract_level(cube: Cube, vertical_level: VerticalLevel) -> Cube:
    """Extract level from cube."""
    coord = cube.coord(vertical_level.coordinate_name)
    coord.convert_units(vertical_level.units)
    return extract_levels(
        cube,
        vertical_level.value,
        "linear",
        coordinate=vertical_level.coordinate_name,
    )


def _get_derived_cube(cubes: CubeList, variable: Variable) -> Cube:
    """Derive variable."""
    logger.debug(f"Data for {variable} not found, trying derivation")
    cubes_for_derivation: CubeList = CubeList([])
    for required_variable in variable.get_required_variables():
        cubes_for_derivation.append(
            cubes.extract_cube(NameConstraint(var_name=required_variable.var_name)),
        )
    return derive(
        cubes_for_derivation,
        short_name=variable.var_name,
        long_name=variable.long_name,
        units=variable.units,
        standard_name=variable.standard_name,
    )


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
