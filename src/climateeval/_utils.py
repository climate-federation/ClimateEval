"""Provide utility functions."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import iris
from esmvalcore.preprocessor import (
    anomalies,
    clip_timerange,
    convert_units,
    derive,
    extract_levels,
    extract_shape,
    mask_landsea,
)
from iris import Constraint, NameConstraint
from iris.coords import CoordExtent
from iris.cube import CubeList
from iris.exceptions import ConstraintMismatchError
from loguru import logger
from ncdata.iris_xarray import cubes_from_xarray
from xarray import Dataset

from climateeval import NO_LEVEL
from climateeval._config import (
    setup_esmvaltool_config_and_logging,
    setup_iris_combine_policy,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from iris.cube import Cube

    from climateeval import Coordinate, Variable, VerticalLevel

logger = logger.opt(colors=True)


def check_coordinates(cube: Cube, expected_coordinates: Iterable[Coordinate]) -> None:
    """Check coordinates of data."""
    covered_dimensions: list[int] = []
    for coordinate in expected_coordinates:
        if coordinate.allowed_coordinates:
            for allowed_coordinate in coordinate.allowed_coordinates:
                if cube.coords(allowed_coordinate.name()):
                    expected_coordinate = allowed_coordinate
                    break
            else:
                msg = (
                    f"Expected general coordinate {coordinate} with allowed "
                    f"coordinates {coordinate.allowed_coordinates} missing in "
                    f"cube\n{cube}"
                )
                raise ValueError(msg)
        else:
            expected_coordinate = coordinate
        coord_name = expected_coordinate.name()
        if not cube.coords(coord_name):
            msg = (
                f"Expected coordinate {coordinate} with name '{coord_name}' "
                f"missing in cube\n{cube}"
            )
            raise ValueError(msg)
        covered_dimensions.extend(cube.coord_dims(coord_name))
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
        with setup_iris_combine_policy():
            data = cubes_from_xarray(data)

    with setup_esmvaltool_config_and_logging():
        try:
            # Use copy() to avoid overwriting input data
            cube = data.extract_cube(NameConstraint(var_name=variable.var_name)).copy()
        except ConstraintMismatchError:
            if not variable.derived:
                raise
            cube = _get_derived_cube(data, variable)

        _check_data(cube, variable)

        # Add bounds if necessary
        cube = _set_coordinate_bounds(cube)

        # Make sure units are correct
        cube = convert_units(cube, variable.units)

        # Make sure longitude is in [0, 360]
        if cube.coords("longitude"):
            cube = _set_lon_to_0_360(cube)

        # Apply masks
        cube = _apply_masks(cube, variable)

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

        # Calculate anomalies
        if variable.anomaly:
            cube = anomalies(cube, variable.anomaly)

    return cube


def load_cubes(uris: str | Path | Iterable[str] | Iterable[Path]) -> CubeList:
    """Load Iris cubes."""
    with setup_iris_combine_policy():
        return iris.load(uris)


def str_to_class(class_str: str) -> type:
    """Convert string to Python class."""
    try:
        module_name, class_name = class_str.rsplit(".", 1)
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
    except Exception as exc:
        msg = f"Failed to load class '{class_str}': {exc}"
        raise ValueError(msg) from exc
    return class_


def _apply_masks(cube: Cube, variable: Variable) -> Cube:
    """Apply masks to cube."""
    if variable.landsea_mask != "*":
        mask_out_options = {"land_only": "sea", "sea-only": "land"}
        if variable.landsea_mask not in mask_out_options:
            msg = (
                f"Got invalid value '{variable.landsea_mask}' for option "
                f"'landsea_mask' of variable {variable}, expected one of "
                f"{list(mask_out_options)}"
            )
            raise ValueError(msg)
        cube = mask_landsea(cube, mask_out_options[variable.landsea_mask])
    if variable.ar6_regions_mask:
        cube = extract_shape(
            cube,
            "ar6",
            method="contains",
            crop=False,
            decomposed=False,
            ids={"Name": variable.ar6_regions_mask},
        )
    return cube


def _check_data(cube: Cube, variable: Variable) -> None:
    """Check data."""
    check_coordinates(cube, variable.coordinates)

    # Time
    if not cube.coords("time"):
        return
    if variable.timerange == "*":
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
    logger.debug(f"Deriving {variable}")
    cubes_for_derivation: CubeList = CubeList([])
    for required_variable in variable.get_required_variables():
        cubes_for_derivation.append(
            cubes.extract_cube(
                NameConstraint(var_name=required_variable.var_name),
            ).copy(),
        )
    return derive(
        cubes_for_derivation,
        short_name=variable.var_name,
        long_name=variable.long_name,
        units=variable.units,
        standard_name=variable.standard_name,
    )


def _set_coordinate_bounds(cube: Cube) -> Cube:
    """Set coordinate bounds (in-place)."""
    for coord_name in ("time", "latitude", "longitude"):
        if cube.coords(coord_name, dim_coords=True):
            coord = cube.coord(coord_name, dim_coords=True)
            if not coord.has_bounds():
                coord.guess_bounds()
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
