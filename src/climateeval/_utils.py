"""Provide utility functions."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import iris
from esmvalcore.preprocessor import (
    clip_timerange,
    convert_units,
    derive,
    extract_levels,
    extract_region,
    extract_shape,
    mask_landsea,
)
from esmvalcore.preprocessor._derive import get_required
from iris import Constraint, NameConstraint
from iris.coords import CoordExtent
from iris.cube import CubeList
from iris.exceptions import ConstraintMismatchError
from loguru import logger
from ncdata.iris_xarray import cubes_from_xarray
from xarray import Dataset

from climateeval import NO_LEVEL
from climateeval._combine import normalise_coordinates, warn_on_split_variables
from climateeval._config import (
    setup_esmvaltool_config_and_logging,
    setup_iris_combine_policy,
)
from climateeval._variable import VARIABLES
from climateeval.exceptions import CoordinateError, MissingDataError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from iris.cube import Cube

    from climateeval import Coordinate, Variable, VerticalLevel

logger = logger.opt(colors=True)


def check_coordinates(cube: Cube, expected_coordinates: Iterable[Coordinate]) -> Cube:
    """Check coordinates of data and unify coordinate names."""
    # Avoid overwriting input cubes
    cube = cube.copy()

    covered_dimensions: list[int] = []
    for coordinate in expected_coordinates:
        # Handle generic coordinates
        if coordinate.allowed_coordinates:
            for allowed_coordinate in coordinate.allowed_coordinates:
                if cube.coords(allowed_coordinate.name()):
                    expected_coordinate = allowed_coordinate
                    break
            else:
                msg = (
                    f"Expected general coordinate {coordinate} with allowed "
                    f"coordinates {coordinate.allowed_coordinates} missing in "
                    f"cube {cube.summary(shorten=True)}"
                )
                raise CoordinateError(msg)
        else:
            expected_coordinate = coordinate

        # Check if coordinate is present
        coord_name = expected_coordinate.name()
        if not cube.coords(coord_name):
            msg = (
                f"Expected coordinate {coordinate} with name '{coord_name}' "
                f"missing in cube {cube.summary(shorten=True)}"
            )
            raise CoordinateError(msg)

        # Unify coordinate names
        cube_coord = cube.coord(coord_name)
        cube_coord.var_name = expected_coordinate.var_name
        cube_coord.long_name = expected_coordinate.long_name
        cube_coord.standard_name = expected_coordinate.standard_name

        covered_dimensions.extend(cube.coord_dims(coord_name))

    # Check cube rank
    if set(covered_dimensions) != set(range(cube.ndim)):
        msg = (
            f"Expected coordinates {expected_coordinates} do not cover all "
            f"dimensions of cube {cube.summary(shorten=True)}"
        )
        raise CoordinateError(msg)

    return cube


def get_prepared_cube(data: CubeList | Dataset, variable: Variable) -> Cube:
    """Get prepared data.

    This performs preprocessing according to
    meth:`Variable.get_preprocessing_settings`.

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

        # General preprocessing
        cube = _check_data(cube, variable)
        cube = _set_coordinate_bounds(cube)
        if cube.coords("longitude"):
            cube = _set_lon_to_0_360(cube)

        # Custom preprocessing
        cube = _apply_masks(cube, variable)
        cube = _extract_time_and_region(cube, variable)
        if variable.vertical_level is not NO_LEVEL:
            cube = _extract_level(cube, variable.vertical_level)
        cube = _run_additional_preprocessors(cube, variable)
        cube = convert_units(cube, variable.units)

    return cube  # noqa: RET504


def load_cubes(uris: str | Path | Iterable[str] | Iterable[Path]) -> CubeList:
    """Load Iris cubes.

    Files of one variable whose *coordinate* metadata disagrees (a per-file
    ``time_origin``, a capitalised ``long_name``, two calendar spellings, a
    float32/float64 grid) do not combine, and iris returns several cubes for
    that variable rather than failing. When that happens, reload without
    combining, reconcile the coordinates (:mod:`climateeval._combine`) and
    combine again; anything still split is reported as a warning naming the
    variable.
    """
    with setup_iris_combine_policy():
        cubes = iris.load(uris)
        if _has_split_variables(cubes):
            raw = CubeList(iris.load_raw(uris))
            normalise_coordinates(raw)
            combined = raw.combine()
            if len(combined) <= len(cubes):
                cubes = combined
    warn_on_split_variables(cubes)
    return cubes


def _has_split_variables(cubes: Iterable[Cube]) -> bool:
    """Whether any variable is represented by more than one cube."""
    seen: set[str] = set()
    for cube in cubes:
        key = cube.var_name or cube.name()
        if key in seen:
            return True
        seen.add(key)
    return False


def str_to_object(object_str: str) -> Any:
    """Convert string to loadable Python object."""
    try:
        module_name, class_name = object_str.rsplit(".", 1)
        module = importlib.import_module(module_name)
        object_ = getattr(module, class_name)
    except Exception as exc:
        msg = f"Failed to load Python object '{object_str}': {exc}"
        raise ValueError(msg) from exc
    return object_


def _apply_masks(cube: Cube, variable: Variable) -> Cube:
    """Apply masks to cube."""
    if variable.landsea_mask != "*":
        mask_out_options = {"land_only": "sea", "sea_only": "land"}
        if variable.landsea_mask not in mask_out_options:
            msg = (
                f"Expected one of ('*', 'land_only', 'sea_only') for option "
                f"'landsea_mask' of variable {variable}, got '{variable.landsea_mask}'"
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


def _check_data(cube: Cube, variable: Variable) -> Cube:
    """Check data and unify some metadata."""
    cube = check_coordinates(cube, variable.coordinates)

    # Time
    if not cube.coords("time"):
        return cube
    if variable.timerange == "*":
        return cube
    time_coord = cube.coord("time")
    datetimes = time_coord.units.num2date(time_coord.points)
    actual_min_year = datetimes[0].year
    actual_max_year = datetimes[-1].year
    expected_min_year = int(variable.timerange.split("/")[0][:4])
    expected_max_year = int(variable.timerange.split("/")[1][:4])
    if expected_min_year < actual_min_year:
        msg = (
            f"Selected timerange is {variable.timerange}, but data starts in "
            f"year {actual_min_year} for {variable}"
        )
        raise MissingDataError(msg)
    if expected_max_year > actual_max_year:
        msg = (
            f"Selected timerange is {variable.timerange}, but data ends in "
            f"year {actual_max_year} for {variable}"
        )
        raise MissingDataError(msg)

    return cube


def _extract_level(cube: Cube, vertical_level: VerticalLevel) -> Cube:
    """Extract level from cube."""
    coord = cube.coord(vertical_level.coordinate_name)
    coord.convert_units(vertical_level.units)
    scheme = "linear_extrapolate" if vertical_level.extrapolate else "linear"
    cube = extract_levels(
        cube,
        vertical_level.value,
        scheme,
        coordinate=vertical_level.coordinate_name,
    )

    # Squeeze cube if necessary
    if cube.coords(vertical_level.coordinate_name, dim_coords=True):
        coord = cube.coord(vertical_level.coordinate_name, dim_coords=True)
        if coord.shape[0] == 1:
            index: list[slice | int] = [slice(None)] * cube.ndim
            index[cube.coord_dims(coord)[0]] = 0
            cube = cube[tuple(index)]

    return cube


def _extract_time_and_region(cube: Cube, variable: Variable) -> Cube:
    """Extract timerange and region."""
    if cube.coords("time") and variable.timerange != "*":
        cube = clip_timerange(cube, variable.timerange)
    try:
        cube = extract_region(
            cube,
            start_longitude=variable.min_lon,
            end_longitude=variable.max_lon,
            start_latitude=variable.min_lat,
            end_latitude=variable.max_lat,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            f"Failed to run extract_region() on cube {cube.summary(shorten=True)} "
            f"({exc}), trying cube.extract() with corresponding constraints",
        )
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
    return cube


def _get_derived_cube(cubes: CubeList, variable: Variable) -> Cube:
    """Derive variable."""
    logger.debug(f"Deriving {variable}")

    # Some derivations accept alternative inputs marked as optional (e.g. amoc
    # can be derived from msftmz or msftyz). Skip optional inputs that are not
    # available and let the derivation use whatever is present.
    optional_var_names = {
        facet["short_name"]
        for facet in get_required(variable.var_name, "CMIP6")
        if facet.get("optional", False)
    }

    cubes_for_derivation: CubeList = CubeList([])
    for required_variable in variable.get_required_variables():
        try:
            cube = cubes.extract_cube(
                NameConstraint(var_name=required_variable.var_name),
            ).copy()
        except ConstraintMismatchError:
            if required_variable.var_name in optional_var_names:
                logger.debug(
                    f"Optional variable '{required_variable.var_name}' not "
                    f"available for deriving {variable}",
                )
                continue
            raise
        cubes_for_derivation.append(cube)
    return derive(
        cubes_for_derivation,
        short_name=variable.var_name,
        long_name=VARIABLES[variable.var_name]["long_name"],
        units=VARIABLES[variable.var_name]["units"],
        standard_name=VARIABLES[variable.var_name]["standard_name"],
    )


def _run_additional_preprocessors(cube: Cube, variable: Variable) -> Cube:
    """Run additional preprocessors."""
    for func_name, settings in variable.additional_preprocessors.items():
        func = str_to_object(func_name)
        try:
            cube = func(cube, **settings)
        except Exception as exc:
            msg = (
                f"Failed to run preprocessor function '{func_name}' with "
                f"settings {settings} on {variable}: {exc}"
            )
            raise ValueError(msg) from exc
    return cube


def _set_coordinate_bounds(cube: Cube) -> Cube:
    """Set coordinate bounds."""
    cube = cube.copy()
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
