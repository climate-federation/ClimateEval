from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

from climateeval import Coordinate, Variable
from climateeval._utils import (
    _apply_masks,
    _check_data,
    _run_additional_preprocessors,
    check_coordinates,
    get_prepared_cube,
)
from climateeval.exceptions import CoordinateError, MissingDataError
from tests._utils import assert_masked_array_allclose

if TYPE_CHECKING:
    import xarray as xr
    from iris.cube import CubeList


@pytest.fixture
def time_coord() -> DimCoord:
    return DimCoord([0.0], standard_name="time")


@pytest.fixture
def lat_coord_1d() -> DimCoord:
    return DimCoord([0.0], standard_name="latitude")


@pytest.fixture
def lat_coord_2d() -> AuxCoord:
    return AuxCoord([[0.0]], standard_name="latitude")


@pytest.fixture
def lon_coord_1d() -> DimCoord:
    return DimCoord([0.0], standard_name="longitude")


@pytest.fixture
def lon_coord_2d() -> AuxCoord:
    return AuxCoord([[0.0]], standard_name="longitude")


@pytest.fixture(params=[("plev", "air_pressure"), ("alt", "altitude")])
def alevel_coord_1d(request: pytest.FixtureRequest) -> DimCoord:
    return DimCoord([0.0], var_name=request.param[0], standard_name=request.param[1])


@pytest.fixture(params=[("plev", "air_pressure"), ("alt", "altitude")])
def alevel_coord_2d(request: pytest.FixtureRequest) -> AuxCoord:
    return AuxCoord([[0.0]], var_name=request.param[0], standard_name=request.param[1])


def test_check_coordinates_1d(
    time_coord: DimCoord,
    lat_coord_1d: DimCoord,
    lon_coord_1d: DimCoord,
) -> None:
    cube = Cube(
        [[[0.0]]],
        dim_coords_and_dims=[(time_coord, 0), (lat_coord_1d, 1), (lon_coord_1d, 2)],
    )

    cube = check_coordinates(
        cube,
        [Coordinate("time"), Coordinate("lat"), Coordinate("lon")],
    )

    for name in ("var_name", "standard_name", "long_name"):
        assert getattr(cube.coord("time"), name) == getattr(Coordinate("time"), name)
        assert getattr(cube.coord("latitude"), name) == getattr(Coordinate("lat"), name)
        assert getattr(cube.coord("longitude"), name) == getattr(
            Coordinate("lon"),
            name,
        )


def test_check_coordinates_1d_fail(
    time_coord: DimCoord,
    lat_coord_1d: DimCoord,
    lon_coord_1d: DimCoord,
) -> None:
    cube = Cube(
        [[[0.0]]],
        dim_coords_and_dims=[(time_coord, 0), (lat_coord_1d, 1), (lon_coord_1d, 2)],
    )
    msg = (
        r"Expected coordinates [Coordinate(var_name='time'), "
        r"Coordinate(var_name='lat')] do not cover all dimensions of cube"
    )
    with pytest.raises(CoordinateError, match=re.escape(msg)):
        check_coordinates(cube, [Coordinate("time"), Coordinate("lat")])


def test_check_coordinates_2d(
    time_coord: DimCoord,
    lat_coord_2d: AuxCoord,
    lon_coord_2d: AuxCoord,
) -> None:
    cube = Cube(
        [[[0.0]]],
        dim_coords_and_dims=[(time_coord, 0)],
        aux_coords_and_dims=[(lat_coord_2d, (1, 2)), (lon_coord_2d, (1, 2))],
    )

    cube = check_coordinates(
        cube,
        [Coordinate("time"), Coordinate("lat"), Coordinate("lon")],
    )

    for name in ("var_name", "standard_name", "long_name"):
        assert getattr(cube.coord("time"), name) == getattr(Coordinate("time"), name)
        assert getattr(cube.coord("latitude"), name) == getattr(Coordinate("lat"), name)
        assert getattr(cube.coord("longitude"), name) == getattr(
            Coordinate("lon"),
            name,
        )


def test_check_coordinates_2d_fail(
    time_coord: DimCoord,
    lat_coord_2d: AuxCoord,
    lon_coord_2d: AuxCoord,
) -> None:
    cube = Cube(
        [[[0.0]]],
        dim_coords_and_dims=[(time_coord, 0)],
        aux_coords_and_dims=[(lat_coord_2d, (1, 2)), (lon_coord_2d, (1, 2))],
    )
    msg = (
        r"Expected coordinates [Coordinate(var_name='lat'), "
        r"Coordinate(var_name='lon')] do not cover all dimensions of cube"
    )
    with pytest.raises(CoordinateError, match=re.escape(msg)):
        check_coordinates(cube, [Coordinate("lat"), Coordinate("lon")])


def test_check_coordinates_missing_coord_fail(time_coord: DimCoord) -> None:
    cube = Cube([0.0], dim_coords_and_dims=[(time_coord, 0)])
    msg = (
        r"Expected coordinate Coordinate(var_name='lat') with name 'latitude' "
        r"missing in cube"
    )
    with pytest.raises(CoordinateError, match=re.escape(msg)):
        check_coordinates(
            cube,
            [Coordinate("time"), Coordinate("lat"), Coordinate("lon")],
        )


def test_check_coordinates_general_coord_1d(
    time_coord: DimCoord,
    alevel_coord_1d: DimCoord,
) -> None:
    cube = Cube([[0.0]], dim_coords_and_dims=[(time_coord, 0), (alevel_coord_1d, 1)])

    cube = check_coordinates(cube, [Coordinate("time"), Coordinate("alevel")])

    for name in ("var_name", "standard_name", "long_name"):
        assert getattr(cube.coord("time"), name) == getattr(Coordinate("time"), name)
        assert getattr(cube.coord(alevel_coord_1d.name()), name) == getattr(
            Coordinate(alevel_coord_1d.var_name),
            name,
        )


def test_check_coordinates_general_coord_1d_fail(
    time_coord: DimCoord,
    alevel_coord_1d: DimCoord,
) -> None:
    cube = Cube([[0.0]], dim_coords_and_dims=[(time_coord, 0), (alevel_coord_1d, 1)])
    msg = (
        r"Expected coordinates [Coordinate(var_name='alevel')] do not cover "
        r"all dimensions of cube"
    )
    with pytest.raises(CoordinateError, match=re.escape(msg)):
        check_coordinates(cube, [Coordinate("alevel")])


def test_check_coordinates_general_coord_2d(
    time_coord: DimCoord,
    alevel_coord_2d: AuxCoord,
) -> None:
    cube = Cube(
        [[[0.0]]],
        dim_coords_and_dims=[(time_coord, 0)],
        aux_coords_and_dims=[(alevel_coord_2d, (1, 2))],
    )

    cube = check_coordinates(cube, [Coordinate("time"), Coordinate("alevel")])

    for name in ("var_name", "standard_name", "long_name"):
        assert getattr(cube.coord("time"), name) == getattr(Coordinate("time"), name)
        assert getattr(cube.coord(alevel_coord_2d.name()), name) == getattr(
            Coordinate(alevel_coord_2d.var_name),
            name,
        )


def test_check_coordinates_general_coord_2d_fail(
    time_coord: DimCoord,
    alevel_coord_2d: AuxCoord,
) -> None:
    cube = Cube(
        [[[0.0]]],
        dim_coords_and_dims=[(time_coord, 0)],
        aux_coords_and_dims=[(alevel_coord_2d, (1, 2))],
    )
    msg = (
        r"Expected coordinates [Coordinate(var_name='alevel')] do not cover "
        r"all dimensions of cube"
    )
    with pytest.raises(CoordinateError, match=re.escape(msg)):
        check_coordinates(cube, [Coordinate("alevel")])


def test_check_coordinates_missing_general_coord_fail(time_coord: DimCoord) -> None:
    cube = Cube([0.0], dim_coords_and_dims=[(time_coord, 0)])
    msg = (
        r"Expected general coordinate Coordinate(var_name='alevel') with "
        r"allowed coordinates (Coordinate(var_name='alt'), "
        r"Coordinate(var_name='plev')) missing in cube"
    )
    with pytest.raises(CoordinateError, match=re.escape(msg)):
        check_coordinates(
            cube,
            [Coordinate("time"), Coordinate("alevel")],
        )


def test__run_additional_preprocessors(cube_2d: Cube) -> None:
    variable = Variable(
        "my_tas",
        "tas",
        "mon",
        additional_preprocessors={
            "esmvalcore.preprocessor.area_statistics": {"operator": "mean"},
        },
    )
    cube = _run_additional_preprocessors(cube_2d, variable)
    assert [c.name() for c in cube.coords(dim_coords=True)] == ["time"]
    assert cube.shape == (3,)


def test__run_additional_preprocessors_invalid_func_fail(cube_2d: Cube) -> None:
    variable = Variable(
        "my_tas",
        "tas",
        "mon",
        additional_preprocessors={"invalid.func": {"test": "1"}},
    )
    msg = r"Failed to load Python object 'invalid.func': No module named 'invalid'"
    with pytest.raises(ValueError, match=re.escape(msg)):
        _run_additional_preprocessors(cube_2d, variable)


def test__run_additional_preprocessors_calculation_error_fail(cube_2d: Cube) -> None:
    variable = Variable(
        "my_tas",
        "tas",
        "mon",
        additional_preprocessors={"tests.unit.raise_value_error": {"test": "1"}},
    )
    msg = (
        r"Failed to run preprocessor function 'tests.unit.raise_value_error' with "
        r"settings {'test': '1'} on Variable(id='my_tas'): value error"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        _run_additional_preprocessors(cube_2d, variable)


def test_get_prepared_cube_landsea_mask(cubes: CubeList) -> None:
    variable = Variable("tas", "tas", "mon", landsea_mask="land_only")
    cube = get_prepared_cube(cubes, variable)
    expected_data = np.ma.masked_invalid(
        [
            [[np.nan, np.nan], [np.nan, 3.0], [np.nan, 5.0]],
            [[np.nan, np.nan], [np.nan, 9.0], [np.nan, 11.0]],
            [[np.nan, np.nan], [np.nan, 15.0], [np.nan, 17.0]],
        ],
    )
    assert_masked_array_allclose(cube.data, expected_data)


def test_get_prepared_cube_ar6_regions_mask(cubes: CubeList) -> None:
    variable = Variable("tas", "tas", "mon", ar6_regions_mask=("E.Asia",))
    cube = get_prepared_cube(cubes, variable)
    expected_data = np.ma.masked_invalid(
        [
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, 5.0]],
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, 11.0]],
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, 17.0]],
        ],
    )
    assert_masked_array_allclose(cube.data, expected_data)


def test_get_prepared_cube_cubes(cubes: CubeList, variable_tas: Variable) -> None:
    cube = get_prepared_cube(cubes, variable_tas)
    assert cube.var_name == "tas"
    assert cube.shape == (3, 3, 2)


def test_get_prepared_cube_xarray_dataset(
    xarray_ds: xr.Dataset,
    variable_tas: Variable,
) -> None:
    cube = get_prepared_cube(xarray_ds, variable_tas)
    assert cube.var_name == "tas"
    assert cube.shape == (3, 3, 2)


def test__apply_masks_invalid_landsea_mask_fail(
    cube_2d: Cube,
) -> None:
    variable = Variable("tas", "tas", "mon", landsea_mask="invalid")
    msg = (
        r"Expected one of ('*', 'land_only', 'sea_only') for option 'landsea_mask' "
        r"of variable Variable(id='tas'), got 'invalid'"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        _apply_masks(cube_2d, variable)


def test_check_data_scalar() -> None:
    cube = Cube(0, var_name="ecs")
    variable = Variable("ecs", "ecs", "fx")
    new_cube = _check_data(cube, variable)
    assert new_cube == cube


def test_check_data_invalid_min_year_fail(cube_2d: Cube) -> None:
    variable = Variable("tas", "tas", "mon", timerange="19600101/19710101")
    msg = (
        r"Selected timerange is 19600101/19710101, but data starts in year 1970 "
        r"for Variable(id='tas')"
    )
    with pytest.raises(MissingDataError, match=re.escape(msg)):
        _check_data(cube_2d, variable)


def test_check_data_invalid_max_year_fail(cube_2d: Cube) -> None:
    variable = Variable("tas", "tas", "mon", timerange="19700101/19800101")
    msg = (
        r"Selected timerange is 19700101/19800101, but data ends in year 1972 "
        r"for Variable(id='tas')"
    )
    with pytest.raises(MissingDataError, match=re.escape(msg)):
        _check_data(cube_2d, variable)
