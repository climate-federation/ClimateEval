import re

import pytest
from climateeval import Coordinate
from climateeval._utils import check_coordinates
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube


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


@pytest.fixture(params=["air_pressure", "altitude"])
def alevel_coord_1d(request: pytest.FixtureRequest) -> DimCoord:
    return DimCoord([0.0], standard_name=request.param)


@pytest.fixture(params=["air_pressure", "altitude"])
def alevel_coord_2d(request: pytest.FixtureRequest) -> AuxCoord:
    return AuxCoord([[0.0]], standard_name=request.param)


def test_check_coordinates_1d(
    time_coord: DimCoord,
    lat_coord_1d: DimCoord,
    lon_coord_1d: DimCoord,
) -> None:
    cube = Cube(
        [[[0.0]]],
        dim_coords_and_dims=[(time_coord, 0), (lat_coord_1d, 1), (lon_coord_1d, 2)],
    )
    check_coordinates(
        cube,
        [Coordinate("time"), Coordinate("lat"), Coordinate("lon")],
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
    with pytest.raises(ValueError, match=re.escape(msg)):
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
    check_coordinates(
        cube,
        [Coordinate("time"), Coordinate("lat"), Coordinate("lon")],
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
    with pytest.raises(ValueError, match=re.escape(msg)):
        check_coordinates(cube, [Coordinate("lat"), Coordinate("lon")])


def test_check_coordinates_missing_coord_fail(time_coord: DimCoord) -> None:
    cube = Cube([0.0], dim_coords_and_dims=[(time_coord, 0)])
    msg = (
        r"Expected coordinate Coordinate(var_name='lat') with name 'latitude' "
        r"missing in cube"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        check_coordinates(
            cube,
            [Coordinate("time"), Coordinate("lat"), Coordinate("lon")],
        )


def test_check_coordinates_general_coord_1d(
    time_coord: DimCoord,
    alevel_coord_1d: DimCoord,
) -> None:
    cube = Cube([[0.0]], dim_coords_and_dims=[(time_coord, 0), (alevel_coord_1d, 1)])
    check_coordinates(cube, [Coordinate("time"), Coordinate("alevel")])


def test_check_coordinates_general_coord_1d_fail(
    time_coord: DimCoord,
    alevel_coord_1d: DimCoord,
) -> None:
    cube = Cube([[0.0]], dim_coords_and_dims=[(time_coord, 0), (alevel_coord_1d, 1)])
    msg = (
        r"Expected coordinates [Coordinate(var_name='alevel')] do not cover "
        r"all dimensions of cube"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
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
    check_coordinates(cube, [Coordinate("time"), Coordinate("alevel")])


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
    with pytest.raises(ValueError, match=re.escape(msg)):
        check_coordinates(cube, [Coordinate("alevel")])


def test_check_coordinates_missing_general_coord_fail(time_coord: DimCoord) -> None:
    cube = Cube([0.0], dim_coords_and_dims=[(time_coord, 0)])
    msg = (
        r"Expected general coordinate Coordinate(var_name='alevel') with "
        r"allowed coordinates (Coordinate(var_name='alt'), "
        r"Coordinate(var_name='plev')) missing in cube"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        check_coordinates(
            cube,
            [Coordinate("time"), Coordinate("alevel")],
        )
