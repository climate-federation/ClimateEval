import re

import pytest
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

from benchmarking._utils import check_coordinates


@pytest.fixture
def time_coord() -> DimCoord:
    return DimCoord([0.0], standard_name="time")


@pytest.fixture
def lat_coord_1d() -> DimCoord:
    return DimCoord([0.0], standard_name="latitude")


@pytest.fixture
def lon_coord_1d() -> DimCoord:
    return DimCoord([0.0], standard_name="longitude")


@pytest.fixture
def lat_coord_2d() -> DimCoord:
    return AuxCoord([[0.0]], standard_name="latitude")


@pytest.fixture
def lon_coord_2d() -> DimCoord:
    return AuxCoord([[0.0]], standard_name="longitude")


def test_check_coordinates_1d(
    time_coord: DimCoord,
    lat_coord_1d: DimCoord,
    lon_coord_1d: DimCoord,
) -> None:
    cube = Cube(
        [[[0.0]]],
        dim_coords_and_dims=[(time_coord, 0), (lat_coord_1d, 1), (lon_coord_1d, 2)],
    )
    check_coordinates(cube, ["time", "latitude", "longitude"])


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
        r"Expected coordinates ['time', 'latitude'] do not cover all dimensions of cube"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        check_coordinates(cube, ["time", "latitude"])


def test_check_coordinates_2d(
    time_coord: DimCoord,
    lat_coord_2d: DimCoord,
    lon_coord_2d: DimCoord,
) -> None:
    cube = Cube(
        [[[0.0]]],
        dim_coords_and_dims=[(time_coord, 0)],
        aux_coords_and_dims=[(lat_coord_2d, (1, 2)), (lon_coord_2d, (1, 2))],
    )
    check_coordinates(cube, ["time", "latitude", "longitude"])


def test_check_coordinates_2d_fail(
    time_coord: DimCoord,
    lat_coord_2d: DimCoord,
    lon_coord_2d: DimCoord,
) -> None:
    cube = Cube(
        [[[0.0]]],
        dim_coords_and_dims=[(time_coord, 0)],
        aux_coords_and_dims=[(lat_coord_2d, (1, 2)), (lon_coord_2d, (1, 2))],
    )
    msg = (
        r"Expected coordinates ['latitude', 'longitude'] do not cover all dimensions "
        r"of cube"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        check_coordinates(cube, ["latitude", "longitude"])


def test_check_coordinates_missing_coord_fail(time_coord: DimCoord) -> None:
    cube = Cube([0.0], dim_coords_and_dims=[(time_coord, 0)])
    msg = r"Expected coordinate 'latitude' (standard_name) missing in cube"
    with pytest.raises(ValueError, match=re.escape(msg)):
        check_coordinates(cube, ["time", "latitude", "longitude"])
