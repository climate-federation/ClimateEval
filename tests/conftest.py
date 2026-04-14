from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube

from benchmarking import Variable
from benchmarking.data import ERA5, GPCP, DiagnosticData, HadCRUT5

if TYPE_CHECKING:
    from pathlib import Path


pytest.register_assert_rewrite("tests.unit")


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def variables_and_data_2d() -> dict[Variable, DiagnosticData]:
    return {
        Variable("my_tas", "tas", "mon"): DiagnosticData(reference=HadCRUT5),
        Variable("my_pr", "pr", "mon"): DiagnosticData(reference=GPCP),
    }


@pytest.fixture
def variables_and_data_3d() -> dict[Variable, DiagnosticData]:
    return {
        Variable("ta", "ta", "mon"): DiagnosticData(reference=ERA5),
    }


@pytest.fixture
def cube_2d() -> Cube:
    return Cube(
        np.arange(1 * 3 * 4).reshape(1, 3, 4),
        var_name="tas",
        standard_name="air_temperature",
        long_name="Near-Surface Air Temperature",
        units="K",
        dim_coords_and_dims=[
            (_get_time_coord(), 0),
            (_get_latitude_coord(), 1),
            (_get_longitude_coord(), 2),
        ],
    )


@pytest.fixture(params=["altitude", "air_pressure"])
def cube_3d(request: pytest.FixtureRequest) -> Cube:
    if request.param == "altitude":
        z_coord = _get_altitude_coord()
    else:
        z_coord = _get_air_pressure_coord()
    return Cube(
        np.arange(1 * 2 * 3 * 4).reshape(1, 2, 3, 4),
        var_name="ta",
        standard_name="air_temperature",
        long_name="Air Temperature",
        units="K",
        dim_coords_and_dims=[
            (_get_time_coord(), 0),
            (z_coord, 1),
            (_get_latitude_coord(), 2),
            (_get_longitude_coord(), 3),
        ],
    )


def _get_time_coord() -> DimCoord:
    return DimCoord(
        [5.0],
        bounds=[[0.0, 10.0]],
        var_name="time",
        standard_name="time",
        long_name="Time Intervals",
        units="days since 1970-01-01 00:00:00",
    )


def _get_altitude_coord() -> DimCoord:
    return DimCoord(
        [100.0, 200.0],
        var_name="alt",
        standard_name="altitude",
        long_name="Altitude",
        units="m",
        attributes={"positive": "up"},
    )


def _get_air_pressure_coord() -> DimCoord:
    return DimCoord(
        [100000.0, 80000.0],
        var_name="plev",
        standard_name="air_pressure",
        long_name="Pressure Levels",
        units="Pa",
        attributes={"positive": "down"},
    )


def _get_latitude_coord() -> DimCoord:
    return DimCoord(
        [0.0, 2.0, 4.0],
        bounds=[[-1.0, 1.0], [1.0, 3.0], [3.0, 5.0]],
        var_name="lat",
        standard_name="latitude",
        long_name="Latitude",
        units="degrees_north",
    )


def _get_longitude_coord() -> DimCoord:
    return DimCoord(
        [10.0, 20.0, 30.0, 40.0],
        bounds=[[5.0, 15.0], [15.0, 25.0], [25.0, 35.0], [35.0, 45.0]],
        var_name="lon",
        standard_name="longitude",
        long_name="Longitude",
        units="degrees_east",
    )
