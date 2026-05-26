from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import xarray as xr
from iris.cube import CubeList

import climateeval.diags._utils
from climateeval import Variable
from climateeval.data import GPCP, MERRA2, DataSourceGenerator, ERA5Monthly, HadCRUT5
from climateeval.diags.simple import SimpleDiagnosticInputData
from tests.sample_data.create_sample_data_sources import (
    get_air_pressure_coord,
    get_altitude_coord,
    get_simple_2d_cube,
    get_simple_3d_cube,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from iris.cube import Cube

    from climateeval.data import DataSource


pytest.register_assert_rewrite("tests._utils")
pytest.register_assert_rewrite("tests.integration")
pytest.register_assert_rewrite("tests.unit")


@pytest.fixture
def sample_data_root_dir() -> Path:
    return Path(str(resources.files("tests"))) / "sample_data" / "data_sources"


@pytest.fixture(autouse=True)
def temporary_data_root_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tmp_data_root_dir = tmp_path / "data"
    monkeypatch.setattr(
        climateeval.diags._base,
        "_DEFAULT_DATA_ROOT_DIR",
        tmp_data_root_dir,
    )
    return tmp_data_root_dir


@pytest.fixture
def variable_tas() -> Variable:
    return Variable(
        "my_tas",
        "tas",
        "mon",
        diagnostic_kwargs={"bin_range": (250.0, 350.0)},
    )


@pytest.fixture
def variable_ta() -> Variable:
    return Variable("ta", "ta", "mon", diagnostic_kwargs={"bin_range": (250.0, 350.0)})


@pytest.fixture
def variable_pr() -> Variable:
    return Variable("my_pr", "pr", "mon", diagnostic_kwargs={"bin_range": (0.0, 15.0)})


@pytest.fixture
def variables_and_data_2d(
    variable_tas: Variable,
    variable_pr: Variable,
) -> dict[Variable, SimpleDiagnosticInputData]:
    return {
        variable_tas: SimpleDiagnosticInputData(reference=HadCRUT5),
        variable_pr: SimpleDiagnosticInputData(reference=GPCP),
    }


@pytest.fixture
def variables_and_data_2d_with_other(
    variable_tas: Variable,
) -> dict[Variable, SimpleDiagnosticInputData]:
    class MockDataSourceGenerator(DataSourceGenerator):
        def generate(self, _variable: Variable) -> Generator[DataSource]:
            """Generate data sources."""
            yield from (ERA5Monthly(),)

    return {
        variable_tas: SimpleDiagnosticInputData(
            reference=HadCRUT5,
            other=(MockDataSourceGenerator, MERRA2),
        ),
    }


@pytest.fixture
def variables_and_data_3d(
    variable_ta: Variable,
) -> dict[Variable, SimpleDiagnosticInputData]:
    return {
        variable_ta: SimpleDiagnosticInputData(reference=ERA5Monthly),
    }


@pytest.fixture
def cube_2d() -> Cube:
    return get_simple_2d_cube(
        var_name="tas",
        standard_name="air_temperature",
        long_name="Near-Surface Air Temperature",
        units="K",
    )


@pytest.fixture(params=["altitude", "air_pressure"])
def cube_3d(request: pytest.FixtureRequest) -> Cube:
    if request.param == "altitude":
        z_coord = get_altitude_coord()
    else:
        z_coord = get_air_pressure_coord()
    return get_simple_3d_cube(
        z_coord,
        var_name="ta",
        standard_name="air_temperature",
        long_name="Air Temperature",
        units="K",
    )


@pytest.fixture
def cubes(cube_2d: Cube, cube_3d: Cube) -> CubeList:
    return CubeList([cube_2d, cube_3d])


@pytest.fixture
def xarray_ds() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "tas": xr.Variable(
                ("time", "lat", "lon"),
                np.arange(3 * 3 * 2, dtype=np.float32).reshape(3, 3, 2),
                attrs={
                    "standard_name": "air_temperature",
                    "long_name": "Near-Surface Air Temperature",
                    "units": "K",
                },
            ),
            "ta": xr.Variable(
                ("time", "plev", "lat", "lon"),
                np.arange(3 * 2 * 3 * 2, dtype=np.float32).reshape(3, 2, 3, 2),
                attrs={
                    "standard_name": "air_temperature",
                    "long_name": "Air Temperature",
                    "units": "K",
                },
            ),
        },
        coords={
            "time": xr.Variable(
                "time",
                [15.0, 380.0, 779.0],
                attrs={
                    "standard_name": "time",
                    "long_name": "Time Intervals",
                    "units": "days since 1970-01-01 00:00:00",
                },
            ),
            "plev": xr.Variable(
                "plev",
                [100000.0, 80000.0],
                attrs={
                    "standard_name": "air_pressure",
                    "long_name": "Pressure Levels",
                    "units": "Pa",
                    "positive": "down",
                },
            ),
            "lat": xr.Variable(
                "lat",
                [-30.0, 0.0, 30.0],
                attrs={
                    "standard_name": "lat",
                    "long_name": "Latitude",
                    "units": "degrees_north",
                },
            ),
            "lon": xr.Variable(
                "lon",
                [50.0, 100.0],
                attrs={
                    "standard_name": "lon",
                    "long_name": "Longitude",
                    "units": "degrees_east",
                },
            ),
        },
        attrs={"test": 1.0},
    )
