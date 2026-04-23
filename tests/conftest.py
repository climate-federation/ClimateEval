from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import benchmarking.diags._utils
from benchmarking import Variable
from benchmarking.data import ERA5, GPCP, MERRA2, DataSourceGenerator, HadCRUT5
from benchmarking.diags.simple import SimpleDiagnosticInputData
from tests.sample_data.create_sample_data_sources import (
    get_air_pressure_coord,
    get_altitude_coord,
    get_simple_2d_cube,
    get_simple_3d_cube,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from iris.cube import Cube

    from benchmarking.data import DataSource


pytest.register_assert_rewrite("tests._utils")
pytest.register_assert_rewrite("tests.unit")


@pytest.fixture
def sample_data_root_dir() -> Path:
    return Path(str(resources.files("tests"))) / "sample_data" / "data_sources"


@pytest.fixture(autouse=True)
def temporary_data_root_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    tmp_data_root_dir = tmp_path / "data"
    monkeypatch.setattr(
        benchmarking.diags._utils,
        "_DEFAULT_DATA_ROOT_DIR",
        tmp_data_root_dir,
    )
    return tmp_data_root_dir


@pytest.fixture
def variables_and_data_2d() -> dict[Variable, SimpleDiagnosticInputData]:
    return {
        Variable("my_tas", "tas", "mon"): SimpleDiagnosticInputData(reference=HadCRUT5),
        Variable("my_pr", "pr", "mon"): SimpleDiagnosticInputData(reference=GPCP),
    }


@pytest.fixture
def variables_and_data_2d_with_other() -> dict[Variable, SimpleDiagnosticInputData]:
    class MockDataSourceGenerator(DataSourceGenerator):
        def generate(self, _variable: Variable) -> Generator[type[DataSource]]:
            """Generate data sources."""
            yield from (ERA5,)

    return {
        Variable("my_tas", "tas", "mon"): SimpleDiagnosticInputData(
            reference=HadCRUT5,
            other=(MockDataSourceGenerator, MERRA2),
        ),
    }


@pytest.fixture
def variables_and_data_3d() -> dict[Variable, SimpleDiagnosticInputData]:
    return {
        Variable("ta", "ta", "mon"): SimpleDiagnosticInputData(reference=ERA5),
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
