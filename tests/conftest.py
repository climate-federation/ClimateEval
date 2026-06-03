from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import climateeval.diags._base
from climateeval import Variable
from tests.sample_data.create_sample_data_sources import (
    get_air_pressure_coord,
    get_altitude_coord,
    get_simple_2d_cube,
    get_simple_3d_cube,
)

if TYPE_CHECKING:
    from pathlib import Path

    from iris.cube import Cube


pytest.register_assert_rewrite("tests._utils")
pytest.register_assert_rewrite("tests.integration")
pytest.register_assert_rewrite("tests.unit")


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
