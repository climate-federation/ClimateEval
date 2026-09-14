from __future__ import annotations

from typing import TYPE_CHECKING, Any

import iris
import numpy as np
import pytest

from climateeval import Variable
from climateeval.data import DataSourceInformation, ERA5Hourly, ERA5Monthly
from climateeval.data._era5 import VARIABLE_MAPPING
from tests.sample_data.create_sample_data_sources import (
    get_air_pressure_coord,
    get_simple_2d_cube,
    get_simple_3d_cube,
)

if TYPE_CHECKING:
    from pathlib import Path

    from iris.cube import Cube

    from climateeval.data._era5 import ERA5

STANDARD_GRAVITY = 9.80665


@pytest.fixture(params=["monthly", "hourly"])
def data_source(request: pytest.FixtureRequest) -> ERA5:
    if request.param == "monthly":
        return ERA5Monthly()
    return ERA5Hourly()


def test_data_source(data_source: ERA5Monthly) -> None:
    assert data_source.name == "ERA5"
    assert data_source.information == DataSourceInformation(
        name="ERA5",
        category="reanalysis",
        institute="ECMWF",
        references=(
            "https://doi.org/10.24381/cds.f17050d7",
            "https://doi.org/10.24381/cds.6860a573",
            "https://doi.org/10.24381/cds.adbb2d47",
        ),
    )


@pytest.mark.parametrize("var_name", sorted(VARIABLE_MAPPING))
def test_variable_mapping_keys_are_known_variables(var_name: str) -> None:
    assert Variable(var_name, var_name, "mon").var_name == var_name


@pytest.mark.parametrize(
    ("var_name", "cds_name"),
    [
        ("ts", "skin_temperature"),
        ("zg", "geopotential"),
    ],
)
def test_variable_mapping(var_name: str, cds_name: str) -> None:
    assert VARIABLE_MAPPING[var_name] == cds_name


@pytest.mark.parametrize(
    ("var_name", "monthly_dataset", "hourly_dataset", "on_pressure_levels"),
    [
        (
            "ts",
            "reanalysis-era5-single-levels-monthly-means",
            "reanalysis-era5-single-levels",
            False,
        ),
        (
            "zg",
            "reanalysis-era5-pressure-levels-monthly-means",
            "reanalysis-era5-pressure-levels",
            True,
        ),
    ],
)
def test_get_cds_client_retrieve_kwargs(
    data_source: ERA5,
    var_name: str,
    monthly_dataset: str,
    hourly_dataset: str,
    on_pressure_levels: bool,
) -> None:
    variable = Variable(var_name, var_name, "mon")

    (dataset, request) = data_source._get_cds_client_retrieve_kwargs(variable)

    if isinstance(data_source, ERA5Monthly):
        assert dataset == monthly_dataset
    else:
        assert dataset == hourly_dataset
    assert ("pressure_level" in request) is on_pressure_levels


@pytest.mark.parametrize("var_name", ["ts", "zg"])
def test_get_esmvaltool_facets(data_source: ERA5, var_name: str) -> None:
    variable = Variable(var_name, var_name, "mon")

    facets = data_source._get_esmvaltool_facets(variable)

    assert facets["short_name"] == var_name
    assert facets["project"] == "native6"
    assert facets["dataset"] == "ERA5"


@pytest.mark.parametrize("var_name", ["ts", "zg"])
def test_download_requests_correct_cds_variable(
    data_source: ERA5,
    var_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the CDS request without actually contacting the CDS."""
    retrieve_calls: list[tuple[str, dict[str, Any]]] = []
    formatted = tmp_path / f"{var_name}.nc"

    class MockResult:
        def download(self, _target: Path) -> None:
            """Pretend the CDS delivered a file (``format`` is mocked, too)."""

    class MockClient:
        def retrieve(self, dataset: str, request: dict[str, Any]) -> MockResult:
            retrieve_calls.append((dataset, request))
            return MockResult()

    monkeypatch.setattr("climateeval.data._era5.cdsapi.Client", MockClient)
    monkeypatch.setattr(
        type(data_source),
        "format",
        lambda *_args, **_kwargs: formatted,
    )

    paths = data_source.download(Variable(var_name, var_name, "mon"), tmp_path)

    assert paths == [formatted]
    assert len(retrieve_calls) == 1
    (_dataset, request) = retrieve_calls[0]
    assert request["variable"] == [VARIABLE_MAPPING[var_name]]
    assert request["data_format"] == "netcdf"


def test_download_unsupported_variable(data_source: ERA5, tmp_path: Path) -> None:
    variable = Variable("tos", "tos", "mon")

    with pytest.raises(NotImplementedError, match="is not supported for ERA5"):
        data_source.download(variable, tmp_path)


def _save_raw_cube(cube: Cube, tmp_path: Path) -> Path:
    """Save a mock raw (i.e., not yet CMORized) ERA5 file."""
    path = tmp_path / "raw.nc"
    iris.save(cube, path)
    return path


def test_format_ts(tmp_path: Path) -> None:
    """ERA5 skin temperature is CMORized to the surface temperature ``ts``."""
    raw_cube = get_simple_2d_cube(var_name="skt", units="K")
    raw_cube.data = raw_cube.core_data() + 250.0
    raw_path = _save_raw_cube(raw_cube, tmp_path)

    target = ERA5Monthly().format(
        raw_path,
        Variable("ts", "ts", "mon"),
        data_root_dir=tmp_path / "data",
    )

    cube = iris.load_cube(target)
    assert cube.var_name == "ts"
    assert cube.standard_name == "surface_temperature"
    assert cube.units == "K"
    np.testing.assert_allclose(
        np.sort(cube.data.ravel()),
        np.sort(raw_cube.data.ravel()),
        rtol=1e-6,
    )


def test_format_zg_converts_geopotential_to_height(tmp_path: Path) -> None:
    """ERA5 geopotential is divided by the standard gravity to give ``zg``."""
    raw_cube = get_simple_3d_cube(
        get_air_pressure_coord(),
        var_name="z",
        units="m2 s-2",
    )
    raw_cube.data = raw_cube.core_data() + 50000.0
    raw_path = _save_raw_cube(raw_cube, tmp_path)

    target = ERA5Monthly().format(
        raw_path,
        Variable("zg", "zg", "mon"),
        data_root_dir=tmp_path / "data",
    )

    cube = iris.load_cube(target)
    assert cube.var_name == "zg"
    assert cube.standard_name == "geopotential_height"
    assert cube.units == "m"
    np.testing.assert_allclose(
        np.sort(cube.data.ravel()),
        np.sort(raw_cube.data.ravel() / STANDARD_GRAVITY),
        rtol=1e-6,
    )
