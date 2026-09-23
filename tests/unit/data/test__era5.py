from __future__ import annotations

from typing import TYPE_CHECKING, Any

import iris
import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube

from climateeval import Variable
from climateeval.data import (
    DataSourceInformation,
    ERA5Daily,
    ERA5Hourly,
    ERA5Monthly,
)
from climateeval.data._era5 import VARIABLE_MAPPING
from tests.sample_data.create_sample_data_sources import (
    get_air_pressure_coord,
    get_simple_2d_cube,
    get_simple_3d_cube,
)

if TYPE_CHECKING:
    from pathlib import Path

    from climateeval.data._era5 import ERA5

SECONDS_PER_HOUR = 3600.0
WATER_DENSITY = 1000.0
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


# ERA5Daily


@pytest.fixture
def daily_data_source() -> ERA5Daily:
    return ERA5Daily()


def _save_raw_hourly_file(
    path: Path,
    var_name: str,
    units: str,
    *,
    year: int,
    n_days: int = 3,
) -> Cube:
    """Save a mock raw (i.e., not yet CMORized) hourly ERA5 file.

    Returns the cube that was saved so that tests can compare the daily
    statistics against the hourly input data.
    """
    n_times = 24 * n_days
    rng = np.random.default_rng(n_times + year)
    time = DimCoord(
        np.arange(n_times, dtype=np.float64),
        var_name="time",
        standard_name="time",
        units=f"hours since {year}-01-01 00:00:00",
    )
    lat = DimCoord(
        [-30.0, 30.0],
        var_name="lat",
        standard_name="latitude",
        units="degrees_north",
    )
    lon = DimCoord(
        [0.0, 90.0],
        var_name="lon",
        standard_name="longitude",
        units="degrees_east",
    )
    cube = Cube(
        rng.random((n_times, 2, 2), dtype=np.float32),
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
    )
    iris.save(cube, path)
    return cube


def test_daily_data_source(daily_data_source: ERA5Daily) -> None:
    assert daily_data_source.name == "ERA5"
    assert daily_data_source.information == DataSourceInformation(
        name="ERA5",
        category="reanalysis",
        institute="ECMWF",
        references=(
            "https://doi.org/10.24381/cds.f17050d7",
            "https://doi.org/10.24381/cds.6860a573",
            "https://doi.org/10.24381/cds.adbb2d47",
        ),
    )


@pytest.mark.parametrize(
    ("var_name", "source_var_name"),
    [("pr", "pr"), ("tas", "tas"), ("tasmax", "tas"), ("tasmin", "tas")],
)
def test_daily_source_var_name(
    daily_data_source: ERA5Daily,
    var_name: str,
    source_var_name: str,
) -> None:
    """The daily temperature extremes are aggregated from hourly ``tas``."""
    variable = Variable(var_name, var_name, "day")

    assert daily_data_source._get_source_var_name(variable) == source_var_name
    assert (
        daily_data_source._get_hourly_esmvaltool_facets(variable)["short_name"]
        == source_var_name
    )


@pytest.mark.parametrize(
    ("timerange", "years"),
    [
        ("1990/1990", [1990]),
        ("1990/1992", [1990, 1991, 1992]),
        ("19900101/19911231", [1990, 1991]),
    ],
)
def test_daily_get_years(
    daily_data_source: ERA5Daily,
    timerange: str,
    years: list[int],
) -> None:
    variable = Variable("tas", "tas", "day", timerange=timerange)

    assert daily_data_source._get_years(variable) == years


@pytest.mark.parametrize("timerange", ["*", "1990", "1990/*", "P5Y/1990"])
def test_daily_get_years_needs_explicit_timerange(
    daily_data_source: ERA5Daily,
    timerange: str,
) -> None:
    variable = Variable("tas", "tas", "day", timerange=timerange)

    with pytest.raises(ValueError, match="needs an explicit timerange"):
        daily_data_source._get_years(variable)


def test_daily_get_years_reversed_timerange(daily_data_source: ERA5Daily) -> None:
    variable = Variable("tas", "tas", "day", timerange="1992/1990")

    with pytest.raises(ValueError, match="must not be later than end year"):
        daily_data_source._get_years(variable)


def test_daily_get_cds_client_retrieve_kwargs(daily_data_source: ERA5Daily) -> None:
    variable = Variable("tasmax", "tasmax", "day", timerange="1990/1991")

    (dataset, request) = daily_data_source._get_cds_client_retrieve_kwargs(variable)

    assert dataset == "reanalysis-era5-single-levels"
    assert request["product_type"] == "reanalysis"
    assert request["year"] == ["1990", "1991"]
    assert len(request["month"]) == 12
    assert len(request["day"]) == 31
    assert len(request["time"]) == 24
    assert "pressure_level" not in request


def test_daily_get_cds_client_retrieve_kwargs_4d(
    daily_data_source: ERA5Daily,
) -> None:
    """Pressure level data cannot be aggregated to daily data (yet)."""
    variable = Variable("ta", "ta", "day", timerange="1990/1991")

    with pytest.raises(ValueError, match="is not supported for daily ERA5"):
        daily_data_source._get_cds_client_retrieve_kwargs(variable)


def test_daily_download_unsupported_variable(
    daily_data_source: ERA5Daily,
    tmp_path: Path,
) -> None:
    variable = Variable("tos", "tos", "day", timerange="1990/1991")

    with pytest.raises(NotImplementedError, match="is not supported for daily ERA5"):
        daily_data_source.download(variable, tmp_path)


def test_daily_download_requests_one_year_at_a_time(
    daily_data_source: ERA5Daily,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check the CDS requests without actually contacting the CDS."""
    retrieve_calls: list[tuple[str, dict[str, Any]]] = []
    format_calls: list[list[Path]] = []
    formatted = tmp_path / "tasmax.nc"

    class MockResult:
        def download(self, _target: Path) -> None:
            """Pretend the CDS delivered a file (``format`` is mocked, too)."""

    class MockClient:
        def retrieve(self, dataset: str, request: dict[str, Any]) -> MockResult:
            retrieve_calls.append((dataset, request))
            return MockResult()

    def mock_format(
        _self: ERA5Daily,
        input_file: list[Path],
        _variable: Variable,
        **_kwargs: Any,
    ) -> Path:
        format_calls.append(input_file)
        return formatted

    monkeypatch.setattr("climateeval.data._era5.cdsapi.Client", MockClient)
    monkeypatch.setattr(ERA5Daily, "format", mock_format)

    variable = Variable("tasmax", "tasmax", "day", timerange="1990/1992")
    paths = daily_data_source.download(variable, tmp_path)

    assert paths == [formatted]
    assert [request["year"] for (_dataset, request) in retrieve_calls] == [
        ["1990"],
        ["1991"],
        ["1992"],
    ]
    # ERA5 does not serve daily temperature extremes, they are aggregated from
    # the hourly 2m temperature
    for _dataset, request in retrieve_calls:
        assert request["variable"] == ["2m_temperature"]
        assert request["data_format"] == "netcdf"
    assert len(format_calls) == 1
    assert len(format_calls[0]) == 3


@pytest.mark.parametrize(
    ("var_name", "operator", "cell_method"),
    [
        ("tas", "mean", "mean"),
        ("tasmax", "max", "maximum"),
        ("tasmin", "min", "minimum"),
    ],
)
def test_daily_format_temperature(
    daily_data_source: ERA5Daily,
    tmp_path: Path,
    var_name: str,
    operator: str,
    cell_method: str,
) -> None:
    """Hourly 2m temperature is aggregated to the daily statistics."""
    raw_path = tmp_path / "raw.nc"
    raw_cube = _save_raw_hourly_file(raw_path, "t2m", "K", year=1990, n_days=3)

    target = daily_data_source.format(
        raw_path,
        Variable(var_name, var_name, "day", timerange="1990/1990"),
        data_root_dir=tmp_path / "data",
    )

    cube = iris.load_cube(target)
    assert cube.var_name == var_name
    assert cube.standard_name == "air_temperature"
    assert cube.units == "K"
    assert cube.coord("height").points == [2.0]
    assert [c.method for c in cube.cell_methods] == [cell_method]
    assert cube.shape == (3, 2, 2)
    expected = getattr(raw_cube.data.reshape(3, 24, 2, 2), operator)(axis=1)
    np.testing.assert_allclose(cube.data, expected, rtol=1e-6)


def test_daily_format_pr(daily_data_source: ERA5Daily, tmp_path: Path) -> None:
    """Accumulated hourly precipitation becomes a daily mean flux."""
    raw_path = tmp_path / "raw.nc"
    raw_cube = _save_raw_hourly_file(raw_path, "tp", "m", year=1990, n_days=3)

    target = daily_data_source.format(
        raw_path,
        Variable("pr", "pr", "day", timerange="1990/1990"),
        data_root_dir=tmp_path / "data",
    )

    cube = iris.load_cube(target)
    assert cube.var_name == "pr"
    assert cube.standard_name == "precipitation_flux"
    assert cube.units == "kg m-2 s-1"
    assert [c.method for c in cube.cell_methods] == ["mean"]
    # The hourly accumulations (m) are converted to fluxes before the
    # aggregation, so the daily mean flux is the daily total divided by the
    # length of a day
    hourly_flux = raw_cube.data * WATER_DENSITY / SECONDS_PER_HOUR
    # The accumulated field is shifted back by 30 minutes, so the raw 00:00
    # step belongs to the previous day (which is dropped by the timerange) and
    # the first day of the requested year runs from 00:30 to 23:30
    assert cube.shape == (3, 2, 2)
    np.testing.assert_allclose(
        cube.data[0],
        hourly_flux[1:25].mean(axis=0),
        rtol=1e-6,
    )


def test_daily_format_concatenates_years(
    daily_data_source: ERA5Daily,
    tmp_path: Path,
) -> None:
    """The raw files of all requested years are aggregated together."""
    raw_paths = [tmp_path / f"raw_{year}.nc" for year in (1990, 1991)]
    for path, year in zip(raw_paths, (1990, 1991), strict=True):
        _save_raw_hourly_file(path, "t2m", "K", year=year, n_days=2)

    target = daily_data_source.format(
        raw_paths,
        Variable("tasmax", "tasmax", "day", timerange="1990/1991"),
        data_root_dir=tmp_path / "data",
    )

    cube = iris.load_cube(target)
    assert cube.shape == (4, 2, 2)


def test_daily_format_output_dir(
    daily_data_source: ERA5Daily,
    tmp_path: Path,
) -> None:
    """Daily data is stored next to, not on top of, other frequencies."""
    raw_path = tmp_path / "raw.nc"
    _save_raw_hourly_file(raw_path, "t2m", "K", year=1990, n_days=2)

    target = daily_data_source.format(
        raw_path,
        Variable("tas", "tas", "day", timerange="1990/1990"),
        data_root_dir=tmp_path / "data",
    )

    assert target.parent == tmp_path / "data" / "reanalysis_ERA5" / "day" / "tas"
    assert target.name.startswith("tas_native6_ERA5_day_day_")
