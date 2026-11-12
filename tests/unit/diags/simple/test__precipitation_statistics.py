"""Test the precipitation-statistics diagnostic on synthetic daily cubes."""

from __future__ import annotations

import datetime

import cf_units
import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from climateeval import Variable
from climateeval.data import DataSourceInformation
from climateeval.diags import _physics
from climateeval.diags.simple import PrecipitationStatistics, SimpleDiagnosticInputData

# The "2x2" grid: regridding to it is exact. Four years of daily data, 1999-2002
LAT = np.arange(-9.0, 10.0, 2.0)
LON = np.arange(1.0, 20.0, 2.0)
UNITS = cf_units.Unit("days since 1990-01-01 00:00:00", calendar="standard")
START = datetime.datetime(1999, 1, 1)  # noqa: DTZ001
N_DAYS = (datetime.datetime(2003, 1, 1) - START).days  # noqa: DTZ001
INFORMATION = DataSourceInformation(name="Model", category="test", exp="amip")


def _time_coord() -> DimCoord:
    points = np.array(
        [
            UNITS.date2num(START + datetime.timedelta(days=d)) + 0.5
            for d in range(N_DAYS)
        ],
    )
    coord = DimCoord(
        points,
        standard_name="time",
        long_name="time",
        var_name="time",
        units=UNITS,
    )
    coord.guess_bounds()
    return coord


def _spatial_coords() -> tuple[DimCoord, DimCoord]:
    lat = DimCoord(
        LAT,
        var_name="lat",
        standard_name="latitude",
        long_name="Latitude",
        units="degrees_north",
    )
    lon = DimCoord(
        LON,
        var_name="lon",
        standard_name="longitude",
        long_name="Longitude",
        units="degrees_east",
    )
    lat.guess_bounds()
    lon.guess_bounds()
    return lat, lon


def _pr_cube(data: np.ndarray, units: str = "mm day-1") -> Cube:
    lat, lon = _spatial_coords()
    return Cube(
        data.astype(np.float32),
        var_name="pr",
        standard_name="precipitation_flux",
        units=units,
        dim_coords_and_dims=[(_time_coord(), 0), (lat, 1), (lon, 2)],
    )


def _rain(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (N_DAYS, LAT.size, LON.size)
    wet = rng.random(shape) < 0.4
    return np.where(wet, 1.0 + rng.gamma(1.5, 4.0, shape), 0.0)


def _variable(**diagnostic_kwargs: object) -> Variable:
    return Variable(
        "pr_box",
        "pr",
        "day",
        units="mm day-1",
        diagnostic_kwargs=dict(diagnostic_kwargs),
    )


def _diagnostic(variable: Variable) -> PrecipitationStatistics:
    return PrecipitationStatistics(
        "precip",
        {variable: SimpleDiagnosticInputData(reference=None)},
    )


def _column(cube: Cube, name: str) -> np.ndarray:
    names = list(cube.coord("statistic").points)
    return cube.data[:, names.index(name)]


def test_annual_statistics_match_the_formulae() -> None:
    data = _rain()
    variable = _variable(grid="2x2", regrid_scheme="linear")

    cube = _diagnostic(variable)._safely_preprocess(_pr_cube(data), variable)

    assert cube.shape[0] == 4  # 1999 ... 2002
    weights = np.broadcast_to(np.cos(np.deg2rad(LAT))[:, None], data.shape[1:])
    edges = _physics.log_rain_bins()
    year = data[:365]
    expected = _physics.precipitation_statistics(
        year,
        edges,
        weights=np.broadcast_to(weights, year.shape),
    )
    for key in ("wet_fraction", "sdii", "mean", "p90", "dry_fraction"):
        assert _column(cube, f"{key}_ann")[0] == pytest.approx(expected[key], rel=1e-4)
    assert _column(cube, "wet_fraction_ann")[0] == pytest.approx(0.4, abs=0.01)


def test_season_selection_and_incomplete_seasons() -> None:
    data = _rain()
    variable = _variable(grid="2x2", regrid_scheme="linear", seasons=["djf", "jja"])

    cube = _diagnostic(variable)._safely_preprocess(_pr_cube(data), variable)

    names = set(cube.coord("statistic").points)
    assert all(n.endswith(("_djf", "_jja")) for n in names)
    djf = _column(cube, "wet_fraction_djf")
    jja = _column(cube, "wet_fraction_jja")
    # DJF of 1999 lacks its December; JJA exists in all four years
    assert np.isnan(djf[0])
    assert np.isfinite(djf[1:]).all()
    assert np.isfinite(jja).all()
    # DJF 2000 = December 1999 + January and February 2000
    days = np.arange(N_DAYS)
    dates = np.array([START + datetime.timedelta(days=int(d)) for d in days])
    in_djf = np.array(
        [
            (d.year == 1999 and d.month == 12) or (d.year == 2000 and d.month <= 2)
            for d in dates
        ],
    )
    weights = np.broadcast_to(np.cos(np.deg2rad(LAT))[:, None], data.shape[1:])
    expected = _physics.wet_day_fraction(
        data[in_djf],
        axis=None,
        weights=np.broadcast_to(weights, data[in_djf].shape),
    )
    assert djf[1] == pytest.approx(expected, rel=1e-4)


def test_flux_units_are_converted_and_missing_data_are_ignored() -> None:
    # Wet days well above the threshold: the float32 round trip through
    # kg m-2 s-1 must not move any of them across 1 mm day-1
    data = np.where(_rain() > 0.0, _rain() + 0.5, 0.0)
    mm_cube = _pr_cube(data)
    masked = np.ma.masked_array(data / 86400.0, mask=False)
    masked[:, :3, :] = np.ma.masked  # a masked (e.g. land) band
    flux_cube = _pr_cube(masked.filled(0.0), units="kg m-2 s-1")
    flux_cube.data = masked.astype(np.float32)
    variable = _variable(grid="2x2", regrid_scheme="linear", seasons=["ann"])

    flux = _diagnostic(variable)._safely_preprocess(flux_cube, variable)
    # The same data in mm/day with the same band masked
    reference_cube = _pr_cube(data)
    reference_cube.data = np.ma.masked_array(
        data.astype(np.float32),
        mask=np.ma.getmaskarray(masked),
    )
    reference = _diagnostic(variable)._safely_preprocess(reference_cube, variable)

    np.testing.assert_allclose(
        _column(flux, "wet_fraction_ann"),
        _column(reference, "wet_fraction_ann"),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        _column(flux, "sdii_ann"),
        _column(reference, "sdii_ann"),
        rtol=1e-3,
    )
    assert mm_cube.units == "mm day-1"


def test_period_aggregation_pools_all_days() -> None:
    data = _rain()
    variable = _variable(
        grid="2x2",
        regrid_scheme="linear",
        seasons=["ann"],
        aggregation="period",
        percentiles=[50],
        statistics=["wet_fraction", "sdii", "p50"],
    )

    cube = _diagnostic(variable)._safely_preprocess(_pr_cube(data), variable)

    assert cube.shape == (1, 3)
    # Whole years only enter an annual-season pool, so use the days of 1999-2002
    weights = np.broadcast_to(np.cos(np.deg2rad(LAT))[:, None], data.shape)
    assert _column(cube, "wet_fraction_ann")[0] == pytest.approx(
        _physics.wet_day_fraction(data, axis=None, weights=weights),
        rel=1e-4,
    )
    assert _column(cube, "sdii_ann")[0] == pytest.approx(
        _physics.sdii(data, axis=None, weights=weights),
        rel=1e-4,
    )


def test_invalid_options_and_non_daily_data() -> None:
    cube = _pr_cube(_rain())
    for options, match in (
        ({"seasons": ["monsoon"]}, "unknown season"),
        ({"aggregation": "decade"}, "aggregation"),
    ):
        variable = _variable(grid="2x2", **options)
        with pytest.raises(ValueError, match=match):
            _diagnostic(variable)._safely_preprocess(cube, variable)

    monthly = cube[::30]
    variable = _variable(grid="2x2")
    with pytest.raises(ValueError, match="daily"):
        _diagnostic(variable)._safely_preprocess(monthly, variable)


def test_get_output_layout() -> None:
    data = _rain()
    variable = _variable(
        grid="2x2",
        regrid_scheme="linear",
        seasons=["ann", "jja"],
        statistics=["wet_fraction", "sdii"],
    )
    diagnostic = _diagnostic(variable)

    output = diagnostic.get_output(CubeList([_pr_cube(data)]), INFORMATION)

    table = output.raw_output.execute().sort_values("time").reset_index(drop=True)
    assert list(table.columns) == [
        "data_id",
        "data_type",
        "time",
        "pr_box_sdii_ann",
        "pr_box_sdii_jja",
        "pr_box_wet_fraction_ann",
        "pr_box_wet_fraction_jja",
    ]
    assert len(table) == 4
    assert set(table["data_type"]) == {"to_benchmark"}
    assert [t.year for t in table["time"]] == [1999, 2000, 2001, 2002]
    assert output.metrics is None
