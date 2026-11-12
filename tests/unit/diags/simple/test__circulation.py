"""Test the circulation-index diagnostics on synthetic cubes with known answers."""

from __future__ import annotations

import datetime

import cf_units
import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from climateeval import Variable
from climateeval.data import DataSourceInformation
from climateeval.diags.simple import (
    EddyDrivenJetLatitude,
    HadleyCellEdge,
    SimpleDiagnosticInputData,
    WalkerCirculationStrength,
)

# The "2x2" grid the diagnostics regrid to: using it makes the regridding exact
LAT = np.arange(-89.0, 90.0, 2.0)
LON = np.arange(1.0, 360.0, 2.0)
PLEV = np.arange(2500.0, 100001.0, 2500.0)
N_MONTHS = 24


def _time_coord(n_months: int = N_MONTHS) -> DimCoord:
    units = cf_units.Unit("days since 1850-01-01 00:00:00", calendar="standard")
    starts = [
        units.date2num(datetime.datetime(1850 + m // 12, m % 12 + 1, 1))  # noqa: DTZ001
        for m in range(n_months + 1)
    ]
    bounds = np.array([[starts[i], starts[i + 1]] for i in range(n_months)])
    return DimCoord(
        bounds.mean(axis=1),
        bounds=bounds,
        var_name="time",
        standard_name="time",
        long_name="Time Intervals",
        units=units,
    )


def _lat_coord() -> DimCoord:
    coord = DimCoord(
        LAT,
        var_name="lat",
        standard_name="latitude",
        long_name="Latitude",
        units="degrees_north",
    )
    coord.guess_bounds()
    return coord


def _lon_coord() -> DimCoord:
    coord = DimCoord(
        LON,
        var_name="lon",
        standard_name="longitude",
        long_name="Longitude",
        units="degrees_east",
    )
    coord.guess_bounds()
    return coord


def _plev_coord() -> DimCoord:
    return DimCoord(
        PLEV,
        var_name="plev",
        standard_name="air_pressure",
        long_name="pressure",
        units="Pa",
        attributes={"positive": "down"},
    )


def _diagnostic(
    cls: type,
    variable: Variable,
) -> HadleyCellEdge | EddyDrivenJetLatitude | WalkerCirculationStrength:
    return cls("diagnostic", {variable: SimpleDiagnosticInputData(reference=None)})


def _va_cube(zero_by_year: tuple[float, float], mask_below: float | None) -> Cube:
    """Get va = A(phi, year) sin(pi p / 1e5) with A crossing zero at +-zero_by_year."""
    time = _time_coord()
    year = np.repeat([0, 1], 12)
    amplitude = np.empty((N_MONTHS, LAT.size))
    for t in range(N_MONTHS):
        amplitude[t] = np.sin(np.pi * LAT / zero_by_year[year[t]]) * (
            1.0 + 0.05 * (t % 12)
        )
    profile = np.sin(np.pi * PLEV / 1.0e5)
    data = amplitude[:, None, :, None] * profile[None, :, None, None]
    data = np.broadcast_to(data, (N_MONTHS, PLEV.size, LAT.size, LON.size)).copy()
    data = np.ma.masked_array(data.astype(np.float32))
    if mask_below is not None:
        data[:, mask_below <= PLEV] = np.ma.masked
    return Cube(
        data,
        var_name="va",
        standard_name="northward_wind",
        units="m s-1",
        dim_coords_and_dims=[
            (time, 0),
            (_plev_coord(), 1),
            (_lat_coord(), 2),
            (_lon_coord(), 3),
        ],
    )


@pytest.mark.parametrize("mask_below", [None, 90000.0])
@pytest.mark.parametrize(("hemisphere", "sign"), [("north", 1.0), ("south", -1.0)])
def test_hadley_cell_edge_annual(
    hemisphere: str,
    sign: float,
    mask_below: float | None,
) -> None:
    variable = Variable(
        "hce",
        "va",
        "mon",
        diagnostic_kwargs={"hemisphere": hemisphere},
    )
    diagnostic = _diagnostic(HadleyCellEdge, variable)
    # The annual-mean streamfunction crosses zero at 30 degrees in year 1, 24 in 2
    cube = _va_cube((30.0, 24.0), mask_below)

    result = diagnostic._safely_preprocess(cube, variable)

    assert result.shape == (2,)
    np.testing.assert_allclose(result.data, sign * np.array([30.0, 24.0]), atol=0.1)
    assert str(result.units) == "degrees_north"
    time = result.coord("time")
    years = [d.year for d in time.units.num2date(time.points)]
    assert years == [1850, 1851]


def test_hadley_cell_edge_monthly_and_season() -> None:
    monthly = Variable("hce", "va", "mon", diagnostic_kwargs={"aggregation": "monthly"})
    cube = _va_cube((30.0, 24.0), None)

    result = _diagnostic(HadleyCellEdge, monthly)._safely_preprocess(cube, monthly)

    assert result.shape == (N_MONTHS,)
    np.testing.assert_allclose(result.data[:12], 30.0, atol=0.1)
    np.testing.assert_allclose(result.data[12:], 24.0, atol=0.1)

    jja = Variable("hce", "va", "mon", diagnostic_kwargs={"aggregation": "JJA"})
    result = _diagnostic(HadleyCellEdge, jja)._safely_preprocess(cube, jja)
    np.testing.assert_allclose(result.data, [30.0, 24.0], atol=0.1)


def test_invalid_aggregation() -> None:
    variable = Variable("hce", "va", "mon", diagnostic_kwargs={"aggregation": "weekly"})
    with pytest.raises(ValueError, match="aggregation"):
        _diagnostic(HadleyCellEdge, variable)._safely_preprocess(
            _va_cube((30.0, 30.0), None),
            variable,
        )


def _ua_cube(centres: tuple[float, float]) -> Cube:
    """Zonal wind with Gaussian jets (centres in the north, south mirrored)."""
    time = _time_coord()
    year = np.repeat([0, 1], 12)
    lon_pattern = 3.0 * np.sin(np.deg2rad(LON))  # zero zonal mean
    data = np.empty((N_MONTHS, LAT.size, LON.size), dtype=np.float32)
    for t in range(N_MONTHS):
        north = 12.0 * np.exp(-(((LAT - centres[year[t]]) / 8.0) ** 2) / 2.0)
        south = 15.0 * np.exp(-(((LAT + centres[year[t]] + 5.0) / 7.0) ** 2) / 2.0)
        data[t] = (north + south + 1.0)[:, None] + lon_pattern[None, :]
    return Cube(
        data,
        var_name="ua",
        standard_name="eastward_wind",
        units="m s-1",
        dim_coords_and_dims=[(time, 0), (_lat_coord(), 1), (_lon_coord(), 2)],
    )


@pytest.mark.parametrize(
    ("hemisphere", "expected"),
    [("north", (47.3, 41.9)), ("south", (-52.3, -46.9))],
)
def test_jet_latitude_annual(hemisphere: str, expected: tuple[float, float]) -> None:
    variable = Variable(
        "jet",
        "ua",
        "mon",
        diagnostic_kwargs={"hemisphere": hemisphere},
    )

    result = _diagnostic(EddyDrivenJetLatitude, variable)._safely_preprocess(
        _ua_cube((47.3, 41.9)),
        variable,
    )

    np.testing.assert_allclose(result.data, expected, atol=0.1)


def test_jet_latitude_selects_850hpa_from_a_pressure_axis() -> None:
    variable = Variable("jet", "ua", "mon")
    ua = _ua_cube((47.3, 41.9))
    # A 3-level field in which only the 85000 Pa level carries the jet
    levels = np.array([70000.0, 85000.0, 100000.0])
    wrong = np.where(LAT > 0, 100.0 - LAT, 0.0)[None, None, :, None] * np.ones(
        (N_MONTHS, 1, 1, LON.size),
    )
    data = np.concatenate([wrong[:, :1], ua.data[:, None], wrong[:, :1]], axis=1)
    plev = DimCoord(
        levels,
        var_name="plev",
        standard_name="air_pressure",
        long_name="pressure",
        units="Pa",
        attributes={"positive": "down"},
    )
    cube = Cube(
        data.astype(np.float32),
        var_name="ua",
        standard_name="eastward_wind",
        units="m s-1",
        dim_coords_and_dims=[
            (ua.coord("time"), 0),
            (plev, 1),
            (_lat_coord(), 2),
            (_lon_coord(), 3),
        ],
    )

    result = _diagnostic(EddyDrivenJetLatitude, variable)._safely_preprocess(
        cube,
        variable,
    )

    np.testing.assert_allclose(result.data, [47.3, 41.9], atol=0.1)


def _psl_cube() -> Cube:
    time = _time_coord()
    scale = 1.0 + (np.arange(N_MONTHS) % 12)  # 1..12 hPa-ish amplitude per month
    data = np.full((N_MONTHS, LAT.size, LON.size), 101000.0)
    east = (np.abs(LAT) <= 5.0)[:, None] & ((LON >= 200.0) & (LON <= 280.0))[None, :]
    west = (np.abs(LAT) <= 5.0)[:, None] & ((LON >= 80.0) & (LON <= 160.0))[None, :]
    data += scale[:, None, None] * (30.0 * east - 20.0 * west)[None]
    data[:, np.abs(LAT) > 10.0, :] += 3000.0
    return Cube(
        data.astype(np.float64),
        var_name="psl",
        standard_name="air_pressure_at_mean_sea_level",
        units="Pa",
        dim_coords_and_dims=[(time, 0), (_lat_coord(), 1), (_lon_coord(), 2)],
    )


def test_walker_strength_monthly_and_annual() -> None:
    cube = _psl_cube()
    scale = 1.0 + (np.arange(N_MONTHS) % 12)

    monthly = Variable(
        "walker_mon",
        "psl",
        "mon",
        diagnostic_kwargs={"aggregation": "monthly"},
    )
    result = _diagnostic(WalkerCirculationStrength, monthly)._safely_preprocess(
        cube,
        monthly,
    )
    np.testing.assert_allclose(result.data, 50.0 * scale)
    assert str(result.units) == "Pa"

    annual = Variable("walker", "psl", "mon")
    result = _diagnostic(WalkerCirculationStrength, annual)._safely_preprocess(
        cube,
        annual,
    )
    np.testing.assert_allclose(result.data, [50.0 * 6.5, 50.0 * 6.5])


def test_walker_strength_custom_boxes() -> None:
    variable = Variable(
        "walker",
        "psl",
        "mon",
        diagnostic_kwargs={
            "aggregation": "monthly",
            "east_box": [[-5, 5], [80, 160]],
            "west_box": [[-5, 5], [200, 280]],
        },
    )

    result = _diagnostic(WalkerCirculationStrength, variable)._safely_preprocess(
        _psl_cube(),
        variable,
    )

    assert result.data[0] == pytest.approx(-50.0)


def test_get_output_layout_of_a_circulation_index() -> None:
    """Write the time-series layout: data_id, data_type, time, variable ID."""
    variable = Variable(
        "walker",
        "psl",
        "mon",
        diagnostic_kwargs={"aggregation": "monthly"},
    )
    diagnostic = _diagnostic(WalkerCirculationStrength, variable)
    information = DataSourceInformation(name="Model", category="test", exp="amip")

    output = diagnostic.get_output(CubeList([_psl_cube()]), information)

    table = output.raw_output.execute()
    assert list(table.columns) == ["data_id", "data_type", "time", "walker"]
    assert set(table["data_type"]) == {"to_benchmark"}
    assert set(table["data_id"]) == {information.id}
    assert len(table) == N_MONTHS
    np.testing.assert_allclose(
        table.sort_values("time")["walker"].to_numpy(),
        50.0 * (1.0 + np.arange(N_MONTHS) % 12),
        rtol=1e-5,
    )
