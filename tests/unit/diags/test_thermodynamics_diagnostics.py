"""Test the MSU-temperature and inversion-strength diagnostics on synthetic cubes."""

from __future__ import annotations

import datetime

import cf_units
import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from climateeval._variable import VARIABLES
from climateeval.data import DataSourceInformation
from climateeval.diags import _physics
from climateeval.diags.complex import InversionStrength, MSUTemperature

# The "2x2" grid the diagnostics regrid to: using it makes the regridding exact
LAT = np.arange(-89.0, 90.0, 2.0)
LON = np.arange(1.0, 360.0, 2.0)
PLEV = np.array(
    [
        100000.0,
        92500.0,
        85000.0,
        70000.0,
        60000.0,
        50000.0,
        40000.0,
        30000.0,
        25000.0,
        20000.0,
        15000.0,
        10000.0,
        7000.0,
        5000.0,
        3000.0,
        2000.0,
        1000.0,
        500.0,
        100.0,
    ],
)
N_MONTHS = 24
INFORMATION = DataSourceInformation(name="Model", category="test", exp="amip")
_OUTPUT_VARIABLES = {
    "tlt": "Temperature of the Lower Troposphere (MSU-Equivalent)",
    "tmt": "Temperature of the Mid Troposphere (MSU-Equivalent)",
    "tts": "Temperature of the Troposphere/Stratosphere (MSU-Equivalent)",
    "tls": "Temperature of the Lower Stratosphere (MSU-Equivalent)",
    "eis": "Estimated Inversion Strength",
    "lts": "Lower-Tropospheric Stability",
}


@pytest.fixture(autouse=True)
def output_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure the output variables exist (they are added to variables.yml)."""
    for name, long_name in _OUTPUT_VARIABLES.items():
        if name not in VARIABLES:
            monkeypatch.setitem(
                VARIABLES,
                name,
                {
                    "coordinates": ["time", "lat", "lon"],
                    "long_name": long_name,
                    "realm": "atmos",
                    "standard_name": "",
                    "units": "K",
                },
            )
    if "hurs" not in VARIABLES:
        monkeypatch.setitem(
            VARIABLES,
            "hurs",
            {
                "coordinates": ["time", "lat", "lon"],
                "long_name": "Near-Surface Relative Humidity",
                "realm": "atmos",
                "standard_name": "relative_humidity",
                "units": "%",
            },
        )


def _time_coord() -> DimCoord:
    units = cf_units.Unit("days since 1850-01-01 00:00:00", calendar="standard")
    starts = [
        units.date2num(datetime.datetime(1850 + m // 12, m % 12 + 1, 1))  # noqa: DTZ001
        for m in range(N_MONTHS + 1)
    ]
    bounds = np.array([[starts[i], starts[i + 1]] for i in range(N_MONTHS)])
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


def _surface_cube(var_name: str, value: float, units: str) -> Cube:
    data = np.full((N_MONTHS, LAT.size, LON.size), value, dtype=np.float32)
    return Cube(
        data,
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(_time_coord(), 0), (_lat_coord(), 1), (_lon_coord(), 2)],
    )


def _ta_cube(profile: np.ndarray, ps: float) -> Cube:
    """Get ta equal to a profile, masked where the level is below the surface."""
    data = np.broadcast_to(
        profile[None, :, None, None],
        (N_MONTHS, PLEV.size, LAT.size, LON.size),
    ).astype(np.float32)
    data = np.ma.masked_array(data.copy())
    data[:, ps < PLEV] = np.ma.masked
    return Cube(
        data,
        var_name="ta",
        standard_name="air_temperature",
        units="K",
        dim_coords_and_dims=[
            (_time_coord(), 0),
            (_plev_coord(), 1),
            (_lat_coord(), 2),
            (_lon_coord(), 3),
        ],
    )


def _table(output: object) -> object:
    return output.raw_output.execute().sort_values("time").reset_index(drop=True)


def test_msu_isothermal_atmosphere_with_masked_levels() -> None:
    ps = 90000.0
    cubes = CubeList(
        [
            _ta_cube(np.full(PLEV.size, 250.0), ps),
            _surface_cube("ts", 250.0, "K"),
            _surface_cube("ps", ps, "Pa"),
        ],
    )

    output = MSUTemperature("msu").get_output({"data": cubes}, INFORMATION)

    table = _table(output)
    assert list(table.columns) == ["data_id", "data_type", "time", "tlt", "tmt"]
    assert len(table) == N_MONTHS
    np.testing.assert_allclose(table["tlt"], 250.0, atol=1e-3)
    np.testing.assert_allclose(table["tmt"], 250.0, atol=1e-3)
    assert set(table["data_type"]) == {"to_benchmark"}


def test_msu_matches_the_formula_for_a_realistic_profile() -> None:
    ps = 101325.0
    # A tropical profile on PLEV, from 1000 hPa (299 K) to 1 hPa, tropopause at 100 hPa
    profile = np.array(
        [
            *(299, 296, 293, 286, 281, 273, 262, 248, 238, 226),
            *(213, 200, 206, 211, 218, 222, 228, 235, 250),
        ],
        dtype=float,
    )
    cubes = CubeList(
        [
            _ta_cube(profile, ps),
            _surface_cube("ts", 301.0, "K"),
            _surface_cube("ps", ps, "Pa"),
        ],
    )
    expected = {
        channel: float(
            _physics.msu_weighted_temperature(
                profile[:, None],
                PLEV,
                np.array([301.0]),
                np.array([ps]),
                channel,
            )[0],
        )
        for channel in ("tlt", "tmt")
    }

    table = _table(MSUTemperature("msu").get_output({"data": cubes}, INFORMATION))

    np.testing.assert_allclose(table["tlt"], expected["tlt"], atol=5e-3)
    np.testing.assert_allclose(table["tmt"], expected["tmt"], atol=5e-3)
    # A tropical profile: TLT is warmer than TMT, both between surface and tropopause
    assert 240.0 < expected["tmt"] < expected["tlt"] < 301.0


def test_msu_gridded_annual_and_other_data_key() -> None:
    ps = 101325.0
    cubes = CubeList(
        [
            _ta_cube(np.full(PLEV.size, 250.0), ps),
            _surface_cube("ts", 250.0, "K"),
            _surface_cube("ps", ps, "Pa"),
        ],
    )
    diagnostic = MSUTemperature(
        "msu",
        channels=("tmt",),
        spatial="gridded",
        aggregation="annual",
        data_key="amip",
    )

    output = diagnostic.get_output({"amip": cubes}, INFORMATION)

    table = output.raw_output.execute()
    assert {"data_id", "data_type", "time", "tmt"} <= set(table.columns)
    assert {"latitude", "longitude"} <= set(table.columns)
    assert len(table) == 2 * LAT.size * LON.size
    np.testing.assert_allclose(table["tmt"], 250.0, atol=1e-3)


def test_msu_uses_the_land_fraction_when_present() -> None:
    ps = 101325.0
    profile = np.linspace(290.0, 220.0, PLEV.size)
    sftlf = Cube(
        np.full((LAT.size, LON.size), 100.0, dtype=np.float32),
        var_name="sftlf",
        units="%",
        dim_coords_and_dims=[(_lat_coord(), 0), (_lon_coord(), 1)],
    )
    cubes = [
        _ta_cube(profile, ps),
        _surface_cube("ts", 300.0, "K"),
        _surface_cube("ps", ps, "Pa"),
    ]

    ocean = _table(
        MSUTemperature("msu", channels=("tlt",)).get_output(
            {"data": CubeList(cubes)},
            INFORMATION,
        ),
    )
    land = _table(
        MSUTemperature("msu", channels=("tlt",)).get_output(
            {"data": CubeList([*cubes, sftlf])},
            INFORMATION,
        ),
    )

    expected_land = _physics.msu_weighted_temperature(
        profile[:, None],
        PLEV,
        np.array([300.0]),
        np.array([ps]),
        "tlt",
        land_fraction=np.array([1.0]),
    )[0]
    np.testing.assert_allclose(land["tlt"], expected_land, atol=5e-3)
    assert not np.allclose(land["tlt"], ocean["tlt"], atol=1e-3)


def test_msu_invalid_options() -> None:
    with pytest.raises(ValueError, match="unknown channel"):
        MSUTemperature("msu", channels=("tmx",))
    with pytest.raises(ValueError, match="spatial"):
        MSUTemperature("msu", spatial="zonal")
    with pytest.raises(ValueError, match="aggregation"):
        MSUTemperature("msu", aggregation="daily")


def _eis_inputs(rh: float | None) -> CubeList:
    ps = 101300.0
    profile = np.full(PLEV.size, 250.0)
    profile[PLEV == 70000.0] = 281.0
    cubes = [
        _ta_cube(profile, ps),
        _surface_cube("ts", 291.0, "K"),
        _surface_cube("ps", ps, "Pa"),
    ]
    if rh is not None:
        cubes.append(_surface_cube("hurs", rh, "%"))
    return CubeList(cubes)


def test_inversion_strength_matches_the_formula() -> None:
    output = InversionStrength("eis", spatial="mean", lat_range=(-30, 30)).get_output(
        {"data": _eis_inputs(rh=None)},
        INFORMATION,
    )

    table = _table(output)
    assert list(table.columns) == ["data_id", "data_type", "time", "eis", "lts"]
    args = (np.array(281.0), np.array(291.0), np.array(101300.0))
    np.testing.assert_allclose(
        table["lts"],
        _physics.lower_tropospheric_stability(*args),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        table["eis"],
        _physics.estimated_inversion_strength(*args),
        rtol=1e-4,
    )


def test_inversion_strength_uses_hurs_when_available() -> None:
    args = (np.array(281.0), np.array(291.0), np.array(101300.0))

    table = _table(
        InversionStrength("eis", spatial="mean", outputs=("eis",)).get_output(
            {"data": _eis_inputs(rh=95.0)},
            INFORMATION,
        ),
    )

    np.testing.assert_allclose(
        table["eis"],
        _physics.estimated_inversion_strength(*args, rh_sfc=95.0),
        rtol=1e-4,
    )
    assert "lts" not in table.columns


def test_inversion_strength_gridded_with_tas_and_sea_mask_option() -> None:
    cubes = _eis_inputs(rh=None)
    cubes.append(_surface_cube("tas", 288.0, "K"))

    table = (
        InversionStrength("eis", surface_temperature="tas")
        .get_output({"data": cubes}, INFORMATION)
        .raw_output.execute()
    )

    assert len(table) == N_MONTHS * LAT.size * LON.size
    expected = _physics.lower_tropospheric_stability(
        np.array(281.0),
        np.array(288.0),
        np.array(101300.0),
    )
    np.testing.assert_allclose(table["lts"], expected, rtol=1e-4)


def test_inversion_strength_invalid_outputs() -> None:
    with pytest.raises(ValueError, match="unknown output"):
        InversionStrength("eis", outputs=("sst",))
