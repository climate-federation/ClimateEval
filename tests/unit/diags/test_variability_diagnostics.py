"""Tests for the coupled variability and tropical energetics diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from climateeval.data import DataSourceInformation
from climateeval.diags.complex import (
    BjerknesCompensation,
    ENSOTeleconnectionPatterns,
    ITCZEnergyFluxEquator,
)
from climateeval.diags.complex._transport import (
    EARTH_RADIUS_M,
    implied_meridional_transport,
)
from climateeval.diags.complex._variability import (
    itcz_transport_regression,
    monthly_anomalies,
    regression_pattern,
    running_mean_correlation,
    standardised,
    zero_crossing_nearest_equator,
)

if TYPE_CHECKING:
    from collections.abc import Callable

N_LAT = 36
N_LON = 36
WATTS_PER_PETAWATT = 1.0e15

#: Cross-equatorial transport (W) implied by a flux profile ``A cos(phi)``
CROSS_EQUATORIAL_PER_UNIT_FLUX = 2.0 * np.pi * EARTH_RADIUS_M**2 * np.pi / 4.0


def _info(exp: str = "piControl") -> DataSourceInformation:
    return DataSourceInformation(
        name="SynthModel",
        category="model",
        institute="Synth",
        exp=exp,
        variant="r1i1p1f1",
    )


def _monthly_cube(
    var_name: str,
    units: str,
    data_fn: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
    *,
    n_years: int,
    n_lat: int = N_LAT,
) -> Cube:
    """Global monthly cube with data from a builder function.

    The default latitude grid is coarser than, and does not reach as far
    poleward as, ClimateEval's 2x2 target grid, so the regridding leaves the
    polar rows masked - the situation a real model grid produces.
    """
    n_time = n_years * 12
    time = DimCoord(
        np.arange(n_time, dtype=float) * 30.0 + 15.0,
        standard_name="time",
        units=Unit("days since 1850-01-01", calendar="360_day"),
    )
    half_cell = 90.0 / n_lat
    lat = DimCoord(
        np.linspace(-90.0 + half_cell, 90.0 - half_cell, n_lat),
        standard_name="latitude",
        units="degrees",
    )
    lon = DimCoord(
        np.linspace(5.0, 355.0, N_LON),
        standard_name="longitude",
        units="degrees",
        circular=True,
    )
    for coord in (time, lat, lon):
        coord.guess_bounds()
    data = data_fn(n_time, lat.points, lon.points).astype(np.float32)
    return Cube(
        data,
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
    )


def _constant(value: float) -> Callable[[int, np.ndarray, np.ndarray], np.ndarray]:
    def build(n_time: int, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        return np.full((n_time, lats.size, lons.size), value)

    return build


def _row(output) -> dict:  # noqa: ANN001
    assert output.raw_output is not None
    return output.raw_output.to_pandas().iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Pure formulae
# ---------------------------------------------------------------------------


def test_zero_crossing_nearest_equator() -> None:
    lats = np.linspace(-40.0, 40.0, 81)
    assert zero_crossing_nearest_equator(lats - 5.0, lats) == pytest.approx(5.0)
    # Several crossings: the one nearest the equator wins
    profile = (lats - 3.0) * (lats - 30.0) * (lats + 25.0)
    assert zero_crossing_nearest_equator(profile, lats) == pytest.approx(3.0, abs=0.5)
    assert np.isnan(zero_crossing_nearest_equator(np.full(81, 2.0), lats))


def test_zero_crossing_makes_no_assumption_about_the_profile_length() -> None:
    for n_points in (7, 33, 181, 1980):
        lats = np.linspace(-40.0, 40.0, n_points)
        assert zero_crossing_nearest_equator(lats + 2.0, lats) == pytest.approx(
            -2.0,
            abs=80.0 / max(n_points - 1, 1),
        )


def test_itcz_transport_regression_recovers_a_slope() -> None:
    rng = np.random.default_rng(3)
    transport = np.sin(np.linspace(0.0, 2 * np.pi, 12, endpoint=False))  # PW
    itcz = -3.0 * transport + rng.normal(0.0, 0.05, 12)
    slope, correlation = itcz_transport_regression(itcz, transport)
    assert slope == pytest.approx(-3.0, abs=0.15)
    assert abs(correlation) > 0.99


def test_monthly_anomalies_removes_the_seasonal_cycle() -> None:
    months = np.arange(240)
    seasonal = 5.0 * np.sin(2 * np.pi * months / 12.0)
    assert np.abs(monthly_anomalies(seasonal + 2.0)).max() < 1e-10


def test_running_mean_correlation_of_anticorrelated_series() -> None:
    rng = np.random.default_rng(4)
    months = np.arange(2400)  # 200 years
    decadal = np.sin(2 * np.pi * months / 240.0)
    series_a = decadal + 0.3 * rng.normal(size=months.size)
    series_b = -decadal + 0.3 * rng.normal(size=months.size)
    assert running_mean_correlation(series_a, series_b, 121) < -0.8
    with pytest.raises(ValueError, match="too short"):
        running_mean_correlation(np.zeros(60), np.zeros(60), 121)


def test_standardised_series() -> None:
    values = standardised(np.array([1.0, 2.0, 3.0, 4.0]))
    assert values.mean() == pytest.approx(0.0)
    assert values.std(ddof=1) == pytest.approx(1.0)
    assert np.isnan(standardised(np.ones(5))).all()


def test_regression_pattern_recovers_a_known_pattern() -> None:
    rng = np.random.default_rng(7)
    index = rng.normal(0.0, 1.0, 200)
    pattern = np.linspace(-2.0, 2.0, 12).reshape(3, 4)
    field = pattern[None] * index[:, None, None] + rng.normal(
        0.0,
        0.01,
        (200, 3, 4),
    )
    np.testing.assert_allclose(regression_pattern(field, index), pattern, atol=0.01)
    # A masked grid point stays masked
    masked = field.copy()
    masked[:, 0, 0] = np.nan
    assert np.isnan(regression_pattern(masked, index)[0, 0])
    # An index without variance gives no pattern at all
    assert np.isnan(regression_pattern(field, np.ones(200))).all()


# ---------------------------------------------------------------------------
# ITCZ and the energy flux equator
# ---------------------------------------------------------------------------


def _itcz_cubes(slope: float, *, n_years: int = 2) -> CubeList:
    """Build a seasonal cycle whose ITCZ follows the energy transport.

    The atmospheric energy convergence is
    ``D (cos(phi) - pi/4) + A(m) cos(phi)``: the first term closes globally
    and carries no cross-equatorial transport, the second sets it to a
    prescribed value each month. The zonal-mean precipitation peaks at
    ``slope`` degrees per PW of that transport - on a latitude grid as fine
    as the target grid, so that the peak is not displaced by the regridding.
    """
    cross_equatorial_pw = 2.0 * np.sin(
        2 * np.pi * np.arange(12) / 12.0,
    )
    flux_amplitude = (
        cross_equatorial_pw * WATTS_PER_PETAWATT / CROSS_EQUATORIAL_PER_UNIT_FLUX
    )
    itcz_latitude = slope * cross_equatorial_pw

    def toa_incoming(
        n_time: int,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> np.ndarray:
        cos_lat = np.cos(np.deg2rad(lats))
        months = np.arange(n_time) % 12
        profile = (
            150.0 * (cos_lat - np.pi / 4.0)[None, :]
            + flux_amplitude[months][:, None] * cos_lat[None, :]
        )
        return np.broadcast_to(
            (400.0 + profile)[:, :, None],
            (n_time, lats.size, lons.size),
        ).copy()

    def precipitation(
        n_time: int,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> np.ndarray:
        months = np.arange(n_time) % 12
        centre = itcz_latitude[months][:, None]
        profile = np.exp(-(((lats[None, :] - centre) / 8.0) ** 2))
        return np.broadcast_to(
            (5.0 * profile / 86400.0)[:, :, None],
            (n_time, lats.size, lons.size),
        ).copy()

    zero = _constant(0.0)
    return CubeList(
        [
            _monthly_cube("rsdt", "W m-2", toa_incoming, n_years=n_years, n_lat=90),
            _monthly_cube("rsut", "W m-2", _constant(300.0), n_years=n_years, n_lat=90),
            _monthly_cube("rlut", "W m-2", _constant(100.0), n_years=n_years, n_lat=90),
            _monthly_cube("rsds", "W m-2", zero, n_years=n_years, n_lat=90),
            _monthly_cube("rsus", "W m-2", zero, n_years=n_years, n_lat=90),
            _monthly_cube("rlds", "W m-2", zero, n_years=n_years, n_lat=90),
            _monthly_cube("rlus", "W m-2", zero, n_years=n_years, n_lat=90),
            _monthly_cube("hfss", "W m-2", zero, n_years=n_years, n_lat=90),
            _monthly_cube("hfls", "W m-2", zero, n_years=n_years, n_lat=90),
            _monthly_cube(
                "pr",
                "kg m-2 s-1",
                precipitation,
                n_years=n_years,
                n_lat=90,
            ),
        ],
    )


def test_itcz_energy_flux_equator_recovers_a_prescribed_slope() -> None:
    diag = ITCZEnergyFluxEquator(
        "itcz_energy_flux_equator",
        fail_on_missing_data=True,
        download_missing_data=False,
    )
    output = diag.get_output({"historical": _itcz_cubes(-3.0)}, _info("historical"))
    row = _row(output)
    assert list(row) == [
        "data_id",
        "data_type",
        "itcz_efe_slope",
        "itcz_efe_correlation",
        "energy_flux_equator_lat",
    ]
    # The ITCZ latitude is quantised to the 2-degree grid, so the recovered
    # slope is close to, not exactly, the prescribed one
    assert row["itcz_efe_slope"] == pytest.approx(-3.0, abs=0.4)
    assert row["itcz_efe_correlation"] < -0.95
    assert abs(row["energy_flux_equator_lat"]) < 20.0


# ---------------------------------------------------------------------------
# Bjerknes compensation
# ---------------------------------------------------------------------------


def test_bjerknes_compensation_of_a_perfectly_compensating_control() -> None:
    """A control with no TOA anomaly compensates exactly: r = -1.

    With the TOA net flux held at zero, the atmospheric energy convergence
    is minus the surface net flux, so the two implied transports are exact
    mirror images of each other.
    """
    n_years = 20

    def surface_shortwave(
        n_time: int,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> np.ndarray:
        oscillation = np.sin(2 * np.pi * np.arange(n_time) / 120.0)
        shape = np.sin(np.deg2rad(lats))
        return np.broadcast_to(
            (200.0 + 20.0 * oscillation[:, None] * shape[None, :])[:, :, None],
            (n_time, lats.size, lons.size),
        ).copy()

    zero = _constant(0.0)
    cubes = CubeList(
        [
            _monthly_cube("rsdt", "W m-2", _constant(340.0), n_years=n_years),
            _monthly_cube("rsut", "W m-2", _constant(240.0), n_years=n_years),
            _monthly_cube("rlut", "W m-2", _constant(100.0), n_years=n_years),
            _monthly_cube("rsds", "W m-2", surface_shortwave, n_years=n_years),
            _monthly_cube("rsus", "W m-2", _constant(200.0), n_years=n_years),
            _monthly_cube("rlds", "W m-2", zero, n_years=n_years),
            _monthly_cube("rlus", "W m-2", zero, n_years=n_years),
            _monthly_cube("hfss", "W m-2", zero, n_years=n_years),
            _monthly_cube("hfls", "W m-2", zero, n_years=n_years),
        ],
    )
    diag = BjerknesCompensation(
        "bjerknes_compensation",
        fail_on_missing_data=True,
        download_missing_data=False,
        running_mean_months=25,
    )
    row = _row(diag.get_output({"picontrol": cubes}, _info()))
    assert list(row) == ["data_id", "data_type", "bjerknes_correlation"]
    assert row["bjerknes_correlation"] == pytest.approx(-1.0, abs=0.01)


# ---------------------------------------------------------------------------
# ENSO teleconnection patterns
# ---------------------------------------------------------------------------


def _enso_cubes(
    *,
    n_years: int = 10,
    seed: int = 21,
    temperature_sign: float = 1.0,
) -> CubeList:
    """Fields driven by one index with prescribed teleconnection patterns."""
    rng = np.random.default_rng(seed)
    index = standardised(rng.normal(0.0, 1.0, n_years * 12))

    def driven(
        pattern_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        background: float,
        amplitude: float,
    ) -> Callable[[int, np.ndarray, np.ndarray], np.ndarray]:
        def build(n_time: int, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
            pattern = pattern_fn(lats, lons)
            return background + amplitude * index[:n_time, None, None] * pattern[None]

        return build

    def temperature_pattern(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        return (
            np.cos(2 * np.pi * lons / 360.0)[None, :]
            * np.exp(-((lats / 15.0) ** 2))[:, None]
        )

    def precipitation_pattern(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        return (
            np.sin(2 * np.pi * lons / 360.0)[None, :]
            * np.exp(-((lats / 15.0) ** 2))[:, None]
        )

    return CubeList(
        [
            _monthly_cube(
                "tos",
                "degC",
                driven(temperature_pattern, 25.0, 1.5),
                n_years=n_years,
            ),
            _monthly_cube(
                "ts",
                "K",
                driven(temperature_pattern, 288.0, 1.5 * temperature_sign),
                n_years=n_years,
            ),
            _monthly_cube(
                "pr",
                "kg m-2 s-1",
                driven(precipitation_pattern, 4.0 / 86400.0, 1.0 / 86400.0),
                n_years=n_years,
            ),
        ],
    )


def _with_observations(
    diag: ENSOTeleconnectionPatterns,
    observations: CubeList,
) -> ENSOTeleconnectionPatterns:
    """Serve the observed cubes from a synthetic CubeList."""
    diag._observation_cube = lambda _source, variable: diag._prepared_cube(
        observations,
        variable,
    )
    return diag


def test_enso_teleconnection_patterns_of_a_perfect_model() -> None:
    """Model and observations sharing a pattern correlate at one."""
    diag = _with_observations(
        ENSOTeleconnectionPatterns(
            "enso_teleconnection_patterns",
            fail_on_missing_data=True,
            download_missing_data=False,
        ),
        _enso_cubes(seed=22),
    )
    row = _row(diag.get_output({"picontrol": _enso_cubes(seed=21)}, _info()))
    assert list(row) == [
        "data_id",
        "data_type",
        "enso_teleconnection_corr_ts",
        "enso_teleconnection_corr_pr",
        "nino34_std",
    ]
    assert row["enso_teleconnection_corr_ts"] == pytest.approx(1.0, abs=0.02)
    assert row["enso_teleconnection_corr_pr"] == pytest.approx(1.0, abs=0.02)
    assert row["nino34_std"] > 0.0


def test_enso_teleconnection_patterns_of_a_model_with_the_wrong_sign() -> None:
    diag = _with_observations(
        ENSOTeleconnectionPatterns(
            "enso_teleconnection_patterns",
            fail_on_missing_data=True,
            download_missing_data=False,
        ),
        _enso_cubes(seed=22),
    )
    row = _row(
        diag.get_output(
            {"picontrol": _enso_cubes(seed=21, temperature_sign=-1.0)},
            _info(),
        ),
    )
    assert row["enso_teleconnection_corr_ts"] == pytest.approx(-1.0, abs=0.02)
    assert row["enso_teleconnection_corr_pr"] == pytest.approx(1.0, abs=0.02)


def test_enso_teleconnection_patterns_without_observations() -> None:
    """Unobtainable observations leave the index amplitude, not an error."""
    diag = ENSOTeleconnectionPatterns(
        "enso_teleconnection_patterns",
        fail_on_missing_data=False,
        download_missing_data=False,
        use_observations=False,
    )
    row = _row(diag.get_output({"picontrol": _enso_cubes()}, _info()))
    assert row["nino34_std"] > 0.0
    assert np.isnan(row["enso_teleconnection_corr_ts"])
    assert np.isnan(row["enso_teleconnection_corr_pr"])


def test_implied_transport_of_the_synthetic_itcz_forcing() -> None:
    """The analytic cross-equatorial transport used by the ITCZ fixture."""
    lats = np.linspace(-89.5, 89.5, 180)
    flux = np.cos(np.deg2rad(lats))
    transport = implied_meridional_transport(flux, lats)
    assert np.interp(0.0, lats, transport) == pytest.approx(
        CROSS_EQUATORIAL_PER_UNIT_FLUX,
        rel=0.02,
    )
