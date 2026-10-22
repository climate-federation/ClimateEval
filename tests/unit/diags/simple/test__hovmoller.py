"""Tests for the Hovmoller (time x longitude) diagnostic."""

from __future__ import annotations

from typing import Any

import cf_units
import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube
from numpy.lib.stride_tricks import sliding_window_view

from climateeval import Variable
from climateeval.diags.simple import Hovmoller, SimpleDiagnosticInputData
from climateeval.diags.simple._hovmoller import (
    _apply_rolling_mean,
    _remove_seasonal_cycle,
)

# Points of the 2x2 degree target grid of the diagnostic
LAT_2DEG = np.arange(-89.0, 90.0, 2.0)
LON_2DEG = np.arange(1.0, 360.0, 2.0)

# Two non-leap years of daily data (1850 and 1851)
N_DAYS = 730
WAVE_AMPLITUDE = 2.0
WAVE_PERIOD_DAYS = 40.0


def _time_coord(n_days: int = N_DAYS) -> DimCoord:
    time = DimCoord(
        np.arange(n_days) + 0.5,
        standard_name="time",
        var_name="time",
        units=cf_units.Unit("days since 1850-01-01", calendar="standard"),
    )
    time.guess_bounds()
    return time


def _wave(n_days: int, lon: np.ndarray, wavenumber: int) -> np.ndarray:
    """Eastward-propagating wave whose second year is the negative of the first.

    Its per-day-of-year climatology is therefore exactly zero.
    """
    day_of_year = np.arange(n_days) % 365
    sign = np.where(np.arange(n_days) < 365, 1.0, -1.0)
    phase = wavenumber * np.deg2rad(lon)[None, :] - (
        2.0 * np.pi * day_of_year[:, None] / WAVE_PERIOD_DAYS
    )
    return WAVE_AMPLITUDE * sign[:, None] * np.cos(phase)


def _seasonal_cycle(n_days: int) -> np.ndarray:
    day_of_year = np.arange(n_days) % 365
    return 5.0 + 3.0 * np.sin(2.0 * np.pi * day_of_year / 365.0)


def _pr_cube(lat_amplitude: np.ndarray | None = None) -> Cube:
    """Get a daily (time, lat, lon) cube on the 2x2 grid.

    Inside 15S-15N the field is a seasonal cycle plus a wavenumber-1 wave
    (scaled by ``lat_amplitude`` if given); outside it is a different,
    much larger wavenumber-3 wave that must not leak into the band mean.
    """
    in_band = np.abs(LAT_2DEG) <= 15.0
    if lat_amplitude is None:
        lat_amplitude = np.ones(LAT_2DEG.size)
    inside = _wave(N_DAYS, LON_2DEG, 1)[:, None, :] * lat_amplitude[None, :, None]
    outside = 50.0 * _wave(N_DAYS, LON_2DEG, 3)[:, None, :]
    data = np.where(in_band[None, :, None], inside, outside)
    data = data + _seasonal_cycle(N_DAYS)[:, None, None]

    lat = DimCoord(
        LAT_2DEG,
        standard_name="latitude",
        var_name="lat",
        units="degrees_north",
    )
    lon = DimCoord(
        LON_2DEG,
        standard_name="longitude",
        var_name="lon",
        units="degrees_east",
        circular=True,
    )
    lat.guess_bounds()
    lon.guess_bounds()
    return Cube(
        data,
        var_name="pr",
        standard_name="precipitation_flux",
        units="kg m-2 s-1",
        dim_coords_and_dims=[(_time_coord(), 0), (lat, 1), (lon, 2)],
    )


@pytest.fixture
def variable_pr_day() -> Variable:
    return Variable("pr", "pr", "day")


def _diagnostic(variable: Variable, **kwargs: Any) -> Hovmoller:
    return Hovmoller(
        "hovmoller",
        {variable: SimpleDiagnosticInputData(reference=None)},
        **kwargs,
    )


def test_default_settings(variable_pr_day: Variable) -> None:
    diagnostic = _diagnostic(variable_pr_day)
    assert diagnostic._lat_min == -15.0
    assert diagnostic._lat_max == 15.0
    assert diagnostic._rolling_window_days == 5


def test_unsmoothed_hovmoller_recovers_the_wave(variable_pr_day: Variable) -> None:
    """Band mean minus daily climatology is exactly the propagating wave."""
    diagnostic = _diagnostic(variable_pr_day, rolling_window_days=0)

    hovmoller = diagnostic._safely_preprocess(_pr_cube(), variable_pr_day)

    assert hovmoller.shape == (N_DAYS, LON_2DEG.size)
    assert [c.name() for c in hovmoller.coords(dim_coords=True)] == [
        "time",
        "longitude",
    ]
    np.testing.assert_allclose(hovmoller.coord("longitude").points, LON_2DEG)
    np.testing.assert_allclose(
        hovmoller.data,
        _wave(N_DAYS, LON_2DEG, 1),
        atol=1e-10,
    )


def test_smoothed_hovmoller_is_running_mean_of_the_wave(
    variable_pr_day: Variable,
) -> None:
    diagnostic = _diagnostic(variable_pr_day, rolling_window_days=5)

    hovmoller = diagnostic._safely_preprocess(_pr_cube(), variable_pr_day)

    expected = sliding_window_view(_wave(N_DAYS, LON_2DEG, 1), 5, axis=0).mean(-1)
    assert hovmoller.shape == (N_DAYS - 4, LON_2DEG.size)
    np.testing.assert_allclose(hovmoller.data, expected, atol=1e-10)
    # The time of each window is the centre of its 5 days
    np.testing.assert_allclose(
        hovmoller.coord("time").points,
        np.arange(2, N_DAYS - 2) + 0.5,
    )
    # A 5-day mean of a 40-day wave damps it by sin(5 pi/40) / (5 sin(pi/40))
    damping = np.sin(5.0 * np.pi / WAVE_PERIOD_DAYS) / (
        5.0 * np.sin(np.pi / WAVE_PERIOD_DAYS)
    )
    assert np.abs(hovmoller.data[:300]).max() == pytest.approx(
        WAVE_AMPLITUDE * damping,
        rel=1e-3,
    )


def test_wave_propagates_eastward(variable_pr_day: Variable) -> None:
    """The crest at 0E on day 0 travels 360 deg / 40 days = 9 deg per day.

    On the 2 degree grid the crest is found within 1 degree of 9 * day.
    """
    diagnostic = _diagnostic(variable_pr_day, rolling_window_days=0)

    hovmoller = diagnostic._safely_preprocess(_pr_cube(), variable_pr_day)

    crest_lon = LON_2DEG[np.argmax(hovmoller.data[:10], axis=1)]
    expected = 9.0 * np.arange(10)
    np.testing.assert_allclose(crest_lon, expected, atol=1.0)


def test_custom_latitude_band(variable_pr_day: Variable) -> None:
    """A band reaching into the outer field picks up the wavenumber-3 wave."""
    diagnostic = _diagnostic(
        variable_pr_day,
        lat_min=17.0,
        lat_max=21.0,
        rolling_window_days=1,
    )

    hovmoller = diagnostic._safely_preprocess(_pr_cube(), variable_pr_day)

    assert hovmoller.shape == (N_DAYS, LON_2DEG.size)
    np.testing.assert_allclose(
        hovmoller.data,
        50.0 * _wave(N_DAYS, LON_2DEG, 3),
        atol=1e-8,
    )


def test_meridional_mean_is_area_weighted(variable_pr_day: Variable) -> None:
    lat_amplitude = 1.0 + (LAT_2DEG / 15.0) ** 2
    diagnostic = _diagnostic(variable_pr_day, rolling_window_days=0)

    hovmoller = diagnostic._safely_preprocess(
        _pr_cube(lat_amplitude),
        variable_pr_day,
    )

    in_band = np.abs(LAT_2DEG) <= 15.0
    weights = np.cos(np.deg2rad(LAT_2DEG[in_band]))
    weighted = np.sum(weights * lat_amplitude[in_band]) / weights.sum()
    np.testing.assert_allclose(
        hovmoller.data,
        weighted * _wave(N_DAYS, LON_2DEG, 1),
        rtol=1e-6,
        atol=1e-10,
    )


def _time_series_cube(data: np.ndarray) -> Cube:
    return Cube(
        data,
        var_name="x",
        units="1",
        dim_coords_and_dims=[(_time_coord(data.shape[0]), 0)],
    )


def test_remove_seasonal_cycle_per_day_of_year() -> None:
    """Each calendar day loses the mean over its years; NaNs are ignored."""
    seasonal = _seasonal_cycle(N_DAYS)
    anomaly = np.where(np.arange(N_DAYS) < 365, 0.5, -0.5)
    data = seasonal + anomaly
    data[3] = np.nan  # day 4 of 1850: the climatology of that day is 1851 only

    result = _remove_seasonal_cycle(_time_series_cube(data))

    expected = anomaly.copy()
    expected[3] = np.nan
    expected[365 + 3] = 0.0
    np.testing.assert_allclose(result.data, expected, atol=1e-12)
    assert result.data.dtype == np.float64


def test_remove_seasonal_cycle_casts_integers_to_float() -> None:
    data = np.tile(np.arange(365), 2) + np.repeat([0, 1], 365)

    result = _remove_seasonal_cycle(_time_series_cube(data))

    np.testing.assert_allclose(result.data, np.repeat([-0.5, 0.5], 365))


@pytest.mark.parametrize("window", [0, 1])
def test_rolling_mean_is_skipped_for_short_windows(window: int) -> None:
    cube = _time_series_cube(np.arange(10.0))
    assert _apply_rolling_mean(cube, window) is cube


def test_rolling_mean_of_a_linear_series() -> None:
    """The 3-day mean of a linear series is the series at the window centre."""
    cube = _time_series_cube(np.arange(10.0))

    result = _apply_rolling_mean(cube, 3)

    np.testing.assert_allclose(result.data, np.arange(1.0, 9.0))
