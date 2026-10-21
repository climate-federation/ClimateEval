"""Tests for the pure physical formulae shared by the diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from climateeval.diags import _physics


def test_running_mean_drift_recovers_a_linear_trend() -> None:
    years = np.arange(200)
    assert _physics.running_mean_drift(0.005 * years) == pytest.approx(0.05, rel=1e-6)


def test_running_mean_drift_ignores_fast_noise() -> None:
    rng = np.random.default_rng(0)
    years = np.arange(500)
    series = 0.002 * years + 0.5 * np.sin(2 * np.pi * years / 3.5)
    drift = _physics.running_mean_drift(series + rng.normal(0.0, 0.05, years.size))
    assert drift == pytest.approx(0.02, abs=0.005)


def test_running_mean_drift_of_a_short_series_fails() -> None:
    with pytest.raises(ValueError, match="annual values"):
        _physics.running_mean_drift(np.zeros(10))


def test_water_budget_residual() -> None:
    precipitation = 3.0 / _physics.SECONDS_PER_DAY  # 3 mm day-1
    latent_heat_flux = precipitation * _physics.LATENT_HEAT_VAPORIZATION
    assert _physics.water_budget_residual(
        precipitation,
        latent_heat_flux,
    ) == pytest.approx(0.0)
    # An evaporation 0.1 mm day-1 short of the precipitation
    assert _physics.water_budget_residual(
        precipitation,
        latent_heat_flux * (1.0 - 0.1 / 3.0),
    ) == pytest.approx(0.1, rel=1e-6)


def test_atmospheric_energy_residual_closes_an_earth_like_column() -> None:
    """L_v P = 80, SHF = 20 and Q_rad = 100 W m-2 balance exactly."""
    latent_heating = 80.0
    precipitation = latent_heating / _physics.LATENT_HEAT_VAPORIZATION
    sensible_heat_flux = 20.0
    toa_net_flux = 0.5
    surface_net_radiation = toa_net_flux + 100.0
    assert _physics.atmospheric_energy_residual(
        precipitation,
        toa_net_flux,
        surface_net_radiation,
        sensible_heat_flux,
    ) == pytest.approx(0.0, abs=1e-9)
    # A real imbalance is reported at its true size
    assert _physics.atmospheric_energy_residual(
        precipitation,
        toa_net_flux,
        surface_net_radiation + 3.0,
        sensible_heat_flux,
    ) == pytest.approx(3.0, abs=1e-9)


def test_gridpoint_regression_slope_recovers_a_known_slope_field() -> None:
    rng = np.random.default_rng(1)
    n_time, n_lat, n_lon = 100, 5, 4
    x = rng.normal(0.0, 1.0, (n_time, n_lat, n_lon))
    true_slope = np.linspace(1.0, 3.0, n_lat * n_lon).reshape(n_lat, n_lon)
    y = true_slope[None] * x + rng.normal(0.0, 0.01, x.shape)
    np.testing.assert_allclose(
        _physics.gridpoint_regression_slope(y, x),
        true_slope,
        atol=0.02,
    )


def test_gridpoint_regression_slope_is_nan_without_variance() -> None:
    x = np.ones((10, 2, 3))
    y = np.zeros((10, 2, 3))
    assert np.isnan(_physics.gridpoint_regression_slope(y, x)).all()


def test_area_weighted_mean_uses_cosine_weights() -> None:
    latitudes = np.array([-60.0, 0.0, 60.0])
    field = np.array([0.0, 1.0, 0.0])
    expected = np.cos(0.0) / (2 * np.cos(np.pi / 3) + np.cos(0.0))
    assert _physics.area_weighted_mean(field, latitudes) == pytest.approx(expected)


def test_area_weighted_mean_ignores_non_finite_points() -> None:
    latitudes = np.array([-30.0, 0.0, 30.0])
    field = np.array([[np.nan, np.nan], [2.0, 2.0], [np.nan, np.nan]])
    assert _physics.area_weighted_mean(field, latitudes) == pytest.approx(2.0)
    assert np.isnan(_physics.area_weighted_mean(np.full(3, np.nan), latitudes))


def test_clausius_clapeyron_scaling() -> None:
    rng = np.random.default_rng(5)
    temperature = 288.0 + rng.normal(0.0, 0.3, 200)
    water_vapour_path = 25.0 * (1 + 0.07 * (temperature - 288.0)) + rng.normal(
        0.0,
        0.05,
        200,
    )
    assert _physics.clausius_clapeyron_scaling(
        water_vapour_path,
        temperature,
    ) == pytest.approx(7.0, abs=0.5)


def test_moist_static_energy_sums_its_three_terms() -> None:
    energy = _physics.moist_static_energy(
        np.array([300.0]),
        np.array([1000.0]),
        np.array([0.01]),
    )
    expected = (
        _physics.SPECIFIC_HEAT_DRY_AIR * 300.0
        + _physics.GRAVITY * 1000.0
        + _physics.LATENT_HEAT_VAPORIZATION * 0.01
    )
    assert energy[0] == pytest.approx(expected)


def test_mass_weighted_column_integral_of_a_uniform_column() -> None:
    """A constant field integrates to ``value dp / g``, in any level order."""
    pressure = np.array([100000.0, 85000.0, 50000.0, 25000.0, 10000.0])
    field = np.full((pressure.size, 3, 4), 3.2e5)
    integral = _physics.mass_weighted_column_integral(field, pressure, axis=0)
    expected = 3.2e5 * (100000.0 - 10000.0) / _physics.GRAVITY
    assert integral.shape == (3, 4)
    np.testing.assert_allclose(integral, expected, rtol=1e-12)
    np.testing.assert_allclose(
        _physics.mass_weighted_column_integral(field[::-1], pressure[::-1], axis=0),
        expected,
    )
    with pytest.raises(ValueError, match="pressure coordinate"):
        _physics.mass_weighted_column_integral(field, pressure[:-1], axis=0)


def test_mass_weighted_column_integral_over_a_middle_axis() -> None:
    pressure = np.array([100000.0, 50000.0])
    field = np.ones((7, 2, 5, 6))
    integral = _physics.mass_weighted_column_integral(field, pressure, axis=1)
    assert integral.shape == (7, 5, 6)
    np.testing.assert_allclose(integral, 50000.0 / _physics.GRAVITY)


def test_hydrostatic_height_matches_an_isothermal_atmosphere() -> None:
    """A dry isothermal column: ``z = (R_d T / g) ln(p0 / p)``."""
    pressure = np.array([100000.0, 85000.0, 70000.0, 50000.0, 25000.0])
    temperature = np.full((pressure.size, 2, 3), 250.0)
    humidity = np.zeros_like(temperature)
    height = _physics.hydrostatic_height(temperature, humidity, pressure, axis=0)
    expected = (_physics.GAS_CONSTANT_DRY_AIR * 250.0 / _physics.GRAVITY) * np.log(
        pressure[0] / pressure,
    )
    np.testing.assert_allclose(height[:, 0, 0], expected, rtol=1e-6)
    # Moisture makes the column thicker (virtual temperature)
    moist = _physics.hydrostatic_height(
        temperature,
        np.full_like(temperature, 0.01),
        pressure,
        axis=0,
    )
    assert (moist[1:] > height[1:]).all()
    with pytest.raises(ValueError, match="pressure coordinate"):
        _physics.hydrostatic_height(temperature, humidity, pressure[:-1], axis=0)


def test_pooled_regression_slope_recovers_a_known_slope() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(0.0, 1.0, (60, 10, 20))  # (time, lat, lon) anomalies
    y = 0.35 * x + rng.normal(0.0, 0.02, x.shape)
    assert _physics.pooled_regression_slope(y, x) == pytest.approx(0.35, abs=0.01)
    # Non-finite pairs are dropped, not propagated
    masked = y.copy()
    masked[:, 0, :] = np.nan
    assert _physics.pooled_regression_slope(masked, x) == pytest.approx(0.35, abs=0.01)
    assert np.isnan(
        _physics.pooled_regression_slope(np.array([1.0]), np.array([1.0])),
    )
    with pytest.raises(ValueError, match="equal shapes"):
        _physics.pooled_regression_slope(np.zeros(4), np.zeros(5))


def test_deseasonalised_anomalies_removes_the_monthly_climatology() -> None:
    months = np.tile(np.arange(1, 13), 5)
    seasonal = 5.0 * np.sin(2 * np.pi * months / 12.0)
    field = (seasonal[:, None, None] + 2.0) * np.ones((months.size, 3, 4))
    assert np.abs(_physics.deseasonalised_anomalies(field, months)).max() < 1e-10
    with pytest.raises(ValueError, match="time steps"):
        _physics.deseasonalised_anomalies(np.zeros((5, 2)), np.arange(4))
