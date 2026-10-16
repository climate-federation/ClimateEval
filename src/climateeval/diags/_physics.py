"""Provide pure physical formulae shared by the diagnostics.

Every function here takes plain :class:`numpy.ndarray` input and returns
plain numbers or arrays, so the physics can be unit-tested without any model
data, Iris cubes or ESMValCore preprocessing. The cube handling lives in the
diagnostics that call them (``climateeval.diags.complex``).

Sign conventions (CMIP6 fluxes, all in W m-2):

- TOA net downward flux ``N = rsdt - rsut - rlut``
- Surface net downward flux
  ``F_sfc = (rsds - rsus) + (rlds - rlus) - hfss - hfls``
- Latitudes in degrees; area weights proportional to cos(lat).

"""

from __future__ import annotations

import numpy as np
from scipy import stats

GRAVITY = 9.80665
"""Standard gravitational acceleration (m s-2)."""

LATENT_HEAT_VAPORIZATION = 2.5008e6
"""Latent heat of vaporization of water, L_v (J kg-1)."""

SECONDS_PER_DAY = 86400.0
"""Seconds per day, converting kg m-2 s-1 precipitation to mm day-1."""

SPECIFIC_HEAT_DRY_AIR = 1004.6
"""Specific heat of dry air at constant pressure, c_p (J kg-1 K-1)."""

GAS_CONSTANT_DRY_AIR = 287.04
"""Gas constant of dry air, R_d (J kg-1 K-1)."""

VIRTUAL_TEMPERATURE_FACTOR = 0.608
"""Factor in the virtual temperature T_v = T (1 + 0.608 q)."""

_MIN_REGRESSION_POINTS = 3
"""Fewest valid pairs a regression is computed from."""

_MIN_COLUMN_LEVELS = 2
"""Fewest pressure levels a column integral is computed from."""


def running_mean_drift(annual_series: np.ndarray, window: int = 10) -> float:
    """Drift per decade of the running mean of an annual series.

    The ordinary-least-squares slope of the ``window``-year running mean,
    expressed per decade. Smoothing first removes the interannual variability
    that would otherwise dominate the slope of a short record, leaving the
    secular drift of e.g. the global-mean TOA net flux of a control run.

    Parameters
    ----------
    annual_series:
        Annual-mean series.
    window:
        Length of the running mean (years).

    Returns
    -------
    :
        Drift in units of the input per decade.

    Raises
    ------
    ValueError
        If the series is too short for the running mean.

    """
    values = np.asarray(annual_series, dtype=float)
    if values.size < window + 2:
        msg = (
            f"need more than {window + 2} annual values for the {window}-year "
            f"running mean, got {values.size}"
        )
        raise ValueError(msg)
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    years = np.arange(smoothed.size, dtype=float)
    slope = stats.linregress(years, smoothed).slope  # per year
    return float(slope * 10.0)


def water_budget_residual(precipitation: float, latent_heat_flux: float) -> float:
    """Global water-budget residual abs(P - E) in mm day-1.

    Evaporation is derived from the surface latent heat flux,
    ``E = hfls / L_v``. In a steady state the global means of precipitation
    and evaporation balance.

    Parameters
    ----------
    precipitation:
        Global-mean precipitation (kg m-2 s-1).
    latent_heat_flux:
        Global-mean surface upward latent heat flux, hfls (W m-2).

    Returns
    -------
    :
        Residual (mm day-1).

    """
    evaporation = latent_heat_flux / LATENT_HEAT_VAPORIZATION
    return float(abs(precipitation - evaporation) * SECONDS_PER_DAY)


def atmospheric_energy_residual(
    precipitation: float,
    toa_net_flux: float,
    surface_net_radiation: float,
    sensible_heat_flux: float,
) -> float:
    """Atmospheric-column energy residual in W m-2.

    The atmospheric column is heated by the latent heat released by
    precipitation and by the surface sensible heat flux, and cooled
    radiatively, so in a steady state ``Q_rad = L_v P + SHF`` with ``Q_rad``
    the magnitude of the net radiative cooling of the column,

    ``Q_rad = [(rsds - rsus) + (rlds - rlus)] - (rsdt - rsut - rlut)``,

    i.e. the surface net radiation minus the TOA net flux (positive, about
    +100 W m-2 for the present-day atmosphere).

    Parameters
    ----------
    precipitation:
        Global-mean precipitation (kg m-2 s-1).
    toa_net_flux:
        Global-mean TOA net downward flux (W m-2).
    surface_net_radiation:
        Global-mean surface net downward radiation (W m-2).
    sensible_heat_flux:
        Global-mean surface upward sensible heat flux, hfss (W m-2).

    Returns
    -------
    :
        Absolute residual (W m-2).

    """
    latent_heating = LATENT_HEAT_VAPORIZATION * precipitation
    radiative_cooling = surface_net_radiation - toa_net_flux
    return float(abs(radiative_cooling - (latent_heating + sensible_heat_flux)))


def gridpoint_regression_slope(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Per-gridpoint least-squares slope of y on x over the leading axis.

    Parameters
    ----------
    y:
        Dependent field of shape ``(n_time, ...)``.
    x:
        Independent field, broadcastable to the shape of ``y``.

    Returns
    -------
    :
        Slope field of shape ``(...)``, NaN where the variance of ``x``
        vanishes.

    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    x_anomaly = x - x.mean(axis=0, keepdims=True)
    y_anomaly = y - y.mean(axis=0, keepdims=True)
    variance = (x_anomaly**2).mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (x_anomaly * y_anomaly).mean(axis=0) / variance
    return np.where(variance > 0, slope, np.nan)


def area_weighted_mean(field: np.ndarray, latitudes: np.ndarray) -> float:
    """Cosine-latitude-weighted mean of a field, ignoring non-finite points.

    Parameters
    ----------
    field:
        Field of shape ``(lat,)`` or ``(lat, lon)``.
    latitudes:
        Latitudes (degrees).

    Returns
    -------
    :
        Weighted mean, NaN if no point is finite.

    """
    field = np.asarray(field, dtype=float)
    weights = np.cos(np.deg2rad(np.asarray(latitudes, dtype=float)))
    if field.ndim == 2:  # noqa: PLR2004 - (lat, lon) versus (lat,)
        weights = np.broadcast_to(weights[:, None], field.shape)
    valid = np.isfinite(field)
    if not valid.any():
        return float("nan")
    return float((field * weights)[valid].sum() / weights[valid].sum())


def clausius_clapeyron_scaling(
    water_vapour_path: np.ndarray,
    temperature: np.ndarray,
) -> float:
    """Fractional water-vapour change per degree of warming (per cent K-1).

    The slope of the fractional anomaly of the column water vapour path on
    the temperature anomaly. Clausius-Clapeyron implies about 7 per cent K-1
    at fixed relative humidity (Held & Soden 2006).

    Parameters
    ----------
    water_vapour_path:
        Global annual-mean water vapour path, prw (kg m-2).
    temperature:
        Global annual-mean near-surface air temperature (K).

    Returns
    -------
    :
        Slope (per cent K-1).

    """
    prw = np.asarray(water_vapour_path, dtype=float)
    tas = np.asarray(temperature, dtype=float)
    prw_percent = (prw / prw.mean() - 1.0) * 100.0
    return float(stats.linregress(tas - tas.mean(), prw_percent).slope)


def moist_static_energy(
    temperature: np.ndarray,
    geopotential_height: np.ndarray,
    specific_humidity: np.ndarray,
) -> np.ndarray:
    """Moist static energy ``h = c_p T + g z + L_v q`` (J kg-1).

    Parameters
    ----------
    temperature:
        Air temperature, ta (K).
    geopotential_height:
        Geopotential height, zg (m) - the CMIP6 variable, hence ``g zg`` for
        the potential energy.
    specific_humidity:
        Specific humidity, hus (kg kg-1).

    Returns
    -------
    :
        Moist static energy (J kg-1).

    """
    return (
        SPECIFIC_HEAT_DRY_AIR * np.asarray(temperature, dtype=float)
        + GRAVITY * np.asarray(geopotential_height, dtype=float)
        + LATENT_HEAT_VAPORIZATION * np.asarray(specific_humidity, dtype=float)
    )


def mass_weighted_column_integral(
    field: np.ndarray,
    pressure: np.ndarray,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Mass-weighted vertical integral ``int field dp / g``.

    The integral spans the levels actually supplied. Monthly CMIP6 data are
    archived on the truncated ``plev19`` set, so this is a partial-column
    integral: an observational reference must be truncated the same way for
    the results to be comparable.

    Parameters
    ----------
    field:
        Field with a pressure axis, e.g. moist static energy (J kg-1).
    pressure:
        Pressure of each level (Pa), in any order.
    axis:
        Axis of ``field`` holding the pressure levels.

    Returns
    -------
    :
        Column integral with the pressure axis removed; J m-2 for a field in
        J kg-1.

    Raises
    ------
    ValueError
        If the pressure coordinate does not match the field.

    """
    field = np.asarray(field, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    if pressure.size != field.shape[axis]:
        msg = (
            f"pressure coordinate has {pressure.size} levels but the field has "
            f"{field.shape[axis]} along axis {axis}"
        )
        raise ValueError(msg)
    order = np.argsort(pressure)  # ascending pressure: top of the column first
    integral = np.trapezoid(
        np.take(field, order, axis=axis),
        pressure[order],
        axis=axis,
    )
    return integral / GRAVITY


def hydrostatic_height(
    temperature: np.ndarray,
    specific_humidity: np.ndarray,
    pressure: np.ndarray,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Geopotential height from the hypsometric equation, integrated upwards.

    ``z(p) = int (R_d T_v / g) d ln p``, integrated from the lowest level
    supplied, whose height is taken as zero. Intended as a fallback for
    observational products that provide temperature and humidity profiles but
    no geopotential height: the missing surface term is constant in time and
    therefore drops out of temporal anomalies.

    Parameters
    ----------
    temperature:
        Air temperature (K).
    specific_humidity:
        Specific humidity (kg kg-1).
    pressure:
        Pressure of each level (Pa), in any order.
    axis:
        Axis holding the pressure levels.

    Returns
    -------
    :
        Height above the lowest level supplied (m).

    Raises
    ------
    ValueError
        If the pressure coordinate does not match the field.

    """
    temperature = np.asarray(temperature, dtype=float)
    specific_humidity = np.asarray(specific_humidity, dtype=float)
    pressure = np.asarray(pressure, dtype=float)
    if pressure.size != temperature.shape[axis]:
        msg = (
            f"pressure coordinate has {pressure.size} levels but the field has "
            f"{temperature.shape[axis]} along axis {axis}"
        )
        raise ValueError(msg)
    order = np.argsort(pressure)[::-1]  # descending: bottom of the column first
    virtual_temperature = np.moveaxis(
        temperature * (1.0 + VIRTUAL_TEMPERATURE_FACTOR * specific_humidity),
        axis,
        0,
    )[order]
    levels = pressure[order]
    height = np.zeros_like(virtual_temperature)
    scale = GAS_CONSTANT_DRY_AIR / GRAVITY
    for k in range(levels.size - 1):
        thickness = (
            scale
            * 0.5
            * (virtual_temperature[k] + virtual_temperature[k + 1])
            * np.log(levels[k] / levels[k + 1])
        )
        height[k + 1] = height[k] + thickness
    out = np.empty_like(height)
    out[order] = height
    return np.moveaxis(out, 0, axis)


def pooled_regression_slope(y: np.ndarray, x: np.ndarray) -> float:
    """Least-squares slope of y on x with all (time, space) pairs pooled.

    Parameters
    ----------
    y:
        Dependent anomaly field.
    x:
        Independent anomaly field of the same shape.

    Returns
    -------
    :
        Slope, NaN if fewer than three finite pairs remain.

    Raises
    ------
    ValueError
        If the two fields have a different number of elements.

    """
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if y.size != x.size:
        msg = f"pooled regression needs equal shapes, got {y.size} and {x.size}"
        raise ValueError(msg)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < _MIN_REGRESSION_POINTS:
        return float("nan")
    return float(stats.linregress(x[valid], y[valid]).slope)


def deseasonalised_anomalies(field: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Remove the per-calendar-month climatology along the leading axis.

    The array counterpart of ESMValCore's ``anomalies(period="month")``, for
    fields that a diagnostic derives itself and therefore never holds as a
    cube.

    Parameters
    ----------
    field:
        Field of shape ``(n_time, ...)``.
    months:
        Calendar month (1-12) of each time step.

    Returns
    -------
    :
        Anomalies with respect to the monthly climatology.

    Raises
    ------
    ValueError
        If the number of months does not match the number of time steps.

    """
    field = np.asarray(field, dtype=float)
    months = np.asarray(months)
    if months.size != field.shape[0]:
        msg = f"got {months.size} months for {field.shape[0]} time steps"
        raise ValueError(msg)
    out = field.copy()
    for month in np.unique(months):
        selection = months == month
        out[selection] = field[selection] - np.nanmean(field[selection], axis=0)
    return out
