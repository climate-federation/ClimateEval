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

import functools
from importlib import resources

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
            f"need at least {window + 2} annual values for the {window}-year "
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


# ---------------------------------------------------------------------------
# Circulation indices (Hadley-cell edge, eddy-driven jet, Walker circulation)
# ---------------------------------------------------------------------------

EARTH_RADIUS = 6.371e6
"""Mean radius of the Earth, a (m)."""

_HEMISPHERE_SIGN = {"north": 1.0, "south": -1.0}
_MIN_FIT_POINTS = 3
_FULL_CIRCLE = 360.0
"""Fewest points a quadratic is fitted through."""


def _hemisphere_sign(hemisphere: str) -> float:
    """Get +1 (north) or -1 (south) for a hemisphere name."""
    try:
        return _HEMISPHERE_SIGN[hemisphere]
    except KeyError:
        msg = f"hemisphere must be 'north' or 'south', got '{hemisphere}'"
        raise ValueError(msg) from None


def meridional_streamfunction(
    v_zonal_mean: np.ndarray,
    plev: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Mean meridional mass streamfunction of the zonal-mean flow.

    ``psi(phi, p) = (2 pi a cos(phi) / g) int_0^p [v] dp'`` with the integral
    taken from the top of the atmosphere (trapezoidal rule between the supplied
    levels, ``[v]`` held constant between ``p = 0`` and the highest level). The
    sign convention is that of Adam et al. (2018, GMD, 11, 4339) and the TropD
    package: positive for the clockwise (northern Hadley) cell.

    Levels at which ``[v]`` is NaN (below the surface, where model output on
    pressure levels is masked) are treated as missing: the streamfunction is
    NaN at and below the first missing level of a column and is unaffected
    above it. Masked values are never read as data; convert masked arrays with
    ``np.ma.filled(x, np.nan)`` first.

    Parameters
    ----------
    v_zonal_mean:
        Zonal-mean meridional wind (m s-1), shape ``(n_lev, n_lat)``.
    plev:
        Pressure of each level (Pa), in any order.
    lat:
        Latitude (degrees), length ``n_lat``.

    Returns
    -------
    :
        Streamfunction (kg s-1) with the shape of ``v_zonal_mean``.

    Raises
    ------
    ValueError
        If the coordinates do not match the field.

    """
    v = np.asarray(v_zonal_mean, dtype=float)
    plev = np.asarray(plev, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if v.ndim != 2 or v.shape != (plev.size, lat.size):  # noqa: PLR2004
        msg = (
            f"expected [v] of shape (n_lev, n_lat) = {(plev.size, lat.size)}, "
            f"got {v.shape}"
        )
        raise ValueError(msg)
    order = np.argsort(plev)  # top of the atmosphere first
    p = plev[order]
    vs = v[order]
    increments = np.empty_like(vs)
    increments[0] = vs[0] * p[0]
    increments[1:] = 0.5 * (vs[1:] + vs[:-1]) * np.diff(p)[:, None]
    integral = np.cumsum(increments, axis=0)  # NaN propagates downwards only
    psi_sorted = (
        2.0 * np.pi * EARTH_RADIUS * np.cos(np.deg2rad(lat))[None, :] / GRAVITY
    ) * integral
    psi = np.empty_like(psi_sorted)
    psi[order] = psi_sorted
    return psi


def interpolate_to_pressure(
    field: np.ndarray,
    plev: np.ndarray,
    pressure: float,
) -> np.ndarray:
    """Linearly interpolate a field in pressure to a single level.

    An exact level is returned as is even when its neighbours are missing;
    otherwise the two bracketing levels must both be finite. No extrapolation:
    a pressure outside the supplied range gives NaN.

    Parameters
    ----------
    field:
        Array whose leading axis is pressure.
    plev:
        Pressure of each level (Pa), in any order.
    pressure:
        Target pressure (Pa).

    Returns
    -------
    :
        Field at ``pressure``, without the leading axis.

    """
    field = np.asarray(field, dtype=float)
    plev = np.asarray(plev, dtype=float)
    order = np.argsort(plev)
    p = plev[order]
    f = field[order]
    exact = np.flatnonzero(np.isclose(p, pressure, rtol=0.0, atol=1e-6))
    if exact.size:
        return f[exact[0]].copy()
    upper = np.searchsorted(p, pressure)
    if upper in (0, p.size):
        return np.full(f.shape[1:], np.nan)
    weight = (pressure - p[upper - 1]) / (p[upper] - p[upper - 1])
    return (1.0 - weight) * f[upper - 1] + weight * f[upper]


def hadley_cell_edge(
    psi500: np.ndarray,
    lat: np.ndarray,
    hemisphere: str = "north",
    search_latitudes: tuple[float, float] = (0.0, 45.0),
) -> float:
    """Hadley-cell edge from the 500 hPa mass streamfunction (TropD ``PSI_500``).

    The edge is the latitude at which the 500 hPa streamfunction crosses zero
    poleward of its subtropical extremum, found by linear interpolation
    between the two grid latitudes that bracket the crossing (Adam et al.,
    2018, GMD, 11, 4339; method ``Psi_500`` of Davis and Birner, 2017). The
    extremum is the maximum of ``psi`` in the northern and the minimum in the
    southern hemisphere, searched equatorward of ``search_latitudes[1]``
    degrees. NaN is returned if the hemisphere has no cell (the extremum has the
    wrong sign) or no crossing is found.

    Parameters
    ----------
    psi500:
        Streamfunction at 500 hPa, length ``n_lat``; NaN means missing.
    lat:
        Latitude (degrees), length ``n_lat``, in any order.
    hemisphere:
        ``"north"`` or ``"south"``.
    search_latitudes:
        Absolute latitude range (degrees) in which the extremum is searched.

    Returns
    -------
    :
        Edge latitude (degrees, negative in the southern hemisphere), NaN if
        undefined.

    """
    sign = _hemisphere_sign(hemisphere)
    psi = np.asarray(psi500, dtype=float)
    lat = np.asarray(lat, dtype=float)
    keep = sign * lat >= 0.0
    x = (sign * lat)[keep]  # absolute latitude, increasing poleward after sort
    y = (sign * psi)[keep]  # a positive hump for both hemispheres
    order = np.argsort(x)
    x, y = x[order], y[order]
    in_window = (x >= search_latitudes[0]) & (x <= search_latitudes[1]) & np.isfinite(y)
    if not in_window.any():
        return float("nan")
    candidates = np.flatnonzero(in_window)
    peak = candidates[np.argmax(y[candidates])]
    if y[peak] <= 0.0:
        return float("nan")
    for j in range(peak, x.size - 1):
        if not (np.isfinite(y[j]) and np.isfinite(y[j + 1])):
            return float("nan")  # the crossing would be in missing data
        if y[j] > 0.0 >= y[j + 1]:
            edge = x[j] + y[j] * (x[j + 1] - x[j]) / (y[j] - y[j + 1])
            return float(sign * edge)
    return float("nan")


def jet_latitude(
    u: np.ndarray,
    lat: np.ndarray,
    hemisphere: str = "north",
    lat_range: tuple[float, float] = (20.0, 70.0),
    half_width: int = 2,
) -> float:
    """Latitude of the eddy-driven jet: maximum of low-level zonal wind.

    The latitude of maximum zonal-mean zonal wind within ``lat_range``
    (absolute latitude), refined by a least-squares parabola through the
    ``2 * half_width + 1`` grid points around the maximum, whose vertex is the
    result (the ``EDJ`` metric of TropD, Adam et al., 2018, GMD; Barnes and
    Polvani, 2013, J. Clim.). The grid maximum is returned instead if the
    parabola is not concave, has its vertex outside the two neighbouring grid
    points, or fewer than three finite points are available.

    Parameters
    ----------
    u:
        Zonal-mean zonal wind (typically 850 hPa), length ``n_lat``; NaN means
        missing.
    lat:
        Latitude (degrees), length ``n_lat``, in any order.
    hemisphere:
        ``"north"`` or ``"south"``.
    lat_range:
        Absolute latitude range (degrees) searched for the maximum.
    half_width:
        Points on each side of the maximum used in the fit.

    Returns
    -------
    :
        Jet latitude (degrees, negative in the southern hemisphere), NaN if no
        finite wind lies within the range.

    """
    sign = _hemisphere_sign(hemisphere)
    u = np.asarray(u, dtype=float)
    lat = np.asarray(lat, dtype=float)
    order = np.argsort(lat)
    lat, u = lat[order], u[order]
    absolute = sign * lat
    in_range = (absolute >= lat_range[0]) & (absolute <= lat_range[1]) & np.isfinite(u)
    if not in_range.any():
        return float("nan")
    candidates = np.flatnonzero(in_range)
    peak = int(candidates[np.argmax(u[candidates])])
    grid_max = float(lat[peak])
    lo, hi = max(peak - half_width, 0), min(peak + half_width + 1, lat.size)
    fit = np.arange(lo, hi)
    fit = fit[np.isfinite(u[fit])]
    if fit.size < _MIN_FIT_POINTS:
        return grid_max
    centre = lat[peak]
    a, b, _c = np.polyfit(lat[fit] - centre, u[fit], 2)
    if a >= 0.0:
        return grid_max
    vertex = -b / (2.0 * a) + centre
    neighbours = lat[[max(peak - 1, 0), min(peak + 1, lat.size - 1)]]
    if not neighbours.min() <= vertex <= neighbours.max():
        return grid_max
    return float(vertex)


def box_mean(
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
) -> np.ndarray:
    """Area-weighted mean over a latitude-longitude box (NaN-aware).

    Grid cells whose centre lies in the box are averaged with weights
    proportional to cos(latitude); NaN cells are ignored. Longitudes are
    compared modulo 360 degrees, so a box may straddle the date line
    (``lon_range = (200, 280)`` and ``(-160, -80)`` are the same box); a range
    spanning 360 degrees or more selects all longitudes.

    Parameters
    ----------
    field:
        Array whose last two axes are latitude and longitude.
    lat:
        Latitude (degrees).
    lon:
        Longitude (degrees).
    lat_range:
        South and north edge of the box (degrees).
    lon_range:
        West and east edge of the box (degrees east), west edge first.

    Returns
    -------
    :
        Box mean for each leading index; NaN where the box has no finite cell.

    Raises
    ------
    ValueError
        If no grid cell centre lies in the box.

    """
    field = np.asarray(field, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    west, east = lon_range[0] % 360.0, lon_range[1] % 360.0
    lon_mod = lon % 360.0
    if lon_range[1] - lon_range[0] >= _FULL_CIRCLE:
        in_lon = np.ones(lon.shape, dtype=bool)
    elif west <= east:
        in_lon = (lon_mod >= west) & (lon_mod <= east)
    else:
        in_lon = (lon_mod >= west) | (lon_mod <= east)
    in_lat = (lat >= lat_range[0]) & (lat <= lat_range[1])
    if not (in_lat.any() and in_lon.any()):
        msg = f"no grid cell centre in box lat={lat_range}, lon={lon_range}"
        raise ValueError(msg)
    sub = field[..., in_lat, :][..., :, in_lon]
    weights = np.broadcast_to(np.cos(np.deg2rad(lat[in_lat]))[:, None], sub.shape[-2:])
    weights = np.broadcast_to(weights, sub.shape)
    valid = np.isfinite(sub)
    total = np.where(valid, weights, 0.0).sum(axis=(-2, -1))
    weighted = np.where(valid, sub * weights, 0.0).sum(axis=(-2, -1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0.0, weighted / total, np.nan)


def walker_circulation_strength(
    psl: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    east_box: tuple[tuple[float, float], tuple[float, float]] = (
        (-5.0, 5.0),
        (200.0, 280.0),
    ),
    west_box: tuple[tuple[float, float], tuple[float, float]] = (
        (-5.0, 5.0),
        (80.0, 160.0),
    ),
) -> np.ndarray:
    """Walker-circulation strength as a sea-level-pressure difference.

    ``Delta p = mean(p_central-east Pacific) - mean(p_Indian Ocean-west
    Pacific)`` of the area-mean sea-level pressure in two boxes. The default
    boxes, 5S-5N, 160W-80W and 5S-5N, 80E-160E, are those of Vecchi et al.
    (2006, Nature, 441, 73); a stronger Walker circulation has a larger
    difference. The result has the unit of the input.

    Parameters
    ----------
    psl:
        Sea-level pressure, shape ``(..., n_lat, n_lon)``.
    lat:
        Latitude (degrees).
    lon:
        Longitude (degrees east).
    east_box, west_box:
        ``((south, north), (west, east))`` of the eastern and western box.

    Returns
    -------
    :
        Pressure difference for each leading index.

    """
    return box_mean(psl, lat, lon, *east_box) - box_mean(psl, lat, lon, *west_box)


# ---------------------------------------------------------------------------
# MSU/AMSU brightness temperature and lower-tropospheric stability
# ---------------------------------------------------------------------------

GAS_CONSTANT_WATER_VAPOR = 461.5
"""Gas constant of water vapour, R_v (J kg-1 K-1)."""

REFERENCE_PRESSURE = 1.0e5
"""Reference pressure of the potential temperature (Pa)."""

KAPPA_DRY_AIR = GAS_CONSTANT_DRY_AIR / SPECIFIC_HEAT_DRY_AIR
"""Exponent R_d / c_p of the potential temperature."""

MSU_CHANNELS = ("tlt", "tmt", "tts", "tls")
"""Channels with a vendored RSS static weighting function."""

_MSU_FILES = {
    "tlt": {"land": "tlt_land", "ocean": "tlt_ocean"},
    "tmt": {"land": "tmt_land", "ocean": "tmt_ocean"},
    "tts": {"land": "tts", "ocean": "tts"},
    "tls": {"land": "tls", "ocean": "tls"},
}
_MSU_TABLE_STEM = "std_atmosphere_wt_function_chan_"


@functools.cache
def load_msu_weighting_function(
    channel: str,
    surface: str = "ocean",
) -> dict[str, np.ndarray | float]:
    """Load a vendored RSS static MSU/AMSU temperature weighting function.

    The tables are the published instrument characteristics of Remote Sensing
    Systems (https://data.remss.com/msu/weighting_functions/, files
    ``std_atmosphere_wt_function_chan_<channel>.txt``): the weight, per km, of
    the temperature of each level of the US standard atmosphere to the
    brightness temperature of channel ``tlt``/``tmt``/``tts``/``tls``, and the
    weight of the surface (a land emissivity of 0.90, a wind-roughened sea
    surface for ocean; the two coincide for ``tts`` and ``tls``). They are
    shipped in ``climateeval/diags/data`` with their source in the header.

    Parameters
    ----------
    channel:
        One of :data:`MSU_CHANNELS`.
    surface:
        ``"land"`` or ``"ocean"``.

    Returns
    -------
    :
        Dictionary with the arrays ``height`` (m), ``temperature`` (K),
        ``pressure`` (Pa) and ``weight`` (km-1) of the standard-atmosphere
        levels, and the float ``surface_weight``.

    Raises
    ------
    ValueError
        For an unknown channel or surface.

    """
    channel = channel.lower()
    if channel not in _MSU_FILES or surface not in ("land", "ocean"):
        msg = (
            f"channel must be one of {MSU_CHANNELS} and surface 'land' or "
            f"'ocean', got '{channel}' and '{surface}'"
        )
        raise ValueError(msg)
    name = f"{_MSU_TABLE_STEM}{_MSU_FILES[channel][surface]}.txt"
    text = (resources.files("climateeval.diags") / "data" / name).read_text(
        encoding="ascii",
    )
    surface_weight = float("nan")
    rows = []
    for line in text.splitlines():
        if line.startswith("Surface Weight"):
            surface_weight = float(line.split()[-1])
        elif line.split()[:1] and line.split()[0].isdigit():
            rows.append([float(x) for x in line.split()])
    table = np.array(rows)
    return {
        "height": table[:, 1],
        "temperature": table[:, 2],
        "pressure": table[:, 3],
        "weight": table[:, 5],
        "surface_weight": surface_weight,
    }


@functools.cache
def _msu_cumulative_weight(
    channel: str,
    surface: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Get the atmospheric weight above a pressure, on the table's levels.

    Returns ``ln(p)`` (increasing), the integral of the weighting function
    from the top of the atmosphere down to that pressure, and the total.
    """
    table = load_msu_weighting_function(channel, surface)
    height_km = np.asarray(table["height"]) / 1000.0
    weight = np.asarray(table["weight"])
    pressure = np.asarray(table["pressure"])
    layer = 0.5 * (weight[1:] + weight[:-1]) * np.diff(height_km)  # RSS read-me
    total = float(layer.sum())
    above_surface_up_to = np.concatenate([[0.0], np.cumsum(layer)])
    from_top = total - above_surface_up_to  # at each level: weight above it
    return np.log(pressure[::-1]), from_top[::-1], total


def _msu_weight_above(pressure: np.ndarray, channel: str, surface: str) -> np.ndarray:
    """Get the atmospheric MSU weight above each pressure (Pa)."""
    ln_p, from_top, total = _msu_cumulative_weight(channel, surface)
    with np.errstate(divide="ignore"):
        ln_pressure = np.log(pressure)
    # Above the table's top level: no weight left; below its surface: all of it
    return np.interp(ln_pressure, ln_p, from_top, left=0.0, right=total)


def msu_layer_weights(
    plev: np.ndarray,
    ps: np.ndarray,
    channel: str,
    surface: str = "ocean",
) -> tuple[np.ndarray, float]:
    """Weights of pressure levels in an MSU/AMSU channel, truncated at ps.

    Each level represents the layer between the mid-points (in pressure) to
    its neighbours; the top level extends to ``p = 0`` and the lowest level at
    or above the surface extends down to the surface pressure. The weight of a
    layer is the integral of the channel's static weighting function (see
    :func:`load_msu_weighting_function`, a function of height on the standard
    atmosphere, mapped to pressure through the standard atmosphere) between
    the layer boundaries. Layers below the surface have no weight; the
    surface itself has the tabulated surface weight. The weights are *not*
    renormalised here (see :func:`msu_weighted_temperature`).

    This is the static-weighting-function approach used for CMIP analyses
    (Santer et al., 2005, JGR, 110, D17103; Po-Chedley et al., 2021, J. Clim.).

    Parameters
    ----------
    plev:
        Pressure of the levels (Pa), in any order, length ``n_lev``.
    ps:
        Surface pressure (Pa), any shape ``S``.
    channel:
        One of :data:`MSU_CHANNELS`.
    surface:
        ``"land"`` or ``"ocean"`` (surface weight and, for the low
        channels, the weighting function).

    Returns
    -------
    :
        Weights of shape ``(n_lev, *S)`` in the order of ``plev``, and the
        surface weight.

    """
    plev = np.asarray(plev, dtype=float)
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(plev)
    p = plev[order]
    n = p.size
    shape = (n,) + (1,) * ps.ndim
    mid = 0.5 * (p[1:] + p[:-1])
    top = np.concatenate([[0.0], mid]).reshape(shape)
    bottom = np.concatenate([mid, [np.inf]]).reshape(shape)
    next_level = np.concatenate([p[1:], [np.inf]]).reshape(shape)
    level = p.reshape(shape)
    ps_b = ps[None]
    # The lowest level above the surface extends down to the surface
    bottom = np.where(next_level <= ps_b, bottom, ps_b)
    top = np.minimum(top, ps_b)
    bottom = np.minimum(bottom, ps_b)
    weights = _msu_weight_above(bottom, channel, surface) - _msu_weight_above(
        top,
        channel,
        surface,
    )
    weights = np.where(level <= ps_b, weights, 0.0)
    out = np.empty_like(weights)
    out[order] = weights
    surface_weight = float(
        load_msu_weighting_function(channel, surface)["surface_weight"],
    )
    return out, surface_weight


def msu_weighted_temperature(  # noqa: PLR0913
    ta: np.ndarray,
    plev: np.ndarray,
    ts: np.ndarray,
    ps: np.ndarray,
    channel: str,
    land_fraction: np.ndarray | None = None,
) -> np.ndarray:
    """MSU/AMSU brightness temperature from model temperature profiles.

    ``T_b = (W_s T_s + sum_k W_k T_k) / (W_s + sum_k W_k)``: the static
    weighting-function approximation with weights truncated at the surface
    pressure and renormalised to sum to one, so that an isothermal atmosphere
    returns its temperature for every channel. Levels with NaN temperature
    (below the surface in model output on pressure levels, where they are
    masked) carry no weight; convert masked arrays with
    ``np.ma.filled(x, np.nan)`` first. For land and ocean the surface weight
    (and for TLT/TMT the weighting function) differ; ``land_fraction`` blends
    the two brightness temperatures, and None means all ocean.

    Parameters
    ----------
    ta:
        Air temperature (K), shape ``(n_lev, *S)``.
    plev:
        Pressure of the levels (Pa), length ``n_lev``.
    ts:
        Surface temperature (K), shape ``S``.
    ps:
        Surface pressure (Pa), shape ``S``.
    channel:
        One of :data:`MSU_CHANNELS`: lower troposphere (``"tlt"``), mid
        troposphere (``"tmt"``), troposphere/stratosphere (``"tts"``) or lower
        stratosphere (``"tls"``).
    land_fraction:
        Land fraction (0-1), shape ``S`` or broadcastable; None for ocean.

    Returns
    -------
    :
        Brightness temperature (K), shape ``S``; NaN where ``ts`` or ``ps``
        is NaN, or (for ``tls``) where no level contributes.

    Raises
    ------
    ValueError
        If the arrays do not match.

    """
    ta = np.asarray(ta, dtype=float)
    plev = np.asarray(plev, dtype=float)
    ts = np.asarray(ts, dtype=float)
    ps = np.asarray(ps, dtype=float)
    if ta.shape[0] != plev.size or ta.shape[1:] != ts.shape or ts.shape != ps.shape:
        msg = (
            f"expected ta of shape (n_lev, *S), ts and ps of shape S; got "
            f"ta {ta.shape}, plev {plev.shape}, ts {ts.shape}, ps {ps.shape}"
        )
        raise ValueError(msg)
    if land_fraction is None:
        blend = (("ocean", np.ones(ts.shape)),)
    else:
        land = np.broadcast_to(np.asarray(land_fraction, dtype=float), ts.shape)
        blend = (("land", land), ("ocean", 1.0 - land))
    valid = np.isfinite(ta)
    ta_filled = np.where(valid, ta, 0.0)
    result = np.zeros(ts.shape)
    for surface, fraction in blend:
        weights, surface_weight = msu_layer_weights(plev, ps, channel, surface)
        weights = np.where(valid, weights, 0.0)
        total = weights.sum(axis=0) + surface_weight
        numerator = (weights * ta_filled).sum(axis=0)
        if surface_weight > 0.0:
            numerator = numerator + surface_weight * ts
        with np.errstate(invalid="ignore", divide="ignore"):
            brightness = np.where(total > 0.0, numerator / total, np.nan)
        result = result + np.where(fraction > 0.0, fraction * brightness, 0.0)
    return np.where(np.isfinite(ts) & np.isfinite(ps), result, np.nan)


def potential_temperature(
    temperature: np.ndarray,
    pressure: np.ndarray,
    reference_pressure: float = REFERENCE_PRESSURE,
) -> np.ndarray:
    """Potential temperature ``theta = T (p_0 / p)^(R_d / c_p)``.

    Parameters
    ----------
    temperature:
        Temperature (K).
    pressure:
        Pressure (Pa).
    reference_pressure:
        Reference pressure p_0 (Pa); 1000 hPa by default.

    Returns
    -------
    :
        Potential temperature (K).

    """
    return (
        np.asarray(temperature, dtype=float)
        * (reference_pressure / np.asarray(pressure, dtype=float)) ** KAPPA_DRY_AIR
    )


def saturation_vapor_pressure(temperature: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure over liquid water (Bolton, 1980, MWR).

    ``e_s = 611.2 exp(17.67 (T - 273.15) / (T - 29.65))`` Pa.

    Parameters
    ----------
    temperature:
        Temperature (K).

    Returns
    -------
    :
        Saturation vapour pressure (Pa).

    """
    t = np.asarray(temperature, dtype=float)
    return 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))


def moist_adiabatic_theta_gradient(
    temperature: np.ndarray,
    pressure: np.ndarray,
) -> np.ndarray:
    """Moist-adiabatic potential-temperature gradient (K m-1).

    ``Gamma_m = (g / c_p) [1 - (1 + L_v q_s / (R_d T)) / (1 + L_v^2 q_s
    epsilon / (c_p R_d T^2))]`` with the saturation mixing ratio ``q_s = epsilon
    e_s / (p - e_s)``: the rate at which the potential temperature increases
    with height along a saturated adiabat, i.e. ``g / c_p`` minus the moist
    adiabatic lapse rate (Wood and Bretherton, 2006, J. Clim., 19, 6425, eq. 5;
    the factor ``theta / T`` between the two is not applied, as there).

    Parameters
    ----------
    temperature:
        Temperature (K).
    pressure:
        Pressure (Pa).

    Returns
    -------
    :
        Gradient (K m-1), about 5e-3 at 850 hPa and 280 K.

    """
    t = np.asarray(temperature, dtype=float)
    p = np.asarray(pressure, dtype=float)
    epsilon = GAS_CONSTANT_DRY_AIR / GAS_CONSTANT_WATER_VAPOR
    e_s = saturation_vapor_pressure(t)
    q_s = epsilon * e_s / (p - e_s)
    numerator = 1.0 + LATENT_HEAT_VAPORIZATION * q_s / (GAS_CONSTANT_DRY_AIR * t)
    denominator = 1.0 + (
        LATENT_HEAT_VAPORIZATION**2
        * q_s
        * epsilon
        / (SPECIFIC_HEAT_DRY_AIR * GAS_CONSTANT_DRY_AIR * t**2)
    )
    return (GRAVITY / SPECIFIC_HEAT_DRY_AIR) * (1.0 - numerator / denominator)


def lower_tropospheric_stability(
    t700: np.ndarray,
    tsfc: np.ndarray,
    ps: np.ndarray,
) -> np.ndarray:
    """Lower-tropospheric stability ``LTS = theta_700 - theta_sfc`` (K).

    Potential temperatures are referenced to 1000 hPa; ``theta_sfc`` uses the
    surface pressure, so that it equals the surface temperature at 1000 hPa
    (Klein and Hartmann, 1993, J. Clim., 6, 1587).

    Parameters
    ----------
    t700:
        Temperature at 700 hPa (K).
    tsfc:
        Surface (or near-surface air) temperature (K).
    ps:
        Surface pressure (Pa).

    Returns
    -------
    :
        LTS (K).

    """
    return potential_temperature(t700, 70000.0) - potential_temperature(tsfc, ps)


def estimated_inversion_strength(
    t700: np.ndarray,
    tsfc: np.ndarray,
    ps: np.ndarray,
    rh_sfc: np.ndarray | float | None = None,
) -> np.ndarray:
    """Estimated inversion strength after Wood and Bretherton (2006).

    ``EIS = LTS - Gamma_m^850 (z_700 - z_LCL)`` (their eq. 4), where

    - ``Gamma_m^850`` is the moist-adiabatic potential-temperature gradient
      (:func:`moist_adiabatic_theta_gradient`) at 850 hPa and at the
      temperature ``T_850 = (T_sfc + T_700) / 2``;
    - ``z_700 = (R_d T_850 / g) ln(p_s / 700 hPa)`` is the hydrostatic height of
      the 700 hPa level above the surface, with the same mean temperature;
    - ``z_LCL = 125 (T - T_d)`` m is the lifting condensation level of surface
      air (Lawrence, 2005, BAMS, 86, 225), with ``T - T_d = (100 - RH) / 5``.

    Parameters
    ----------
    t700:
        Temperature at 700 hPa (K).
    tsfc:
        Surface (or near-surface air) temperature (K).
    ps:
        Surface pressure (Pa).
    rh_sfc:
        Near-surface relative humidity (%), default 80 % where not supplied.

    Returns
    -------
    :
        EIS (K).

    """
    t700 = np.asarray(t700, dtype=float)
    tsfc = np.asarray(tsfc, dtype=float)
    ps = np.asarray(ps, dtype=float)
    rh = 80.0 if rh_sfc is None else np.asarray(rh_sfc, dtype=float)
    t850 = 0.5 * (tsfc + t700)
    z700 = GAS_CONSTANT_DRY_AIR * t850 / GRAVITY * np.log(ps / 70000.0)
    z_lcl = 125.0 * (100.0 - rh) / 5.0
    gamma = moist_adiabatic_theta_gradient(t850, 85000.0)
    return lower_tropospheric_stability(t700, tsfc, ps) - gamma * (z700 - z_lcl)
