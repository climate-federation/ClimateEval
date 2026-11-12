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
    (``lon_range = (200, 280)`` and ``(-160, -80)`` are the same box).

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
    if west <= east:
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
