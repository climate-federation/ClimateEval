"""Provide large-scale dynamics diagnostics of daily model output.

Two properties of a dynamical atmosphere that only daily fields resolve:
the midlatitude flow has to be close to geostrophic balance with the
model's own geopotential field, and near-equatorial convection has to
propagate predominantly eastward on intraseasonal timescales (the MJO).

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from esmvalcore.preprocessor import extract_levels, extract_region
from loguru import logger

from climateeval import Variable
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval.diags.complex._scalar import (
    ScalarExperimentDiagnostic,
    as_float_array,
)
from climateeval.diags.complex._transport import EARTH_RADIUS_M

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval.data import ComplexDataSource

logger = logger.opt(colors=True)

EARTH_ANGULAR_VELOCITY = 7.292e-5
"""Angular velocity of the Earth's rotation, Omega (rad s-1)."""

_GRAVITY = 9.80665
"""Standard gravitational acceleration (m s-2)."""

_MIN_CORIOLIS_LATITUDE = 10.0
"""Latitude (degrees) below which the Coriolis parameter is too small."""

_MIN_CORRELATION_POINTS = 3
"""Fewest valid pairs a pattern correlation is computed from."""


def geostrophic_wind_u(
    geopotential_height: np.ndarray,
    latitudes: np.ndarray,
) -> np.ndarray:
    """Zonal geostrophic wind from a geopotential height field.

    ``u_g = -(g / f) dZ/dy`` with ``f = 2 Omega sin(lat)``, the meridional
    derivative taken on the sphere. Rows within
    ``_MIN_CORIOLIS_LATITUDE`` of the equator are returned as NaN, where the
    geostrophic approximation breaks down.

    Parameters
    ----------
    geopotential_height:
        Geopotential height of shape ``(..., lat, lon)`` in m.
    latitudes:
        Latitudes (degrees).

    Returns
    -------
    :
        Zonal geostrophic wind (m s-1) of the same shape as the input.

    """
    geopotential_height = np.asarray(geopotential_height, dtype=float)
    latitudes = np.asarray(latitudes, dtype=float)
    coriolis = 2.0 * EARTH_ANGULAR_VELOCITY * np.sin(np.deg2rad(latitudes))
    d_height_d_y = np.gradient(
        geopotential_height,
        np.deg2rad(latitudes) * EARTH_RADIUS_M,
        axis=-2,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        wind = -(_GRAVITY / coriolis)[..., :, None] * d_height_d_y
    wind[..., np.abs(latitudes) < _MIN_CORIOLIS_LATITUDE, :] = np.nan
    return wind


def banded_pattern_correlation(
    field_a: np.ndarray,
    field_b: np.ndarray,
    latitudes: np.ndarray,
    *,
    band: tuple[float, float] = (-90.0, 90.0),
    absolute_latitude: bool = False,
) -> float:
    """Centred, cosine-weighted correlation of two fields over a latitude band.

    The fields are pooled over every leading dimension (time, if any) and
    over the band. Non-finite points are dropped *pairwise*, so a field that
    is masked somewhere (an SST product over land, a sub-surface pressure
    level) masks both sides of the comparison consistently. "Centred" means
    each field has its own weighted band mean removed before the covariance.

    Parameters
    ----------
    field_a:
        First field of shape ``(..., lat, lon)``.
    field_b:
        Second field of the same shape.
    latitudes:
        Latitudes (degrees).
    band:
        Latitude interval to correlate over.
    absolute_latitude:
        Apply the interval to ``|lat|``, i.e. to the same band in both
        hemispheres.

    Returns
    -------
    :
        Correlation coefficient, NaN if fewer than three valid pairs remain.

    """
    field_a = np.asarray(field_a, dtype=float)
    field_b = np.asarray(field_b, dtype=float)
    latitudes = np.asarray(latitudes, dtype=float)
    latitude_key = np.abs(latitudes) if absolute_latitude else latitudes
    in_band = (latitude_key >= band[0]) & (latitude_key <= band[1])
    banded_a = field_a[..., in_band, :]
    banded_b = field_b[..., in_band, :]
    weights = np.broadcast_to(
        np.broadcast_to(
            np.cos(np.deg2rad(latitudes[in_band]))[:, None],
            banded_a.shape[-2:],
        ),
        banded_a.shape,
    ).ravel()
    values_a = banded_a.ravel()
    values_b = banded_b.ravel()
    valid = np.isfinite(values_a) & np.isfinite(values_b)
    values_a, values_b, weights = (
        values_a[valid],
        values_b[valid],
        weights[valid],
    )
    if values_a.size < _MIN_CORRELATION_POINTS:
        return float("nan")
    mean_a = np.average(values_a, weights=weights)
    mean_b = np.average(values_b, weights=weights)
    covariance = np.average(
        (values_a - mean_a) * (values_b - mean_b),
        weights=weights,
    )
    variance_a = np.average((values_a - mean_a) ** 2, weights=weights)
    variance_b = np.average((values_b - mean_b) ** 2, weights=weights)
    return float(covariance / np.sqrt(variance_a * variance_b))


def east_west_power_ratio(
    equatorial_field: np.ndarray,
    *,
    wavenumbers: tuple[int, int] = (1, 3),
    period_days: tuple[float, float] = (30.0, 90.0),
) -> float:
    """Eastward-to-westward power ratio of an equatorial Hovmoeller field.

    The Wheeler & Kiladis (1999) wavenumber-frequency decomposition of a
    daily, meridionally averaged field, restricted to the MJO band. The
    field is centred in time and Hann-tapered before the two-dimensional
    FFT. With numpy's ``exp(-i (omega t + k x))`` convention, eastward
    propagation has ``omega k < 0``.

    Parameters
    ----------
    equatorial_field:
        Daily, meridionally averaged field of shape ``(n_time, n_lon)``.
    wavenumbers:
        Inclusive range of zonal wavenumbers.
    period_days:
        Inclusive range of periods (days).

    Returns
    -------
    :
        Eastward power divided by westward power; infinity if there is no
        westward power at all.

    """
    field = np.asarray(equatorial_field, dtype=float)
    n_time, n_lon = field.shape
    field = field - field.mean(axis=0, keepdims=True)
    field = field * np.hanning(n_time)[:, None]
    power = np.abs(np.fft.fft2(field)) ** 2
    frequencies = np.fft.fftfreq(n_time, d=1.0)  # cycles per day
    zonal_wavenumbers = np.fft.fftfreq(n_lon, d=1.0 / n_lon)

    frequency_grid, wavenumber_grid = np.meshgrid(
        frequencies,
        zonal_wavenumbers,
        indexing="ij",
    )
    in_band = (
        (np.abs(frequency_grid) >= 1.0 / period_days[1])
        & (np.abs(frequency_grid) <= 1.0 / period_days[0])
        & (np.abs(wavenumber_grid) >= wavenumbers[0])
        & (np.abs(wavenumber_grid) <= wavenumbers[1])
    )
    eastward = frequency_grid * wavenumber_grid < 0
    east = float(power[in_band & eastward].sum())
    west = float(power[in_band & ~eastward].sum())
    return east / west if west > 0 else float("inf")


class GeostrophicBalance(ScalarExperimentDiagnostic):
    """Manage midlatitude geostrophic balance diagnostics.

    The daily midlatitude flow of any model with a dynamical atmosphere has
    to be close to geostrophic balance with its own geopotential field. The
    diagnostic correlates the daily zonal wind at ``pressure_level`` with the
    geostrophic wind ``u_g = -(g / f) dZ/dy`` diagnosed from the geopotential
    height at the same level, pooling all days and both midlatitude bands;
    the correlation of a balanced atmosphere is well above 0.9.

    The 850 hPa surface intersects orography over much of the midlatitude
    band (the Rockies, the Andes, the Tibetan Plateau, Antarctica), where
    pressure-level values are extrapolated below ground. Grid points whose
    daily surface pressure is below ``surface_pressure_threshold`` are
    therefore masked out of both winds, whenever ``ps`` is part of the daily
    data; without it the correlation is computed unmasked and a warning is
    logged.

    Additional keyword arguments
    ----------------------------
    pressure_level: float (default: 85000.0)
        Pressure level the balance is evaluated at (Pa).
    latitude_band: tuple[float, float] (default: (30.0, 60.0))
        Absolute-latitude band the correlation is pooled over (degrees); it
        is applied in both hemispheres.
    surface_pressure_threshold: float (default: 87000.0)
        Grid points with a lower daily surface pressure are masked as
        sub-surface (Pa).

    """

    _required_data_keys = ("day",)

    _default_pressure_level: ClassVar[float] = 85000.0
    _default_latitude_band: ClassVar[tuple[float, float]] = (30.0, 60.0)
    _default_surface_pressure_threshold: ClassVar[float] = 87000.0

    _ua: ClassVar[Variable] = Variable("ua", "ua", "day")
    _zg: ClassVar[Variable] = Variable("zg", "zg", "day")
    _ps: ClassVar[Variable] = Variable("ps", "ps", "day")
    _input_variables: ClassVar[tuple[Variable, ...]] = (_ua, _zg)

    _correlation: ClassVar[Variable] = Variable(
        "geostrophic_correlation",
        "geostrophic_correlation",
        "fx",
    )
    _output_variables: ClassVar[tuple[Variable, ...]] = (_correlation,)

    # No CMIP6 comparison ensemble: daily 3-D winds and geopotential heights
    # of the whole r1i1p1f1 ensemble are far too large to pull in.

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize class instance."""
        super().__init__(name, **kwargs)
        self._pressure_level: float = float(
            self._kwargs.get("pressure_level", self._default_pressure_level),
        )
        self._latitude_band: tuple[float, float] = tuple(
            self._kwargs.get("latitude_band", self._default_latitude_band),
        )
        self._surface_pressure_threshold: float = float(
            self._kwargs.get(
                "surface_pressure_threshold",
                self._default_surface_pressure_threshold,
            ),
        )

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        data = complex_data_source.data["day"]
        wind_cube = self._level_cube(data, self._ua)
        height_cube = self._level_cube(data, self._zg)
        latitudes = wind_cube.coord("latitude").points.astype(float)
        wind = np.squeeze(as_float_array(wind_cube.data))
        height = np.squeeze(as_float_array(height_cube.data))
        wind, height = self._mask_below_surface(data, wind, height)
        correlation = banded_pattern_correlation(
            wind,
            geostrophic_wind_u(height, latitudes),
            latitudes,
            band=self._latitude_band,
            absolute_latitude=True,
        )
        return self._scalar_cubes({self._correlation: correlation})

    def _level_cube(self, data: Any, variable: Variable) -> Cube:
        """Get one daily variable at the evaluation pressure level."""
        cube = self._prepared_cube(data, variable)
        with setup_esmvaltool_config_and_logging():
            cube = extract_levels(cube, [self._pressure_level], "linear")
        return cube  # noqa: RET504 - named for readability

    def _mask_below_surface(
        self,
        data: Any,
        wind: np.ndarray,
        height: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mask grid points whose surface pressure puts them below ground."""
        try:
            surface_pressure = np.squeeze(
                as_float_array(self._prepared_cube(data, self._ps).data),
            )
        except Exception as exc:  # noqa: BLE001 - ps is genuinely optional
            logger.warning(
                f"Diagnostic '{self.name}': no daily surface pressure "
                f"({type(exc).__name__}: {exc}); the geostrophic balance is "
                f"evaluated without masking sub-surface grid points",
            )
            return wind, height
        n_time = min(wind.shape[0], surface_pressure.shape[0])
        below_surface = surface_pressure[:n_time] < self._surface_pressure_threshold
        wind, height = wind[:n_time].copy(), height[:n_time].copy()
        wind[below_surface] = np.nan
        height[below_surface] = np.nan
        return wind, height


class MJOEastWestPowerRatio(ScalarExperimentDiagnostic):
    """Manage MJO wavenumber-frequency diagnostics.

    The Madden-Julian Oscillation dominates tropical intraseasonal
    variability and is strongly eastward-propagating. Following the
    Wheeler & Kiladis (1999) decomposition, the diagnostic takes the
    meridional mean of daily precipitation over ``latitude_band``, computes
    its wavenumber-frequency spectrum and returns the ratio of eastward to
    westward power in the MJO band (zonal wavenumbers 1-3, periods
    30-90 days). Observations give a ratio well above one; a model with no
    coherent eastward propagation gives about one.

    No red-noise background is removed and the spectrum is not split into
    symmetric and antisymmetric parts, so the ratio is that of the raw
    spectrum.

    Additional keyword arguments
    ----------------------------
    latitude_band: tuple[float, float] (default: (-15.0, 15.0))
        Near-equatorial band averaged over before the transform (degrees).
    wavenumbers: tuple[int, int] (default: (1, 3))
        Inclusive range of zonal wavenumbers.
    period_days: tuple[float, float] (default: (30.0, 90.0))
        Inclusive range of periods (days).

    """

    _required_data_keys = ("day",)

    _default_latitude_band: ClassVar[tuple[float, float]] = (-15.0, 15.0)
    _default_wavenumbers: ClassVar[tuple[int, int]] = (1, 3)
    _default_period_days: ClassVar[tuple[float, float]] = (30.0, 90.0)

    _pr: ClassVar[Variable] = Variable("pr", "pr", "day")
    _input_variables: ClassVar[tuple[Variable, ...]] = (_pr,)

    _ratio: ClassVar[Variable] = Variable(
        "mjo_east_west_ratio",
        "mjo_east_west_ratio",
        "fx",
    )
    _output_variables: ClassVar[tuple[Variable, ...]] = (_ratio,)

    # No CMIP6 comparison ensemble: decades of daily precipitation of the
    # whole r1i1p1f1 ensemble are too large to pull in.

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize class instance."""
        super().__init__(name, **kwargs)
        self._latitude_band: tuple[float, float] = tuple(
            self._kwargs.get("latitude_band", self._default_latitude_band),
        )
        self._wavenumbers: tuple[int, int] = tuple(
            self._kwargs.get("wavenumbers", self._default_wavenumbers),
        )
        self._period_days: tuple[float, float] = tuple(
            self._kwargs.get("period_days", self._default_period_days),
        )

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        cube = self._prepared_cube(complex_data_source.data["day"], self._pr)
        with setup_esmvaltool_config_and_logging():
            cube = extract_region(
                cube,
                start_longitude=0.0,
                end_longitude=360.0,
                start_latitude=self._latitude_band[0],
                end_latitude=self._latitude_band[1],
            )
        equatorial = np.nanmean(
            as_float_array(cube.data),
            axis=cube.coord_dims("latitude")[0],
        )
        ratio = east_west_power_ratio(
            equatorial,
            wavenumbers=self._wavenumbers,
            period_days=self._period_days,
        )
        return self._scalar_cubes({self._ratio: ratio})
