"""Provide coupled variability and tropical energetics diagnostics.

Three properties of the coupled system that follow from its energy
transport and its dominant mode of interannual variability: the ITCZ
migrates with the cross-equatorial atmospheric energy transport, the
atmospheric and oceanic heat transports compensate each other on decadal
timescales, and ENSO imprints a characteristic pattern on tropical surface
temperature and precipitation.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from esmvalcore.preprocessor import (
    anomalies,
    area_statistics,
    climate_statistics,
    extract_region,
    regrid,
    zonal_statistics,
)
from loguru import logger
from scipy import stats

from climateeval import Variable
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval.data import (
    GPCP,
    CMIP6HistoricalR1I1P1F1,
    CMIP6PiControlR1I1P1F1,
    HadISST,
)
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.complex._dynamics import banded_pattern_correlation
from climateeval.diags.complex._scalar import (
    ScalarExperimentDiagnostic,
    as_float_array,
    calendar_years_and_months,
)
from climateeval.diags.complex._transport import implied_meridional_transport

if TYPE_CHECKING:
    from iris.cube import Cube, CubeList
    from xarray import Dataset

    from climateeval.data import ComplexDataSource, DataSource

logger = logger.opt(colors=True)

MONTHS_PER_YEAR = 12
"""Number of calendar months, i.e. the length of a monthly climatology."""

_WATTS_PER_PETAWATT = 1.0e15

_TOA_FLUXES: tuple[tuple[str, float], ...] = (
    ("rsdt", +1.0),
    ("rsut", -1.0),
    ("rlut", -1.0),
)
"""Contributions to the TOA net downward flux."""

_SURFACE_FLUXES: tuple[tuple[str, float], ...] = (
    ("rsds", +1.0),
    ("rsus", -1.0),
    ("rlds", +1.0),
    ("rlus", -1.0),
    ("hfss", -1.0),
    ("hfls", -1.0),
)
"""Contributions to the surface net downward flux."""

_FLUX_VARIABLES: tuple[Variable, ...] = tuple(
    Variable(name, name, "mon") for name, _sign in (*_TOA_FLUXES, *_SURFACE_FLUXES)
)


def zero_crossing_nearest_equator(
    profile: np.ndarray,
    latitudes: np.ndarray,
    band: float = 40.0,
) -> float:
    """Latitude of a profile's zero crossing nearest the equator.

    Linearly interpolated between the two bracketing latitudes. Applied to a
    zonal-mean atmospheric energy transport this is the energy flux equator.

    Parameters
    ----------
    profile:
        Profile of shape ``(lat,)``.
    latitudes:
        Latitudes (degrees), ascending or descending.
    band:
        Only crossings within this absolute latitude are considered.

    Returns
    -------
    :
        Latitude of the crossing (degrees), NaN if there is none.

    """
    profile = np.asarray(profile, dtype=float)
    latitudes = np.asarray(latitudes, dtype=float)
    order = np.argsort(latitudes)
    sorted_latitudes, sorted_profile = latitudes[order], profile[order]
    in_band = np.abs(sorted_latitudes) <= band
    sorted_latitudes = sorted_latitudes[in_band]
    sorted_profile = sorted_profile[in_band]
    crossings = np.where(np.diff(np.sign(sorted_profile)) != 0)[0]
    if crossings.size == 0:
        return float("nan")
    latitudes_of_crossings = np.array(
        [
            sorted_latitudes[i]
            - sorted_profile[i]
            * (sorted_latitudes[i + 1] - sorted_latitudes[i])
            / (sorted_profile[i + 1] - sorted_profile[i])
            for i in crossings
        ],
    )
    return float(latitudes_of_crossings[np.argmin(np.abs(latitudes_of_crossings))])


def itcz_transport_regression(
    itcz_latitudes: np.ndarray,
    cross_equatorial_transport: np.ndarray,
) -> tuple[float, float]:
    """Regression of the ITCZ latitude on the cross-equatorial transport.

    Parameters
    ----------
    itcz_latitudes:
        Latitude of the zonal-mean precipitation maximum (degrees), one
        value per climatological month.
    cross_equatorial_transport:
        Northward atmospheric energy transport across the equator (PW), one
        value per climatological month.

    Returns
    -------
    :
        Slope (degrees per PW) and correlation coefficient.

    """
    regression = stats.linregress(
        np.asarray(cross_equatorial_transport, dtype=float),
        np.asarray(itcz_latitudes, dtype=float),
    )
    return float(regression.slope), float(regression.rvalue)


def monthly_anomalies(series: np.ndarray) -> np.ndarray:
    """Remove the monthly climatology from a monthly series.

    Parameters
    ----------
    series:
        Monthly series starting in January; its length need not be a
        multiple of twelve.

    Returns
    -------
    :
        Anomalies with respect to the climatology of each calendar month.

    """
    values = np.asarray(series, dtype=float)
    out = values.copy()
    for month in range(MONTHS_PER_YEAR):
        selection = np.arange(month, values.size, MONTHS_PER_YEAR)
        out[selection] = values[selection] - np.nanmean(values[selection])
    return out


def running_mean_correlation(
    series_a: np.ndarray,
    series_b: np.ndarray,
    window: int,
) -> float:
    """Correlation of two series after a centred running mean.

    Parameters
    ----------
    series_a:
        First monthly series.
    series_b:
        Second monthly series of the same length.
    window:
        Length of the running mean (time steps).

    Returns
    -------
    :
        Pearson correlation of the smoothed series.

    Raises
    ------
    ValueError
        If the series are too short for the running mean.

    """
    values_a = monthly_anomalies(series_a)
    values_b = monthly_anomalies(series_b)
    if values_a.size < window + 2:
        msg = (
            f"the series are too short ({values_a.size}) for a running mean of "
            f"{window} time steps"
        )
        raise ValueError(msg)
    kernel = np.ones(window) / window
    smoothed_a = np.convolve(values_a, kernel, mode="valid")
    smoothed_b = np.convolve(values_b, kernel, mode="valid")
    return float(stats.pearsonr(smoothed_a, smoothed_b)[0])


def standardised(series: np.ndarray) -> np.ndarray:
    """Standardise a series to zero mean and unit standard deviation.

    Parameters
    ----------
    series:
        Series to standardise.

    Returns
    -------
    :
        Standardised series, all-NaN if the series has no spread.

    """
    values = np.asarray(series, dtype=float)
    spread = np.nanstd(values, ddof=1)
    if not np.isfinite(spread) or spread == 0.0:
        return np.full_like(values, np.nan)
    return (values - np.nanmean(values)) / spread


def regression_pattern(field: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Per-gridpoint regression of a field on a one-dimensional index.

    Parameters
    ----------
    field:
        Field of shape ``(n_time, lat, lon)``.
    index:
        Index of shape ``(n_time,)``.

    Returns
    -------
    :
        Slope map of shape ``(lat, lon)``, NaN where the field is masked and
        everywhere if the index has no variance.

    """
    field = np.asarray(field, dtype=float)
    index = np.asarray(index, dtype=float)
    index_anomaly = index - index.mean()
    variance = float((index_anomaly**2).mean())
    if not np.isfinite(variance) or variance <= 0.0:
        return np.full(field.shape[1:], np.nan)
    field_anomaly = field - field.mean(axis=0, keepdims=True)
    covariance = (field_anomaly * index_anomaly[:, None, None]).mean(axis=0)
    return covariance / variance


class _EnergyTransportDiagnostic(ScalarExperimentDiagnostic):
    """Manage diagnostics of the implied meridional energy transport.

    Shares the flux bookkeeping of the residual method (Trenberth & Caron
    2001): the atmospheric energy convergence is the TOA net flux minus the
    surface net flux, and the oceanic one is the surface net flux itself.
    """

    @staticmethod
    def _integrable(profile: np.ndarray) -> np.ndarray:
        """Set latitudes without any data to zero flux.

        A latitude circle that the model grid does not reach (the poles of a
        grid that stops short of them) is masked by the regridding. Its
        contribution to the cumulative integral is weighted by cos(lat) and
        so negligible, but a NaN there would propagate to every latitude
        poleward of it.
        """
        return np.nan_to_num(profile, nan=0.0, posinf=0.0, neginf=0.0)

    def _zonal_mean_series(
        self,
        data: CubeList | Dataset,
        variable: Variable,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the zonal-mean series of one variable and its latitudes."""
        cube = self._prepared_cube(data, variable)
        with setup_esmvaltool_config_and_logging():
            cube = zonal_statistics(cube, "mean")
        return (
            self._integrable(as_float_array(cube.data)),
            cube.coord("latitude").points.astype(float),
        )

    def _monthly_climatology_zonal_mean(
        self,
        data: CubeList | Dataset,
        variable: Variable,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the (12, lat) climatological zonal mean of one variable."""
        cube = self._prepared_cube(data, variable)
        with setup_esmvaltool_config_and_logging():
            cube = climate_statistics(cube, "mean", "month")
            cube = zonal_statistics(cube, "mean")
        return (
            self._integrable(as_float_array(cube.data)),
            cube.coord("latitude").points.astype(float),
        )

    @staticmethod
    def _net_fluxes(
        profiles: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the TOA and surface net downward fluxes from the components."""
        toa_net = np.sum(
            [sign * profiles[name] for name, sign in _TOA_FLUXES],
            axis=0,
        )
        surface_net = np.sum(
            [sign * profiles[name] for name, sign in _SURFACE_FLUXES],
            axis=0,
        )
        return toa_net, surface_net


class ITCZEnergyFluxEquator(_EnergyTransportDiagnostic):
    """Manage ITCZ energy-flux-equator diagnostics.

    The energetic constraint on tropical rainfall: the ITCZ sits near the
    energy flux equator and migrates with the cross-equatorial atmospheric
    energy transport over the seasonal cycle (Kang et al. 2008; Donohoe et
    al. 2013; Schneider et al. 2014). On the climatological seasonal cycle
    (twelve monthly means), the latitude of the zonal-mean precipitation
    maximum is regressed on the northward atmospheric energy transport
    across the equator; observations give a slope of about -3 degrees per PW
    with a correlation close to one.

    The atmospheric transport comes from the residual method used by
    :class:`~climateeval.diags.complex.MeridionalHeatTransport`: the
    atmospheric energy convergence is the TOA net flux minus the surface net
    flux, integrated from the South Pole
    (:func:`~climateeval.diags.complex._transport.implied_meridional_transport`).

    Additional keyword arguments
    ----------------------------
    itcz_latitude_band: tuple[float, float] (default: (-30.0, 30.0))
        Band searched for the zonal-mean precipitation maximum (degrees).

    """

    _required_data_keys = ("historical",)

    _default_itcz_latitude_band: ClassVar[tuple[float, float]] = (-30.0, 30.0)

    _pr: ClassVar[Variable] = Variable("pr", "pr", "mon")
    _input_variables: ClassVar[tuple[Variable, ...]] = (*_FLUX_VARIABLES, _pr)

    _slope: ClassVar[Variable] = Variable("itcz_efe_slope", "itcz_efe_slope", "fx")
    _correlation: ClassVar[Variable] = Variable(
        "itcz_efe_correlation",
        "itcz_efe_correlation",
        "fx",
    )
    _efe_latitude: ClassVar[Variable] = Variable(
        "energy_flux_equator_lat",
        "energy_flux_equator_lat",
        "fx",
    )
    _output_variables: ClassVar[tuple[Variable, ...]] = (
        _slope,
        _correlation,
        _efe_latitude,
    )

    _other_data_generator = CMIP6HistoricalR1I1P1F1
    _other_data_exp = "historical"

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize class instance."""
        super().__init__(name, **kwargs)
        self._itcz_latitude_band: tuple[float, float] = tuple(
            self._kwargs.get(
                "itcz_latitude_band",
                self._default_itcz_latitude_band,
            ),
        )

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        data = complex_data_source.data["historical"]
        profiles: dict[str, np.ndarray] = {}
        latitudes = np.array([])
        for variable in _FLUX_VARIABLES:
            profiles[variable.var_name], latitudes = (
                self._monthly_climatology_zonal_mean(data, variable)
            )
        toa_net, surface_net = self._net_fluxes(profiles)
        atmospheric_convergence = toa_net - surface_net

        pr_zonal, pr_latitudes = self._monthly_climatology_zonal_mean(data, self._pr)
        in_band = (pr_latitudes >= self._itcz_latitude_band[0]) & (
            pr_latitudes <= self._itcz_latitude_band[1]
        )

        itcz_latitudes = np.empty(MONTHS_PER_YEAR)
        cross_equatorial = np.empty(MONTHS_PER_YEAR)
        efe_latitudes = np.empty(MONTHS_PER_YEAR)
        order = np.argsort(latitudes)
        for month in range(MONTHS_PER_YEAR):
            transport = implied_meridional_transport(
                atmospheric_convergence[month],
                latitudes,
            )
            cross_equatorial[month] = (
                np.interp(0.0, latitudes[order], transport[order]) / _WATTS_PER_PETAWATT
            )
            efe_latitudes[month] = zero_crossing_nearest_equator(transport, latitudes)
            itcz_latitudes[month] = pr_latitudes[in_band][
                np.argmax(pr_zonal[month][in_band])
            ]

        slope, correlation = itcz_transport_regression(
            itcz_latitudes,
            cross_equatorial,
        )
        return self._scalar_cubes(
            {
                self._slope: slope,
                self._correlation: correlation,
                self._efe_latitude: float(np.nanmean(efe_latitudes)),
            },
        )


class BjerknesCompensation(_EnergyTransportDiagnostic):
    """Manage Bjerknes compensation diagnostics.

    Bjerknes (1964) compensation: with the TOA radiative imbalance and the
    ocean heat content varying only slowly, anomalies of the atmospheric and
    oceanic meridional heat transports must largely cancel on decadal
    timescales. The diagnostic forms both transports of a control run with
    the residual method, averages them over ``latitude_band``, removes the
    monthly climatology and correlates them after a centred running mean of
    ``running_mean_months`` (Outten et al. 2018). Coupled models compensate
    strongly in the subpolar North Atlantic latitudes, with correlations
    well below zero.

    Additional keyword arguments
    ----------------------------
    latitude_band: tuple[float, float] (default: (40.0, 70.0))
        Latitude band the transports are averaged over (degrees north).
    running_mean_months: int (default: 121)
        Length of the centred running mean (months), about a decade.

    """

    _required_data_keys = ("picontrol",)

    _default_latitude_band: ClassVar[tuple[float, float]] = (40.0, 70.0)
    _default_running_mean_months: ClassVar[int] = 121

    _input_variables: ClassVar[tuple[Variable, ...]] = _FLUX_VARIABLES

    _correlation: ClassVar[Variable] = Variable(
        "bjerknes_correlation",
        "bjerknes_correlation",
        "fx",
    )
    _output_variables: ClassVar[tuple[Variable, ...]] = (_correlation,)

    _other_data_generator = CMIP6PiControlR1I1P1F1
    _other_data_exp = "piControl"

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize class instance."""
        super().__init__(name, **kwargs)
        self._latitude_band: tuple[float, float] = tuple(
            self._kwargs.get("latitude_band", self._default_latitude_band),
        )
        self._running_mean_months: int = int(
            self._kwargs.get(
                "running_mean_months",
                self._default_running_mean_months,
            ),
        )

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        data = complex_data_source.data["picontrol"]
        profiles: dict[str, np.ndarray] = {}
        latitudes = np.array([])
        for variable in _FLUX_VARIABLES:
            profiles[variable.var_name], latitudes = self._zonal_mean_series(
                data,
                variable,
            )
        toa_net, surface_net = self._net_fluxes(profiles)
        in_band = (latitudes >= self._latitude_band[0]) & (
            latitudes <= self._latitude_band[1]
        )

        n_time = toa_net.shape[0]
        atmospheric = np.empty(n_time)
        oceanic = np.empty(n_time)
        for step in range(n_time):
            atmospheric[step] = implied_meridional_transport(
                toa_net[step] - surface_net[step],
                latitudes,
            )[in_band].mean()
            oceanic[step] = implied_meridional_transport(
                surface_net[step],
                latitudes,
            )[in_band].mean()

        return self._scalar_cubes(
            {
                self._correlation: running_mean_correlation(
                    atmospheric,
                    oceanic,
                    self._running_mean_months,
                ),
            },
        )


class ENSOTeleconnectionPatterns(ScalarExperimentDiagnostic):
    """Manage ENSO teleconnection pattern diagnostics.

    ENSO imprints a characteristic pattern on tropical surface temperature
    and precipitation. The diagnostic regresses monthly anomalies of ``ts``
    and ``pr`` at every grid point of the tropical band onto the
    standardised Nino 3.4 index of the same run, does the same for the
    observations, and returns the centred, cosine-weighted spatial
    correlation of the two regression patterns - one number per field, above
    0.7 for a model that reproduces the observed teleconnections. Only the
    tropical band is used; extratropical teleconnections are poorly
    constrained even in long ensembles (Deser et al. 2017).

    The reference patterns come from HadISST sea surface temperature and
    GPCP precipitation over ``observation_timerange``, both regressed on the
    HadISST Nino 3.4 index (GPCP carries no temperature) and aligned on
    their common months. HadISST is an SST product, so its temperature
    pattern is missing over land; the pattern correlation drops non-finite
    pairs, which masks the model's ``ts`` to the same ocean points without a
    separate land mask.

    A pattern correlation is scale-free and ENSO teleconnections are a
    property of internal variability, so the model patterns of a control run
    and the observed patterns of the satellite era need not be
    contemporaneous - they cannot be.

    Additional keyword arguments
    ----------------------------
    latitude_band: tuple[float, float] (default: (-30.0, 30.0))
        Tropical band the patterns are compared over (degrees).
    use_observations: bool (default: True)
        Compute the observed reference patterns. With ``False``, or whenever
        the observations cannot be obtained and ``fail_on_missing_data`` is
        ``False``, only the amplitude of the model's Nino 3.4 index is
        emitted.
    observation_timerange: str (default: "19790101/20141231")
        Time range of the observational reference (GPCP starts in 1979).

    """

    _required_data_keys = ("picontrol",)

    _default_latitude_band: ClassVar[tuple[float, float]] = (-30.0, 30.0)
    _default_observation_timerange: ClassVar[str] = "19790101/20141231"
    _nino34_region: ClassVar[tuple[float, float, float, float]] = (
        190.0,
        240.0,
        -5.0,
        5.0,
    )
    _min_common_months: ClassVar[int] = 24

    _tos: ClassVar[Variable] = Variable("tos", "tos", "mon")
    _ts: ClassVar[Variable] = Variable("ts", "ts", "mon")
    _pr: ClassVar[Variable] = Variable("pr", "pr", "mon")
    _input_variables: ClassVar[tuple[Variable, ...]] = (_tos, _ts, _pr)

    _corr_ts: ClassVar[Variable] = Variable(
        "enso_teleconnection_corr_ts",
        "enso_teleconnection_corr_ts",
        "fx",
    )
    _corr_pr: ClassVar[Variable] = Variable(
        "enso_teleconnection_corr_pr",
        "enso_teleconnection_corr_pr",
        "fx",
    )
    _nino34_std: ClassVar[Variable] = Variable("nino34_std", "nino34_std", "fx")
    _output_variables: ClassVar[tuple[Variable, ...]] = (
        _corr_ts,
        _corr_pr,
        _nino34_std,
    )

    _other_data_generator = CMIP6PiControlR1I1P1F1
    _other_data_exp = "piControl"

    #: Observational references, as classes so a test can substitute them.
    _sst_reference: ClassVar[type[DataSource]] = HadISST
    _precipitation_reference: ClassVar[type[DataSource]] = GPCP

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize class instance."""
        super().__init__(name, **kwargs)
        self._latitude_band: tuple[float, float] = tuple(
            self._kwargs.get("latitude_band", self._default_latitude_band),
        )
        self._use_observations: bool = bool(
            self._kwargs.get("use_observations", True),
        )
        self._observation_timerange: str = str(
            self._kwargs.get(
                "observation_timerange",
                self._default_observation_timerange,
            ),
        )

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        data = complex_data_source.data["picontrol"]
        sea_surface_temperature = self._prepared_cube(data, self._tos)
        index, index_months = self._nino34_index(sea_surface_temperature)
        latitudes = np.array([])
        model_patterns: dict[str, np.ndarray | None] = {}
        for variable in (self._ts, self._pr):
            field, latitudes, months = self._tropical_anomalies(
                self._prepared_cube(data, variable),
            )
            model_patterns[variable.var_name] = self._pattern(
                field,
                months,
                index,
                index_months,
            )

        values = {self._nino34_std: self._nino34_amplitude(sea_surface_temperature)}
        observed_patterns = self._observed_patterns()
        if observed_patterns is None:
            return self._scalar_cubes(values)
        for variable, output in ((self._ts, self._corr_ts), (self._pr, self._corr_pr)):
            model = model_patterns[variable.var_name]
            observed = observed_patterns[variable.var_name]
            if model is None or observed is None:
                continue
            values[output] = banded_pattern_correlation(
                model,
                observed,
                latitudes,
                band=self._latitude_band,
            )
        return self._scalar_cubes(values)

    # -- the pieces of one regression pattern -------------------------------

    def _nino34_index(
        self,
        sea_surface_temperature: Cube,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the standardised Nino 3.4 index and its (year, month) keys.

        The unsmoothed index: the operational three-month running mean is
        left out so the regression uses every month, and the index is
        standardised so the slopes are per standard deviation of Nino 3.4 and
        therefore comparable between a model and the observations.
        """
        cube = self._nino34_cube(sea_surface_temperature)
        return standardised(as_float_array(cube.data)), self._month_keys(cube)

    def _nino34_amplitude(self, sea_surface_temperature: Cube) -> float:
        """Get the standard deviation of the raw Nino 3.4 index (K)."""
        cube = self._nino34_cube(sea_surface_temperature)
        return float(np.nanstd(as_float_array(cube.data), ddof=1))

    def _nino34_cube(self, sea_surface_temperature: Cube) -> Cube:
        """Get the deseasonalised Nino 3.4 box mean."""
        min_lon, max_lon, min_lat, max_lat = self._nino34_region
        with setup_esmvaltool_config_and_logging():
            cube = extract_region(
                sea_surface_temperature,
                start_longitude=min_lon,
                end_longitude=max_lon,
                start_latitude=min_lat,
                end_latitude=max_lat,
            )
            cube = anomalies(cube, period="month")
            cube = area_statistics(cube, "mean")
        return cube  # noqa: RET504 - named for readability

    def _tropical_anomalies(
        self,
        cube: Cube,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the monthly anomalies over the band, latitudes and keys."""
        with setup_esmvaltool_config_and_logging():
            cube = extract_region(
                cube,
                start_longitude=0.0,
                end_longitude=360.0,
                start_latitude=self._latitude_band[0],
                end_latitude=self._latitude_band[1],
            )
            cube = anomalies(cube, period="month")
        return (
            as_float_array(cube.data),
            cube.coord("latitude").points.astype(float),
            self._month_keys(cube),
        )

    @staticmethod
    def _month_keys(cube: Cube) -> np.ndarray:
        """Get a sortable ``year * 12 + month`` key for every time step."""
        years, months = calendar_years_and_months(cube)
        return years * MONTHS_PER_YEAR + months

    def _pattern(
        self,
        field: np.ndarray,
        field_keys: np.ndarray,
        index: np.ndarray,
        index_keys: np.ndarray,
    ) -> np.ndarray | None:
        """Get the regression pattern of a field on the index."""
        common = np.intersect1d(field_keys, index_keys)
        if common.size < self._min_common_months:
            logger.warning(
                f"Diagnostic '{self.name}': only {common.size} months are shared "
                f"between the field and the Nino 3.4 index (at least "
                f"{self._min_common_months} are needed); no regression pattern",
            )
            return None
        return regression_pattern(
            field[np.isin(field_keys, common)],
            index[np.isin(index_keys, common)],
        )

    # -- the observed reference patterns ------------------------------------

    def _observed_patterns(self) -> dict[str, np.ndarray | None] | None:
        """Get the observed patterns, or ``None`` if they are unobtainable."""
        if not self._use_observations:
            return None
        try:
            return self._compute_observed_patterns()
        except Exception as exc:
            if self._fail_on_missing_data:
                raise
            logger.warning(
                f"Diagnostic '{self.name}': no observed ENSO teleconnection "
                f"patterns ({type(exc).__name__}: {exc}); emitting the model's "
                f"Nino 3.4 amplitude without the pattern correlations",
            )
            return None

    def _compute_observed_patterns(self) -> dict[str, np.ndarray | None]:
        """Get the HadISST and GPCP regression patterns."""
        sea_surface_temperature = self._observation_cube(
            self._sst_reference,
            self._tos,
        )
        index, index_keys = self._nino34_index(sea_surface_temperature)
        temperature, _latitudes, temperature_keys = self._tropical_anomalies(
            sea_surface_temperature,
        )
        precipitation, _pr_latitudes, precipitation_keys = self._tropical_anomalies(
            self._observation_cube(self._precipitation_reference, self._pr),
        )
        return {
            self._ts.var_name: self._pattern(
                temperature,
                temperature_keys,
                index,
                index_keys,
            ),
            self._pr.var_name: self._pattern(
                precipitation,
                precipitation_keys,
                index,
                index_keys,
            ),
        }

    def _observation_cube(
        self,
        data_source: type[DataSource],
        variable: Variable,
    ) -> Cube:
        """Get one observed variable on the common grid."""
        cube = data_source().get_cube(
            self.data_root_dir,
            Variable(
                variable.id,
                variable.var_name,
                variable.frequency,
                timerange=self._observation_timerange,
            ),
            download_missing_data=self._download_missing_data,
        )
        with setup_esmvaltool_config_and_logging():
            cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
        return cube  # noqa: RET504 - named for readability
