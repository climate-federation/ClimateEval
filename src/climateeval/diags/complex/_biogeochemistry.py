"""Provide ocean biogeochemistry diagnostics.

:class:`ChlorophyllDriverCoupling` ports the physical-biogeochemical coupling
analysis of surface chlorophyll-a developed by G. G. Navarra for CMIP6 Earth
system models (Navarra et al., "Global and Polar Failures in
Physical-Biogeochemical Coupling of Chlorophyll-a Across CMIP6 Earth System
Models", submitted): is the surface chlorophyll of a model coupled to its
mixed-layer depth, its sea surface temperature and its sea ice with the
sign, strength and timing seen in the observations?

The statistics are plain array functions (:func:`lag_correlation`,
:func:`monthly_latitude_correlation`, ...) so they can be tested on their
own; the diagnostic class only arranges the data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from esmvalcore.preprocessor import regrid
from loguru import logger

from climateeval import Variable
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._variable import VerticalLevel
from climateeval.data import (
    ESACCIOC,
    ORAS5,
    CMIP6HistoricalR1I1P1F1,
    HadISST,
)
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.complex._scalar import (
    ScalarExperimentDiagnostic,
    as_float_array,
    calendar_years_and_months,
)

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval.data import ComplexDataSource, DataSource

logger = logger.opt(colors=True)

MONTHS_PER_YEAR = 12

#: Basin boxes ``(lat_min, lat_max, lon_min, lon_max)`` of the paper
#: (longitudes 0-360, all edges inclusive).
BASINS: dict[str, tuple[float, float, float, float]] = {
    "Global": (-90.0, 90.0, 0.0, 360.0),
    "Atlantic": (-60.0, 70.0, 290.0, 360.0),
    "Pacific": (-60.0, 60.0, 130.0, 280.0),
    "Indian": (-60.0, 30.0, 20.0, 130.0),
    "Southern": (-90.0, -45.0, 0.0, 360.0),
    "Arctic": (65.0, 90.0, 0.0, 360.0),
}

#: Basins in which the sea-ice coupling is evaluated.
SICONC_BASINS: tuple[str, ...] = ("Arctic", "Southern")

#: Pairs needed for one correlation coefficient.
_MIN_CORRELATION_PAIRS = 3


# ---------------------------------------------------------------------------
# Pure formulae
# ---------------------------------------------------------------------------


def basin_mask(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    box: tuple[float, float, float, float],
) -> np.ndarray:
    """Get the boolean mask of a basin box on a regular grid.

    Parameters
    ----------
    latitudes:
        Grid latitudes (degrees north).
    longitudes:
        Grid longitudes (degrees east, 0-360).
    box:
        ``(lat_min, lat_max, lon_min, lon_max)``; a box whose ``lon_min``
        exceeds ``lon_max`` wraps across 0 degrees.

    Returns
    -------
    :
        Mask of shape ``(n_lat, n_lon)``.

    """
    lat_min, lat_max, lon_min, lon_max = box
    lats = np.asarray(latitudes, dtype=float)
    lons = np.mod(np.asarray(longitudes, dtype=float), 360.0)
    in_lat = (lats >= lat_min) & (lats <= lat_max)
    if lon_min <= lon_max:
        in_lon = (lons >= lon_min) & (lons <= lon_max)
    else:
        in_lon = (lons >= lon_min) | (lons <= lon_max)
    return in_lat[:, None] & in_lon[None, :]


def area_weighted_basin_mean(
    field: np.ndarray,
    latitudes: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Get the cosine-weighted mean series of a field over a basin.

    Parameters
    ----------
    field:
        Field of shape ``(n_time, n_lat, n_lon)``; NaN marks missing cells.
    latitudes:
        Grid latitudes (degrees north).
    mask:
        Basin mask of shape ``(n_lat, n_lon)``.

    Returns
    -------
    :
        Series of length ``n_time``; NaN for steps without a valid cell.

    """
    field = np.asarray(field, dtype=float)
    weights = np.cos(np.deg2rad(np.asarray(latitudes, dtype=float)))[:, None]
    weights = np.where(mask, weights, 0.0)[None]
    valid = np.isfinite(field)
    weighted = np.where(valid, field, 0.0) * weights
    norm = np.sum(np.where(valid, weights, 0.0), axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(norm > 0.0, np.sum(weighted, axis=(1, 2)) / norm, np.nan)


def deseasonalised_standardised(
    field: np.ndarray,
    months: np.ndarray,
    *,
    detrend: bool = True,
) -> np.ndarray:
    """Get standardised monthly anomalies along the leading (time) axis.

    The calendar-month climatology is removed, the anomalies are (optionally)
    linearly detrended against the time index, and the result is divided by
    its standard deviation; every step is done per cell on the valid steps
    of that cell only, so gaps stay gaps.

    Parameters
    ----------
    field:
        Series or field of shape ``(n_time, ...)``.
    months:
        Calendar month (1-12) of every time step.
    detrend:
        Remove a linear trend from the anomalies.

    Returns
    -------
    :
        Standardised anomalies with the input's shape; NaN wherever the
        input is NaN or the series is constant.

    """
    field = np.asarray(field, dtype=float)
    months = np.asarray(months)
    if months.size != field.shape[0]:
        msg = f"got {months.size} months for {field.shape[0]} time steps"
        raise ValueError(msg)
    anomalies = np.full_like(field, np.nan)
    with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        for month in np.unique(months):
            in_month = months == month
            anomalies[in_month] = field[in_month] - np.nanmean(field[in_month], axis=0)
        if detrend:
            valid = np.isfinite(anomalies)
            index = np.arange(field.shape[0], dtype=float).reshape(
                (-1,) + (1,) * (field.ndim - 1),
            )
            index = np.broadcast_to(index, field.shape)
            index_mean = np.nanmean(np.where(valid, index, np.nan), axis=0)
            centred = np.where(valid, index - index_mean, 0.0)
            denominator = np.sum(centred**2, axis=0)
            slope = np.where(
                denominator > 0.0,
                np.sum(centred * np.where(valid, anomalies, 0.0), axis=0)
                / np.where(denominator > 0.0, denominator, 1.0),
                0.0,
            )
            anomalies = anomalies - slope * centred
        std = np.nanstd(anomalies, axis=0, ddof=1)
        return np.where(std > 0.0, anomalies / np.where(std > 0.0, std, 1.0), np.nan)


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Get the Pearson correlation of the finite pairs of two arrays.

    Parameters
    ----------
    x:
        First array (flattened).
    y:
        Second array (flattened), same size.

    Returns
    -------
    :
        Correlation coefficient; NaN with fewer than three pairs or a
        constant series.

    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < _MIN_CORRELATION_PAIRS:
        return float("nan")
    xv, yv = x[valid] - x[valid].mean(), y[valid] - y[valid].mean()
    norm = np.sqrt(np.sum(xv**2) * np.sum(yv**2))
    if norm <= 0.0:
        return float("nan")
    return float(np.sum(xv * yv) / norm)


def lag_correlation(
    chlorophyll: np.ndarray,
    driver: np.ndarray,
    *,
    lag_max: int = 12,
    min_pairs: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Get the lag correlation of chlorophyll with a driver series.

    Both series are already aligned in time (one value per month). A
    positive lag ``k`` correlates the driver at month ``t - k`` with the
    chlorophyll at month ``t``: the driver leads the chlorophyll. A negative
    lag has the chlorophyll leading.

    Parameters
    ----------
    chlorophyll:
        Chlorophyll anomaly series.
    driver:
        Driver anomaly series, same length.
    lag_max:
        Largest lag in months, both ways.
    min_pairs:
        Pairs needed for a coefficient; shorter overlaps give NaN.

    Returns
    -------
    :
        The lags ``-lag_max .. lag_max`` and the correlation at each.

    """
    chl = np.asarray(chlorophyll, dtype=float)
    drv = np.asarray(driver, dtype=float)
    if chl.shape != drv.shape:
        msg = f"series of different lengths: {chl.shape} and {drv.shape}"
        raise ValueError(msg)
    n_time = chl.size
    lags = np.arange(-lag_max, lag_max + 1)
    correlations = np.full(lags.size, np.nan)
    for i, lag in enumerate(lags):
        if lag >= 0:
            x, y = drv[: n_time - lag], chl[lag:]
        else:
            x, y = drv[-lag:], chl[: n_time + lag]
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() >= max(min_pairs, _MIN_CORRELATION_PAIRS):
            correlations[i] = pearson_correlation(x[valid], y[valid])
    return lags, correlations


def peak_lag(lags: np.ndarray, correlations: np.ndarray) -> tuple[float, float]:
    """Get the lag of the strongest correlation and that correlation.

    Parameters
    ----------
    lags:
        Lags (months).
    correlations:
        Correlation at every lag.

    Returns
    -------
    :
        ``(lag, correlation)`` at the largest absolute correlation; NaN
        twice if no correlation is finite.

    """
    correlations = np.asarray(correlations, dtype=float)
    if not np.any(np.isfinite(correlations)):
        return float("nan"), float("nan")
    index = int(np.nanargmax(np.abs(correlations)))
    return float(lags[index]), float(correlations[index])


def monthly_latitude_correlation(
    chlorophyll: np.ndarray,
    driver: np.ndarray,
    months: np.ndarray,
    *,
    min_years: int = 10,
    min_cells: int = 10,
) -> np.ndarray:
    """Get the chlorophyll-driver correlation by calendar month and latitude.

    At every grid cell and for every calendar month the anomalies of the two
    fields are correlated across the years, and the coefficients are then
    averaged along the longitudes: the seasonality of the coupling, zonally
    (the month-by-latitude panel of the newer analysis, which has an
    observed counterpart, in place of the paper's model-only Hovmöller).

    Parameters
    ----------
    chlorophyll:
        Chlorophyll anomalies of shape ``(n_time, n_lat, n_lon)``.
    driver:
        Driver anomalies, same shape.
    months:
        Calendar month (1-12) of every time step.
    min_years:
        Years a cell needs in a month for a coefficient.
    min_cells:
        Cells a latitude row needs for a zonal mean.

    Returns
    -------
    :
        Correlations of shape ``(12, n_lat)``, NaN where undetermined.

    """
    chl = np.asarray(chlorophyll, dtype=float)
    drv = np.asarray(driver, dtype=float)
    months = np.asarray(months)
    if chl.shape != drv.shape or chl.ndim != 3:  # noqa: PLR2004 - (time, lat, lon)
        msg = f"expected two (time, lat, lon) fields, got {chl.shape} and {drv.shape}"
        raise ValueError(msg)
    n_lat = chl.shape[1]
    result = np.full((MONTHS_PER_YEAR, n_lat), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        for month in range(1, MONTHS_PER_YEAR + 1):
            a, b = chl[months == month], drv[months == month]
            if a.shape[0] == 0:
                continue
            valid = np.isfinite(a) & np.isfinite(b)
            count = valid.sum(axis=0)
            a0, b0 = np.where(valid, a, 0.0), np.where(valid, b, 0.0)
            safe = np.where(count > 0, count, 1)
            da = np.where(valid, a0 - a0.sum(axis=0) / safe, 0.0)
            db = np.where(valid, b0 - b0.sum(axis=0) / safe, 0.0)
            norm = np.sqrt(np.sum(da**2, axis=0) * np.sum(db**2, axis=0))
            cell = np.where(
                (count >= max(min_years, _MIN_CORRELATION_PAIRS)) & (norm > 0.0),
                np.sum(da * db, axis=0) / np.where(norm > 0.0, norm, 1.0),
                np.nan,
            )
            enough = np.sum(np.isfinite(cell), axis=1) >= min_cells
            row = np.nanmean(np.where(np.isfinite(cell), cell, np.nan), axis=1)
            result[month - 1] = np.where(enough, row, np.nan)
    return result


# ---------------------------------------------------------------------------
# The diagnostic
# ---------------------------------------------------------------------------


class ChlorophyllDriverCoupling(ScalarExperimentDiagnostic):
    """Manage chlorophyll-driver coupling diagnostics.

    Surface chlorophyll responds to the physical state of the upper ocean:
    deep winter mixing dilutes it and brings nutrients, spring shoaling of
    the mixed layer releases the bloom, warm anomalies go with stratified,
    nutrient-poor water, and in the polar oceans the retreat of sea ice
    opens the growing season. The diagnostic measures that coupling the
    same way for the benchmarked model and for the observations:

    - **Lag correlations per basin.** Basin-mean series (cosine-weighted, the
      six boxes of :data:`BASINS`) of the surface chlorophyll and of a driver
      are turned into standardised monthly anomalies (log10 chlorophyll,
      calendar-month climatology removed, linear trend removed) and
      correlated at lags of -12 to +12 months, a positive lag meaning the
      driver leads. Drivers: mixed-layer depth and sea surface temperature in
      every basin, sea-ice concentration in the Arctic and Southern boxes.
    - **Seasonality of the coupling.** At every grid cell the anomalies are
      correlated across the years for each calendar month and the
      coefficients averaged zonally, giving the coupling by month and
      latitude.

    The scalars emitted compare the model with the observations:
    ``chl_<driver>_coupling_corr`` is the correlation between the model's and
    the observed basin-by-lag curves (one for a model that reproduces the
    sign, strength and timing of the coupling everywhere, below zero for one
    with the wrong sign), ``chl_<driver>_seasonal_coupling_corr`` the same
    for the month-by-latitude field. For context the model's own global
    correlation at lag zero and the lag of its strongest global correlation
    are emitted as well. Without observations only the model's own numbers
    are produced.

    Observations: ESA CCI Ocean Colour chlorophyll, HadISST sea surface
    temperature and sea-ice concentration, ORAS5 mixed-layer depth (staged
    locally; see :class:`~climateeval.data.ORAS5`). Every source is regridded
    to the common 2x2 grid, and the model's chlorophyll is restricted to the
    cells the satellite product covers so both see the same ocean.

    Additional keyword arguments
    ----------------------------
    lag_max: int (default: 12)
        Largest lag of the basin correlations (months).
    log_chlorophyll: bool (default: True)
        Correlate log10 chlorophyll (non-positive values become missing).
    detrend: bool (default: True)
        Remove a linear trend from the monthly anomalies.
    min_pairs: int (default: 24)
        Months needed for one lag correlation.
    min_years_per_month: int (default: 10)
        Years a cell needs for one calendar month of the seasonal coupling.
    use_observations: bool (default: True)
        Compute the observed coupling. With ``False``, or whenever the
        observations cannot be obtained and ``fail_on_missing_data`` is
        ``False``, only the model's own numbers are emitted.
    observation_timerange: str (default: "19980101/20141231")
        Time range of the observations (the satellite record starts in
        September 1997; the paper's lag analysis uses 1998-2014).

    """

    _required_data_keys = ("historical",)

    _default_lag_max: ClassVar[int] = 12
    _default_min_pairs: ClassVar[int] = 24
    _default_min_years_per_month: ClassVar[int] = 10
    _default_observation_timerange: ClassVar[str] = "19980101/20141231"
    _min_cells_per_latitude: ClassVar[int] = 10

    _chl: ClassVar[Variable] = Variable(
        "chl",
        "chl",
        "mon",
        units="mg m-3",
        vertical_level=VerticalLevel("depth", "m", 0.0, extrapolate=True),
    )
    _tos: ClassVar[Variable] = Variable("tos", "tos", "mon")
    _mlotst: ClassVar[Variable] = Variable("mlotst", "mlotst", "mon")
    _siconc: ClassVar[Variable] = Variable("siconc", "siconc", "mon")
    _input_variables: ClassVar[tuple[Variable, ...]] = (_chl, _tos, _mlotst, _siconc)

    #: Drivers: (name, variable, basins the coupling is evaluated in).
    _drivers: ClassVar[tuple[tuple[str, Variable, tuple[str, ...]], ...]] = (
        ("mld", _mlotst, tuple(BASINS)),
        ("sst", _tos, tuple(BASINS)),
        ("siconc", _siconc, SICONC_BASINS),
    )
    #: Drivers whose seasonal coupling and global lag numbers are emitted.
    _context_drivers: ClassVar[tuple[str, ...]] = ("mld", "sst")

    _mld_coupling: ClassVar[Variable] = Variable(
        "chl_mld_coupling_corr",
        "chl_mld_coupling_corr",
        "fx",
    )
    _sst_coupling: ClassVar[Variable] = Variable(
        "chl_sst_coupling_corr",
        "chl_sst_coupling_corr",
        "fx",
    )
    _siconc_coupling: ClassVar[Variable] = Variable(
        "chl_siconc_coupling_corr",
        "chl_siconc_coupling_corr",
        "fx",
    )
    _mld_seasonal: ClassVar[Variable] = Variable(
        "chl_mld_seasonal_coupling_corr",
        "chl_mld_seasonal_coupling_corr",
        "fx",
    )
    _sst_seasonal: ClassVar[Variable] = Variable(
        "chl_sst_seasonal_coupling_corr",
        "chl_sst_seasonal_coupling_corr",
        "fx",
    )
    _mld_lag0: ClassVar[Variable] = Variable(
        "chl_mld_corr_lag0",
        "chl_mld_corr_lag0",
        "fx",
    )
    _sst_lag0: ClassVar[Variable] = Variable(
        "chl_sst_corr_lag0",
        "chl_sst_corr_lag0",
        "fx",
    )
    _mld_peak_lag: ClassVar[Variable] = Variable(
        "chl_mld_peak_lag",
        "chl_mld_peak_lag",
        "fx",
    )
    _sst_peak_lag: ClassVar[Variable] = Variable(
        "chl_sst_peak_lag",
        "chl_sst_peak_lag",
        "fx",
    )
    _output_variables: ClassVar[tuple[Variable, ...]] = (
        _mld_coupling,
        _sst_coupling,
        _siconc_coupling,
        _mld_seasonal,
        _sst_seasonal,
        _mld_lag0,
        _sst_lag0,
        _mld_peak_lag,
        _sst_peak_lag,
    )

    _other_data_generator = CMIP6HistoricalR1I1P1F1
    _other_data_exp = "historical"

    #: Observational references, as classes so a test can substitute them.
    _chlorophyll_reference: ClassVar[type[DataSource]] = ESACCIOC
    _sst_reference: ClassVar[type[DataSource]] = HadISST
    _sea_ice_reference: ClassVar[type[DataSource]] = HadISST
    _mld_reference: ClassVar[type[DataSource]] = ORAS5

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize class instance."""
        super().__init__(name, **kwargs)
        self._lag_max = int(self._kwargs.get("lag_max", self._default_lag_max))
        self._log_chlorophyll = bool(self._kwargs.get("log_chlorophyll", True))
        self._detrend = bool(self._kwargs.get("detrend", True))
        self._min_pairs = int(self._kwargs.get("min_pairs", self._default_min_pairs))
        self._min_years_per_month = int(
            self._kwargs.get(
                "min_years_per_month",
                self._default_min_years_per_month,
            ),
        )
        self._use_observations = bool(self._kwargs.get("use_observations", True))
        self._observation_timerange = str(
            self._kwargs.get(
                "observation_timerange",
                self._default_observation_timerange,
            ),
        )
        #: The observed coupling, computed once for every dataset compared.
        self._observed: dict[str, Any] | None = None
        self._observed_known = False

    # -- the diagnostic -----------------------------------------------------

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        data = complex_data_source.data[self._data_key]
        observed = self._observed_coupling()
        coverage = None if observed is None else observed["coverage"]
        model = self._coupling(
            lambda variable: self._prepared_cube(data, variable),
            coverage=coverage,
        )
        values: dict[Variable, float] = {}
        comparisons = {
            "mld": (self._mld_coupling, self._mld_seasonal),
            "sst": (self._sst_coupling, self._sst_seasonal),
            "siconc": (self._siconc_coupling, None),
        }
        for driver, (curve_output, seasonal_output) in comparisons.items():
            if observed is None:
                continue
            model_curves = model["curves"].get(driver)
            observed_curves = observed["curves"].get(driver)
            if model_curves is not None and observed_curves is not None:
                values[curve_output] = pearson_correlation(
                    model_curves,
                    observed_curves,
                )
            if seasonal_output is None:
                continue
            model_seasonal = model["seasonal"].get(driver)
            observed_seasonal = observed["seasonal"].get(driver)
            if model_seasonal is not None and observed_seasonal is not None:
                values[seasonal_output] = pearson_correlation(
                    model_seasonal,
                    observed_seasonal,
                )
        context = {
            "mld": (self._mld_lag0, self._mld_peak_lag),
            "sst": (self._sst_lag0, self._sst_peak_lag),
        }
        for driver, (lag0_output, peak_output) in context.items():
            if driver in model["global_lag0"]:
                values[lag0_output] = model["global_lag0"][driver]
                values[peak_output] = model["global_peak_lag"][driver]
        return self._scalar_cubes(values)

    # -- the coupling of one source -------------------------------------------

    def _coupling(
        self,
        get_cube: Any,
        *,
        coverage: np.ndarray | None,
    ) -> dict[str, Any]:
        """Get the coupling statistics of one source.

        Parameters
        ----------
        get_cube:
            Callable returning the prepared cube of an input variable on the
            common grid.
        coverage:
            Cells the chlorophyll is restricted to (``None``: all).

        Returns
        -------
        :
            ``curves``: per driver the basin-by-lag correlations;
            ``seasonal``: per driver the month-by-latitude correlations;
            ``global_lag0`` / ``global_peak_lag``: per driver the global
            correlation at lag zero and the lag of the strongest one;
            ``coverage``: cells with chlorophyll data.

        """
        chl_values, chl_keys, latitudes, longitudes = self._field(get_cube(self._chl))
        if self._log_chlorophyll:
            with np.errstate(invalid="ignore", divide="ignore"):
                chl_values = np.where(chl_values > 0.0, np.log10(chl_values), np.nan)
        own_coverage = np.sum(np.isfinite(chl_values), axis=0) >= self._min_pairs
        if coverage is not None:
            chl_values = np.where(coverage[None], chl_values, np.nan)
        masks = {
            name: basin_mask(latitudes, longitudes, box) for name, box in BASINS.items()
        }
        result: dict[str, Any] = {
            "curves": {},
            "seasonal": {},
            "global_lag0": {},
            "global_peak_lag": {},
            "coverage": own_coverage,
        }
        for driver, variable, basins in self._drivers:
            try:
                driver_values, driver_keys, _lats, _lons = self._field(
                    get_cube(variable),
                )
            except Exception as exc:
                if self._fail_on_missing_data:
                    raise
                logger.warning(
                    f"Diagnostic '{self.name}': no {variable.var_name} for the "
                    f"{driver} coupling ({type(exc).__name__}: {exc})",
                )
                continue
            common = np.intersect1d(chl_keys, driver_keys)
            if common.size < self._min_pairs:
                logger.warning(
                    f"Diagnostic '{self.name}': only {common.size} months shared "
                    f"between chl and {variable.var_name}; no {driver} coupling",
                )
                continue
            chl = chl_values[np.isin(chl_keys, common)]
            drv = driver_values[np.isin(driver_keys, common)]
            months = common % MONTHS_PER_YEAR + 1
            curves = np.full((len(basins), 2 * self._lag_max + 1), np.nan)
            for i, basin in enumerate(basins):
                chl_series = deseasonalised_standardised(
                    area_weighted_basin_mean(chl, latitudes, masks[basin]),
                    months,
                    detrend=self._detrend,
                )
                driver_series = deseasonalised_standardised(
                    area_weighted_basin_mean(drv, latitudes, masks[basin]),
                    months,
                    detrend=self._detrend,
                )
                lags, curves[i] = lag_correlation(
                    chl_series,
                    driver_series,
                    lag_max=self._lag_max,
                    min_pairs=self._min_pairs,
                )
                if basin == "Global":
                    result["global_lag0"][driver] = float(curves[i][self._lag_max])
                    result["global_peak_lag"][driver] = peak_lag(lags, curves[i])[0]
            result["curves"][driver] = curves
            if driver in self._context_drivers:
                result["seasonal"][driver] = monthly_latitude_correlation(
                    deseasonalised_standardised(chl, months, detrend=self._detrend),
                    deseasonalised_standardised(drv, months, detrend=self._detrend),
                    months,
                    min_years=self._min_years_per_month,
                    min_cells=self._min_cells_per_latitude,
                )
        return result

    @staticmethod
    def _field(cube: Cube) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get a cube's values, month keys, latitudes and longitudes."""
        years, months = calendar_years_and_months(cube)
        keys = years * MONTHS_PER_YEAR + (months - 1)
        return (
            as_float_array(cube.data),
            keys,
            cube.coord("latitude").points.astype(float),
            cube.coord("longitude").points.astype(float),
        )

    # -- the observed coupling ------------------------------------------------

    def _observed_coupling(self) -> dict[str, Any] | None:
        """Get the observed coupling, or ``None`` if it is unobtainable."""
        if self._observed_known:
            return self._observed
        self._observed_known = True
        if not self._use_observations:
            return None
        try:
            self._observed = self._compute_observed_coupling()
        except Exception as exc:
            if self._fail_on_missing_data:
                raise
            logger.warning(
                f"Diagnostic '{self.name}': no observed chlorophyll-driver "
                f"coupling ({type(exc).__name__}: {exc}); emitting the model's "
                f"own coupling without the comparison",
            )
            self._observed = None
        return self._observed

    def _compute_observed_coupling(self) -> dict[str, Any]:
        """Get the coupling of the observational references."""
        sources = {
            self._chl.var_name: self._chlorophyll_reference,
            self._tos.var_name: self._sst_reference,
            self._mlotst.var_name: self._mld_reference,
            self._siconc.var_name: self._sea_ice_reference,
        }
        return self._coupling(
            lambda variable: self._observation_cube(
                sources[variable.var_name],
                variable,
            ),
            coverage=None,
        )

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
                units=variable.units,
                vertical_level=variable.vertical_level,
            ),
            download_missing_data=self._download_missing_data,
        )
        with setup_esmvaltool_config_and_logging():
            cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
        return cube  # noqa: RET504 - named for readability
