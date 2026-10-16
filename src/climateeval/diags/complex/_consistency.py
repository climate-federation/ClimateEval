"""Provide physical-consistency diagnostics of a single experiment.

Global conservation checks of an unforced control run (energy balance and
drift, water and atmospheric-energy budget closure) and the thermodynamic
covariances a physically plausible atmosphere has to reproduce (clear-sky
longwave feedback, tropical precipitation-buoyancy relationship,
Clausius-Clapeyron scaling of the water vapour path).

All of them reduce one experiment to a few scalars, so they are built on
:class:`~climateeval.diags.complex._scalar.ScalarExperimentDiagnostic`.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from esmvalcore.preprocessor import (
    anomalies,
    extract_levels,
    extract_region,
    regrid,
)
from loguru import logger

from climateeval import Variable
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval.data import (
    GPCP,
    CMIP6HistoricalR1I1P1F1,
    CMIP6PiControlR1I1P1F1,
    ERA5Monthly,
)
from climateeval.diags import _physics
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

_MIN_COLUMN_LEVELS = 2
"""Fewest common pressure levels a column integral needs."""


class EnergyBalance(ScalarExperimentDiagnostic):
    """Manage energy-balance diagnostics of a control run.

    An unforced control run must conserve energy: the global annual-mean TOA
    net downward flux ``N = rsdt - rsut - rlut`` has to be close to zero and
    free of secular drift, otherwise the run is still equilibrating and any
    forced response computed against it is biased. Both statistics are
    evaluated over the last ``evaluation_years`` years of the control - the
    segment contemporaneous with a historical branch point - so a model still
    equilibrating early in its control is not penalised. Well-behaved CMIP6
    controls have a mean of order 0.1 W m-2 and a drift well below
    0.1 W m-2 per decade.

    Additional keyword arguments
    ----------------------------
    evaluation_years: int (default: 100)
        Number of final control years the two statistics are computed over. A
        shorter control is used whole, with a warning; the number of years
        actually used is reported as ``toa_net_n_years``.
    running_mean_years: int (default: 10)
        Length of the running mean whose least-squares slope defines the
        drift. The drift is reported per decade regardless of this length.

    """

    _required_data_keys = ("picontrol",)

    _default_evaluation_years: ClassVar[int] = 100
    _default_running_mean_years: ClassVar[int] = 10

    _input_variables: ClassVar[tuple[Variable, ...]] = (
        Variable("rsdt", "rsdt", "mon"),
        Variable("rsut", "rsut", "mon"),
        Variable("rlut", "rlut", "mon"),
    )

    _toa_net_mean: ClassVar[Variable] = Variable("toa_net_mean", "toa_net_mean", "fx")
    _toa_net_drift: ClassVar[Variable] = Variable(
        "toa_net_drift",
        "toa_net_drift",
        "fx",
    )
    _toa_net_n_years: ClassVar[Variable] = Variable(
        "toa_net_n_years",
        "toa_net_n_years",
        "fx",
    )
    _output_variables: ClassVar[tuple[Variable, ...]] = (
        _toa_net_mean,
        _toa_net_drift,
        _toa_net_n_years,
    )

    _other_data_generator = CMIP6PiControlR1I1P1F1
    _other_data_exp = "piControl"

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize class instance."""
        super().__init__(name, **kwargs)
        self._evaluation_years: int = int(
            self._kwargs.get("evaluation_years", self._default_evaluation_years),
        )
        self._running_mean_years: int = int(
            self._kwargs.get("running_mean_years", self._default_running_mean_years),
        )

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        toa_net = self._toa_net_global_annual_mean(
            complex_data_source.data["picontrol"],
        )
        if toa_net.size < self._evaluation_years:
            logger.warning(
                f"Diagnostic '{self.name}': the control run has {toa_net.size} "
                f"years, fewer than the {self._evaluation_years} evaluation "
                f"years; using the whole record (reduced power to detect slow "
                f"drifts)",
            )
        else:
            toa_net = toa_net[-self._evaluation_years :]
        drift = _physics.running_mean_drift(toa_net, self._running_mean_years)
        return self._scalar_cubes(
            {
                self._toa_net_mean: float(toa_net.mean()),
                self._toa_net_drift: drift,
                self._toa_net_n_years: float(toa_net.size),
            },
        )


class BudgetClosure(ScalarExperimentDiagnostic):
    """Manage water- and energy-budget closure diagnostics of a control run.

    Two global multidecadal-mean closure residuals of an unforced control
    run:

    - the water budget, ``|<P> - <E>|`` with ``E = hfls / L_v``, which
      vanishes in a steady state (reported in mm day-1);
    - the atmospheric energy budget,
      ``|<Q_rad> - (<L_v P> + <SHF>)|`` with ``Q_rad`` the net radiative
      cooling of the atmospheric column, i.e. the surface net radiation minus
      the TOA net flux (reported in W m-2).

    Both are of order 0.01 mm day-1 and 1 W m-2 in a well-closed model.
    """

    _required_data_keys = ("picontrol",)

    _input_variables: ClassVar[tuple[Variable, ...]] = (
        Variable("pr", "pr", "mon"),
        Variable("hfls", "hfls", "mon"),
        Variable("hfss", "hfss", "mon"),
        Variable("rsdt", "rsdt", "mon"),
        Variable("rsut", "rsut", "mon"),
        Variable("rlut", "rlut", "mon"),
        Variable("rsds", "rsds", "mon"),
        Variable("rsus", "rsus", "mon"),
        Variable("rlds", "rlds", "mon"),
        Variable("rlus", "rlus", "mon"),
    )

    _water_budget_residual: ClassVar[Variable] = Variable(
        "water_budget_residual",
        "water_budget_residual",
        "fx",
    )
    _atmospheric_energy_residual: ClassVar[Variable] = Variable(
        "atmospheric_energy_residual",
        "atmospheric_energy_residual",
        "fx",
    )
    _output_variables: ClassVar[tuple[Variable, ...]] = (
        _water_budget_residual,
        _atmospheric_energy_residual,
    )

    _other_data_generator = CMIP6PiControlR1I1P1F1
    _other_data_exp = "piControl"

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        data = complex_data_source.data["picontrol"]
        means = {
            variable.var_name: float(self._global_annual_mean(data, variable).mean())
            for variable in self._input_variables
        }
        toa_net = means["rsdt"] - means["rsut"] - means["rlut"]
        surface_net_radiation = (
            means["rsds"] - means["rsus"] + means["rlds"] - means["rlus"]
        )
        return self._scalar_cubes(
            {
                self._water_budget_residual: _physics.water_budget_residual(
                    means["pr"],
                    means["hfls"],
                ),
                self._atmospheric_energy_residual: (
                    _physics.atmospheric_energy_residual(
                        means["pr"],
                        toa_net,
                        surface_net_radiation,
                        means["hfss"],
                    )
                ),
            },
        )


class ClausiusClapeyronScaling(ScalarExperimentDiagnostic):
    """Manage Clausius-Clapeyron scaling diagnostics of a control run.

    Regression of the fractional global annual-mean water vapour path
    anomaly (per cent) on the global annual-mean surface air temperature
    anomaly (K). At roughly fixed relative humidity, Clausius-Clapeyron gives
    about 7 per cent K-1 (Held & Soden 2006); the interannual variability of
    a control run is enough to estimate it.
    """

    _required_data_keys = ("picontrol",)

    _input_variables: ClassVar[tuple[Variable, ...]] = (
        Variable("prw", "prw", "mon"),
        Variable("tas", "tas", "mon"),
    )

    _scaling: ClassVar[Variable] = Variable(
        "clausius_clapeyron_scaling",
        "clausius_clapeyron_scaling",
        "fx",
    )
    _output_variables: ClassVar[tuple[Variable, ...]] = (_scaling,)

    _other_data_generator = CMIP6PiControlR1I1P1F1
    _other_data_exp = "piControl"

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        data = complex_data_source.data["picontrol"]
        water_vapour_path = self._global_annual_mean(data, self._input_variables[0])
        temperature = self._global_annual_mean(data, self._input_variables[1])
        n_years = min(water_vapour_path.size, temperature.size)
        return self._scalar_cubes(
            {
                self._scaling: _physics.clausius_clapeyron_scaling(
                    water_vapour_path[:n_years],
                    temperature[:n_years],
                ),
            },
        )


class ClearSkyLongwaveFeedback(ScalarExperimentDiagnostic):
    """Manage clear-sky longwave feedback diagnostics.

    Clear-sky outgoing longwave radiation is tightly and quasi-linearly tied
    to surface temperature (Koll & Cronin 2018); CERES gives an area-mean
    local slope of about 2.2 W m-2 K-1. Here the slope of deseasonalised
    monthly ``rlutcs`` on ``ts`` is computed at every grid point and averaged
    with cosine-latitude weights. Monthly anomalies rather than annual means
    are used so the seasonal covariance that carries most of the signal is
    retained.
    """

    _required_data_keys = ("historical",)

    _input_variables: ClassVar[tuple[Variable, ...]] = (
        Variable("rlutcs", "rlutcs", "mon"),
        Variable("ts", "ts", "mon"),
    )

    _feedback: ClassVar[Variable] = Variable(
        "clear_sky_lw_feedback",
        "clear_sky_lw_feedback",
        "fx",
    )
    _output_variables: ClassVar[tuple[Variable, ...]] = (_feedback,)

    _other_data_generator = CMIP6HistoricalR1I1P1F1
    _other_data_exp = "historical"

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        data = complex_data_source.data["historical"]
        rlutcs = self._prepared_cube(data, self._input_variables[0])
        surface_temperature = self._prepared_cube(data, self._input_variables[1])
        with setup_esmvaltool_config_and_logging():
            rlutcs = anomalies(rlutcs, period="month")
            surface_temperature = anomalies(surface_temperature, period="month")
        n_time = min(rlutcs.shape[0], surface_temperature.shape[0])
        slopes = _physics.gridpoint_regression_slope(
            as_float_array(rlutcs.data[:n_time]),
            as_float_array(surface_temperature.data[:n_time]),
        )
        feedback = _physics.area_weighted_mean(
            slopes,
            rlutcs.coord("latitude").points.astype(float),
        )
        return self._scalar_cubes({self._feedback: feedback})


class PrecipitationBuoyancy(ScalarExperimentDiagnostic):
    """Manage tropical precipitation-buoyancy diagnostics.

    Convective coupling: tropical precipitation increases with column
    instability, so precipitation anomalies covary with anomalies of the
    column-integrated moist static energy (Neelin-type precipitation-buoyancy
    relations). The diagnostic regresses monthly precipitation anomalies
    (mm day-1) on anomalies of the mass-weighted column integral of
    ``h = c_p ta + g zg + L_v hus`` (MJ m-2) over the tropical band, pooling
    time and space.

    The column integral spans the pressure levels the monthly data actually
    carry (``plev19`` for CMIP6 Amon output), so it is a partial column; the
    optional observational reference is truncated the same way. That
    reference is the same statistic computed from GPCP precipitation and ERA5
    temperature and humidity over ``observation_timerange``, and is emitted
    as ``precip_buoyancy_slope_obs`` when those data can be obtained. ERA5 is
    served without geopotential height, so the reference column uses the
    hydrostatic height of its own temperature and humidity profiles
    (:func:`~climateeval.diags._physics.hydrostatic_height`); the missing
    surface term is constant in time and drops out of the anomalies.

    Additional keyword arguments
    ----------------------------
    latitude_band: tuple[float, float] (default: (-20.0, 20.0))
        Tropical band the regression is pooled over (degrees north).
    use_observations: bool (default: True)
        Compute the GPCP/ERA5 reference slope. With ``False``, or whenever
        the observations cannot be obtained and ``fail_on_missing_data`` is
        ``False``, only the model slope is emitted.
    observation_timerange: str (default: "19790101/20141231")
        Time range of the observational reference (GPCP starts in 1979).

    """

    _required_data_keys = ("historical",)

    _default_latitude_band: ClassVar[tuple[float, float]] = (-20.0, 20.0)
    _default_observation_timerange: ClassVar[str] = "19790101/20141231"
    _joules_per_megajoule: ClassVar[float] = 1.0e6

    _pr: ClassVar[Variable] = Variable("pr", "pr", "mon")
    _ta: ClassVar[Variable] = Variable("ta", "ta", "mon")
    _zg: ClassVar[Variable] = Variable("zg", "zg", "mon")
    _hus: ClassVar[Variable] = Variable("hus", "hus", "mon")
    _input_variables: ClassVar[tuple[Variable, ...]] = (_pr, _ta, _zg, _hus)

    _slope_variable: ClassVar[Variable] = Variable(
        "precip_buoyancy_slope",
        "precip_buoyancy_slope",
        "fx",
    )
    _slope_obs_variable: ClassVar[Variable] = Variable(
        "precip_buoyancy_slope_obs",
        "precip_buoyancy_slope_obs",
        "fx",
    )
    _output_variables: ClassVar[tuple[Variable, ...]] = (
        _slope_variable,
        _slope_obs_variable,
    )

    _other_data_generator = CMIP6HistoricalR1I1P1F1
    _other_data_exp = "historical"

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
        data = complex_data_source.data["historical"]
        slope = self._pooled_slope(
            *(self._prepared_cube(data, var) for var in self._input_variables),
        )
        values = {self._slope_variable: slope}
        observed_slope = self._observed_slope()
        if observed_slope is not None:
            values[self._slope_obs_variable] = observed_slope
        return self._scalar_cubes(values)

    def _pooled_slope(self, pr: Cube, ta: Cube, zg: Cube, hus: Cube) -> float:
        """Get the pooled slope of P' (mm day-1) on column MSE' (MJ m-2)."""
        pr, ta, zg, hus = (self._tropical_band(cube) for cube in (pr, ta, zg, hus))
        column, level_cube = self._column_moist_static_energy(ta, zg, hus)
        column_anomaly = (
            _physics.deseasonalised_anomalies(
                column,
                calendar_years_and_months(level_cube)[1],
            )
            / self._joules_per_megajoule
        )
        with setup_esmvaltool_config_and_logging():
            pr = anomalies(pr, period="month")
        precipitation_anomaly = as_float_array(pr.data) * _physics.SECONDS_PER_DAY
        n_time = min(precipitation_anomaly.shape[0], column_anomaly.shape[0])
        return _physics.pooled_regression_slope(
            precipitation_anomaly[:n_time],
            column_anomaly[:n_time],
        )

    def _tropical_band(self, cube: Cube) -> Cube:
        """Extract the tropical band the regression is pooled over."""
        with setup_esmvaltool_config_and_logging():
            return extract_region(
                cube,
                start_longitude=0.0,
                end_longitude=360.0,
                start_latitude=self._latitude_band[0],
                end_latitude=self._latitude_band[1],
            )

    @staticmethod
    def _column_moist_static_energy(
        ta: Cube,
        zg: Cube,
        hus: Cube,
    ) -> tuple[np.ndarray, Cube]:
        """Get the column-integrated MSE (J m-2) and the cube it came from.

        Raises
        ------
        ValueError
            If the three profiles share fewer than two pressure levels.

        """
        levels = sorted(
            set(ta.coord("air_pressure").points.astype(float))
            & set(zg.coord("air_pressure").points.astype(float))
            & set(hus.coord("air_pressure").points.astype(float)),
        )
        if len(levels) < _MIN_COLUMN_LEVELS:
            msg = (
                f"the column integral of the moist static energy needs at least "
                f"{_MIN_COLUMN_LEVELS} common pressure levels, got {levels}"
            )
            raise ValueError(msg)
        with setup_esmvaltool_config_and_logging():
            ta, zg, hus = (
                extract_levels(cube, levels, "linear") for cube in (ta, zg, hus)
            )
        moist_static_energy = _physics.moist_static_energy(
            as_float_array(ta.data),
            as_float_array(zg.data),
            as_float_array(hus.data),
        )
        column = _physics.mass_weighted_column_integral(
            moist_static_energy,
            np.asarray(levels, dtype=float),
            axis=ta.coord_dims("air_pressure")[0],
        )
        return column, ta

    def _observed_slope(self) -> float | None:
        """Get the GPCP/ERA5 reference slope, or ``None`` if unobtainable."""
        if not self._use_observations:
            return None
        try:
            return self._pooled_slope(*self._observation_cubes())
        except Exception as exc:
            if self._fail_on_missing_data:
                raise
            logger.warning(
                f"Diagnostic '{self.name}': no GPCP/ERA5 reference slope "
                f"({type(exc).__name__}: {exc}); emitting the model slope alone",
            )
            return None

    def _observation_cubes(self) -> tuple[Cube, Cube, Cube, Cube]:
        """Get the observed (pr, ta, zg, hus) cubes of the reference slope."""
        pr = self._observation_cube(GPCP, self._pr)
        ta = self._observation_cube(ERA5Monthly, self._ta)
        hus = self._observation_cube(ERA5Monthly, self._hus)
        try:
            zg = self._observation_cube(ERA5Monthly, self._zg)
        except Exception as exc:  # noqa: BLE001 - ERA5 is served without zg
            logger.warning(
                f"Diagnostic '{self.name}': no observed geopotential height "
                f"({type(exc).__name__}: {exc}); using the hydrostatic height of "
                f"the observed temperature and humidity profiles instead",
            )
            zg = ta.copy(
                _physics.hydrostatic_height(
                    as_float_array(ta.data),
                    as_float_array(hus.data),
                    ta.coord("air_pressure").points.astype(float),
                    axis=ta.coord_dims("air_pressure")[0],
                ),
            )
        return (pr, ta, zg, hus)

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
