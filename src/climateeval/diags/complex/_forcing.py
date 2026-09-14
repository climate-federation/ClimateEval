"""Provide effective radiative forcing and pattern-effect diagnostics.

The forcing a model actually realises, from the experiments designed to
isolate it: the fixed-SST forcing of quadrupled CO2 (``amip-4xCO2``), the
dependence of the radiative feedback on where the ocean warms (the GFMIP
SST patch experiments), and the aerosol forcing of the DAMIP aerosol-only
historical run (``hist-aer``).

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from loguru import logger
from scipy import stats

from climateeval import Variable
from climateeval.diags.complex._branch_time import (
    branch_year_in_parent,
    parallel_control_window,
)
from climateeval.diags.complex._scalar import ScalarExperimentDiagnostic

if TYPE_CHECKING:
    from iris.cube import Cube, CubeList
    from xarray import Dataset

    from climateeval.data import ComplexDataSource

logger = logger.opt(colors=True)

_TOA_VARIABLES: tuple[Variable, ...] = (
    Variable("rsdt", "rsdt", "mon"),
    Variable("rsut", "rsut", "mon"),
    Variable("rlut", "rlut", "mon"),
)
"""Fluxes the TOA net downward flux is assembled from."""


def clip_window_to_record(
    window: tuple[int, int],
    first_year: int,
    last_year: int,
) -> tuple[int, int]:
    """Fit a fixed calendar window into the record actually supplied.

    The window keeps its length and slides back so that it ends no later
    than the record does; a record shorter than the window is used whole.
    DAMIP ``hist-aer`` runs, for instance, often stop in 2014, before the
    end of a window centred on 2015.

    Parameters
    ----------
    window:
        Requested inclusive ``(first, last)`` calendar years.
    first_year:
        First year of the record.
    last_year:
        Last year of the record.

    Returns
    -------
    :
        Inclusive ``(first, last)`` calendar years actually usable.

    Raises
    ------
    ValueError
        If the window or the record is empty.

    """
    requested_first, requested_last = int(window[0]), int(window[1])
    length = requested_last - requested_first + 1
    if length < 1:
        msg = f"empty window {window}"
        raise ValueError(msg)
    first_year, last_year = int(first_year), int(last_year)
    if last_year < first_year:
        msg = f"empty record {first_year}-{last_year}"
        raise ValueError(msg)
    if last_year - first_year + 1 <= length:
        return (first_year, last_year)
    last = min(requested_last, last_year)
    first = last - length + 1
    if first < first_year:
        first = first_year
        last = first + length - 1
    return (first, last)


class _ForcingDiagnostic(ScalarExperimentDiagnostic):
    """Manage diagnostics of the TOA response to a prescribed perturbation.

    All of them contrast the global-mean TOA net flux of two or more
    experiments, so none has a single data key and none has a CMIP6
    comparison ensemble: the archive holds no ``r1i1p1f1`` ensemble of the
    fixed-SST and patch experiments.
    """

    def _toa_net_time_mean(self, data: CubeList | Dataset) -> float:
        """Get the time mean of the global annual-mean TOA net flux."""
        return float(self._toa_net_global_annual_mean(data).mean())


class Amip4xCO2ERF(_ForcingDiagnostic):
    """Manage fixed-SST effective radiative forcing diagnostics.

    The effective radiative forcing of quadrupled CO2, as the difference in
    global-mean TOA net downward flux between the fixed-SST ``amip-4xCO2``
    and ``amip`` experiments. Holding the sea surface fixed removes the
    surface-temperature-mediated feedbacks, so what remains is the forcing
    including rapid adjustments; CMIP6 models give roughly 6.5-9 W m-2. No
    land-warming correction is applied.

    This is a different quantity from the ``f4x`` intercept that
    :class:`~climateeval.diags.complex.ECS` obtains from the Gregory
    regression of the coupled abrupt-4xCO2 run.
    """

    _required_data_keys = ("amip", "amip4xco2")

    _input_variables: ClassVar[tuple[Variable, ...]] = _TOA_VARIABLES

    _erf: ClassVar[Variable] = Variable("amip_4xco2_erf", "amip_4xco2_erf", "fx")
    _output_variables: ClassVar[tuple[Variable, ...]] = (_erf,)

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        data = complex_data_source.data
        forcing = self._toa_net_time_mean(data["amip4xco2"]) - self._toa_net_time_mean(
            data["amip"],
        )
        return self._scalar_cubes({self._erf: forcing})


class GFMIPPatchFeedback(_ForcingDiagnostic):
    """Manage SST pattern-effect diagnostics.

    The radiative feedback depends on *where* the ocean warms: warming the
    east Pacific destabilises the tropical circulation and gives a more
    negative feedback than warming the west Pacific warm pool. Following the
    GFMIP protocol, the diagnostic takes two fixed-SST experiments in which
    a +1 K patch is added over the east and the west Pacific, forms the
    feedback of each against the unperturbed ``amip`` run,

    ``lambda_x = (R_x - R_amip) / (Ts_x - Ts_amip)``

    from global, time-mean TOA net fluxes and surface temperatures, and
    returns the difference ``lambda_EP - lambda_WP`` alongside both
    feedbacks.

    The patch experiments are not part of the CMIP6 protocol, so they have
    to be supplied by the modelling group.
    """

    _required_data_keys = ("amip", "patch_ep", "patch_wp")

    _ts: ClassVar[Variable] = Variable("ts", "ts", "mon")
    _input_variables: ClassVar[tuple[Variable, ...]] = (*_TOA_VARIABLES, _ts)

    _difference: ClassVar[Variable] = Variable(
        "patch_feedback_difference",
        "patch_feedback_difference",
        "fx",
    )
    _east_pacific: ClassVar[Variable] = Variable(
        "patch_feedback_ep",
        "patch_feedback_ep",
        "fx",
    )
    _west_pacific: ClassVar[Variable] = Variable(
        "patch_feedback_wp",
        "patch_feedback_wp",
        "fx",
    )
    _output_variables: ClassVar[tuple[Variable, ...]] = (
        _difference,
        _east_pacific,
        _west_pacific,
    )

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        data = complex_data_source.data
        east = self._patch_feedback(data["patch_ep"], data["amip"])
        west = self._patch_feedback(data["patch_wp"], data["amip"])
        return self._scalar_cubes(
            {
                self._difference: east - west,
                self._east_pacific: east,
                self._west_pacific: west,
            },
        )

    def _patch_feedback(
        self,
        patch: CubeList | Dataset,
        reference: CubeList | Dataset,
    ) -> float:
        """Get the feedback of one patch experiment against the amip run."""
        delta_flux = self._toa_net_time_mean(patch) - self._toa_net_time_mean(reference)
        delta_temperature = float(
            self._global_annual_mean(patch, self._ts).mean()
            - self._global_annual_mean(reference, self._ts).mean(),
        )
        return delta_flux / delta_temperature


class AerosolERF(_ForcingDiagnostic):
    """Manage aerosol effective radiative forcing diagnostics.

    The aerosol-only historical run of DAMIP (``hist-aer``) isolates the
    response to anthropogenic aerosols. Its forcing follows from the
    energy-balance inversion ``ERF = dN - lambda dT`` (Forster et al. 2021)
    with ``lambda`` the model's own Gregory feedback from its abrupt-4xCO2
    run (negative, hence the minus sign), and ``dN`` and ``dT`` the
    global-mean TOA net flux and surface air temperature anomalies over
    ``evaluation_window``. Present-day aerosol forcing is negative, of order
    -1 W m-2, and the accompanying temperature anomaly is a cooling.

    The anomalies are taken against the **parallel segment** of the control
    run - the piControl years running alongside the evaluation window,
    located through the CMIP6 branch-time attributes
    (:mod:`~climateeval.diags.complex._branch_time`) - so that the control's
    own drift cancels. When the branch metadata is absent, or the implied
    segment falls outside the control record, the control's long-term mean
    is used instead and a warning is logged; ``aerosol_parallel_segment``
    records which of the two was used.

    Additional keyword arguments
    ----------------------------
    evaluation_window: tuple[int, int] (default: (2010, 2019))
        Inclusive calendar years the anomalies are averaged over. Many
        ``hist-aer`` runs stop in 2014, in which case the window keeps its
        length and slides back to the end of the record; the years actually
        used are reported as ``aerosol_window_first_year`` and
        ``aerosol_window_last_year``.
    gregory_years: int (default: 150)
        Number of abrupt-4xCO2 years the Gregory regression uses.

    """

    _required_data_keys = ("picontrol", "4xco2", "histaer")

    _default_evaluation_window: ClassVar[tuple[int, int]] = (2010, 2019)
    _default_gregory_years: ClassVar[int] = 150

    _tas: ClassVar[Variable] = Variable("tas", "tas", "mon")
    _input_variables: ClassVar[tuple[Variable, ...]] = (*_TOA_VARIABLES, _tas)

    _erf: ClassVar[Variable] = Variable("aerosol_erf", "aerosol_erf", "fx")
    _delta_flux: ClassVar[Variable] = Variable(
        "aerosol_delta_n",
        "aerosol_delta_n",
        "fx",
    )
    _delta_temperature: ClassVar[Variable] = Variable(
        "aerosol_delta_t",
        "aerosol_delta_t",
        "fx",
    )
    _feedback: ClassVar[Variable] = Variable(
        "aerosol_lambda",
        "aerosol_lambda",
        "fx",
    )
    _window_first_year: ClassVar[Variable] = Variable(
        "aerosol_window_first_year",
        "aerosol_window_first_year",
        "fx",
    )
    _window_last_year: ClassVar[Variable] = Variable(
        "aerosol_window_last_year",
        "aerosol_window_last_year",
        "fx",
    )
    _parallel_segment: ClassVar[Variable] = Variable(
        "aerosol_parallel_segment",
        "aerosol_parallel_segment",
        "fx",
    )
    _output_variables: ClassVar[tuple[Variable, ...]] = (
        _erf,
        _delta_flux,
        _delta_temperature,
        _feedback,
        _window_first_year,
        _window_last_year,
        _parallel_segment,
    )

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize class instance."""
        super().__init__(name, **kwargs)
        self._evaluation_window: tuple[int, int] = tuple(
            self._kwargs.get("evaluation_window", self._default_evaluation_window),
        )
        self._gregory_years: int = int(
            self._kwargs.get("gregory_years", self._default_gregory_years),
        )

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        data = complex_data_source.data
        control_temperature, control_temperature_years = (
            self._global_annual_mean_and_years(data["picontrol"], self._tas)
        )
        control_flux, control_flux_years = self._toa_net_global_annual_mean_and_years(
            data["picontrol"],
        )
        feedback = self._gregory_feedback(
            data["4xco2"],
            control_temperature.mean(),
            control_flux.mean(),
        )

        temperature, flux, years = self._aligned_series(data["histaer"])
        first, last = clip_window_to_record(
            self._evaluation_window,
            int(years[0]),
            int(years[-1]),
        )
        if (first, last) != tuple(self._evaluation_window):
            logger.warning(
                f"Diagnostic '{self.name}': the aerosol-only run covers "
                f"{years[0]}-{years[-1]}, so the evaluation window "
                f"{self._evaluation_window[0]}-{self._evaluation_window[1]} is "
                f"shifted to {first}-{last}",
            )
        in_window = (years >= first) & (years <= last)

        baseline_temperature, baseline_flux, is_parallel = self._control_baseline(
            data["histaer"],
            (first, last),
            int(years[0]),
            (control_temperature, control_temperature_years),
            (control_flux, control_flux_years),
        )
        delta_temperature = float(temperature[in_window].mean() - baseline_temperature)
        delta_flux = float(flux[in_window].mean() - baseline_flux)

        return self._scalar_cubes(
            {
                self._erf: delta_flux - feedback * delta_temperature,
                self._delta_flux: delta_flux,
                self._delta_temperature: delta_temperature,
                self._feedback: feedback,
                self._window_first_year: float(first),
                self._window_last_year: float(last),
                self._parallel_segment: float(is_parallel),
            },
        )

    def _aligned_series(
        self,
        data: CubeList | Dataset,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get the temperature and TOA net flux series on their common years."""
        temperature, temperature_years = self._global_annual_mean_and_years(
            data,
            self._tas,
        )
        flux, flux_years = self._toa_net_global_annual_mean_and_years(data)
        years = np.intersect1d(temperature_years, flux_years)
        return (
            temperature[np.isin(temperature_years, years)],
            flux[np.isin(flux_years, years)],
            years,
        )

    def _gregory_feedback(
        self,
        data: CubeList | Dataset,
        control_temperature: float,
        control_flux: float,
    ) -> float:
        """Get the Gregory feedback of the abrupt-4xCO2 run (W m-2 K-1).

        The slope of the TOA net flux anomaly against the surface warming of
        the first ``gregory_years`` years, as in
        :class:`~climateeval.diags.complex.ECS`. Only the slope is used, so
        the choice of control baseline does not matter here.
        """
        temperature = self._global_annual_mean(data, self._tas)
        flux = self._toa_net_global_annual_mean(data)
        n_years = min(temperature.size, flux.size, self._gregory_years)
        regression = stats.linregress(
            temperature[:n_years] - control_temperature,
            flux[:n_years] - control_flux,
        )
        return float(regression.slope)

    def _control_baseline(
        self,
        child: CubeList | Dataset,
        window: tuple[int, int],
        child_first_year: int,
        control_temperature: tuple[np.ndarray, np.ndarray],
        control_flux: tuple[np.ndarray, np.ndarray],
    ) -> tuple[float, float, bool]:
        """Get the control means the aerosol anomalies are taken against.

        The parallel segment when the branch metadata allows it, the
        long-term control mean otherwise; the third element says which.
        """
        temperature, temperature_years = control_temperature
        flux, flux_years = control_flux
        branch_year = branch_year_in_parent(child, context=self.name)
        if branch_year is not None:
            first, last = parallel_control_window(
                window,
                branch_year=branch_year,
                child_first_year=child_first_year,
            )
            in_segment_temperature = (temperature_years >= first) & (
                temperature_years <= last
            )
            in_segment_flux = (flux_years >= first) & (flux_years <= last)
            if in_segment_temperature.any() and in_segment_flux.any():
                logger.info(
                    f"Diagnostic '{self.name}': removing the control drift with "
                    f"the parallel piControl segment {first}-{last} "
                    f"({int(in_segment_temperature.sum())} yr; branch year "
                    f"{branch_year})",
                )
                return (
                    float(temperature[in_segment_temperature].mean()),
                    float(flux[in_segment_flux].mean()),
                    True,
                )
            logger.warning(
                f"Diagnostic '{self.name}': the parallel piControl segment "
                f"{first}-{last} lies outside the control record "
                f"{temperature_years[0]}-{temperature_years[-1]}",
            )
        logger.warning(
            f"Diagnostic '{self.name}': taking the aerosol anomalies against the "
            f"long-term piControl mean, so the control drift is not removed",
        )
        return float(temperature.mean()), float(flux.mean()), False
