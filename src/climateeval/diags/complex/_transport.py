"""Provide meridional heat transport diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from esmvalcore.preprocessor import (
    climate_statistics,
    regrid,
    zonal_statistics,
)
from iris.cube import Cube, CubeList
from loguru import logger

from climateeval import Variable
from climateeval._config import (
    setup_esmvaltool_config_and_logging,
    setup_iris_combine_policy,
)
from climateeval._utils import get_prepared_cube
from climateeval.data import (
    CMIP6PiControlR1I1P1F1,
    ComplexDataSource,
    DataSourceInformation,
)
from climateeval.data._esmvaltool_intake import filter_intake_data_sources
from climateeval.diags._base import DiagnosticOutput
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.complex._base import ComplexDiagnostic

if TYPE_CHECKING:
    from collections.abc import Generator

    from xarray import Dataset


logger = logger.opt(colors=True)

EARTH_RADIUS_M = 6.371e6


def implied_meridional_transport(
    zonal_mean_flux: np.ndarray,
    latitudes: np.ndarray,
) -> np.ndarray:
    """Implied northward energy transport from a zonal-mean flux profile.

    MET(phi) = 2 pi a^2 int_{-pi/2}^{phi} F(phi') cos(phi') dphi',
    integrated cumulatively from the southernmost latitude (Trenberth &
    Caron 2001).

    Parameters
    ----------
    zonal_mean_flux:
        Zonal-mean net downward flux profile (W m-2).
    latitudes:
        Latitudes (degrees), ascending or descending.

    Returns
    -------
    :
        Transport (W) at each latitude, in the input latitude order.

    """
    flux = np.asarray(zonal_mean_flux, dtype=float)
    lats = np.asarray(latitudes, dtype=float)
    order = np.argsort(lats)
    phi = np.deg2rad(lats[order])
    integrand = flux[order] * np.cos(phi) * np.gradient(phi)
    transport_sorted = 2.0 * np.pi * EARTH_RADIUS_M**2 * np.cumsum(integrand)
    out = np.empty_like(transport_sorted)
    out[order] = transport_sorted
    return out


class MeridionalHeatTransport(ComplexDiagnostic):
    """Manage meridional heat transport diagnostics.

    Partitions the implied poleward energy transport of a coupled control
    run into its atmospheric (AMET) and oceanic (OMET) components with the
    residual method: AMET from the atmospheric energy convergence
    (TOA net minus surface net flux), OMET from the net surface flux
    (Trenberth & Caron 2001). Outputs the Northern-Hemisphere peak
    magnitudes (PW) and latitudes of both components — observed benchmarks
    are an OMET peak of 1.5-2.0 PW near 15-20 degN and an AMET peak of 4-5 PW
    near 45 degN (ECCO/ERA5).

    Additional keyword arguments
    ----------------------------
    amet_search_band: tuple[float, float] (default: (25.0, 55.0))
        NH latitude band (degrees north) searched for the AMET peak.
    omet_search_band: tuple[float, float] (default: (5.0, 30.0))
        NH latitude band (degrees north) searched for the OMET peak.

    """

    _required_data_keys = ("picontrol",)

    _default_amet_search_band: ClassVar[tuple[float, float]] = (25.0, 55.0)
    _default_omet_search_band: ClassVar[tuple[float, float]] = (5.0, 30.0)

    # Input variables (piControl monthly means)
    _input_variables: ClassVar[tuple[Variable, ...]] = (
        Variable("rsdt", "rsdt", "mon"),
        Variable("rsut", "rsut", "mon"),
        Variable("rlut", "rlut", "mon"),
        Variable("rsds", "rsds", "mon"),
        Variable("rsus", "rsus", "mon"),
        Variable("rlds", "rlds", "mon"),
        Variable("rlus", "rlus", "mon"),
        Variable("hfss", "hfss", "mon"),
        Variable("hfls", "hfls", "mon"),
    )

    # Output variables
    _amet_peak: ClassVar[Variable] = Variable("amet_peak", "amet_peak", "fx")
    _amet_peak_lat: ClassVar[Variable] = Variable(
        "amet_peak_lat",
        "amet_peak_lat",
        "fx",
    )
    _omet_peak: ClassVar[Variable] = Variable("omet_peak", "omet_peak", "fx")
    _omet_peak_lat: ClassVar[Variable] = Variable(
        "omet_peak_lat",
        "omet_peak_lat",
        "fx",
    )

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize class instance."""
        super().__init__(name, **kwargs)
        self._amet_search_band: tuple[float, float] = tuple(
            self._kwargs.get("amet_search_band", self._default_amet_search_band),
        )
        self._omet_search_band: tuple[float, float] = tuple(
            self._kwargs.get("omet_search_band", self._default_omet_search_band),
        )

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        data = complex_data_source.data["picontrol"]
        profiles: dict[str, np.ndarray] = {}
        latitudes = np.array([])
        for variable in self._input_variables:
            cube = self._get_zonal_mean(data, variable)
            profiles[variable.var_name] = np.asarray(cube.data, dtype=float)
            latitudes = cube.coord("latitude").points.astype(float)

        toa_net = profiles["rsdt"] - profiles["rsut"] - profiles["rlut"]
        sfc_net = (
            profiles["rsds"]
            - profiles["rsus"]
            + profiles["rlds"]
            - profiles["rlus"]
            - profiles["hfss"]
            - profiles["hfls"]
        )
        amet = implied_meridional_transport(toa_net - sfc_net, latitudes) / 1e15
        omet = implied_meridional_transport(sfc_net, latitudes) / 1e15

        amet_peak, amet_lat = self._nh_peak(amet, latitudes, *self._amet_search_band)
        omet_peak, omet_lat = self._nh_peak(omet, latitudes, *self._omet_search_band)

        return {
            self._amet_peak: Cube(amet_peak, var_name="amet_peak"),
            self._amet_peak_lat: Cube(amet_lat, var_name="amet_peak_lat"),
            self._omet_peak: Cube(omet_peak, var_name="omet_peak"),
            self._omet_peak_lat: Cube(omet_lat, var_name="omet_peak_lat"),
        }

    def _get_zonal_mean(self, data: CubeList | Dataset, variable: Variable) -> Cube:
        """Get time-mean zonal-mean profile of a variable."""
        cube = get_prepared_cube(data, variable)
        with setup_esmvaltool_config_and_logging():
            cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
            cube = climate_statistics(cube, "mean", "full")
            cube = zonal_statistics(cube, "mean")
        return cube  # noqa: RET504 - assignments keep the pipeline readable

    @staticmethod
    def _nh_peak(
        profile: np.ndarray,
        latitudes: np.ndarray,
        lat_min: float,
        lat_max: float,
    ) -> tuple[float, float]:
        """Get (peak value, peak latitude) within a NH latitude band.

        The peak is refined to sub-gridscale precision by fitting a parabola
        through the band's argmax and its two neighbours and returning the
        vertex (three-point parabolic interpolation). On the common 2° grid
        the plain argmax quantises the reported latitude to grid lines, so
        models whose true peaks differ by a fraction of a cell were reported
        as identical, or on opposite sides of a bound that falls between two
        grid lines. Falls back to the grid-cell value when the argmax sits at
        an edge of the band or the three points are flat/collinear.
        """
        mask = (latitudes >= lat_min) & (latitudes <= lat_max)
        if not mask.any():
            return float("nan"), float("nan")
        band_profile = np.asarray(profile, dtype=float)[mask]
        band_lats = np.asarray(latitudes, dtype=float)[mask]
        i = int(np.nanargmax(band_profile))
        peak_value, peak_lat = float(band_profile[i]), float(band_lats[i])
        if i == 0 or i == band_profile.size - 1:
            return peak_value, peak_lat
        y0, y1, y2 = band_profile[i - 1], band_profile[i], band_profile[i + 1]
        denom = y0 - 2.0 * y1 + y2
        if not np.isfinite(denom) or denom == 0.0:
            return peak_value, peak_lat
        half_step = 0.5 * (band_lats[i + 1] - band_lats[i - 1])
        delta = 0.5 * (y0 - y2) / denom
        return (
            float(y1 - 0.25 * (y0 - y2) * delta),
            float(peak_lat + delta * half_step),
        )

    def _get_other_data(self) -> Generator[ComplexDataSource]:
        """Get the CMIP6 r1i1p1f1 piControl reference ensemble (as in ECS)."""
        logger.debug(f"Retrieving other data for complex diagnostic '{self.name}'")
        variables = self._input_variables
        generators = [CMIP6PiControlR1I1P1F1().generate(var) for var in variables]
        grouped_data_sources = filter_intake_data_sources(
            *generators,
            group_by="dataset",
            required_facets=tuple(
                {"exp": "piControl", "short_name": var.var_name} for var in variables
            ),
        )
        for dataset, data_sources in grouped_data_sources.items():
            logger.debug(f"Found other data of CMIP6 model {dataset}")
            data_source = next(
                d
                for d in data_sources
                if d.__class__.__name__ == "CMIP6PiControlR1I1P1F1DataSource"
            )
            data_information = DataSourceInformation(
                name=dataset,
                category=data_source.information.category,
                institute=data_source.information.institute,
                exp="piControl",
                variant=data_source.information.variant,
                references=data_source.information.references,
            )
            with setup_iris_combine_policy():
                try:
                    data = {
                        "picontrol": CubeList(
                            data_source.get_cube(
                                self.data_root_dir,
                                var,
                                download_missing_data=self._download_missing_data,
                            )
                            for var in variables
                        ),
                    }
                except Exception as exc:  # noqa: BLE001
                    self._handle_missing_data(data_source.id, variables, exc)
                    continue
            yield ComplexDataSource(data=data, information=data_information)

    def _postprocess_output(self, output: DiagnosticOutput) -> DiagnosticOutput:
        """Postprocess diagnostic output."""
        if output.raw_output is None:
            return output
        unique_columns = (self._data_id_column, self._data_type_column)
        variables = (
            self._amet_peak.id,
            self._amet_peak_lat.id,
            self._omet_peak.id,
            self._omet_peak_lat.id,
        )
        raw_output = self._remove_superfluous_rows(
            output.raw_output,
            unique_columns=unique_columns,
            variable_columns=variables,
        )
        raw_output = raw_output.select((*unique_columns, *variables))
        return DiagnosticOutput(
            raw_output=raw_output,
            metrics=output.metrics,
            variables=output.variables,
            data_sources=output.data_sources,
        )
