"""Provide forced-response diagnostics (abrupt-4xCO2 vs piControl)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from esmvalcore.preprocessor import (
    annual_statistics,
    area_statistics,
    extract_region,
    mask_landsea,
    regrid,
)
from iris.cube import Cube

from climateeval import Variable
from climateeval._config import (
    setup_esmvaltool_config_and_logging,
)
from climateeval._utils import get_prepared_cube
from climateeval.diags._base import DiagnosticOutput
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.complex._base import ComplexDiagnostic

if TYPE_CHECKING:
    from iris.cube import CubeList
    from xarray import Dataset

    from climateeval.data import ComplexDataSource


class _ForcedResponseDiagnostic(ComplexDiagnostic):
    """Manage forced-response diagnostics.

    Shared machinery for diagnostics comparing the equilibrium response of
    an ``abrupt-4xCO2`` run (mean over the last ``equilibrium_years``,
    default 50) with the full piControl mean over a spatial domain.

    Additional keyword arguments
    ----------------------------
    equilibrium_years: int (default: 50)
        Number of final abrupt-4xCO2 years averaged as the equilibrium
        state.

    """

    _required_data_keys = ("4xco2", "picontrol")

    _tas: ClassVar[Variable] = Variable("tas", "tas", "mon")

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize class instance."""
        super().__init__(name, **kwargs)
        self._equilibrium_years: int = int(self._kwargs.get("equilibrium_years", 50))

    def _domain_mean_anomaly(
        self,
        complex_data_source: ComplexDataSource,
        *,
        region: tuple[float, float] | None = None,
        mask_out: str | None = None,
    ) -> float:
        """Get equilibrium abrupt-4xCO2 minus piControl domain-mean tas."""

        def domain_series(data: CubeList | Dataset) -> np.ndarray:
            cube = get_prepared_cube(data, self._tas)
            with setup_esmvaltool_config_and_logging():
                cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
                if region is not None:
                    cube = extract_region(
                        cube,
                        start_longitude=0.0,
                        end_longitude=360.0,
                        start_latitude=region[0],
                        end_latitude=region[1],
                    )
                if mask_out is not None:
                    cube = mask_landsea(cube, mask_out)
                cube = area_statistics(cube, "mean")
                cube = annual_statistics(cube, "mean")
            return np.asarray(cube.data, dtype=float)

        response = domain_series(complex_data_source.data["4xco2"])
        control = domain_series(complex_data_source.data["picontrol"])
        return float(
            response[-self._equilibrium_years :].mean() - control.mean(),
        )

    def _select_output(
        self,
        output: DiagnosticOutput,
        variables: tuple[str, ...],
    ) -> DiagnosticOutput:
        """Drop superfluous rows and order the output columns."""
        if output.raw_output is None:
            return output
        unique_columns = (self._data_id_column, self._data_type_column)
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


class LandOceanWarmingRatio(_ForcedResponseDiagnostic):
    """Manage land-ocean warming ratio diagnostics.

    Land must warm faster than ocean under greenhouse forcing (thermal
    inertia and lapse-rate/moisture constraints; Sutton et al. 2007, Joshi
    et al. 2008). The ratio dT_land / dT_ocean of the equilibrium
    abrupt-4xCO2 response is > 1 for any physically plausible model, with
    an expected range of roughly 1.2-1.6.
    """

    _ratio: ClassVar[Variable] = Variable(
        "land_ocean_warming_ratio",
        "land_ocean_warming_ratio",
        "fx",
    )
    _delta_t_land: ClassVar[Variable] = Variable(
        "delta_t_land",
        "delta_t_land",
        "fx",
    )
    _delta_t_ocean: ClassVar[Variable] = Variable(
        "delta_t_ocean",
        "delta_t_ocean",
        "fx",
    )

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        delta_t_land = self._domain_mean_anomaly(complex_data_source, mask_out="sea")
        delta_t_ocean = self._domain_mean_anomaly(complex_data_source, mask_out="land")
        return {
            self._ratio: Cube(
                delta_t_land / delta_t_ocean,
                var_name="land_ocean_warming_ratio",
            ),
            self._delta_t_land: Cube(delta_t_land, var_name="delta_t_land"),
            self._delta_t_ocean: Cube(delta_t_ocean, var_name="delta_t_ocean"),
        }

    def _postprocess_output(self, output: DiagnosticOutput) -> DiagnosticOutput:
        """Postprocess diagnostic output."""
        return self._select_output(
            output,
            (self._ratio.id, self._delta_t_land.id, self._delta_t_ocean.id),
        )


class ArcticAmplification(_ForcedResponseDiagnostic):
    """Manage Arctic amplification diagnostics.

    Polar amplification from ice-albedo and lapse-rate feedbacks and
    poleward energy transport (Pithan & Mauritsen 2014): the equilibrium
    Arctic (> 66.5 degN) warming of an abrupt-4xCO2 run exceeds the global
    mean by a factor of at least ~1.5 in observations and CMIP models.

    Additional keyword arguments
    ----------------------------
    arctic_latitude: float (default: 66.5)
        Southern boundary of the Arctic domain (degrees north).

    """

    _ratio: ClassVar[Variable] = Variable(
        "arctic_amplification",
        "arctic_amplification",
        "fx",
    )
    _delta_t_arctic: ClassVar[Variable] = Variable(
        "delta_t_arctic",
        "delta_t_arctic",
        "fx",
    )
    _delta_t_global: ClassVar[Variable] = Variable(
        "delta_t_global",
        "delta_t_global",
        "fx",
    )

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize class instance."""
        super().__init__(name, **kwargs)
        self._arctic_latitude: float = float(
            self._kwargs.get("arctic_latitude", 66.5),
        )

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        delta_t_arctic = self._domain_mean_anomaly(
            complex_data_source,
            region=(self._arctic_latitude, 90.0),
        )
        delta_t_global = self._domain_mean_anomaly(complex_data_source)
        return {
            self._ratio: Cube(
                delta_t_arctic / delta_t_global,
                var_name="arctic_amplification",
            ),
            self._delta_t_arctic: Cube(delta_t_arctic, var_name="delta_t_arctic"),
            self._delta_t_global: Cube(delta_t_global, var_name="delta_t_global"),
        }

    def _postprocess_output(self, output: DiagnosticOutput) -> DiagnosticOutput:
        """Postprocess diagnostic output."""
        return self._select_output(
            output,
            (self._ratio.id, self._delta_t_arctic.id, self._delta_t_global.id),
        )
