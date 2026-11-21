"""Provide seasonal chlorophyll-driver covariability diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from esmvalcore.preprocessor import extract_time, regrid
from iris.coords import AuxCoord
from iris.cube import Cube
from loguru import logger
from xarray import DataArray

from climateeval import Variable
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._utils import get_prepared_cube
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.complex import _covariability
from climateeval.diags.complex._base import ComplexDiagnostic
from climateeval.diags.complex._covariability import (
    PERIOD,
    REGIONS,
    metrics_from_fields,
)

if TYPE_CHECKING:
    from pathlib import Path

    from iris.cube import CubeList
    from xarray import Dataset

    from climateeval.data import ComplexDataSource

logger = logger.opt(colors=True)

# The extracted analysis code reports progress through its own `log`,
# which writes to stdout because it came from a standalone script. Send
# it to the logger instead: a library should not print.
_covariability.log = logger.debug

#: Metrics returned by
#: :func:`climateeval.diags.complex._covariability.metrics_from_fields`,
#: in the order they are reported. Every one becomes an output variable
#: over the ``basin`` coordinate. The names are those of the published
#: table, so renaming one changes a published metric.
#:
#: ``cov_<driver>_lag`` is the lag in months at which chlorophyll and
#: the driver covary most strongly, ``cov_<driver>_r`` the correlation
#: there, and ``cov_<driver>_r0`` the correlation at zero lag.
METRICS: tuple[str, ...] = (
    # seasonal cycle of chlorophyll
    "bloom_phase",
    "bloom_peak",
    "bloom_phase_sd",
    "A1",
    "A2",
    "A1_rel",
    "A2_over_A1",
    "A2_over_A1_regmean",
    "bimodal_frac",
    "two_peak_frac",
    "cells_frac",
    "chl_mean",
    "n_months_chl",
    "n_years_phase",
    # seasonal cycle of the physical drivers
    "mld_max_month",
    "restrat_month",
    "ice_retreat_month",
    "ice_area_frac",
    # bloom timing relative to the drivers
    "lag_bloom_restrat",
    "lag_bloom_ice",
    # covariability with each driver
    "cov_mld_lag",
    "cov_mld_r",
    "cov_mld_r0",
    "cov_sst_lag",
    "cov_sst_r",
    "cov_sst_r0",
    "cov_ice_lag",
    "cov_ice_r",
    "cov_ice_r0",
)


class ChlDriverCovariability(ComplexDiagnostic):
    """Manage seasonal chlorophyll-driver covariability diagnostics.

    Compares the seasonal cycle of surface chlorophyll with the seasonal
    cycles of the physical properties that drive it -- mixed-layer depth,
    sea surface temperature and sea-ice concentration -- and reports, per
    ocean basin, when the bloom peaks, when the water column
    restratifies, when sea ice retreats, and the lags between them.

    The motivation is that a model can reproduce the amplitude of the
    seasonal chlorophyll cycle and still get its timing wrong, and the
    timing is what ties the biology to the physics. Where the physical
    driver is well simulated and the bloom still peaks in the wrong
    month, the error is biological rather than physical, which the
    separate ``restrat_month`` and ``bloom_phase`` outputs make visible.

    The analysis itself lives in
    :mod:`climateeval.diags.complex._covariability` and this class only
    adapts cubes to it. That division is deliberate: those routines
    produced the published metrics and are kept verbatim, and three of
    their properties are easy to lose in a reimplementation. Months
    wrap, so phases are averaged circularly and never arithmetically.
    For a near-sinusoidal pair a lag of L and one of L+6 fit equally
    well with opposite signs, so each driver's expected sign is declared
    rather than inferred. And two metrics -- the second-harmonic ratio
    and the ice-retreat month -- are computed cell by cell and only then
    averaged over a basin, because a basin mean cannot distinguish a
    real autumn bloom from single blooms peaking at different times in
    different places, nor a retreat date in a band that is partly
    perennial ice.

    Additional keyword arguments
    ----------------------------
    period: tuple[int, int] (default: ``_covariability.PERIOD``)
        First and last calendar year to evaluate.
    regions: dict (default: all of ``_covariability.REGIONS``)
        Mapping of basin name to a latitude/longitude selection. Pass a
        subset to restrict the output.
    grid: str (default: ``climateeval.diags._utils.DEFAULT_GRID``)
        Target grid for the regridding. The published metrics used
        ``"1x1"``; pass that to reproduce them exactly.
    scales: dict (default: None)
        Per-variable unit factor, overriding the unit handling in
        ``FIELD_SPEC``. Only needed when the fields have already been
        normalised before they arrive, which is not the case for cubes
        from a data source: chlorophyll comes in kg m-3 and is converted
        from its units attribute. Pass ``{"chl": 1.0}`` for a field
        already in mg m-3, or it is multiplied by a million.

    """

    _required_data_keys = ("historical",)

    #: Input variables. Chlorophyll is required; a driver that is absent
    #: leaves the metrics depending on it missing rather than dropping
    #: the dataset, because several CMIP6 models archive no mixed-layer
    #: depth and the bloom phase is still informative without it.
    _chl: ClassVar[Variable] = Variable("chl", "chl", "mon")
    _mlotst: ClassVar[Variable] = Variable("mlotst", "mlotst", "mon")
    _siconc: ClassVar[Variable] = Variable("siconc", "siconc", "mon")
    _tos: ClassVar[Variable] = Variable("tos", "tos", "mon")

    #: Field name used by ``FIELD_SPEC`` -> input variable.
    _FIELDS: ClassVar[dict[str, str]] = {
        "chl": "_chl",
        "mlotst": "_mlotst",
        "siconc": "_siconc",
        "tos": "_tos",
    }

    #: Output variables, one per metric, each over ``basin``. Built from
    #: METRICS rather than written out one by one: there are 29 of them
    #: and a hand-maintained list would drift from what
    #: ``metrics_from_fields`` actually returns. ``_calculate_raw_output``
    #: checks the two still agree on every call.
    _outputs: ClassVar[dict[str, Variable]] = {
        metric: Variable(metric, metric, "fx") for metric in METRICS
    }

    def __init__(
        self,
        name: str,
        *,
        data_root_dir: Path | str | None = None,
        fail_on_missing_data: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize class instance."""
        super().__init__(
            name,
            data_root_dir=data_root_dir,
            fail_on_missing_data=fail_on_missing_data,
            **kwargs,
        )
        period = self._kwargs.get("period", PERIOD)
        self._period: tuple[int, int] = (int(period[0]), int(period[1]))
        self._regions: dict = self._kwargs.get("regions", REGIONS)
        # The default is ClimateEval's 2x2 so this diagnostic behaves
        # like the others, but the grid is settable so a run can be
        # compared like-for-like against the published table, which used
        # 1x1 -- otherwise a regridding difference and a bug look alike.
        self._grid: str = str(self._kwargs.get("grid", DEFAULT_GRID))
        self._scales: dict | None = self._kwargs.get("scales")
        logger.debug(
            f"Evaluating {len(self._regions)} basin(s) over "
            f"{self._period[0]}-{self._period[1]} on the {self._grid} grid",
        )

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        data = complex_data_source.data["historical"]
        name = complex_data_source.information.name

        fields = {
            field: self._get_field(
                data, getattr(self, attribute), required=(field == "chl"),
            )
            for field, attribute in self._FIELDS.items()
        }

        # metrics_from_fields is the seam this diagnostic exists to feed:
        # it builds the per-cell climatologies for chlorophyll and sea
        # ice, merges the metrics that only exist per cell, and derives
        # the bloom-to-ice lag from the gridded retreat month rather than
        # from a basin mean. Reproducing any of that here would be a
        # second implementation of the published method.
        metrics_by_basin = metrics_from_fields(
            fields,
            self._regions,
            self._period,
            name=name,
            scales=self._scales,
        )

        self._check_metrics(metrics_by_basin, name)
        return self._as_cubes(metrics_by_basin)

    def _as_cubes(
        self,
        metrics_by_basin: dict[str, dict[str, float]],
    ) -> dict[Variable, Cube]:
        """Assemble per-basin metrics into one cube per output variable."""
        basins = list(metrics_by_basin)
        basin_coord = AuxCoord(
            np.array(basins, dtype=object),
            long_name="Ocean Basin",
            standard_name="region",
            var_name="basin",
        )
        output: dict[Variable, Cube] = {}
        for metric, variable in self._outputs.items():
            values = np.array(
                [
                    self._as_float(metrics_by_basin[basin].get(metric))
                    for basin in basins
                ],
                dtype=float,
            )
            output[variable] = Cube(
                values,
                var_name=variable.var_name,
                long_name=metric,
                aux_coords_and_dims=[(basin_coord, 0)],
            )
        return output

    @staticmethod
    def _as_float(value: Any) -> float:
        """Coerce a metric to a float, with anything unusable as NaN."""
        if value is None:
            return float("nan")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    def _check_metrics(
        self,
        metrics_by_basin: dict[str, dict[str, float]],
        name: str,
    ) -> None:
        """Warn if the metrics produced and those declared disagree."""
        produced: set[str] = set()
        for metrics in metrics_by_basin.values():
            produced |= set(metrics)
        undeclared = sorted(produced - set(METRICS))
        unproduced = sorted(set(METRICS) - produced)
        if undeclared:
            logger.warning(
                f"{name}: metrics_from_fields returned {undeclared}, which "
                f"METRICS does not declare, so they are not written out",
            )
        if unproduced:
            logger.warning(
                f"{name}: METRICS declares {unproduced}, which "
                f"metrics_from_fields did not return; written as missing",
            )

    def _get_field(
        self,
        data: CubeList | Dataset,
        variable: Variable,
        *,
        required: bool = True,
    ) -> DataArray | None:
        """Get a preprocessed field as a data array, or None if absent."""
        try:
            cube = self._preprocess(get_prepared_cube(data, variable))
        except Exception as exc:  # noqa: BLE001
            if required:
                raise
            logger.warning(
                f"No {variable.var_name}: metrics depending on it will be "
                f"missing ({exc})",
            )
            return None
        return DataArray.from_iris(cube)

    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess data."""
        cube = self._surface(cube)
        with setup_esmvaltool_config_and_logging():
            cube = extract_time(
                cube, self._period[0], 1, 1, self._period[1], 12, 31,
            )
            cube = regrid(
                cube,
                self._grid,
                "linear",
                cache_weights=True,
            )  # to ensure we are on a regular grid
        return cube  # noqa: RET504

    def _surface(self, cube: Cube) -> Cube:
        """Reduce a cube with a vertical coordinate to its surface level.

        ``FIELD_SPEC`` marks chlorophyll as a surface field and the
        reduction routines would take the top level themselves, so this
        is for cost rather than correctness: chlorophyll is an Omon
        field on tens of depth levels, and regridding all of them to
        feed one to the analysis is the most expensive thing this
        diagnostic would otherwise do.

        The shallowest level is taken rather than index 0, because the
        ordering of a depth coordinate is a model's choice and not a
        guarantee.
        """
        try:
            z_coord = cube.coord(axis="Z", dim_coords=True)
        except Exception:  # noqa: BLE001 - iris raises several types here
            return cube

        axis = cube.coord_dims(z_coord)[0]
        shallowest = int(np.argmin(np.abs(np.asarray(z_coord.points, float))))
        index: list[Any] = [slice(None)] * cube.ndim
        index[axis] = shallowest
        logger.debug(
            f"Reducing '{z_coord.name()}' to its shallowest level "
            f"({z_coord.points[shallowest]} {z_coord.units})",
        )
        return cube[tuple(index)]
