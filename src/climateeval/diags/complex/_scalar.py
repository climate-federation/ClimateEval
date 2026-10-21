"""Provide the base class of complex diagnostics with scalar output.

A number of complex diagnostics reduce one experiment to a handful of
scalars (an energy-budget residual, a feedback parameter, a correlation).
They share the regridding to ClimateEval's common grid, the global-mean
series, the scalar output cubes, the output-table layout, and the CMIP6
r1i1p1f1 comparison ensemble that :class:`~climateeval.diags.complex.ECS`
introduced; :class:`ScalarExperimentDiagnostic` holds that machinery.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from esmvalcore.preprocessor import (
    annual_statistics,
    area_statistics,
    regrid,
)
from iris.cube import Cube, CubeList
from loguru import logger

from climateeval import Variable
from climateeval._config import (
    setup_esmvaltool_config_and_logging,
    setup_iris_combine_policy,
)
from climateeval._utils import get_prepared_cube
from climateeval.data import ComplexDataSource, DataSourceInformation
from climateeval.data._esmvaltool_intake import filter_intake_data_sources
from climateeval.diags._base import DiagnosticOutput
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.complex._base import ComplexDiagnostic

if TYPE_CHECKING:
    from collections.abc import Generator

    from xarray import Dataset

    from climateeval.data._esmvaltool_intake import (
        ESMValToolIntakeDataSourceGenerator,
    )

logger = logger.opt(colors=True)


def as_float_array(data: Any) -> np.ndarray:
    """Get a float array in which a masked array's mask has become NaN.

    :func:`numpy.asarray` on a masked array silently exposes the values
    *under* the mask, which for an observational product is typically land or
    a sub-surface pressure level.

    Parameters
    ----------
    data:
        Cube data, masked or not.

    Returns
    -------
    :
        Float array with NaN wherever the input was masked.

    """
    if np.ma.isMaskedArray(data):
        return np.ma.filled(data.astype(float), np.nan)
    return np.asarray(data, dtype=float)


def calendar_years_and_months(cube: Cube) -> tuple[np.ndarray, np.ndarray]:
    """Get the calendar year and month of every step of a cube's time axis.

    Parameters
    ----------
    cube:
        Cube with a time coordinate.

    Returns
    -------
    :
        Years and months, one entry per time step.

    """
    coord = cube.coord("time")
    dates = coord.units.num2date(coord.points)
    return (
        np.array([d.year for d in dates], dtype=int),
        np.array([d.month for d in dates], dtype=int),
    )


class ScalarExperimentDiagnostic(ComplexDiagnostic):
    """Manage complex diagnostics that reduce experiments to scalars.

    Subclasses declare the CMIP variables they read (``_input_variables``),
    the scalars they emit (``_output_variables``) and, optionally, the CMIP6
    generator supplying the comparison ensemble (``_other_data_generator``
    and ``_other_data_exp``; the ensemble is only available for diagnostics
    of a single experiment that the CMIP6 archive actually holds).
    """

    _input_variables: ClassVar[tuple[Variable, ...]] = ()
    """Variables read from the experiment."""

    _output_variables: ClassVar[tuple[Variable, ...]] = ()
    """Scalars emitted by the diagnostic, in output-column order."""

    _other_data_generator: ClassVar[
        type[ESMValToolIntakeDataSourceGenerator] | None
    ] = None
    """Generator of the CMIP6 comparison ensemble (``None``: no other data)."""

    _other_data_exp: ClassVar[str] = ""
    """CMIP6 ``exp`` facet of the comparison ensemble."""

    @property
    def _data_key(self) -> str:
        """Get the data key of a single-experiment diagnostic."""
        return self._required_data_keys[0]

    def _prepared_cube(self, data: CubeList | Dataset, variable: Variable) -> Cube:
        """Get one variable of an experiment on the common grid."""
        cube = get_prepared_cube(data, variable)
        with setup_esmvaltool_config_and_logging():
            cube = regrid(cube, DEFAULT_GRID, "linear", cache_weights=True)
        return cube  # noqa: RET504 - named for readability

    def _global_annual_mean(
        self,
        data: CubeList | Dataset,
        variable: Variable,
    ) -> np.ndarray:
        """Get the global annual-mean series of one variable."""
        cube = self._prepared_cube(data, variable)
        with setup_esmvaltool_config_and_logging():
            cube = area_statistics(cube, "mean")
            cube = annual_statistics(cube, "mean")
        return as_float_array(cube.data)

    def _global_annual_mean_and_years(
        self,
        data: CubeList | Dataset,
        variable: Variable,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the global annual-mean series and its calendar years.

        The years matter wherever a diagnostic names a calendar window rather
        than a position in the record.
        """
        cube = self._prepared_cube(data, variable)
        with setup_esmvaltool_config_and_logging():
            cube = area_statistics(cube, "mean")
            cube = annual_statistics(cube, "mean")
        years, _months = calendar_years_and_months(cube)
        return as_float_array(cube.data), years

    def _toa_net_global_annual_mean(self, data: CubeList | Dataset) -> np.ndarray:
        """Get the global annual-mean TOA net downward flux series.

        ``N = rsdt - rsut - rlut``, assembled from the three archived fluxes
        rather than from the derived ``rtnt`` variable so the diagnostic works
        on any set of cubes carrying them.
        """
        return self._toa_net_global_annual_mean_and_years(data)[0]

    def _toa_net_global_annual_mean_and_years(
        self,
        data: CubeList | Dataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the global annual-mean TOA net flux series and its years."""
        series = {
            name: self._global_annual_mean_and_years(data, Variable(name, name, "mon"))
            for name in ("rsdt", "rsut", "rlut")
        }
        years = series["rsdt"][1]
        for _values, own_years in series.values():
            years = np.intersect1d(years, own_years)

        def aligned(name: str) -> np.ndarray:
            values, own_years = series[name]
            return values[np.isin(own_years, years)]

        return aligned("rsdt") - aligned("rsut") - aligned("rlut"), years

    @staticmethod
    def _scalar_cubes(values: dict[Variable, float]) -> dict[Variable, Cube]:
        """Get one scalar cube per output variable."""
        return {
            variable: Cube(float(value), var_name=variable.var_name)
            for variable, value in values.items()
        }

    def _get_other_data(self) -> Generator[ComplexDataSource]:
        """Get the CMIP6 r1i1p1f1 comparison ensemble (as in ECS)."""
        if self._other_data_generator is None:
            logger.debug(
                f"No other data for complex diagnostic '{self.name}' available",
            )
            yield from ()
            return
        logger.debug(f"Retrieving other data for complex diagnostic '{self.name}'")
        variables = self._input_variables
        generator = self._other_data_generator
        grouped_data_sources = filter_intake_data_sources(
            *[generator().generate(var) for var in variables],
            group_by="dataset",
            required_facets=tuple(
                {"exp": self._other_data_exp, "short_name": var.var_name}
                for var in variables
            ),
        )
        data_source_class_name = f"{generator.__name__}DataSource"
        for dataset, data_sources in grouped_data_sources.items():
            logger.debug(f"Found other data of CMIP6 model {dataset}")
            data_source = next(
                d
                for d in data_sources
                if d.__class__.__name__ == data_source_class_name
            )
            data_information = DataSourceInformation(
                name=dataset,
                category=data_source.information.category,
                institute=data_source.information.institute,
                exp=self._other_data_exp,
                variant=data_source.information.variant,
                references=data_source.information.references,
            )
            with setup_iris_combine_policy():
                try:
                    data = {
                        self._data_key: CubeList(
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
        """Drop superfluous rows and order the output columns."""
        if output.raw_output is None:
            return output
        unique_columns = (self._data_id_column, self._data_type_column)
        variables = tuple(var.id for var in self._output_variables)
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
