"""Provide map diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from esmvalcore.preprocessor import histogram, regrid
from loguru import logger

from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval.diags._utils import DEFAULT_GRID, union
from climateeval.diags.simple._base import SimpleDiagnostic
from climateeval.exceptions import CoordinateError

if TYPE_CHECKING:
    import ibis
    from iris.cube import Cube

    from climateeval import Variable


logger = logger.opt(colors=True)


class Histogram(SimpleDiagnostic):
    """Manage histogram diagnostics.

    Custom variable-specific diagnostic keyword arguments
    -----------------------------------------------------
    bins: int (default: 150)
        Number of bins.
    bin_range: tuple[float, float] (required)
        Bin range used for the histogram.

    """

    _OUTPUT_METRICS_TABLE = False

    def _check_variables(self) -> None:
        """Check variables."""
        for variable in self._variables:
            if "bin_range" not in variable.diagnostic_kwargs:
                msg = (
                    f"Expected diagnostic keyword argument 'bin_range' for {variable} "
                    f"(required by diagnostic {self.name})"
                )
                raise ValueError(msg)

    def _combine_raw_output_tables(
        self,
        *raw_output_tables: ibis.Table,
    ) -> ibis.Table | None:
        """Combine raw output tables."""
        if not raw_output_tables:
            return None
        raw_output_table = union(*raw_output_tables)
        new_column_order = [self._DATA_ID_COLUMN, self._DATA_TYPE_COLUMN]
        for variable in sorted(self._variables):
            new_column_order.extend([variable.id, f"density_{variable.id}"])
        return raw_output_table.select(new_column_order)

    def _get_raw_output_table(
        self,
        cube: Cube,
        *,
        variable: Variable,
        data_id: str,
        data_type: Literal["to_benchmark", "reference", "other"],
    ) -> ibis.Table:
        """Get data table of a single cube."""
        raw_output_table = super()._get_raw_output_table(
            cube,
            variable=variable,
            data_id=data_id,
            data_type=data_type,
        )
        raw_output_table = raw_output_table.rename(
            **{f"density_{variable.id}": variable.id},
            **{variable.id: cube.coords(dim_coords=True)[0].name()},
        )
        return raw_output_table  # noqa: RET504

    def _preprocess(self, cube: Cube, variable: Variable) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(cube, DEFAULT_GRID, "area_weighted", cache_weights=True)
            cube = histogram(
                cube,
                bins=variable.diagnostic_kwargs.get("bins", 150),
                bin_range=variable.diagnostic_kwargs["bin_range"],
                normalization="integral",
            )
        return cube  # noqa: RET504

    def _safely_preprocess(self, cube: Cube, variable: Variable) -> Cube:
        """Safely preprocess cube."""
        cube = self._preprocess(cube, variable)
        if cube.ndim != 1 or not cube.coords(dim_coords=True):
            msg = (
                f"Expected 1D data with a single dimensional coordinate after "
                f"preprocessing {variable} in diagnostic {self.name}, got "
                f"{cube.ndim}D cube {cube.summary(shorten=True)}"
            )
            raise CoordinateError(msg)
        dim_coord = cube.coords(dim_coords=True)[0]
        if dim_coord.var_name != variable.var_name:
            msg = (
                f"Expected single dimensional coordinate '{variable.var_name}' "
                f"in cube after preprocessing {variable} in diagnostic {self.name}, "
                f"got coordinate '{dim_coord.var_name}'"
            )
            raise CoordinateError(msg)
        return cube
