"""Benchmarking Diagnostics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import ibis
import ibis.selectors
import pandas as pd
from esmvalcore.preprocessor import distance_metric
from iris.pandas import as_data_frame

from benchmarking import _get_prepared_cube

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from iris.cube import Cube, CubeList
    from xarray import Dataset

    from benchmarking import Config, Variable
    from benchmarking.data import DataSource


class Diagnostic(ABC):
    """Manage diagnostics."""

    DATASET_STR: ClassVar[str] = "Dataset"

    def __init__(
        self,
        *,
        reference_data: Mapping[Variable, type[DataSource]],
        other_data: Mapping[Variable, Iterable[type[DataSource]]],
        config: Config,
    ) -> None:
        """Initialize class instance."""
        self._config = config

        if not reference_data:
            msg = "At least a single variable and reference dataset must be given."
            raise ValueError(msg)

        self._variables = sorted(reference_data.keys())
        self._reference_data = {v: r(self.config) for v, r in reference_data.items()}

        _other_data: dict[Variable, tuple[DataSource, ...]] = {}
        for var in self.variables:
            if var.var_name in other_data:
                val = tuple(o(self.config) for o in other_data[var])
            else:
                val = ()
            _other_data[var] = val
        self._other_data = _other_data

        self._tags = self._get_tags()

    @property
    def config(self) -> Config:
        """Get configuration."""
        return self._config

    @property
    def name(self) -> str:
        """Get diagnostic name."""
        return self.__class__.__name__

    @property
    def reference_data(self) -> dict[Variable, DataSource]:
        """Get reference data."""
        return self._reference_data

    @property
    def other_data(self) -> dict[Variable, tuple[DataSource, ...]]:
        """Get other data."""
        return self._other_data

    @property
    def tags(self) -> tuple[str, ...]:
        """Get tags."""
        return self._tags

    @property
    def variables(self) -> list[Variable]:
        """Get variables."""
        return self._variables

    def get_tables(
        self,
        data: CubeList | Dataset,
        data_name: str,
    ) -> tuple[ibis.Table, ibis.Table]:
        """Get diagnostic output tables.

        Parameters
        ----------
        data:
            Data to benchmark.
        data_name:
            Name of data to benchmark.

        Returns
        -------
        :
            (Data table, metrics table).

        """
        cube_dict = {var: self._get_cube(data, var) for var in self.variables}
        table = self._get_table_of_data(cube_dict, data_name)

        metrics = ibis.memtable(
            pd.DataFrame(
                {
                    "var_name": pd.Series(dtype="str"),
                    **dict.fromkeys(self._get_metric_names(), pd.Series(dtype="float")),
                },
            ),
        )

        # Reference data
        for variable, reference_data in self.reference_data.items():
            reference_cube = self._preprocess(reference_data.get_cube(variable))

            # Expand table
            new_table = self._get_table_of_cube(
                reference_cube,
                reference_data.name,
                variable,
            )
            new_table = self._add_missing_columns(new_table, table)
            table = ibis.union(table, new_table)

            # Calculate metrics
            new_metrics = ibis.memtable(
                pd.DataFrame(
                    {
                        "var_name": variable.var_name,
                        **self._get_metrics(cube_dict[variable], reference_cube),
                    },
                    index=[0],
                ),
            )
            metrics = ibis.union(metrics, new_metrics)

        # Other data
        for variable, other_datas in self.other_data.items():
            for other_data in other_datas:
                other_cube = self._preprocess(other_data.get_cube(variable))
                new_table = self._get_table_of_cube(
                    other_cube,
                    other_data.name,
                    variable,
                )
                new_table = self._add_missing_columns(new_table, table)
                table = ibis.union(table, new_table)

        return table, metrics

    def _add_missing_columns(
        self,
        table: ibis.Table,
        target_table: ibis.Table,
    ) -> ibis.Table:
        """Add missing columns to table (compared to target table)."""
        missing_columns = set(target_table.columns) - set(table.columns)
        if missing_columns:
            table = table.mutate(**{col: ibis.literal(None) for col in missing_columns})
            table = table.cast(
                **{col: target_table[col].type() for col in missing_columns},
            )
        return table

    def _get_cube(self, data: CubeList | Dataset, variable: Variable) -> Cube:
        """Get data cube."""
        cube = _get_prepared_cube(data, variable, self.config)
        return self._preprocess(cube)

    def _get_metric_names(self) -> tuple[str, ...]:
        """Get metric names."""
        return ("weighted_rmse", "weighted_pearsonr", "weighted_emd")

    def _get_metrics(self, cube: Cube, reference_cube: Cube) -> dict[str, float]:
        """Get metrics."""
        metrics: dict[str, float] = {}
        for metric in self._get_metric_names():
            metrics[metric] = float(
                distance_metric([cube], metric, reference=reference_cube)[0].data,
            )
        return metrics

    def _get_table_of_cube(
        self,
        cube: Cube,
        name: str,
        variable: Variable,
    ) -> ibis.Table:
        """Get table of a single cube."""
        df = as_data_frame(cube).reset_index()
        if "time" in df.columns:
            df["time"] = df["time"].astype(str)
        table = ibis.memtable(df)
        table = table.rename(**{variable.var_name: cube.name()})
        table = table.mutate(**{self.DATASET_STR: ibis.literal(name)})
        if "time" in table.columns:
            table = table.cast(time="timestamp")
        return table

    def _get_table_of_data(
        self,
        cube_dict: dict[Variable, Cube],
        data_name: str,
    ) -> ibis.Table:
        """Get table of data to benchmark."""
        variable = self.variables[0]  # self.variables will never be empty
        cube = cube_dict[variable]
        table = self._get_table_of_cube(cube, data_name, variable)
        for variable in self.variables[1:]:
            cube = cube_dict[variable]
            new_table = self._get_table_of_cube(cube, data_name, variable)
            table = table.join(new_table, how="outer", rname="{name}_DUPLICATE")
            table = table.select(ibis.selectors.matches(r"^(?!.*_DUPLICATE)"))

        return table

    def _get_tags(self) -> tuple[str, ...]:
        """Get diagnostic tags."""
        tags: list[str] = []
        tags.append(self.name.lower())
        tags.extend(v.var_name for v in self.variables)
        tags.extend(v.realm for v in self.variables)
        return tuple(sorted(set(tags)))

    @abstractmethod
    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess cube."""
