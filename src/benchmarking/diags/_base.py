"""Benchmarking Diagnostics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import ibis
import ibis.selectors
import pandas as pd
from esmvalcore.preprocessor import distance_metric
from iris.pandas import as_data_frame
from loguru import logger

from benchmarking._utils import check_coordinates, get_prepared_cube

if TYPE_CHECKING:
    from iris.cube import Cube, CubeList
    from xarray import Dataset

    from benchmarking import Variable
    from benchmarking.data import DataSource, DiagnosticData


logger = logger.opt(colors=True)


class Diagnostic(ABC):
    """Manage diagnostics."""

    DATASET_STR: ClassVar[str] = "Dataset"
    DATA_TYPE_STR: ClassVar[str] = "Data Type"
    DTYPE: ClassVar[ibis.DataType] = "float32"
    FINAL_COORDINATES: ClassVar[tuple[str, ...]] = ()
    """Coordinates (standard names) of preprocessed output."""

    def __init__(
        self,
        data: dict[Variable, DiagnosticData],
        *,
        data_dir: Path | None = None,
    ) -> None:
        """Initialize class instance."""
        logger.debug(f"Initializing diagnostic {self.name}")

        if data_dir is None:
            data_dir = Path.home() / ".benchmarking" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir

        if not data:
            msg = "At least a single variable must be given."
            raise ValueError(msg)

        self._variables = sorted(data.keys())
        self._reference_data = {v: d.reference(self.data_dir) for v, d in data.items()}

        _other_data: dict[Variable, tuple[DataSource, ...]] = {}
        for variable, data_item in data.items():
            if data_item.other:
                val = tuple(o(self.data_dir) for o in data_item.other)
            else:
                val = ()
            _other_data[variable] = val
        self._other_data = _other_data

        self._tags = self._get_tags()

    @property
    def data_dir(self) -> Path:
        """Get path to data directory."""
        return self._data_dir

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
        logger.debug(f"Loading {data_name}")
        ref_cube_dict = {
            v: self._safely_preprocess(r.get_cube(v))
            for v, r in self.reference_data.items()
        }

        # Create empty tables with correct columns
        table = ibis.memtable(
            pd.DataFrame(
                {
                    self.DATASET_STR: pd.Series(dtype="str"),
                    self.DATA_TYPE_STR: pd.Series(dtype="str"),
                    **dict.fromkeys(
                        self.FINAL_COORDINATES,
                        pd.Series(dtype=self.DTYPE),
                    ),
                    **dict.fromkeys(
                        sorted(v.id for v in self.variables),
                        pd.Series(dtype=self.DTYPE),
                    ),
                },
            ),
        )
        if "time" in self.FINAL_COORDINATES:
            table = table.cast(time="timestamp")
        metrics = ibis.memtable(
            pd.DataFrame(
                {
                    "var_id": pd.Series(dtype="str"),
                    **dict.fromkeys(
                        self._get_metric_names(),
                        pd.Series(dtype=self.DTYPE),
                    ),
                    self.DATASET_STR: pd.Series(dtype="str"),
                    "reference": pd.Series(dtype="str"),
                },
            ),
        )

        # Process data to benchmark
        for variable in self.variables:
            cube = self._get_cube(data, variable)
            new_table = self._get_table_of_cube(
                cube,
                variable=variable,
                data_name=data_name,
                data_type="to_benchmark",
            )
            table = ibis.union(table, new_table)
            new_metrics = self._get_metrics_table(
                cube,
                ref_cube_dict[variable],
                variable=variable,
                data_name=data_name,
            )
            metrics = ibis.union(metrics, new_metrics)

        # Process other data
        for variable, other_datas in self.other_data.items():
            for other_data in other_datas:
                other_cube = self._safely_preprocess(other_data.get_cube(variable))
                new_table = self._get_table_of_cube(
                    other_cube,
                    variable=variable,
                    data_name=other_data.name,
                    data_type="other",
                )
                table = ibis.union(table, new_table)
                new_metrics = self._get_metrics_table(
                    other_cube,
                    ref_cube_dict[variable],
                    variable=variable,
                    data_name=other_data.name,
                )
                metrics = ibis.union(metrics, new_metrics)

        # Add reference data to table
        for variable, reference_cube in ref_cube_dict.items():
            new_table = self._get_table_of_cube(
                reference_cube,
                variable=variable,
                data_name=self.reference_data[variable].name,
                data_type="reference",
            )
            table = ibis.union(table, new_table)

        return table, metrics

    def _get_metrics_table(
        self,
        cube: Cube,
        reference_cube: Cube,
        *,
        variable: Variable,
        data_name: str,
    ) -> ibis.Table:
        """Get metrics."""
        metrics = ibis.memtable(
            pd.DataFrame(
                {
                    "var_id": variable.id,
                    **self._get_metrics(cube, reference_cube),
                    self.DATASET_STR: data_name,
                    "reference": self.reference_data[variable].name,
                },
                index=[0],
            ),
        )
        cast_operations = dict.fromkeys(self._get_metric_names(), self.DTYPE)
        return metrics.cast(**cast_operations)

    def _add_missing_columns(self, table: ibis.Table) -> ibis.Table:
        """Add missing columns to table (one column per variable)."""
        missing_columns = {v.id for v in self.variables} - set(table.columns)
        if missing_columns:
            table = table.mutate(**{col: ibis.literal(None) for col in missing_columns})
            table = table.cast(**dict.fromkeys(missing_columns, self.DTYPE))
        new_col_order = (
            self.DATASET_STR,
            self.DATA_TYPE_STR,
            *self.FINAL_COORDINATES,
            *sorted(v.id for v in self.variables),
        )
        return table.select(new_col_order)

    def _get_cube(self, data: CubeList | Dataset, variable: Variable) -> Cube:
        """Get data cube."""
        cube = get_prepared_cube(data, variable)
        return self._safely_preprocess(cube)

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
        *,
        variable: Variable,
        data_name: str,
        data_type: Literal["to_benchmark", "reference", "other"],
    ) -> ibis.Table:
        """Get table of a single cube."""
        df = as_data_frame(cube).reset_index()
        if "time" in df.columns:
            df["time"] = df["time"].astype(str)
        table = ibis.memtable(df)
        cast_operations = {
            c: {"time": "timestamp"}.get(c, self.DTYPE) for c in table.columns
        }
        table = table.cast(**cast_operations)
        table = table.rename(**{variable.id: cube.name()})
        table = table.mutate(**{self.DATASET_STR: ibis.literal(data_name)})
        table = table.mutate(**{self.DATA_TYPE_STR: ibis.literal(data_type)})
        return self._add_missing_columns(table)

    def _get_table_of_data(
        self,
        cube_dict: dict[Variable, Cube],
        data_name: str,
    ) -> ibis.Table:
        """Get table of data to benchmark."""
        variable = self.variables[0]  # self.variables will never be empty
        cube = cube_dict[variable]
        table = self._get_table_of_cube(
            cube,
            variable=variable,
            data_name=data_name,
            data_type="to_benchmark",
        )
        for variable in self.variables[1:]:
            cube = cube_dict[variable]
            new_table = self._get_table_of_cube(
                cube,
                variable=variable,
                data_name=data_name,
                data_type="to_benchmark",
            )

            # Join tables (add columns)
            table = table.join(
                new_table,
                self.FINAL_COORDINATES,
                how="outer",
                rname="{name}_DUPLICATE",
            )

            # Coalesce duplicated columns (take first non-null value)
            dup_cols = [col for col in table.columns if col.endswith("_DUPLICATE")]
            mutate_kwargs = {}
            for dup_col in dup_cols:
                orig_col = dup_col.replace("_DUPLICATE", "")
                mutate_kwargs[orig_col] = ibis.coalesce(
                    table[orig_col],
                    table[dup_col],
                )
            table = table.mutate(**mutate_kwargs)
            table = table.select(ibis.selectors.matches(r"^(?!.*_DUPLICATE)"))

        # Sort columns (variables to the right)
        new_col_order = (
            self.DATASET_STR,
            self.DATA_TYPE_STR,
            *self.FINAL_COORDINATES,
            *sorted(v.id for v in self.variables),
        )
        return table.select(new_col_order)

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

    def _safely_preprocess(self, cube: Cube) -> Cube:
        """Safely preprocess cube."""
        cube = self._preprocess(cube)
        check_coordinates(cube, self.FINAL_COORDINATES)
        return cube
