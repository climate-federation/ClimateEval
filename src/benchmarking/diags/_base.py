"""Provide base class for diagnostics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import ibis
import ibis.selectors
import pandas as pd
from esmvalcore.preprocessor import distance_metric
from ibis import _
from iris.pandas import as_data_frame
from loguru import logger

from benchmarking import NO_LEVEL
from benchmarking._utils import check_coordinates, get_prepared_cube

if TYPE_CHECKING:
    from iris.cube import Cube, CubeList
    from xarray import Dataset

    from benchmarking import Coordinate, Variable
    from benchmarking.data import DataSource, DiagnosticData


logger = logger.opt(colors=True)

DEFAULT_GRID: str = "2x2"
PLEV19_LEVELS: list[float] = [
    100000.0,
    92500.0,
    85000.0,
    70000.0,
    60000.0,
    50000.0,
    40000.0,
    30000.0,
    25000.0,
    20000.0,
    15000.0,
    10000.0,
    7000.0,
    5000.0,
    3000.0,
    2000.0,
    1000.0,
    500.0,
    100.0,
]


@dataclass(frozen=True, kw_only=True)
class DiagnosticOutput:
    data: ibis.Table
    metrics: ibis.Table
    variables: ibis.Table

    @classmethod
    def union(
        cls,
        *diagnostic_output: DiagnosticOutput,
        perform_variables_union: bool = False,
    ) -> DiagnosticOutput:
        """Create union of output."""
        data_union = ibis.union(*(o.data for o in diagnostic_output))
        metrics_union = ibis.union(*(o.metrics for o in diagnostic_output))
        if perform_variables_union:
            variables_union = ibis.union(*(o.variables for o in diagnostic_output))
        else:
            variables_union = diagnostic_output[0].variables
        return cls(data=data_union, metrics=metrics_union, variables=variables_union)


class Diagnostic(ABC):
    """Manage diagnostics."""

    DATASET_COLUMN: ClassVar[str] = "Dataset"
    DATA_TYPE_COLUMN: ClassVar[str] = "Data Type"
    DTYPE: ClassVar[ibis.DataType] = "float32"
    FINAL_COORDINATES: ClassVar[tuple[Coordinate, ...]] = ()
    """Coordinates of preprocessed output."""

    def __init__(
        self,
        variables_and_data: dict[Variable, DiagnosticData],
        *,
        data_dir: Path | None = None,
        fail_on_missing_data: bool = False,
    ) -> None:
        """Initialize class instance."""
        logger.debug(f"Initializing diagnostic {self.name}")

        if data_dir is None:
            data_dir = Path.home() / ".benchmarking" / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        self._data_dir = data_dir

        if not variables_and_data:
            msg = "At least a single variable must be given."
            raise ValueError(msg)

        self._variables = sorted(variables_and_data.keys())
        self._reference_data = {
            v: d.reference(self.data_dir) for v, d in variables_and_data.items()
        }
        variable_ids = [v.id for v in self.variables]
        if len(set(variable_ids)) != len(variable_ids):
            msg = (
                f"Variable IDs must be unique for diagnostic {self.name}, got "
                f"{variable_ids}"
            )
            raise ValueError(msg)

        _other_data: dict[Variable, tuple[DataSource, ...]] = {}
        for variable, data_item in variables_and_data.items():
            if data_item.other:
                val = tuple(o(self.data_dir) for o in data_item.other)
            else:
                val = ()
            _other_data[variable] = val
        self._other_data = _other_data

        self._tags = self._get_tags()

        self._fail_on_missing_data = fail_on_missing_data

    @property
    def data_dir(self) -> Path:
        """Get path to data directory."""
        return self._data_dir

    @property
    def final_coordinate_names(self) -> tuple[str, ...]:
        """Get names of final coordinate."""
        return tuple(c.name() for c in self.FINAL_COORDINATES)

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

    def get_output(
        self,
        data: CubeList | Dataset,
        data_name: str,
    ) -> DiagnosticOutput:
        """Get diagnostic output.

        Parameters
        ----------
        data:
            Data to benchmark.
        data_name:
            Name of data to benchmark.

        Returns
        -------
        :
            Diagnostic output.

        """
        # Load reference cubes
        reference_cubes: dict[Variable, Cube] = {}
        for variable, reference_data in self.reference_data.items():
            try:
                reference_cube = self._safely_preprocess(
                    reference_data.get_cube(variable),
                )
            except Exception as exc:
                if self._fail_on_missing_data:
                    msg = (
                        f"Failed to get reference data {reference_data.name} for "
                        f"variable {variable}: {exc}"
                    )
                    raise ValueError(msg) from exc
                logger.warning(msg)
                continue
            reference_cubes[variable] = reference_cube

        # Load cube data ahead to avoid potential bug in ESMValCore (see
        # https://github.com/ESMValGroup/ESMValCore/pull/3038)
        for cube in reference_cubes.values():
            cube.data  # noqa: B018

        # Process data to benchmark
        diagnostic_output = self._get_output_of_to_benchmark_data(
            data,
            data_name,
            reference_cubes,
        )

        # Process other data
        other_output = self._get_output_of_other_data(reference_cubes)
        diagnostic_output = DiagnosticOutput.union(diagnostic_output, other_output)

        # Add reference data to table
        reference_output = self._get_output_of_reference_data(reference_cubes)
        diagnostic_output = DiagnosticOutput.union(diagnostic_output, reference_output)

        return diagnostic_output  # noqa: RET504

    def _get_empty_data_table(self) -> ibis.Table:
        """Get empty data table with correct schema."""
        data_table = ibis.memtable(
            pd.DataFrame(
                {
                    self.DATASET_COLUMN: pd.Series(dtype="str"),
                    self.DATA_TYPE_COLUMN: pd.Series(dtype="str"),
                    **dict.fromkeys(
                        self.final_coordinate_names,
                        pd.Series(dtype=self.DTYPE),
                    ),
                    **dict.fromkeys(
                        sorted(v.id for v in self.variables),
                        pd.Series(dtype=self.DTYPE),
                    ),
                },
            ),
        )
        if "time" in self.final_coordinate_names:
            data_table = data_table.cast(time="timestamp")
        return data_table

    def _get_empty_metrics_table(self) -> ibis.Table:
        """Get empty metrics table with correct schema."""
        return ibis.memtable(
            pd.DataFrame(
                {
                    "var_id": pd.Series(dtype="str"),
                    **dict.fromkeys(
                        self._get_metric_names(),
                        pd.Series(dtype=self.DTYPE),
                    ),
                    self.DATASET_COLUMN: pd.Series(dtype="str"),
                    "reference": pd.Series(dtype="str"),
                },
            ),
        )

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
                    self.DATASET_COLUMN: data_name,
                    "reference": self.reference_data[variable].name,
                },
                index=[0],
            ),
        )
        cast_operations = dict.fromkeys(self._get_metric_names(), self.DTYPE)
        return metrics.cast(**cast_operations)

    def _add_missing_columns(self, data_table: ibis.Table) -> ibis.Table:
        """Add missing columns to data table (one column per variable)."""
        missing_columns = {v.id for v in self.variables} - set(data_table.columns)
        if missing_columns:
            data_table = data_table.mutate(
                **{col: ibis.literal(None) for col in missing_columns},
            )
            data_table = data_table.cast(**dict.fromkeys(missing_columns, self.DTYPE))
        new_col_order = (
            self.DATASET_COLUMN,
            self.DATA_TYPE_COLUMN,
            *self.final_coordinate_names,
            *sorted(v.id for v in self.variables),
        )
        return data_table.select(new_col_order)

    def _get_cube(self, data: CubeList | Dataset, variable: Variable) -> Cube:
        """Get data cube."""
        cube = get_prepared_cube(data, variable)
        return self._safely_preprocess(cube)

    def _get_data_table_of_cube(
        self,
        cube: Cube,
        *,
        variable: Variable,
        data_name: str,
        data_type: Literal["to_benchmark", "reference", "other"],
    ) -> ibis.Table:
        """Get data table of a single cube."""
        df = as_data_frame(cube).reset_index()
        if "time" in df.columns:
            df["time"] = df["time"].astype(str)
        data_table = ibis.memtable(df)
        cast_operations = {
            c: {"time": "timestamp"}.get(c, self.DTYPE) for c in data_table.columns
        }
        data_table = data_table.cast(**cast_operations)
        data_table = data_table.rename(**{variable.id: cube.name()})
        data_table = data_table.mutate(**{self.DATASET_COLUMN: ibis.literal(data_name)})
        data_table = data_table.mutate(
            **{self.DATA_TYPE_COLUMN: ibis.literal(data_type)},
        )
        return self._add_missing_columns(data_table)

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

    def _get_output_of_other_data(
        self,
        reference_cubes: dict[Variable, Cube],
    ) -> DiagnosticOutput:
        """Get output of other data."""
        data_table = self._get_empty_data_table()
        metrics_table = self._get_empty_metrics_table()

        for variable, other_datas in self.other_data.items():
            for other_data in other_datas:
                try:
                    other_cube = self._safely_preprocess(other_data.get_cube(variable))
                except Exception as exc:
                    msg = (
                        f"Failed to get other data {other_data.name} for variable "
                        f"{variable}: {exc}"
                    )
                    if self._fail_on_missing_data:
                        raise ValueError(msg) from exc
                    logger.warning(msg)
                    continue

                # Expand table
                new_data_table = self._get_data_table_of_cube(
                    other_cube,
                    variable=variable,
                    data_name=other_data.name,
                    data_type="other",
                )
                data_table = ibis.union(data_table, new_data_table)

                # Expand metrics
                if variable not in reference_cubes:
                    continue
                new_metrics_table = self._get_metrics_table(
                    other_cube,
                    reference_cubes[variable],
                    variable=variable,
                    data_name=other_data.name,
                )
                metrics_table = ibis.union(metrics_table, new_metrics_table)

        # Drop superfluous rows by removing NaN entries
        data_table = self._remove_superfluous_rows(data_table)

        variables_table = self._get_variables_table()

        return DiagnosticOutput(
            data=data_table,
            metrics=metrics_table,
            variables=variables_table,
        )

    def _get_output_of_reference_data(
        self,
        reference_cubes: dict[Variable, Cube],
    ) -> DiagnosticOutput:
        """Get output of other data."""
        data_table = self._get_empty_data_table()
        metrics_table = self._get_empty_metrics_table()  # will stay empty

        for variable, reference_cube in reference_cubes.items():
            new_data_table = self._get_data_table_of_cube(
                reference_cube,
                variable=variable,
                data_name=self.reference_data[variable].name,
                data_type="reference",
            )
            data_table = ibis.union(data_table, new_data_table)

        # Drop superfluous rows by removing NaN entries
        data_table = self._remove_superfluous_rows(data_table)

        variables_table = self._get_variables_table()

        return DiagnosticOutput(
            data=data_table,
            metrics=metrics_table,
            variables=variables_table,
        )

    def _get_output_of_to_benchmark_data(
        self,
        data: CubeList | Dataset,
        data_name: str,
        reference_cubes: dict[Variable, Cube],
    ) -> DiagnosticOutput:
        """Get output of data to benchmark."""
        data_table = self._get_empty_data_table()
        metrics_table = self._get_empty_metrics_table()

        for variable in self.variables:
            logger.debug(f"Loading {variable} of {data_name}")
            try:
                cube = self._get_cube(data, variable)
            except Exception as exc:
                msg = f"Failed to get data {data_name} for variable {variable}: {exc}"
                if self._fail_on_missing_data:
                    raise ValueError(msg) from exc
                logger.warning(msg)
                continue

            # Expand table
            new_data_table = self._get_data_table_of_cube(
                cube,
                variable=variable,
                data_name=data_name,
                data_type="to_benchmark",
            )
            data_table = ibis.union(data_table, new_data_table)

            # Expand metrics
            if variable not in reference_cubes:
                continue
            new_metrics_table = self._get_metrics_table(
                cube,
                reference_cubes[variable],
                variable=variable,
                data_name=data_name,
            )
            metrics_table = ibis.union(metrics_table, new_metrics_table)

        # Drop superfluous rows by removing NaN entries
        data_table = self._remove_superfluous_rows(data_table)

        variables_table = self._get_variables_table()

        return DiagnosticOutput(
            data=data_table,
            metrics=metrics_table,
            variables=variables_table,
        )

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

    def _remove_superfluous_rows(self, table: ibis.Table) -> ibis.Table:
        """Drop superfluous rows by removing NaN entries.

        Note
        ----
        Currently, we first add NaNs and then remove it again. The reason is
        that the logic is simply and the code robust. There are certainly
        smarter and more efficient ways to do this, but since our tables are
        usually quite small, it's probably not necessary.

        """
        unique_columns = (
            self.DATASET_COLUMN,
            self.DATA_TYPE_COLUMN,
            *self.final_coordinate_names,
        )
        table = table.group_by(unique_columns).aggregate(
            **{v.id: _[v.id].first() for v in self.variables},
        )
        return table.drop_null([v.id for v in self.variables], how="all")

    def _safely_preprocess(self, cube: Cube) -> Cube:
        """Safely preprocess cube."""
        cube = self._preprocess(cube)
        check_coordinates(cube, self.FINAL_COORDINATES)
        return cube

    def _get_variables_table(self) -> ibis.Table:
        """Get variables table."""
        df = pd.DataFrame(self.variables)
        df = df.drop("settings", axis=1)
        df["coordinates"] = df["coordinates"].map(
            lambda r: " ".join(c["var_name"] for c in r),
        )

        def vertical_level_dict_to_str(
            vertical_level_dict: dict[str, str | float],
        ) -> str:
            """Convert vertical level dictionary to string."""
            if vertical_level_dict == asdict(NO_LEVEL):
                return ""
            return (
                f"{vertical_level_dict['coordinate_name']} = "
                f"{vertical_level_dict['value']} {vertical_level_dict['units']}"
            )

        df["vertical_level"] = df["vertical_level"].map(vertical_level_dict_to_str)
        return ibis.memtable(df)
