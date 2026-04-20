"""Provide basic features of diagnostics."""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import ibis
import pandas as pd
from esmvalcore.preprocessor import distance_metric
from ibis import _
from iris.pandas import as_data_frame
from iris.warnings import IrisVagueMetadataWarning
from loguru import logger

from benchmarking import NO_LEVEL
from benchmarking._utils import check_coordinates, get_prepared_cube

if TYPE_CHECKING:
    from iris.cube import Cube, CubeList
    from xarray import Dataset

    from benchmarking import Coordinate, Variable
    from benchmarking.data import DataSource, DataSourceInformation, DiagnosticData


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
    """Manage diagnostic output."""

    raw_output: ibis.Table | None
    metrics: ibis.Table | None
    variables: ibis.Table
    data_sources: ibis.Table

    @classmethod
    def union(cls, *diagnostic_output: DiagnosticOutput) -> DiagnosticOutput:
        """Create union of output."""
        # data
        all_raw_output = [o.raw_output for o in diagnostic_output]
        if None in all_raw_output and not all(d is None for d in all_raw_output):
            msg = "Table 'data' must be None for either all or no diagnostics"
            raise ValueError(msg)
        raw_output_union = (
            None if None in all_raw_output else ibis.union(*all_raw_output)
        )

        # metrics
        all_metrics = [o.metrics for o in diagnostic_output]
        if None in all_metrics and not all(d is None for d in all_metrics):
            msg = "Table 'metrics' must be None for either all or no diagnostics"
            raise ValueError(msg)
        metrics_union = None if None in all_metrics else ibis.union(*all_metrics)

        # variables
        variables_union = ibis.union(*(o.variables for o in diagnostic_output))

        # data_sources
        data_sources_union = ibis.union(*(o.data_sources for o in diagnostic_output))

        return cls(
            raw_output=raw_output_union,
            metrics=metrics_union,
            variables=variables_union,
            data_sources=data_sources_union,
        )


class Diagnostic(ABC):
    """Manage diagnostics."""

    DATA_ID_COLUMN: ClassVar[str] = "data_id"
    """Name of table column containing data ID."""

    REFERENCE_DATA_ID_COLUMN: ClassVar[str] = "reference_data_id"
    """Name of table column containing reference data ID."""

    DATA_TYPE_COLUMN: ClassVar[str] = "data_type"
    """Name of table column containing data type."""

    DTYPE: ClassVar[ibis.DataType] = "float32"
    """Data type of numerical values."""

    FINAL_COORDINATES: ClassVar[tuple[Coordinate, ...]] = ()
    """Coordinates of preprocessed output."""

    OUTPUT_DATA_TABLE: ClassVar[bool] = True
    """Output diagnostic data when creating database."""

    OUTPUT_METRICS_TABLE: ClassVar[bool] = True
    """Output metrics data when creating database."""

    def __init__(
        self,
        variables_and_data: dict[Variable, DiagnosticData],
        *,
        data_root_dir: Path | None = None,
        fail_on_missing_data: bool = False,
    ) -> None:
        """Initialize class instance."""
        logger.debug(f"Initializing diagnostic {self.name}")

        if data_root_dir is None:
            data_root_dir = Path.home() / ".benchmarking" / "data"
        data_root_dir.mkdir(parents=True, exist_ok=True)
        self._data_root_dir = data_root_dir
        logger.debug(f"Using data root directory {self.data_root_dir}")

        if not variables_and_data:
            msg = "At least a single variable must be given."
            raise ValueError(msg)

        self._variables = sorted(variables_and_data.keys())
        self._reference_data = {v: d.reference() for v, d in variables_and_data.items()}
        variable_ids = [v.id for v in self.variables]
        if len(set(variable_ids)) != len(variable_ids):
            msg = (
                f"Variable IDs must be unique for diagnostic {self.name}, got "
                f"{variable_ids}"
            )
            raise ValueError(msg)

        _other_data: dict[Variable, tuple[DataSource, ...]] = {}
        for variable, data_item in variables_and_data.items():
            _other_data[variable] = tuple(o() for o in data_item.other)
        self._other_data = _other_data

        self._fail_on_missing_data = fail_on_missing_data

    @property
    def data_root_dir(self) -> Path:
        """Get path to data directory."""
        return self._data_root_dir

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
    def variables(self) -> list[Variable]:
        """Get variables."""
        return self._variables

    def get_output(
        self,
        data: CubeList | Dataset,
        data_information: DataSourceInformation,
    ) -> DiagnosticOutput:
        """Get diagnostic output.

        Parameters
        ----------
        data:
            Data to benchmark.
        data_information:
            Information on data to benchmark.

        Returns
        -------
        :
            Diagnostic output.

        """
        # Load reference cubes
        logger.debug("Loading reference data")
        reference_cubes: dict[Variable, Cube] = {}
        for variable, reference_data in self.reference_data.items():
            try:
                reference_cube = self._safely_preprocess(
                    reference_data.get_cube(self.data_root_dir, variable),
                )
            except Exception as exc:
                msg = (
                    f"Failed to get reference data {reference_data.id} for "
                    f"variable {variable}: {exc}"
                )
                if self._fail_on_missing_data:
                    raise ValueError(msg) from exc
                logger.warning(msg)
                continue
            reference_cubes[variable] = reference_cube

            # Load cube data ahead to avoid potential bug in ESMValCore (see
            # https://github.com/ESMValGroup/ESMValCore/pull/3038)
            reference_cube.data  # noqa: B018

        # Process data to benchmark
        logger.debug("Loading to_benchmark data")
        diagnostic_output = self._get_output_of_to_benchmark_data(
            data,
            data_information,
            reference_cubes,
        )

        # Process other data
        logger.debug("Loading other data")
        other_output = self._get_output_of_other_data(reference_cubes)
        diagnostic_output = DiagnosticOutput.union(diagnostic_output, other_output)

        # Add reference data to table
        reference_output = self._get_output_of_reference_data(reference_cubes)
        diagnostic_output = DiagnosticOutput.union(diagnostic_output, reference_output)

        return diagnostic_output  # noqa: RET504

    def _get_empty_data_sources_table(self) -> ibis.Table:
        """Get empty data_sources table with correct schema."""
        example_data_source_info = self.reference_data[self.variables[0]].information
        return ibis.memtable(
            pd.DataFrame(
                {
                    f.name: pd.Series(dtype="str")
                    for f in fields(example_data_source_info)
                },
            ),
        )

    def _get_empty_raw_output_table(self) -> ibis.Table | None:
        """Get empty raw_output table with correct schema."""
        if not self.OUTPUT_DATA_TABLE:
            return None

        raw_output_table = ibis.memtable(
            pd.DataFrame(
                {
                    self.DATA_ID_COLUMN: pd.Series(dtype="str"),
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
            raw_output_table = raw_output_table.cast(time="timestamp")
        return raw_output_table

    def _get_empty_metrics_table(self) -> ibis.Table | None:
        """Get empty metrics table with correct schema."""
        if not self.OUTPUT_METRICS_TABLE:
            return None

        return ibis.memtable(
            pd.DataFrame(
                {
                    "var_id": pd.Series(dtype="str"),
                    **dict.fromkeys(
                        self._get_metric_names(),
                        pd.Series(dtype=self.DTYPE),
                    ),
                    self.DATA_ID_COLUMN: pd.Series(dtype="str"),
                    self.REFERENCE_DATA_ID_COLUMN: pd.Series(dtype="str"),
                },
            ),
        )

    def _get_empty_variables_table(self) -> ibis.Table:
        """Get empty variables table with correct schema.

        Note
        ----
        This code is far from ideal, but since the number of variables is in
        the order of 10s this should not be a problem.

        """
        variables_df = self._get_variables_table().to_pandas()
        empty_variables_df = pd.DataFrame(columns=variables_df.columns).astype(
            variables_df.dtypes,
        )
        return ibis.memtable(empty_variables_df)

    def _get_metrics_table(
        self,
        cube: Cube,
        reference_cube: Cube,
        *,
        variable: Variable,
        data_id: str,
    ) -> ibis.Table:
        """Get metrics table."""
        metrics = ibis.memtable(
            pd.DataFrame(
                {
                    "var_id": variable.id,
                    **self._get_metrics(cube, reference_cube),
                    self.DATA_ID_COLUMN: data_id,
                    self.REFERENCE_DATA_ID_COLUMN: self.reference_data[variable].id,
                },
                index=[0],
            ),
        )
        cast_operations = dict.fromkeys(self._get_metric_names(), self.DTYPE)
        return metrics.cast(**cast_operations)

    def _add_missing_columns(self, raw_output_table: ibis.Table) -> ibis.Table:
        """Add missing columns to data table (one column per variable)."""
        missing_columns = {v.id for v in self.variables} - set(raw_output_table.columns)
        if missing_columns:
            raw_output_table = raw_output_table.mutate(
                **{col: ibis.literal(None) for col in missing_columns},
            )
            raw_output_table = raw_output_table.cast(
                **dict.fromkeys(missing_columns, self.DTYPE),
            )
        new_col_order = (
            self.DATA_ID_COLUMN,
            self.DATA_TYPE_COLUMN,
            *self.final_coordinate_names,
            *sorted(v.id for v in self.variables),
        )
        return raw_output_table.select(new_col_order)

    def _get_cube(self, data: CubeList | Dataset, variable: Variable) -> Cube:
        """Get data cube."""
        cube = get_prepared_cube(data, variable)
        return self._safely_preprocess(cube)

    def _get_raw_output_table_of_cube(
        self,
        cube: Cube,
        *,
        variable: Variable,
        data_id: str,
        data_type: Literal["to_benchmark", "reference", "other"],
    ) -> ibis.Table:
        """Get data table of a single cube."""
        df = as_data_frame(cube).reset_index()
        if "time" in df.columns:
            df["time"] = df["time"].astype(str)
        raw_output_table = ibis.memtable(df)
        cast_operations = {
            c: {"time": "timestamp"}.get(c, self.DTYPE)
            for c in raw_output_table.columns
        }
        raw_output_table = raw_output_table.cast(**cast_operations)
        raw_output_table = raw_output_table.rename(**{variable.id: cube.name()})
        raw_output_table = raw_output_table.mutate(
            **{self.DATA_ID_COLUMN: ibis.literal(data_id)},
        )
        raw_output_table = raw_output_table.mutate(
            **{self.DATA_TYPE_COLUMN: ibis.literal(data_type)},
        )
        return self._add_missing_columns(raw_output_table)

    def _get_metric_names(self) -> tuple[str, ...]:
        """Get metric names."""
        return ("weighted_rmse", "weighted_pearsonr", "weighted_emd")

    def _get_metrics(self, cube: Cube, reference_cube: Cube) -> dict[str, float]:
        """Get metrics."""
        metrics: dict[str, float] = {}
        for metric in self._get_metric_names():
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Cannot check if coordinate is contiguous: .*",
                    category=IrisVagueMetadataWarning,
                    module="iris",
                )
                metrics[metric] = float(
                    distance_metric([cube], metric, reference=reference_cube)[0].data,
                )
        return metrics

    def _get_output_of_other_data(
        self,
        reference_cubes: dict[Variable, Cube],
    ) -> DiagnosticOutput:
        """Get output of other data."""
        raw_output_table = self._get_empty_raw_output_table()
        metrics_table = self._get_empty_metrics_table()
        variables_table = self._get_empty_variables_table()
        data_sources_table = self._get_empty_data_sources_table()

        for variable, other_datas in self.other_data.items():
            for other_data in other_datas:
                try:
                    other_cube = self._safely_preprocess(
                        other_data.get_cube(self.data_root_dir, variable),
                    )
                except Exception as exc:
                    msg = (
                        f"Failed to get other data {other_data.id} for variable "
                        f"{variable}: {exc}"
                    )
                    if self._fail_on_missing_data:
                        raise ValueError(msg) from exc
                    logger.warning(msg)
                    continue

                # Expand table
                if self.OUTPUT_DATA_TABLE:
                    new_raw_output_table = self._get_raw_output_table_of_cube(
                        other_cube,
                        variable=variable,
                        data_id=other_data.id,
                        data_type="other",
                    )
                    raw_output_table = ibis.union(
                        raw_output_table,
                        new_raw_output_table,
                    )

                # Expand metrics
                if self.OUTPUT_METRICS_TABLE and variable in reference_cubes:
                    new_metrics_table = self._get_metrics_table(
                        other_cube,
                        reference_cubes[variable],
                        variable=variable,
                        data_id=other_data.id,
                    )
                    metrics_table = ibis.union(metrics_table, new_metrics_table)

        # Drop superfluous rows by removing NaN entries
        if raw_output_table is not None:
            raw_output_table = self._remove_superfluous_rows(raw_output_table)

        return DiagnosticOutput(
            raw_output=raw_output_table,
            metrics=metrics_table,
            variables=variables_table,
            data_sources=data_sources_table,
        )

    def _get_output_of_reference_data(
        self,
        reference_cubes: dict[Variable, Cube],
    ) -> DiagnosticOutput:
        """Get output of other data."""
        raw_output_table = self._get_empty_raw_output_table()
        metrics_table = self._get_empty_metrics_table()  # will stay empty
        variables_table = self._get_empty_variables_table()
        data_sources_table = self._get_empty_data_sources_table()

        if raw_output_table is None:
            return DiagnosticOutput(
                raw_output=raw_output_table,
                metrics=metrics_table,
                variables=variables_table,
                data_sources=data_sources_table,
            )

        for variable, reference_cube in reference_cubes.items():
            new_raw_output_table = self._get_raw_output_table_of_cube(
                reference_cube,
                variable=variable,
                data_id=self.reference_data[variable].id,
                data_type="reference",
            )
            raw_output_table = ibis.union(raw_output_table, new_raw_output_table)

        # Drop superfluous rows by removing NaN entries
        raw_output_table = self._remove_superfluous_rows(raw_output_table)

        return DiagnosticOutput(
            raw_output=raw_output_table,
            metrics=metrics_table,
            variables=variables_table,
            data_sources=data_sources_table,
        )

    def _get_output_of_to_benchmark_data(
        self,
        data: CubeList | Dataset,
        data_information: DataSourceInformation,
        reference_cubes: dict[Variable, Cube],
    ) -> DiagnosticOutput:
        """Get output of data to benchmark."""
        data_id = data_information.id

        raw_output_table = self._get_empty_raw_output_table()
        metrics_table = self._get_empty_metrics_table()
        variables_table = self._get_variables_table()
        data_sources_table = self._get_data_sources_table(data_information)

        for variable in self.variables:
            logger.debug(f"Loading {variable} of {data_id}")
            try:
                cube = self._get_cube(data, variable)
            except Exception as exc:
                msg = f"Failed to get data {data_id} for variable {variable}: {exc}"
                if self._fail_on_missing_data:
                    raise ValueError(msg) from exc
                logger.warning(msg)
                continue

            # Expand table
            if self.OUTPUT_DATA_TABLE:
                new_raw_output_table = self._get_raw_output_table_of_cube(
                    cube,
                    variable=variable,
                    data_id=data_id,
                    data_type="to_benchmark",
                )
                raw_output_table = ibis.union(raw_output_table, new_raw_output_table)

            # Expand metrics
            if self.OUTPUT_METRICS_TABLE and variable in reference_cubes:
                new_metrics_table = self._get_metrics_table(
                    cube,
                    reference_cubes[variable],
                    variable=variable,
                    data_id=data_id,
                )
                metrics_table = ibis.union(metrics_table, new_metrics_table)

        # Drop superfluous rows by removing NaN entries
        if raw_output_table is not None:
            raw_output_table = self._remove_superfluous_rows(raw_output_table)

        return DiagnosticOutput(
            raw_output=raw_output_table,
            metrics=metrics_table,
            variables=variables_table,
            data_sources=data_sources_table,
        )

    @abstractmethod
    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess cube."""

    def _remove_superfluous_rows(self, table: ibis.Table) -> ibis.Table:
        """Drop superfluous rows by removing NaN entries.

        Note
        ----
        Currently, we first add NaNs and then remove it again. The reason for
        this is that the logic is simple and the code robust. There are
        certainly smarter and more efficient ways to do this, but since our
        tables are usually quite small, it's probably not necessary.

        """
        unique_columns = (
            self.DATA_ID_COLUMN,
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

    def _get_data_sources_table(
        self,
        *additional_data_source_information: DataSourceInformation,
    ) -> ibis.Table:
        """Get data sources table."""
        unique_data_sources_info = (
            set(additional_data_source_information)
            | {d.information for d in self.reference_data.values()}
            | {d.information for o in self.other_data.values() for d in o}
        )
        df = pd.DataFrame(unique_data_sources_info)
        df["references"] = df["references"].map(" ".join)
        return ibis.memtable(df)

    def _get_variables_table(self) -> ibis.Table:
        """Get variables table."""
        df = pd.DataFrame(self.variables)
        df = df.drop("settings", axis=1)
        df["coordinates"] = df["coordinates"].map(
            lambda r: " ".join(c["var_name"] for c in r),
        )
        df["ar6_regions_mask"] = df["ar6_regions_mask"].map(" ".join)

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
