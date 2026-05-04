"""Provide base classes for simple diagnostics."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self

import ibis
import pandas as pd
from climateeval import Variable, VerticalLevel, setup_esmvaltool_config_and_logging
from climateeval._utils import check_coordinates, get_prepared_cube, str_to_class
from climateeval.diags._base import Diagnostic, DiagnosticOutput
from climateeval.diags._utils import union
from climateeval.diags.simple._utils import SimpleDiagnosticInputData
from esmvalcore.preprocessor import distance_metric
from ibis import _
from iris.warnings import IrisVagueMetadataWarning
from loguru import logger

if TYPE_CHECKING:
    from pathlib import Path

    from climateeval import Coordinate
    from climateeval.data import DataSource, DataSourceInformation
    from iris.cube import Cube, CubeList
    from xarray import Dataset


logger = logger.opt(colors=True)


@dataclass(frozen=True, kw_only=True)
class _IntermediateDiagnosticOutput:
    """Manage intermediate diagnostic output."""

    raw_output: list[ibis.Table]
    metrics: list[ibis.Table]


class SimpleDiagnostic(Diagnostic):
    """Manage simple diagnostics."""

    OUTPUT_DATA_TABLE: ClassVar[bool] = True
    """Output diagnostic data when creating database."""

    OUTPUT_METRICS_TABLE: ClassVar[bool] = True
    """Output metrics data when creating database."""

    FINAL_COORDINATES: ClassVar[tuple[Coordinate, ...]] = ()
    """Coordinates of preprocessed output."""

    def __init__(
        self,
        variables_and_data: dict[Variable, SimpleDiagnosticInputData],
        *,
        data_root_dir: Path | None = None,
        fail_on_missing_data: bool = False,
    ) -> None:
        """Initialize class instance."""
        logger.debug(f"Initializing simple diagnostic {self.name}")

        self._data_root_dir = self._get_data_root_dir(data_root_dir)

        if not variables_and_data:
            msg = "At least a single variable must be given."
            raise ValueError(msg)

        self._variables = tuple(sorted(variables_and_data.keys()))
        self._reference_data = {v: d.reference() for v, d in variables_and_data.items()}
        variable_ids = [v.id for v in self.variables]
        if len(set(variable_ids)) != len(variable_ids):
            msg = (
                f"Variable IDs must be unique for diagnostic {self.name}, got "
                f"{variable_ids}"
            )
            raise ValueError(msg)

        self._other_data: dict[Variable, tuple[DataSource, ...]] = {}
        for variable, diag_datas in variables_and_data.items():
            other_data_sources: list[DataSource] = []
            for data_source_cls in diag_datas.other:
                if hasattr(data_source_cls, "generate"):
                    other_data_sources.extend(
                        d
                        for d in data_source_cls().generate(variable)  # type: ignore[union-attr]
                    )
                else:
                    other_data_sources.append(data_source_cls())
            self._other_data[variable] = tuple(other_data_sources)

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
    def other_data(self) -> dict[Variable, tuple[DataSource, ...]]:
        """Get other data."""
        return self._other_data

    @property
    def reference_data(self) -> dict[Variable, DataSource]:
        """Get reference data."""
        return self._reference_data

    @property
    def variables(self) -> tuple[Variable, ...]:
        """Get variables."""
        return self._variables

    @classmethod
    def from_suite_dict(
        cls,
        suite_settings: dict[str, Any] | None,
        additional_variable_kwargs: dict[str, Any],
        **additional_diagnostic_kwargs: Any,
    ) -> Self:
        """Load simple diagnostic from suite dictionary (given by YAML file)."""
        if suite_settings is None:
            suite_settings = {}
        variables_and_data: dict[Variable, SimpleDiagnosticInputData] = {}
        for variable_id, variable_kwargs_ in suite_settings.items():
            variable_kwargs = dict(variable_kwargs_)
            reference_data = str_to_class(variable_kwargs.pop("reference_data"))
            other_data = tuple(
                str_to_class(o) for o in variable_kwargs.pop("other_data", ())
            )
            if "vertical_level" in variable_kwargs:
                variable_kwargs["vertical_level"] = VerticalLevel(
                    **variable_kwargs["vertical_level"],
                )
            if "ar6_regions_mask" in variable_kwargs:
                variable_kwargs["ar6_regions_mask"] = tuple(
                    variable_kwargs["ar6_regions_mask"],
                )
            variable_kwargs.update(additional_variable_kwargs)
            variable = Variable(variable_id, **variable_kwargs)
            variables_and_data[variable] = SimpleDiagnosticInputData(
                reference=reference_data,
                other=other_data,
            )
        return cls(variables_and_data, **additional_diagnostic_kwargs)

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
            except Exception as exc:  # noqa: BLE001
                self._handle_missing_data(reference_data.id, (variable,), exc)
                continue
            reference_cubes[variable] = reference_cube

            # Load cube data ahead to avoid potential bug in ESMValCore (see
            # https://github.com/ESMValGroup/ESMValCore/pull/3038)
            reference_cube.data  # noqa: B018

        # Process data to benchmark
        logger.debug("Loading to_benchmark data")
        to_benchmark_output = self._get_output_of_to_benchmark_data(
            data,
            data_information,
            reference_cubes,
        )

        # Process other data
        logger.debug("Loading other data")
        other_output = self._get_output_of_other_data(reference_cubes)

        # Add reference data to table
        reference_output = self._get_output_of_reference_data(reference_cubes)

        # Combine raw output tables and drop superfluous rows by removing NaN
        # entries
        raw_output_table: ibis.Table | None
        raw_output_tables = [
            *to_benchmark_output.raw_output,
            *other_output.raw_output,
            *reference_output.raw_output,
        ]
        if raw_output_tables:
            raw_output_table = union(*raw_output_tables)
            raw_output_table = self._remove_superfluous_rows(raw_output_table)
            raw_output_table = self._sort_raw_output_columns(raw_output_table)
        else:
            raw_output_table = None

        # Combine metrics tables
        metrics_table: ibis.Table | None
        metrics_tables = [
            *to_benchmark_output.metrics,
            *other_output.metrics,
            *reference_output.metrics,
        ]
        metrics_table = union(*metrics_tables) if metrics_tables else None

        # Metadata tables
        variables_table = self._get_variables_table(self.variables)
        all_data_source_information = (
            [data_information]
            + [d.information for d in self.reference_data.values()]
            + [d.information for o in self.other_data.values() for d in o]
        )
        data_sources_table = self._get_data_sources_table(all_data_source_information)

        return DiagnosticOutput(
            raw_output=raw_output_table,
            metrics=metrics_table,
            variables=variables_table,
            data_sources=data_sources_table,
        )

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
                    self.VARIABLE_ID_COLUMN: variable.id,
                    **self._get_metrics(cube, reference_cube),
                    self.DATA_ID_COLUMN: data_id,
                    self.REFERENCE_DATA_ID_COLUMN: self.reference_data[variable].id,
                },
                index=[0],
            ),
        )
        cast_operations = dict.fromkeys(self._get_metric_names(), self.DTYPE)
        return metrics.cast(**cast_operations)

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
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Cannot check if coordinate is contiguous: .*",
                    category=IrisVagueMetadataWarning,
                    module="iris",
                )
                with setup_esmvaltool_config_and_logging():
                    metrics[metric] = float(
                        distance_metric([cube], metric, reference=reference_cube)[
                            0
                        ].data,
                    )
        return metrics

    def _get_output_of_other_data(
        self,
        reference_cubes: dict[Variable, Cube],
    ) -> _IntermediateDiagnosticOutput:
        """Get output of other data."""
        raw_output_tables: list[ibis.Table] = []
        metrics_tables: list[ibis.Table] = []

        for variable, other_datas in self.other_data.items():
            for other_data in other_datas:
                try:
                    other_cube = self._safely_preprocess(
                        other_data.get_cube(self.data_root_dir, variable),
                    )
                except Exception as exc:  # noqa: BLE001
                    self._handle_missing_data(other_data.id, (variable,), exc)
                    continue

                # Expand table
                if self.OUTPUT_DATA_TABLE:
                    raw_output_tables.append(
                        self._get_raw_output_table(
                            other_cube,
                            variable=variable,
                            data_id=other_data.id,
                            data_type="other",
                        ),
                    )

                # Expand metrics
                if self.OUTPUT_METRICS_TABLE and variable in reference_cubes:
                    metrics_tables.append(
                        self._get_metrics_table(
                            other_cube,
                            reference_cubes[variable],
                            variable=variable,
                            data_id=other_data.id,
                        ),
                    )

        return _IntermediateDiagnosticOutput(
            raw_output=raw_output_tables,
            metrics=metrics_tables,
        )

    def _get_output_of_reference_data(
        self,
        reference_cubes: dict[Variable, Cube],
    ) -> _IntermediateDiagnosticOutput:
        """Get output of other data."""
        raw_output_tables: list[ibis.Table] = []
        metrics_tables: list[ibis.Table] = []

        if not self.OUTPUT_DATA_TABLE:
            return _IntermediateDiagnosticOutput(
                raw_output=raw_output_tables,
                metrics=metrics_tables,
            )

        for variable, reference_cube in reference_cubes.items():
            raw_output_tables.append(
                self._get_raw_output_table(
                    reference_cube,
                    variable=variable,
                    data_id=self.reference_data[variable].id,
                    data_type="reference",
                ),
            )

        return _IntermediateDiagnosticOutput(
            raw_output=raw_output_tables,
            metrics=metrics_tables,
        )

    def _get_output_of_to_benchmark_data(
        self,
        data: CubeList | Dataset,
        data_information: DataSourceInformation,
        reference_cubes: dict[Variable, Cube],
    ) -> _IntermediateDiagnosticOutput:
        """Get raw output and metrics of data to benchmark."""
        data_id = data_information.id

        raw_output_tables: list[ibis.Table] = []
        metrics_tables: list[ibis.Table] = []

        for variable in self.variables:
            logger.debug(f"Loading {variable} of {data_id}")
            try:
                cube = self._get_cube(data, variable)
            except Exception as exc:  # noqa: BLE001
                self._handle_missing_data(data_id, (variable,), exc)
                continue

            # Expand table
            if self.OUTPUT_DATA_TABLE:
                raw_output_tables.append(
                    self._get_raw_output_table(
                        cube,
                        variable=variable,
                        data_id=data_id,
                        data_type="to_benchmark",
                    ),
                )

            # Expand metrics
            if self.OUTPUT_METRICS_TABLE and variable in reference_cubes:
                metrics_tables.append(
                    self._get_metrics_table(
                        cube,
                        reference_cubes[variable],
                        variable=variable,
                        data_id=data_id,
                    ),
                )

        return _IntermediateDiagnosticOutput(
            raw_output=raw_output_tables,
            metrics=metrics_tables,
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

    def _sort_raw_output_columns(self, raw_output_table: ibis.Table) -> ibis.Table:
        """Sort columns of raw output table."""
        new_column_order = (
            self.DATA_ID_COLUMN,
            self.DATA_TYPE_COLUMN,
            *self.final_coordinate_names,
            *sorted(v.id for v in self.variables),
        )
        return raw_output_table.select(new_column_order)
