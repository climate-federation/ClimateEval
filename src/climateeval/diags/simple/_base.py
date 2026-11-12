"""Provide base classes for simple diagnostics."""

from __future__ import annotations

import warnings
from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar, Self

import ibis
import pandas as pd
from esmvalcore.preprocessor import distance_metric
from iris import Constraint
from iris.warnings import IrisVagueMetadataWarning
from loguru import logger

from climateeval import Variable, VerticalLevel
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._utils import (
    check_coordinates,
    clip_to_timerange_overlap,
    get_prepared_cube,
    get_time_window,
    str_to_object,
)
from climateeval.diags._base import Diagnostic, DiagnosticOutput
from climateeval.diags._utils import union
from climateeval.diags.simple._utils import SimpleDiagnosticInputData
from climateeval.exceptions import MetricError, MissingDataError

if TYPE_CHECKING:
    from pathlib import Path

    from iris.cube import Cube, CubeList
    from xarray import Dataset

    from climateeval import Coordinate
    from climateeval.data import DataSource, DataSourceInformation


logger = logger.opt(colors=True)


@dataclass(frozen=True, kw_only=True)
class _IntermediateDiagnosticOutput:
    """Manage intermediate diagnostic output."""

    raw_output: list[ibis.Table]
    metrics: list[ibis.Table]


class SimpleDiagnostic(Diagnostic):
    """Manage simple diagnostics."""

    _output_data_table: ClassVar[bool] = True
    """Output diagnostic data when creating database."""

    _output_metrics_table: ClassVar[bool] = True
    """Output metrics data when creating database."""

    _final_coordinates: ClassVar[tuple[Coordinate, ...]] = ()
    """Coordinates of preprocessed output."""

    def __init__(  # noqa: PLR0913
        self,
        name: str,
        variables_and_data: dict[Variable, SimpleDiagnosticInputData],
        *,
        data_root_dir: Path | str | None = None,
        fail_on_missing_data: bool = False,
        download_missing_data: bool = True,
        fail_on_metric_error: bool = True,
        calculate_metrics: bool | None = None,
        clip_reference_timerange: bool = False,
        grid: str | None = None,
    ) -> None:
        """Initialize class instance.

        Parameters
        ----------
        clip_reference_timerange:
            If True, a reference that covers the requested window only partly
            (e.g. CERES-EBAF, which starts in 2000, for a 1985-2024 window)
            is clipped to the overlap instead of being dropped as missing
            data. A reference without any overlap is still missing data.
        grid:
            Regular grid (any ESMValCore ``target_grid`` specification, e.g.
            ``"1x1"`` or ``"2.5x2.5"``) that the diagnostic regrids to. The
            default is ``DEFAULT_GRID`` (``"2x2"``), or ``SEA_ICE_GRID``
            (``"1x1"``) for the sea-ice diagnostics. Diagnostics that do not
            regrid ignore it.

        """
        self._name = name
        logger.debug(f"Initializing simple diagnostic '{self.name}'")

        if not variables_and_data:
            msg = "At least a single variable must be given."
            raise ValueError(msg)

        # Set up variables and reference data
        self._variables = tuple(sorted(variables_and_data.keys()))
        self._check_variables()

        self._reference_data: dict[Variable, DataSource] = {}
        self._reference_timeranges: dict[Variable, str] = {}
        for variable, diag_input_data in variables_and_data.items():
            if diag_input_data.reference is not None:
                self._reference_data[variable] = diag_input_data.reference()
                if diag_input_data.reference_timerange is not None:
                    self._reference_timeranges[variable] = (
                        diag_input_data.reference_timerange
                    )
        variable_ids = [v.id for v in self._variables]
        if len(set(variable_ids)) != len(variable_ids):
            msg = (
                f"Variable IDs must be unique for diagnostic '{self.name}', got "
                f"{variable_ids}"
            )
            raise ValueError(msg)

        # Set up other data
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

        self._data_root_dir = self._get_data_root_dir(data_root_dir)
        self._fail_on_missing_data = fail_on_missing_data
        self._download_missing_data = download_missing_data
        self._fail_on_metric_error = fail_on_metric_error
        self._clip_reference_timerange = clip_reference_timerange
        self._grid_spec = grid
        if calculate_metrics is None:
            self._calculate_metrics = self._output_metrics_table
        else:
            self._calculate_metrics = calculate_metrics

    @property
    def _final_coordinate_names(self) -> tuple[str, ...]:
        """Get names of final coordinate."""
        return tuple(c.name() for c in self._final_coordinates)

    @classmethod
    def from_suite_dict(
        cls,
        name: str,
        suite_settings: dict[str, Any],
        additional_variable_kwargs: dict[str, Any],
        **additional_diagnostic_kwargs: Any,
    ) -> Self:
        """Load simple diagnostic from suite dictionary (given by YAML file)."""
        if "variables" not in suite_settings:
            msg = (
                f"Expected key 'variables' in suite definition of simple diagnostic "
                f"'{name}'"
            )
            raise KeyError(msg)

        variables_and_data: dict[Variable, SimpleDiagnosticInputData] = {}
        for variable_kwargs_ in suite_settings["variables"]:
            variable_kwargs = dict(variable_kwargs_)
            given_reference_data = variable_kwargs.pop("reference_data", None)
            reference_data: type[DataSource] | None
            if given_reference_data is None:
                reference_data = None
            else:
                reference_data = str_to_object(given_reference_data)
            other_data = tuple(
                str_to_object(o) for o in variable_kwargs.pop("other_data", ())
            )
            if "vertical_level" in variable_kwargs:
                variable_kwargs["vertical_level"] = VerticalLevel(
                    **variable_kwargs["vertical_level"],
                )
            if "ar6_regions_mask" in variable_kwargs:
                variable_kwargs["ar6_regions_mask"] = tuple(
                    variable_kwargs["ar6_regions_mask"],
                )
            # The reference window belongs to the reference, not to the Variable
            # (which is also used for the data to benchmark and other data)
            reference_timerange = variable_kwargs.pop("reference_timerange", None)
            additional_kwargs = dict(additional_variable_kwargs)
            reference_timerange = additional_kwargs.pop(
                "reference_timerange",
                reference_timerange,
            )
            variable_kwargs.update(additional_kwargs)
            variable = Variable(**variable_kwargs)
            variables_and_data[variable] = SimpleDiagnosticInputData(
                reference=reference_data,
                other=other_data,
                reference_timerange=reference_timerange,
            )

        diagnostic_kwargs = dict(suite_settings.get("diagnostic_kwargs", {}))
        diagnostic_kwargs.update(additional_diagnostic_kwargs)

        return cls(name, variables_and_data, **diagnostic_kwargs)

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
        # Load and process reference data
        logger.debug("Loading reference data")
        reference_cubes = self._get_reference_cubes()
        reference_output = self._get_output_of_reference_data(reference_cubes)

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

        # Combine raw output tables
        raw_output_table = self._combine_raw_output_tables(
            *to_benchmark_output.raw_output,
            *other_output.raw_output,
            *reference_output.raw_output,
        )

        # Combine metrics tables
        metrics_table: ibis.Table | None
        metrics_tables = [
            *to_benchmark_output.metrics,
            *other_output.metrics,
            *reference_output.metrics,
        ]
        metrics_table = union(*metrics_tables) if metrics_tables else None

        # Metadata tables
        variables_table = self._get_variables_table(
            self._variables,
            reference_timeranges=self._reference_timeranges,
        )
        all_data_source_information = (
            [data_information]
            + [d.information for d in self._reference_data.values()]
            + [d.information for o in self._other_data.values() for d in o]
        )
        data_sources_table = self._get_data_sources_table(all_data_source_information)

        return DiagnosticOutput(
            raw_output=raw_output_table,
            metrics=metrics_table,
            variables=variables_table,
            data_sources=data_sources_table,
        )

    def _check_variables(self) -> None:
        """Check variables."""

    def _combine_raw_output_tables(
        self,
        *raw_output_tables: ibis.Table,
    ) -> ibis.Table | None:
        """Combine raw output tables."""
        if not raw_output_tables:
            return None
        raw_output_table = union(*raw_output_tables)
        unique_columns = (
            self._data_id_column,
            self._data_type_column,
            *self._final_coordinate_names,
        )
        raw_output_table = self._remove_superfluous_rows(
            raw_output_table,
            unique_columns=unique_columns,
            variable_columns=(v.id for v in self._variables),
        )
        return self._sort_raw_output_columns(raw_output_table)

    def _get_metrics_table(
        self,
        cube: Cube,
        reference_cube: Cube,
        *,
        variable: Variable,
        data_id: str,
    ) -> ibis.Table:
        """Get metrics table."""
        return ibis.memtable(
            pd.DataFrame(
                {
                    self._data_id_column: data_id,
                    self._reference_data_id_column: self._reference_data[variable].id,
                    self._variable_id_column: variable.id,
                    **self._get_metrics(cube, reference_cube),
                },
                index=[0],
            ),
        )

    def _get_cube(self, data: CubeList | Dataset, variable: Variable) -> Cube:
        """Get data cube."""
        cube = get_prepared_cube(data, variable)
        return self._safely_preprocess(cube, variable)

    def _align_time(self, cube: Cube, reference_cube: Cube) -> tuple[Cube, Cube]:
        """Restrict two time series to their common period.

        Only acts if both cubes have a time dimension of different length, which
        happens when the reference has its own (``reference_timerange``) or a
        clipped (``clip_reference_timerange``) window. Time-mean diagnostics
        have no time dimension and are not affected: their model and reference
        windows may differ freely. Equal-length series are never touched.

        Raises
        ------
        MissingDataError
            The two series do not share any period.

        """
        coords = [c.coords("time", dim_coords=True) for c in (cube, reference_cube)]
        if not all(coords):
            return cube, reference_cube
        time, ref_time = (c[0] for c in coords)
        if time.shape == ref_time.shape:
            return cube, reference_cube

        def key(date: Any) -> tuple[int, int, int]:
            # Tuples compare across calendars (cftime objects do not)
            return (date.year, date.month, date.day)

        def span(coord: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
            if not coord.has_bounds() and coord.shape[0] > 1:
                coord = coord.copy()
                coord.guess_bounds()
            values = coord.bounds if coord.has_bounds() else coord.points[:, None]
            dates = coord.units.num2date(values)
            return key(dates[0, 0]), key(dates[-1, -1])

        start = max(span(time)[0], span(ref_time)[0])
        end = min(span(time)[1], span(ref_time)[1])

        def in_period(cell: Any) -> bool:
            return start <= key(cell.point) < end

        constraint = Constraint(time=in_period)
        aligned = [c.extract(constraint) for c in (cube, reference_cube)]
        if aligned[0] is None or aligned[1] is None:
            msg = (
                f"Model and reference time series of diagnostic '{self.name}' "
                f"have no common period"
            )
            raise MissingDataError(msg)
        logger.info(
            f"Computing metrics of '{self.name}' on the common period "
            f"{get_time_window(aligned[0])} (model {get_time_window(cube)}, "
            f"reference {get_time_window(reference_cube)})",
        )
        return aligned[0], aligned[1]

    def _get_metrics(self, cube: Cube, reference_cube: Cube) -> dict[str, pd.Series]:
        """Get metrics."""
        cube, reference_cube = self._align_time(cube, reference_cube)
        metrics: dict[str, pd.Series] = {}
        for metric in ("weighted_rmse", "weighted_pearsonr", "weighted_emd"):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"Cannot check if coordinate is contiguous: .*",
                    category=IrisVagueMetadataWarning,
                    module="iris",
                )
                with setup_esmvaltool_config_and_logging():
                    try:
                        metric_cube = distance_metric(
                            [cube],
                            metric,
                            reference=reference_cube,
                        )[0]
                    except Exception as exc:
                        msg = (
                            f"Calculation of metric '{metric}' of diagnostic "
                            f"'{self.name}' for cube {cube.summary(shorten=True)} "
                            f"failed: {exc}"
                        )
                        raise ValueError(msg) from exc
                    metrics[metric] = pd.Series(metric_cube.data, dtype=self._dtype)
        return metrics

    def _get_output_of_other_data(
        self,
        reference_cubes: dict[Variable, Cube],
    ) -> _IntermediateDiagnosticOutput:
        """Get output of other data."""
        raw_output_tables: list[ibis.Table] = []
        metrics_tables: list[ibis.Table] = []

        for variable, other_datas in self._other_data.items():
            for other_data in other_datas:
                try:
                    other_cube = self._safely_preprocess(
                        other_data.get_cube(
                            self.data_root_dir,
                            variable,
                            download_missing_data=self._download_missing_data,
                        ),
                        variable,
                    )
                except Exception as exc:  # noqa: BLE001
                    self._handle_missing_data(other_data.id, (variable,), exc)
                    continue

                # Expand raw output
                if self._output_data_table:
                    raw_output_tables.append(
                        self._get_raw_output_table(
                            other_cube,
                            variable=variable,
                            data_id=other_data.id,
                            data_type="other",
                        ),
                    )

                # Expand metrics
                if not (self._calculate_metrics and variable in reference_cubes):
                    continue
                try:
                    metrics_tables.append(
                        self._get_metrics_table(
                            other_cube,
                            reference_cubes[variable],
                            variable=variable,
                            data_id=other_data.id,
                        ),
                    )
                except Exception as exc:  # noqa: BLE001
                    self._handle_metric_error(other_data.id, variable, exc)

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

        if not self._output_data_table:
            return _IntermediateDiagnosticOutput(
                raw_output=raw_output_tables,
                metrics=metrics_tables,
            )

        for variable, reference_cube in reference_cubes.items():
            raw_output_tables.append(
                self._get_raw_output_table(
                    reference_cube,
                    variable=variable,
                    data_id=self._reference_data[variable].id,
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

        for variable in self._variables:
            logger.debug(f"Loading {variable} of {data_id}")
            try:
                cube = self._get_cube(data, variable)
            except Exception as exc:  # noqa: BLE001
                self._handle_missing_data(data_id, (variable,), exc)
                continue

            # Expand raw output
            if self._output_data_table:
                raw_output_tables.append(
                    self._get_raw_output_table(
                        cube,
                        variable=variable,
                        data_id=data_id,
                        data_type="to_benchmark",
                    ),
                )

            # Expand metrics
            if not self._calculate_metrics:
                continue
            if variable not in reference_cubes:
                logger.warning(
                    f"No reference data available for {variable}: skipping "
                    f"metrics calculation",
                )
                continue
            try:
                metrics_tables.append(
                    self._get_metrics_table(
                        cube,
                        reference_cubes[variable],
                        variable=variable,
                        data_id=data_id,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                self._handle_metric_error(data_id, variable, exc)

        return _IntermediateDiagnosticOutput(
            raw_output=raw_output_tables,
            metrics=metrics_tables,
        )

    def _get_reference_cubes(self) -> dict[Variable, Cube]:
        """Get reference cubes."""
        reference_cubes: dict[Variable, Cube] = {}
        for variable, reference_data in self._reference_data.items():
            try:
                reference_cube = self._safely_preprocess(
                    self._get_raw_reference_cube(reference_data, variable),
                    variable,
                )
            except Exception as exc:  # noqa: BLE001
                self._handle_missing_data(reference_data.id, (variable,), exc)
                continue
            reference_cubes[variable] = reference_cube

            # Load cube data ahead to avoid potential bug in ESMValCore (see
            # https://github.com/ESMValGroup/ESMValCore/pull/3038)
            reference_cube.data  # noqa: B018
        return reference_cubes

    def _get_raw_reference_cube(
        self,
        reference_data: DataSource,
        variable: Variable,
    ) -> Cube:
        """Get the reference cube for its own window (not yet preprocessed).

        The window is the variable's ``timerange`` unless the suite gives a
        ``reference_timerange``. With ``clip_reference_timerange``, a reference
        that does not reach the requested window is clipped to the overlap.
        """
        timerange = self._reference_timeranges.get(variable)
        reference_variable = (
            variable if timerange is None else replace(variable, timerange=timerange)
        )

        def get_cube(var: Variable) -> Cube:
            return reference_data.get_cube(
                self.data_root_dir,
                var,
                download_missing_data=self._download_missing_data,
            )

        try:
            return get_cube(reference_variable)
        except MissingDataError:
            if (
                not self._clip_reference_timerange
                or reference_variable.timerange == "*"
            ):
                raise
            logger.debug(
                f"{reference_data.id} does not cover {reference_variable.timerange} "
                f"for {variable}: trying to clip to the overlap",
            )

        # Load without a window (so nothing is checked or clipped), then keep
        # the overlap; no overlap at all raises MissingDataError again.
        cube = clip_to_timerange_overlap(
            get_cube(replace(reference_variable, timerange="*")),
            reference_variable.timerange,
        )
        if cube.coords("time"):
            logger.info(
                f"Reference {reference_data.id} of {variable} only partly covers "
                f"{reference_variable.timerange}: clipped to "
                f"{get_time_window(cube)}",
            )
        return cube

    def _handle_metric_error(
        self,
        data_id: str,
        variable: Variable,
        exception: Exception,
    ) -> None:
        """Handle metric calculation error."""
        msg = f"Failed to calculate metrics for {variable} of {data_id}: {exception}"
        if self._fail_on_metric_error:
            raise MetricError(msg) from exception
        logger.warning(msg)

    @abstractmethod
    def _preprocess(self, cube: Cube, variable: Variable) -> Cube:
        """Preprocess cube."""

    def _safely_preprocess(self, cube: Cube, variable: Variable) -> Cube:
        """Safely preprocess cube."""
        cube = self._preprocess(cube, variable)
        return check_coordinates(cube, self._final_coordinates)

    def _sort_raw_output_columns(self, raw_output_table: ibis.Table) -> ibis.Table:
        """Sort columns of raw output table."""
        present = set(raw_output_table.columns)
        missing = [v.id for v in self._variables if v.id not in present]
        if missing and self._fail_on_missing_data:
            msg = f"Missing variable columns in raw output: {missing}"
            raise MissingDataError(msg)
        new_column_order = (
            self._data_id_column,
            self._data_type_column,
            *self._final_coordinate_names,
            *sorted(v.id for v in self._variables if v.id in present),
        )
        return raw_output_table.select(new_column_order)
