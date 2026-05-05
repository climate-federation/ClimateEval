"""Provide base classes for complex diagnostics."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Self

from loguru import logger

from climateeval.data import ComplexDataSource
from climateeval.diags._base import Diagnostic, DiagnosticOutput
from climateeval.diags._utils import (
    union,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    import ibis
    from iris.cube import Cube, CubeList
    from xarray import Dataset

    from climateeval import Variable
    from climateeval.data import DataSourceInformation
    from climateeval.diags._utils import DataTypeValue

logger = logger.opt(colors=True)


class ComplexDiagnostic(Diagnostic):
    """Manage complex diagnostics."""

    _REQUIRED_DATA_KEYS: ClassVar[tuple[str, ...]] = ()
    """Dictionary keys required for multi-data input."""

    def __init__(
        self,
        *,
        data_root_dir: Path | None = None,
        fail_on_missing_data: bool = False,
        download_missing_data: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize class instance."""
        logger.debug(f"Initializing complex diagnostic {self.name}")
        self._data_root_dir = self._get_data_root_dir(data_root_dir)
        self._fail_on_missing_data = fail_on_missing_data
        self._download_missing_data = download_missing_data
        self._kwargs = kwargs

    @classmethod
    def from_suite_dict(
        cls,
        suite_settings: dict[str, Any] | None,
        additional_variable_kwargs: dict[str, Any],  # noqa: ARG003
        **additional_diagnostic_kwargs: Any,
    ) -> Self:
        """Load complex diagnostic from suite dictionary (given by YAML file)."""
        if suite_settings is None:
            suite_settings = {}
        all_diagnostic_kwargs = {**suite_settings, **additional_diagnostic_kwargs}
        return cls(**all_diagnostic_kwargs)

    def get_output(
        self,
        data: dict[str, CubeList | Dataset],
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
        self._check_required_dict_keys(data, "data")
        raw_output_tables: list[ibis.Table] = []
        all_variables: list[Variable] = []
        all_data_sources: list[DataSourceInformation] = []

        # Process to_benchmark data
        logger.debug("Processing to_benchmark data")
        to_benchmark_data = ComplexDataSource(data=data, information=data_information)
        raw_output = self._calculate_raw_output(to_benchmark_data)
        raw_output_tables.extend(
            self._get_raw_output_tables(
                raw_output,
                data_id=to_benchmark_data.id,
                data_type="to_benchmark",
            ),
        )
        all_data_sources.append(to_benchmark_data.information)
        all_variables.extend(raw_output)

        # Process other data
        logger.debug("Processing other data")
        for other_data in self._get_other_data():
            other_raw_output = self._calculate_raw_output(other_data)
            raw_output_tables.extend(
                self._get_raw_output_tables(
                    other_raw_output,
                    data_id=other_data.id,
                    data_type="other",
                ),
            )
            all_data_sources.append(other_data.information)
            all_variables.extend(other_raw_output)

        data_sources_table = self._get_data_sources_table(all_data_sources)
        variables_table = self._get_variables_table(all_variables)

        diagnostic_output = DiagnosticOutput(
            raw_output=union(*raw_output_tables),
            metrics=None,
            variables=variables_table,
            data_sources=data_sources_table,
        )
        return self._postprocess_output(diagnostic_output)

    @abstractmethod
    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""

    def _check_required_dict_keys(self, dict_: dict[str, Any], dict_name: str) -> None:
        """Check if required dictionary keys are present."""
        if set(dict_) != set(self._REQUIRED_DATA_KEYS):
            msg = (
                f"Expected keys {self._REQUIRED_DATA_KEYS} for {dict_name} dictionary, "
                f"got {tuple(dict_)}"
            )
            raise ValueError(msg)

    def _get_other_data(self) -> Generator[ComplexDataSource]:
        """Get other data."""
        logger.debug(f"No other data for complex diagnostic {self.name} available")
        yield from ()

    def _get_raw_output_tables(
        self,
        raw_output: dict[Variable, Cube],
        *,
        data_id: str,
        data_type: DataTypeValue,
    ) -> Generator[ibis.Table]:
        """Get raw output table."""
        for variable, cube in raw_output.items():
            yield self._get_raw_output_table(
                cube,
                variable=variable,
                data_id=data_id,
                data_type=data_type,
            )

    def _postprocess_output(self, output: DiagnosticOutput) -> DiagnosticOutput:
        """Postprocess diagnostic output."""
        logger.debug(f"No postprocessing of output in diagnostic {self.name} defined")
        return output
