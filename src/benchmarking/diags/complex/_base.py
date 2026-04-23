"""Provide base classes for complex diagnostics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger

from benchmarking.data import ComplexDataSource
from benchmarking.diags._utils import (
    DiagnosticOutput,
    get_data_root_dir,
    get_data_sources_table,
    get_raw_output_table,
    get_variables_table,
    union,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    import ibis
    from iris.cube import Cube, CubeList
    from xarray import Dataset

    from benchmarking import Variable
    from benchmarking.data import DataSourceInformation
    from benchmarking.diags._utils import DataTypeValue

logger = logger.opt(colors=True)


class ComplexDiagnostic(ABC):
    """Manage complex diagnostics."""

    REQUIRED_DATA_KEYS: ClassVar[tuple[str, ...]] = ()
    """Dictionary keys required for multi-data input."""

    def __init__(
        self,
        *,
        data_root_dir: Path | None = None,
        fail_on_missing_data: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize class instance."""
        logger.debug(f"Initializing complex diagnostic {self.name}")
        self._data_root_dir = get_data_root_dir(data_root_dir)
        self._fail_on_missing_data = fail_on_missing_data
        self._kwargs = kwargs

    @property
    def data_root_dir(self) -> Path:
        """Get path to data directory."""
        return self._data_root_dir

    @property
    def name(self) -> str:
        """Get diagnostic name."""
        return self.__class__.__name__

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

        data_sources_table = get_data_sources_table(all_data_sources)
        variables_table = get_variables_table(all_variables)

        return DiagnosticOutput(
            raw_output=union(*raw_output_tables),
            metrics=None,
            variables=variables_table,
            data_sources=data_sources_table,
        )

    @abstractmethod
    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""

    def _check_required_dict_keys(self, dict_: dict[str, Any], dict_name: str) -> None:
        """Check if required dictionary keys are present."""
        if set(dict_) != set(self.REQUIRED_DATA_KEYS):
            msg = (
                f"Expected keys {self.REQUIRED_DATA_KEYS} for {dict_name} dictionary, "
                f"got {tuple(dict_)}"
            )
            raise ValueError(msg)

    def _get_raw_output_tables(
        self,
        raw_output: dict[Variable, Cube],
        *,
        data_id: str,
        data_type: DataTypeValue,
    ) -> Generator[ibis.Table]:
        """Get raw output table."""
        for variable, cube in raw_output.items():
            yield get_raw_output_table(
                cube,
                variable=variable,
                data_id=data_id,
                data_type=data_type,
            )

    def _get_other_data(self) -> Generator[ComplexDataSource]:
        """Get other data."""
        yield from ()
