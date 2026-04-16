"""Provide base class for evaluation suites."""

from __future__ import annotations

from dataclasses import fields
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ibis
import yaml
from loguru import logger

from benchmarking import Variable, VerticalLevel
from benchmarking._utils import str_to_class
from benchmarking.data import DiagnosticData

if TYPE_CHECKING:
    from iris.cube import CubeList
    from xarray import Dataset

    from benchmarking.diags import Diagnostic

logger = logger.opt(colors=True)


class Suite:
    """Manage evaluation suites."""

    def __init__(
        self,
        suite_definition: str | Path,
        *,
        diagnostic_kwargs: dict[str, Any] | None = None,
        variable_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize class instance."""
        self._suite_definition_file = self._get_suite_defintion_file(suite_definition)
        self._suite_definition = self._get_suite_defintion()
        logger.info(f"Initializing evaluation suite {self.name}")
        if diagnostic_kwargs is None:
            diagnostic_kwargs = {}
        if variable_kwargs is None:
            variable_kwargs = {}
        self._diagnostic_kwargs = diagnostic_kwargs
        self._variable_kwargs = variable_kwargs

    @property
    def name(self) -> str:
        """Get evaluation suite name."""
        return self._suite_definition_file.stem

    def get_database(
        self,
        data: CubeList | Dataset,
        data_name: str,
        *,
        database_resource: str = "duckdb://",
        **database_kwargs: Any,
    ) -> ibis.BaseBackend:
        """Get database from evaluation suite."""
        logger.debug(f"Using database {database_resource}")

        connection = ibis.connect(database_resource, **database_kwargs)

        for diagnostic in self._get_diagnostics():
            logger.info(
                f"Processing diagnostic {diagnostic.name} within suite {self.name}",
            )
            output = diagnostic.get_output(data, data_name)
            connection.create_database(diagnostic.name)
            for output_field in fields(output):
                table_name = output_field.name
                connection.create_table(
                    table_name,
                    getattr(output, table_name),
                    database=diagnostic.name,
                )
            logger.info(
                f"Created database for diagnostic {diagnostic.name} within suite "
                f"{self.name} at {database_resource}",
            )

        return connection

    def _get_diagnostics(self) -> list[Diagnostic]:
        """Get diagnostics to run."""
        diagnostics: list[Diagnostic] = []
        for diag_str, variables in self._suite_definition.items():
            diag_cls: type[Diagnostic] = str_to_class(diag_str)
            variables_and_data: dict[Variable, DiagnosticData] = {}
            for variable_id, variable_kwargs in variables.items():
                all_variable_kwargs = dict(variable_kwargs)
                reference_data = str_to_class(all_variable_kwargs.pop("reference_data"))
                other_data = tuple(
                    str_to_class(o) for o in all_variable_kwargs.pop("other_data", ())
                )
                if "vertical_level" in all_variable_kwargs:
                    all_variable_kwargs["vertical_level"] = VerticalLevel(
                        **all_variable_kwargs["vertical_level"],
                    )
                all_variable_kwargs.update(self._variable_kwargs)
                variable = Variable(variable_id, **all_variable_kwargs)
                variables_and_data[variable] = DiagnosticData(
                    reference=reference_data,
                    other=other_data,
                )
            diagnostics.append(
                diag_cls(variables_and_data, **self._diagnostic_kwargs),
            )
        return diagnostics

    def _get_suite_defintion(self) -> dict[str, Any]:
        """Get suite definition."""
        with self._suite_definition_file.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _get_suite_defintion_file(self, suite_definition_file: str | Path) -> Path:
        """Get suite definition file."""
        suite_dir = Path(str(resources.files("benchmarking.suites")))
        if (
            isinstance(suite_definition_file, str)
            and (suite_dir / suite_definition_file).is_file()
        ):
            return suite_dir / suite_definition_file
        if (
            isinstance(suite_definition_file, str)
            and (suite_dir / f"{suite_definition_file}.yml").is_file()
        ):
            return suite_dir / f"{suite_definition_file}.yml"
        suite_definition_file = Path(suite_definition_file)
        if not suite_definition_file.is_file():
            msg = f"Suite definition file {suite_definition_file} does not exist"
            raise FileNotFoundError(msg)
        return suite_definition_file
