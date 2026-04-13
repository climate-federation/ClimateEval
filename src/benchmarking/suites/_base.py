"""Provide base class for evaluation suites."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ibis
from loguru import logger

if TYPE_CHECKING:
    from iris.cube import CubeList
    from xarray import Dataset

    from benchmarking import Variable
    from benchmarking.data import DiagnosticData
    from benchmarking.diags import Diagnostic

logger = logger.opt(colors=True)


class Suite(ABC):
    """Manage evaluation suites."""

    def __init__(
        self,
        *,
        diagnostic_kwargs: dict[str, Any] | None = None,
        variable_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize class instance."""
        logger.debug(f"Initializing evaluation suite {self.name}")
        if diagnostic_kwargs is None:
            diagnostic_kwargs = {}
        if variable_kwargs is None:
            variable_kwargs = {}
        self._diagnostic_kwargs = diagnostic_kwargs
        self._variable_kwargs = variable_kwargs

    @property
    def name(self) -> str:
        """Get evaluation suite name."""
        return self.__class__.__name__

    def run(
        self,
        data: CubeList | Dataset,
        data_name: str,
        *,
        database_backend: str = "duckdb",
        database_path: str | Path | None = None,
        overwrite: bool = False,
        **database_kwargs: Any,
    ) -> ibis.BaseBackend:
        """Run evaluation suite."""
        if database_path is None:
            database_path = Path.home() / ".benchmarking" / "benchmarking.ddb"
        database_path = Path(database_path)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        if database_path.is_file():
            if overwrite:
                logger.debug(f"{database_path} exists, overwriting it")
                database_path.unlink()
            else:
                msg = f"Database file {database_path} already exists"
                raise FileExistsError(msg)
        database_resource = f"{database_backend}://{database_path}"
        logger.debug(f"Using database {database_resource}")

        connection = ibis.connect(database_resource, **database_kwargs)

        variables_and_data = self._get_variables_and_data()
        for diag_cls in self._get_diagnostics():
            diagnostic = diag_cls(variables_and_data, **self._diagnostic_kwargs)
            output = diagnostic.get_output(data, data_name)
            connection.create_database(diagnostic.name)
            for output_field in fields(output):
                table_name = output_field.name
                connection.create_table(
                    table_name,
                    getattr(output, table_name),
                    database=diagnostic.name,
                )
            logger.debug(f"Created database for diagnostic {diagnostic.name}")

        return connection

    @abstractmethod
    def _get_variables_and_data(self) -> dict[Variable, DiagnosticData]:
        """Get data to evaluate."""

    @abstractmethod
    def _get_diagnostics(self) -> tuple[type[Diagnostic], ...]:
        """Get diagnostics to run."""
