"""Provide base class for evaluation suites."""

from __future__ import annotations

from dataclasses import fields
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ibis
import yaml
from loguru import logger

from climateeval._utils import str_to_object
from climateeval.diags.complex import ComplexDiagnostic
from climateeval.diags.simple import SimpleDiagnostic

if TYPE_CHECKING:
    from iris.cube import CubeList
    from xarray import Dataset

    from climateeval.data import DataSourceInformation
    from climateeval.diags._base import Diagnostic

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
        logger.info(f"Initializing evaluation suite '{self.name}'")
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
        data: dict[str, CubeList | Dataset] | CubeList | Dataset,
        data_information: DataSourceInformation,
        *,
        database_resource: str = "duckdb://",
        **database_kwargs: Any,
    ) -> ibis.BaseBackend:
        """Get database from evaluation suite."""
        logger.debug(f"Using database {database_resource}")

        connection = ibis.connect(database_resource, **database_kwargs)

        for diag_name, diag in self._get_diagnostics().items():
            logger.info(
                f"Processing diagnostic '{diag_name}' within suite '{self.name}'",
            )
            output = diag.get_output(data, data_information)
            connection.create_database(diag_name)
            for output_field in fields(output):
                table_name = output_field.name
                table = getattr(output, table_name)
                if table is None:
                    logger.debug(
                        f"Table '{table_name}' of diagnostic '{diag_name}' "
                        f"within suite '{self.name}' contains no data",
                    )
                    continue
                connection.create_table(table_name, table, database=diag_name)
            logger.info(
                f"Created database for diagnostic '{diag_name}' within suite "
                f"'{self.name}' at {database_resource}",
            )

        return connection

    def _get_diagnostics(self) -> dict[str, Diagnostic]:
        """Get diagnostics to run."""
        diagnostics: dict[str, Diagnostic] = {}
        for diag_index, diag_definition in enumerate(self._suite_definition):
            general_required_keys = ("name", "diagnostic")
            for required_key in general_required_keys:
                if required_key not in diag_definition:
                    msg = (
                        f"Expected key '{required_key}' in diagnostic located "
                        f"at index {diag_index} in suite '{self.name}'"
                    )
                    raise KeyError(msg)

            diag_name = diag_definition["name"]
            diag_cls = str_to_object(diag_definition["diagnostic"])

            if diag_name in diagnostics:
                msg = (
                    f"Diagnostic names must be unique for suite '{self.name}', "
                    f"got multiple diagnostics named '{diag_name}'"
                )
                raise ValueError(msg)

            diag_settings: dict[str, Any]
            if issubclass(diag_cls, SimpleDiagnostic):
                if "variables" not in diag_definition:
                    msg = (
                        f"Expected key 'variables' in simple diagnostic "
                        f"'{diag_name}' of suite '{self.name}'"
                    )
                    raise KeyError(msg)
                diag_settings = diag_definition["variables"]
            elif issubclass(diag_cls, ComplexDiagnostic):
                if "additional_diagnostic_kwargs" not in diag_definition:
                    msg = (
                        f"Expected key 'additional_diagnostic_kwargs' in complex "
                        f"diagnostic '{diag_name}' of suite '{self.name}'"
                    )
                    raise KeyError(msg)
                diag_settings = diag_definition["additional_diagnostic_kwargs"]
            else:
                msg = (
                    f"Diagnostic '{diag_name}' is not a valid diagnostic; should "
                    f"be a subclass of climateeval.diags.simple.SimpleDiagnostic or "
                    f"climateeval.diags.copmlex.ComplexDiagnostic"
                )
                raise TypeError(msg)

            diag = diag_cls.from_suite_dict(
                diag_name,
                diag_settings,
                self._variable_kwargs,
                **self._diagnostic_kwargs,
            )
            diagnostics[diag_name] = diag

        return diagnostics

    def _get_suite_defintion(self) -> list[dict[str, Any]]:
        """Get suite definition."""
        with self._suite_definition_file.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _get_suite_defintion_file(self, suite_definition_file: str | Path) -> Path:
        """Get suite definition file."""
        suite_dir = Path(str(resources.files("climateeval.suites")))
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
