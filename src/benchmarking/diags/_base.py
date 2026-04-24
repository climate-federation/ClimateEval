"""Provide diagnostic base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

from loguru import logger

if TYPE_CHECKING:

    import ibis
    from iris.cube import CubeList
    from xarray import Dataset

    from benchmarking.data import DataSourceInformation

logger = logger.opt(colors=True)


class Diagnostic(ABC):
    """Manage diagnostics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Diagnostic name."""

    @classmethod
    @abstractmethod
    def from_suite_dict(
        cls,
        suite_settings: dict[str, Any],
        additional_variable_kwargs: dict[str, Any],
        **additional_diagnostic_kwargs: Any,
    ) -> Self:
        """Load diagnostic from suite dictionary (given by YAML file)."""

    @abstractmethod
    def get_output(
        self,
        data: dict[str, CubeList | Dataset],
        data_information: DataSourceInformation,
    ) -> DiagnosticOutput:
        """Get diagnostic output."""


@dataclass(frozen=True, kw_only=True)
class DiagnosticOutput:
    """Manage diagnostic output."""

    raw_output: ibis.Table | None
    metrics: ibis.Table | None
    variables: ibis.Table
    data_sources: ibis.Table
