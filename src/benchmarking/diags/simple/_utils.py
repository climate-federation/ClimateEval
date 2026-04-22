"""Provide simple diagnostic utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from benchmarking.data import DataSource, DataSourceGenerator

logger = logger.opt(colors=True)


@dataclass(frozen=True, kw_only=True)
class SimpleDiagnosticInputData:
    """Manage input data for simple diagnostics."""

    reference: type[DataSource]
    other: tuple[type[DataSource | DataSourceGenerator], ...] = ()
