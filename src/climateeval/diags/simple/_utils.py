"""Provide simple diagnostic utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from climateeval.data import DataSource, DataSourceGenerator

logger = logger.opt(colors=True)

_TIMERANGE_RE = re.compile(r"\d{4}(\d{2}(\d{2})?)?/\d{4}(\d{2}(\d{2})?)?")


@dataclass(frozen=True, kw_only=True)
class SimpleDiagnosticInputData:
    """Manage input data for simple diagnostics."""

    reference: type[DataSource] | None
    other: tuple[type[DataSource | DataSourceGenerator], ...] = ()
    reference_timerange: str | None = None
    """Window (``"YYYYMMDD/YYYYMMDD"``) of the reference only.

    ``None`` (default) means the reference uses the variable's ``timerange``,
    like the data to benchmark and the other data.
    """

    def __post_init__(self) -> None:
        """Validate the reference timerange."""
        if self.reference_timerange is not None and not _TIMERANGE_RE.fullmatch(
            self.reference_timerange,
        ):
            msg = (
                f"reference_timerange must have the form 'YYYYMMDD/YYYYMMDD' "
                f"(YYYY or YYYYMM are also accepted), got "
                f"'{self.reference_timerange}'"
            )
            raise ValueError(msg)
