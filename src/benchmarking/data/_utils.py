"""Provide data source utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from benchmarking.data._base import DataSource

logger = logger.opt(colors=True)


@dataclass(frozen=True, kw_only=True)
class DataSourceInformation:
    """Manage data source information."""

    id: str = field(init=False)

    name: str = field(repr=False)
    category: str = field(repr=False)  # e.g., observation, CMIP6 model, etc.
    institute: str = field(default="", repr=False)
    exp: str = field(default="", repr=False)
    variant: str = field(default="", repr=False)  # e.g., r1i1p1f1, etc.
    references: tuple[str, ...] = field(default_factory=tuple, repr=False)

    def __post_init__(self) -> None:
        """Initialize class instance."""
        id_fields = ("category", "name", "exp", "variant")

        id_fields_vals = [
            getattr(self, f).replace(" ", "-")
            for f in id_fields
            if getattr(self, f) != ""
        ]

        # See https://docs.python.org/3/library/dataclasses.html#frozen-instances
        object.__setattr__(self, "id", "_".join(id_fields_vals))


@dataclass(frozen=True, kw_only=True)
class DiagnosticData:
    """Manage diagnostic data."""

    reference: type[DataSource]
    other: tuple[type[DataSource], ...] = ()
