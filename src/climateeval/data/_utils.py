"""Provide data source utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

logger = logger.opt(colors=True)


def facet_to_str(facet_value: Any) -> str:
    """Convert a facet value to a string.

    Lists and tuples of facet values (e.g. ``exp = ["historical", "ssp245"]``
    for experiments that are concatenated into a single time series) are joined
    with ``-`` so that they can be used in data source IDs and file names.

    Parameters
    ----------
    facet_value:
        Facet value.

    Returns
    -------
    :
        String representation of the facet value.

    """
    if isinstance(facet_value, list | tuple):
        return "-".join(str(value) for value in facet_value)
    return str(facet_value)


@dataclass(frozen=True, order=True, kw_only=True)
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


def get_cmip6_reference(*, mip: str, institute: str, model: str) -> str:
    """Get reference for CMIP model."""
    return f"https://www.wdc-climate.de/ui/cmip6?input=CMIP6.{mip}.{institute}.{model}"
