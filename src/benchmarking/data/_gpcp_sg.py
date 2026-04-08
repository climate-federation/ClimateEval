"""Manage datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.data import DataSource

if TYPE_CHECKING:
    from benchmarking import Config, Variable


class GPCPSG(DataSource):
    """Manage GPCP-SG data."""

    def __init__(self, variable: str | Variable, config: Config) -> None:
        """Initialize class instance."""
        super().__init__(variable, config)
        self._name = "GPCP-SG"

    def _download(self) -> None:
        """Download data."""
        self._intake_esgf_download(
            experiment_id="historical",
            source_id="CanESM5",
            frequency="mon",
            variable_id=["tas", "gpp"],
            member_id="r1i1p1f1",
        )
