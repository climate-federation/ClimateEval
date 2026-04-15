"""Provide CERES-EBAF data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.data._base import DataSource

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking import Variable


class CERESEBAF(DataSource):
    """Manage CERES-EBAF data."""

    def __init__(self, data_dir: Path) -> None:
        """Initialize class instance."""
        super().__init__(data_dir)
        self._name = "CERES-EBAF"

    def _download(self, variable: Variable) -> list[Path]:
        """Download data."""
        return self._esmvaltool_intake_download(
            variable,
            project="obs4MIPs",
            dataset="CERES-EBAF-4-2-1",
            mip="Amon",
            short_name=variable.var_name,
            frequency=variable.frequency,
        )
