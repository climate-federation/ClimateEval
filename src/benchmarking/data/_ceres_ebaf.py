"""Manage HadCRUT5 data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.data import DataSource

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
        return self._intake_esgf_download(
            variable,
            "obs4MIPs",
            supported_var_names=[
                "rlut",
                "rlutcs",
                "rsdt",
                "rsut",
                "rsutcs",
                "rtmt",
                "rtnt",
            ],
            variable_id=variable.var_name,
            source_id="CERES-EBAF-4-2-1",
            frequency=variable.frequency,
        )
