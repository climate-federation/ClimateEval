"""Manage HadCRUT5 data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.data import DataSource

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking import Variable


class HadCRUT5(DataSource):
    """Manage HadCRUT5 data."""

    def _download(self, variable: Variable) -> list[Path]:
        """Download data."""
        return self._esmvaltool_download(
            variable,
            "HadCRUT5",
            supported_var_names=["tas", "tasa"],
            file_prefix_pattern="*_5.0.1.0-analysis_*_",
        )
