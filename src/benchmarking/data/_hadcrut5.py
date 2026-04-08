"""Manage datasets."""

from __future__ import annotations

from benchmarking.data import DataSource


class HadCRUT5(DataSource):
    """Manage HadCRUT5 data."""

    def _download(self) -> None:
        """Download data."""
        self._esmvaltool_download(
            "HadCRUT5",
            file_prefix_pattern="*_5.0.1.0-analysis_*_",
        )
