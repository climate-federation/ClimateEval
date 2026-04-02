"""Manage datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking._data import Data

if TYPE_CHECKING:
    from benchmarking._config import Config


class HadCRUT5(Data):
    """Manage HadCRUT5 data."""

    def _download(self, config: Config) -> None:
        """Download data."""
        self._esmvaltool_download(
            "HadCRUT5",
            config,
            file_prefix_pattern="*_5.0.1.0-analysis_*_",
        )
