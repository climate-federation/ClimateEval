"""Manage datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking._data import Data

if TYPE_CHECKING:
    from benchmarking._config import Config


class GPCPSG(Data):
    """Manage GPCP-SG data."""

    def _download(self, config: Config) -> None:
        """Download data."""
        self._esmvaltool_download("GPCP-SG", config)
