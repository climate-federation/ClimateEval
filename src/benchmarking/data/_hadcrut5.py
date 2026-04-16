"""Provide HadCRUT5 data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.data._base import DataSource

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking import Variable


class HadCRUT5(DataSource):
    """Manage HadCRUT5 data."""

    def download(self, data_root_dir: Path, variable: Variable) -> list[Path]:
        """Download data."""
        return self._esmvaltool_cmorizer_prepare(
            data_root_dir,
            variable,
            supported_var_names=["tas", "tasa"],
            file_prefix_pattern="*_5.0.1.0-analysis_*",
        )
