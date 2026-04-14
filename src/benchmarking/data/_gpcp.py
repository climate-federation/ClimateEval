"""Provide GPCP data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.data._base import DataSource

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking import Variable


class GPCP(DataSource):
    """Manage GPCP data."""

    def _download(self, variable: Variable) -> list[Path]:
        """Download data."""
        return self._esmvaltool_intake_download(
            variable,
            supported_var_names=["pr"],
            project="obs4MIPs",
            dataset="GPCP-3-3",
            mip="Amon",
            short_name=variable.var_name,
            frequency=variable.frequency,
        )
