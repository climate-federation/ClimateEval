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
        return self._intake_esgf_download(
            variable,
            "obs4MIPs",
            supported_var_names=["pr"],
            variable_id=variable.var_name,
            source_id="GPCP-3-3",
            frequency=variable.frequency,
        )
