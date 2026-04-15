"""Provide ESACCI-CLOUD data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.data._base import DataSource

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking import Variable


class ESACCICloud(DataSource):
    """Manage ESACCI-CLOUD data."""

    def __init__(self, data_dir: Path) -> None:
        """Initialize class instance."""
        super().__init__(data_dir)
        self._name = "ESACCI-CLOUD"

    def _download(self, variable: Variable) -> list[Path]:
        """Download data."""
        return self._esmvaltool_cmorizer_prepare(
            variable,
            "ESACCI-CLOUD",
            supported_var_names=[
                "clivi",
                "clt",
                "cltStderr",
                "clwvi",
                "lwp",
                "rlut",
                "rlutcs",
                "rsut",
                "rsutcs",
                "rsdt",
                "rlus",
                "rsus",
                "rsuscs",
            ],
        )
