"""Provide ESACCI-CLOUD data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.data._base import DataSource

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking import Variable


class ESACCICloud(DataSource):
    """Manage ESACCI-CLOUD data."""

    def __init__(self) -> None:
        """Initialize class instance."""
        super().__init__()
        self._name = "ESACCI-CLOUD"

    def download(self, data_root_dir: Path, variable: Variable) -> list[Path]:
        """Download data."""
        return self._esmvaltool_cmorizer_prepare(
            data_root_dir,
            variable,
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
