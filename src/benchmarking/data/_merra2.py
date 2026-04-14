"""Provide MERRA2 data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.data._base import DataSource

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking import Variable


class MERRA2(DataSource):
    """Manage MERRA2 data."""

    def _download(self, variable: Variable) -> list[Path]:
        """Download data."""
        return self._esmvaltool_cmorizer_download(
            variable,
            "MERRA2",
            supported_var_names=[
                "sm",
                "clt",
                "pr",
                "evspsbl",
                "hfss",
                "hfls",
                "huss",
                "prc",
                "prsn",
                "prw",
                "ps",
                "psl",
                "rlds",
                "rldscs",
                "rlus",
                "rlut",
                "rlutcs",
                "rsds",
                "rsdscs",
                "rsdt",
                "tas",
                "tasmin",
                "tasmax",
                "tauu",
                "tauv",
                "ts",
                "uas",
                "vas",
                "rsus",
                "rsuscs",
                "rsut",
                "rsutcs",
                "ta",
                "ua",
                "va",
                "tro3",
                "zg",
                "hus",
                "wap",
                "hur",
                "cl",
                "clw",
                "cli",
                "clwvi",
                "clivi",
            ],
        )
