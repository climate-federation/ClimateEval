"""Provide ESACCI-CLOUD data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.data._base import DataSource
from benchmarking.data._utils import DataSourceInformation

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking import Variable


class ESACCICloud(DataSource):
    """Manage ESACCI-CLOUD data."""

    _INFORMATION = DataSourceInformation(
        name="ESACCI-CLOUD",
        category="observation",
        institute="DWD",
        references=("https://doi.org/10.5676/DWD/ESA_Cloud_cci/AVHRR-PM/V003",),
    )

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
            file_prefix_pattern="*_v3.0-AVHRR-AMPM_*",
        )
