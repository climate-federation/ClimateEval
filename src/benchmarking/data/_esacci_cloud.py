"""Provide ESACCI-CLOUD data."""

from __future__ import annotations

from benchmarking.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from benchmarking.data._utils import DataSourceInformation


class ESACCICloud(ESMValToolCMORizerDataSource):
    """Manage ESACCI-CLOUD data."""

    _INFORMATION = DataSourceInformation(
        name="ESACCI-CLOUD",
        category="observation",
        institute="DWD",
        references=("https://doi.org/10.5676/DWD/ESA_Cloud_cci/AVHRR-PM/V003",),
    )

    SUPPORTED_VAR_NAMES = (
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
    )
    _FILE_PREFIX_PATTERN = "*_v3.0-AVHRR-AMPM_*"
