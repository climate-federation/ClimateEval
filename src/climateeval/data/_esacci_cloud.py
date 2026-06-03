"""Provide ESACCI-CLOUD data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class ESACCICloud(ESMValToolCMORizerDataSource):
    """Manage ESACCI-CLOUD data."""

    _information = DataSourceInformation(
        name="ESACCI-CLOUD",
        category="observation",
        institute="DWD",
        references=("https://doi.org/10.5676/DWD/ESA_Cloud_cci/AVHRR-PM/V003",),
    )

    _file_prefix_pattern = "*_v3.0-AVHRR-AMPM_*"
    _supported_var_names = (
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
