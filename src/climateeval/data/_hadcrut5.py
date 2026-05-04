"""Provide HadCRUT5 data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class HadCRUT5(ESMValToolCMORizerDataSource):
    """Manage HadCRUT5 data."""

    _INFORMATION = DataSourceInformation(
        name="HadCRUT5",
        category="observation",
        institute="Met Office Hadley Centre",
        references=("https://doi.org/10.1029/2019JD032361",),
    )

    SUPPORTED_VAR_NAMES = ("tas", "tasa")
    _FILE_PREFIX_PATTERN = "*_5.0.1.0-analysis_*"
