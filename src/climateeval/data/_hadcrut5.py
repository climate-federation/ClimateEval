"""Provide HadCRUT5 data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class HadCRUT5(ESMValToolCMORizerDataSource):
    """Manage HadCRUT5 data."""

    _information = DataSourceInformation(
        name="HadCRUT5",
        category="observation",
        institute="Met Office Hadley Centre",
        references=("https://doi.org/10.1029/2019JD032361",),
    )

    _file_prefix_pattern = "*_5.0.1.0-analysis_*"
    _supported_var_names = ("tas", "tasa")
