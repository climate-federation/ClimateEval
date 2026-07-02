"""Provide EN4 data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class EN4(ESMValToolCMORizerDataSource):
    """Manage EN4 data.

    Warning
    -------
    Automatic download and formatting does not work since no ESMValTool
    downloading script is available.

    """

    _information = DataSourceInformation(
        name="EN4",
        category="reanalysis",
        institute="Met Office Hadley Centre",
        references=("https://doi.org/10.1002/2013JC009067",),
    )

    _supported_var_names = ("so", "sos", "thetao", "tos")
