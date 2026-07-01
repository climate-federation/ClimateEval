"""Provide Met Office EN4 ocean temperature/salinity data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class EN4(ESMValToolCMORizerDataSource):
    """Manage EN4 data.

    EN4 is the Met Office Hadley Centre EN4 objective analysis of subsurface
    ocean temperature and salinity, CMORized via ESMValTool. It provides
    ``thetao`` (and ``so``), from which the integrated ocean heat content
    (``phcint``) is derived.
    """

    _information = DataSourceInformation(
        name="EN4",
        category="reanalysis",
        institute="Met Office Hadley Centre",
        references=("https://doi.org/10.1002/2013JC009067",),
    )

    _supported_var_names = ("thetao", "so")
