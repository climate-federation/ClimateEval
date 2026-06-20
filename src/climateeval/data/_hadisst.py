"""Provide HadISST data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class HadISST(ESMValToolCMORizerDataSource):
    """Manage HadISST data."""

    _information = DataSourceInformation(
        name="HadISST",
        category="reanalysis",
        institute="Met Office Hadley Centre",
        references=("https://doi.org/10.1029/2002JD002670",),
    )

    _supported_var_names = ("tos", "siconc")
