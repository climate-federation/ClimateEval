"""Provide WOA data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class WOA(ESMValToolCMORizerDataSource):
    """Manage World Ocean Atlas (WOA) data."""

    _information = DataSourceInformation(
        name="WOA",
        category="observation",
        institute="NOAA NCEI",
        references=("https://doi.org/10.7289/V5/ATLAS-NESDIS-87",),
    )

    _supported_var_names = ("so", "thetao", "no3", "o2", "po4", "si")
