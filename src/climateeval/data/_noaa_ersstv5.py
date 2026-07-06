"""Provide NOAA-ERSSTv5 data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class NOAAERSSTv5(ESMValToolCMORizerDataSource):
    """Manage NOAA-ERSSTv5 data."""

    _information = DataSourceInformation(
        name="NOAA-ERSSTv5",
        category="reanalysis",
        institute="NOAA",
        references=("https://doi.org/10.1175/JCLI-D-16-0836.1",),
    )

    _supported_var_names = ("tos",)
