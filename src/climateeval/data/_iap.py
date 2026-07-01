"""Provide IAP ocean temperature data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class IAP(ESMValToolCMORizerDataSource):
    """Manage IAP data.

    IAP is the Institute of Atmospheric Physics gridded ocean temperature
    analysis, CMORized via ESMValTool. It provides ``thetao``, from which the
    integrated ocean heat content (``phcint``) is derived.
    """

    _information = DataSourceInformation(
        name="IAP",
        category="reanalysis",
        institute="Institute of Atmospheric Physics",
        references=("https://doi.org/10.1175/JCLI-D-16-0743.1",),
    )

    _supported_var_names = ("thetao",)
