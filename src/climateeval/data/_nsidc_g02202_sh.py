"""Provide NSIDC-G02202 (Southern Hemisphere) sea-ice concentration data.

NSIDC-G02202 is the NOAA/NSIDC Climate Data Record of passive-microwave sea-ice
concentration. Here the Southern Hemisphere record is used, CMORized via
ESMValTool (tier 3).
"""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class NSIDCG02202SH(ESMValToolCMORizerDataSource):
    """Manage NSIDC-G02202 Southern Hemisphere sea-ice concentration data."""

    _information = DataSourceInformation(
        name="NSIDC-G02202-sh",
        category="reanalysis",
        institute="NSIDC",
        references=("https://doi.org/10.7265/efmz-2t65",),
    )

    _supported_var_names = ("siconc",)

    _tier = 3
