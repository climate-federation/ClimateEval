"""Provide ESACCI-CLOUD data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class ESACCIOC(ESMValToolCMORizerDataSource):
    """Manage ESACCI-OC chlorophyll-a data."""

    _INFORMATION = DataSourceInformation(
        name="ESACCI-OC",
        category="observation",
        institute="ESA",
        references=("https://climate.esa.int/en/projects/ocean-colour/",),
    )

    SUPPORTED_VAR_NAMES = ("chl",)
    _FILE_PREFIX_PATTERN = "OBS6_ESACCI-OC_sat_fv5.0_Omon_chl_*.nc"
