"""Provide ESACCI-OC data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class ESACCIOC(ESMValToolCMORizerDataSource):
    """Manage ESACCI-OC data."""

    _information = DataSourceInformation(
        name="ESACCI-OC",
        category="observation",
        institute="ESA",
        references=("http://doi.org/10.5285/1dbe7a109c0244aaad713e078fd3059a",),
    )

    _supported_var_names = ("chl",)
