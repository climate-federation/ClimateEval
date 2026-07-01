"""Provide ESACCI-SST data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class ESACCISST(ESMValToolCMORizerDataSource):
    """Manage ESACCI-SST data."""

    _information = DataSourceInformation(
        name="ESACCI-SST",
        category="observation",
        institute="ESA",
        references=("https://doi.org/10.5285/62c0f97b1eac4e0197a674870afe1ee6",),
    )

    _supported_var_names = ("tos",)
