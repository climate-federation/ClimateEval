"""Provide Berkeley Earth data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class BerkeleyEarth(ESMValToolCMORizerDataSource):
    """Manage Berkeley Earth data.

    The land and ocean gridded product provides near-surface temperature
    anomalies (``tasa``) and, by adding the CMORizer's climatology, absolute
    near-surface temperatures (``tas``).

    """

    _information = DataSourceInformation(
        name="BerkeleyEarth",
        category="observation",
        institute="Berkeley Earth",
        references=("https://doi.org/10.4172/2327-4581.1000101",),
    )

    # The CMORizer also writes a time-independent land-sea mask (``sftlf``,
    # MIP ``fx``), which is not a ClimateEval variable; only pick the monthly
    # temperature fields.
    _file_prefix_pattern = "*_Amon_*"
    _supported_var_names = ("tas", "tasa")
