"""Provide NOAAGlobalTemp data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class NOAAGlobalTemp(ESMValToolCMORizerDataSource):
    """Manage NOAAGlobalTemp data.

    The NOAA Global Surface Temperature dataset (v5) provides gridded
    near-surface temperature *anomalies* (``tasa``) only; there is no absolute
    temperature field.

    """

    _information = DataSourceInformation(
        name="NOAAGlobalTemp",
        category="observation",
        institute="NOAA-NCEI",
        references=("https://doi.org/10.25921/9qth-2p70",),
    )

    _supported_var_names = ("tasa",)
