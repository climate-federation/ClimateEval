"""Provide GISTEMP data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class GISTEMP(ESMValToolCMORizerDataSource):
    """Manage GISTEMP data.

    The NASA GISS Surface Temperature Analysis (GISTEMP v4) provides gridded
    near-surface temperature *anomalies* (``tasa``) only; there is no absolute
    temperature field.

    """

    _information = DataSourceInformation(
        name="GISTEMP",
        category="observation",
        institute="NASA-GISS",
        references=("https://doi.org/10.1029/2018JD029522",),
    )

    _supported_var_names = ("tasa",)
