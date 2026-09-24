"""Provide CRU TS data."""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class CRU(ESMValToolCMORizerDataSource):
    """Manage CRU TS data.

    CRU TS (v4.07) is a station-interpolated dataset of monthly land-surface
    climate; all fields are masked over the ocean.

    """

    _information = DataSourceInformation(
        name="CRU",
        category="observation",
        institute="University of East Anglia CRU",
        references=("https://doi.org/10.1038/s41597-020-0453-3",),
    )

    # The CMORizer also writes the potential evapotranspiration ``evspsblpot``
    # (MIP ``Emon``), which is not a ClimateEval variable; only pick the
    # ``Amon`` fields.
    _file_prefix_pattern = "*_Amon_*"
    _supported_var_names = ("clt", "pr", "tas", "tasmax", "tasmin")
