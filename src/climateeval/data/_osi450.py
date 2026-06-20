"""Provide OSI SAF OSI-450 sea-ice concentration data.

OSI-450 is the EUMETSAT OSI SAF global sea-ice concentration climate data
record, provided separately for the Northern (``OSI-450-nh``) and Southern
(``OSI-450-sh``) hemispheres. Both are CMORized via ESMValTool.
"""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class OSI450NH(ESMValToolCMORizerDataSource):
    """Manage OSI-450 Northern Hemisphere sea-ice concentration data."""

    _information = DataSourceInformation(
        name="OSI-450-nh",
        category="reanalysis",
        institute="EUMETSAT OSI SAF",
        references=("http://osisaf.met.no/p/ice/",),
    )

    _supported_var_names = ("siconc",)


class OSI450SH(ESMValToolCMORizerDataSource):
    """Manage OSI-450 Southern Hemisphere sea-ice concentration data."""

    _information = DataSourceInformation(
        name="OSI-450-sh",
        category="reanalysis",
        institute="EUMETSAT OSI SAF",
        references=("http://osisaf.met.no/p/ice/",),
    )

    _supported_var_names = ("siconc",)
