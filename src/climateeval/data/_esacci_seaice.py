"""Provide ESACCI-SEAICE sea-ice concentration data.

ESACCI-SEAICE is the ESA Climate Change Initiative sea-ice concentration
climate data record (L4, SSM/I and SSMIS, 12.5 km EASE2 grid, v3.0). The
Northern (``NH``) and Southern (``SH``) hemispheres are stored on separate polar
grids and are exposed here as two data sources, giving both hemispheres a
satellite counterpart to the reanalysis-based ``NSIDCG02202SH``.

Both hemispheres are produced by the same ESMValTool CMORizer data set
(``ESACCI-SEAICE``), so the two classes share their ``name`` and are
distinguished by their ``variant`` (and hence by their ``id`` and their data
directory).
"""

from __future__ import annotations

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation


class _ESACCISeaIce(ESMValToolCMORizerDataSource):
    """Manage ESACCI-SEAICE sea-ice concentration data (one hemisphere).

    Notes
    -----
    For each hemisphere the CMORizer writes daily (MIP ``SIday``) and monthly
    (MIP ``SImon``) ``siconc`` plus a time-independent grid-cell area
    (``areacello``, MIP ``Ofx``, not a ClimateEval variable). Formatted files are
    sorted into directories by variable name and by the frequency that was
    requested, so a mixture of these products would end up in the same
    directory. The hemisphere is part of the ESMValTool ``version`` facet of the
    file name (``..._L4-SICONC-RE-SSMI-12.5kmEASE2-fv3.0-NH_SImon_siconc_...``),
    so one hemisphere at monthly frequency is selected with the file prefix
    pattern of the respective subclass.

    Downloading is therefore also shared: fetching either hemisphere runs the
    ESMValTool downloader and CMORizer for *both* of them (daily source files
    for 1991-2020), and everything that does not match the subclass' prefix
    pattern is discarded.

    """

    _supported_frequencies = ("mon",)
    _supported_var_names = ("siconc",)


class ESACCISeaIceNH(_ESACCISeaIce):
    """Manage ESACCI-SEAICE Northern Hemisphere sea-ice concentration data."""

    _information = DataSourceInformation(
        name="ESACCI-SEAICE",
        category="observation",
        institute="ESA",
        variant="nh",
        references=("https://doi.org/10.5285/eade27004395466aaa006135e1b2ad1a",),
    )

    _file_prefix_pattern = "*-NH_SImon_*"


class ESACCISeaIceSH(_ESACCISeaIce):
    """Manage ESACCI-SEAICE Southern Hemisphere sea-ice concentration data."""

    _information = DataSourceInformation(
        name="ESACCI-SEAICE",
        category="observation",
        institute="ESA",
        variant="sh",
        references=("https://doi.org/10.5285/eade27004395466aaa006135e1b2ad1a",),
    )

    _file_prefix_pattern = "*-SH_SImon_*"
