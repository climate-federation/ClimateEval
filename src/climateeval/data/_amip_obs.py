"""Provide the locally staged observational products for AMIP-style comparisons.

Each product is prepared into CMOR-like NetCDF files by a prep script of the
observation bundle that stages it and read from
``<data_root>/<source id>/<frequency>/<var_name>/*.nc`` (see
:mod:`climateeval.data._local_obs`). Only :class:`GPCC` is different: ESMValTool
has a CMORizer for it, which ClimateEval can run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._local_obs import LocalObservationDataSource
from climateeval.data._utils import DataSourceInformation

if TYPE_CHECKING:
    from typing import Self


class GPCPDaily(LocalObservationDataSource):
    """GPCP 1DD (v1.3 CDR) daily precipitation, 1 degree, global."""

    _information = DataSourceInformation(
        name="GPCP-1DD",
        category="observation",
        institute="NASA-GSFC",
        references=(
            "https://doi.org/10.1175/1525-7541(2001)002<0036:GPAODD>2.0.CO;2",
            "https://www.ncei.noaa.gov/products/climate-data-records/precipitation-gpcp-daily",
        ),
    )
    _supported_var_names = ("pr",)
    _supported_frequencies = ("day",)
    _prep_script = "obs/prep/gpcp_daily.py"


class CMORPH(LocalObservationDataSource):
    """CMORPH-CDR daily precipitation (59.5S-59.5N)."""

    _information = DataSourceInformation(
        name="CMORPH-CDR",
        category="observation",
        institute="NOAA-NCEI",
        references=(
            "https://doi.org/10.1175/JHM-D-16-0168.1",
            "https://www.ncei.noaa.gov/products/climate-data-records/precipitation-cmorph",
        ),
    )
    _supported_var_names = ("pr",)
    _supported_frequencies = ("day",)
    _prep_script = "obs/prep/cmorph.py"


class BerkeleyEarthDaily(LocalObservationDataSource):
    """Berkeley Earth daily land temperature (``tas``, ``tasmax``, ``tasmin``)."""

    _information = DataSourceInformation(
        name="BerkeleyEarth-daily",
        category="observation",
        institute="Berkeley Earth",
        references=(
            "https://doi.org/10.5194/essd-12-3469-2020",
            "https://berkeleyearth.org/data/",
        ),
    )
    _supported_var_names = ("tas", "tasmax", "tasmin")
    _supported_frequencies = ("day",)
    _prep_script = "obs/prep/berkeley_earth_daily.py"


class HadGHCND(LocalObservationDataSource):
    """HadGHCND daily land temperature extremes (``tasmax``, ``tasmin``)."""

    _information = DataSourceInformation(
        name="HadGHCND",
        category="observation",
        institute="UK Met Office Hadley Centre",
        references=(
            "https://doi.org/10.1029/2005JD006280",
            "https://www.metoffice.gov.uk/hadobs/hadghcnd/",
        ),
    )
    _supported_var_names = ("tasmax", "tasmin")
    _supported_frequencies = ("day",)
    _prep_script = "obs/prep/hadghcnd.py"


class RSSPrw(LocalObservationDataSource):
    """RSS merged monthly column water vapour (``prw``), ocean only."""

    _information = DataSourceInformation(
        name="RSS-TPW",
        category="observation",
        institute="Remote Sensing Systems",
        references=("https://www.remss.com/measurements/atmospheric-water-vapor/",),
    )
    _supported_var_names = ("prw",)
    _supported_frequencies = ("mon",)
    _prep_script = "obs/prep/rss_prw.py"


class RSSMSU(LocalObservationDataSource):
    """RSS MSU/AMSU lower- and mid-tropospheric temperature (``tlt``, ``tmt``)."""

    _information = DataSourceInformation(
        name="RSS-MSU",
        category="observation",
        institute="Remote Sensing Systems",
        references=("https://www.remss.com/measurements/upper-air-temperature/",),
    )
    _supported_var_names = ("tlt", "tmt")
    _supported_frequencies = ("mon",)
    _prep_script = "obs/prep/msu_rss.py"


class UAHMSU(LocalObservationDataSource):
    """UAH MSU/AMSU lower- and mid-tropospheric temperature (``tlt``, ``tmt``)."""

    _information = DataSourceInformation(
        name="UAH-MSU",
        category="observation",
        institute="University of Alabama in Huntsville",
        references=(
            "https://doi.org/10.1007/s13143-017-0010-y",
            "https://www.nsstc.uah.edu/data/msu/",
        ),
    )
    _supported_var_names = ("tlt", "tmt")
    _supported_frequencies = ("mon",)
    _prep_script = "obs/prep/msu_uah.py"


class RICH(LocalObservationDataSource):
    """RICH homogenised radiosonde temperature on pressure levels (``ta``)."""

    _information = DataSourceInformation(
        name="RICH",
        category="observation",
        institute="University of Vienna",
        references=(
            "https://doi.org/10.1175/JCLI-D-11-00668.1",
            "https://www.univie.ac.at/theoret-met/research/raobcore/",
        ),
    )
    _supported_var_names = ("ta",)
    _supported_frequencies = ("mon",)


class RAOBCORE(LocalObservationDataSource):
    """RAOBCORE homogenised radiosonde temperature on pressure levels (``ta``)."""

    _information = DataSourceInformation(
        name="RAOBCORE",
        category="observation",
        institute="University of Vienna",
        references=(
            "https://doi.org/10.1175/JCLI-D-11-00668.1",
            "https://www.univie.ac.at/theoret-met/research/raobcore/",
        ),
    )
    _supported_var_names = ("ta",)
    _supported_frequencies = ("mon",)


class GPCC(ESMValToolCMORizerDataSource):
    """GPCC Full Data Monthly v2018 precipitation (1 degree, land only).

    Unlike the other products of this module, GPCC has an ESMValTool CMORizer,
    so it is served like :class:`~climateeval.data.CRU`: formatted files are
    looked up in ``<data_root>/observation_GPCC/mon/pr/*.nc`` first and only
    otherwise downloaded and CMORized. The CMORizer writes the four versions
    (0.25, 0.5, 1.0 and 2.5 degrees, 1891-2016) with and without the mask of
    cells without gauges; this class only uses the unmasked 1.0 degree
    version (``v2018-10``). On DKRZ Levante the file is available as
    ``/work/bd0854/DATA/ESMValTool2/OBS/Tier2/GPCC/OBS_GPCC_reanaly_v2018-10_Amon_pr_189101-201612.nc``
    and can be symlinked into the data root.
    """

    _information = DataSourceInformation(
        name="GPCC",
        category="observation",
        institute="DWD",
        references=("https://doi.org/10.5676/DWD_GPCC/FD_M_V2018_100",),
    )
    _file_prefix_pattern = "*_v2018-10_Amon_*"
    _supported_var_names = ("pr",)
    _supported_frequencies = ("mon",)


class AMIPBoundaryConditions(LocalObservationDataSource):
    """Prescribed AMIP sea surface temperature and sea-ice concentration.

    The dataset behind the boundary conditions is not fixed (the choice is
    open), so the name is configurable: use :meth:`for_dataset` to get a class
    with the name (and references) of the dataset that is actually staged. The
    default name is ``AMIPBoundaryConditions``. Serves ``tos`` and ``siconc``
    (monthly).
    """

    _information = DataSourceInformation(
        name="AMIPBoundaryConditions",
        category="observation",
    )
    _supported_var_names = ("tos", "siconc")
    _supported_frequencies = ("mon",)

    @classmethod
    def for_dataset(
        cls,
        name: str,
        *,
        institute: str = "",
        references: tuple[str, ...] = (),
    ) -> type[Self]:
        """Create a class for a specific boundary-condition dataset.

        The data source ID is ``observation_<name>``, so the files must be
        staged under that directory name.
        """
        information = DataSourceInformation(
            name=name,
            category="observation",
            institute=institute,
            references=references,
        )
        class_name = "".join(c if c.isalnum() else "_" for c in f"AMIP_{name}")
        return type(
            class_name,
            (cls,),
            {
                "__doc__": f"AMIP boundary conditions of {information.id}.",
                "_information": information,
            },
        )
