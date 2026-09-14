"""Provide data source."""

from __future__ import annotations

from climateeval.data._base import ComplexDataSource, DataSource, DataSourceGenerator
from climateeval.data._berkeleyearth import BerkeleyEarth
from climateeval.data._ceres_ebaf import CERESEBAF
from climateeval.data._cmip6_generators import (
    CMIP6Abrupt4xCO2R1I1P1F1,
    CMIP6DataSourceGenerator,
    CMIP6HistoricalR1I1P1F1,
    CMIP6PiControlR1I1P1F1,
)
from climateeval.data._cru import CRU
from climateeval.data._en4 import EN4
from climateeval.data._era5 import ERA5Hourly, ERA5Monthly
from climateeval.data._esacci_cloud import ESACCICloud
from climateeval.data._esacci_oc import ESACCIOC
from climateeval.data._esacci_sst import ESACCISST
from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._esmvaltool_intake import (
    ESMValToolIntakeDataSource,
    ESMValToolIntakeDataSourceGenerator,
)
from climateeval.data._gistemp import GISTEMP
from climateeval.data._gpcp import GPCP
from climateeval.data._hadcrut5 import HadCRUT5
from climateeval.data._hadisst import HadISST
from climateeval.data._iap import IAP
from climateeval.data._merra2 import MERRA2
from climateeval.data._noaa_ersstv5 import NOAAERSSTv5
from climateeval.data._noaaglobaltemp import NOAAGlobalTemp
from climateeval.data._nsidc_g02202_sh import NSIDCG02202SH
from climateeval.data._oras5 import ORAS5
from climateeval.data._osi450 import OSI450NH, OSI450SH
from climateeval.data._rapid import RAPID
from climateeval.data._utils import DataSourceInformation
from climateeval.data._woa import WOA

__all__ = [
    "CERESEBAF",
    "CRU",
    "EN4",
    "ESACCIOC",
    "ESACCISST",
    "GISTEMP",
    "GPCP",
    "IAP",
    "MERRA2",
    "NSIDCG02202SH",
    "ORAS5",
    "OSI450NH",
    "OSI450SH",
    "RAPID",
    "WOA",
    "BerkeleyEarth",
    "CMIP6Abrupt4xCO2R1I1P1F1",
    "CMIP6DataSourceGenerator",
    "CMIP6HistoricalR1I1P1F1",
    "CMIP6PiControlR1I1P1F1",
    "ComplexDataSource",
    "DataSource",
    "DataSourceGenerator",
    "DataSourceInformation",
    "ERA5Hourly",
    "ERA5Monthly",
    "ESACCICloud",
    "ESMValToolCMORizerDataSource",
    "ESMValToolIntakeDataSource",
    "ESMValToolIntakeDataSourceGenerator",
    "HadCRUT5",
    "HadISST",
    "NOAAERSSTv5",
    "NOAAGlobalTemp",
]
