"""Provide data source."""

from __future__ import annotations

from climateeval.data._base import ComplexDataSource, DataSource, DataSourceGenerator
from climateeval.data._ceres_ebaf import CERESEBAF
from climateeval.data._cmip6_generators import (
    CMIP6Abrupt4xCO2R1I1P1F1,
    CMIP6DataSourceGenerator,
    CMIP6HistoricalR1I1P1F1,
    CMIP6PiControlR1I1P1F1,
)
from climateeval.data._era5 import ERA5Hourly, ERA5Monthly
from climateeval.data._esacci_cloud import ESACCICloud
from climateeval.data._esacci_oc import ESACCIOC
from climateeval.data._esacci_sst import ESACCISST
from climateeval.data._esmvalcore_native6 import ESMValCoreNative6DataSource
from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._esmvaltool_intake import (
    ESMValToolIntakeDataSource,
    ESMValToolIntakeDataSourceGenerator,
)
from climateeval.data._gpcp import GPCP
from climateeval.data._hadcrut5 import HadCRUT5
from climateeval.data._hadisst import HadISST
from climateeval.data._merra2 import MERRA2
from climateeval.data._noaa_ersstv5 import NOAAERSSTv5
from climateeval.data._oras5 import ORAS5
from climateeval.data._utils import DataSourceInformation
from climateeval.data._woa import WOA

__all__ = [
    "CERESEBAF",
    "ESACCIOC",
    "ESACCISST",
    "GPCP",
    "MERRA2",
    "ORAS5",
    "WOA",
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
    "ESMValCoreNative6DataSource",
    "ESMValToolCMORizerDataSource",
    "ESMValToolIntakeDataSource",
    "ESMValToolIntakeDataSourceGenerator",
    "HadCRUT5",
    "HadISST",
    "NOAAERSSTv5",
]
