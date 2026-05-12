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
from climateeval.data._era5 import ERA5
from climateeval.data._esacci_chl import ESACCIOC
from climateeval.data._esacci_cloud import ESACCICloud
from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._esmvaltool_intake import (
    ESMValToolIntakeDataSource,
    ESMValToolIntakeDataSourceGenerator,
)
from climateeval.data._gpcp import GPCP
from climateeval.data._hadcrut5 import HadCRUT5
from climateeval.data._merra2 import MERRA2
from climateeval.data._utils import DataSourceInformation

__all__ = [
    "CERESEBAF",
    "ERA5",
    "ESACCIOC",
    "GPCP",
    "MERRA2",
    "CMIP6Abrupt4xCO2R1I1P1F1",
    "CMIP6DataSourceGenerator",
    "CMIP6HistoricalR1I1P1F1",
    "CMIP6PiControlR1I1P1F1",
    "ComplexDataSource",
    "DataSource",
    "DataSourceGenerator",
    "DataSourceInformation",
    "ESACCICloud",
    "ESMValToolCMORizerDataSource",
    "ESMValToolIntakeDataSource",
    "ESMValToolIntakeDataSourceGenerator",
    "HadCRUT5",
]
