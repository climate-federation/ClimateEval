"""Provide data source."""

from __future__ import annotations

from benchmarking.data._base import ComplexDataSource, DataSource, DataSourceGenerator
from benchmarking.data._ceres_ebaf import CERESEBAF
from benchmarking.data._cmip6_historical import CMIP6HistoricalR1I1P1F1
from benchmarking.data._era5 import ERA5
from benchmarking.data._esacci_cloud import ESACCICloud
from benchmarking.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from benchmarking.data._esmvaltool_intake import (
    ESMValToolIntakeDataSource,
    ESMValToolIntakeDataSourceGenerator,
)
from benchmarking.data._gpcp import GPCP
from benchmarking.data._hadcrut5 import HadCRUT5
from benchmarking.data._merra2 import MERRA2
from benchmarking.data._utils import DataSourceInformation

__all__ = [
    "CERESEBAF",
    "ERA5",
    "GPCP",
    "MERRA2",
    "CMIP6HistoricalR1I1P1F1",
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
