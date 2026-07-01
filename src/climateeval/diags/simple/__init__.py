"""Provide simple diagnostics.

Simple diagnostics are defined to work with any variable. They required exactly
one reference data source per variable. Each variable of each input data source
is processed separately and compared with the corresponding reference to
calculate a set of metrics.

"""

from __future__ import annotations

from climateeval.diags.simple._amoc import AMOCTimeSeries
from climateeval.diags.simple._annual_cycle import AnnualCycle
from climateeval.diags.simple._annual_cycle_spatial_distribution import (
    AnnualCycleSpatialDistribution,
)
from climateeval.diags.simple._base import SimpleDiagnostic
from climateeval.diags.simple._diurnal_cycle import DiurnalCycle
from climateeval.diags.simple._histogram import Histogram
from climateeval.diags.simple._hovmoller import Hovmoller
from climateeval.diags.simple._map import Map
from climateeval.diags.simple._nino34 import Nino34
from climateeval.diags.simple._ohc import OceanHeatContentTimeSeries
from climateeval.diags.simple._qbo import QBO
from climateeval.diags.simple._sea_ice import (
    SeaIceAreaAnnualCycle,
    SeaIceAreaTimeSeries,
)
from climateeval.diags.simple._timeseries import (
    AnnualMeanTimeSeries,
    MeanTimeSeries,
    TimeSeriesNoPreprocessing,
)
from climateeval.diags.simple._utils import SimpleDiagnosticInputData
from climateeval.diags.simple._zonal_line import ZonalLine
from climateeval.diags.simple._zonal_profile import ZonalProfileAirPressure

__all__ = [
    "AMOCTimeSeries",
    "AnnualCycle",
    "AnnualCycleSpatialDistribution",
    "AnnualMeanTimeSeries",
    "DiurnalCycle",
    "Histogram",
    "Hovmoller",
    "Map",
    "QBO",
    "MeanTimeSeries",
    "Nino34",
    "OceanHeatContentTimeSeries",
    "SeaIceAreaAnnualCycle",
    "SeaIceAreaTimeSeries",
    "SimpleDiagnostic",
    "SimpleDiagnosticInputData",
    "TimeSeriesNoPreprocessing",
    "ZonalLine",
    "ZonalProfileAirPressure",
]
