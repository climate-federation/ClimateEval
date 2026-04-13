"""Provide diagnostics."""

from __future__ import annotations

from benchmarking.diags._annual_cycle import AnnualCycle
from benchmarking.diags._base import Diagnostic
from benchmarking.diags._diurnal_cycle import DiurnalCycle
from benchmarking.diags._map import Map
from benchmarking.diags._timeseries import TimeSeries
from benchmarking.diags._zonal_line import ZonalLine
from benchmarking.diags._zonal_profile import ZonalProfileAirPressure

__all__ = [
    "AnnualCycle",
    "Diagnostic",
    "DiurnalCycle",
    "Map",
    "TimeSeries",
    "ZonalLine",
    "ZonalProfileAirPressure",
]
