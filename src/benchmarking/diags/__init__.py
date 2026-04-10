"""Provide diagnostics."""

from __future__ import annotations

from benchmarking.diags._base import Diagnostic
from benchmarking.diags._map import Map
from benchmarking.diags._timeseries import TimeSeries

__all__ = [
    "Diagnostic",
    "Map",
    "TimeSeries",
]
