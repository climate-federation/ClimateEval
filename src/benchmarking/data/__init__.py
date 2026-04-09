"""Manage datasets."""

from __future__ import annotations

from benchmarking.data._base import DataSource, DiagnosticData
from benchmarking.data._ceres_ebaf import CERESEBAF
from benchmarking.data._era5 import ERA5
from benchmarking.data._gpcp import GPCP
from benchmarking.data._hadcrut5 import HadCRUT5

__all__ = [
    "CERESEBAF",
    "ERA5",
    "GPCP",
    "DataSource",
    "DiagnosticData",
    "HadCRUT5",
]
