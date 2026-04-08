"""Benchmarking Diagnostics."""

from __future__ import annotations

from benchmarking.diags._base import Diagnostic
from benchmarking.diags._timeseries import TimeSeries

__all__ = [
    "Diagnostic",
    "TimeSeries",
]
