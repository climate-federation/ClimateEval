"""Manage datasets."""

from __future__ import annotations

from benchmarking.data._base import DataSource
from benchmarking.data._gpcp_sg import GPCPSG
from benchmarking.data._hadcrut5 import HadCRUT5

__all__ = [
    "GPCPSG",
    "DataSource",
    "HadCRUT5",
]
