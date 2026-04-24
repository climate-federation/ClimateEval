"""Provide diagnostics."""

from __future__ import annotations

from benchmarking.diags._base import Diagnostic, DiagnosticOutput
from benchmarking.diags._utils import DataTypeValue

__all__ = [
    "DataTypeValue",
    "Diagnostic",
    "DiagnosticOutput",
]
