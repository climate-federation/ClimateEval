"""Provide complex diagnostics.

Complex diagnostics require a predefined set of variables and/or experiments
defined for each diagnostic separately.

"""

from __future__ import annotations

from climateeval.diags.complex._base import ComplexDiagnostic
from climateeval.diags.complex._ecs import ECS
from climateeval.diags.complex._tem import TEMDiagnostic

__all__ = [
    "ECS",
    "ComplexDiagnostic",
    "TEMDiagnostic",
]
