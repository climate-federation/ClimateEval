"""Provide complex diagnostics.

Complex diagnostics require a predefined set of variables and/or experiments
defined for each diagnostic separately.

"""

from __future__ import annotations

from climateeval.diags.complex._base import ComplexDiagnostic
from climateeval.diags.complex._dynamics import (
    GeostrophicBalance,
    MJOEastWestPowerRatio,
)
from climateeval.diags.complex._ecs import ECS
from climateeval.diags.complex._forced_response import (
    ArcticAmplification,
    LandOceanWarmingRatio,
)
from climateeval.diags.complex._tem import TEMDiagnostic
from climateeval.diags.complex._transport import MeridionalHeatTransport
from climateeval.diags.complex._variability import (
    BjerknesCompensation,
    ENSOTeleconnectionPatterns,
    ITCZEnergyFluxEquator,
)

__all__ = [
    "ECS",
    "ArcticAmplification",
    "BjerknesCompensation",
    "ComplexDiagnostic",
    "ENSOTeleconnectionPatterns",
    "GeostrophicBalance",
    "ITCZEnergyFluxEquator",
    "LandOceanWarmingRatio",
    "MJOEastWestPowerRatio",
    "MeridionalHeatTransport",
    "TEMDiagnostic",
]
