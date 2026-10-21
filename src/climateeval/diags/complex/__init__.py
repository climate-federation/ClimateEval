"""Provide complex diagnostics.

Complex diagnostics require a predefined set of variables and/or experiments
defined for each diagnostic separately.

"""

from __future__ import annotations

from climateeval.diags.complex._base import ComplexDiagnostic
from climateeval.diags.complex._consistency import (
    BudgetClosure,
    ClausiusClapeyronScaling,
    ClearSkyLongwaveFeedback,
    EnergyBalance,
    PrecipitationBuoyancy,
)
from climateeval.diags.complex._ecs import ECS
from climateeval.diags.complex._forced_response import (
    ArcticAmplification,
    LandOceanWarmingRatio,
)
from climateeval.diags.complex._tem import TEMDiagnostic
from climateeval.diags.complex._transport import MeridionalHeatTransport

__all__ = [
    "ECS",
    "ArcticAmplification",
    "BudgetClosure",
    "ClausiusClapeyronScaling",
    "ClearSkyLongwaveFeedback",
    "ComplexDiagnostic",
    "EnergyBalance",
    "LandOceanWarmingRatio",
    "MeridionalHeatTransport",
    "PrecipitationBuoyancy",
    "TEMDiagnostic",
]
