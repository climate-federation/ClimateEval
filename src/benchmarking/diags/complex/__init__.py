"""Provide complex diagnostics.

Complex diagnostics require a predefined set of variables and/or experiments
defined for each diagnostic separately.

"""

from __future__ import annotations

from benchmarking.diags.complex._ecs import ECS

__all__ = [
    "ECS",
]
