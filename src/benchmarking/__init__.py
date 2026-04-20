"""Benchmark Earth system models."""

from __future__ import annotations

from benchmarking._config import (
    esmvaltool_only_log_errors,
    setup_esmvaltool_config_and_logging,
    setup_iris_combine_policy,
    use_default_esmvaltool_config,
)
from benchmarking._variable import (
    COORDINATES,
    NO_LEVEL,
    VARIABLES,
    Coordinate,
    Variable,
    VerticalLevel,
)

__all__ = [
    "COORDINATES",
    "NO_LEVEL",
    "VARIABLES",
    "Coordinate",
    "Variable",
    "VerticalLevel",
    "esmvaltool_only_log_errors",
    "setup_esmvaltool_config_and_logging",
    "setup_iris_combine_policy",
    "use_default_esmvaltool_config",
]
