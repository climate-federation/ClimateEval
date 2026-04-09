"""Benchmark Earth system models."""

from __future__ import annotations

import logging

import iris

from benchmarking._config import NO_LEVEL, VerticalLevel, reset_esmvaltool_config
from benchmarking._variable import COORDINATES, VARIABLES, Coordinate, Variable

__all__ = [
    "COORDINATES",
    "NO_LEVEL",
    "VARIABLES",
    "Coordinate",
    "Variable",
    "VerticalLevel",
]

# Make Iris as flexible as possible when combining cubes (e.g., during loading)
iris.COMBINE_POLICY.equalise_cubes_kwargs = {
    "equalise_attributes": True,
    "unify_time_units": True,
}
iris.COMBINE_POLICY.merge_concat_sequence = "mc"
iris.COMBINE_POLICY.repeat_until_unchanged = True

# Use new behavior of iris.pandas.as_data_frame
iris.FUTURE.pandas_ndim = True

# Configure ESMValTool
reset_esmvaltool_config()
logging.getLogger("esmvalcore").setLevel(logging.ERROR)
logging.getLogger("esmvaltool").setLevel(logging.ERROR)
