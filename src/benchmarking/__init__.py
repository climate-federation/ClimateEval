"""Benchmark Earth system models."""

from __future__ import annotations

import iris

from benchmarking._config import Config
from benchmarking._variable import COORDINATES, VARIABLES, Coordinate, Variable

__all__ = [
    "COORDINATES",
    "VARIABLES",
    "Config",
    "Coordinate",
    "Variable",
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
