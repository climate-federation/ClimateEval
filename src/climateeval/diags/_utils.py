"""Provide diagnostic utilities."""

from __future__ import annotations

from typing import Literal

import ibis
from loguru import logger

logger = logger.opt(colors=True)

DataTypeValue = Literal["to_benchmark", "reference", "other"]
"""Allowed values for data type."""

DEFAULT_GRID: str = "2x2"
SEA_ICE_GRID: str = "1x1"  # finer grid for sea ice, matching ICONEval recipes
PLEV19_LEVELS: list[float] = [
    100000.0,
    92500.0,
    85000.0,
    70000.0,
    60000.0,
    50000.0,
    40000.0,
    30000.0,
    25000.0,
    20000.0,
    15000.0,
    10000.0,
    7000.0,
    5000.0,
    3000.0,
    2000.0,
    1000.0,
    500.0,
    100.0,
]


def union(*tables: ibis.Table) -> ibis.Table:
    """Create union of tables (accounts for different schemas)."""
    columns_and_dtypes: dict[str, ibis.DataType] = {}
    for table in tables:
        for column, dtype in table.schema().items():
            if column not in columns_and_dtypes:
                columns_and_dtypes[column] = dtype

    # Add missing columns (filled with NaNs), cast to common dtype and unify
    # column order
    unified_tables: list[ibis.Table] = []
    for table in tables:
        missing_columns = set(columns_and_dtypes) - set(table.columns)
        if missing_columns:
            unified_table = table.mutate(
                **{col: ibis.literal(None) for col in missing_columns},
            )
        else:
            unified_table = table
        unified_table = unified_table.cast(columns_and_dtypes)
        unified_table = unified_table.select(tuple(columns_and_dtypes))
        unified_tables.append(unified_table)

    return ibis.union(*unified_tables)
