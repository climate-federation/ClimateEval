"""Provide diagnostic utilities."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import ibis
import iris
import pandas as pd
from iris.pandas import as_data_frame
from loguru import logger

from benchmarking import NO_LEVEL

if TYPE_CHECKING:
    from collections.abc import Iterable

    from iris.cube import Cube

    from benchmarking import Variable
    from benchmarking.data import DataSourceInformation

logger = logger.opt(colors=True)

# Public constants
DATA_ID_COLUMN: str = "data_id"
"""Name of table column containing data ID."""

REFERENCE_DATA_ID_COLUMN: str = "reference_data_id"
"""Name of table column containing reference data ID."""

DATA_TYPE_COLUMN: str = "data_type"
"""Name of table column containing data type."""

DTYPE: ibis.DataType = "float32"
"""Dtype of numerical values."""

VARIABLE_ID_COLUMN: str = "var_id"
"""Name of table column containing variable ID."""

DataTypeValue = Literal["to_benchmark", "reference", "other"]
"""Allowed values for data type."""


# Private constants
DEFAULT_GRID: str = "2x2"
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


@dataclass(frozen=True, kw_only=True)
class DiagnosticOutput:
    """Manage diagnostic output."""

    raw_output: ibis.Table | None
    metrics: ibis.Table | None
    variables: ibis.Table
    data_sources: ibis.Table


def get_data_root_dir(data_root_dir: Path | None) -> Path:
    """Get data root directory."""
    if data_root_dir is None:
        data_root_dir = Path.home() / ".benchmarking" / "data"
    data_root_dir.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Using data root directory {data_root_dir}")
    return data_root_dir


def get_data_sources_table(
    data_source_information: Iterable[DataSourceInformation],
) -> ibis.Table:
    """Get data sources table."""
    df = pd.DataFrame(sorted(set(data_source_information)))
    df["references"] = df["references"].map(" ".join)
    return ibis.memtable(df)


def get_raw_output_table(
    cube: Cube,
    *,
    variable: Variable,
    data_id: str,
    data_type: Literal["to_benchmark", "reference", "other"],
) -> ibis.Table:
    """Get data table of a single cube."""
    with iris.FUTURE.context(pandas_ndim=True):
        df = as_data_frame(cube).reset_index()
    if "time" in df.columns:
        df["time"] = df["time"].astype(str)
    raw_output_table = ibis.memtable(df)
    cast_operations = {
        c: {"time": "timestamp"}.get(c, DTYPE) for c in raw_output_table.columns
    }
    raw_output_table = raw_output_table.cast(**cast_operations)
    raw_output_table = raw_output_table.rename(**{variable.id: cube.name()})
    raw_output_table = raw_output_table.mutate(
        **{DATA_ID_COLUMN: ibis.literal(data_id)},
    )
    raw_output_table = raw_output_table.mutate(
        **{DATA_TYPE_COLUMN: ibis.literal(data_type)},
    )
    return raw_output_table  # noqa: RET504


def get_variables_table(variables: Iterable[Variable]) -> ibis.Table:
    """Get variables table."""
    df = pd.DataFrame(sorted(set(variables)))
    df = df.drop("settings", axis=1)
    df["coordinates"] = df["coordinates"].map(
        lambda r: " ".join(c["var_name"] for c in r),
    )
    df["ar6_regions_mask"] = df["ar6_regions_mask"].map(" ".join)

    def vertical_level_dict_to_str(
        vertical_level_dict: dict[str, str | float],
    ) -> str:
        """Convert vertical level dictionary to string."""
        if vertical_level_dict == asdict(NO_LEVEL):
            return ""
        return (
            f"{vertical_level_dict['coordinate_name']} = "
            f"{vertical_level_dict['value']} {vertical_level_dict['units']}"
        )

    df["vertical_level"] = df["vertical_level"].map(vertical_level_dict_to_str)
    return ibis.memtable(df)


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
