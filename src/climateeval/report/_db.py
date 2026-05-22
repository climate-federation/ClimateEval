"""Database reading and diagnostic-type detection utilities."""

from __future__ import annotations

from pathlib import Path

import ibis
import pandas as pd

# ---------------------------------------------------------------------------
# Database reading
# ---------------------------------------------------------------------------

_SKIP_DBS = {"main", "memory", "information_schema", "temp", "system"}


def read_database(db_path: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Return {diagnostic_name: {table_name: DataFrame}} from a DuckDB file."""
    conn = ibis.duckdb.connect(db_path, read_only=True)
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for db_name in conn.list_databases():
        if db_name.lower() in _SKIP_DBS:
            continue
        tables: dict[str, pd.DataFrame] = {}
        try:
            for table_name in conn.list_tables(database=db_name):
                tables[table_name] = conn.table(
                    table_name,
                    database=db_name,
                ).to_pandas()
        except Exception:
            continue
        if tables:
            result[db_name] = tables
    return result


def get_var_cols(data_df: pd.DataFrame) -> list[str]:
    """Return variable (non-coordinate, non-metadata) column names."""
    skip = {
        "data_id",
        "data_type",
        "latitude",
        "longitude",
        "time",
        "air_pressure",
        "month_number",
        "hour",
    }
    return [c for c in data_df.columns if c not in skip]


def get_density_cols(data_df: pd.DataFrame) -> list[str]:
    """Return density column names."""
    return [c for c in data_df.columns if "density_" in c]


def detect_diag_type(columns: list[str]) -> str:
    cols = set(columns)
    if "latitude" in cols and "longitude" in cols:
        return "Map"
    if "air_pressure" in cols and "latitude" in cols:
        return "ZonalProfile"
    if "latitude" in cols:
        return "ZonalLine"
    if "time" in cols:
        return "AnnualMeanTimeSeries"
    if "month_number" in cols:
        return "AnnualCycle"
    if "hour" in cols:
        return "DiurnalCycle"
    for col in cols:
        if "density_" in col:
            return "Histogram"
    coord_columns = {
        "latitude",
        "longitude",
        "time",
        "air_pressure",
        "month_number",
        "hour",
    }
    if not (cols & coord_columns):
        return "Scalar"
    return "Unknown"
