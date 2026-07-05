"""Merge per-model ClimateEval tier databases into one combined database.

Each source .ddb holds one schema per diagnostic with up to four tables
(raw_output, metrics, variables, data_sources). Rows shared between runs
(references like ERA5, the CMIP6 comparison ensemble) appear in every
per-model file; the merge inserts each dataset's rows only once, so the
result matches what a sequential ``Suite.get_database(..., append=True)``
loop produces — minus the duplicated reference rows.

Source order defines precedence: pass the benchmarked model's db first.

Usage
-----
  python merge_tier_dbs.py OUT.ddb SRC1.ddb SRC2.ddb [SRC3.ddb ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
from loguru import logger

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
)

_SYSTEM_SCHEMAS = {"main", "information_schema", "pg_catalog"}

# Dedup key per table: a dataset's rows arrive complete with its own run,
# so presence of the key means the rows are already there.
_DEDUP_KEYS = {
    "data_sources": "id",
    "variables": "id",
    "raw_output": "data_id",
    "metrics": "data_id",
}


def _q(identifier: str) -> str:
    """Quote a SQL identifier."""
    return '"' + identifier.replace('"', '""') + '"'


def _schemas(con: duckdb.DuckDBPyConnection, catalog: str) -> list[str]:
    rows = con.execute(
        "SELECT schema_name FROM information_schema.schemata "
        "WHERE catalog_name = ? ORDER BY schema_name",
        [catalog],
    ).fetchall()
    return [r[0] for r in rows if r[0] not in _SYSTEM_SCHEMAS]


def _tables(con: duckdb.DuckDBPyConnection, catalog: str, schema: str) -> list[str]:
    rows = con.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_catalog = ? AND table_schema = ? ORDER BY table_name",
        [catalog, schema],
    ).fetchall()
    return [r[0] for r in rows]


def _columns(
    con: duckdb.DuckDBPyConnection, catalog: str, schema: str, table: str
) -> dict[str, str]:
    rows = con.execute(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_catalog = ? AND table_schema = ? AND table_name = ?",
        [catalog, schema, table],
    ).fetchall()
    return dict(rows)


def _merge_table(
    con: duckdb.DuckDBPyConnection, out_catalog: str, schema: str, table: str
) -> tuple[int, int]:
    """Merge src.schema.table into out.schema.table; return (inserted, skipped)."""
    src = f"src.{_q(schema)}.{_q(table)}"
    tgt = f"{_q(out_catalog)}.{_q(schema)}.{_q(table)}"
    n_src = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]

    if table not in _tables(con, out_catalog, schema):
        con.execute(f"CREATE TABLE {tgt} AS SELECT * FROM {src}")
        return n_src, 0

    # Tolerate column drift (e.g. a model missing a variable column):
    # add source-only columns to the target; INSERT BY NAME NULL-fills the rest.
    tgt_cols = _columns(con, out_catalog, schema, table)
    for col, dtype in _columns(con, "src", schema, table).items():
        if col not in tgt_cols:
            logger.warning("Adding missing column {}.{}.{} ({})", schema, table, col, dtype)
            con.execute(f"ALTER TABLE {tgt} ADD COLUMN {_q(col)} {dtype}")

    key = _DEDUP_KEYS.get(table)
    if key is None:
        logger.warning("No dedup key for table '{}' — inserting all rows", table)
        con.execute(f"INSERT INTO {tgt} BY NAME SELECT * FROM {src}")
        return n_src, 0

    new_rows = f"SELECT * FROM {src} WHERE {_q(key)} NOT IN (SELECT DISTINCT {_q(key)} FROM {tgt})"
    n_new = con.execute(f"SELECT count(*) FROM ({new_rows})").fetchone()[0]
    con.execute(f"INSERT INTO {tgt} BY NAME {new_rows}")

    # A model benchmarked by this source may already sit in the target as an
    # 'other' (comparison-ensemble) dataset with identical values — its rows
    # were skipped above, so promote the label instead.
    if "data_type" in _columns(con, "src", schema, table):
        con.execute(
            f"UPDATE {tgt} SET data_type = 'to_benchmark' "
            f"WHERE data_type <> 'to_benchmark' AND data_id IN "
            f"(SELECT DISTINCT data_id FROM {src} WHERE data_type = 'to_benchmark')"
        )
    return n_new, n_src - n_new


def merge(out_path: Path, sources: list[Path]) -> None:
    con = duckdb.connect(str(out_path))
    out_catalog = out_path.stem
    for src_path in sources:
        con.execute(f"ATTACH '{src_path}' AS src (READ_ONLY)")
        n_inserted = n_skipped = 0
        for schema in _schemas(con, "src"):
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {_q(out_catalog)}.{_q(schema)}")
            for table in _tables(con, "src", schema):
                ins, skip = _merge_table(con, out_catalog, schema, table)
                n_inserted += ins
                n_skipped += skip
        con.execute("DETACH src")
        logger.info(
            "Merged {} — {} rows inserted, {} duplicate rows skipped",
            src_path,
            n_inserted,
            n_skipped,
        )
    con.close()
    logger.info("Wrote {}", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Combined output .ddb (created)")
    parser.add_argument(
        "sources",
        type=Path,
        nargs="+",
        help="Per-model source .ddb files (benchmarked model first)",
    )
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    for src in args.sources:
        if not src.is_file():
            parser.error(f"Source not found: {src}")

    merge(args.output, args.sources)


if __name__ == "__main__":
    main()
