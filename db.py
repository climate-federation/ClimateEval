"""Small script to test everything manually."""

from __future__ import annotations

from pathlib import Path

import ibis

db_path = Path.home() / "benchmarking" / "benchmarking.ddb"
con = ibis.duckdb.connect(db_path)

for catalog_name in con.list_catalogs():
    # print(f"===== {catalog_name} =====")
    for database_name in con.list_databases(catalog=catalog_name):
        # print(f"  --- {database_name} ---")
        for table_name in con.list_tables(database=f"{catalog_name}.{database_name}"):
            full_name = f"{catalog_name}.{database_name}.{table_name}"
            print(f"{full_name}")
            table = con.table(table_name, database=database_name)
            print(table)
            print()
