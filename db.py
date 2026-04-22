"""Small script to test everything manually."""

from __future__ import annotations

from pathlib import Path

import ibis

root = Path.home() / "benchmarking"
db_path = root / "Tier2_atmosphere.ddb"
# db_path = root / "Tier2_atmosphere.ddb.BAK"
con = ibis.duckdb.connect(db_path)

for catalog_name in con.list_catalogs():
    for database_name in con.list_databases(catalog=catalog_name):
        for table_name in con.list_tables(database=f"{catalog_name}.{database_name}"):
            full_name = f"{catalog_name}.{database_name}.{table_name}"
            table = con.table(table_name, database=database_name)
            if table_name != "raw_output":
                continue
            # print(full_name)
            # print(table)
            # print(table.schema())
            # print((table.count().to_pandas(), len(table.schema())))
            print(table.to_pandas())
            # html_path = root / f"{full_name}.html"
            # print(table.to_pandas().to_html(html_path))
            # print(f"Wrote {html_path}")
            print()
