"""Create a TEM diagnostic database from legoESM AMIP output."""

from __future__ import annotations

import sys
from pathlib import Path

import iris
from iris.cube import CubeList
from loguru import logger

from climateeval._utils import load_cubes
from climateeval.data import DataSourceInformation
from climateeval.diags.complex import TEMDiagnostic
from climateeval.diags._base import DiagnosticOutput
from climateeval.report import serve

import ibis
from dataclasses import fields

logger.remove()
logger.add(sys.stdout, level="INFO",
           format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")

CMOR_DIR = Path("/scratch/b/b309178/amip_1979_270d_25124195/cmor/Amon")
DB_PATH = Path("tem_diagnostic.ddb")


def main() -> None:
    # --- Load model data --------------------------------------------------
    logger.info("Loading model CubeList from {}", CMOR_DIR)
    cubes = CubeList([])
    for var in ("va", "ta"):
        nc = CMOR_DIR / f"{var}_Amon_legoESM-1-0_amip_r1i1p1f1_gn.nc"
        if nc.is_file():
            loaded = iris.load(str(nc))
            cubes.extend(loaded)
            logger.info("  Loaded {} cubes from {}", len(loaded), nc.name)
        else:
            logger.warning("  File not found: {}", nc)

    # wap is optional (w_star and w_dia will be NaN without it)
    wap_nc = CMOR_DIR / "wap_Amon_legoESM-1-0_amip_r1i1p1f1_gn.nc"
    if wap_nc.is_file():
        loaded = iris.load(str(wap_nc))
        cubes.extend(loaded)
        logger.info("  Loaded wap ({} cubes)", len(loaded))
    else:
        logger.info("  wap not found – w_star and w_dia will be NaN")

    data_info = DataSourceInformation(
        name="legoESM-1-0",xit
        
        category="legoESM",
        institute="CU",
        exp="amip",
        variant="r1i1p1f1",
        references=("https://github.com/google-deepmind/neural_lam",),
    )

    # --- Run TEM diagnostic ------------------------------------------------
    logger.info("Running TEMDiagnostic")
    diag = TEMDiagnostic("TEM")
    output = diag.get_output(data={"data": cubes}, data_information=data_info)

    # --- Write to DuckDB ---------------------------------------------------
    if DB_PATH.is_file():
        DB_PATH.unlink()

    conn = ibis.connect(f"duckdb://{DB_PATH}")
    conn.create_database("TEM")
    for output_field in fields(output):
        table_name = output_field.name
        table = getattr(output, table_name)
        if table is None:
            logger.info("  Table '{}' is None, skipping", table_name)
            continue
        conn.create_table(table_name, table, database="TEM")
        logger.info("  Written table '{}' ({} rows)", table_name,
                    table.count().execute() if hasattr(table, "count") else "?")

    conn.disconnect()
    logger.info("Database written to {}", DB_PATH)

    # --- Launch HTML report ------------------------------------------------
    logger.info("Launching report at http://127.0.0.1:8765 (Ctrl-C to stop)")
    serve(
        db_paths=[DB_PATH],
        model_name="legoESM-1-0",
        host="127.0.0.1",
        port=8765,
        open_browser=False,
    )


if __name__ == "__main__":
    main()
