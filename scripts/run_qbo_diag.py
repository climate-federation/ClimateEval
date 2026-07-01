"""Run the QBO diagnostic on MPI-ESM1-2-LR historical ua (Amon, 1990-2009).

Writes qbo_diagnostic.ddb and serves it at http://127.0.0.1:8767.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import fields
from pathlib import Path

import ibis
import iris
from iris.cube import CubeList
from loguru import logger

from climateeval import Variable
from climateeval.data import DataSourceInformation
from climateeval.diags.simple import QBO
from climateeval.diags.simple._utils import SimpleDiagnosticInputData
from climateeval.report import serve

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TIMERANGE = "19900101/20091231"  # 20 years — well above the 5-year QBO minimum

_V = "v20190710"
_BASE = (
    "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR"
    "/historical/r1i1p1f1/Amon"
)

_NC = {
    "ua": (
        f"{_BASE}/ua/gn/{_V}/"
        "ua_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc"
    ),
}

DB_PATH = Path("qbo_diagnostic.ddb")
PORT = 8767

# ---------------------------------------------------------------------------


def main() -> None:
    # ---- Load data --------------------------------------------------------
    cubes = CubeList([])
    available_vars: list[str] = []
    for var_id, nc_path in _NC.items():
        nc = Path(nc_path)
        if nc.is_file():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loaded = iris.load(str(nc))
            cubes.extend(loaded)
            available_vars.append(var_id)
            logger.info("Loaded {}: {} cube(s)", var_id, len(loaded))
        else:
            logger.info("Skipping {} (file not found)", var_id)

    if not cubes:
        logger.error("No data loaded — aborting")
        return

    data_info = DataSourceInformation(
        name="MPI-ESM1-2-LR",
        category="CMIP6",
        institute="MPI-M",
        exp="historical",
        variant="r1i1p1f1",
        references=(
            "https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.MPI-M.MPI-ESM1-2-LR",
        ),
    )

    variables_and_data = {
        Variable(v, v, "mon", timerange=TIMERANGE): SimpleDiagnosticInputData(
            reference=None, other=()
        )
        for v in available_vars
    }

    logger.info("QBO: variables={}, period={}", available_vars, TIMERANGE)
    diag = QBO(
        "QBO",
        variables_and_data,
        fail_on_missing_data=False,
        calculate_metrics=False,
    )
    output = diag.get_output(data=cubes, data_information=data_info)

    # ---- Write to DuckDB --------------------------------------------------
    if DB_PATH.is_file():
        DB_PATH.unlink()

    conn = ibis.connect(f"duckdb://{DB_PATH}")
    conn.create_database("QBO")
    for output_field in fields(output):
        table_name = output_field.name
        table = getattr(output, table_name)
        if table is None:
            logger.info("  Skipping table '{}' (None)", table_name)
            continue
        conn.create_table(table_name, table, database="QBO")
        n = table.count().execute() if hasattr(table, "count") else "?"
        logger.info("  Written '{}': {} rows", table_name, n)

    conn.disconnect()
    logger.info("Database written to {}", DB_PATH)

    # ---- Serve HTML report ------------------------------------------------
    logger.info("Serving at http://127.0.0.1:{} — Ctrl-C to stop", PORT)
    serve(
        db_paths=[DB_PATH],
        model_name="MPI-ESM1-2-LR",
        host="127.0.0.1",
        port=PORT,
        open_browser=False,
    )


if __name__ == "__main__":
    main()
