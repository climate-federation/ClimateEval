"""Create a Hovmöller diagnostic database from MPI-ESM1-2-LR CMIP6 data.

Configurable options (edit below):
  TIMERANGE           : period to plot (e.g. 3 years → readable diagram)
  LAT_MIN / LAT_MAX   : latitude band for the meridional average
  ROLLING_WINDOW_DAYS : 0=none, 3=Kelvin waves, 5=MJO, 30=ENSO
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import fields
from pathlib import Path

import iris
from iris.cube import CubeList
from loguru import logger

import ibis

from climateeval import Variable
from climateeval.data import DataSourceInformation
from climateeval.diags.simple import Hovmoller
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
TIMERANGE = "20000101/20021231"   # 3 years
LAT_MIN = -15.0                   # degrees N
LAT_MAX = 15.0
ROLLING_WINDOW_DAYS = 5           # 0=none, 3=Kelvin, 5=MJO, 30=ENSO

_V = "v20190710"
_ATMOS = (
    "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR"
    "/historical/r1i1p1f1/day"
)
_OCEAN = (
    "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR"
    "/historical/r1i1p1f1/Oday"
)

_NC = {
    "pr":   f"{_ATMOS}/pr/gn/{_V}/pr_day_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_19900101-20091231.nc",
    "rlut": f"{_ATMOS}/rlut/gn/{_V}/rlut_day_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_19900101-20091231.nc",
    "tos":  f"{_OCEAN}/tos/gn/{_V}/tos_Oday_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_19900101-20091231.nc",
}

DB_PATH = Path("hovmoller_diagnostic.ddb")
# ---------------------------------------------------------------------------


def main() -> None:
    # --- Load whatever files exist ----------------------------------------
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
        Variable(v, v, "day", timerange=TIMERANGE): SimpleDiagnosticInputData(
            reference=None, other=()
        )
        for v in available_vars
    }

    logger.info(
        "Hovmöller: variables={}, lat=[{}°, {}°], rolling={}d, period={}",
        available_vars, LAT_MIN, LAT_MAX, ROLLING_WINDOW_DAYS, TIMERANGE,
    )
    diag = Hovmoller(
        "Hovmoller",
        variables_and_data,
        lat_min=LAT_MIN,
        lat_max=LAT_MAX,
        rolling_window_days=ROLLING_WINDOW_DAYS,
        fail_on_missing_data=False,   # skip variables that fail preprocessing
        calculate_metrics=False,
    )
    output = diag.get_output(data=cubes, data_information=data_info)

    # --- Write to DuckDB --------------------------------------------------
    if DB_PATH.is_file():
        DB_PATH.unlink()

    conn = ibis.connect(f"duckdb://{DB_PATH}")
    conn.create_database("Hovmoller")
    for output_field in fields(output):
        table_name = output_field.name
        table = getattr(output, table_name)
        if table is None:
            logger.info("  Skipping table '{}' (None)", table_name)
            continue
        conn.create_table(table_name, table, database="Hovmoller")
        n = table.count().execute() if hasattr(table, "count") else "?"
        logger.info("  Written '{}': {} rows", table_name, n)

    conn.disconnect()
    logger.info("Database written to {}", DB_PATH)

    # --- Serve HTML report ------------------------------------------------
    logger.info("Serving at http://127.0.0.1:8766 — Ctrl-C to stop")
    serve(
        db_paths=[DB_PATH],
        model_name="MPI-ESM1-2-LR",
        host="127.0.0.1",
        port=8766,
        open_browser=False,
    )


if __name__ == "__main__":
    main()
