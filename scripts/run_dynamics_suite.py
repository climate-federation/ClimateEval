"""Showcase suite: tropical convection (Hovmöller) + stratospheric QBO.

Runs the Tier3_dynamics suite on MPI-ESM1-2-LR historical data and serves
the result at http://127.0.0.1:8768.

Data loaded:
  - Daily atmosphere (pr, rlut)  — 1990-2009, only 2000-2002 used by Hovmöller
  - Daily ocean     (tos)        — 1990-2009, only 2000-2002 used by Hovmöller
  - Monthly atmosphere (ua)      — 1990-2009, full period used by QBO

Port: 8768 (Hovmöller alone: 8766, QBO alone: 8767)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import iris
from iris.cube import CubeList
from loguru import logger

from climateeval.data import DataSourceInformation
from climateeval.suites import Suite
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

_V = "v20190710"
_ATMOS = (
    "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR"
    "/historical/r1i1p1f1"
)

# Files to load — all variables needed by Hovmöller and QBO diagnostics
_NC_FILES = {
    # Hovmöller inputs (daily)
    "pr_day":   f"{_ATMOS}/day/pr/gn/{_V}/pr_day_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_19900101-20091231.nc",
    "rlut_day": f"{_ATMOS}/day/rlut/gn/{_V}/rlut_day_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_19900101-20091231.nc",
    "tos_day":  f"{_ATMOS}/Oday/tos/gn/{_V}/tos_Oday_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_19900101-20091231.nc",
    # QBO input (monthly)
    "ua_mon":   f"{_ATMOS}/Amon/ua/gn/{_V}/ua_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc",
}

DB_PATH = Path("Tier3_dynamics.ddb")
PORT = 8768

# ---------------------------------------------------------------------------


def main() -> None:
    # ---- Load all available data ------------------------------------------
    cubes = CubeList([])
    for label, nc_path in _NC_FILES.items():
        nc = Path(nc_path)
        if nc.is_file():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loaded = iris.load(str(nc))
            cubes.extend(loaded)
            logger.info("Loaded {}: {} cube(s)", label, len(loaded))
        else:
            logger.warning("File not found — skipping: {}", nc_path)

    if not cubes:
        logger.error("No cubes loaded — aborting")
        sys.exit(1)

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

    # ---- Run suite --------------------------------------------------------
    if DB_PATH.is_file():
        DB_PATH.unlink()
    Path(str(DB_PATH) + ".wal").unlink(missing_ok=True)

    suite = Suite(
        "Tier3_dynamics",
        diagnostic_kwargs={
            "data_root_dir": Path("/work/bd1179/b309141/climateeval_input"),
            "fail_on_missing_data": False,
            "fail_on_metric_error": False,
            "download_missing_data": False,
        },
    )

    logger.info("Running Tier3_dynamics suite → {}", DB_PATH)
    suite.get_database(
        cubes,
        data_info,
        database_resource=f"duckdb://{DB_PATH}",
    )
    logger.info("Database written to {}", DB_PATH)

    # ---- Serve ------------------------------------------------------------
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
