"""Sea-ice monthly diagnostic suite for NH and SH siconc.

Runs the Tier2_sea_ice_monthly suite on MPI-ESM1-2-LR historical SImon data
and serves the result at http://127.0.0.1:8769.

Diagnostics run:
  - annual_cycle          (annual cycle of total hemispheric sea-ice area,
                           ref: OSI-450, other: HadISST/NSIDC-G02202)
  - annual_mean_timeseries (annual time series of the seasonal sea-ice
                           minimum: NH September, SH February)
  - map                   (climatological maps of siconc: full, Feb, Sep)

References (OSI-450, HadISST, NSIDC-G02202) must be staged locally; if
missing, the corresponding metrics are skipped (fail_on_missing_data=False).

Port: 8769
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
_V = "v20190710"
_BASE = (
    "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR"
    "/historical/r1i1p1f1/SImon/siconc/gn"
)

_NC_FILES = [
    f"{_BASE}/{_V}/siconc_SImon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc",
]

DB_PATH = Path("Tier2_sea_ice_monthly.ddb")
PORT = 8769
# ---------------------------------------------------------------------------


def main() -> None:
    cubes = CubeList([])
    for nc_path in _NC_FILES:
        nc = Path(nc_path)
        if nc.is_file():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loaded = iris.load(str(nc))
            cubes.extend(loaded)
            logger.info("Loaded {} cube(s) from {}", len(loaded), nc.name)
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

    if DB_PATH.is_file():
        DB_PATH.unlink()
    Path(str(DB_PATH) + ".wal").unlink(missing_ok=True)

    suite = Suite(
        "Tier2_sea_ice_monthly",
        diagnostic_kwargs={
            "data_root_dir": Path("/work/bd1179/b309141/climateeval_input"),
            "fail_on_missing_data": False,
            "fail_on_metric_error": False,
            "download_missing_data": False,
        },
    )

    logger.info("Running Tier2_sea_ice_monthly suite → {}", DB_PATH)
    suite.get_database(
        cubes,
        data_info,
        database_resource=f"duckdb://{DB_PATH}",
    )
    logger.info("Database written to {}", DB_PATH)

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
