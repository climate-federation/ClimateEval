"""Benchmark legoESM (amip_1979_270d_25124195) using Tier2_atmosphere_monthly suite.

Produces one .ddb per suite (per-tier convention) and exports a portable HTML report.
CMIP6 historical comparison models are loaded automatically from the suite definition
via climateeval.data.CMIP6HistoricalR1I1P1F1.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import iris
from loguru import logger

from climateeval.data import DataSourceInformation
from climateeval.report import serve
from climateeval.suites import Suite

logger = logger.opt(colors=True)

LEGOESM_CMOR_DIR = Path("/scratch/b/b309178/amip_1979_270d_25124195/cmor/Amon")
LEGOESM_TIMERANGE = "19790101/19790930"

SUITES = [
    "Tier2_atmosphere_monthly",
]

DIAGNOSTIC_KWARGS = {
    "data_root_dir": Path("/work/bd1179/b309141/climateeval_input"),
    "fail_on_missing_data": False,
    "download_missing_data": False,
}


def run_suite(suite_name: str, cubes: object, data_info: DataSourceInformation) -> Path:
    db_path = Path(f"{suite_name}.ddb")
    if db_path.is_file():
        db_path.unlink()
    suite = Suite(
        suite_name,
        diagnostic_kwargs=DIAGNOSTIC_KWARGS,
        variable_kwargs={"timerange": LEGOESM_TIMERANGE},
    )
    suite.get_database(cubes, data_info, database_resource=f"duckdb://{db_path}")
    logger.info(f"Database written to {db_path}")
    return db_path


def main() -> None:
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | <level>{message}</level>"
    )
    logger.add(sys.stdout, level="DEBUG", format=fmt, colorize=True)

    legoesm_info = DataSourceInformation(
        name="legoESM-1-0",
        category="legoESM",
        institute="MPI-M",
        exp="amip",
        variant="r1i1p1f1",
    )

    legoesm_paths = sorted(LEGOESM_CMOR_DIR.glob("*.nc"))
    logger.info(f"Loading {len(legoesm_paths)} legoESM files from {LEGOESM_CMOR_DIR}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legoesm_cubes = iris.load([str(p) for p in legoesm_paths])
    logger.info(f"Loaded {len(legoesm_cubes)} cubes")

    db_paths = []
    for suite_name in SUITES:
        db_paths.append(run_suite(suite_name, legoesm_cubes, legoesm_info))

    html_path = Path("legoesm_amip_1979_25124195_report.html")
    logger.info(f"Generating HTML report → {html_path}")
    serve(
        db_paths,
        model_name="legoESM-1-0",
        save_html=html_path,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
