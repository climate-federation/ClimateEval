"""Benchmark legoESM amip run and MPI-ESM1-2-LR CMIP6 for comparison."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import iris
from loguru import logger

from climateeval.data import DataSourceInformation
from climateeval.suites import Suite

logger = logger.opt(colors=True)

LEGOESM_CMOR_DIR = Path("/scratch/b/b309178/amip_1979_270d_25073918/cmor/Amon")

CMIP6_BASE = Path(
    "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR"
    "/historical/r1i1p1f1/Amon"
)
CMIP6_VARIABLES = (
    "clivi", "clt", "clwvi",
    "hfls", "hfss", "hus",
    "pr", "prw",
    "rlut", "rlutcs", "rsdt", "rsut", "rsutcs",
    "ta", "tas", "ua",
)
CMIP6_VERSION = "v20190710"
CMIP6_FILE_SUFFIX = "MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc"

DIAGNOSTIC_KWARGS = {
    "data_root_dir": Path("/work/bd1179/b309141/climateeval_input"),
    "fail_on_missing_data": False,
    "download_missing_data": False,
}


def run_suite(
    suite_name: str,
    cubes: object,
    data_info: DataSourceInformation,
    variable_kwargs: dict,
    db_path: Path,
) -> None:
    if db_path.is_file():
        db_path.unlink()
    suite = Suite(
        suite_name,
        diagnostic_kwargs=DIAGNOSTIC_KWARGS,
        variable_kwargs=variable_kwargs,
    )
    suite.get_database(
        cubes,
        data_info,
        database_resource=f"duckdb://{db_path}",
    )
    logger.info(f"Database written to {db_path}")


def main() -> None:
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | <level>{message}</level>"
    )
    logger.add(sys.stdout, level="DEBUG", format=fmt, colorize=True)

    # --- legoESM (amip, Jan–Sep 1979) ---
    legoesm_info = DataSourceInformation(
        name="legoESM-1-0",
        category="legoESM",
        institute="MPI-M",
        exp="amip",
        variant="r1i1p1f1",
    )
    legoesm_paths = sorted(LEGOESM_CMOR_DIR.glob("*.nc"))
    logger.info(
        f"Loading {len(legoesm_paths)} legoESM files from {LEGOESM_CMOR_DIR}"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legoesm_cubes = iris.load([str(p) for p in legoesm_paths])
    logger.info(f"Loaded {len(legoesm_cubes)} cubes")

    run_suite(
        "Tier2_atmosphere_monthly",
        legoesm_cubes,
        legoesm_info,
        {"timerange": "19790101/19790930"},
        Path("legoesm_amip_1979.ddb"),
    )

    # --- MPI-ESM1-2-LR CMIP6 historical ---
    cmip6_info = DataSourceInformation(
        name="MPI-ESM1-2-LR",
        category="CMIP6",
        institute="MPI-M",
        exp="historical",
        variant="r1i1p1f1",
        references=(
            "https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.MPI-M.MPI-ESM1-2-LR",
        ),
    )
    cmip6_paths = []
    for var in CMIP6_VARIABLES:
        p = CMIP6_BASE / var / "gn" / CMIP6_VERSION / f"{var}_Amon_{CMIP6_FILE_SUFFIX}"
        if p.is_file():
            cmip6_paths.append(str(p))
        else:
            logger.warning(f"CMIP6 file not found, skipping: {p}")
    logger.info(f"Loading {len(cmip6_paths)} CMIP6 files")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmip6_cubes = iris.load(cmip6_paths)
    logger.info(f"Loaded {len(cmip6_cubes)} cubes")

    run_suite(
        "Tier2_atmosphere_monthly",
        cmip6_cubes,
        cmip6_info,
        {"timerange": "20000101/20011231"},
        Path("mpi_esm_historical.ddb"),
    )


if __name__ == "__main__":
    main()
