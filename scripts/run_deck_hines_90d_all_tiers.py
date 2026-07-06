"""Benchmark amip_deck_era5_hines_90d across all feasible ClimateEval tier suites.

Dataset: /scratch/b/b309178/amip_deck_era5_hines_90d/cmor/Amon  (job 25492319)
         C48/L40, 90 days from 1979-01-01, CMIP6-deck protocol, Hines GWD, ERA5 IC.

Feasible suites (monthly Amon output available):
  - Tier1_sanity_checks       : reasonable-range checks, no reference comparison
  - Tier1_consistency_checks  : air-mass and water-vapour anomalies vs ERA5
  - Tier2_atmosphere_monthly  : full set of 2-D/3-D atmosphere diagnostics vs ERA5

Skipped suites:
  - Tier3_ecs                : needs abrupt-4xCO2 experiment
  - Tier3_atmosphere_subdaily: needs 1-hourly data (only daily available)
  - Tier2_ocean_monthly      : needs ocean variables (atm-only run)

Each suite writes its own .ddb file; a combined portable HTML is written at the end.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from importlib import resources
from pathlib import Path

import iris
import yaml
from loguru import logger

from climateeval.data import DataSourceInformation
from climateeval.report import serve
from climateeval.suites import Suite

logger = logger.opt(colors=True)

LEGOESM_CMOR_DIR = Path("/scratch/b/b309178/amip_deck_era5_hines_90d/cmor/Amon")
LEGOESM_TIMERANGE = "19790101/19790331"  # 90-day run → Jan–Mar 1979 only
MODEL_NAME = "legoESM-1-0 (deck-ERA5-Hines-90d)"

FEASIBLE_SUITES = [
    "Tier1_sanity_checks",
    "Tier1_consistency_checks",
    "Tier2_atmosphere_monthly",
]

DIAGNOSTIC_KWARGS = {
    "data_root_dir": Path("/work/bd1179/b309141/climateeval_input"),
    "fail_on_missing_data": False,
    "download_missing_data": False,
}


def _suite_yml_era5_only(suite_name: str) -> Path:
    """Return a temp suite YAML with CMIP6 other_data stripped and all
    variable-level reference_data replaced by ERA5Monthly.

    ERA5 is the only reference dataset covering Jan 1979; MERRA2 starts 1980,
    GPCP starts 1983, CERES-EBAF starts 2000.
    """
    suite_pkg = resources.files("climateeval.suites")
    with (Path(str(suite_pkg)) / f"{suite_name}.yml").open() as f:
        suite_def = yaml.safe_load(f)

    for diag_entry in suite_def:
        if not isinstance(diag_entry, dict):
            continue
        variables = diag_entry.get("variables")
        if not isinstance(variables, list):
            continue
        for var_cfg in variables:
            if not isinstance(var_cfg, dict):
                continue
            var_cfg.pop("other_data", None)
            if "reference_data" in var_cfg:
                var_cfg["reference_data"] = "climateeval.data.ERA5Monthly"

    tmp = Path(tempfile.mktemp(suffix=".yml"))
    with tmp.open("w") as f:
        yaml.dump(suite_def, f)
    return tmp


def run_suite(
    suite_name: str,
    cubes: object,
    data_info: DataSourceInformation,
) -> Path:
    db_path = Path(f"deck_hines_90d_{suite_name}.ddb")
    if db_path.is_file():
        db_path.unlink()
    yml_path = _suite_yml_era5_only(suite_name)
    try:
        suite = Suite(
            yml_path,
            diagnostic_kwargs=DIAGNOSTIC_KWARGS,
            variable_kwargs={"timerange": LEGOESM_TIMERANGE},
        )
        suite.get_database(cubes, data_info, database_resource=f"duckdb://{db_path}")
    finally:
        yml_path.unlink(missing_ok=True)
    logger.info(f"Database written to {db_path}")
    return db_path


def main() -> None:
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | <level>{message}</level>"
    )
    logger.add(sys.stdout, level="DEBUG", format=fmt, colorize=True)

    if not LEGOESM_CMOR_DIR.is_dir():
        logger.error(f"CMOR directory not found: {LEGOESM_CMOR_DIR}")
        sys.exit(1)

    legoesm_info = DataSourceInformation(
        name="legoESM-1-0",
        category="legoESM",
        institute="MPI-M",
        exp="amip",
        variant="r1i1p1f1",
    )

    nc_paths = sorted(LEGOESM_CMOR_DIR.glob("*.nc"))
    logger.info(f"Loading {len(nc_paths)} legoESM files from {LEGOESM_CMOR_DIR}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cubes = iris.load([str(p) for p in nc_paths])
    logger.info(f"Loaded {len(cubes)} cubes")

    db_paths: list[Path] = []
    for suite_name in FEASIBLE_SUITES:
        logger.info(f"Running suite: {suite_name}")
        db_paths.append(run_suite(suite_name, cubes, legoesm_info))

    html_path = Path("deck_hines_90d_report.html")
    logger.info(f"Generating portable HTML report → {html_path}")
    serve(
        db_paths,
        model_name=MODEL_NAME,
        save_html=html_path,
        open_browser=False,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
