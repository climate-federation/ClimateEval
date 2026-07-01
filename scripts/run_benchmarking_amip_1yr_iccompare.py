"""Benchmark the two 2026-06-30 1-year lat-lon (res24/L20) AMIP runs against ERA5.

Compares legoESM against observational reference data only (ERA5, CERES-EBAF, etc.).
CMIP6 other_data entries are stripped so no additional models are loaded.

Two runs, differing ONLY in initial condition (PR #688 lat-lon ERA5-IC phis-smoothing
fix validation, full 365-day length; convective_cloud=false in both):
  - std : amip_test_latlon_1yr_std  (job 25964732, ic=standard)
  - era5: amip_test_latlon_1yr_era5 (job 25964733, ic=era5)
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

RUNS = {
    "std": Path("/scratch/b/b309178/amip_test_latlon_1yr_std/cmor/Amon"),
    "era5": Path("/scratch/b/b309178/amip_test_latlon_1yr_era5/cmor/Amon"),
}
LEGOESM_TIMERANGE = "19790101/19791231"

SUITES = [
    "Tier2_atmosphere_monthly",
]

DIAGNOSTIC_KWARGS = {
    "data_root_dir": Path("/scratch/b/b309178/climateeval_data"),
    "fail_on_missing_data": False,
    "download_missing_data": False,
}


def _suite_yml_era5_only(suite_name: str) -> Path:
    """Return path to a temporary suite YAML using ERA5Monthly as reference for all
    variables and with other_data removed."""
    suite_dir = Path(str(resources.files("climateeval.suites")))
    with (suite_dir / f"{suite_name}.yml").open() as f:
        suite_def = yaml.safe_load(f)

    for diag_settings in suite_def:
        if not isinstance(diag_settings, dict):
            continue
        for var_settings in diag_settings.values():
            if isinstance(var_settings, dict):
                var_settings.pop("other_data", None)
                if "reference_data" in var_settings:
                    var_settings["reference_data"] = "climateeval.data.ERA5Monthly"

    tmp = Path(tempfile.mktemp(suffix=".yml"))
    with tmp.open("w") as f:
        yaml.dump(suite_def, f)
    return tmp


def run_suite(tag: str, suite_name: str, cubes: object, data_info: DataSourceInformation) -> Path:
    db_path = Path(f"amip_1yr_{tag}_{suite_name}.ddb")
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

    db_paths = []
    for tag, cmor_dir in RUNS.items():
        info = DataSourceInformation(
            name=f"legoESM-1-0-{tag}",
            category="legoESM",
            institute="MPI-M",
            exp="amip",
            variant="r1i1p1f1",
        )
        paths = sorted(cmor_dir.glob("*.nc"))
        logger.info(f"[{tag}] Loading {len(paths)} legoESM files from {cmor_dir}")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cubes = iris.load([str(p) for p in paths])
        logger.info(f"[{tag}] Loaded {len(cubes)} cubes")

        for suite_name in SUITES:
            db_paths.append(run_suite(tag, suite_name, cubes, info))

    html_path = Path("amip_1yr_iccompare_report.html")
    logger.info(f"Generating HTML report -> {html_path}")
    serve(
        db_paths,
        model_name="legoESM-1-0",
        save_html=html_path,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
