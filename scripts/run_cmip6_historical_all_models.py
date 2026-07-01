"""Benchmark all CMIP6 historical r1i1p1f1 models from Manuel's input data.

Each model is processed through the Tier2_atmosphere_monthly suite.
Results are stored as one DuckDB file per model in OUTPUT_DIR.

Usage:
    # Run a single model by index (for SLURM array jobs, 0-based)
    python run_cmip6_historical_all_models.py --model-index 0

    # Run a single model by name
    python run_cmip6_historical_all_models.py --model MPI-ESM1-2-LR

    # Run all models sequentially (no argument)
    python run_cmip6_historical_all_models.py
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import iris
from loguru import logger

from climateeval.data import DataSourceInformation
from climateeval.suites import Suite

logger = logger.opt(colors=True)

DATA_ROOT_DIR = Path("/work/bd1179/b309141/climateeval_input")
OUTPUT_DIR = Path("/work/bd1083/b309178/diffESM/ClimateEval/cmip6_historical_dbs")
TIMERANGE = "19790101/20141231"
SUITE_NAME = "Tier2_atmosphere_monthly"

DIAGNOSTIC_KWARGS = {
    "data_root_dir": DATA_ROOT_DIR,
    "fail_on_missing_data": False,
    "download_missing_data": False,
}

# All CMIP6 historical r1i1p1f1 models available in Manuel's input data
MODELS: list[tuple[str, str]] = [
    ("ACCESS-CM2", "CSIRO-ARCCSS"),
    ("ACCESS-ESM1-5", "CSIRO"),
    ("AWI-CM-1-1-MR", "AWI"),
    ("AWI-ESM-1-1-LR", "AWI"),
    ("AWI-ESM-1-REcoM", "AWI"),
    ("BCC-CSM2-MR", "BCC"),
    ("BCC-ESM1", "BCC"),
    ("CAMS-CSM1-0", "CAMS"),
    ("CanESM5", "CCCma"),
    ("CAS-ESM2-0", "CAS"),
    ("CESM2-FV2", "NCAR"),
    ("CESM2", "NCAR"),
    ("CESM2-WACCM-FV2", "NCAR"),
    ("CESM2-WACCM", "NCAR"),
    ("CMCC-CM2-HR4", "CMCC"),
    ("CMCC-CM2-SR5", "CMCC"),
    ("CMCC-ESM2", "CMCC"),
    ("E3SM-1-0", "E3SM-Project"),
    ("E3SM-1-1-ECA", "E3SM-Project"),
    ("E3SM-1-1", "E3SM-Project"),
    ("EC-Earth3-AerChem", "EC-Earth-Consortium"),
    ("EC-Earth3-CC", "EC-Earth-Consortium"),
    ("EC-Earth3", "EC-Earth-Consortium"),
    ("EC-Earth3-Veg", "EC-Earth-Consortium"),
    ("EC-Earth3-Veg-LR", "EC-Earth-Consortium"),
    ("FGOALS-f3-L", "CAS"),
    ("FGOALS-g3", "CAS"),
    ("FIO-ESM-2-0", "FIO-QLNM"),
    ("GFDL-CM4", "NOAA-GFDL"),
    ("GFDL-ESM4", "NOAA-GFDL"),
    ("GISS-E2-1-G-CC", "NASA-GISS"),
    ("GISS-E2-1-G", "NASA-GISS"),
    ("GISS-E2-1-H", "NASA-GISS"),
    ("ICON-ESM-LR", "MPI-M"),
    ("IITM-ESM", "CCCR-IITM"),
    ("INM-CM4-8", "INM"),
    ("INM-CM5-0", "INM"),
    ("IPSL-CM5A2-INCA", "IPSL"),
    ("IPSL-CM6A-LR", "IPSL"),
    ("KACE-1-0-G", "NIMS-KMA"),
    ("KIOST-ESM", "KIOST"),
    ("MCM-UA-1-0", "UA"),
    ("MIROC6", "MIROC"),
    ("MPI-ESM-1-2-HAM", "HAMMOZ-Consortium"),
    ("MPI-ESM1-2-HR", "MPI-M"),
    ("MPI-ESM1-2-LR", "MPI-M"),
    ("MRI-ESM2-0", "MRI"),
    ("NESM3", "NUIST"),
    ("NorESM2-LM", "NCC"),
    ("NorESM2-MM", "NCC"),
    ("SAM0-UNICON", "SNU"),
    ("TaiESM1", "AS-RCEC"),
]


def load_model_cubes(model: str) -> iris.cube.CubeList:
    """Load all monthly NetCDF files for a model from Manuel's input data."""
    model_dir = DATA_ROOT_DIR / f"CMIP6_{model}_historical_r1i1p1f1" / "mon"
    nc_paths = sorted(model_dir.rglob("*.nc"))
    if not nc_paths:
        logger.warning(f"No NetCDF files found for {model} in {model_dir}")
        return iris.cube.CubeList()
    logger.info(f"Loading {len(nc_paths)} files for {model}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return iris.load([str(p) for p in nc_paths])


def run_model(model: str, institute: str) -> Path:
    """Run Tier2_atmosphere_monthly suite for one CMIP6 historical model."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    db_path = OUTPUT_DIR / f"CMIP6_{model}_historical_r1i1p1f1.ddb"

    if db_path.is_file():
        logger.info(f"Skipping {model}: {db_path} already exists")
        return db_path

    cubes = load_model_cubes(model)
    if not cubes:
        logger.error(f"No data for {model}, skipping")
        return db_path

    data_info = DataSourceInformation(
        name=model,
        category="CMIP6",
        institute=institute,
        exp="historical",
        variant="r1i1p1f1",
        references=(
            f"https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.{institute}.{model}",
        ),
    )

    logger.info(f"Running suite '{SUITE_NAME}' for {model}")
    suite = Suite(
        SUITE_NAME,
        diagnostic_kwargs=DIAGNOSTIC_KWARGS,
        variable_kwargs={"timerange": TIMERANGE},
    )
    suite.get_database(cubes, data_info, database_resource=f"duckdb://{db_path}")
    logger.info(f"Written: {db_path}")
    return db_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark CMIP6 historical models with ClimateEval"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--model-index",
        type=int,
        metavar="N",
        help="0-based index into the MODELS list (for SLURM array jobs)",
    )
    group.add_argument(
        "--model",
        type=str,
        metavar="NAME",
        help="Model name, e.g. MPI-ESM1-2-LR",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    logger.remove()
    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | <level>{message}</level>"
    )
    logger.add(sys.stdout, level="DEBUG", format=fmt, colorize=True)

    args = parse_args()

    if args.model_index is not None:
        if not 0 <= args.model_index < len(MODELS):
            logger.error(
                f"--model-index {args.model_index} out of range "
                f"(0–{len(MODELS) - 1})"
            )
            sys.exit(1)
        models_to_run = [MODELS[args.model_index]]
    elif args.model is not None:
        lookup = {m: i for m, i in MODELS}
        if args.model not in lookup:
            logger.error(f"Unknown model '{args.model}'. Available: {list(lookup)}")
            sys.exit(1)
        models_to_run = [(args.model, lookup[args.model])]
    else:
        models_to_run = MODELS

    logger.info(f"Processing {len(models_to_run)} model(s)")
    for model, institute in models_to_run:
        logger.info(f"=== {model} ({institute}) ===")
        try:
            run_model(model, institute)
        except Exception:
            logger.exception(f"Failed to process {model}")


if __name__ == "__main__":
    main()
