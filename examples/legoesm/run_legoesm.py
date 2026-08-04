"""Benchmark an external (non-CMIP archive) model against observations.

This example evaluates output from the legoESM AMIP experiment with the
``Tier2_atmosphere_monthly`` suite and writes a self-contained HTML report. It
is a template for benchmarking *any* model whose CMOR-compliant NetCDF files you
can point Iris at -- replace the paths and the ``DataSourceInformation`` block
with your own.

Configure it through environment variables (with sensible fall-backs):

    LEGOESM_CMOR_DIR          directory with the model's CMORized ``*.nc`` files
    CLIMATEEVAL_DATA_ROOT     directory with the CMORized observational references
    CLIMATEEVAL_RESULTS_DIR   where to write the ``.ddb`` database and report

Run it from inside the pixi environment::

    export LEGOESM_CMOR_DIR=/path/to/legoesm/cmor/Amon
    export CLIMATEEVAL_DATA_ROOT=~/climateeval_input
    export CLIMATEEVAL_RESULTS_DIR=~/legoesm_results
    pixi run python run_legoesm.py

See ``docs/benchmarking-legoesm.md`` for the full walkthrough.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import iris
from loguru import logger

from climateeval.data import DataSourceInformation
from climateeval.report import generate_portable_html
from climateeval.suites import Suite

logger = logger.opt(colors=True)

# --- Configuration ---------------------------------------------------------
# Directory with the model's CMORized NetCDF output (edit or set the env var).
MODEL_CMOR_DIR = Path(
    os.environ.get("LEGOESM_CMOR_DIR", "/path/to/legoesm/cmor/Amon"),
)
# Directory with the CMORized observational references.
DATA_ROOT_DIR = Path(
    os.environ.get("CLIMATEEVAL_DATA_ROOT", str(Path.home() / "climateeval_input")),
)
RESULTS_DIR = Path(
    os.environ.get("CLIMATEEVAL_RESULTS_DIR", str(Path.home() / "legoesm_results")),
)
SUITE_NAME = "Tier2_atmosphere_monthly"
# ClimateEval applies one time range to every dataset; pick one your model and
# the references both cover.
TIMERANGE = "19790101/19790930"


def run_suite(cubes: iris.cube.CubeList, data_info: DataSourceInformation) -> Path:
    """Run the suite on ``cubes`` and return the path of the written database."""
    db_path = RESULTS_DIR / f"{SUITE_NAME}.ddb"
    if db_path.is_file():
        db_path.unlink()
    suite = Suite(
        SUITE_NAME,
        diagnostic_kwargs={
            "data_root_dir": DATA_ROOT_DIR,
            "fail_on_missing_data": False,
            "download_missing_data": False,
        },
        variable_kwargs={"timerange": TIMERANGE},
    )
    suite.get_database(cubes, data_info, database_resource=f"duckdb://{db_path}")
    logger.info(f"Database written to {db_path}")
    return db_path


def build_report(db_path: Path, model_name: str) -> None:
    """Write a self-contained HTML report next to the database."""
    html = generate_portable_html([db_path], offline=True, model_name=model_name)
    report = RESULTS_DIR / "report.html"
    report.write_text(html)
    logger.info(f"Report written to {report}")


def main() -> None:
    """Load the model output, run the suite and build the report."""
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | <level>{message}</level>"
        ),
        colorize=True,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Provenance recorded in the report. `category` is free-form -- it just
    # groups the dataset in the output; use "CMIP6" for archive models.
    model_info = DataSourceInformation(
        name="legoESM-1-0",
        category="legoESM",
        institute="MPI-M",
        exp="amip",
        variant="r1i1p1f1",
    )

    model_paths = sorted(MODEL_CMOR_DIR.glob("*.nc"))
    if not model_paths:
        msg = f"No NetCDF files found in {MODEL_CMOR_DIR}"
        raise SystemExit(msg)
    logger.info(f"Loading {len(model_paths)} files from {MODEL_CMOR_DIR}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cubes = iris.load([str(p) for p in model_paths])
    logger.info(f"Loaded {len(cubes)} cubes")

    db_path = run_suite(cubes, model_info)
    build_report(db_path, model_info.name)


if __name__ == "__main__":
    main()
