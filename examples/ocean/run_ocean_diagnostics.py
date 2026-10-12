"""Run the ClimateEval ocean and sea-ice diagnostics on a CMIP6 model.

This is a worked example configured for MPI-ESM1-2-LR. To evaluate a different
model, edit the ``MODEL`` block below (the data directory and the
``DataSourceInformation``) and run::

    python run_ocean_diagnostics.py

It produces four result databases (``seaice.ddb``, ``ocean_ohc.ddb``,
``ocean_sst.ddb``, ``ocean_amoc.ddb``) and a single self-contained
``report.html``.

Separate runs are used because ClimateEval applies one time range to all
datasets, and each observational reference covers a different period (a
reference that does not cover the requested start year is dropped):

* sea ice (1997-2009): OSI-450, HadISST, NSIDC-G02202
* ocean heat content (1979-1989): EN4, IAP (covers the 1979-1988 OHC baseline)
* SST / salinity / chlorophyll (1997-2009): ESACCI-SST, NOAA-ERSSTv5, HadISST,
  WOA, ESACCI-OC
* AMOC (2004-2014): RAPID (which only starts in 2004, hence its own window)

See README.md for the full setup (installing, staging the references, env vars).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import iris
from iris.util import equalise_attributes, unify_time_units

from climateeval.data import DataSourceInformation
from climateeval.suites import Suite

# ---------------------------------------------------------------------------
# Paths (override via environment variables, see README.md)
# ---------------------------------------------------------------------------
# Directory where the observational references were staged (stage_obs_refs.sh).
DATA_ROOT = os.environ.get("CLIMATEEVAL_DATA_ROOT", "./climateeval_input")
# Where to write the result databases and the report.
RESULTS_DIR = Path(os.environ.get("CLIMATEEVAL_RESULTS_DIR", "./results"))

# ORAS5 (mixed-layer depth) and RAPID (AMOC) live outside the staged tree. The
# data sources already default to the shared Levante copies; override only if
# they ever move.
os.environ.setdefault(
    "CLIMATEEVAL_ORAS5_RAW_DIR",
    "/work/bd1083/b382555/extraobsraw/Tier2/ORAS5/data",
)
os.environ.setdefault(
    "CLIMATEEVAL_ORAS5_GRID_DIR",
    "/work/bd1083/b382555/extraobsraw/Tier2/ORAS5/grids",
)

# ---------------------------------------------------------------------------
# MODEL — edit this block to evaluate a different model
# ---------------------------------------------------------------------------
MODEL_DIR = (
    "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r10i1p1f1"
)
MODEL_GRID = "gn"  # native ("gn") or regridded ("gr") grid label in the filename
MODEL_VERSION = "v20190710"  # data version subdirectory; use "*" to match any
MODEL_INFO = DataSourceInformation(
    name="MPI-ESM1-2-LR",
    category="CMIP6",
    institute="MPI-M",
    exp="historical",
    variant="r10i1p1f1",
)

# ---------------------------------------------------------------------------
# The runs (suite, [(variable, MIP table)], timerange, output database).
# The variables must all exist for the chosen variant. AMOC gets its own run
# because its reference (RAPID) only starts in 2004.
# ---------------------------------------------------------------------------
RUNS = (
    (
        "Tier2_seaice_monthly",
        (("siconc", "SImon"),),
        "19970101/20071231",
        "seaice.ddb",
    ),
    (
        "Tier2_ocean_3d",
        (
            ("so", "Omon"),
            ("thetao", "Omon"),
        ),
        "19790101/19821231",
        "ocean_3d.ddb",
    ),
    (
        "Tier2_ocean_acc",
        (("uo", "Omon"),),
        "19850101/19901231",
        "ocean_acc.ddb",
    ),
    (
        "Tier2_ocean_amoc",
        (
            ("msftmz", "Omon"),
            ("hfbasin", "Omon"),
        ),
        "20050101/20131231",
        "ocean_amoc.ddb",
    ),
    (
        "Tier2_ocean_2d",
        (
            ("mlotst", "Omon"),
            ("zos", "Omon"),
            ("tos", "Omon"),
            ("sos", "Omon"),
        ),
        "19850101/19901231",
        "ocean_2d.ddb",
    ),
)


def model_files(var: str, mip: str) -> list[str]:
    """Return the model files for one variable (matches any time span)."""
    base = Path(MODEL_DIR) / mip / var / MODEL_GRID
    pattern = f"{MODEL_VERSION}/{var}_{mip}_*.nc"
    files = sorted(str(path) for path in base.glob(pattern))
    if not files:
        pattern = f"*/{var}_{mip}_*.nc"
        files = sorted(str(path) for path in base.glob(pattern))
        if not files:
            msg = f"No model files found for '{var}' ({mip}): {base / pattern}"
            raise FileNotFoundError(msg)
    return files


def run(suite_name: str, variables: tuple, timerange: str, db: str) -> Path:
    """Run one suite and write its result database."""
    paths = [f for var, mip in variables for f in model_files(var, mip)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cubes = iris.load(paths)
        # CMIP6 output is split into per-decade files: concatenate the time
        # chunks back into one cube per variable (the per-file global
        # attributes and time units must be reconciled first, otherwise the
        # chunks stay separate and the single-cube extraction would reject
        # them).
        equalise_attributes(cubes)
        unify_time_units(cubes)
        cubes = cubes.concatenate()

    db_path = RESULTS_DIR / db
    if db_path.is_file():
        db_path.unlink()
    Suite(
        suite_name,
        diagnostic_kwargs={
            "fail_on_missing_data": False,
            "download_missing_data": False,
            "data_root_dir": DATA_ROOT,
        },
        variable_kwargs={"timerange": timerange},
    ).get_database(cubes, MODEL_INFO, database_resource=f"duckdb://{db_path}")
    print(f"WROTE {db_path}")
    return db_path


def build_report(db_paths: list[Path]) -> None:
    """Build the HTML report if the (optional) report module is available."""
    try:
        from climateeval.report import generate_portable_html  # noqa: PLC0415
    except ImportError:
        print(
            "climateeval.report is not installed (it lives on the "
            "'kd/ocean-diag-report' branch); skipping the HTML report. "
            f"Result databases are in {RESULTS_DIR}.",
        )
        return
    html = generate_portable_html(db_paths, offline=True, model_name=MODEL_INFO.name)
    report = RESULTS_DIR / "report.html"
    report.write_text(html)
    print(f"WROTE {report}  (open in a browser)")


def main() -> None:
    """Run every suite and build the HTML report."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    db_paths = [run(*spec) for spec in RUNS]
    build_report(db_paths)


if __name__ == "__main__":
    main()
