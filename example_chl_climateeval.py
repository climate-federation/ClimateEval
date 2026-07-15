"""
Run ChlBGCEvaluation through the ClimateEval pipeline.

Produces:
  - Tier2_ocean.ddb           — DuckDB database for HTML report
  - paper_figures/*.png       — all paper figures

The suite loads five variables per model (chl, tos, mlotst, siconc, rsds).
Variables that are missing for a given model are skipped with a warning;
the diagnostic handles partial data gracefully.

Usage:
    python example_chl_climateeval.py
    python generate_html.py Tier2_ocean.ddb -o Tier2_ocean_report.html
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import iris
from iris.util import equalise_attributes, unify_time_units
from loguru import logger

logger = logger.opt(colors=True)
logger.remove()
logger.add(
    sys.stdout, level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    colorize=True,
)

# ── All 7 models from the manuscript ─────────────────────────────────────────
MODELS = {
    "ACCESS-ESM1-5": {
        "institute": "CSIRO",
        "variant":   "r1i1p1f1",
        "grid":      "gn",
        "version":   "v20191115",
    },
    "CanESM5": {
        "institute": "CCCma",
        "variant":   "r1i1p1f1",
        "grid":      "gn",
        "version":   "v20190429",
    },
    "CESM2": {
        "institute": "NCAR",
        "variant":   "r1i1p1f1",
        "grid":      "gn",
        "version":   "v20190308",
    },
    "GFDL-ESM4": {
        "institute": "NOAA-GFDL",
        "variant":   "r1i1p1f1",
        "grid":      "gr",
        "version":   "v20190726",
    },
    "IPSL-CM6A-LR": {
        "institute": "IPSL",
        "variant":   "r1i1p1f1",
        "grid":      "gn",
        "version":   "v20180803",
    },
    "MPI-ESM-1-2-HAM": {
        "institute": "HAMMOZ-Consortium",
        "variant":   "r1i1p1f1",
        "grid":      "gn",
        "version":   "v20190627",
    },
    "NorESM2-LM": {
        "institute": "NCC",
        "variant":   "r1i1p1f1",
        "grid":      "gn",
        "version":   "v20190815",
    },
}

# ── Variables to load per model ───────────────────────────────────────────────
# Each entry: (var_name, mip_table, grid_override_or_None)
# grid_override allows per-variable grid (e.g. rsds uses Amon not Omon)
VARIABLES = [
    ("chl",     "Omon", None),    # surface chlorophyll-a  (required)
    ("tos",     "Omon", None),    # sea surface temperature
    ("mlotst",  "Omon", None),    # mixed-layer depth
    ("siconc",  "SImon", None),   # sea-ice concentration
    ("rsds",    "Amon", None),    # surface downwelling shortwave
]

CMIP6_ROOT = Path("/work/ik1017/CMIP6/data/CMIP6/CMIP")
DATA_ROOT  = Path("/work/bd1179/b384036/climateeval_input")
SUITE_NAME = "Tier2_ocean"
DB_PATH    = Path(f"{SUITE_NAME}.ddb")


# ── Path helpers ──────────────────────────────────────────────────────────────

def _var_paths(
    model: str,
    cfg: dict,
    var: str,
    mip: str,
    grid_override: str | None = None,
) -> list[Path]:
    """Return sorted netCDF paths for one variable of one model.

    Falls back to the pre-staged DATA_ROOT directory if the CMIP6 pool
    does not contain the files.
    """
    grid = grid_override or cfg["grid"]
    base = (
        CMIP6_ROOT
        / cfg["institute"]
        / model
        / "historical"
        / cfg["variant"]
        / mip
        / var
        / grid
        / cfg["version"]
    )
    paths = sorted(base.glob(f"{var}_{mip}_*.nc"))
    if not paths:
        staged = sorted(DATA_ROOT.glob(f"{var}_{mip}_{model}_historical_*.nc"))
        if staged:
            return staged
        logger.warning(f"  [{model}] no {var} files found under {base}")
    return paths


def _load_var(
    model: str,
    cfg: dict,
    var: str,
    mip: str,
    grid_override: str | None = None,
) -> iris.cube.CubeList:
    """Load and concatenate one variable for one model.

    Returns an empty CubeList if files are missing.
    """
    paths = _var_paths(model, cfg, var, mip, grid_override)
    if not paths:
        return iris.cube.CubeList()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cubes = iris.load([str(p) for p in paths])
    # Concatenate per-decade chunks (CMIP6 files are often split by decade)
    try:
        equalise_attributes(cubes)
        unify_time_units(cubes)
        cubes = cubes.concatenate()
    except Exception as e:
        logger.warning(f"  [{model}] {var} concatenation failed: {e}")
    return cubes


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from climateeval.data._utils import DataSourceInformation
    from climateeval.suites._base import Suite

    if DB_PATH.is_file():
        DB_PATH.unlink()
        logger.info(f"Removed existing {DB_PATH}")

    suite = Suite(
        SUITE_NAME,
        diagnostic_kwargs={
            "data_root_dir":         DATA_ROOT,
            "fail_on_missing_data":  False,
            "download_missing_data": False,
        },
        variable_kwargs={"timerange": "19970101/20141231"},
    )

    for model, cfg in MODELS.items():
        logger.info(f"Loading {model}...")

        # Collect cubes for all variables into one CubeList
        all_cubes = iris.cube.CubeList()
        for var, mip, grid_override in VARIABLES:
            var_cubes = _load_var(model, cfg, var, mip, grid_override)
            if var_cubes:
                logger.info(f"  [{model}] loaded {var} ({len(var_cubes)} cube(s))")
                all_cubes.extend(var_cubes)
            # Missing variables are skipped — the diagnostic handles this

        # Skip model entirely if chl is missing (required for metrics table)
        chl_present = any(
            c.var_name == "chl" or "chlorophyll" in c.name().lower()
            for c in all_cubes
        )
        if not chl_present:
            logger.warning(f"  [{model}] skipping: no chl data found")
            continue

        data_info = DataSourceInformation(
            name=model,
            category="CMIP6",
            institute=cfg["institute"],
            exp="historical",
            variant=cfg["variant"],
            references=(),
        )

        logger.info(f"Running suite for {model} ({len(all_cubes)} cubes total)...")
        try:
            suite.get_database(
                all_cubes,
                data_info,
                database_resource=f"duckdb://{DB_PATH}",
            )
            logger.success(f"  {model} done")
        except Exception as e:
            logger.warning(f"  {model} failed: {e}")
            import traceback
            logger.opt(colors=False).debug(traceback.format_exc())

    logger.info(f"Database written to {DB_PATH}")
    logger.info("Generate HTML report with:")
    logger.info(f"  python3 generate_html.py {DB_PATH} -o Tier2_ocean_report.html")


if __name__ == "__main__":
    main()
