"""Showcase: run every tier on MPI-ESM1-2-HR historical and build one report.

Benchmarks MPI-ESM1-2-HR (historical, r1i1p1f1) against ERA5 / observations and
CMIP6 (via each suite's built-in references and ``other_data``) across ALL
tiers, writing one DuckDB database per tier and a single combined interactive
HTML report. It exists to develop and sanity-check the report/visualizations:
running every tier exercises every diagnostic and figure type in one go.

Fail-soft: a tier whose input data is missing or errors is logged and skipped,
so the report is built from whatever succeeded (useful for spotting what does
and does not work).

Meant to run on DKRZ Levante (the paths point at the shared CMIP6 archive and
Manuel's reference pool). It reads a lot of data, so submit it to a compute
node -- see scripts/submit_showcase.sh.

    pixi run python scripts/showcase_all_tiers.py                 # write showcase_output/report.html
    pixi run python scripts/showcase_all_tiers.py --serve         # interactive server instead
    pixi run python scripts/showcase_all_tiers.py --skip-existing # reuse existing .ddb files
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import iris
from iris.cube import CubeList
from iris.util import equalise_attributes, unify_time_units
from loguru import logger

from climateeval._config import setup_iris_combine_policy
from climateeval.data import DataSourceInformation
from climateeval.report import generate_portable_html, serve
from climateeval.suites import Suite

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
)

# --- Configuration ---------------------------------------------------------
MODEL = "MPI-ESM1-2-HR"
INSTITUTE = "MPI-M"
VARIANT = "r1i1p1f1"

_ARCHIVE = Path("/work/ik1017/CMIP6/data/CMIP6/CMIP") / INSTITUTE / MODEL
HIST = _ARCHIVE / "historical" / VARIANT
ABRUPT = _ARCHIVE / "abrupt-4xCO2" / VARIANT
PICTRL = _ARCHIVE / "piControl" / VARIANT

DATA_ROOT = Path("/work/bd1179/b309141/climateeval_input")
OUTDIR = Path(__file__).resolve().parent.parent / "showcase_output"
PORT = 8771

# One time range per tier. ClimateEval drops any reference that does not cover
# the requested start year, so the monthly window starts in 2001 -- after the
# latest-starting atmosphere reference (CERES-EBAF begins in 2000) -- so ERA5/
# obs comparisons and metrics are actually produced. (A few later references,
# e.g. RAPID/AMOC from 2004, may still be dropped; narrow the window per tier if
# you need them.) The daily/sub-daily tiers use shorter windows to bound volume.
MON_RANGE = "20010101/20141231"
DYN_RANGE = "20050101/20141231"
SUB_RANGE = "20130101/20141231"
_MON_YEARS = (2001, 2014)
_DYN_YEARS = (2005, 2014)
_SUB_YEARS = (2013, 2014)
_ECS_YEARS = (1850, 2009)  # ~150 yr of abrupt-4xCO2 / piControl for the Gregory fit

_OCEAN_VARS = ["tos", "so", "thetao", "chl", "msftmz", "msftyz", "mlotst"]
_ECS_VARS = ["rsdt", "rsut", "rlut", "tas"]

DATA_INFO = DataSourceInformation(
    name=MODEL,
    category="CMIP6",
    institute=INSTITUTE,
    exp="historical",
    variant=VARIANT,
    references=(
        "https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.MPI-M.MPI-ESM1-2-HR",
    ),
)
DIAG_KWARGS = {
    "data_root_dir": DATA_ROOT,
    "fail_on_missing_data": False,
    "fail_on_metric_error": False,
    "download_missing_data": False,
}

_PERIOD = re.compile(r"_(\d{4})\d+-(\d{4})\d+\.nc$")
# ---------------------------------------------------------------------------


def _overlaps(name: str, years: tuple[int, int]) -> bool:
    """True if the file's time span (parsed from its name) overlaps ``years``."""
    m = _PERIOD.search(name)
    if not m:
        return True  # time-invariant or unparseable: keep it
    start, end = int(m.group(1)), int(m.group(2))
    return start <= years[1] and end >= years[0]


def _latest_version_dir(var_dir: Path) -> Path | None:
    """Return the newest ``v<date>`` directory under a variable's grid folder."""
    for grid in ("gn", "gr", "gn1", "gr1"):
        grid_dir = var_dir / grid
        if grid_dir.is_dir():
            versions = sorted(p for p in grid_dir.glob("v*") if p.is_dir())
            if versions:
                return versions[-1]
    return None


def _load_table(
    exp_dir: Path,
    mip: str,
    variables: list[str] | None,
    years: tuple[int, int],
) -> CubeList:
    """Load a CMOR table's variables (latest version, files overlapping years)."""
    table_dir = exp_dir / mip
    cubes: CubeList = CubeList([])
    if not table_dir.is_dir():
        logger.warning("No {} table under {}", mip, exp_dir)
        return cubes
    var_dirs = (
        [table_dir / v for v in variables]
        if variables
        else sorted(p for p in table_dir.iterdir() if p.is_dir())
    )
    for var_dir in var_dirs:
        vdir = _latest_version_dir(var_dir) if var_dir.is_dir() else None
        if vdir is None:
            continue
        for nc in sorted(vdir.glob("*.nc")):
            if not _overlaps(nc.name, years):
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cubes.extend(iris.load(str(nc)))
    if cubes:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            equalise_attributes(cubes)
            unify_time_units(cubes)
            cubes = cubes.concatenate()
    logger.info("Loaded {} cube(s) from {}/{}", len(cubes), exp_dir.name, mip)
    return cubes


def _safe(thunk: Callable[[], Any], default: Any) -> Any:
    """Run a loader, returning ``default`` (and logging) if it raises."""
    try:
        return thunk()
    except Exception as exc:  # noqa: BLE001
        logger.error("Data load failed: {}", exc)
        return default


def _load_dynamics() -> CubeList:
    """Daily pr/rlut (day), daily tos (Oday) and monthly ua (Amon) for Tier 3."""
    cubes: CubeList = CubeList([])
    cubes.extend(_load_table(HIST, "day", ["pr", "rlut"], _DYN_YEARS))
    cubes.extend(_load_table(HIST, "Oday", ["tos"], _DYN_YEARS))
    cubes.extend(_load_table(HIST, "Amon", ["ua"], _DYN_YEARS))
    return cubes


def _load_ecs() -> dict[str, CubeList]:
    """abrupt-4xCO2 and piControl radiation/temperature for the ECS diagnostic."""
    with setup_iris_combine_policy():
        return {
            "4xco2": _load_table(ABRUPT, "Amon", _ECS_VARS, _ECS_YEARS),
            "picontrol": _load_table(PICTRL, "Amon", _ECS_VARS, _ECS_YEARS),
        }


def _run_suite(
    suite_name: str,
    data: Any,
    timerange: str | None,
    *,
    skip_existing: bool,
) -> Path | None:
    """Run one tier suite (fail-soft) and return its database path, or None."""
    db = OUTDIR / f"{suite_name}.ddb"
    if skip_existing and db.is_file():
        logger.info("Reusing existing {}", db.name)
        return db
    if not data:
        logger.warning("{}: no input data, skipping", suite_name)
        return None
    try:
        if db.is_file():
            db.unlink()
        Path(str(db) + ".wal").unlink(missing_ok=True)
        variable_kwargs = {"timerange": timerange} if timerange else {}
        suite = Suite(
            suite_name,
            diagnostic_kwargs=DIAG_KWARGS,
            variable_kwargs=variable_kwargs,
        )
        logger.info("Running {} ...", suite_name)
        suite.get_database(data, DATA_INFO, database_resource=f"duckdb://{db}")
        logger.success("  wrote {}", db.name)
    except Exception as exc:  # noqa: BLE001
        logger.error("  {} FAILED: {}", suite_name, exc)
        if db.is_file():
            db.unlink()
        return None
    return db


def main() -> None:
    """Run all tiers on MPI-ESM1-2-HR and build the combined report."""
    parser = argparse.ArgumentParser(
        description="Run every ClimateEval tier on MPI-ESM1-2-HR and build a report.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve interactively instead of writing report.html.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse any tier .ddb that already exists.",
    )
    args = parser.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    def run(name: str, data: Any, timerange: str | None) -> None:
        db = _run_suite(name, data, timerange, skip_existing=args.skip_existing)
        if db is not None:
            written.append(db)

    # Tier 1 + Tier 2 atmosphere all read monthly Amon: load it once and reuse.
    amon = _safe(lambda: _load_table(HIST, "Amon", None, _MON_YEARS), CubeList([]))
    run("Tier1_sanity_checks", amon, MON_RANGE)
    run("Tier1_consistency_checks", amon, MON_RANGE)
    run("Tier2_atmosphere_monthly", amon, MON_RANGE)
    del amon

    # Tier 2 ocean + sea ice.
    run(
        "Tier2_ocean_monthly",
        _safe(lambda: _load_table(HIST, "Omon", _OCEAN_VARS, _MON_YEARS), CubeList([])),
        MON_RANGE,
    )
    run(
        "Tier2_sea_ice_monthly",
        _safe(lambda: _load_table(HIST, "SImon", ["siconc"], _MON_YEARS), CubeList([])),
        MON_RANGE,
    )

    # Tier 3 variability / processes / feedbacks.
    run("Tier3_dynamics", _safe(_load_dynamics, CubeList([])), DYN_RANGE)
    run(
        "Tier3_atmosphere_subdaily",
        _safe(lambda: _load_table(HIST, "3hr", ["pr"], _SUB_YEARS), CubeList([])),
        SUB_RANGE,
    )
    run("Tier3_ecs", _safe(_load_ecs, {}), None)

    if not written:
        logger.error("No tier databases were produced -- check the data paths.")
        raise SystemExit(1)
    logger.info(
        "Produced {} tier(s): {}",
        len(written),
        ", ".join(p.stem for p in sorted(written)),
    )

    if args.serve:
        logger.info("Serving at http://127.0.0.1:{} (Ctrl-C to stop)", PORT)
        serve(sorted(written), model_name=MODEL, host="127.0.0.1", port=PORT, open_browser=False)
        return

    report = OUTDIR / "report.html"
    report.write_text(
        generate_portable_html(sorted(written), offline=True, model_name=MODEL),
    )
    logger.success("Report written: {}", report)


if __name__ == "__main__":
    main()
