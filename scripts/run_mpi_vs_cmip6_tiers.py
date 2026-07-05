"""Benchmark MPI-ESM1-2-LR vs ERA5 + 4 CMIP6 models across tier suites.

Runs the tier suites for MPI-ESM1-2-LR (period 1990-2009) and appends four
CMIP6 comparison models to the Tier2 atmosphere database, so the report shows
MPI-ESM1-2-LR against the observational references (ERA5, GPCP, CERES-EBAF)
*and* against CESM2, EC-Earth3, IPSL-CM6A-LR and MIROC6.

Databases written (one per tier, report-ready):
  output/Tier1_sanity_checks.ddb       — MPI-ESM1-2-LR reasonable-range checks
  output/Tier1_consistency_checks.ddb  — MPI-ESM1-2-LR vs ERA5 consistency
  output/Tier2_atmosphere_monthly.ddb  — all 5 models vs ERA5/GPCP/CERES-EBAF

Afterwards build the report outputs:
  pixi run python report.py output/Tier*.ddb --model MPI-ESM1-2-LR \
      --save-portable-html-output output/MPI-ESM1-2-LR_report.html --no-browser
  pixi run python report.py output/Tier*.ddb --model MPI-ESM1-2-LR \
      --static-out output/static_figures
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import iris
from iris.util import equalise_attributes, unify_time_units
from iris.cube import CubeList
from loguru import logger

from climateeval.data import DataSourceInformation
from climateeval.suites import Suite

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
)

# ---------------------------------------------------------------------------
_CMIP6_ROOT = Path("/work/ik1017/CMIP6/data/CMIP6/CMIP")
_TIMERANGE = "19900101/20091231"
_ENSEMBLE = "r1i1p1f1"
OUT_DIR = Path("output")

# (institute, dataset) — first entry is the model under evaluation
MODELS = [
    ("MPI-M", "MPI-ESM1-2-LR"),
    ("NCAR", "CESM2"),
    ("EC-Earth-Consortium", "EC-Earth3"),
    ("IPSL", "IPSL-CM6A-LR"),
    ("MIROC", "MIROC6"),
]

_DIAG_KWARGS: dict[str, Any] = {
    "data_root_dir": Path("/work/bd1179/b309141/climateeval_input"),
    "fail_on_missing_data": False,
    "fail_on_metric_error": False,
    "download_missing_data": False,
}
_VARIABLE_KWARGS: dict[str, Any] = {"timerange": _TIMERANGE}
# ---------------------------------------------------------------------------


def _overlaps_period(filename: str) -> bool:
    """Return True if the file's date token overlaps 1990-2009."""
    date_token = Path(filename).stem.rsplit("_", 1)[-1]
    if "-" not in date_token:
        return True
    start_str, end_str = date_token.split("-", 1)
    try:
        return int(end_str[:4]) >= 1990 and int(start_str[:4]) <= 2009
    except ValueError:
        return True


def _load_amon(institute: str, dataset: str) -> CubeList:
    """Load all Amon files (any grid, latest version) overlapping 1990-2009."""
    amon_root = _CMIP6_ROOT / institute / dataset / "historical" / _ENSEMBLE / "Amon"
    cubes: CubeList = CubeList([])
    if not amon_root.is_dir():
        return cubes
    for var_dir in sorted(amon_root.iterdir()):
        grid_dirs = sorted(d for d in var_dir.iterdir() if d.is_dir())
        if not grid_dirs:
            continue
        # Prefer the native grid if present, otherwise take the first grid
        grid_dir = next((d for d in grid_dirs if d.name == "gn"), grid_dirs[0])
        versions = sorted(grid_dir.iterdir())
        if not versions:
            continue
        relevant = [
            f for f in sorted(versions[-1].glob("*.nc")) if _overlaps_period(f.name)
        ]
        if not relevant:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = iris.load([str(f) for f in relevant])
            # Some models (e.g. EC-Earth3) split Amon output into one file
            # per year — concatenate to one cube per variable.
            equalise_attributes(loaded)
            unify_time_units(loaded)
            loaded = loaded.concatenate()
        cubes.extend(loaded)
    return cubes


def _data_info(institute: str, dataset: str) -> DataSourceInformation:
    return DataSourceInformation(
        name=dataset,
        category="CMIP6",
        institute=institute,
        exp="historical",
        variant=_ENSEMBLE,
        references=(
            f"https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.{institute}.{dataset}",
        ),
    )


def _run_suite(
    suite_name: str,
    db_path: Path,
    cubes: CubeList,
    info: DataSourceInformation,
    *,
    append: bool,
) -> None:
    if not append:
        db_path.unlink(missing_ok=True)
        Path(str(db_path) + ".wal").unlink(missing_ok=True)
    suite = Suite(
        suite_name,
        diagnostic_kwargs=_DIAG_KWARGS,
        variable_kwargs=_VARIABLE_KWARGS,
    )
    logger.info("Running {} for {} → {}", suite_name, info.name, db_path)
    suite.get_database(
        cubes, info, database_resource=f"duckdb://{db_path}", append=append
    )
    logger.info("Written {} → {}", info.name, db_path)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    # ── MPI-ESM1-2-LR: Tier1 suites + first entry of the Tier2 database ─────
    institute, dataset = MODELS[0]
    logger.info("Loading Amon data for {}/{}", institute, dataset)
    mpi_cubes = _load_amon(institute, dataset)
    logger.info("Loaded {} cube(s) for {}", len(mpi_cubes), dataset)
    mpi_info = _data_info(institute, dataset)

    _run_suite(
        "Tier1_sanity_checks",
        OUT_DIR / "Tier1_sanity_checks.ddb",
        mpi_cubes,
        mpi_info,
        append=False,
    )
    _run_suite(
        "Tier1_consistency_checks",
        OUT_DIR / "Tier1_consistency_checks.ddb",
        mpi_cubes,
        mpi_info,
        append=False,
    )

    tier2_db = OUT_DIR / "Tier2_atmosphere_monthly.ddb"
    _run_suite("Tier2_atmosphere_monthly", tier2_db, mpi_cubes, mpi_info, append=False)
    del mpi_cubes

    # ── Append the 4 CMIP6 comparison models to the Tier2 database ──────────
    for institute, dataset in MODELS[1:]:
        logger.info("Loading Amon data for {}/{}", institute, dataset)
        cubes = _load_amon(institute, dataset)
        if not cubes:
            logger.warning("No Amon cubes for {} — skipping", dataset)
            continue
        logger.info("Loaded {} cube(s) for {}", len(cubes), dataset)
        _run_suite(
            "Tier2_atmosphere_monthly",
            tier2_db,
            cubes,
            _data_info(institute, dataset),
            append=True,
        )
        del cubes

    logger.info("All done: {}", sorted(str(p) for p in OUT_DIR.glob("*.ddb")))


if __name__ == "__main__":
    main()
