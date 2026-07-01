"""Run all MPI-ESM1-2-LR tier suites and serve them together.

Computes (or reuses) the following databases:
  Tier1_sanity_checks.ddb      — reasonable-range checks (Amon)
  Tier1_consistency_checks.ddb — air-mass / water-vapour vs ERA5 (Amon)
  Tier2_ocean_monthly.ddb      — chl + siconc NH/SH vs ESA CCI OC (Omon + SImon)
  Tier3_dynamics.ddb           — Hovmöller (daily) + QBO vs ERA5 (Amon)

Then serves those together with any other Tier*.ddb files already present
in the working directory (e.g. Tier2_atmosphere_monthly.ddb).

Usage
-----
  # Recompute everything and serve:
  python run_mpi_esm1_2_lr.py

  # Skip suites whose .ddb already exists, only serve:
  python run_mpi_esm1_2_lr.py --skip-existing

Port: 8770
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import iris
from iris.cube import CubeList
from loguru import logger

from climateeval.data import DataSourceInformation
from climateeval.report import serve
from climateeval.suites import Suite

logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
)

# ---------------------------------------------------------------------------
_V = "v20190710"
_BASE = "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1"
DATA_ROOT = Path("/work/bd1179/b309141/climateeval_input")
PORT = 8770

_DATA_INFO = DataSourceInformation(
    name="MPI-ESM1-2-LR",
    category="CMIP6",
    institute="MPI-M",
    exp="historical",
    variant="r1i1p1f1",
    references=(
        "https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.MPI-M.MPI-ESM1-2-LR",
    ),
)

# Amon 1990-2009: one file per variable, covers Tier1 + Tier2_atmosphere_monthly
_AMON_DIR = Path(f"{_BASE}/Amon")
_AMON_PERIOD = "199001-200912"

# Ocean + sea ice (Tier2_ocean_monthly)
_OCEAN_FILES = [
    f"{_BASE}/Omon/chl/gn/{_V}/chl_Omon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_{_AMON_PERIOD}.nc",
    f"{_BASE}/SImon/siconc/gn/{_V}/siconc_SImon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_{_AMON_PERIOD}.nc",
]

# Daily atmosphere + ocean + monthly ua (Tier3_dynamics)
_DYNAMICS_FILES = {
    "pr_day":   f"{_BASE}/day/pr/gn/{_V}/pr_day_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_19900101-20091231.nc",
    "rlut_day": f"{_BASE}/day/rlut/gn/{_V}/rlut_day_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_19900101-20091231.nc",
    "tos_day":  f"{_BASE}/Oday/tos/gn/{_V}/tos_Oday_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_19900101-20091231.nc",
    "ua_mon":   f"{_BASE}/Amon/ua/gn/{_V}/ua_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_{_AMON_PERIOD}.nc",
}

_DIAG_KWARGS = {
    "data_root_dir": DATA_ROOT,
    "fail_on_missing_data": False,
    "fail_on_metric_error": False,
    "download_missing_data": False,
}
# ---------------------------------------------------------------------------


def _load_amon() -> CubeList:
    """Load the 1990-2009 Amon file for every available variable."""
    cubes: CubeList = CubeList([])
    for var_dir in sorted(_AMON_DIR.iterdir()):
        nc_files = sorted(
            (var_dir / "gn" / _V).glob(f"*{_AMON_PERIOD}.nc")
            if (var_dir / "gn" / _V).is_dir()
            else []
        )
        for nc in nc_files:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loaded = iris.load(str(nc))
            cubes.extend(loaded)
    logger.info("Loaded {} Amon cube(s) for period {}", len(cubes), _AMON_PERIOD)
    return cubes


def _load(paths: list[str] | dict[str, str]) -> CubeList:
    items = list(paths.items()) if isinstance(paths, dict) else [(Path(p).name, p) for p in paths]
    cubes: CubeList = CubeList([])
    for label, path in items:
        nc = Path(path)
        if nc.is_file():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                loaded = iris.load(str(nc))
            cubes.extend(loaded)
            logger.info("Loaded {}: {} cube(s)", label, len(loaded))
        else:
            logger.warning("File not found — skipping: {}", path)
    return cubes


def _run_suite(
    suite_name: str,
    db_path: Path,
    cubes: CubeList,
    *,
    skip_existing: bool,
) -> None:
    if skip_existing and db_path.is_file():
        logger.info("Reusing existing: {}", db_path)
        return
    if db_path.is_file():
        db_path.unlink()
    Path(str(db_path) + ".wal").unlink(missing_ok=True)
    suite = Suite(suite_name, diagnostic_kwargs=_DIAG_KWARGS)
    logger.info("Running {} → {}", suite_name, db_path)
    suite.get_database(cubes, _DATA_INFO, database_resource=f"duckdb://{db_path}")
    logger.info("Written: {}", db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip suites whose .ddb file already exists",
    )
    args = parser.parse_args()

    # ── Tier1: sanity + consistency checks (Amon) ───────────────────────────
    amon_cubes = _load_amon()
    _run_suite("Tier1_sanity_checks", Path("Tier1_sanity_checks.ddb"), amon_cubes, skip_existing=args.skip_existing)
    _run_suite("Tier1_consistency_checks", Path("Tier1_consistency_checks.ddb"), amon_cubes, skip_existing=args.skip_existing)

    # ── Tier2_ocean_monthly (chl + siconc NH/SH) ────────────────────────────
    ocean_cubes = _load(_OCEAN_FILES)
    _run_suite("Tier2_ocean_monthly", Path("Tier2_ocean_monthly.ddb"), ocean_cubes, skip_existing=args.skip_existing)

    # ── Tier3_dynamics (Hovmöller + QBO) ────────────────────────────────────
    dynamics_cubes = _load(_DYNAMICS_FILES)
    _run_suite("Tier3_dynamics", Path("Tier3_dynamics.ddb"), dynamics_cubes, skip_existing=args.skip_existing)

    # ── Serve all Tier*.ddb files in the working directory ───────────────────
    db_paths = sorted(Path(".").glob("Tier*.ddb"))
    if not db_paths:
        logger.error("No Tier*.ddb files found — nothing to serve")
        sys.exit(1)

    logger.info("Serving {} tier(s): {}", len(db_paths), [p.stem for p in db_paths])
    logger.info("http://127.0.0.1:{} — Ctrl-C to stop", PORT)
    serve(
        db_paths=db_paths,
        model_name="MPI-ESM1-2-LR",
        host="127.0.0.1",
        port=PORT,
        open_browser=False,
    )


if __name__ == "__main__":
    main()
