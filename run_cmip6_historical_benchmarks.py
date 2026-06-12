"""Benchmark all locally available CMIP6 historical r1i1p1f1 models.

Discovers every CMIP6 historical r1i1p1f1 model that has tas available on
the DKRZ filesystem, loads their Amon data, and runs Tier2_atmosphere_monthly,
appending each model to the same DuckDB so the leaderboard shows all of them.

Output: Tier2_atmosphere_cmip6_historical.ddb

Usage
-----
  # Benchmark all available models (may take hours):
  python run_cmip6_historical_benchmarks.py

  # Quick test with first N models only:
  python run_cmip6_historical_benchmarks.py --max-models 5

  # Resume a partial run (skip models already in the database):
  python run_cmip6_historical_benchmarks.py --skip-existing

  # Serve the result alongside other tier databases:
  python run_cmip6_historical_benchmarks.py --serve-only
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import ibis
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
_CMIP6_ROOT = Path("/work/ik1017/CMIP6/data/CMIP6/CMIP")
_TIMERANGE = "19900101/20091231"
_ENSEMBLE = "r1i1p1f1"
_GRID = "gn"

DB_PATH = Path("Tier2_atmosphere_cmip6_historical.ddb")
PORT = 8771

_DIAG_KWARGS: dict[str, Any] = {
    "data_root_dir": Path("/work/bd1179/b309141/climateeval_input"),
    "fail_on_missing_data": False,
    "fail_on_metric_error": False,
    "download_missing_data": False,
}
_VARIABLE_KWARGS: dict[str, Any] = {
    "timerange": _TIMERANGE,
}
# ---------------------------------------------------------------------------


def _discover_models() -> list[dict[str, str]]:
    """Return list of {institute, dataset, grid, version} for all models with tas."""
    models = []
    for tas_dir in sorted(
        _CMIP6_ROOT.glob(
            f"*/*/historical/{_ENSEMBLE}/Amon/tas/{_GRID}"
        )
    ):
        # Pick the latest version
        versions = sorted(tas_dir.iterdir())
        if not versions:
            continue
        version = versions[-1].name
        parts = tas_dir.parts
        institute = parts[-7]
        dataset = parts[-6]
        models.append(
            {
                "institute": institute,
                "dataset": dataset,
                "grid": _GRID,
                "version": version,
            }
        )
    logger.info("Discovered {} CMIP6 historical models with Amon/tas/{}", len(models), _GRID)
    return models


def _load_amon(institute: str, dataset: str, grid: str) -> CubeList:
    """Load all Amon NC files for a model that overlap with the target period."""
    amon_root = _CMIP6_ROOT / institute / dataset / "historical" / _ENSEMBLE / "Amon"
    cubes: CubeList = CubeList([])
    if not amon_root.is_dir():
        return cubes

    for var_dir in sorted(amon_root.iterdir()):
        grid_dir = var_dir / grid
        if not grid_dir.is_dir():
            continue
        versions = sorted(grid_dir.iterdir())
        if not versions:
            continue
        version_dir = versions[-1]
        nc_files = sorted(version_dir.glob("*.nc"))
        # Keep only files whose time range overlaps 1990-2009
        relevant = [f for f in nc_files if _overlaps_period(f.name)]
        if not relevant:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = iris.load([str(f) for f in relevant])
        cubes.extend(loaded)

    return cubes


def _overlaps_period(filename: str) -> bool:
    """Return True if the file's time range overlaps 1990-2009.

    Parses the trailing date token from standard CMIP6 filenames:
    ``{var}_{mip}_{model}_{exp}_{ensemble}_{grid}_{start}-{end}.nc``
    """
    stem = Path(filename).stem
    date_token = stem.rsplit("_", 1)[-1]
    if "-" not in date_token:
        return True  # can't parse — include to be safe
    start_str, end_str = date_token.split("-", 1)
    try:
        start_year = int(start_str[:4])
        end_year = int(end_str[:4])
    except ValueError:
        return True
    return end_year >= 1990 and start_year <= 2009


def _existing_model_ids(db_path: Path) -> set[str]:
    """Return data_ids already stored in the DuckDB (so we can skip them)."""
    if not db_path.is_file():
        return set()
    try:
        conn = ibis.duckdb.connect(db_path, read_only=True)
        dbs = [d for d in conn.list_databases() if d.lower() not in
               {"main", "memory", "information_schema", "temp", "system", "pg_catalog"}]
        if not dbs:
            return set()
        # Any diagnostic's data_sources table will do
        first_db = dbs[0]
        tables = conn.list_tables(database=first_db)
        if "data_sources" not in tables:
            return set()
        df = conn.table("data_sources", database=first_db).to_pandas()
        conn.disconnect()
        return set(df["id"].unique()) if "id" in df.columns else set()
    except Exception:
        return set()


def _run_model(
    model: dict[str, str],
    *,
    first: bool,
    skip_existing: bool,
    existing_ids: set[str],
) -> bool:
    """Run Tier2_atmosphere_monthly for one model. Returns True on success."""
    institute = model["institute"]
    dataset = model["dataset"]
    grid = model["grid"]

    data_info = DataSourceInformation(
        name=dataset,
        category="CMIP6",
        institute=institute,
        exp="historical",
        variant=_ENSEMBLE,
        references=(
            f"https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.{institute}.{dataset}",
        ),
    )

    if skip_existing and data_info.id in existing_ids:
        logger.info("Skipping {} — already in database", dataset)
        return True

    logger.info("Loading Amon data for {}/{}", institute, dataset)
    cubes = _load_amon(institute, dataset, grid)
    if not cubes:
        logger.warning("No Amon cubes loaded for {} — skipping", dataset)
        return False

    logger.info("Loaded {} cube(s) for {}", len(cubes), dataset)

    suite = Suite(
        "Tier2_atmosphere_monthly",
        diagnostic_kwargs=_DIAG_KWARGS,
        variable_kwargs=_VARIABLE_KWARGS,
    )

    try:
        suite.get_database(
            cubes,
            data_info,
            database_resource=f"duckdb://{DB_PATH}",
            append=(not first),
        )
        logger.info("Written {} → {}", dataset, DB_PATH)
        return True
    except Exception as exc:
        logger.error("Suite failed for {}: {}", dataset, exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N models (useful for testing)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models already present in the output database",
    )
    parser.add_argument(
        "--serve-only",
        action="store_true",
        help="Skip computation and just serve existing databases",
    )
    args = parser.parse_args()

    if not args.serve_only:
        models = _discover_models()
        if args.max_models:
            models = models[: args.max_models]
            logger.info("Limiting to first {} models", args.max_models)

        existing_ids = _existing_model_ids(DB_PATH) if args.skip_existing else set()

        if DB_PATH.is_file() and not args.skip_existing:
            DB_PATH.unlink()
            Path(str(DB_PATH) + ".wal").unlink(missing_ok=True)
            logger.info("Removed existing {}", DB_PATH)

        n_ok = 0
        for i, model in enumerate(models):
            ok = _run_model(
                model,
                first=(i == 0 and not DB_PATH.is_file()),
                skip_existing=args.skip_existing,
                existing_ids=existing_ids,
            )
            if ok:
                n_ok += 1

        logger.info(
            "Finished: {}/{} models written to {}",
            n_ok,
            len(models),
            DB_PATH,
        )

    # Serve all Tier*.ddb files found in the working directory
    db_paths = sorted(Path(".").glob("Tier*.ddb"))
    if not db_paths:
        logger.error("No Tier*.ddb files found — nothing to serve")
        sys.exit(1)

    logger.info("Serving {} tier(s) at http://127.0.0.1:{}", len(db_paths), PORT)
    serve(
        db_paths=db_paths,
        model_name="CMIP6 Historical",
        host="127.0.0.1",
        port=PORT,
        open_browser=False,
    )


if __name__ == "__main__":
    main()
