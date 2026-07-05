"""Run tier suite(s) for ONE model into its own DuckDB (parallel-safe).

One invocation = one model = one scratch database per suite, named
``<out-dir>/<suite>__<dataset>.ddb``. Designed for SLURM array jobs: each
array task processes one line of the models file; afterwards
``merge_tier_dbs.py`` unions the per-model files into one combined database
per suite (see ``submit_tier_benchmark.sh``).

Usage
-----
  # by array index (0-based) into a models file with "INSTITUTE/DATASET" lines
  python run_tier_benchmark_model.py --models-file models.txt --model-index 0

  # or a single model directly
  python run_tier_benchmark_model.py --model MPI-M/MPI-ESM1-2-LR \
      --suites Tier1_sanity_checks,Tier1_consistency_checks,Tier2_atmosphere_monthly
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

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

_CMIP6_ROOT = Path("/work/ik1017/CMIP6/data/CMIP6/CMIP")
_ENSEMBLE = "r1i1p1f1"


def _overlaps(filename: str, start: int, end: int) -> bool:
    date_token = Path(filename).stem.rsplit("_", 1)[-1]
    if "-" not in date_token:
        return True
    s, e = date_token.split("-", 1)
    try:
        return int(e[:4]) >= start and int(s[:4]) <= end
    except ValueError:
        return True


def _load_amon(institute: str, dataset: str, start: int, end: int) -> CubeList:
    """Load all Amon files (native grid preferred, latest version)."""
    amon_root = _CMIP6_ROOT / institute / dataset / "historical" / _ENSEMBLE / "Amon"
    cubes: CubeList = CubeList([])
    if not amon_root.is_dir():
        return cubes
    for var_dir in sorted(amon_root.iterdir()):
        grid_dirs = sorted(d for d in var_dir.iterdir() if d.is_dir())
        if not grid_dirs:
            continue
        grid_dir = next((d for d in grid_dirs if d.name == "gn"), grid_dirs[0])
        versions = sorted(grid_dir.iterdir())
        if not versions:
            continue
        relevant = [
            f
            for f in sorted(versions[-1].glob("*.nc"))
            if _overlaps(f.name, start, end)
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


def _read_models_file(path: Path) -> list[str]:
    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="INSTITUTE/DATASET, e.g. MPI-M/MPI-ESM1-2-LR")
    group.add_argument(
        "--model-index", type=int, help="0-based line index into --models-file"
    )
    parser.add_argument(
        "--models-file", type=Path, help="File with one INSTITUTE/DATASET per line"
    )
    parser.add_argument(
        "--suites",
        default="Tier2_atmosphere_monthly",
        help="Comma-separated suite names (default: Tier2_atmosphere_monthly)",
    )
    parser.add_argument("--timerange", default="19900101/20091231")
    parser.add_argument("--out-dir", type=Path, default=Path("output/parallel"))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/work/bd1179/b309141/climateeval_input"),
        help="Directory with staged reference datasets",
    )
    args = parser.parse_args()

    if args.model_index is not None:
        if args.models_file is None:
            parser.error("--model-index requires --models-file")
        models = _read_models_file(args.models_file)
        if not 0 <= args.model_index < len(models):
            parser.error(f"--model-index out of range (0..{len(models) - 1})")
        model = models[args.model_index]
    else:
        model = args.model
    try:
        institute, dataset = model.split("/", 1)
    except ValueError:
        parser.error(f"Model must be INSTITUTE/DATASET, got: {model}")

    start, end = (int(t[:4]) for t in args.timerange.split("/", 1))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading Amon data for {}/{} ({}–{})", institute, dataset, start, end)
    cubes = _load_amon(institute, dataset, start, end)
    if not cubes:
        logger.error("No Amon cubes for {} — aborting", dataset)
        raise SystemExit(1)
    logger.info("Loaded {} cube(s)", len(cubes))

    info = DataSourceInformation(
        name=dataset,
        category="CMIP6",
        institute=institute,
        exp="historical",
        variant=_ENSEMBLE,
        references=(
            f"https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.{institute}.{dataset}",
        ),
    )
    diag_kwargs = {
        "data_root_dir": args.data_root,
        "fail_on_missing_data": False,
        "fail_on_metric_error": False,
        "download_missing_data": False,
    }

    for suite_name in [s.strip() for s in args.suites.split(",") if s.strip()]:
        db_path = args.out_dir / f"{suite_name}__{dataset}.ddb"
        db_path.unlink(missing_ok=True)
        Path(str(db_path) + ".wal").unlink(missing_ok=True)
        suite = Suite(
            suite_name,
            diagnostic_kwargs=diag_kwargs,
            variable_kwargs={"timerange": args.timerange},
        )
        logger.info("Running {} for {} → {}", suite_name, dataset, db_path)
        suite.get_database(cubes, info, database_resource=f"duckdb://{db_path}")
        logger.info("Written {}", db_path)


if __name__ == "__main__":
    main()
