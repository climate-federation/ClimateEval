#!/usr/bin/env python
r"""Quickly benchmark a model against observations and CMIP6, and build a report.

Point this at a directory of CMOR-compliant NetCDF files from your model. It runs
an evaluation suite (by default ``Tier2_atmosphere_monthly``, which compares
against ERA5 / GPCP / CERES-EBAF references *and* CMIP6 historical models) and
writes a self-contained interactive HTML report.

Examples
--------
Compare your model against ERA5 + CMIP6 and write ``MyModel_benchmark.html``::

    python benchmark.py /path/to/model/cmor/Amon --name MyModel

Use staged references and open an interactive server instead of a file::

    python benchmark.py /path/to/model/cmor/Amon --name MyModel \\
        --data-root ~/climateeval_input --serve

Download any missing references automatically (ERA5 needs ``~/.cdsapirc``)::

    python benchmark.py /path/to/model/cmor/Amon --name MyModel --download

See ``docs/usage.md`` and ``docs/reporting.md`` for details.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import iris
from iris.util import equalise_attributes, unify_time_units
from loguru import logger

from climateeval.data import DataSourceInformation
from climateeval.report import generate_portable_html, serve
from climateeval.suites import Suite

logger = logger.opt(colors=True)

DEFAULT_SUITE = "Tier2_atmosphere_monthly"
DEFAULT_TIMERANGE = "19790101/20141231"


def _find_files(model: Path) -> list[Path]:
    """Return the NetCDF files for a directory (glob ``*.nc``) or a single file."""
    if model.is_dir():
        return sorted(model.glob("*.nc"))
    return [model]


def _load_cubes(paths: list[Path]) -> iris.cube.CubeList:
    """Load and concatenate the model's CMORized NetCDF files."""
    logger.info(f"Loading {len(paths)} file(s)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cubes = iris.load([str(p) for p in paths])
        equalise_attributes(cubes)
        unify_time_units(cubes)
        cubes = cubes.concatenate()
    logger.info(f"Loaded {len(cubes)} cube(s)")
    return cubes


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a model against observational references (ERA5, GPCP, "
            "CERES-EBAF, ...) and CMIP6 historical models, and build an "
            "interactive HTML report."
        ),
    )
    parser.add_argument(
        "model",
        type=Path,
        help="Directory of the model's CMOR-compliant NetCDF files (or one file).",
    )
    parser.add_argument(
        "--name",
        default="model",
        help="Model name shown in the report (default: model).",
    )
    parser.add_argument(
        "--suite",
        default=DEFAULT_SUITE,
        help=f"Suite name or YAML path (default: {DEFAULT_SUITE}).",
    )
    parser.add_argument(
        "--timerange",
        default=DEFAULT_TIMERANGE,
        help=(
            f"ISO time range applied to every dataset (default: {DEFAULT_TIMERANGE})."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory with staged reference datasets.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download missing references (ERA5 needs ~/.cdsapirc).",
    )
    parser.add_argument("--institute", default="", help="Institute (provenance).")
    parser.add_argument("--exp", default="", help="Experiment (provenance).")
    parser.add_argument("--variant", default="", help="Variant label (provenance).")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="HTML report path (default: <name>_benchmark.html).",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start an interactive server instead of writing an HTML file.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for --serve (default: 8765).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the benchmark and produce the report."""
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
    args = _parse_args()

    files = _find_files(args.model)
    if not files:
        logger.error(f"No NetCDF (*.nc) files found in {args.model}")
        raise SystemExit(1)
    cubes = _load_cubes(files)

    info = DataSourceInformation(
        name=args.name,
        category="model",
        institute=args.institute,
        exp=args.exp,
        variant=args.variant,
    )

    diagnostic_kwargs: dict[str, Any] = {
        "fail_on_missing_data": False,
        "download_missing_data": args.download,
    }
    if args.data_root is not None:
        diagnostic_kwargs["data_root_dir"] = args.data_root

    db_path = Path(f"{args.suite}.ddb")
    if db_path.exists():
        db_path.unlink()
    suite = Suite(
        args.suite,
        diagnostic_kwargs=diagnostic_kwargs,
        variable_kwargs={"timerange": args.timerange},
    )
    logger.info(f"Running suite '{args.suite}' (timerange {args.timerange})")
    suite.get_database(cubes, info, database_resource=f"duckdb://{db_path}")
    logger.info(f"Wrote database {db_path}")

    if args.serve:
        serve([db_path], model_name=args.name, port=args.port)
        return

    output = args.output or Path(f"{args.name}_benchmark.html")
    output.write_text(
        generate_portable_html([db_path], offline=True, model_name=args.name),
    )
    logger.info(f"Wrote report {output}")


if __name__ == "__main__":
    main()
