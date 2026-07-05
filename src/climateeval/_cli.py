"""Command-line interface for ClimateEval.

Installed as the ``climateeval`` console script (see ``[project.scripts]``
in ``pyproject.toml``); also runnable as ``python -m climateeval``.

Subcommands
-----------
``climateeval benchmark MODEL --name NAME``
    Run evaluation suite(s) on a model's CMOR output and compare it against
    the observational references configured in each suite (ERA5, GPCP,
    CERES-EBAF, ...) and the CMIP6 comparison ensemble. Writes one DuckDB
    per suite plus a portable HTML report (and optionally static figures)
    to the output directory.
``climateeval report DB [DB ...]``
    Serve or export a report for existing result databases.
"""

from __future__ import annotations

import argparse
import getpass
import os
import re
import socket
import sys
from pathlib import Path
from typing import Any

try:
    import pwd
except ImportError:  # pragma: no cover - Windows
    pwd = None  # type: ignore[assignment]

from loguru import logger

from climateeval._loader import load_cmor_dir
from climateeval.data import DataSourceInformation
from climateeval.report import (
    generate_portable_html,
    generate_static_plots,
    serve,
)
from climateeval.suites import Suite

DEFAULT_SUITE = "Tier2_atmosphere_monthly"
DEFAULT_TIMERANGE = "19790101/20141231"

# CMOR filename convention: <var>_<table>_<source_id>_<experiment>_...
_CMOR_FNAME_RE = re.compile(r"^[A-Za-z0-9]+_[A-Za-z0-9]+_([A-Za-z0-9-]+)_")


def _infer_model_name(model: Path) -> str | None:
    """Derive the model name (CMOR source_id) from NetCDF file names."""
    files = [model] if model.is_file() else sorted(model.rglob("*.nc"))
    for path in files[:50]:
        match = _CMOR_FNAME_RE.match(path.name)
        if match:
            return match.group(1)
    return None


def _full_user_name() -> str:
    """Return the user's full name (GECOS field), falling back to the login."""
    login = getpass.getuser()
    if pwd is not None:
        full_name = pwd.getpwuid(os.getuid()).pw_gecos.split(",")[0].strip()
        if full_name:
            return full_name
    return login


def _add_benchmark_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "benchmark",
        help="Evaluate a model against observations, reanalyses and CMIP6.",
        description=(
            "Run evaluation suite(s) on CMOR-compliant model output and "
            "write result databases plus an HTML report."
        ),
    )
    parser.add_argument(
        "model",
        type=Path,
        help=(
            "Model output: a NetCDF file, a flat directory of NetCDF files, "
            "or a CMOR DRS tree (e.g. an Amon table directory)."
        ),
    )
    parser.add_argument(
        "--name",
        default=None,
        help=(
            "Model name shown in the report "
            "(default: derived from the CMOR file names)."
        ),
    )
    parser.add_argument(
        "--suite",
        action="append",
        metavar="SUITE",
        help=(
            f"Suite name or YAML path; repeat for several suites "
            f"(default: {DEFAULT_SUITE})."
        ),
    )
    parser.add_argument(
        "--timerange",
        default=DEFAULT_TIMERANGE,
        help=f"ISO time range applied to every dataset (default: {DEFAULT_TIMERANGE}).",
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
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        metavar="DIR",
        help="Output directory (default: <name>_benchmark/).",
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="Also render static PNG figures into <out>/static_figures/.",
    )
    parser.add_argument("--institute", default="", help="Institute (provenance).")
    parser.add_argument("--exp", default="", help="Experiment (provenance).")
    parser.add_argument("--variant", default="", help="Variant label (provenance).")
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
    parser.set_defaults(func=_cmd_benchmark)


def _add_report_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "report",
        help="Serve or export a report for existing result databases.",
        description=(
            "By default a local web server is started. Use --html or "
            "--static-out to export files instead."
        ),
    )
    parser.add_argument("db", type=Path, nargs="+", help="DuckDB .ddb file(s).")
    parser.add_argument("--model", default="", help="Model name in the header.")
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write a self-contained HTML file instead of serving.",
    )
    parser.add_argument(
        "--static-out",
        type=Path,
        default=None,
        metavar="DIR",
        help="Render static figures into DIR instead of serving.",
    )
    parser.add_argument(
        "--static-format",
        default="png",
        metavar="FMT[,FMT...]",
        help="Static image format(s) (default: png).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Static figure DPI (default: 300).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Server port (default: 8765).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    parser.set_defaults(func=_cmd_report)


def _cmd_benchmark(args: argparse.Namespace) -> None:

    if not args.model.exists():
        msg = f"Model path not found: {args.model}"
        raise SystemExit(msg)

    if not args.name:
        args.name = _infer_model_name(args.model) or "model"
        logger.info(f"Model name '{args.name}' derived from the input files")

    cubes = load_cmor_dir(args.model, timerange=args.timerange)
    if not cubes:
        msg = f"No NetCDF data found in {args.model}"
        raise SystemExit(msg)

    info = DataSourceInformation(
        name=args.name,
        category="model",
        institute=args.institute,
        exp=args.exp,
        variant=args.variant,
    )
    diagnostic_kwargs: dict[str, Any] = {
        "fail_on_missing_data": False,
        "fail_on_metric_error": False,
        "download_missing_data": args.download,
    }
    if args.data_root is not None:
        diagnostic_kwargs["data_root_dir"] = args.data_root

    out_dir = args.out or Path(f"{args.name}_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_names = args.suite or [DEFAULT_SUITE]
    user_name = _full_user_name()
    run_info = {
        "Data": str(args.model.resolve()),
        "Time range": args.timerange,
        "Suites": ", ".join(Path(s).stem for s in suite_names),
        "Host": socket.gethostname(),
    }

    db_paths: list[Path] = []
    for suite_name in suite_names:
        db_path = out_dir / f"{Path(suite_name).stem}.ddb"
        db_path.unlink(missing_ok=True)
        Path(str(db_path) + ".wal").unlink(missing_ok=True)
        suite = Suite(
            suite_name,
            diagnostic_kwargs=diagnostic_kwargs,
            variable_kwargs={"timerange": args.timerange},
        )
        logger.info(f"Running suite '{suite_name}' (timerange {args.timerange})")
        suite.get_database(cubes, info, database_resource=f"duckdb://{db_path}")
        logger.info(f"Wrote database {db_path}")
        db_paths.append(db_path)

    if args.serve:
        serve(
            db_paths,
            model_name=args.name,
            user_name=user_name,
            run_info=run_info,
            port=args.port,
        )
        return

    html_path = out_dir / f"{args.name}_report.html"
    html_path.write_text(
        generate_portable_html(
            db_paths,
            offline=True,
            model_name=args.name,
            user_name=user_name,
            run_info=run_info,
        ),
        encoding="utf-8",
    )
    logger.info(f"Wrote report {html_path}")

    if args.static:
        generate_static_plots(db_paths, out_dir / "static_figures")
        logger.info(f"Wrote static figures to {out_dir / 'static_figures'}")


def _cmd_report(args: argparse.Namespace) -> None:

    db_paths = [p.resolve() for p in args.db]
    for path in db_paths:
        if not path.exists():
            msg = f"Database not found: {path}"
            raise SystemExit(msg)

    if args.static_out is not None:
        formats = [f.strip() for f in args.static_format.split(",") if f.strip()]
        generate_static_plots(
            db_paths,
            args.static_out.resolve(),
            formats=formats,
            dpi=args.dpi,
        )
        return

    if args.html is not None:
        args.html.write_text(
            generate_portable_html(
                db_paths,
                offline=True,
                model_name=args.model,
                user_name=_full_user_name(),
            ),
            encoding="utf-8",
        )
        logger.info(f"Wrote report {args.html}")
        return

    serve(
        db_paths,
        model_name=args.model,
        user_name=_full_user_name(),
        port=args.port,
        open_browser=not args.no_browser,
    )


def main(argv: list[str] | None = None) -> None:
    """Run the climateeval command-line interface."""
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

    parser = argparse.ArgumentParser(
        prog="climateeval",
        description=(
            "Evaluate and benchmark Earth System Model output against "
            "observations, reanalyses and CMIP6 models."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_benchmark_parser(subparsers)
    _add_report_parser(subparsers)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
