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
``climateeval check MODEL``
    Validate model output against a suite's input requirements without
    running any diagnostics.
``climateeval report DB [DB ...]``
    Serve or export a report for existing result databases.
``climateeval suites list``
    List the shipped evaluation suites.
"""

from __future__ import annotations

import argparse
import getpass
import json
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

import yaml
from loguru import logger

# Configure logging before the climateeval imports below: package modules log
# at import time, and loguru's default sink would print that as DEBUG noise.
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

from climateeval import suites as _suites_pkg  # noqa: E402
from climateeval._check import check_model_data  # noqa: E402
from climateeval._loader import load_cmor_dir  # noqa: E402
from climateeval.data import DataSourceInformation  # noqa: E402
from climateeval.report import (  # noqa: E402
    generate_portable_html,
    generate_static_plots,
    serve,
)
from climateeval.suites import Suite  # noqa: E402

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


def _add_check_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "check",
        help="Check that model output meets a suite's input requirements.",
        description=(
            "Validate model output against the input requirements of the "
            "selected evaluation suites without running any diagnostics. "
            "Only metadata is read (fast, no reference data needed) and a "
            "per-variable PASS/FAIL report is printed. Exit status 0 means "
            "every required variable is usable by `climateeval benchmark`; "
            "the requirements are documented in docs/input-data-spec.md."
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
        help=(
            f"ISO time range the benchmark will use; the data must cover it "
            f"(default: {DEFAULT_TIMERANGE})."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON report instead of the table.",
    )
    parser.set_defaults(func=_cmd_check)


def _cmd_check(args: argparse.Namespace) -> None:
    if not args.model.exists():
        msg = f"Model path not found: {args.model}"
        raise SystemExit(msg)

    suite_names = args.suite or [DEFAULT_SUITE]
    report = check_model_data(
        args.model,
        suites=suite_names,
        timerange=args.timerange,
    )
    if report.n_cubes == 0:
        msg = f"No NetCDF data found in {args.model}"
        raise SystemExit(msg)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))  # noqa: T201
    else:
        _print_check_report(report)

    if not report.passed:
        raise SystemExit(1)


def _print_check_report(report: Any) -> None:
    """Print a human-readable check report."""
    print(  # noqa: T201
        f"Checked {report.model_path} against {', '.join(report.suites)} "
        f"(timerange {report.timerange})",
    )
    print(f"Loaded {report.n_cubes} cube(s)")  # noqa: T201
    print()  # noqa: T201

    counts = {"pass": 0, "warn": 0, "fail": 0}
    name_w = max(len(v.var_name) for v in report.variables)
    for variable in report.variables:
        counts[variable.status] += 1
        details = "; ".join(
            f"{check.name}: {check.message}"
            if check.status != "pass"
            else check.message
            for check in variable.checks
            if check.message or check.status != "pass"
        )
        label = variable.long_name or ""
        text = " — ".join(part for part in (label, details) if part)
        status = variable.status.upper()
        print(f"  {status:<4}  {variable.var_name:<{name_w}}  {text}")  # noqa: T201

    print()  # noqa: T201
    if report.extra_var_names:
        print(  # noqa: T201
            f"Extra cubes not used by the selected suites: "
            f"{', '.join(report.extra_var_names)}",
        )
    print(  # noqa: T201
        f"Summary: {counts['pass']} pass, {counts['warn']} warning(s), "
        f"{counts['fail']} fail",
    )
    print("Input data requirements: docs/input-data-spec.md")  # noqa: T201


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


def _add_suites_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "suites",
        help="Inspect the shipped evaluation suites.",
        description=(
            "List the evaluation suites shipped with ClimateEval. Pass a "
            "suite name (or your own YAML path) to `climateeval benchmark "
            "--suite`."
        ),
    )
    sub = parser.add_subparsers(dest="suites_command")
    list_parser = sub.add_parser("list", help="List the shipped suites.")
    list_parser.set_defaults(func=_cmd_suites_list)
    # bare `climateeval suites` behaves like `suites list`
    parser.set_defaults(func=_cmd_suites_list)


def _cmd_suites_list(args: argparse.Namespace) -> None:  # noqa: ARG001
    suites_dir = Path(_suites_pkg.__file__).parent
    rows: list[tuple[str, int, int]] = []
    for path in sorted(suites_dir.glob("*.yml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        diagnostics = spec if isinstance(spec, list) else []
        var_ids = {
            str(v.get("id", v.get("var_name", "")))
            for d in diagnostics
            for v in d.get("variables", [])
        }
        rows.append((path.stem, len(diagnostics), len(var_ids)))

    name_w = max(len(name) for name, *_ in rows)
    print("Shipped suites (use with `climateeval benchmark --suite NAME`):")  # noqa: T201
    print()  # noqa: T201
    print(f"  {'NAME':<{name_w}}  DIAGNOSTICS  VARIABLES")  # noqa: T201
    for name, n_diags, n_vars in rows:
        print(f"  {name:<{name_w}}  {n_diags:>11}  {n_vars:>9}")  # noqa: T201
    print()  # noqa: T201
    print("Suite format and contents: docs/suites.md")  # noqa: T201


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
    parser = argparse.ArgumentParser(
        prog="climateeval",
        description=(
            "Evaluate and benchmark Earth System Model output against "
            "observations, reanalyses and CMIP6 models."
        ),
        epilog=(
            "examples:\n"
            "  climateeval benchmark /path/to/model/cmor/Amon\n"
            "      benchmark a model against the default Tier 2 atmosphere\n"
            "      suite (ERA5, GPCP, CERES-EBAF references + CMIP6 models)\n"
            "  climateeval check /path/to/model/cmor/Amon\n"
            "      verify converted model output is usable before benchmarking\n"
            "  climateeval benchmark model/Amon --suite my_suite.yml --serve\n"
            "      run a custom suite and open the interactive report\n"
            "  climateeval report results/*.ddb --html report.html\n"
            "      export a self-contained HTML report for existing results\n"
            "  climateeval suites list\n"
            "      list the shipped evaluation suites\n"
            "\n"
            "documentation: docs/usage.md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")
    _add_benchmark_parser(subparsers)
    _add_check_parser(subparsers)
    _add_report_parser(subparsers)
    _add_suites_parser(subparsers)

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stderr)
        raise SystemExit(2)
    args.func(args)


if __name__ == "__main__":
    main()
