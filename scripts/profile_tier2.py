"""Profile where the wall time goes in a single tier-suite run (task D1).

Monkey-patches timing wrappers around the hot paths of ``SimpleDiagnostic``
(no core changes) and runs one model through a suite with an in-memory
database. Produces a per-phase breakdown:

    diagnostic x {references, to_benchmark, other, combine} and
    step totals for {load (DataSource.get_cube / get_prepared_cube),
    preprocess (_safely_preprocess), metrics (_get_metrics),
    to_table (_get_raw_output_table)}

Caveat: load timing wraps the base ``DataSource.get_cube``; the few ocean
sources that override it (ORAS5, RAPID, OSI450, WOA) are not step-timed —
irrelevant for the atmosphere suites.

Usage
-----
  # full run (expect ~1 h for Tier2_atmosphere_monthly)
  python profile_tier2.py --model MPI-M/MPI-ESM1-2-LR

  # quick look: 3 ensemble members, two diagnostics
  python profile_tier2.py --max-other 3 --diags map,annual_cycle
"""

from __future__ import annotations

import argparse
import functools
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

import climateeval._utils
import climateeval.diags.simple._base as simple_base
from climateeval.data import DataSourceInformation
from climateeval.data._base import DataSource
from climateeval.diags.simple import SimpleDiagnostic
from climateeval.suites import Suite

# Reuse the model loader from the sibling driver script (sys.path[0] is scripts/).
from run_tier_benchmark_model import _load_amon, logger as _  # noqa: F401

RECORDS: list[dict[str, Any]] = []
CURRENT = {"diag": "-", "phase": "-"}


def _record(step: str, detail: str, seconds: float) -> None:
    RECORDS.append(
        {
            "diag": CURRENT["diag"],
            "phase": CURRENT["phase"],
            "step": step,
            "detail": detail,
            "seconds": seconds,
        }
    )


def _timed(step: str, detail_fn):  # noqa: ANN001, ANN201
    """Wrap a callable; record wall time under the current diag/phase."""

    def deco(orig):  # noqa: ANN001, ANN202
        @functools.wraps(orig)
        def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            try:
                detail = detail_fn(*args, **kwargs)
            except Exception:  # noqa: BLE001
                detail = "?"
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                _record(step, detail, time.perf_counter() - t0)

        return wrapper

    return deco


def _phased(phase: str):  # noqa: ANN201
    """Wrap a phase method; set the current phase for nested step records."""

    def deco(orig):  # noqa: ANN001, ANN202
        @functools.wraps(orig)
        def wrapper(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
            prev = CURRENT["phase"]
            CURRENT["phase"] = phase
            t0 = time.perf_counter()
            try:
                return orig(self, *args, **kwargs)
            finally:
                CURRENT["phase"] = prev
                _record("phase_total", phase, time.perf_counter() - t0)

        return wrapper

    return deco


def _install_patches(max_other: int | None) -> None:
    # Per-diagnostic total + current-diag context (+ optional ensemble trim)
    orig_get_output = SimpleDiagnostic.get_output

    @functools.wraps(orig_get_output)
    def get_output(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        CURRENT["diag"] = self.name
        if max_other is not None:
            self._other_data = {
                var: srcs[:max_other] for var, srcs in self._other_data.items()
            }
        t0 = time.perf_counter()
        try:
            return orig_get_output(self, *args, **kwargs)
        finally:
            _record("diag_total", self.name, time.perf_counter() - t0)
            CURRENT["diag"] = "-"

    SimpleDiagnostic.get_output = get_output

    SimpleDiagnostic._get_reference_cubes = _phased("references")(
        SimpleDiagnostic._get_reference_cubes
    )
    SimpleDiagnostic._get_output_of_reference_data = _phased("references_tables")(
        SimpleDiagnostic._get_output_of_reference_data
    )
    SimpleDiagnostic._get_output_of_to_benchmark_data = _phased("to_benchmark")(
        SimpleDiagnostic._get_output_of_to_benchmark_data
    )
    SimpleDiagnostic._get_output_of_other_data = _phased("other")(
        SimpleDiagnostic._get_output_of_other_data
    )

    DataSource.get_cube = _timed(
        "load", lambda self, root, variable, **kw: f"{self.id}/{variable.id}"
    )(DataSource.get_cube)
    SimpleDiagnostic._safely_preprocess = _timed(
        "preprocess", lambda self, cube, variable: variable.id
    )(SimpleDiagnostic._safely_preprocess)
    SimpleDiagnostic._get_metrics = _timed(
        "metrics", lambda self, cube, ref: str(cube.var_name)
    )(SimpleDiagnostic._get_metrics)
    SimpleDiagnostic._get_raw_output_table = _timed(
        "to_table", lambda self, cube, *, variable, data_id, data_type: f"{data_id}/{variable.id}"
    )(SimpleDiagnostic._get_raw_output_table)

    # get_prepared_cube is imported by name into the diagnostics module
    timed_gpc = _timed(
        "load", lambda data, variable, *a, **k: f"to_benchmark/{variable.id}"
    )(climateeval._utils.get_prepared_cube)
    climateeval._utils.get_prepared_cube = timed_gpc
    simple_base.get_prepared_cube = timed_gpc


def _report(csv_path: Path, total: float) -> None:
    df = pd.DataFrame(RECORDS)
    df.to_csv(csv_path, index=False)

    steps = df[~df["step"].isin(["phase_total", "diag_total"])]
    print(f"\n=== Total suite wall time: {total:,.1f} s ===")

    print("\n--- Per diagnostic (s) ---")
    diag_totals = (
        df[df["step"] == "diag_total"].groupby("diag")["seconds"].sum().sort_values(ascending=False)
    )
    print(diag_totals.to_string(float_format="%.1f"))

    print("\n--- Diagnostic × phase (s) ---")
    phases = df[df["step"] == "phase_total"]
    pivot = phases.pivot_table(
        index="diag", columns="detail", values="seconds", aggfunc="sum", fill_value=0.0
    )
    # residual = table union/combine work outside the three phases
    pivot["combine(resid)"] = diag_totals.reindex(pivot.index).fillna(0) - pivot.sum(axis=1)
    print(pivot.to_string(float_format="%.1f"))

    print("\n--- Step totals across run (s) ---")
    step_totals = steps.groupby(["phase", "step"])["seconds"].agg(["sum", "count", "mean"])
    print(step_totals.sort_values("sum", ascending=False).to_string(float_format="%.2f"))

    print("\n--- Top 15 single steps (s) ---")
    top = steps.nlargest(15, "seconds")[["diag", "phase", "step", "detail", "seconds"]]
    print(top.to_string(index=False, float_format="%.1f"))
    print(f"\nFull records: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="MPI-M/MPI-ESM1-2-LR")
    parser.add_argument("--suite", default="Tier2_atmosphere_monthly")
    parser.add_argument("--timerange", default="19900101/20091231")
    parser.add_argument(
        "--max-other", type=int, default=None,
        help="Trim the 'other' (ensemble) sources per variable to N (quick runs)",
    )
    parser.add_argument(
        "--diags", default=None,
        help="Comma-separated diagnostic names to keep (default: all)",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("/work/bd1179/b309141/climateeval_input")
    )
    parser.add_argument("--csv", type=Path, default=Path("profile_tier2_timings.csv"))
    args = parser.parse_args()

    institute, dataset = args.model.split("/", 1)
    start, end = (int(t[:4]) for t in args.timerange.split("/", 1))

    logger.info("Loading Amon data for {}/{}", institute, dataset)
    cubes = _load_amon(institute, dataset, start, end)
    if not cubes:
        msg = f"No Amon cubes for {dataset}"
        raise SystemExit(msg)
    logger.info("Loaded {} cube(s)", len(cubes))

    info = DataSourceInformation(
        name=dataset, category="CMIP6", institute=institute,
        exp="historical", variant="r1i1p1f1",
    )
    suite = Suite(
        args.suite,
        diagnostic_kwargs={
            "data_root_dir": args.data_root,
            "fail_on_missing_data": False,
            "fail_on_metric_error": False,
            "download_missing_data": False,
        },
        variable_kwargs={"timerange": args.timerange},
    )
    if args.diags:
        wanted = {d.strip() for d in args.diags.split(",")}
        suite._suite_definition = [  # noqa: SLF001
            d for d in suite._suite_definition if d["name"] in wanted  # noqa: SLF001
        ]
        if not suite._suite_definition:  # noqa: SLF001
            msg = f"No diagnostics left after --diags={args.diags}"
            raise SystemExit(msg)

    _install_patches(args.max_other)

    logger.info("Profiling {} for {} (in-memory db)", args.suite, dataset)
    t0 = time.perf_counter()
    suite.get_database(cubes, info, database_resource="duckdb://")
    total = time.perf_counter() - t0

    _report(args.csv, total)


if __name__ == "__main__":
    main()
