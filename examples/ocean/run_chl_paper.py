"""Run the manuscript's seven CMIP6 models through ClimateEval.

Modelled on examples/ocean/run_ocean_diagnostics.py, extended from one
model to the seven of

    Navarra et al., "Global and Polar Failures in Physical-Biogeochemical
    Coupling of Chlorophyll-a Across CMIP6 Earth System Models", JGR
    Oceans (submitted).

One database per model, because get_database writes a single
DataSourceInformation per call and the ocean example deletes the file
before writing. The report is built from all of them together.

Usage:
    python run_chl_paper.py                 # all seven
    CHL_MODELS=CESM2,GFDL-ESM4 python run_chl_paper.py   # a subset
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import iris
import numpy as np
from iris.util import equalise_attributes, unify_time_units

from climateeval.data import DataSourceInformation
from climateeval.suites import Suite

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_ROOT = os.environ.get("CLIMATEEVAL_DATA_ROOT", "./climateeval_input")
RESULTS_DIR = Path(os.environ.get("CLIMATEEVAL_RESULTS_DIR", "./results_chl_paper"))

os.environ.setdefault(
    "CLIMATEEVAL_ORAS5_RAW_DIR",
    "/work/bd1083/b382555/extraobsraw/Tier2/ORAS5/data",
)
os.environ.setdefault(
    "CLIMATEEVAL_ORAS5_GRID_DIR",
    "/work/bd1083/b382555/extraobsraw/Tier2/ORAS5/grids",
)

CMIP6_ROOT = Path("/work/ik1017/CMIP6/data/CMIP6/CMIP")

# ---------------------------------------------------------------------------
# The seven models. One configuration per ocean-biogeochemistry family, so
# the ensemble is not weighted toward duplicated formulations: CESM2 stands
# in for the CESM2-FV2/WACCM variants (all sharing MARBL), MPI-ESM-1-2-HAM
# for the MPI-ESM1-2-LR/HR variants (all sharing HAMOCC6).
# ---------------------------------------------------------------------------
MODELS = {
    "ACCESS-ESM1-5":   {"institute": "CSIRO",             "variant": "r1i1p1f1", "grid": "gn", "version": "v20191115"},
    "CanESM5":         {"institute": "CCCma",             "variant": "r1i1p1f1", "grid": "gn", "version": "v20190429"},
    "CESM2":           {"institute": "NCAR",              "variant": "r1i1p1f1", "grid": "gn", "version": "v20190308"},
    "GFDL-ESM4":       {"institute": "NOAA-GFDL",         "variant": "r1i1p1f1", "grid": "gr", "version": "v20190726"},
    "IPSL-CM6A-LR":    {"institute": "IPSL",              "variant": "r1i1p1f1", "grid": "gn", "version": "v20180803"},
    "MPI-ESM-1-2-HAM": {"institute": "HAMMOZ-Consortium", "variant": "r1i1p1f1", "grid": "gn", "version": "v20190627"},
    "NorESM2-LM":      {"institute": "NCC",               "variant": "r1i1p1f1", "grid": "gn", "version": "v20190815"},
}

# ---------------------------------------------------------------------------
# What to run per model: (suite, [(variable, MIP)], timerange, db suffix).
#
# Split by suite rather than run as one, following the ocean example: the
# sea-ice references and the ocean references cover different periods, and
# one timerange is applied to EVERY dataset in a run, model and reference
# alike, so a reference not covering the requested start year is dropped.
#
# rsds is absent deliberately. It is a model-SELECTION criterion in the
# manuscript - the seven models are those archiving it alongside the other
# four - but no figure plots it.
# ---------------------------------------------------------------------------
# `chl`, not `chlos`. The SUITE declares the variable ids it wants, and
# Tier2_ocean_monthly asks for chl; feeding it chlos produces "Got 0
# cubes for constraint NameConstraint(var_name='chl')". The suite's own
# preprocessing takes the surface level.
#
# The earlier "Unable to allocate 124 MiB" was not really about chl
# being three-dimensional: 124 MiB is trivial on an exclusive node, so
# the job was running under a cgroup memory cap. --mem=0 in the submit
# script is the fix.
CHL_SURFACE_ONLY = ()

# Tier2_ocean, not Tier2_ocean_monthly.
#
# Tier2_ocean_monthly is the generic ocean benchmark: annual cycle, map,
# zonal line, AMOC, OHC, Nino3.4, and mixed-layer depth in the Labrador
# and Weddell Seas. None of that is in the manuscript, and its ORAS5
# regional MLD variables failed on coordinate problems while consuming
# most of a 50-minute run.
#
# Tier2_ocean is the suite carrying the Chl-a biogeochemistry diagnostic
# (climateeval.diags.simple._chl_bgc), which is what the manuscript
# figures come from. The original example script named it and it was
# right; pointing at the monthly suite was the error.
RUNS = (
    (
        "Tier2_ocean",
        (("chl", "Omon"), ("tos", "Omon"), ("mlotst", "Omon"),
         ("siconc", "SImon")),
        "19970101/20141231",
        "chl_bgc",
    ),
    # Kept as a separate run only until Tier2_ocean.yml confirms whether
    # it already consumes siconc. If it does, this is redundant and
    # should be removed rather than run twice.
    (
        "Tier2_seaice_monthly",
        (("siconc", "SImon"),),
        "19970101/20141231",
        "seaice",
    ),
)


def model_files(model: str, cfg: dict, var: str, mip: str) -> list[str]:
    """Return the files for one variable of one model, or an empty list.

    Unlike the single-model example this does NOT raise when a variable is
    missing. Only three of the seven models archive mlotst, so a hard
    failure here would stop the run on the fourth model rather than
    producing the six databases that are obtainable.
    """
    root = CMIP6_ROOT / cfg["institute"] / model / "historical" / cfg["variant"]
    root = root / mip / var

    # The grid label is per VARIABLE, not per model. NorESM2-LM archives
    # chl on the regular grid (gr) while its siconc, tos and mlotst are
    # native (gn), so a single label in MODELS silently loses one
    # variable - which is exactly how that model ended up with no chl
    # and no metrics. Try the model's declared grid first, then the
    # others, then whatever directories exist.
    candidates = [cfg["grid"], "gn", "gr", "gr1"]
    seen = set()
    grids = [g for g in candidates if not (g in seen or seen.add(g))]
    if root.is_dir():
        grids += sorted(d.name for d in root.iterdir()
                        if d.is_dir() and d.name not in seen)

    for grid in grids:
        base = root / grid
        if not base.is_dir():
            continue
        files = sorted(str(p) for p in base.glob(
            f"{cfg['version']}/{var}_{mip}_*.nc"))
        if not files:
            files = sorted(str(p) for p in base.glob(f"*/{var}_{mip}_*.nc"))
            if files:
                print(f"  [{model}] {var}: version {cfg['version']} absent, "
                      f"using {Path(files[0]).parent.name}")
        if files:
            if grid != cfg["grid"]:
                print(f"  [{model}] {var}: on grid '{grid}', "
                      f"not the model default '{cfg['grid']}'")
            return files

    print(f"  [{model}] {var} ({mip}): NOT FOUND under {root} "
          f"(tried grids {grids})")
    return []


def surface_chl(cubes: iris.cube.CubeList) -> iris.cube.CubeList:
    """Reduce a 3D `chl` cube to its surface level and relabel it `chlos`.

    Only used for models that do not archive chlos. The shallowest level
    is taken rather than interpolating to a fixed depth: the models have
    different vertical grids, and the surface layer is what chlos itself
    reports.
    """
    out = iris.cube.CubeList()
    for cube in cubes:
        if cube.var_name != "chl":
            out.append(cube)
            continue
        vertical = None
        for name in ("depth", "olevel", "lev", "model_level_number"):
            try:
                vertical = cube.coord(name)
                break
            except Exception:
                continue
        if vertical is None or vertical.shape[0] == 1:
            print("    chl has no usable vertical coordinate; left as is")
            out.append(cube)
            continue
        top = int(np.argmin(vertical.points))
        surface = cube.subset(vertical[top]) if hasattr(cube, "subset") else None
        if surface is None:
            index = [slice(None)] * cube.ndim
            index[cube.coord_dims(vertical)[0]] = top
            surface = cube[tuple(index)]
        surface.var_name = "chlos"
        surface.long_name = "Surface Mass Concentration of Total Phytoplankton Chlorophyll"
        print(f"    chl -> chlos: took level {top} "
              f"({vertical.points[top]:.1f} {vertical.units}) "
              f"of {vertical.shape[0]}")
        out.append(surface)
    return out


def run_one(model: str, cfg: dict, spec: tuple) -> Path | None:
    """Run one suite for one model and write its database."""
    suite_name, variables, timerange, tag = spec

    paths, missing = [], []
    needs_surface = False
    for var, mip in variables:
        found = model_files(model, cfg, var, mip)
        if not found and var == "chlos" and model in CHL_SURFACE_ONLY:
            found = model_files(model, cfg, "chl", mip)
            if found:
                needs_surface = True
                print(f"  [{model}] chlos absent; using chl and taking "
                      f"the surface level")
        paths.extend(found)
        if not found:
            missing.append(var)
    if not paths:
        print(f"  [{model}] {tag}: no data at all, skipping")
        return None
    if missing:
        print(f"  [{model}] {tag}: proceeding without {missing}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cubes = iris.load(paths)
        # CMIP6 output is split into per-decade files: concatenate the
        # time chunks back into one cube per variable. The per-file global
        # attributes and time units must be reconciled first, otherwise
        # the chunks stay separate and single-cube extraction rejects them.
        equalise_attributes(cubes)
        unify_time_units(cubes)
        cubes = cubes.concatenate()
        if needs_surface:
            cubes = surface_chl(cubes)

    info = DataSourceInformation(
        name=model,
        category="CMIP6",
        institute=cfg["institute"],
        exp="historical",
        variant=cfg["variant"],
    )

    db_path = RESULTS_DIR / f"{tag}_{model}.ddb"
    if db_path.is_file():
        db_path.unlink()

    Suite(
        suite_name,
        diagnostic_kwargs={
            "fail_on_missing_data": False,
            "download_missing_data": False,
            "data_root_dir": DATA_ROOT,
        },
        variable_kwargs={"timerange": timerange},
    ).get_database(cubes, info, database_resource=f"duckdb://{db_path}")

    print(f"  WROTE {db_path}")
    return db_path


def build_report(db_paths: list[Path]) -> None:
    """Build the HTML report if the (optional) report module is available."""
    try:
        from climateeval.report import generate_portable_html  # noqa: PLC0415
    except ImportError:
        print(
            "climateeval.report is not installed (it lives on the "
            "'kd/ocean-diag-report' branch); skipping the HTML report. "
            f"Result databases are in {RESULTS_DIR}.",
        )
        return
    html = generate_portable_html(db_paths, offline=True,
                                  model_name="CMIP6 ensemble")
    report = RESULTS_DIR / "report.html"
    report.write_text(html)
    print(f"WROTE {report}  (open in a browser)")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    wanted = os.environ.get("CHL_MODELS")
    models = ([m.strip() for m in wanted.split(",")] if wanted
              else list(MODELS))
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}")

    print(f"Models : {models}")
    print(f"Runs   : {[s[0] + '/' + s[3] for s in RUNS]}")
    print(f"Results: {RESULTS_DIR}")

    db_paths, failed = [], []
    for model in models:
        print(f"\n=== {model} ===")
        for spec in RUNS:
            try:
                path = run_one(model, MODELS[model], spec)
                if path is not None:
                    db_paths.append(path)
            except Exception as exc:
                # One model failing must not take the other six with it.
                print(f"  [{model}] {spec[3]} FAILED: {exc}")
                failed.append((model, spec[3], str(exc)))

    print(f"\n{len(db_paths)} database(s) written")
    if failed:
        print(f"{len(failed)} run(s) failed:")
        for model, tag, exc in failed:
            print(f"  {model:18s} {tag:8s} {exc[:70]}")

    if db_paths:
        build_report(db_paths)


if __name__ == "__main__":
    main()
