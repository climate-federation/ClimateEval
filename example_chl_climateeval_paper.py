"""
Run ChlBGCEvaluation through the ClimateEval pipeline, restricted to the
figures that appear in the manuscript.

Reproduces Figures 1-9 of
    Navarra et al., "Global and Polar Failures in Physical-Biogeochemical
    Coupling of Chlorophyll-a Across CMIP6 Earth System Models
    Investigated Through Correlation Analysis", JGR Oceans (submitted).

Figure 10 (surface dissolved-iron bias against GEOTRACES) is not
produced here. It compares against point observations with no gridded
counterpart, which the cube-based pipeline cannot express, so it stays a
standalone script (so_fe_bias.py) reading the CMIP6 dfe fields directly.
The same script produces supplementary Fig. S3.

Produces:
  - Tier2_ocean_paper.ddb     — DuckDB database for the HTML report
  - paper_figures/*.png       — the nine pipeline figures

Usage:
    python example_chl_climateeval_paper.py
    python generate_html.py Tier2_ocean_paper.ddb -o Tier2_ocean_paper.html
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import yaml

import iris
from iris.util import equalise_attributes, unify_time_units
from loguru import logger

logger = logger.opt(colors=True)
logger.remove()
logger.add(
    sys.stdout, level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    colorize=True,
)

# ── Configuration ────────────────────────────────────────────────────────────
# Everything that describes WHAT to run lives in the YAML; this file holds
# only the machinery. Keeping them apart means a change of model list,
# analysis window or figure set is a config edit reviewable on its own,
# and the config can be archived alongside the manuscript as the record
# of what produced the figures.
CONFIG_PATH = Path(os.environ.get("CHL_RECIPE", "recipe_chl_paper.yml"))


def load_config(path: Path) -> dict:
    """Read the recipe and derive what the loader needs from it."""
    if not path.is_file():
        raise FileNotFoundError(
            f"recipe not found: {path}. Set CHL_RECIPE to point at it.")
    with open(path) as handle:
        cfg = yaml.safe_load(handle)

    # Variables are DERIVED from the figure list, not declared
    # separately: a variable no figure requires is never read, and
    # adding a figure automatically pulls in what it needs. rsds is
    # absent for exactly this reason - it selects the models but appears
    # in no figure.
    needed = sorted({v for f in cfg["figures"].values() for v in f["vars"]})
    unknown = [v for v in needed if v not in cfg["variables"]]
    if unknown:
        raise ValueError(f"figures require undeclared variables: {unknown}")
    cfg["_variables"] = [
        (v, cfg["variables"][v]["mip"], cfg["variables"][v].get("grid"))
        for v in needed
    ]

    # Fail here rather than three hours into a run: a figure naming a
    # reference that does not exist would otherwise be discovered only
    # when the diagnostic tried to load it.
    missing = sorted({
        obs for spec in cfg["figures"].values() for obs in spec["obs"]
        if obs not in cfg["references"]
    })
    if missing:
        raise ValueError(f"figures reference undeclared datasets: {missing}")

    mask = cfg.get("masking", {})
    if mask.get("mask_models_to_obs"):
        ref = mask.get("reference")
        if ref not in cfg["references"]:
            raise ValueError(
                f"masking.reference '{ref}' is not a declared reference")
        if not mask.get("output_suffix"):
            raise ValueError(
                "masking is enabled but no output_suffix is set; a masked "
                "run would overwrite the unmasked one under the same "
                "filenames and the two would be indistinguishable")

    return cfg


# ── Path helpers ──────────────────────────────────────────────────────────────

def _var_paths(
    model: str,
    model_cfg: dict,
    var: str,
    mip: str,
    grid_override: str | None,
    paths_cfg: dict,
) -> list[Path]:
    """Return sorted netCDF paths for one variable of one model.

    Falls back to the pre-staged DATA_ROOT directory if the CMIP6 pool
    does not contain the files.
    """
    grid = grid_override or model_cfg["grid"]
    base = (
        Path(paths_cfg["cmip6_root"])
        / model_cfg["institute"]
        / model
        / "historical"
        / model_cfg["variant"]
        / mip
        / var
        / grid
        / model_cfg["version"]
    )
    paths = sorted(base.glob(f"{var}_{mip}_*.nc"))
    if not paths:
        staged = sorted(Path(paths_cfg["data_root"]).glob(
            f"{var}_{mip}_{model}_historical_*.nc"))
        if staged:
            return staged
        logger.warning(f"  [{model}] no {var} files found under {base}")
    return paths


def _load_var(
    model: str,
    model_cfg: dict,
    var: str,
    mip: str,
    grid_override: str | None,
    paths_cfg: dict,
) -> iris.cube.CubeList:
    """Load and concatenate one variable for one model.

    Returns an empty CubeList if files are missing.
    """
    paths = _var_paths(model, model_cfg, var, mip, grid_override,
                       paths_cfg)
    if not paths:
        return iris.cube.CubeList()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cubes = iris.load([str(p) for p in paths])
    # Concatenate per-decade chunks (CMIP6 files are often split by decade)
    try:
        equalise_attributes(cubes)
        unify_time_units(cubes)
        cubes = cubes.concatenate()
    except Exception as e:
        logger.warning(f"  [{model}] {var} concatenation failed: {e}")
    return cubes


def _report_coverage(loaded: dict[str, set[str]], cfg: dict) -> None:
    """Say which figures the loaded data can actually support.

    Printed before the suite runs rather than discovered afterwards from
    a missing file: mlotst in particular is archived by only three of the
    seven models, and a reader comparing the ensemble rows of Figs 7 and
    8 needs to know that the two rest on different sample sizes.
    """
    logger.info("Figure coverage:")
    for fig, spec in cfg["figures"].items():
        have = [m for m, v in loaded.items() if set(spec["vars"]) <= v]
        n, total = len(have), len(cfg["models"])
        level = "OK" if n == total else ("PARTIAL" if n else "NO DATA")
        logger.info(f"  {fig:24s} {n}/{total} models  [{level}]  "
                    f"obs: {'+'.join(spec['obs'])}")
        if 0 < n < total:
            logger.warning(f"    missing: "
                           f"{sorted(set(cfg['models']) - set(have))}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from climateeval.data._utils import DataSourceInformation
    from climateeval.suites._base import Suite

    cfg = load_config(CONFIG_PATH)
    logger.info(f"Recipe: {CONFIG_PATH}")

    suite_cfg = cfg["suite"]
    mask_cfg = cfg.get("masking", {})
    masking = bool(mask_cfg.get("mask_models_to_obs"))

    # Separate outputs when masking, so the two runs can be compared
    # rather than one silently replacing the other.
    suffix = mask_cfg.get("output_suffix", "") if masking else ""
    db_path = Path(suite_cfg["database"]).with_name(
        Path(suite_cfg["database"]).stem + suffix
        + Path(suite_cfg["database"]).suffix)
    fig_dir = Path(str(suite_cfg["figure_dir"]) + suffix)

    if db_path.is_file():
        db_path.unlink()
        logger.info(f"Removed existing {db_path}")
    fig_dir.mkdir(exist_ok=True)

    logger.info(f"Figures requested: {len(cfg['figures'])}")
    logger.info(f"Variables required: {[v[0] for v in cfg['_variables']]}")
    if masking:
        logger.info(
            f"Models masked to the {mask_cfg['reference']} sampling, "
            f"month by month, before the climatology is removed")
        logger.info(f"  -> {db_path} and {fig_dir}/")
    else:
        logger.info("Models NOT masked to the observational sampling; "
                    "model and observed correlations may rest on "
                    "different sets of months")

    suite = Suite(
        suite_cfg["name"],
        diagnostic_kwargs={
            "data_root_dir": Path(cfg["paths"]["data_root"]),
            **cfg.get("diagnostic_kwargs", {}),
            # Restrict the diagnostic to the manuscript figures; the
            # suite otherwise emits its full set, most of which does not
            # appear in the paper.
            "figures": tuple(cfg["figures"]),
            "figure_options": {k: v.get("options", {})
                               for k, v in cfg["figures"].items()},
            "output_dir": fig_dir,
            # Blank each model wherever the reference has no retrieval
            # in that month, applied after regridding and before the
            # climatology is removed so that climatology, trend and
            # anomalies all come from the same calendar sampling.
            "mask_to_obs": masking,
            "mask_reference": mask_cfg.get("reference"),
            "mask_min_retained": mask_cfg.get("min_retained_fraction", 0.0),
            "mask_warn_below": mask_cfg.get("warn_below_fraction", 0.0),
        },
        variable_kwargs={"timerange": suite_cfg["timerange"]},
    )

    loaded: dict[str, set[str]] = {}

    for model, model_cfg in cfg["models"].items():
        logger.info(f"Loading {model}...")

        all_cubes = iris.cube.CubeList()
        got: set[str] = set()
        for var, mip, grid_override in cfg["_variables"]:
            var_cubes = _load_var(model, model_cfg, var, mip, grid_override,
                                  cfg["paths"])
            if var_cubes:
                logger.info(f"  [{model}] loaded {var} "
                            f"({len(var_cubes)} cube(s))")
                all_cubes.extend(var_cubes)
                got.add(var)

        # chl is required: it is the subject of the paper and every
        # figure except 4, 5 and 6 uses it.
        if "chl" not in got:
            logger.warning(f"  [{model}] skipping: no chl data found")
            continue
        loaded[model] = got

        data_info = DataSourceInformation(
            name=model,
            category="CMIP6",
            institute=model_cfg["institute"],
            exp="historical",
            variant=model_cfg["variant"],
            references=(),
        )

        logger.info(f"Running suite for {model} "
                    f"({len(all_cubes)} cubes total)...")
        try:
            suite.get_database(
                all_cubes,
                data_info,
                database_resource=f"duckdb://{db_path}",
            )
            logger.success(f"  {model} done")
        except Exception as e:
            logger.warning(f"  {model} failed: {e}")
            import traceback
            logger.opt(colors=False).debug(traceback.format_exc())

    _report_coverage(loaded, cfg)

    logger.info(f"Database written to {db_path}")
    logger.info(f"Figures written to {fig_dir}/")
    for name, spec in cfg.get("external_figures", {}).items():
        logger.info(f"{name} is not produced here; run {spec['script']} "
                    f"separately")
    logger.info("Generate HTML report with:")
    logger.info(f"  python3 generate_html.py {db_path} "
                f"-o {suite_cfg['name']}_paper.html")


if __name__ == "__main__":
    main()
