#!/usr/bin/env python3
"""
seaice_chl_analysis.py
======================
Standalone script for the sea-ice / Chl-a relationship analysis.

Imports all data loaders, path constants, and utilities directly from
example_chl_nodatabse5.py so that file discovery, unit conversion,
calendar handling and regridding are identical to the main diagnostic.

Set MAIN_SCRIPT below to the absolute path of example_chl_nodatabse5.py.

Produces in OUT_DIR:
  figS_seaice_chl_polar_indiv_Arctic.png
  figS_seaice_chl_polar_indiv_Southern.png
  figS_seaice_chl_polar_ens_Arctic.png
  figS_seaice_chl_polar_ens_Southern.png
  figS_seaice_chl_corr_Arctic.png
  figS_seaice_chl_corr_Southern.png
  figS_seaice_chl_regression.png
  figS_seaice_eof.png
  figS_chl_eof.png
  figS_bloom_phenology_Arctic.png
  figS_bloom_phenology_Southern.png
"""
from __future__ import annotations
import importlib.util
import sys
import warnings
from pathlib import Path

# ============================================================
# Set this to the absolute path of the main diagnostic script
# ============================================================
MAIN_SCRIPT = Path(
    "/home/b/b384036/ClimateEval/example_chl_nodatabse5_new.py"
)  # ← point to your updated script

OUT_DIR = Path.cwd() / "paper_figures_seaice"
OUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Import everything from the main script
# ============================================================
print(f"Importing from {MAIN_SCRIPT.name} ...", flush=True)
_spec = importlib.util.spec_from_file_location("main_diag", str(MAIN_SCRIPT))
_mod  = importlib.util.module_from_spec(_spec)
sys.modules["main_diag"] = _mod
_spec.loader.exec_module(_mod)

# Pull every name we need into this module's global namespace
from main_diag import (
    # Constants
    MODELS, PALETTE, LAG_PERIOD,
    # Data loaders
    load_obs_chl, load_model_chl,
    load_obs_siconc, load_model_siconc,
    # Utilities
    regrid_timeseries, snap_to_month_start,
    # Figure functions (sea-ice / Chl-a)
    make_figS_seaice_chl_polar_corr,
    make_figS_seaice_chl_spatial_corr,
    make_figS_seaice_chl_regression,
    make_figS_seaice_eof,
    make_figS_chl_eof,
    make_figS_bloom_phenology,
)

# Override OUT_DIR in the imported module so figures go to our directory
import main_diag as _md
_md.OUT_DIR = OUT_DIR

import numpy as np
import xarray as xr
from loguru import logger

print("Import complete.", flush=True)

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    logger.info("=== Sea-ice / Chl-a analysis ===")
    logger.info(f"Output directory: {OUT_DIR}")

    # ---- Common 1x1 grid ----
    _lat1 = np.arange(-89.5, 90.5, 1.0)
    _lon1 = np.arange(  0.5, 360.5, 1.0)
    common_grid = xr.DataArray(
        np.zeros((len(_lat1), len(_lon1))),
        dims=("lat", "lon"),
        coords={"lat": _lat1, "lon": _lon1},
    )

    # ---- Load observations ----
    logger.info("Loading OBS Chl-a...")
    obs_chl = None
    try:
        obs_chl = load_obs_chl()
        if obs_chl is not None:
            logger.success(f"OBS Chl-a loaded: {obs_chl.shape}")
    except Exception as e:
        logger.warning(f"OBS Chl-a load failed: {e}")

    logger.info("Loading OBS siconc...")
    obs_siconc = None
    try:
        obs_siconc = load_obs_siconc()
        if obs_siconc is not None:
            logger.success(f"OBS siconc loaded: {obs_siconc.shape}")
    except Exception as e:
        logger.warning(f"OBS siconc load failed: {e}")

    # ---- Load models ----
    model_chl_ts:    dict[str, xr.DataArray] = {}
    model_siconc_ts: dict[str, xr.DataArray] = {}

    for model in MODELS:
        logger.info(f"  Loading [{model}] Chl-a...")
        try:
            chl = load_model_chl(model)
            if chl is not None:
                model_chl_ts[model] = chl
                logger.success(f"  [{model}] Chl-a loaded: {chl.shape}")
        except Exception as e:
            logger.warning(f"  [{model}] Chl-a load failed: {e}")

        logger.info(f"  Loading [{model}] siconc...")
        try:
            sic = load_model_siconc(model)
            if sic is not None:
                model_siconc_ts[model] = sic
                logger.success(f"  [{model}] siconc loaded: {sic.shape}")
        except Exception as e:
            logger.warning(f"  [{model}] siconc load failed: {e}")

    logger.info(
        f"Loaded: {len(model_chl_ts)} models with Chl-a, "
        f"{len(model_siconc_ts)} models with siconc"
    )

    if obs_chl is None and not model_chl_ts:
        logger.error("No Chl-a data available — aborting")
        sys.exit(1)

    obs_full = obs_chl

    def _run(name, fn, **kwargs):
        try:
            fn(**kwargs)
        except Exception as e:
            import traceback
            logger.warning(f"  {name} crashed: {e}")
            logger.warning(traceback.format_exc())

    # ---- Fig 1+2: Individual model maps + ensemble ----
    if model_chl_ts:
        _run("polar_corr", make_figS_seaice_chl_polar_corr,
             obs_chl_ts=obs_full,
             obs_siconc_ts=obs_siconc,
             model_chl_ts=model_chl_ts,
             model_siconc_ts=model_siconc_ts,
             common_grid=common_grid)

    # ---- Fig 3: OBS + ensemble mean at observed peak lag ----
    if model_chl_ts:
        _run("spatial_corr", make_figS_seaice_chl_spatial_corr,
             obs_chl_ts=obs_full,
             obs_siconc_ts=obs_siconc,
             model_chl_ts=model_chl_ts,
             model_siconc_ts=model_siconc_ts,
             common_grid=common_grid)

    # ---- Fig 4: Explained variance (multi-PC regression) ----
    if model_chl_ts and model_siconc_ts:
        _run("regression", make_figS_seaice_chl_regression,
             obs_chl_ts=obs_full,
             obs_siconc_ts=obs_siconc,
             model_chl_ts=model_chl_ts,
             model_siconc_ts=model_siconc_ts,
             common_grid=common_grid)

    # ---- Fig 5: Sea-ice EOF analysis ----
    _run("seaice_eof", make_figS_seaice_eof,
         obs_siconc_ts=obs_siconc,
         model_siconc_ts=model_siconc_ts,
         common_grid=common_grid)

    # ---- Fig 6: Chl-a EOF analysis ----
    if obs_full is not None or model_chl_ts:
        _run("chl_eof", make_figS_chl_eof,
             obs_chl_ts=obs_full,
             model_chl_ts=model_chl_ts,
             common_grid=common_grid)

    # ---- Fig 7+8: Bloom phenology + sea-ice retreat ----
    if model_chl_ts:
        _run("bloom_phenology", make_figS_bloom_phenology,
             obs_chl_ts=obs_full,
             obs_siconc_ts=obs_siconc,
             model_chl_ts=model_chl_ts,
             model_siconc_ts=model_siconc_ts,
             common_grid=common_grid)

    logger.info(f"Done. All figures in {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("figS_seaice*.png")):
        logger.info(f"  {f.name}")
