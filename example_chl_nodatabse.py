"""
CMIP6 surface chlorophyll-a diagnostic: 12 figures total.

Phase A — mean state and seasonal cycle (paper Figs 1-4):
    fig1_obs_climatology.png    ESA-CCI time-mean Chl-a, log scale
    fig2_bias_maps.png          per-model (model - obs) Chl-a bias maps  [2x2]
    fig3_basin_bias.png         basin-mean Chl-a bias bar chart
    fig4_seasonal_cycle.png     mean seasonal cycle, all models + obs

Phase B — physical driver biases (paper Figs 5-7):
    fig5_mld_bias.png           MLD bias (model - ORAS5)                 [2x2]
    fig6_sst_bias.png           SST bias (model - NOAA-ERSSTv3b)         [2x2]
    fig7_clt_bias.png           cloud cover bias (model - ESA-CCI AVHRR) [2x2]

Phase C — MLD-Chl coupling diagnostics:
    fig8_hovmoller_mld_chl.png  Hovmoller r(MLD anomaly, Chl-a bias)
                                on (latitude, month-of-year)             [2x2]
    fig8b_pattern_corr.png      Basin-mean spatial pattern correlation
                                r(Chl-a, MLD) and r(Chl-a, SST),
                                OBS + models, one bar group per basin    [2 panels]

Phase D — lag correlation of basin-mean anomalies:
    fig9_lag_corr_chl_sst.png   Pearson r(Chl, SST) at lags -12..+12 months
                                per ocean basin, OBS + 4 models          [6 subplots]
    fig10_lag_corr_chl_mld.png  Pearson r(Chl, MLD) at lags -12..+12 months
                                per ocean basin, OBS + 4 models          [6 subplots]

Phase E — sea-ice memory diagnostics:
    fig11_lag_corr_chl_siconc.png  Pearson r(Chl, siconc) at lags -12..+12 mo,
                                   Arctic & Southern Ocean, OBS + 4 models
    fig12_siconc_memory.png        12-month autocorrelation of basin-mean
                                   siconc anomalies, OBS + 4 models — direct
                                   test of the broken-sea-ice-memory hypothesis
    figS1_siconc_memory_seasonal_zone.png
                                   supplementary: same as Fig 12 but cells
                                   restricted to the seasonal-ice zone of
                                   each source (0.05 < mean siconc < 0.95);
                                   robustness check against static-cell inflation
    figS2_picontrol_envelope.png
                                   supplementary: internal-variability
                                   envelope of the 12-month siconc
                                   autocorrelation. The Fig 12 statistic is
                                   recomputed on overlapping 17-yr segments of
                                   each model's piControl run; violins show the
                                   resulting distribution, against the observed
                                   value. Tests whether the historical
                                   sea-ice-persistence mismatch exceeds internal
                                   variability.

NOTE on detrending: the variability diagnostics (Figs 9-12, S1, S2)
operate on basin-mean anomaly series that are deseasonalised, then
LINEARLY DETRENDED, then standardised — the order specified in
Section 2.6 of the manuscript. See _deseasonalise_zscore().

CLI: `python <thisfile> --audit [historical|piControl]` lists, per
model, the DKRZ-pool variant labels that carry every variable the
diagnostic needs (the intersection across variables).
"""

from __future__ import annotations
import cartopy
import shapely
print(f"cartopy={cartopy.__version__}  shapely={shapely.__version__}")
print(f"cartopy from: {cartopy.__file__}")

import sys
import warnings
import re
import string
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from loguru import logger
from matplotlib.colors import LogNorm, TwoSlopeNorm


# ============================================================
# Logger
# ============================================================

logger = logger.opt(colors=True)
logger.remove()
logger.add(
    sys.stdout, level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    colorize=True,
)


# ============================================================
# Paths and configuration
# ============================================================

MODEL_DIR = Path("/work/bd1179/b384036/climateeval_input")
OBS_DIR   = Path("/work/bd0854/DATA/ESMValTool2/OBS/Tier2/ESACCI-OC")
OUT_DIR   = Path.cwd() / "paper_figures"
OUT_DIR.mkdir(exist_ok=True)

# ORAS5 monthly mlotst (somxl010), 1x1 grid, OPeNDAP
ORAS5_OPENDAP = (
    "http://apdrc.soest.hawaii.edu:80/dods/"
    "public_data/Reanalysis_Data/ORAS5/1x1_grid/somxl010/opa0"
)

MODELS = [
    "ACCESS-ESM1-5",     # WOMBAT
    "CanESM5",           # CMOC
    "CESM2",             # MARBL (CESM2-FV2/WACCM/WACCM-FV2 share BGC; omitted)
    "GFDL-ESM4",         # COBALTv2
    "IPSL-CM6A-LR",      # PISCES
    "MPI-ESM-1-2-HAM",   # HAMOCC6 (MPI-ESM1-2-HR/LR share BGC; omitted)
    "NorESM2-LM",        # iHAMOCC
]
# Full seven-model ensemble, matching the manuscript (Table 1).
# ACCESS-ESM1-5 (WOMBAT) and CanESM5 (CMOC) lack dissolved iron (dfe)
# in their CMIP6 historical archives on Levante, so they are absent only
# from the iron-limitation analysis (Section 3.7): make_fig_limitation
# filters to models with dfe/rsds and excludes them automatically. All
# other diagnostics (mean state, lag correlations, sea-ice bias maps)
# use all seven where the required field is available.

PERIOD     = "1997-2014"
CHL_LABEL  = r"Chl-a (mg m$^{-3}$)"
LOG_VMIN   = 0.02
LOG_VMAX   = 5.0
BIAS_LIMIT = 1.0

BASINS = {
    "Global":   (-90,  90,   0, 360),
    "Atlantic": (-60,  70, 290, 360),
    "Pacific":  (-60,  60, 130, 280),
    "Indian":   (-60,  30,  20, 130),
    # Polar basins bounded to where both the satellite reference
    # (ESA-CCI) and the models carry meaningful Chl-a coverage.
    # ESA-CCI retrievals essentially disappear poleward of ~80 deg
    # (no winter coverage, low solar elevation, ice and cloud
    # contamination), and CMIP6 ocean BGC models carry only marginal
    # Chl-a poleward of the perennial ice edge. These bounds apply to
    # ALL diagnostics that consume BASINS (Fig 3 mean-state; Figs 9-10
    # SST/MLD lag-correlation; any other figure using _build_masks).
    # The original variability results in Section 3.6 of the manuscript
    # (e.g. Southern Ocean Chl/SST r ~ -0.40 at lag 0; Arctic Chl/SST
    # r ~ +0.43 at +11 mo) were computed on the previous boxes
    # (Southern -90 to -45; Arctic 65 to 90) and will change when these
    # figures are regenerated.
    "Southern": (-80, -30,   0, 360),
    "Arctic":   ( 60,  80,   0, 360),
}

# --- Southern Ocean decomposition -------------------------------------
# The single "Southern" box above mixes the seasonal-ice zone (where
# sea ice advances and retreats every year, and where the cryosphere
# memory mechanism actually operates) with the open Antarctic
# Circumpolar Current band (essentially ice-free year-round). Averaging
# the two together dilutes the Chl-a/sea-ice signal. SOUTHERN_SUBBASINS
# splits the Southern Ocean into:
#   Southern-SIZ : seasonal-ice zone, ~55-70 S  -- the memory zone
#   Southern-ACC : open ACC band,     ~45-55 S  -- the control zone
# These are added to the basin dictionary used by the variability
# diagnostics (Figs 9-12, S1, S2) but NOT to the mean-state figures,
# which keep the original six-basin decomposition for comparability
# with Fig 3.
SOUTHERN_SUBBASINS = {
    "Southern-SIZ": (-70, -55,   0, 360),
    "Southern-ACC": (-55, -45,   0, 360),
}
BASINS.update(SOUTHERN_SUBBASINS)

# The original six basins, used by the mean-state figures (Fig 3) and
# the SST/MLD lag-correlation grid (Figs 9, 10) so those panels keep
# the decomposition used in the manuscript. The Southern Ocean
# sub-basins enter only through SICONC_BASINS (Figs 11, 12, S1, S2).
MEANSTATE_BASINS = ["Global", "Atlantic", "Pacific",
                    "Indian", "Southern", "Arctic"]

# --- Regions for the seasonal cycle figure (Fig 4) --------------------
# Fig 4 uses a different decomposition from the rest of the paper: it
# resolves the global mean into 8 physically meaningful regions that
# span distinct seasonal-cycle regimes. The point is to expose the
# tropical / mid-latitude / polar contrast (and the ITCZ-controlled
# tropical regimes specifically) that the global mean would mask.
#
# Boundaries follow the conventions used by NOAA Climate Prediction
# Center / WCRP ocean-basin diagnostics. The four tropical and
# mid-latitude regions are basin-bounded; the two polar regions are
# zonal bands consistent with the polar caps used by BASINS.
#
# These regions are USED ONLY by make_fig4. The rest of the paper
# continues to use BASINS / MEANSTATE_BASINS as before.
SEASONAL_REGIONS = {
    "Global":           (-90,  90,   0, 360),
    "Arctic":           ( 60,  80,   0, 360),
    "Southern Ocean":   (-80, -30,   0, 360),
    "North Atlantic":   ( 30,  70, 290, 360),
    "North Pacific":    ( 30,  60, 130, 260),
    "Indian":           (-30,  30,  20, 130),
    "Tropical Pacific": (-20,  20, 130, 280),
    "South Pacific":    (-30,  -5, 150, 290),
}

# Display order for the 2x4 panel grid. Global first as a reference,
# then poles (Arctic and Southern Ocean), then the four mid-latitude /
# tropical basin regions in roughly geographic order.
SEASONAL_REGION_ORDER = [
    "Global", "Arctic", "Southern Ocean", "North Atlantic",
    "North Pacific", "Indian", "Tropical Pacific", "South Pacific",
]

PALETTE = ["#4477AA", "#EE6677", "#228833", "#AA3377",
           "#CCBB44", "#66CCEE", "#EE7733", "#000000"]


# ============================================================
# Driver file paths (MLD, SST, clt) per model
# ============================================================

MLD_PATHS = {
    "GFDL-ESM4": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NOAA-GFDL/GFDL-ESM4/"
             "historical/r1i1p1f1/Omon/mlotst/gr/v20190726"),
        "mlotst_Omon_GFDL-ESM4_historical_r1i1p1f1_gr_*.nc",
    ),
    "CESM2": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NCAR/CESM2/"
             "historical/r1i1p1f1/Omon/mlotst/gn/v20190308"),
        "mlotst_Omon_CESM2_historical_r1i1p1f1_gn_*.nc",
    ),
    "MPI-ESM-1-2-HAM": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/HAMMOZ-Consortium/"
             "MPI-ESM-1-2-HAM/historical/r1i1p1f1/Omon/mlotst/gn/v20190627"),
        "mlotst_Omon_MPI-ESM-1-2-HAM_historical_r1i1p1f1_gn_*.nc",
    ),
    "UKESM1-0-LL": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/"
             "historical/r1i1p1f2/Omon/mlotst/gn/v20190406"),
        "mlotst_Omon_UKESM1-0-LL_historical_r1i1p1f2_gn_*.nc",
    ),
    # ---- New models (one config per BGC family) -----------------------
    "ACCESS-ESM1-5": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/"
             "historical/r1i1p1f1/Omon/mlotst/gn/v20191115"),
        "mlotst_Omon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_*.nc",
    ),
    "CanESM5": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/CCCma/CanESM5/"
             "historical/r1i1p1f1/Omon/mlotst/gn/v20190429"),
        "mlotst_Omon_CanESM5_historical_r1i1p1f1_gn_*.nc",
    ),
    "IPSL-CM6A-LR": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/"
             "historical/r1i1p1f1/Omon/mlotst/gn/v20180803"),
        "mlotst_Omon_IPSL-CM6A-LR_historical_r1i1p1f1_gn_*.nc",
    ),
    "NorESM2-LM": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NCC/NorESM2-LM/"
             "historical/r1i1p1f1/Omon/mlotst/gn/v20190815"),
        "mlotst_Omon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc",
    ),
}

SST_OBS_FILE = Path(
    "/work/bd0854/DATA/ESMValTool2/OBS/Tier2/NOAA-ERSSTv3b/"
    "OBS6_NOAA-ERSSTv3b_reanaly_v3b_Omon_tos_185401-202001.nc"
)

SST_PATHS = {
    "GFDL-ESM4": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NOAA-GFDL/GFDL-ESM4/"
             "historical/r1i1p1f1/Omon/tos/gr/v20190726"),
        "tos_Omon_GFDL-ESM4_historical_r1i1p1f1_gr_*.nc",
    ),
    "CESM2": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NCAR/CESM2/"
             "historical/r1i1p1f1/Omon/tos/gn/v20190308"),
        "tos_Omon_CESM2_historical_r1i1p1f1_gn_*.nc",
    ),
    "MPI-ESM-1-2-HAM": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/HAMMOZ-Consortium/"
             "MPI-ESM-1-2-HAM/historical/r1i1p1f1/Omon/tos/gn/v20190627"),
        "tos_Omon_MPI-ESM-1-2-HAM_historical_r1i1p1f1_gn_*.nc",
    ),
    "UKESM1-0-LL": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/"
             "historical/r2i1p1f2/Omon/tos/gn/v20190708"),
        "tos_Omon_UKESM1-0-LL_historical_r2i1p1f2_gn_*.nc",
    ),
    # ---- New models (one config per BGC family) -----------------------
    "ACCESS-ESM1-5": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/"
             "historical/r1i1p1f1/Omon/tos/gn/v20191115"),
        "tos_Omon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_*.nc",
    ),
    "CanESM5": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/CCCma/CanESM5/"
             "historical/r1i1p1f1/Omon/tos/gn/v20190429"),
        "tos_Omon_CanESM5_historical_r1i1p1f1_gn_*.nc",
    ),
    "IPSL-CM6A-LR": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/"
             "historical/r1i1p1f1/Omon/tos/gn/v20180803"),
        "tos_Omon_IPSL-CM6A-LR_historical_r1i1p1f1_gn_*.nc",
    ),
    "NorESM2-LM": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NCC/NorESM2-LM/"
             "historical/r1i1p1f1/Omon/tos/gn/v20190815"),
        "tos_Omon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc",
    ),
}

CLT_OBS_FILE = Path(
    "/work/bd0854/DATA/ESMValTool2/OBS/Tier2/ESACCI-CLOUD/"
    "OBS6_ESACCI-CLOUD_sat_v3.0-AVHRR-AMPM_Amon_clt_198201-201612.nc"
)

CLT_PATHS = {
    "GFDL-ESM4": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NOAA-GFDL/GFDL-ESM4/"
             "historical/r1i1p1f1/Amon/clt/gr1/v20190726"),
        "clt_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_*.nc",
    ),
    "CESM2": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NCAR/CESM2/"
             "historical/r1i1p1f1/Amon/clt/gn/v20190308"),
        "clt_Amon_CESM2_historical_r1i1p1f1_gn_*.nc",
    ),
    "MPI-ESM-1-2-HAM": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/HAMMOZ-Consortium/"
             "MPI-ESM-1-2-HAM/historical/r1i1p1f1/Amon/clt/gn/v20190627"),
        "clt_Amon_MPI-ESM-1-2-HAM_historical_r1i1p1f1_gn_*.nc",
    ),
    "UKESM1-0-LL": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/"
             "historical/r1i1p1f2/Amon/clt/gn/v20190406"),
        "clt_Amon_UKESM1-0-LL_historical_r1i1p1f2_gn_*.nc",
    ),
    # ---- New models (one config per BGC family) -----------------------
    "ACCESS-ESM1-5": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/"
             "historical/r1i1p1f1/Amon/clt/gn/v20191115"),
        "clt_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_*.nc",
    ),
    "CanESM5": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/CCCma/CanESM5/"
             "historical/r1i1p1f1/Amon/clt/gn/v20190429"),
        "clt_Amon_CanESM5_historical_r1i1p1f1_gn_*.nc",
    ),
    "IPSL-CM6A-LR": (
        # IPSL clt is on regular grid 'gr' (not native 'gn' like the
        # ocean vars). Both grids are accepted by the loader.
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/"
             "historical/r1i1p1f1/Amon/clt/gr/v20180803"),
        "clt_Amon_IPSL-CM6A-LR_historical_r1i1p1f1_gr_*.nc",
    ),
    "NorESM2-LM": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NCC/NorESM2-LM/"
             "historical/r1i1p1f1/Amon/clt/gn/v20190815"),
        "clt_Amon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc",
    ),
}


# ------------------------------------------------------------
# Sea-ice concentration (siconc) — for Phase E (Figs 11 & 12)
#
# Observational reference: HadISST sea-ice fraction, 1° regular grid.
# If a different obs product is preferred (e.g. NSIDC on a regular
# grid), update SICONC_OBS_FILE accordingly; the loader uses the
# variable name 'sic' or 'siconc' or 'sea_ice_fraction', falling
# back to the first data variable.
#
# Model siconc lives in the SImon table. Version subdirectories
# (v20...) follow the convention used by each modelling group;
# they may differ from the Omon/Amon versions used elsewhere in
# this script. The loader skips silently and logs a warning if a
# given path doesn't resolve, so the rest of the pipeline still
# runs even if one or two siconc paths need updating after the
# first run.
# ------------------------------------------------------------

SICONC_OBS_FILE = Path(
    "/work/bd0854/DATA/ESMValTool2/OBS/Tier2/HadISST/"
    "OBS6_HadISST_reanaly_1_SImon_siconc_187001-202112.nc"
)

SICONC_PATHS = {
    "GFDL-ESM4": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NOAA-GFDL/GFDL-ESM4/"
             "historical/r1i1p1f1/SImon/siconc/gr/v20190726"),
        "siconc_SImon_GFDL-ESM4_historical_r1i1p1f1_gr_*.nc",
    ),
    "CESM2": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NCAR/CESM2/"
             "historical/r1i1p1f1/SImon/siconc/gn/v20190308"),
        "siconc_SImon_CESM2_historical_r1i1p1f1_gn_*.nc",
    ),
    "MPI-ESM-1-2-HAM": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/HAMMOZ-Consortium/"
             "MPI-ESM-1-2-HAM/historical/r1i1p1f1/SImon/siconc/gn/v20190627"),
        "siconc_SImon_MPI-ESM-1-2-HAM_historical_r1i1p1f1_gn_*.nc",
    ),
    "UKESM1-0-LL": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/MOHC/UKESM1-0-LL/"
             "historical/r1i1p1f2/SImon/siconc/gn/v20190406"),
        "siconc_SImon_UKESM1-0-LL_historical_r1i1p1f2_gn_*.nc",
    ),
    # ---- New models (one config per BGC family) -----------------------
    "ACCESS-ESM1-5": (
        # Two siconc versions exist (v20191115, v20200817). Pick the
        # later (v20200817) which supersedes the earlier release.
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/CSIRO/ACCESS-ESM1-5/"
             "historical/r1i1p1f1/SImon/siconc/gn/v20200817"),
        "siconc_SImon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_*.nc",
    ),
    "CanESM5": (
        # Two siconc versions exist (v20190306, v20190429). Pick later.
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/CCCma/CanESM5/"
             "historical/r1i1p1f1/SImon/siconc/gn/v20190429"),
        "siconc_SImon_CanESM5_historical_r1i1p1f1_gn_*.nc",
    ),
    "IPSL-CM6A-LR": (
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/IPSL/IPSL-CM6A-LR/"
             "historical/r1i1p1f1/SImon/siconc/gn/v20180803"),
        "siconc_SImon_IPSL-CM6A-LR_historical_r1i1p1f1_gn_*.nc",
    ),
    "NorESM2-LM": (
        # Assumed gn/v20190815 (consistent with the other NorESM2-LM
        # variable versions). If the loader logs a missing-file error,
        # check `ls /work/ik1017/.../NorESM2-LM/.../SImon/siconc/` for
        # the correct grid/version and update this entry.
        Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NCC/NorESM2-LM/"
             "historical/r1i1p1f1/SImon/siconc/gn/v20190815"),
        "siconc_SImon_NorESM2-LM_historical_r1i1p1f1_gn_*.nc",
    ),
}


# ------------------------------------------------------------
# piControl siconc -- for the internal-variability envelope (Fig S2)
#
# Fig S2 recomputes the Fig 12 statistic on overlapping 17-yr segments
# of each model's preindustrial-control run, to characterise how much
# of the historical sea-ice-persistence spread is internal variability.
#
# piControl variant labels and version directories are NOT generally
# the same as the historical ones, so the loader globs the variant and
# version levels rather than hard-coding them. CMIP6_ROOT and the
# institution map below are used to build the search path:
#   {CMIP6_ROOT}/{institution}/{model}/piControl/*/SImon/siconc/*/*/*.nc
# ------------------------------------------------------------

CMIP6_ROOT = Path("/work/ik1017/CMIP6/data/CMIP6/CMIP")

INSTITUTIONS = {
    "GFDL-ESM4":        "NOAA-GFDL",
    "CESM2":            "NCAR",
    "MPI-ESM-1-2-HAM":  "HAMMOZ-Consortium",
    "UKESM1-0-LL":      "MOHC",
}


# ============================================================
# Unit handling -- convert Chl to mg m-3
# ============================================================

def to_mg_m3(
    da: xr.DataArray,
    units_str: str | None,
    model: str | None = None,
) -> xr.DataArray:
    """Convert chlorophyll-a DataArray to mg m-3.

    Per-model overrides take precedence over the units attribute,
    because some CMIP6 models ship chl with a units label that does
    NOT match the actual numerical values stored in the file. Known
    cases handled here:

      IPSL-CM6A-LR (PISCES) -- The file metadata declares
          units = "kg m-3" but the values themselves are stored as
          if they were g m-3. A naive "kg m-3 -> *1e6" conversion
          yields chl ~380 mg/m3 globally, which is ~1000x too large.
          The fix is to multiply by 1e3 instead of 1e6, treating the
          stored numbers as g/m3 rather than kg/m3.
          Verified empirically with cdo / xarray: surface global mean
          of the raw stored numbers is ~3.8e-4, which corresponds to
          0.38 mg/m3 if interpreted as g/m3 -- consistent with the
          ESA-CCI observed global mean of ~0.2-0.4 mg/m3.

    Add further entries to MODEL_CHL_OVERRIDES below as new models
    are added to the ensemble and their unit conventions are
    verified.
    """
    # Per-model overrides: model name -> multiplicative factor that
    # produces mg/m3 regardless of what the units attribute claims.
    MODEL_CHL_OVERRIDES: dict[str, float] = {
        "IPSL-CM6A-LR": 1e3,   # units string says kg/m3 (would give
                               # ~380 mg/m3 with the standard *1e6
                               # conversion -- 1000x too high). The
                               # file actually stores values in g/m3,
                               # so multiplying by 1000 yields the
                               # correct ~0.38 mg/m3 global mean.
    }
    if model is not None and model in MODEL_CHL_OVERRIDES:
        factor = MODEL_CHL_OVERRIDES[model]
        logger.info(
            f"  [{model}] applying model-specific chl scale "
            f"x{factor:g} (units attr says '{units_str}' but is "
            f"known to be mislabelled)"
        )
        return da * factor

    # ---- standard units-string-based dispatch ----
    u = (units_str or "").strip().lower().replace(" ", "")
    if u in {"kgm-3", "kg/m3", "kgm**-3", "kg.m-3"}:
        return da * 1e6
    if u in {"mgm-3", "mg/m3", "mgm**-3", "mg.m-3"}:
        return da * 1.0
    if u in {"mgl-1", "mg/l", "mgl**-1", "mgdm-3", "gm-3"}:
        return da * 1.0
    if u in {"ugl-1", "ug/l", "\u00b5gl-1", "\u00b5g/l", "ugm-3", "ug/m3"}:
        return da * 1.0
    finite = da.values[np.isfinite(da.values)]
    if finite.size == 0:
        logger.warning("Cannot convert: array is all NaN")
        return da
    mx = float(np.nanmax(finite))
    if mx < 1e-3:
        logger.warning(f"Units unknown (max={mx:.2e}); assuming kg m-3 -> *1e6")
        return da * 1e6
    if mx < 1e3:
        logger.warning(f"Units unknown (max={mx:.2f}); assuming mg m-3 (no scale)")
        return da * 1.0
    logger.warning(f"Units unknown (max={mx:.2f}); leaving as-is")
    return da * 1.0


# ============================================================
# Coord helpers
# ============================================================

def standardize(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    for old in ("latitude", "Latitude", "nav_lat"):
        if old in da.coords:
            rename[old] = "lat"
    for old in ("longitude", "Longitude", "nav_lon"):
        if old in da.coords:
            rename[old] = "lon"
    if rename:
        da = da.rename(rename)
    if "lon" in da.coords and da["lon"].ndim == 1 and "lon" in da.dims:
        if float(da.lon.min()) < 0:
            da = da.assign_coords(lon=(da.lon % 360)).sortby("lon")
    return da


def snap_to_month_start(da: xr.DataArray) -> xr.DataArray:
    """Snap all time coordinates to the first day of each month.

    Two code paths. If the axis is already cftime (as for the
    piControl runs, whose synthetic years can exceed the pandas
    datetime64 ceiling of 2262), snap within cftime and never touch
    pandas -- this avoids 'Out of bounds timestamp' overflow. For an
    ordinary pandas/datetime64 axis, snap via pandas as before.
    """
    import pandas as pd
    import cftime
    if "time" not in da.coords:
        return da
    times = da.time.values
    if len(times) and isinstance(times[0], cftime.datetime):
        cal = getattr(times[0], "calendar", "standard")
        new_times = [
            cftime.datetime(t.year, t.month, 1, calendar=cal)
            for t in times
        ]
        return da.assign_coords(time=xr.DataArray(new_times, dims="time"))
    new_times = pd.to_datetime(
        [pd.Timestamp(str(t)).replace(day=1, hour=0, minute=0, second=0,
                                       microsecond=0, nanosecond=0)
         for t in times]
    )
    return da.assign_coords(time=new_times)


# ============================================================
# Chl loaders
# ============================================================

def find_obs_chl_file() -> Path:
    candidates = sorted(OBS_DIR.glob("OBS6_ESACCI-OC_sat_*Omon_chl_*.nc"))
    if not candidates:
        candidates = sorted(OBS_DIR.glob("OBS_ESACCI-OC_sat_*Omon_chl_*.nc"))
    if not candidates:
        raise FileNotFoundError(f"No ESACCI-OC chl file in {OBS_DIR}")
    return candidates[-1]


def load_obs_chl() -> xr.DataArray:
    obs_file = find_obs_chl_file()
    logger.info(f"Obs Chl file: {obs_file.name}")
    ds = xr.open_dataset(obs_file)
    var = "chl" if "chl" in ds.data_vars else list(ds.data_vars)[0]
    da = ds[var].squeeze()
    units_str = ds[var].attrs.get("units")
    da = standardize(da)
    if "time" in da.dims:
        da = da.sel(time=slice("1997", "2014"))
    return to_mg_m3(da, units_str)


def load_model_chl(model: str) -> xr.DataArray:
    candidates = sorted(
        MODEL_DIR.glob(f"chl_Omon_{model}_historical_*_199701-201412.nc")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No chl file for {model}. "
            f"Expected pattern: chl_Omon_{model}_historical_*_199701-201412.nc "
            f"in directory: {MODEL_DIR}. "
            f"chl is loaded from a pre-staged directory (not directly from "
            f"the CMIP6 archive) — stage the file there before running."
        )
    ds = xr.open_dataset(candidates[0], mask_and_scale=True)
    da = ds["chl"]
    units_str = da.attrs.get("units")

    # ---- Strip any vertical dimension and keep only the surface ----
    # The variable name for the depth axis is not standardised across
    # CMIP6 models: GFDL uses "lev", IPSL uses "olevel", some NEMO-based
    # configurations use "deptht", others use "depth". A hard-coded
    # list misses cases (this caused the IPSL chl values to come in
    # ~1000x too large because all 75 depth levels were retained and
    # later averaged). Detect any candidate depth dim defensively.
    KNOWN_Z_NAMES = {
        "lev", "olevel", "depth", "deptht", "depthu", "depthv",
        "depthw", "lev_2", "z_l", "z_t", "st_ocean", "sigma",
        "rho", "pressure", "alt",
    }
    z_dims = []
    # Match by dim name against the known list.
    for d in da.dims:
        if d in KNOWN_Z_NAMES or d.lower().startswith(("lev", "depth", "z_")):
            z_dims.append(d)
    # Also match by the standard_name / axis attribute of coordinate
    # variables, which is the CF-compliant way to identify the
    # vertical axis. This catches exotic names like "olevel_zoo".
    for coord_name in list(da.coords) + list(ds.coords):
        if coord_name not in da.dims and coord_name not in da.coords:
            continue
        coord_attrs = ds[coord_name].attrs if coord_name in ds.coords else {}
        if (coord_attrs.get("axis", "").upper() == "Z" or
            "depth" in coord_attrs.get("standard_name", "").lower() or
            coord_attrs.get("positive", "").lower() in ("up", "down")):
            if coord_name in da.dims and coord_name not in z_dims:
                z_dims.append(coord_name)
    if z_dims:
        logger.info(f"  [{model}] stripping vertical dim(s): {z_dims}; "
                    f"keeping surface level only")
        for zname in z_dims:
            da = da.isel({zname: 0}, drop=True)
            if zname in da.coords:
                da = da.drop_vars(zname, errors="ignore")
            for bnd in (f"{zname}_bnds", f"{zname}_bounds"):
                if bnd in da.coords:
                    da = da.drop_vars(bnd, errors="ignore")
    else:
        logger.info(f"  [{model}] no vertical dim detected (surface 2D)")

    da = da.where(np.abs(da) < 1.0)
    da = da.squeeze()
    da = standardize(da)
    return to_mg_m3(da, units_str, model=model)


# ============================================================
# MLD loaders
# ============================================================

def load_obs_mld() -> xr.DataArray:
    """Stream ORAS5 monthly mlotst from APDRC OPeNDAP, clip to 1997-2014."""
    logger.info(f"Streaming ORAS5 MLD from {ORAS5_OPENDAP} ...")
    ds = xr.open_dataset(ORAS5_OPENDAP)
    da = ds["somxl010"].squeeze()
    da = standardize(da)
    da = da.sel(time=slice("1997", "2014"))
    da = da.where(np.abs(da) < 1e3)
    da.attrs["units"] = "m"
    return snap_to_month_start(da)


def load_model_mld(model: str) -> xr.DataArray | None:
    """Load model mlotst directly from CMIP6 directories."""
    if model not in MLD_PATHS:
        logger.warning(f"  [{model}] no MLD path configured")
        return None
    mld_dir, pattern = MLD_PATHS[model]
    paths = sorted(mld_dir.glob(pattern))
    if not paths:
        logger.warning(f"  [{model}] no mlotst files in {mld_dir}")
        return None
    logger.info(f"  [{model}] loading MLD: {len(paths)} files")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_mfdataset([str(p) for p in paths],
                                combine="by_coords", use_cftime=True)
    da = ds["mlotst"].sel(time=slice("1997", "2014"))
    da = da.where(np.abs(da) < 1e4)
    da = da.squeeze()
    da = standardize(da)
    return da


# ============================================================
# SST loaders
# ============================================================

def load_obs_sst() -> xr.DataArray:
    """Load NOAA-ERSSTv3b SST obs, clip to 1997-2014."""
    logger.info(f"Loading SST obs: {SST_OBS_FILE.name}")
    ds  = xr.open_dataset(str(SST_OBS_FILE))
    var = "tos" if "tos" in ds else list(ds.data_vars)[0]
    da  = ds[var].sel(time=slice("1997", "2014")).squeeze()
    da  = standardize(da)
    da  = da.where(np.abs(da) < 100)
    return da


def load_model_sst(model: str) -> xr.DataArray | None:
    """Load model tos directly from CMIP6 directories."""
    if model not in SST_PATHS:
        logger.warning(f"  [{model}] no SST path configured")
        return None
    sst_dir, pattern = SST_PATHS[model]
    paths = sorted(sst_dir.glob(pattern))
    if not paths:
        logger.warning(f"  [{model}] no tos files in {sst_dir}")
        return None
    logger.info(f"  [{model}] loading SST: {len(paths)} files")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_mfdataset([str(p) for p in paths],
                                combine="by_coords", use_cftime=True)
    da = ds["tos"].sel(time=slice("1997", "2014"))
    da = da.where(np.abs(da) < 100)
    da = da.squeeze()
    da = standardize(da)
    return da


# ============================================================
# Cloud cover (clt) loaders
# ============================================================

def load_obs_clt() -> xr.DataArray:
    logger.info(f"Loading cloud obs: {CLT_OBS_FILE.name}")
    ds  = xr.open_dataset(str(CLT_OBS_FILE))
    var = "clt" if "clt" in ds else list(ds.data_vars)[0]
    da  = ds[var].sel(time=slice("1997", "2014")).squeeze()
    da  = standardize(da)
    da  = da.where(da < 200)   # mask fill values
    return da


def load_model_clt(model: str) -> xr.DataArray | None:
    if model not in CLT_PATHS:
        logger.warning(f"  [{model}] no clt path configured")
        return None
    vdir, pattern = CLT_PATHS[model]
    paths = sorted(vdir.glob(pattern))
    if not paths:
        logger.warning(f"  [{model}] no clt files in {vdir}")
        return None
    logger.info(f"  [{model}] loading clt: {len(paths)} files")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_mfdataset([str(p) for p in paths],
                                combine="by_coords", use_cftime=True)
    da = ds["clt"].sel(time=slice("1997", "2014"))
    da = da.squeeze()
    da = standardize(da)
    return da


# ============================================================
# Sea-ice concentration (siconc) loaders
# ============================================================

def _pick_siconc_var(ds: xr.Dataset) -> str:
    """siconc obs products use various variable names — try the
    common ones in order."""
    for name in ("siconc", "sic", "sea_ice_fraction",
                 "ice_concentration"):
        if name in ds.data_vars:
            return name
    return list(ds.data_vars)[0]


def load_obs_siconc() -> xr.DataArray | None:
    """Load observational siconc, clip to 1997-2014.

    Default reference is HadISST (`sic`, monthly, 1° regular grid).
    If the file is missing, return None and let the caller skip
    the Phase E figures rather than crash the rest of the script.
    """
    if not SICONC_OBS_FILE.exists():
        logger.warning(f"  obs siconc file not found: {SICONC_OBS_FILE}")
        return None
    logger.info(f"Loading siconc obs: {SICONC_OBS_FILE.name}")
    ds  = xr.open_dataset(str(SICONC_OBS_FILE))
    var = _pick_siconc_var(ds)
    da  = ds[var].sel(time=slice("1997", "2014")).squeeze()
    da  = standardize(da)
    # HadISST uses fraction 0-1 with -1e30 fill values; mask anything
    # outside the physical range [0, 1] (or [0, 100] for some products).
    if float(np.nanmax(da.values)) > 1.5:
        da = da / 100.0   # convert percent → fraction
    da = da.where((da >= 0) & (da <= 1))
    da.attrs["units"] = "fraction"
    return snap_to_month_start(da)


def load_model_siconc(model: str) -> xr.DataArray | None:
    """Load model siconc directly from CMIP6 directories.

    siconc lives in the SImon table on each model's native sea-ice
    grid; this function makes no attempt to regrid here — that is
    handled in the lag-correlation routine.
    """
    if model not in SICONC_PATHS:
        logger.warning(f"  [{model}] no siconc path configured")
        return None
    vdir, pattern = SICONC_PATHS[model]
    paths = sorted(vdir.glob(pattern))
    if not paths:
        logger.warning(f"  [{model}] no siconc files in {vdir}")
        return None
    logger.info(f"  [{model}] loading siconc: {len(paths)} files")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_mfdataset([str(p) for p in paths],
                                combine="by_coords", use_cftime=True)
    var = "siconc" if "siconc" in ds else _pick_siconc_var(ds)
    da  = ds[var].sel(time=slice("1997", "2014"))
    da  = da.squeeze()
    da  = standardize(da)
    # CMIP6 siconc is in percent (0-100); convert to fraction for
    # consistency with obs.
    if float(np.nanmax(da.values)) > 1.5:
        da = da / 100.0
    da = da.where((da >= 0) & (da <= 1))
    return da


# ============================================================
# Dissolved iron (dfe) loader
# ============================================================
# CMIP6 dfe is mole_concentration_of_dissolved_iron_in_sea_water
# (mol m-3) on the 4D ocean grid (time, lev, lat, lon). We
# depth-integrate the top 100 m to get a quantity that reflects
# the iron pool the surface mixed layer can draw on, expressed
# as mmol Fe m-2. The exact upper-boundary choice (50 m vs 100 m
# vs MLD) is a tradeoff: 100 m captures the seasonal nutricline
# in most models without being so deep that deep iron pools
# dominate the comparison. We document and stick with 100 m.

# Institute string per model, needed to construct CMIP6 paths
_INSTITUTE_OF: dict[str, str] = {
    "CESM2":            "NCAR",
    "GFDL-ESM4":        "NOAA-GFDL",
    "IPSL-CM6A-LR":     "IPSL",
    "MPI-ESM-1-2-HAM":  "HAMMOZ-Consortium",
    "NorESM2-LM":       "NCC",
    # Models in the wider list that don't have dfe but might be in MODELS
    "ACCESS-ESM1-5":    "CSIRO",
    "CanESM5":          "CCCma",
    "UKESM1-0-LL":      "MOHC",
}

# Ensemble member per model (defaults to r1i1p1f1, override per model)
_ENSEMBLE_OF: dict[str, str] = {
    "UKESM1-0-LL":      "r1i1p1f2",
}

CMIP6_ROOT = Path("/work/ik1017/CMIP6/data/CMIP6/CMIP")


def _find_cmip6_var_dir(model: str, table: str, var: str,
                        ensemble: str | None = None) -> Path | None:
    """Locate the data directory for a (model, table, variable) triple.

    Returns the most recent version subdirectory found under the
    standard CMIP6 layout. Returns None if no directory exists.
    Searches both the typical ensemble (r1i1p1f1 or model-specific
    override) and falls back to whatever single ensemble is present.
    """
    institute = _INSTITUTE_OF.get(model)
    if institute is None:
        return None
    if ensemble is None:
        ensemble = _ENSEMBLE_OF.get(model, "r1i1p1f1")

    base = (CMIP6_ROOT / institute / model / "historical" /
            ensemble / table / var)
    if not base.is_dir():
        # Try any ensemble member
        parent = CMIP6_ROOT / institute / model / "historical"
        if not parent.is_dir():
            return None
        candidates = sorted(parent.glob(f"*/{table}/{var}"))
        if not candidates:
            return None
        base = candidates[0]

    # Iterate over grid labels and version dirs; pick the most recent
    # version inside any grid label.
    version_dirs = sorted(base.glob("*/v*"), key=lambda p: p.name)
    if not version_dirs:
        return None
    return version_dirs[-1]


def load_model_dfe(model: str,
                   z_max: float = 100.0) -> xr.DataArray | None:
    """Load dfe and depth-integrate over 0 to z_max metres.

    Returns mmol Fe m-2 (column integral) as a (time, lat, lon)
    DataArray over 1997-2014.
    """
    var_dir = _find_cmip6_var_dir(model, "Omon", "dfe")
    if var_dir is None:
        logger.warning(f"  [{model}] no dfe data found in CMIP6 archive")
        return None

    paths = sorted(var_dir.glob("dfe_Omon_*.nc"))
    if not paths:
        logger.warning(f"  [{model}] no dfe files in {var_dir}")
        return None

    logger.info(f"  [{model}] loading dfe: {len(paths)} files from {var_dir}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_mfdataset([str(p) for p in paths],
                                combine="by_coords", use_cftime=True)

    da = ds["dfe"].sel(time=slice("1997", "2014"))

    # Identify the depth dimension using the same defensive logic as
    # load_model_chl, since dfe shares the same heterogeneous
    # vertical-axis naming.
    KNOWN_Z_NAMES = {
        "lev", "olevel", "depth", "deptht", "depthu", "depthv",
        "depthw", "lev_2", "z_l", "z_t", "st_ocean",
    }
    z_dim = None
    for d in da.dims:
        if d in KNOWN_Z_NAMES or d.lower().startswith(("lev", "depth", "z_")):
            z_dim = d
            break
    if z_dim is None:
        logger.warning(f"  [{model}] dfe: no vertical dim detected; "
                       f"returning surface only")
        da = da.squeeze()
        # Convert mol/m3 to mmol/m3 to be in same family as integrated
        return standardize(snap_to_month_start(da * 1000.0))

    # Get the depth coordinate values (assumed positive-down in metres
    # for CMIP6 ocean BGC fields). If the units are something else
    # (centimetres for some POP outputs), warn and convert.
    z_coord = da[z_dim]
    z_units = z_coord.attrs.get("units", "m").lower()
    z_vals = z_coord.values.astype(float)
    if z_units in ("cm", "centimeter", "centimetres"):
        z_vals = z_vals / 100.0
        logger.info(f"  [{model}] dfe depth in cm; converting to m")
    elif z_units in ("km",):
        z_vals = z_vals * 1000.0
    # Handle positive-up convention (rare for CMIP6 ocean but possible)
    if z_coord.attrs.get("positive", "down").lower() == "up":
        z_vals = -z_vals

    # Layer thicknesses: midpoint-based, with closed boundaries at 0
    # and at the deepest selected level. This is approximate but
    # adequate at 0-100 m where layers are ~10 m thick in most models.
    keep_mask = z_vals <= z_max
    if not keep_mask.any():
        logger.warning(f"  [{model}] dfe: no levels at or above {z_max} m")
        return None
    keep_idx = np.where(keep_mask)[0]
    z_kept = z_vals[keep_idx]
    # Layer edges: half-way between successive level midpoints; top
    # edge at 0 m, bottom edge at z_max (clipped if last level is less)
    edges = np.empty(len(z_kept) + 1)
    edges[0] = 0.0
    edges[-1] = min(z_max, float(z_kept[-1]) +
                    (float(z_kept[-1]) - float(z_kept[-2])) / 2.0
                    if len(z_kept) > 1 else z_max)
    for i in range(len(z_kept) - 1):
        edges[i + 1] = 0.5 * (z_kept[i] + z_kept[i + 1])
    thick = np.diff(edges)
    if (thick < 0).any():
        logger.warning(f"  [{model}] dfe: negative layer thicknesses; "
                       f"check depth coord ordering")
        return None

    # Subset to kept levels
    da_sub = da.isel({z_dim: keep_idx})
    # Broadcast thickness against the depth dimension
    thick_da = xr.DataArray(thick, coords=[da_sub[z_dim]], dims=[z_dim])
    # Integrate: sum(dfe * dz) over z. Units: mol/m3 * m = mol/m2.
    # Convert to mmol/m2 by multiplying by 1000.
    integrated = (da_sub * thick_da).sum(dim=z_dim, skipna=True)
    integrated = integrated * 1000.0
    integrated.attrs["units"] = "mmol m-2"
    integrated.attrs["long_name"] = (
        f"Dissolved iron, depth-integrated 0-{z_max:g} m"
    )
    integrated = integrated.where(np.abs(integrated) < 1e4)
    integrated = standardize(snap_to_month_start(integrated))
    return integrated




def make_driver_bias_with_obs(
    bias_dict:  dict[str, xr.DataArray],
    obs_clim:   xr.DataArray,
    obs_title:  str,
    obs_cmap:   str,
    obs_label:  str,
    obs_vmin:   float,
    obs_vmax:   float,
    bias_vmin:  float,
    bias_vmax:  float,
    bias_label: str,
    filename:   str,
    stat_fmt:   str = "mean={mean:+.2f}  RMSE={rmse:.2f}",
) -> None:
    """Physical-driver bias figure in Fig. 1 style.

    Layout: 2 columns × ceil((n_models + 1) / 2) rows.
      Panel (a): obs climatology on a sequential colour scale.
      Panels (b)–(h): model − obs bias on a diverging scale.

    Two colourbars are placed below the figure:
      left  → sequential obs scale
      right → diverging bias scale

    Latitude tick labels appear only on the left column.
    The figure is saved to OUT_DIR / filename.

    Parameters
    ----------
    bias_dict  : {model_name: (model_clim − obs_clim)} already on the obs grid.
    obs_clim   : Time-mean obs field on the shared 1° grid.
    obs_title  : Panel (a) title, e.g. "ORAS5 (obs)".
    obs_cmap   : Sequential colourmap for the obs panel, e.g. "Blues_r".
    obs_label  : Colourbar label for the obs panel, e.g. r"MLD (m)".
    obs_vmin/vmax : Colour scale limits for the obs panel.
    bias_vmin/vmax: Colour scale limits for the bias panels (symmetric recommended).
    bias_label : Colourbar label for the bias panels, e.g. r"Bias (m)".
    filename   : Output filename under OUT_DIR.
    stat_fmt   : f-string template receiving {mean} and {rmse} for the inset box.
    """
    logger.info(f"Building driver bias figure (obs + bias, 2col vertical): {filename}")

    n_models = len(bias_dict)
    if n_models == 0:
        logger.warning(f"  no model data; skipping {filename}")
        return

    # ------------------------------------------------------------------
    # Grid: 2 columns, enough rows to hold obs panel + all model panels.
    # ------------------------------------------------------------------
    ncols  = 2
    nrows  = int(np.ceil((n_models + 1) / ncols))   # +1 for the obs panel
    height = 4.0 * nrows                             # ~4 in per row, matches Fig.1

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(10, height),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.03, right=0.97,
        bottom=0.06, top=0.97,
        hspace=0.15, wspace=0.06,
    )
    axes_flat = axes.flatten()

    # ------------------------------------------------------------------
    # Panel (a): obs climatology
    # ------------------------------------------------------------------
    ax0 = axes_flat[0]
    add_panel_label(ax0, 0)

    im_obs = ax0.pcolormesh(
        obs_clim.lon, obs_clim.lat, obs_clim.values,
        vmin=obs_vmin, vmax=obs_vmax, cmap=obs_cmap,
        transform=ccrs.PlateCarree(), shading="auto",
    )
    ax0.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax0.coastlines(linewidth=0.4)
    gl = ax0.gridlines(
        linestyle=":", linewidth=0.3, draw_labels=True,
        alpha=0.5, x_inline=False, y_inline=False,
    )
    gl.top_labels   = False
    gl.right_labels = False
    gl.left_labels  = True
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    ax0.set_title(obs_title, fontsize=10, fontweight="bold", pad=3)

    # ------------------------------------------------------------------
    # Panels (b)–(h): per-model bias maps
    # ------------------------------------------------------------------
    norm_bias = TwoSlopeNorm(vmin=bias_vmin, vcenter=0, vmax=bias_vmax)
    im_bias   = None

    for idx, (model, bias) in enumerate(bias_dict.items()):
        panel_idx = idx + 1                   # slot 0 is the obs panel
        col       = panel_idx % ncols         # 0 = left, 1 = right
        ax        = axes_flat[panel_idx]
        add_panel_label(ax, panel_idx)

        im_bias = ax.pcolormesh(
            obs_clim.lon, obs_clim.lat, bias.values,
            norm=norm_bias, cmap="RdBu_r",
            transform=ccrs.PlateCarree(), shading="auto",
        )
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        ax.coastlines(linewidth=0.4)

        gl = ax.gridlines(
            linestyle=":", linewidth=0.3, draw_labels=True,
            alpha=0.4, x_inline=False, y_inline=False,
        )
        gl.top_labels   = False
        gl.right_labels = False
        gl.left_labels  = (col == 0)   # lat labels only on left column
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}

        ax.set_title(f"Bias: {model} − obs", fontsize=9, fontweight="bold", pad=3)

        v = bias.values[np.isfinite(bias.values)]
        if v.size:
            mean_v = float(np.mean(v))
            rmse_v = float(np.sqrt(np.mean(v ** 2)))
            ax.text(
                0.02, 0.04,
                stat_fmt.format(mean=mean_v, rmse=rmse_v),
                transform=ax.transAxes, fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="black", alpha=0.9),
            )

    # Hide any unused slots (last panel slot if n_models is odd)
    for ax in axes_flat[n_models + 1:]:
        ax.set_visible(False)

    # ------------------------------------------------------------------
    # Two colourbars below the figure (mirrors Fig. 1 layout)
    # ------------------------------------------------------------------
    cbar_y      = 0.010
    cbar_height = 0.015

    ax_cbar_obs = fig.add_axes([0.05, cbar_y, 0.40, cbar_height])
    cbar_obs    = fig.colorbar(
        im_obs, cax=ax_cbar_obs,
        orientation="horizontal", extend="both",
    )
    cbar_obs.set_label(obs_label, fontsize=13)
    cbar_obs.ax.tick_params(labelsize=12)

    if im_bias is not None:
        ax_cbar_bias = fig.add_axes([0.55, cbar_y, 0.40, cbar_height])
        cbar_bias    = fig.colorbar(
            im_bias, cax=ax_cbar_bias,
            orientation="horizontal", extend="both",
        )
        cbar_bias.set_label(bias_label, fontsize=13)
        cbar_bias.ax.tick_params(labelsize=12)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    out_path = OUT_DIR / filename
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.success(f"  wrote {out_path.name}")







# ============================================================
# Surface downwelling shortwave (rsds) loader
# ============================================================

def load_model_rsds(model: str) -> xr.DataArray | None:
    """Load monthly rsds from the CMIP6 archive (Amon, W m-2)."""
    var_dir = _find_cmip6_var_dir(model, "Amon", "rsds")
    if var_dir is None:
        logger.warning(f"  [{model}] no rsds data found in CMIP6 archive")
        return None

    paths = sorted(var_dir.glob("rsds_Amon_*.nc"))
    if not paths:
        logger.warning(f"  [{model}] no rsds files in {var_dir}")
        return None

    logger.info(f"  [{model}] loading rsds: {len(paths)} files from {var_dir}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_mfdataset([str(p) for p in paths],
                                combine="by_coords", use_cftime=True)
    da = ds["rsds"].sel(time=slice("1997", "2014"))
    da = da.squeeze()
    da = da.where((da >= 0) & (da < 600))  # physical sanity
    da = standardize(snap_to_month_start(da))
    da.attrs["units"] = "W m-2"
    return da


# ============================================================
# Observational reference for rsds: ISCCP-FH (surface SW)
# ============================================================
# IMPORTANT: the CERES-EBAF directory on Levante is TOA-ONLY
# (rsut, rsutcs, rlut, rsdt, rtmt, rtnt, ...). It contains NO surface
# downwelling shortwave, so the previous loader globbed "*Amon*.nc",
# picked the last file (rtnt = net TOA radiation), and produced a
# bogus uniform "rsds bias" of ~+189 W m-2 (each model's absolute
# surface SW minus a ~0/TOA reference). ISCCP-FH is a genuine GLOBAL
# surface downwelling-SW product (W m-2). TROPFLUX is tropics-only and
# is deliberately NOT used for a global/polar diagnostic.
RSDS_OBS_DIR  = Path("/work/bd0854/DATA/ESMValTool2/OBS/Tier2/ISCCP-FH")
RSDS_OBS_NAME = "ISCCP-FH"


def load_obs_rsds() -> xr.DataArray | None:
    """Load ISCCP-FH surface downwelling shortwave (rsds, W m-2).

    Refuses anything that is not a plausible surface SW field (the
    area-weighted global mean must be O(100) W m-2) so a wrong file
    can never again masquerade as the light reference. Returns None
    when no usable surface rsds is found, in which case the limitation
    figure falls back to the ensemble-mean reference.
    """
    if not RSDS_OBS_DIR.is_dir():
        logger.warning(f"  rsds obs dir not found: {RSDS_OBS_DIR}")
        return None

    candidates = sorted(RSDS_OBS_DIR.glob("*_rsds_*.nc"))
    if not candidates:
        candidates = sorted(RSDS_OBS_DIR.glob("*rsds*.nc"))
    if not candidates:
        logger.warning(f"  no surface rsds file in {RSDS_OBS_DIR}")
        return None

    fpath = candidates[-1]
    logger.info(f"  {RSDS_OBS_NAME} surface rsds file: {fpath.name}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_dataset(fpath)

    if "rsds" not in ds.data_vars:
        logger.error(f"  {fpath.name} has no 'rsds' (vars={list(ds.data_vars)})")
        return None

    da = ds["rsds"].squeeze()
    if "time" in da.dims:
        da = da.sel(time=slice("1997", "2014"))
    da = standardize(snap_to_month_start(da))

    src_units = (ds["rsds"].attrs.get("units") or "").strip().lower()
    if src_units not in ("w/m2", "w m-2", "w m**-2", "w/m**2", ""):
        logger.warning(f"  rsds units '{src_units}' != W m-2; no conversion")
    da.attrs["units"] = "W m-2"

    # area-weighted sanity check (unweighted mean under-counts: it
    # over-weights the low-insolation poles on a regular lat-lon grid)
    w = np.cos(np.deg2rad(da.lat))
    gm = float(da.mean("time").weighted(w).mean(("lat", "lon")))
    if not (120.0 <= gm <= 240.0):
        logger.error(f"  {RSDS_OBS_NAME} area-wtd global mean {gm:.1f} W m-2 "
                     f"implausible for surface SW; refusing")
        return None
    logger.info(f"  {RSDS_OBS_NAME} area-weighted global-mean rsds = "
                f"{gm:.1f} W m-2")
    return da


# ============================================================
# Regridding
# ============================================================

def regrid_to_obs(model_da: xr.DataArray, obs_da: xr.DataArray) -> xr.DataArray:
    lat_1d = model_da["lat"].ndim == 1 and "lat" in model_da.dims
    lon_1d = model_da["lon"].ndim == 1 and "lon" in model_da.dims
    if lat_1d and lon_1d:
        return model_da.interp(lat=obs_da.lat, lon=obs_da.lon, method="linear")

    from scipy.interpolate import griddata

    src_val = np.asarray(model_da.values)
    spatial_shape = src_val.shape[-2:]
    lat_arr = np.asarray(model_da["lat"].values)
    lon_arr = np.asarray(model_da["lon"].values)

    if lat_arr.ndim == 1:
        if lat_arr.size == spatial_shape[0]:
            lat_arr = np.broadcast_to(lat_arr[:, None], spatial_shape).copy()
        elif lat_arr.size == spatial_shape[1]:
            lat_arr = np.broadcast_to(lat_arr[None, :], spatial_shape).copy()
    if lon_arr.ndim == 1:
        if lon_arr.size == spatial_shape[1]:
            lon_arr = np.broadcast_to(lon_arr[None, :], spatial_shape).copy()
        elif lon_arr.size == spatial_shape[0]:
            lon_arr = np.broadcast_to(lon_arr[:, None], spatial_shape).copy()
    lon_arr = lon_arr % 360

    pts   = np.column_stack([lat_arr.ravel(), lon_arr.ravel()])
    vals  = src_val.ravel()
    valid = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1]) & np.isfinite(vals)
    pts, vals = pts[valid], vals[valid]
    obs_lon2d, obs_lat2d = np.meshgrid(obs_da.lon.values, obs_da.lat.values)
    target = np.column_stack([obs_lat2d.ravel(), obs_lon2d.ravel()])
    out = griddata(pts, vals, target, method="nearest").reshape(obs_lat2d.shape)
    return xr.DataArray(out, dims=("lat", "lon"),
                        coords={"lat": obs_da.lat, "lon": obs_da.lon})


def regrid_timeseries(model_da: xr.DataArray, obs_da: xr.DataArray) -> xr.DataArray:
    """Regrid a (time, lat, lon) field timestep by timestep.
    Returns result with time snapped to month-start datetime64."""
    out_list = []
    for t in model_da.time.values:
        slab = model_da.sel(time=t)
        re = regrid_to_obs(slab, obs_da)
        out_list.append(re.expand_dims(time=[t]))
    result = xr.concat(out_list, dim="time")
    return snap_to_month_start(result)


# ============================================================
# Stats
# ============================================================

def rmse_and_r(model: xr.DataArray, obs: xr.DataArray) -> tuple[float, float]:
    m = model.values.ravel()
    o = obs.values.ravel()
    mask = np.isfinite(m) & np.isfinite(o)
    m, o = m[mask], o[mask]
    if len(m) < 2:
        return np.nan, np.nan
    return (float(np.sqrt(np.mean((m - o) ** 2))),
            float(np.corrcoef(m, o)[0, 1]))


def basin_mean(da: xr.DataArray, lat_min, lat_max, lon_min, lon_max) -> float:
    weights = np.cos(np.deg2rad(da.lat))
    if lon_min <= lon_max:
        sub = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        w   = weights.sel(lat=slice(lat_min, lat_max))
    else:
        sub = xr.concat(
            [da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, 360)),
             da.sel(lat=slice(lat_min, lat_max), lon=slice(0, lon_max))],
            dim="lon",
        )
        w = weights.sel(lat=slice(lat_min, lat_max))
    return float(sub.weighted(w).mean(dim=("lat", "lon")))


# ============================================================
# Fig 1: ESA-CCI Chl climatology
# ============================================================
def make_fig1_and_fig2_combined(
    obs_clim: xr.DataArray,
    model_clim: dict[str, xr.DataArray],
) -> None:
    """
    Single figure combining the ESA-CCI Chl-a climatology (panel a)
    and the per-model bias maps (panels b-h) in a 2-column x 4-row
    vertical layout.

    Panel (a): ESA-CCI time-mean Chl-a on a log colour scale (YlGn).
    Panels (b)-(h): model minus ESA-CCI bias on a diverging scale (RdBu_r).
    The last panel slot is hidden if the number of models is less than 7.

    Two colourbars are placed below the figure:
      - left  colourbar: log-scale Chl-a for panel (a)
      - right colourbar: diverging bias scale for panels (b)-(h)

    Lat labels appear only on the left column.
    No superfluous white space.
    """
    logger.info("Building combined Fig 1+2 (obs + bias maps, 2col x 4row)...")

    nrows, ncols = 4, 2
    n_panels     = nrows * ncols   # 8
    n_models     = len(model_clim) # up to 7

    # ----------------------------------------------------------------
    # Figure geometry — tight, no extra height for colourbar strip
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(10, 4.2 * nrows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.03, right=0.97,
        bottom=0.06, top=0.97,
        hspace=0.15, wspace=0.06,
    )

    axes_flat = axes.flatten()   # [ax_a, ax_b, ax_c, ...]

    # ----------------------------------------------------------------
    # Panel (a): ESA-CCI climatology  —  left column, top row
    # ----------------------------------------------------------------
    ax0      = axes_flat[0]
    add_panel_label(ax0, 0)

    norm_log = LogNorm(vmin=LOG_VMIN, vmax=LOG_VMAX)
    im_obs   = ax0.pcolormesh(
        obs_clim.lon, obs_clim.lat, obs_clim.values,
        norm=norm_log, cmap="YlGn",
        transform=ccrs.PlateCarree(), shading="auto",
    )
    ax0.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax0.coastlines(linewidth=0.4)
    gl = ax0.gridlines(
        linestyle=":", linewidth=0.3, draw_labels=True,
        alpha=0.5, x_inline=False, y_inline=False,
    )
    gl.top_labels   = False
    gl.right_labels = False
    gl.left_labels  = True
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}
    ax0.set_title("ESA-CCI (obs)", fontsize=10, fontweight="bold", pad=3)

    # ----------------------------------------------------------------
    # Panels (b)-(h): per-model bias maps
    # ----------------------------------------------------------------
    norm_bias = TwoSlopeNorm(vmin=-BIAS_LIMIT, vcenter=0, vmax=BIAS_LIMIT)
    im_bias   = None

    for idx, (model, m_re) in enumerate(model_clim.items()):
        panel_idx = idx + 1            # slot in the 8-panel grid
        col       = panel_idx % ncols  # 0 = left column, 1 = right column
        ax        = axes_flat[panel_idx]
        add_panel_label(ax, panel_idx)

        bias    = m_re - obs_clim
        rmse, r = rmse_and_r(m_re, obs_clim)

        im_bias = ax.pcolormesh(
            obs_clim.lon, obs_clim.lat, bias.values,
            norm=norm_bias, cmap="RdBu_r",
            transform=ccrs.PlateCarree(), shading="auto",
        )
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        ax.coastlines(linewidth=0.4)

        gl = ax.gridlines(
            linestyle=":", linewidth=0.3, draw_labels=True,
            alpha=0.4, x_inline=False, y_inline=False,
        )
        gl.top_labels   = False
        gl.right_labels = False
        gl.left_labels  = (col == 0)   # lat labels only on left column
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}

        ax.set_title(model, fontsize=9, fontweight="bold", pad=3)
        ax.text(
            0.02, 0.04,
            f"RMSE={rmse:.3f}  r={r:.3f}",
            transform=ax.transAxes, fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor="black", alpha=0.9),
        )

    # Hide any unused panel slots
    for ax in axes_flat[n_models + 1:]:
        ax.set_visible(False)

    # ----------------------------------------------------------------
    # Two colourbars below the figure
    # ----------------------------------------------------------------
    cbar_y      = 0.010
    cbar_height = 0.015

    ax_cbar_obs = fig.add_axes([0.05, cbar_y, 0.40, cbar_height])
    cbar_obs    = fig.colorbar(
        im_obs, cax=ax_cbar_obs,
        orientation="horizontal", extend="both",
    )
    cbar_obs.set_label(CHL_LABEL, fontsize=13)
    cbar_obs.ax.tick_params(labelsize=12)

    if im_bias is not None:
        ax_cbar_bias = fig.add_axes([0.55, cbar_y, 0.40, cbar_height])
        cbar_bias    = fig.colorbar(
            im_bias, cax=ax_cbar_bias,
            orientation="horizontal", extend="both",
        )
        cbar_bias.set_label(f"Bias ({CHL_LABEL})", fontsize=13)
        cbar_bias.ax.tick_params(labelsize=12)

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    out_path = OUT_DIR / "fig1_2_obs_and_bias_combined.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.success(f"  wrote {out_path.name}")

# ============================================================
# Panel-label helper — bold AGU-style (a), (b), ... tags
# ============================================================

def add_panel_label(ax, index: int, x: float = 0.02, y: float = 0.98,
                     fontsize: int = 12) -> None:
    """Stamp a bold AGU-style panel label, e.g. (a), in the upper-left
    corner inside the axes. `index` is 0-based: 0 -> (a), 1 -> (b)...
    Works for ordinary Axes and Cartopy GeoAxes (transAxes is
    projection-independent). A faint white box keeps the letter
    legible over busy pcolormesh or line content."""
    letter = string.ascii_lowercase[index % 26]
    ax.text(x, y, f"({letter})",
            transform=ax.transAxes,
            fontsize=fontsize, fontweight="bold",
            va="top", ha="left", zorder=20,
            bbox=dict(boxstyle="round,pad=0.2",
                      facecolor="white", edgecolor="none", alpha=0.7))



# ============================================================
# Fig 3: basin-mean bias bars
# ============================================================

def make_fig3_hovmoller(
    obs_full: xr.DataArray,
    model_seasonal: dict[str, xr.DataArray],
    obs_clim: xr.DataArray,
) -> None:
    """
    Fig 3 (revised): Hovmöller diagrams of Chl-a seasonal anomalies.

    Layout: 2 columns x 4 rows = 8 panels.
      Panel (a): ESA-CCI observed Chl-a anomaly (zonal mean, lat x month)
      Panels (b)-(h): per-model Chl-a anomaly minus ESA-CCI anomaly
                      (zonal mean, lat x month)

    The anomaly is defined as the deviation of the monthly climatology
    from the annual mean:
        A(lat, month) = zonal_mean_clim(lat, month)
                        - zonal_mean_annual_mean(lat)

    For the model bias panels:
        delta_A(lat, month) = A_model(lat, month) - A_obs(lat, month)

    This reveals where and when in the seasonal cycle models diverge
    from observations, separately from any mean-state bias.

    Parameters
    ----------
    obs_full : xr.DataArray
        Full monthly ESA-CCI time series (time, lat, lon).
    model_seasonal : dict[str, xr.DataArray]
        Per-model monthly climatology (month, lat, lon) already
        regridded to the obs grid. Built in the main pipeline loop.
    obs_clim : xr.DataArray
        ESA-CCI annual-mean climatology (lat, lon), used only to
        provide the lat coordinate for the obs seasonal cycle.
    """
    logger.info("Building Fig 3 Hovmöller (seasonal Chl-a anomalies)...")

    models   = list(model_seasonal.keys())
    n_models = len(models)
    nrows, ncols = 4, 2

    # ----------------------------------------------------------------
    # Build obs monthly climatology: group by month, take zonal mean
    # ----------------------------------------------------------------
    # obs_full: (time, lat, lon)  -> group by calendar month
    obs_monthly_clim = (
        obs_full
        .groupby("time.month")
        .mean("time", keep_attrs=True)
    )   # shape: (12, lat, lon)
    obs_monthly_clim = obs_monthly_clim.sortby("lat")
    # Zonal mean (lon average, NaN-aware)
    obs_zonal = obs_monthly_clim.mean("lon", skipna=True)   # (12, lat)

    # Annual mean of the zonal climatology
    obs_annual = obs_zonal.mean("month", skipna=True)       # (lat,)

    # Anomaly: deviation from annual mean
    obs_anom = obs_zonal - obs_annual                       # (12, lat)
    # Transpose to (lat, month) for pcolormesh
    obs_anom = obs_anom.transpose("lat", "month")

    # ----------------------------------------------------------------
    # Build per-model zonal anomalies
    # ----------------------------------------------------------------
    model_anom: dict[str, xr.DataArray] = {}
    for model, m_seas in model_seasonal.items():
        # m_seas: (month, lat, lon) — already on obs grid
        m_zonal  = m_seas.mean("lon", skipna=True)          # (12, lat)
        m_annual = m_zonal.mean("month", skipna=True)       # (lat,)
        m_anom   = (m_zonal - m_annual).transpose("lat", "month")
        model_anom[model] = m_anom

    # ----------------------------------------------------------------
    # Colour scale limits
    # ----------------------------------------------------------------
    # Obs anomaly: symmetric around zero, use 95th percentile of abs
    obs_vals   = obs_anom.values[np.isfinite(obs_anom.values)]
    obs_vlim   = float(np.nanpercentile(np.abs(obs_vals), 95))
    obs_vlim   = max(obs_vlim, 0.05)   # floor to avoid flat colourbar

    # Bias panels: smaller range — difference of anomalies
    all_diff_vals = []
    for model in models:
        diff = model_anom[model] - obs_anom
        all_diff_vals.append(diff.values[np.isfinite(diff.values)])
    if all_diff_vals:
        all_diff_vals = np.concatenate(all_diff_vals)
        bias_vlim = float(np.nanpercentile(np.abs(all_diff_vals), 95))
        bias_vlim = max(bias_vlim, 0.02)
    else:
        bias_vlim = 0.2

    # ----------------------------------------------------------------
    # Figure layout
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(12, 4.0 * nrows),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.07, right=0.93,
        bottom=0.08, top=0.97,
        hspace=0.30, wspace=0.25,
    )
    axes_flat = axes.flatten()

    # Shared x-axis: months 1-12
    month_labels = ["J", "F", "M", "A", "M", "J",
                    "J", "A", "S", "O", "N", "D"]
    month_ticks  = np.arange(1, 13)

    # ----------------------------------------------------------------
    # Panel (a): ESA-CCI observed anomaly
    # ----------------------------------------------------------------
    ax0  = axes_flat[0]
    add_panel_label(ax0, 0)

    lats   = obs_anom.lat.values
    months = obs_anom.month.values

    im_obs = ax0.pcolormesh(
        months, lats, obs_anom.values,
        cmap="RdBu_r",
        vmin=-obs_vlim, vmax=obs_vlim,
        shading="auto",
    )
    ax0.set_title("ESA-CCI (obs anomaly)", fontsize=10,
                  fontweight="bold")
    ax0.set_ylabel("Latitude (°)", fontsize=9)
    ax0.set_xticks(month_ticks)
    ax0.set_xticklabels(month_labels, fontsize=8)
    ax0.set_yticks([-60, -30, 0, 30, 60])
    ax0.axhline(0, color="k", linewidth=0.5, linestyle=":")
    ax0.tick_params(labelsize=14)

    # Colourbar for obs panel
    cb_obs = fig.colorbar(
        im_obs, ax=ax0,
        orientation="vertical", pad=0.02,
        fraction=0.046, extend="both",
    )
    cb_obs.set_label(CHL_LABEL, fontsize=12)
    cb_obs.ax.tick_params(labelsize=14)

    # ----------------------------------------------------------------
    # Panels (b)-(h): per-model anomaly bias (model - obs)
    # ----------------------------------------------------------------
    im_bias_last = None
    for idx, model in enumerate(models):
        ax  = axes_flat[idx + 1]
        add_panel_label(ax, idx + 1)

        diff = model_anom[model] - obs_anom   # (lat, month)

        im_bias_last = ax.pcolormesh(
            months, lats, diff.values,
            cmap="RdBu_r",
            vmin=-bias_vlim, vmax=bias_vlim,
            shading="auto",
        )
        ax.set_title(model, fontsize=9, fontweight="bold")
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels, fontsize=9)
        ax.set_yticks([-60, -30, 0, 30, 60])
        ax.axhline(0, color="k", linewidth=0.5, linestyle=":")
        ax.tick_params(labelsize=14)

        # y-label only on left column
        col = (idx + 1) % ncols
        if col == 0:
            ax.set_ylabel("Latitude (°)", fontsize=9)

        # Colourbar on each panel
        cb = fig.colorbar(
            im_bias_last, ax=ax,
            orientation="vertical", pad=0.02,
            fraction=0.046, extend="both",
        )
        cb.set_label(f"Δ {CHL_LABEL}", fontsize=12)
        cb.ax.tick_params(labelsize=14)

    # Hide unused panels
    for ax in axes_flat[n_models + 1:]:
        ax.set_visible(False)

    # ----------------------------------------------------------------
    # Shared x-axis label on bottom row
    # ----------------------------------------------------------------
    for ax in axes_flat[nrows * ncols - ncols: nrows * ncols]:
        if ax.get_visible():
            ax.set_xlabel("Month", fontsize=12)

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    out_path = OUT_DIR / "fig3_hovmoller_chl_anomaly.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    logger.success(f"  wrote {out_path.name}")

# ============================================================
# Fig 4: seasonal cycle
# ============================================================

def make_fig4(obs_da: xr.DataArray,
              model_seasonal: dict[str, xr.DataArray]) -> None:
    """Fig 4: mean seasonal cycle of surface Chl-a, decomposed into
    8 physically motivated regions (Global + Arctic + Southern Ocean
    + North Atlantic + North Pacific + Indian + Tropical Pacific +
    South Pacific). See SEASONAL_REGIONS at the top of this file
    for the exact lat/lon bounds.

    The motivation for using these regions rather than the six
    standard ocean basins is to expose the ITCZ-controlled tropical
    regimes separately from the mid-latitude and polar regimes. A
    model with the right ITCZ position should show a tropical-Pacific
    Chl-a seasonal cycle that matches ESA-CCI; a model with a
    misplaced or doubled ITCZ should not. The Southern Ocean and
    Arctic panels carry the polar bloom signal; North Atlantic and
    North Pacific carry the subpolar mid-latitude bloom.

    Within each region the seasonal cycle is the cos(lat)-weighted
    mean of the monthly climatology over cells where data exist
    (NaN-aware, so ESA-CCI Arctic months without coverage are simply
    excluded).
    """
    logger.info("Building Fig 4 (seasonal cycle, 8-region decomposition)...")
    n_m = len(model_seasonal)

    regions = list(SEASONAL_REGION_ORDER)
    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 8),
                              sharex=True, squeeze=False)
    axes_flat = axes.flatten()

    # For larger ensembles use thinner model lines and lower marker
    # transparency, so the obs (black, bold) remains the eye anchor.
    model_lw    = 1.4 if n_m <= 4 else 1.1
    model_alpha = 0.85 if n_m <= 4 else 0.7

    # Helper: region-mean monthly climatology of a DataArray with
    # either a `time` dim (full monthly series) or a `month` dim
    # (already a climatology). NaN-aware area average using cos(lat)
    # weights.
    def _region_seasonal(da: xr.DataArray, region: str) -> xr.DataArray:
        lat_min, lat_max, lon_min, lon_max = SEASONAL_REGIONS[region]
        if lon_min <= lon_max:
            sub = da.sel(lat=slice(lat_min, lat_max),
                         lon=slice(lon_min, lon_max))
        else:
            sub = xr.concat(
                [da.sel(lat=slice(lat_min, lat_max),
                        lon=slice(lon_min, 360)),
                 da.sel(lat=slice(lat_min, lat_max),
                        lon=slice(0, lon_max))],
                dim="lon",
            )
        w = np.cos(np.deg2rad(sub.lat))
        if "time" in sub.dims:
            sc = sub.groupby("time.month").mean("time")
        else:
            sc = sub
        return sc.weighted(w).mean(dim=("lat", "lon"))

    handles_for_legend = None
    labels_for_legend  = None

    for idx, region in enumerate(regions):
        ax = axes_flat[idx]
        add_panel_label(ax, idx)

        # --- OBS ---
        try:
            obs_r = _region_seasonal(obs_da, region)
            ax.plot(obs_r.month, obs_r.values, color="black",
                    linewidth=2.4, marker="o", label="ESA CCI-OC",
                    zorder=10)
        except Exception as e:
            logger.warning(f"  [{region:17s}] OBS seasonal cycle failed: {e}")

        # --- models ---
        for i, (model, mda) in enumerate(model_seasonal.items()):
            try:
                m_r = _region_seasonal(mda, region)
                ax.plot(m_r.month, m_r.values,
                        color=PALETTE[i % len(PALETTE)],
                        linewidth=model_lw, marker="o", markersize=4,
                        alpha=model_alpha, label=model)
            except Exception as e:
                logger.warning(
                    f"  [{region:17s}] [{model}] seasonal cycle failed: {e}"
                )

        ax.set_title(region, fontsize=11, fontweight="bold")
        ax.set_xticks(range(1, 13))
        ax.grid(alpha=0.3)

        # y-label only on leftmost column; x-label only on bottom row.
        if idx % ncols == 0:
            ax.set_ylabel(CHL_LABEL)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Month")

        # Capture legend handles from the first panel only.
        if idx == 0:
            handles_for_legend, labels_for_legend = \
                ax.get_legend_handles_labels()

    # Shared legend at the bottom; ncol scales with ensemble size.
    if handles_for_legend is not None:
        n_entries = len(handles_for_legend)
        ncol = min(n_entries, 4 if n_m <= 4 else 5)
        fig.legend(handles_for_legend, labels_for_legend,
                   loc="lower center", ncol=ncol, fontsize=9,
                   frameon=True, bbox_to_anchor=(0.5, -0.02))

    # No fig-level suptitle per the manuscript convention.
    fig.tight_layout(rect=[0, 0.04, 1, 1.0])
    fig.savefig(OUT_DIR / "fig4_seasonal_cycle.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote fig4_seasonal_cycle.png")


# ============================================================
# Generic 2x2 bias map (used for Figs 5, 6, 7)
# ============================================================

def make_bias_map_2x2(
    bias_dict: dict[str, xr.DataArray],
    obs_grid: xr.DataArray,
    vmin: float,
    vmax: float,
    unit_label: str,
    cmap: str,
    fig_num: int,
    title: str,
    filename: str,
    stat_fmt: str = "mean={mean:.2f}  RMSE={rmse:.2f}",
) -> None:
    """Generic bias map figure. Grid adapts to number of models:
    2x2 for <=4, 2x4 for 5-8, 3x4 for 9+."""
    logger.info(f"Building Fig {fig_num} ({title})...")
    n = len(bias_dict)
    if n == 0:
        logger.warning(f"  no data; skipping Fig {fig_num}")
        return
    if n <= 4:
        ncols, nrows = 2, max(1, int(np.ceil(n / 2)))
        figsize = (10, 4 * nrows)
    elif n <= 8:
        ncols, nrows = 4, int(np.ceil(n / 4))
        figsize = (18, 4 * nrows)
    else:
        ncols, nrows = 4, int(np.ceil(n / 4))
        figsize = (18, 3.5 * nrows)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False,
    )
    axes_flat = axes.flatten()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    last_im = None

    for _pidx, (ax, (model, bias)) in enumerate(zip(axes_flat, bias_dict.items())):
        add_panel_label(ax, _pidx)
        im = ax.pcolormesh(
            obs_grid.lon, obs_grid.lat, bias.values,
            norm=norm, cmap=cmap,
            transform=ccrs.PlateCarree(), shading="auto",
        )
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        ax.coastlines(linewidth=0.4)
        ax.gridlines(linestyle=":", linewidth=0.3, draw_labels=True,
                     alpha=0.4, x_inline=False, y_inline=False,
                     xlabel_style={"size": 7}, ylabel_style={"size": 7})
        ax.set_title(f"Bias: {model} − obs", fontsize=10)
        v = bias.values[np.isfinite(bias.values)]
        if v.size:
            mean = float(np.mean(v))
            rmse = float(np.sqrt(np.mean(v ** 2)))
            ax.text(0.02, 0.04,
                    stat_fmt.format(mean=mean, rmse=rmse),
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", edgecolor="black", alpha=0.9))
        last_im = im

    for ax in axes_flat[n:]:
        ax.axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, orientation="horizontal",
                            fraction=0.04, pad=0.07, extend="both")
        cbar.set_label(f"Bias [{unit_label}]")

    #fig.suptitle(f"Fig {fig_num} — {title}\nPeriod: {PERIOD}",
    #             fontsize=13, y=0.99)
    fig.savefig(OUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {filename}")


def make_polar_bias_map(
    bias_dict: dict[str, xr.DataArray],
    obs_grid: xr.DataArray,
    pole: str,
    vmin: float,
    vmax: float,
    unit_label: str,
    cmap: str,
    title: str,
    filename: str,
    stat_fmt: str = "mean={mean:+.1f}  RMSE={rmse:.1f}",
) -> None:
    """Multi-model bias map on a polar-stereographic projection.

    pole = "north" -> NorthPolarStereo, cap 50-90 N (Arctic).
    pole = "south" -> SouthPolarStereo, cap 50-90 S (Southern Ocean /
    seasonal-ice zone).

    Same per-model grid logic as make_bias_map_2x2, but each panel is a
    circular polar cap and the mean/RMSE inset is computed over the cells
    inside the cap only, so the statistic is not diluted by the ice-free
    rest of the ocean.
    """
    import matplotlib.path as mpath
    from cartopy.util import add_cyclic_point

    logger.info(f"Building polar bias map ({title})...")
    n = len(bias_dict)
    if n == 0:
        logger.warning(f"  no data; skipping {filename}")
        return

    if pole == "north":
        proj = ccrs.NorthPolarStereo()
        extent = [-180, 180, 50, 90]
        cap_sel = obs_grid.lat.values >= 50
    else:
        proj = ccrs.SouthPolarStereo()
        extent = [-180, 180, -90, -50]
        cap_sel = obs_grid.lat.values <= -50

    ncols = 4 if n > 4 else 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 3.9 * nrows),
        subplot_kw={"projection": proj}, squeeze=False,
    )
    axes_flat = axes.flatten()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Circular boundary so each panel is a clean polar cap.
    theta = np.linspace(0, 2 * np.pi, 200)
    circle = mpath.Path(
        np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + [0.5, 0.5])

    lat = obs_grid.lat.values
    lon = obs_grid.lon.values
    last_im = None

    for _pidx, (ax, (model, bias)) in enumerate(zip(axes_flat,
                                                    bias_dict.items())):
        add_panel_label(ax, _pidx)
        ax.set_extent(extent, ccrs.PlateCarree())
        ax.set_boundary(circle, transform=ax.transAxes)

        # Close the longitude seam so there is no wedge gap at 0 deg.
        data_c, lon_c = add_cyclic_point(bias.values, coord=lon)
        im = ax.pcolormesh(
            lon_c, lat, data_c, norm=norm, cmap=cmap,
            transform=ccrs.PlateCarree(), shading="auto",
        )
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        ax.coastlines(linewidth=0.4)
        ax.gridlines(linestyle=":", linewidth=0.3, alpha=0.4)
        ax.set_title(f"Bias: {model} − obs", fontsize=10)

        cap = bias.values[cap_sel, :]
        v = cap[np.isfinite(cap)]
        if v.size:
            mean = float(np.mean(v))
            rmse = float(np.sqrt(np.mean(v ** 2)))
            ax.text(0.02, 0.02, stat_fmt.format(mean=mean, rmse=rmse),
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", edgecolor="black", alpha=0.9))
        last_im = im

    for ax in axes_flat[n:]:
        ax.axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, orientation="horizontal",
                            fraction=0.04, pad=0.07, extend="both")
        cbar.set_label(f"Bias [{unit_label}]")

    # No fig-level suptitle per the manuscript convention.
    #fig.suptitle(f"{title}\nPeriod: {PERIOD}", fontsize=12, y=0.99)
    fig.savefig(OUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {filename}")


# ============================================================
# Fig 8: Hovmöller r(MLD_anomaly, Chl_bias) — lat × month  2×2
# ============================================================

def compute_hovmoller_mld_chl_correlation(
    mld_ts: xr.DataArray,
    chl_model_ts: xr.DataArray,
    obs_chl_ts: xr.DataArray,
) -> xr.DataArray:
    """Zonal-mean Pearson r(MLD_anomaly, Chl_bias) as function of (lat, month).

    For each (latitude band, month-of-year):
      1. Average MLD and Chl zonally
      2. Remove monthly climatology to get MLD anomaly
      3. Correlate MLD_anomaly vs Chl_bias over the inter-annual dimension

    Returns (month, lat) DataArray of Pearson r.
    """
    mld_ts, chl_model_ts, obs_chl_ts = xr.align(
        mld_ts, chl_model_ts, obs_chl_ts, join="inner"
    )

    mld_z   = mld_ts.mean("lon", skipna=True)
    chl_m_z = chl_model_ts.mean("lon", skipna=True)
    obs_z   = obs_chl_ts.mean("lon", skipna=True)

    mld_clim = mld_z.groupby("time.month").mean("time")
    mld_anom = mld_z.groupby("time.month") - mld_clim

    chl_bias_z = chl_m_z - obs_z

    nlat = mld_z.sizes["lat"]
    r_out = np.full((12, nlat), np.nan)

    for m in range(1, 13):
        mld_m  = mld_anom.sel(time=mld_anom["time.month"] == m).values
        bias_m = chl_bias_z.sel(time=chl_bias_z["time.month"] == m).values
        for j in range(nlat):
            x = mld_m[:, j]
            y = bias_m[:, j]
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 3:
                continue
            r_out[m - 1, j] = float(np.corrcoef(x[ok], y[ok])[0, 1])

    return xr.DataArray(
        r_out,
        dims=("month", "lat"),
        coords={"month": np.arange(1, 13), "lat": mld_z.lat},
    )


MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def make_fig8(hov_maps: dict[str, xr.DataArray]) -> None:
    """2×2 Hovmöller diagrams: latitude × month of r(MLD_anom, Chl_bias)."""
    logger.info("Building Fig 8 (Hovmöller MLD-Chl coupling, 2x2)...")
    n = len(hov_maps)
    if n == 0:
        logger.warning("  no data; skipping Fig 8")
        return

    ncols = 2
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows),
                              squeeze=False)
    axes_flat = axes.flatten()

    vmax = 0.5
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    months = np.arange(1, 13)
    last_im = None

    for ax, (model, hov) in zip(axes_flat, hov_maps.items()):
        im = ax.pcolormesh(
            hov.month.values, hov.lat.values, hov.values.T,
            norm=norm, cmap="RdBu_r", shading="auto",
        )
        ax.set_xticks(months)
        ax.set_xticklabels(MONTH_LABELS, fontsize=9, rotation=45, ha="right")
        ax.set_yticks(range(-90, 91, 30))
        ax.set_ylabel("Latitude [°]", fontsize=10)
        ax.set_xlabel("Month", fontsize=10)
        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.4)
        ax.axhline(40,  color="k", lw=0.4, ls=":", alpha=0.3)
        ax.axhline(-40, color="k", lw=0.4, ls=":", alpha=0.3)
        ax.grid(alpha=0.15)
        last_im = im

    for ax in axes_flat[n:]:
        ax.axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes,
                            orientation="horizontal",
                            fraction=0.04, pad=0.12)
        cbar.set_label("Pearson r  [MLD anomaly vs Chl-a bias]", fontsize=11)

    #fig.suptitle(
    #    f"Fig 8 — Hovmöller: r(MLD anomaly, model−CCI Chl bias)\n"
    #    f"Latitude × Month  |  Period: {PERIOD}\n"
    #    r"Red: deep MLD → Chl overestimate  |  "
    #    r"Blue: deep MLD → Chl underestimate",
    #    fontsize=11, y=1.02,
    #)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig8_hovmoller_mld_chl.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote fig8_hovmoller_mld_chl.png")


# ============================================================
# Fig 8b: Spatial pattern correlation r(Chl-a, MLD) and r(Chl-a, SST)
# at the basin level, time-mean fields.
#
# For each basin and each (Chl, driver) pair, this is the
# area-weighted Pearson correlation between two 2D fields: the
# time-mean Chl-a and the time-mean physical driver. It measures
# whether high-Chl regions inside the basin tend to co-locate with
# regions of deeper MLD (panel a) or warmer SST (panel b).
#
# OBS bars: ESA-CCI Chl-a vs ORAS5 MLD (panel a) or NOAA-ERSSTv3b
# SST (panel b) -- two independent observational products per panel.
# Model bars: each model's own Chl-a vs its own MLD / SST
# (self-correlation), so any structural disagreement with OBS is a
# property of the model and not of cross-product noise.
#
# This complements the lag-correlation diagnostic of Section 3.6 by
# answering a structurally different question: do the time-mean
# spatial patterns covary as they do in observations? A model that
# matches OBS in the lag-correlation panels but not here, or vice
# versa, has gotten one aspect of the Chl/driver coupling right and
# the other wrong.
# ============================================================

def _area_weighted_pattern_corr(
    field_a: xr.DataArray,
    field_b: xr.DataArray,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> float:
    """Area-weighted spatial Pearson correlation between two 2D fields
    inside a basin box. NaN cells (in either field) are excluded.

    Returns nan if no overlapping valid cells exist.
    """
    # cos(lat) weights, broadcast over lon
    w = np.cos(np.deg2rad(field_a.lat))
    if lon_min <= lon_max:
        a = field_a.sel(lat=slice(lat_min, lat_max),
                        lon=slice(lon_min, lon_max))
        b = field_b.sel(lat=slice(lat_min, lat_max),
                        lon=slice(lon_min, lon_max))
        ww = w.sel(lat=slice(lat_min, lat_max))
    else:
        a = xr.concat(
            [field_a.sel(lat=slice(lat_min, lat_max),
                         lon=slice(lon_min, 360)),
             field_a.sel(lat=slice(lat_min, lat_max),
                         lon=slice(0, lon_max))],
            dim="lon",
        )
        b = xr.concat(
            [field_b.sel(lat=slice(lat_min, lat_max),
                         lon=slice(lon_min, 360)),
             field_b.sel(lat=slice(lat_min, lat_max),
                         lon=slice(0, lon_max))],
            dim="lon",
        )
        ww = w.sel(lat=slice(lat_min, lat_max))

    av = a.values
    bv = b.values
    ww_arr = np.broadcast_to(ww.values[:, None], av.shape)

    # Common valid mask
    valid = np.isfinite(av) & np.isfinite(bv) & np.isfinite(ww_arr)
    if not valid.any():
        return float("nan")

    wv = ww_arr[valid]
    avv = av[valid]
    bvv = bv[valid]

    wsum = wv.sum()
    a_mean = (wv * avv).sum() / wsum
    b_mean = (wv * bvv).sum() / wsum
    a_anom = avv - a_mean
    b_anom = bvv - b_mean

    cov = (wv * a_anom * b_anom).sum() / wsum
    var_a = (wv * a_anom**2).sum() / wsum
    var_b = (wv * b_anom**2).sum() / wsum
    denom = np.sqrt(var_a * var_b)
    if denom == 0 or not np.isfinite(denom):
        return float("nan")
    return float(cov / denom)


def make_fig8b_pattern_corr(
    obs_chl_clim:   xr.DataArray,
    obs_mld_clim:   xr.DataArray | None,
    obs_sst_clim:   xr.DataArray | None,
    model_chl_clim: dict[str, xr.DataArray],
    model_mld_clim: dict[str, xr.DataArray],
    model_sst_clim: dict[str, xr.DataArray],
) -> None:
    """Two-panel basin-mean spatial pattern correlation:
      (a) r(Chl-a, MLD) -- OBS = ESA-CCI vs ORAS5
      (b) r(Chl-a, SST) -- OBS = ESA-CCI vs NOAA-ERSSTv3b

    Each panel is a grouped bar chart with one group per MEANSTATE
    basin and one bar per source (OBS first, then models). No
    figure-level title (per the manuscript convention for Fig 8).
    """
    logger.info("Building Fig 8b (basin-mean pattern correlations)...")

    basins = list(MEANSTATE_BASINS)
    sources_a: list[tuple[str, str]] = []  # (label, color)
    sources_b: list[tuple[str, str]] = []

    # ---- compute r(Chl, MLD) per (source, basin) -----------------------
    table_mld: dict[str, dict[str, float]] = {b: {} for b in basins}
    if obs_mld_clim is not None:
        # Align obs MLD to obs Chl grid (Chl is the master grid here)
        obs_mld_re = obs_mld_clim.interp(
            lat=obs_chl_clim.lat, lon=obs_chl_clim.lon,
            kwargs={"fill_value": np.nan},
        )
        for basin in basins:
            box = BASINS[basin]
            table_mld[basin]["OBS"] = _area_weighted_pattern_corr(
                obs_chl_clim, obs_mld_re, *box)
        sources_a.append(("OBS (ESA-CCI vs ORAS5)", "black"))
    for i, model in enumerate(MODELS):
        if model not in model_chl_clim or model not in model_mld_clim:
            continue
        m_chl = model_chl_clim[model]
        m_mld = model_mld_clim[model].interp(
            lat=m_chl.lat, lon=m_chl.lon, kwargs={"fill_value": np.nan},
        )
        for basin in basins:
            box = BASINS[basin]
            table_mld[basin][model] = _area_weighted_pattern_corr(
                m_chl, m_mld, *box)
        sources_a.append((model, PALETTE[i % len(PALETTE)]))

    # ---- compute r(Chl, SST) per (source, basin) -----------------------
    table_sst: dict[str, dict[str, float]] = {b: {} for b in basins}
    if obs_sst_clim is not None:
        obs_sst_re = obs_sst_clim.interp(
            lat=obs_chl_clim.lat, lon=obs_chl_clim.lon,
            kwargs={"fill_value": np.nan},
        )
        for basin in basins:
            box = BASINS[basin]
            table_sst[basin]["OBS"] = _area_weighted_pattern_corr(
                obs_chl_clim, obs_sst_re, *box)
        sources_b.append(("OBS (ESA-CCI vs ERSSTv3b)", "black"))
    for i, model in enumerate(MODELS):
        if model not in model_chl_clim or model not in model_sst_clim:
            continue
        m_chl = model_chl_clim[model]
        m_sst = model_sst_clim[model].interp(
            lat=m_chl.lat, lon=m_chl.lon, kwargs={"fill_value": np.nan},
        )
        for basin in basins:
            box = BASINS[basin]
            table_sst[basin][model] = _area_weighted_pattern_corr(
                m_chl, m_sst, *box)
        sources_b.append((model, PALETTE[i % len(PALETTE)]))

    # ---- Plot ----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    x = np.arange(len(basins))

    def _draw_panel(ax, table, sources, panel_title):
        n_src = len(sources)
        if n_src == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            return
        width = 0.8 / n_src
        for i, (label, color) in enumerate(sources):
            # Internal key in table is "OBS" for the obs entry, model
            # name for models. Recover it from the label.
            key = "OBS" if label.startswith("OBS") else label
            vals = [table[b].get(key, np.nan) for b in basins]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                   color=color, label=label, edgecolor="white",
                   linewidth=0.4)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(basins, rotation=20)
        ax.set_ylim(-1, 1)
        ax.grid(axis="y", alpha=0.3)
        #ax.set_title(panel_title, fontsize=11, fontweight="bold")
        # Per-panel legend removed; a single shared key is drawn below
        # the panels (see fig.legend after both panels are built).

    add_panel_label(ax1, 0)
    add_panel_label(ax2, 1)
    ax1.set_ylabel("Area-weighted Pearson r")
    _draw_panel(ax1, table_mld, sources_a, "Pattern corr.  r(Chl-a, MLD)")
    _draw_panel(ax2, table_sst, sources_b, "Pattern corr.  r(Chl-a, SST)")

    # Single shared key below both panels (replaces the old per-panel
    # legends). Merge entries from both panels and de-duplicate by label,
    # so both OBS references (ORAS5 for MLD, ERSSTv3b for SST) and the
    # model bars each appear once.
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    merged: dict[str, object] = {}
    for h, l in zip(h1 + h2, l1 + l2):
        if l not in merged:
            merged[l] = h
    fig.legend(list(merged.values()), list(merged.keys()),
               loc="lower center", ncol=4, fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    # No fig.suptitle per the manuscript convention.
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out_path = OUT_DIR / "fig8b_pattern_corr.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {out_path.name}")


# ============================================================
# Fig 9 & Fig 10: lag correlation of basin-mean anomalies
#   Fig  9 — Chl-a vs SST  (ref: ESA-CCI vs NOAA-ERSSTv3b)
#   Fig 10 — Chl-a vs MLD  (ref: ESA-CCI vs ORAS5)
#
# Per (source, basin):
#   1. Regrid both fields to a common 1° grid.
#   2. Mask to basin, take cos(lat)-weighted basin-mean monthly
#      time series for each variable.
#   3. Deseasonalise (subtract monthly climatology) and z-score
#      (divide by std).
#   4. Compute Pearson r(lag) for lag ∈ [-12, +12] months.
#      Convention: lag > 0 means the driver leads Chl, i.e. r(k)
#      is the correlation between driver(t-k) and chl(t).
# ============================================================

LAG_MAX = 12   # months
LAG_PERIOD = slice("1998", "2014")   # clean full-year window


def _build_masks(ref_grid: xr.DataArray) -> dict[str, xr.DataArray]:
    """Build ocean-basin masks on the reference grid.

    Basin definitions follow the BASINS dict at the top of the file —
    the same boxes used by the basin-mean bias chart (Fig 3) — so that
    the basins in Figs 9 and 10 are directly comparable to those in
    Fig 3.
    """
    lat = ref_grid.lat
    lon = ref_grid.lon
    lat2d, lon2d = xr.broadcast(lat, lon)

    masks: dict[str, xr.DataArray] = {}
    for basin, (lat_min, lat_max, lon_min, lon_max) in BASINS.items():
        in_lat = (lat2d >= lat_min) & (lat2d <= lat_max)
        if lon_min <= lon_max:
            in_lon = (lon2d >= lon_min) & (lon2d <= lon_max)
        else:
            in_lon = (lon2d >= lon_min) | (lon2d <= lon_max)
        masks[basin] = in_lat & in_lon
    return masks


def _basin_mean_timeseries(
    da: xr.DataArray,
    mask: xr.DataArray,
) -> xr.DataArray:
    """cos(lat)-weighted basin-mean monthly time series.

    Input must have dims (time, lat, lon) on the same grid as `mask`.
    Returns a 1-D DataArray indexed by time.
    """
    da = da.where(mask)
    w  = np.cos(np.deg2rad(da.lat)) * xr.ones_like(da.lon)
    w  = w.where(mask)
    return da.weighted(w.fillna(0)).mean(dim=("lat", "lon"), skipna=True)


def _detrend_linear(anom: xr.DataArray) -> xr.DataArray:
    """Remove a least-squares straight line from a 1-D time series.

    A least-squares straight line is fitted to the (already
    deseasonalised) anomaly series and subtracted. NaNs are ignored
    in the fit and preserved in the output. This implements the
    linear detrending described in Section 2.6 of the manuscript:
    over the 1998-2014 polar window both siconc and SST carry
    pronounced trends, and a shared trend in two series inflates
    their lag correlation while a trend in a single series inflates
    its autocorrelation, in both cases independently of any genuine
    interannual coupling.
    """
    v = np.asarray(anom.values, dtype=float)
    t = np.arange(v.size, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 24:                 # too few points to fit a trend
        return anom
    slope, intercept = np.polyfit(t[ok], v[ok], deg=1)
    resid = v - (slope * t + intercept)
    resid[~ok] = np.nan
    return anom.copy(data=resid)


def _deseasonalise_zscore(ts: xr.DataArray) -> xr.DataArray:
    """Deseasonalise, LINEARLY DETREND, then standardise to unit variance.

    Section 2.6 of the manuscript specifies the order explicitly:
    the seasonal cycle is removed first so the annual cycle does not
    bias the trend slope, the linear trend is removed second, and the
    residual is standardised last. The previous version of this
    function omitted the detrending step, so the lag-correlation and
    12-month autocorrelation diagnostics were computed on UN-detrended
    series and were therefore trend-inflated. With the step restored,
    Figs 9-12 and S1/S2 are all measures of detrended interannual
    variability, consistent with the text.
    """
    ts = snap_to_month_start(ts)
    clim = ts.groupby("time.month").mean("time")
    anom = ts.groupby("time.month") - clim     # 1. deseasonalise
    anom = _detrend_linear(anom)               # 2. linear detrend (Sec 2.6)
    s = float(anom.std("time", skipna=True))
    if not np.isfinite(s) or s == 0:
        return anom * np.nan
    return anom / s                            # 3. standardise


def _lag_correlation(
    chl_z: xr.DataArray,
    drv_z: xr.DataArray,
    lag_max: int = LAG_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson r at lags k ∈ [-lag_max, +lag_max].

    Convention: positive k means the driver leads Chl, i.e.
    r(k) correlates driver(t-k) with chl(t).

    Returns (lags, r_values) as numpy arrays.
    """
    chl_z, drv_z = xr.align(chl_z, drv_z, join="inner")
    c = chl_z.values
    d = drv_z.values
    n = len(c)
    lags = np.arange(-lag_max, lag_max + 1)
    rs   = np.full(lags.shape, np.nan)
    for i, k in enumerate(lags):
        if k >= 0:
            # driver(t-k) vs chl(t)  →  drop first k of chl, last k of drv
            x = d[: n - k] if k > 0 else d
            y = c[k:]      if k > 0 else c
        else:
            kk = -k
            # chl(t-|k|) vs drv(t)
            x = d[kk:]
            y = c[: n - kk]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 24:   # need at least 2 years
            continue
        rs[i] = float(np.corrcoef(x[ok], y[ok])[0, 1])
    return lags, rs


def _prep_for_lag(
    chl_ts: xr.DataArray,
    drv_ts: xr.DataArray,
    common_grid: xr.DataArray,
    period: slice = LAG_PERIOD,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Regrid two time series to the common grid and clip to period."""
    chl_re = regrid_timeseries(chl_ts.sel(time=period), common_grid)
    drv_re = regrid_timeseries(drv_ts.sel(time=period), common_grid)
    return chl_re, drv_re


# Bootstrap parameters for the lag-correlation diagnostic. The
# moving-block bootstrap preserves the residual autocorrelation of
# the anomaly series rather than assuming independent monthly
# samples; this matches Section 2.5 of the manuscript and yields
# more conservative significance than the naive ±2/√N threshold,
# which would overstate significance for autocorrelated series.
# BLOCK_LEN ≈ 1.5 × autocorrelation timescale (months); 12 is a
# standard choice for monthly polar series with ~year-scale memory.
LAG_BOOT_N        = 1000
LAG_BOOT_BLOCKLEN = 12
LAG_BOOT_ALPHA    = 0.05   # 95% CI; significance threshold


def _moving_block_indices(n: int, block_len: int,
                           rng: np.random.Generator) -> np.ndarray:
    """Build a length-`n` index series by sampling overlapping blocks
    with replacement. Used by both the CI and the null-distribution
    bootstraps."""
    n_blocks = int(np.ceil(n / block_len))
    # Each block is anchored at a random start in [0, n-block_len]
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    # Build offsets for each block
    offs = np.arange(block_len)[None, :] + starts[:, None]   # (n_blocks, block_len)
    idx  = offs.ravel()[:n]                                  # trim to n
    return idx


def _lag_correlation_with_bootstrap(
    chl_z: xr.DataArray,
    drv_z: xr.DataArray,
    lag_max: int = LAG_MAX,
    n_boot: int = LAG_BOOT_N,
    block_len: int = LAG_BOOT_BLOCKLEN,
    alpha: float = LAG_BOOT_ALPHA,
    seed: int = 12345,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Lag correlation with moving-block bootstrap CI and p-values.

    Returns (lags, r, ci_lo, ci_hi, pvalue) where:
      - r      : point estimate of Pearson r at each lag (same as
                 _lag_correlation)
      - ci_lo, : (1-alpha)*100% confidence interval bounds on r at
        ci_hi   each lag, from `n_boot` moving-block resamples of the
                 PAIRED series (resampling the same time indices in
                 both chl and driver so the cross-correlation is
                 preserved within blocks)
      - pvalue : two-sided p-value at each lag, from `n_boot`
                 INDEPENDENT-resampling surrogates (chl and driver
                 resampled with different block placements so the
                 cross-correlation is destroyed while each series'
                 own autocorrelation is preserved). This is the
                 procedure described in Section 2.5 of the manuscript.

    NaNs in either series at any lag are propagated as NaN in r,
    ci_lo, ci_hi, and pvalue; tests with fewer than 24 finite pairs
    return NaN.
    """
    chl_z, drv_z = xr.align(chl_z, drv_z, join="inner")
    c = np.asarray(chl_z.values, dtype=float)
    d = np.asarray(drv_z.values, dtype=float)
    n = len(c)
    lags = np.arange(-lag_max, lag_max + 1)
    r_point = np.full(lags.shape, np.nan)
    ci_lo   = np.full(lags.shape, np.nan)
    ci_hi   = np.full(lags.shape, np.nan)
    pval    = np.full(lags.shape, np.nan)

    if n < 24:
        return lags, r_point, ci_lo, ci_hi, pval

    rng = np.random.default_rng(seed)

    def _r_at_lag(c_arr: np.ndarray, d_arr: np.ndarray, k: int) -> float:
        # Convention: positive k means driver leads Chl, i.e.
        # corr(driver(t-k), chl(t)).
        if k >= 0:
            x = d_arr[: n - k] if k > 0 else d_arr
            y = c_arr[k:]      if k > 0 else c_arr
        else:
            kk = -k
            x = d_arr[kk:]
            y = c_arr[: n - kk]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 24:
            return np.nan
        return float(np.corrcoef(x[ok], y[ok])[0, 1])

    # 1. Point estimate r at each lag
    for i, k in enumerate(lags):
        r_point[i] = _r_at_lag(c, d, int(k))

    # 2. Paired moving-block bootstrap for CI (resample SAME indices
    #    in both series to preserve cross-coupling within blocks).
    boot_r_paired = np.full((n_boot, lags.size), np.nan)
    for b in range(n_boot):
        idx = _moving_block_indices(n, block_len, rng)
        cb = c[idx]
        db = d[idx]
        for i, k in enumerate(lags):
            boot_r_paired[b, i] = _r_at_lag(cb, db, int(k))
    ci_lo = np.nanpercentile(boot_r_paired,       100 * (alpha / 2.0), axis=0)
    ci_hi = np.nanpercentile(boot_r_paired, 100 * (1.0 - alpha / 2.0), axis=0)

    # 3. Independent-resampling bootstrap for the null distribution
    #    (resample chl and driver with DIFFERENT block placements so
    #    the cross-coupling is destroyed while each series' own
    #    autocorrelation is preserved). Two-sided p-value at each lag.
    boot_r_null = np.full((n_boot, lags.size), np.nan)
    for b in range(n_boot):
        idx_c = _moving_block_indices(n, block_len, rng)
        idx_d = _moving_block_indices(n, block_len, rng)
        cb = c[idx_c]
        db = d[idx_d]
        for i, k in enumerate(lags):
            boot_r_null[b, i] = _r_at_lag(cb, db, int(k))
    # Two-sided p: fraction of null draws with |r| >= |observed r|.
    for i in range(lags.size):
        if not np.isfinite(r_point[i]):
            continue
        null = boot_r_null[:, i]
        null = null[np.isfinite(null)]
        if null.size == 0:
            continue
        pval[i] = float(np.mean(np.abs(null) >= abs(r_point[i])))

    return lags, r_point, ci_lo, ci_hi, pval


def _make_lag_figure(
    lag_curves: dict[str, dict[str, tuple]],
    obs_label: str,
    driver_name: str,
    title: str,
    filename: str,
    fig_num: str,
) -> None:
    """Plot lag-correlation curves as 6 subplots (one per basin).

    lag_curves: {basin: {source_name: (lags, r, ci_lo, ci_hi, pval)}}
                where source_name is "OBS" or a model name and the
                last four entries come from
                _lag_correlation_with_bootstrap. Backward
                compatibility: a 2-tuple (lags, r) is accepted but
                in that case no CI band or significance marker is
                drawn for the curve.

    Significance display (Section 2.5 of the manuscript):
      - Translucent ribbon: 95% paired moving-block bootstrap CI.
      - Filled circles on the curve: lags where p < LAG_BOOT_ALPHA
        from the independent-resampling null distribution.
    The legacy ±2/√N band has been removed in favour of the
    per-curve bootstrap CI; the suptitle no longer advertises it.
    """
    logger.info(f"Building Fig {fig_num} ({title})...")
    # Figs 9/10 keep the original six-basin 3x2 grid; the Southern
    # Ocean sub-basins are shown only in the sea-ice diagnostics.
    basins = [b for b in MEANSTATE_BASINS if b in lag_curves]
    n_b = len(basins)
    ncols = 3
    nrows = int(np.ceil(n_b / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.0 * nrows),
                              sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    handles_for_legend = None
    labels_for_legend  = None

    for idx, basin in enumerate(basins):
        ax = axes_flat[idx]
        add_panel_label(ax, idx)
        curves = lag_curves.get(basin, {})

        # Zero-line and lag-zero guide
        ax.axhline(0, color="k", lw=0.5, alpha=0.6)
        ax.axvline(0, color="k", lw=0.5, alpha=0.6, ls=":")

        # We draw model curves first (under) and OBS last (on top)
        # by sorting OBS to the end of the iteration.
        sources_ordered = sorted(curves.keys(), key=lambda s: (s == "OBS", s))
        models_list = [s for s in curves.keys() if s != "OBS"]

        peak_info_lines = []
        for source in sources_ordered:
            entry = curves[source]
            # Unpack: support both 5-tuple (with bootstrap) and
            # legacy 2-tuple. Missing bootstrap fields become NaN
            # arrays so the ribbon/marker code degrades gracefully.
            if len(entry) == 5:
                lags, rs, ci_lo, ci_hi, pval = entry
            elif len(entry) == 2:
                lags, rs = entry
                nan_arr = np.full_like(rs, np.nan, dtype=float)
                ci_lo, ci_hi, pval = nan_arr, nan_arr.copy(), nan_arr.copy()
            else:
                logger.warning(
                    f"  [{basin}] [{source}] unexpected entry length "
                    f"{len(entry)}; skipping"
                )
                continue

            if source == "OBS":
                color = "black"
                lw    = 2.4
                zord  = 10
            else:
                # Thinner model lines for many-model ensembles so the
                # OBS curve remains the visual anchor.
                lw    = 1.4 if len(curves) <= 5 else 1.0
                color = PALETTE[models_list.index(source) % len(PALETTE)]
                zord  = 5

            # 1. 95% bootstrap CI as translucent ribbon. We draw the
            #    ribbon underneath the line so the line stays
            #    visible; alpha is low so multiple overlapping ribbons
            #    do not turn the panel into mud.
            if np.any(np.isfinite(ci_lo)) and np.any(np.isfinite(ci_hi)):
                ax.fill_between(
                    lags, ci_lo, ci_hi,
                    color=color, alpha=0.12, lw=0,
                    zorder=zord - 1,
                )

            # 2. Main line
            ax.plot(lags, rs, ls="-", lw=lw, color=color,
                    label=source, alpha=0.9, zorder=zord)

            # 3. Significance markers at lags where p < alpha.
            #    Slightly larger markers for OBS so the eye picks
            #    them out first.
            sig_mask = np.isfinite(pval) & (pval < LAG_BOOT_ALPHA)
            if np.any(sig_mask):
                msz = 5.5 if source == "OBS" else 3.5
                ax.plot(
                    lags[sig_mask], rs[sig_mask],
                    "o", ms=msz, mfc=color, mec="white",
                    mew=0.6, zorder=zord + 1,
                )

            # Peak |r| annotation. Keep only the OBS entry visible
            # in the inset box when there are many models so the box
            # does not occupy half the panel; per-model peaks are
            # still printed to the log.
            if np.any(np.isfinite(rs)):
                k_peak = int(lags[np.nanargmax(np.abs(rs))])
                r_peak = float(rs[np.nanargmax(np.abs(rs))])
                p_peak = (float(pval[np.nanargmax(np.abs(rs))])
                          if np.any(np.isfinite(pval)) else np.nan)
                if np.isfinite(p_peak):
                    line = (f"{source}: r={r_peak:+.2f} @ lag={k_peak:+d} "
                            f"(p={p_peak:.3f})")
                else:
                    line = f"{source}: r={r_peak:+.2f} @ lag={k_peak:+d}"
                if len(curves) <= 5 or source == "OBS":
                    peak_info_lines.append(line)
                else:
                    logger.info(f"    [{basin:10s}] {line}")

        ax.set_title(basin, fontsize=11, fontweight="bold")
        ax.set_xlim(-LAG_MAX, LAG_MAX)
        ax.set_ylim(-1, 1)
        ax.set_xticks(np.arange(-LAG_MAX, LAG_MAX + 1, 4))
        ax.grid(alpha=0.25)

        if idx % ncols == 0:
            ax.set_ylabel("Pearson r", fontsize=10)
        if idx // ncols == nrows - 1:
            ax.set_xlabel(f"Lag (months)  —  {driver_name} leads → ",
                          fontsize=9)

        # Small peak-r annotation box
# [removed per-panel stat box]         if peak_info_lines:
# [removed per-panel stat box]             ax.text(0.02, 0.02, "\n".join(peak_info_lines),
# [removed per-panel stat box]                     transform=ax.transAxes, fontsize=6.5, va="bottom",
# [removed per-panel stat box]                     family="monospace",
# [removed per-panel stat box]                     bbox=dict(boxstyle="round,pad=0.25",
# [removed per-panel stat box]                               facecolor="white", edgecolor="gray",
# [removed per-panel stat box]                               alpha=0.85))

        if idx == 0:
            handles_for_legend, labels_for_legend = ax.get_legend_handles_labels()

    # Hide any unused panels
    for idx in range(n_b, nrows * ncols):
        axes_flat[idx].axis("off")

    # Figure-level legend in the unused panel area or below
    if handles_for_legend is not None:
        fig.legend(handles_for_legend, labels_for_legend,
                   loc="lower center", ncol=min(len(handles_for_legend), 4),
                   fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    # Suptitle: drop the ±2/√N reference. The significance is now
    # carried by the per-curve CI ribbons and the markers.
    #fig.suptitle(
    #    f"Fig {fig_num} — {title}\n"
    #    f"Lag correlation of basin-mean deseasonalised z-score anomalies "
    #    f"(period: {LAG_PERIOD.start}–{LAG_PERIOD.stop})\n"
    #    f"OBS reference: {obs_label}  |  "
    #    f"shaded ribbons: 95% moving-block bootstrap CI ({LAG_BOOT_N} resamples); "
    #    f"filled markers: p < {LAG_BOOT_ALPHA:.2f}",
    #    fontsize=11, y=1.0,
    #)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(OUT_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {filename}")


def make_fig9_lag_sst(
    obs_chl_ts: xr.DataArray,
    obs_sst_ts: xr.DataArray | None,
    model_chl_ts: dict[str, xr.DataArray],
    model_sst_ts: dict[str, xr.DataArray],
    common_grid: xr.DataArray,
) -> None:
    """Fig 9: lag correlation between Chl and SST anomalies, by basin."""
    logger.info("=== Fig 9: lag correlation Chl vs SST (basin-mean) ===")
    masks = _build_masks(common_grid)
    lag_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        b: {} for b in masks
    }

    # ---- OBS ----
    if obs_sst_ts is not None:
        try:
            logger.info("  [OBS  ] regridding Chl + SST to common grid...")
            chl_re, sst_re = _prep_for_lag(
                obs_chl_ts, obs_sst_ts, common_grid)
            for basin, mask in masks.items():
                chl_b = _basin_mean_timeseries(chl_re, mask)
                sst_b = _basin_mean_timeseries(sst_re, mask)
                chl_z = _deseasonalise_zscore(chl_b)
                sst_z = _deseasonalise_zscore(sst_b)
                lags, rs, ci_lo, ci_hi, pval = _lag_correlation_with_bootstrap(chl_z, sst_z)
                lag_curves[basin]["OBS"] = (lags, rs, ci_lo, ci_hi, pval)
                k_peak = int(lags[np.nanargmax(np.abs(rs))]) \
                    if np.any(np.isfinite(rs)) else 0
                r_peak = float(rs[np.nanargmax(np.abs(rs))]) \
                    if np.any(np.isfinite(rs)) else float("nan")
                logger.success(
                    f"  [OBS  ] {basin:10s}  peak r={r_peak:+.3f} "
                    f"@ lag={k_peak:+d} mo")
        except Exception as e:
            logger.warning(f"  OBS lag-corr failed: {e}")
    else:
        logger.warning("  OBS SST not available; skipping OBS curve")

    # ---- models ----
    for model in MODELS:
        if model not in model_chl_ts or model not in model_sst_ts:
            continue
        try:
            logger.info(f"  [{model:9s}] regridding Chl + SST...")
            chl_re, sst_re = _prep_for_lag(
                model_chl_ts[model], model_sst_ts[model], common_grid)
            for basin, mask in masks.items():
                chl_b = _basin_mean_timeseries(chl_re, mask)
                sst_b = _basin_mean_timeseries(sst_re, mask)
                chl_z = _deseasonalise_zscore(chl_b)
                sst_z = _deseasonalise_zscore(sst_b)
                lags, rs, ci_lo, ci_hi, pval = _lag_correlation_with_bootstrap(chl_z, sst_z)
                lag_curves[basin][model] = (lags, rs, ci_lo, ci_hi, pval)
        except Exception as e:
            logger.warning(f"  [{model}] lag-corr failed: {e}")

    _make_lag_figure(
        lag_curves,
        obs_label="ESA-CCI Chl vs NOAA-ERSSTv3b SST",
        driver_name="SST",
        title="Lag correlation: basin-mean Chl-a vs SST anomalies",
        filename="fig9_lag_corr_chl_sst.png",
        fig_num="9",
    )


def make_fig10_lag_mld(
    obs_chl_ts: xr.DataArray,
    obs_mld_ts: xr.DataArray | None,
    model_chl_ts: dict[str, xr.DataArray],
    model_mld_ts: dict[str, xr.DataArray],
    common_grid: xr.DataArray,
) -> None:
    """Fig 10: lag correlation between Chl and MLD anomalies, by basin."""
    logger.info("=== Fig 10: lag correlation Chl vs MLD (basin-mean) ===")
    masks = _build_masks(common_grid)
    lag_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        b: {} for b in masks
    }

    if obs_mld_ts is not None:
        try:
            logger.info("  [OBS  ] regridding Chl + MLD to common grid...")
            chl_re, mld_re = _prep_for_lag(
                obs_chl_ts, obs_mld_ts, common_grid)
            for basin, mask in masks.items():
                chl_b = _basin_mean_timeseries(chl_re, mask)
                mld_b = _basin_mean_timeseries(mld_re, mask)
                chl_z = _deseasonalise_zscore(chl_b)
                mld_z = _deseasonalise_zscore(mld_b)
                lags, rs, ci_lo, ci_hi, pval = _lag_correlation_with_bootstrap(chl_z, mld_z)
                lag_curves[basin]["OBS"] = (lags, rs, ci_lo, ci_hi, pval)
                k_peak = int(lags[np.nanargmax(np.abs(rs))]) \
                    if np.any(np.isfinite(rs)) else 0
                r_peak = float(rs[np.nanargmax(np.abs(rs))]) \
                    if np.any(np.isfinite(rs)) else float("nan")
                logger.success(
                    f"  [OBS  ] {basin:10s}  peak r={r_peak:+.3f} "
                    f"@ lag={k_peak:+d} mo")
        except Exception as e:
            logger.warning(f"  OBS lag-corr failed: {e}")
    else:
        logger.warning("  OBS MLD not available; skipping OBS curve")

    for model in MODELS:
        if model not in model_chl_ts or model not in model_mld_ts:
            continue
        try:
            logger.info(f"  [{model:9s}] regridding Chl + MLD...")
            chl_re, mld_re = _prep_for_lag(
                model_chl_ts[model], model_mld_ts[model], common_grid)
            for basin, mask in masks.items():
                chl_b = _basin_mean_timeseries(chl_re, mask)
                mld_b = _basin_mean_timeseries(mld_re, mask)
                chl_z = _deseasonalise_zscore(chl_b)
                mld_z = _deseasonalise_zscore(mld_b)
                lags, rs, ci_lo, ci_hi, pval = _lag_correlation_with_bootstrap(chl_z, mld_z)
                lag_curves[basin][model] = (lags, rs, ci_lo, ci_hi, pval)
        except Exception as e:
            logger.warning(f"  [{model}] lag-corr failed: {e}")

    _make_lag_figure(
        lag_curves,
        obs_label="ESA-CCI Chl vs ORAS5 MLD",
        driver_name="MLD",
        title="Lag correlation: basin-mean Chl-a vs MLD anomalies",
        filename="fig10_lag_corr_chl_mld.png",
        fig_num="10",
    )


# ============================================================
# Fig 11: lag correlation of basin-mean Chl-a vs sea-ice
# concentration anomalies. Same machinery as Figs 9 & 10, with
# siconc replacing SST / MLD as the driver. This is the direct
# test of the sea-ice-mediated memory hypothesis raised by the
# Arctic SST lag finding in Fig 9.
#
# Arctic-only:
# The Atlantic/Pacific/Indian/Southern siconc time series carry
# almost no variance — they are essentially zero or near-zero
# year-round outside the polar basins — so we run this diagnostic
# on the Arctic basin only (and as a comparison, the Southern
# Ocean, where the same physics is also plausible but with
# year-round ice variability). Including Atlantic/Pacific/Indian
# would just plot noise.
# ============================================================

# Basins used by the sea-ice diagnostics (Figs 11, 12, S1, S2).
# In addition to the Arctic and the full Southern Ocean box, the
# Southern Ocean is resolved into its seasonal-ice zone (Southern-SIZ)
# and the open ACC band (Southern-ACC). The seasonal-ice zone is where
# the cryosphere memory mechanism operates; separating it from the
# ice-free ACC band prevents the basin average from diluting the
# Chl-a/sea-ice coupling and the sea-ice persistence signal.
SICONC_BASINS = ("Arctic", "Southern", "Southern-SIZ", "Southern-ACC")


def make_fig11_lag_siconc(
    obs_chl_ts: xr.DataArray,
    obs_siconc_ts: xr.DataArray | None,
    model_chl_ts: dict[str, xr.DataArray],
    model_siconc_ts: dict[str, xr.DataArray],
    common_grid: xr.DataArray,
) -> None:
    """Fig 11: lag correlation Chl-a vs siconc, Arctic + Southern Ocean only.

    Parallel construction to Figs 9 and 10; only the basin list is
    restricted because mid-latitude siconc carries no signal.
    """
    logger.info("=== Fig 11: lag correlation Chl vs siconc (basin-mean) ===")

    # Build masks for the two polar basins only
    all_masks = _build_masks(common_grid)
    masks = {b: all_masks[b] for b in SICONC_BASINS if b in all_masks}

    lag_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        b: {} for b in masks
    }

    # ---- OBS ----
    if obs_siconc_ts is not None:
        try:
            logger.info("  [OBS  ] regridding Chl + siconc to common grid...")
            chl_re, sic_re = _prep_for_lag(
                obs_chl_ts, obs_siconc_ts, common_grid)
            for basin, mask in masks.items():
                chl_b = _basin_mean_timeseries(chl_re, mask)
                sic_b = _basin_mean_timeseries(sic_re, mask)
                chl_z = _deseasonalise_zscore(chl_b)
                sic_z = _deseasonalise_zscore(sic_b)
                lags, rs, ci_lo, ci_hi, pval = _lag_correlation_with_bootstrap(chl_z, sic_z)
                lag_curves[basin]["OBS"] = (lags, rs, ci_lo, ci_hi, pval)
                if np.any(np.isfinite(rs)):
                    k_peak = int(lags[np.nanargmax(np.abs(rs))])
                    r_peak = float(rs[np.nanargmax(np.abs(rs))])
                    logger.success(
                        f"  [OBS  ] {basin:10s}  peak r={r_peak:+.3f} "
                        f"@ lag={k_peak:+d} mo")
        except Exception as e:
            logger.warning(f"  OBS lag-corr failed: {e}")
    else:
        logger.warning("  OBS siconc not available; skipping OBS curve")

    # ---- models ----
    for model in MODELS:
        if model not in model_chl_ts or model not in model_siconc_ts:
            continue
        try:
            logger.info(f"  [{model:9s}] regridding Chl + siconc...")
            chl_re, sic_re = _prep_for_lag(
                model_chl_ts[model], model_siconc_ts[model], common_grid)
            for basin, mask in masks.items():
                chl_b = _basin_mean_timeseries(chl_re, mask)
                sic_b = _basin_mean_timeseries(sic_re, mask)
                chl_z = _deseasonalise_zscore(chl_b)
                sic_z = _deseasonalise_zscore(sic_b)
                lags, rs, ci_lo, ci_hi, pval = _lag_correlation_with_bootstrap(chl_z, sic_z)
                lag_curves[basin][model] = (lags, rs, ci_lo, ci_hi, pval)
        except Exception as e:
            logger.warning(f"  [{model}] lag-corr failed: {e}")

    # The shared helper expects a curve per BASIN dict entry; we built
    # only the two polar basins, so we need a basin-restricted variant
    # of _make_lag_figure. The simplest path is to call the helper but
    # pre-fill the missing basins with empty dicts so it skips them.
    # We instead build a dedicated 1x2 figure here.
    if not any(lag_curves.values()):
        logger.warning("  Fig 11 SKIPPED: no curves built")
        return

    basins = list(masks.keys())
    n_b = len(basins)
    fig, axes = plt.subplots(1, n_b, figsize=(7 * n_b, 5),
                              sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    handles_for_legend = None
    labels_for_legend  = None

    for idx, basin in enumerate(basins):
        ax = axes_flat[idx]
        add_panel_label(ax, idx)
        curves = lag_curves.get(basin, {})

        ax.axhline(0, color="k", lw=0.5, alpha=0.6)
        ax.axvline(0, color="k", lw=0.5, alpha=0.6, ls=":")

        sources_ordered = sorted(curves.keys(), key=lambda s: (s == "OBS", s))
        models_in = [s for s in curves.keys() if s != "OBS"]

        peak_info_lines = []
        for source in sources_ordered:
            entry = curves[source]
            if len(entry) == 5:
                lags, rs, ci_lo, ci_hi, pval = entry
            else:
                lags, rs = entry
                nan_arr = np.full_like(rs, np.nan, dtype=float)
                ci_lo, ci_hi, pval = nan_arr, nan_arr.copy(), nan_arr.copy()

            if source == "OBS":
                color, lw, zord = "black", 2.4, 10
            else:
                color = PALETTE[models_in.index(source) % len(PALETTE)] \
                        if source in models_in else "gray"
                # Thinner model lines for many-model ensembles
                lw    = 1.4 if len(curves) <= 5 else 1.0
                zord  = 5

            # 95% bootstrap CI ribbon
            if np.any(np.isfinite(ci_lo)) and np.any(np.isfinite(ci_hi)):
                ax.fill_between(lags, ci_lo, ci_hi,
                                color=color, alpha=0.12, lw=0,
                                zorder=zord - 1)
            ax.plot(lags, rs, lw=lw, color=color, label=source,
                    alpha=0.9, zorder=zord)

            # Significance markers
            sig_mask = np.isfinite(pval) & (pval < LAG_BOOT_ALPHA)
            if np.any(sig_mask):
                msz = 5.5 if source == "OBS" else 3.5
                ax.plot(lags[sig_mask], rs[sig_mask],
                        "o", ms=msz, mfc=color, mec="white",
                        mew=0.6, zorder=zord + 1)

            if np.any(np.isfinite(rs)):
                k_peak = int(lags[np.nanargmax(np.abs(rs))])
                r_peak = float(rs[np.nanargmax(np.abs(rs))])
                p_peak = (float(pval[np.nanargmax(np.abs(rs))])
                          if np.any(np.isfinite(pval)) else np.nan)
                if np.isfinite(p_peak):
                    line = (f"{source}: r={r_peak:+.2f} @ lag={k_peak:+d} "
                            f"(p={p_peak:.3f})")
                else:
                    line = f"{source}: r={r_peak:+.2f} @ lag={k_peak:+d}"
                # With many models the inset box overflows. Show OBS
                # only and log model peaks instead.
                if len(curves) <= 5 or source == "OBS":
                    peak_info_lines.append(line)
                else:
                    logger.info(f"    {line}")

        ax.set_title(basin, fontsize=11, fontweight="bold")
        ax.set_xlim(-LAG_MAX, LAG_MAX)
        ax.set_ylim(-1, 1)
        ax.set_xticks(np.arange(-LAG_MAX, LAG_MAX + 1, 4))
        ax.grid(alpha=0.25)
        ax.set_xlabel("Lag (months)  —  siconc leads → ", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Pearson r", fontsize=10)

# [removed per-panel stat box]         if peak_info_lines:
# [removed per-panel stat box]             ax.text(0.02, 0.02, "\n".join(peak_info_lines),
# [removed per-panel stat box]                     transform=ax.transAxes, fontsize=6.5, va="bottom",
# [removed per-panel stat box]                     family="monospace",
# [removed per-panel stat box]                     bbox=dict(boxstyle="round,pad=0.25",
# [removed per-panel stat box]                               facecolor="white", edgecolor="gray",
# [removed per-panel stat box]                               alpha=0.85))

        if idx == 0:
            handles_for_legend, labels_for_legend = \
                ax.get_legend_handles_labels()

    if handles_for_legend is not None:
        fig.legend(handles_for_legend, labels_for_legend,
                   loc="lower center", ncol=min(len(handles_for_legend), 4),
                   fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    #fig.suptitle(
    #    "Fig 11 — Lag correlation: basin-mean Chl-a vs sea-ice concentration anomalies\n"
    #    f"OBS reference: ESA-CCI Chl vs HadISST siconc  |  "
    #    f"period: {LAG_PERIOD.start}–{LAG_PERIOD.stop}  |  "
    #    f"shaded ribbons: 95% moving-block bootstrap CI ({LAG_BOOT_N} resamples); "
    #    f"filled markers: p < {LAG_BOOT_ALPHA:.2f}",
    #    fontsize=11, y=1.0,
    #)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(OUT_DIR / "fig11_lag_corr_chl_siconc.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote fig11_lag_corr_chl_siconc.png")


# ============================================================
# Fig 11b: longitudinal sector decomposition of the Southern Ocean
# Chl-a / sea-ice lag correlation.
#
# The Fig 11 Southern-SIZ panel averages over the entire circumpolar
# seasonal-ice zone. That belt spans regimes with very different bloom
# physics (the marginal-ice-zone Weddell, the polynya-dominated Ross,
# the comparatively narrow Indian and Bellingshausen-Amundsen sectors,
# etc.), so a model can produce a basin-mean correlation near zero by
# getting one sector right and another wrong with opposing sign --
# qualitatively different from a uniform failure. To distinguish these
# cases, Fig 11b repeats the Fig 11 diagnostic on five circumpolar SIZ
# sectors plus a Southern-ACC negative-control panel (55-45 S,
# year-round ice-free; a flat curve there is the expected result and
# provides a within-figure check that any signals in the sector
# panels are not artefacts of the masking or pipeline).
#
# Sector definitions follow the NSIDC convention used in Southern Ocean
# sea-ice reporting:
#   Weddell                : 60 W  -  20 E   (wraps 0/360)
#   Indian                 : 20 E  -  90 E
#   West Pacific           : 90 E  - 160 E
#   Ross                   :160 E  - 130 W   (wraps 180)
#   Bellingshausen-Amundsen:130 W  -  60 W
# Each sector is intersected with the SIZ latitude band (70 S - 55 S,
# matching SOUTHERN_SUBBASINS["Southern-SIZ"]).
#
# Methodology -- deseasonalise, linearly detrend, z-score, Pearson r
# at lags -12..+12 months, sign convention "siconc leads Chl-a" for
# positive lag -- is identical to Fig 11; the helpers
# _basin_mean_timeseries, _deseasonalise_zscore, _lag_correlation and
# _prep_for_lag are reused unchanged, so any difference between Fig 11
# and Fig 11b reflects geography only, not a methodological mismatch.
# ============================================================

# Each entry is a list of (lat_min, lat_max, lon_min, lon_max) boxes
# in degrees east [0, 360). Sectors that cross the 0/360 wrap or the
# dateline (Weddell, Ross) are expressed as two boxes and OR-ed in
# _build_sector_mask below.
SOUTHERN_SECTORS = {
    "Weddell":      [(-70, -55, 300, 360),
                     (-70, -55,   0,  20)],
    "Indian":       [(-70, -55,  20,  90)],
    "West-Pacific": [(-70, -55,  90, 160)],
    "Ross":         [(-70, -55, 160, 230)],
    "Bell-Amund":   [(-70, -55, 230, 300)],
}

# Display order: Weddell first (the canonical anchor of the SO bloom
# literature), then eastward around the continent. The open-ACC band
# (Southern-ACC, 55-45 S, year-round ice-free) is appended last as a
# negative-control panel: no cryospheric memory is expected there,
# so a flat curve in (f) is the right answer and confirms that the
# sector-resolved signals in (a)-(e) are not basin-averaging artefacts.
SECTOR_ORDER = ["Weddell", "Indian", "West-Pacific",
                "Ross", "Bell-Amund", "Southern-ACC"]


def _build_sector_mask(
    ref_grid: xr.DataArray,
    boxes: list[tuple[float, float, float, float]],
) -> xr.DataArray:
    """Union of one or more lat-lon boxes as a boolean (lat, lon) mask.

    Sectors that straddle the 0/360 wrap (Weddell) or the dateline
    (Ross) are passed in as two contiguous boxes and combined here, so
    the rest of the pipeline never has to special-case wrap-around.
    """
    lat = ref_grid.lat
    lon = ref_grid.lon
    lat2d, lon2d = xr.broadcast(lat, lon)
    out = xr.zeros_like(lat2d, dtype=bool)
    for lat_min, lat_max, lon_min, lon_max in boxes:
        in_lat = (lat2d >= lat_min) & (lat2d <= lat_max)
        in_lon = (lon2d >= lon_min) & (lon2d <= lon_max)
        out = out | (in_lat & in_lon)
    return out


def make_fig11b_lag_siconc_sectors(
    obs_chl_ts: xr.DataArray,
    obs_siconc_ts: xr.DataArray | None,
    model_chl_ts: dict[str, xr.DataArray],
    model_siconc_ts: dict[str, xr.DataArray],
    common_grid: xr.DataArray,
) -> None:
    """Fig 11b: longitudinal sector decomposition of the Fig 11 Southern
    Ocean panel.

    Six subplots (2x3): five NSIDC-style SIZ sectors plus a
    Southern-ACC negative-control panel. The control panel (55-45 S,
    year-round ice-free) is drawn against a light grey background so
    the eye can immediately separate it from the cryospherically active
    sectors. A flat curve in the Southern-ACC panel is the expected
    result and confirms that any signals seen in (a)-(e) are not
    basin-averaging or pipeline artefacts.
    """
    logger.info("=== Fig 11b: sector decomposition of Chl/siconc coupling ===")

    # ---- Build sector masks on the common grid + reuse Southern-ACC ----
    sector_masks: dict[str, xr.DataArray] = {
        name: _build_sector_mask(common_grid, boxes)
        for name, boxes in SOUTHERN_SECTORS.items()
    }
    # Reuse the existing Southern-ACC basin mask for the negative-control
    # panel. The open-ACC band (55-45 S) is essentially ice-free
    # year-round; a flat curve in panel (f) is the expected result and
    # provides a within-figure check that any signals seen in (a)-(e)
    # are not artefacts of the masking or pipeline.
    all_basin_masks = _build_masks(common_grid)
    if "Southern-ACC" not in all_basin_masks:
        logger.error("Fig 11b: Southern-ACC mask not available; aborting")
        return
    sector_masks["Southern-ACC"] = all_basin_masks["Southern-ACC"]

    # Log cell counts so the user can spot anomalously thin sectors
    # before interpreting wide bootstrap CIs as model failure.
    for name in SECTOR_ORDER:
        n_cells = int(sector_masks[name].sum())
        logger.info(f"  {name:14s}  {n_cells:6d} cells")

    lag_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        s: {} for s in SECTOR_ORDER
    }

    # ---- OBS ----
    if obs_siconc_ts is not None:
        try:
            logger.info("  [OBS  ] regridding Chl + siconc...")
            chl_re, sic_re = _prep_for_lag(
                obs_chl_ts, obs_siconc_ts, common_grid)
            for sector in SECTOR_ORDER:
                mask = sector_masks[sector]
                chl_b = _basin_mean_timeseries(chl_re, mask)
                sic_b = _basin_mean_timeseries(sic_re, mask)
                chl_z = _deseasonalise_zscore(chl_b)
                sic_z = _deseasonalise_zscore(sic_b)
                lags, rs, ci_lo, ci_hi, pval = _lag_correlation_with_bootstrap(chl_z, sic_z)
                lag_curves[sector]["OBS"] = (lags, rs, ci_lo, ci_hi, pval)
                if np.any(np.isfinite(rs)):
                    k_peak = int(lags[np.nanargmax(np.abs(rs))])
                    r_peak = float(rs[np.nanargmax(np.abs(rs))])
                    logger.success(
                        f"  [OBS  ] {sector:14s} peak r={r_peak:+.3f} "
                        f"@ lag={k_peak:+d} mo")
        except Exception as e:
            logger.warning(f"  OBS sector lag-corr failed: {e}")
    else:
        logger.warning("  OBS siconc not available; skipping OBS curves")

    # ---- models ----
    for model in MODELS:
        if model not in model_chl_ts or model not in model_siconc_ts:
            continue
        try:
            logger.info(f"  [{model:9s}] regridding Chl + siconc...")
            chl_re, sic_re = _prep_for_lag(
                model_chl_ts[model], model_siconc_ts[model], common_grid)
            for sector in SECTOR_ORDER:
                mask = sector_masks[sector]
                chl_b = _basin_mean_timeseries(chl_re, mask)
                sic_b = _basin_mean_timeseries(sic_re, mask)
                chl_z = _deseasonalise_zscore(chl_b)
                sic_z = _deseasonalise_zscore(sic_b)
                lags, rs, ci_lo, ci_hi, pval = _lag_correlation_with_bootstrap(chl_z, sic_z)
                lag_curves[sector][model] = (lags, rs, ci_lo, ci_hi, pval)
        except Exception as e:
            logger.warning(f"  [{model}] sector lag-corr failed: {e}")

    if not any(lag_curves.values()):
        logger.warning("  Fig 11b SKIPPED: no curves built")
        return

    # ---- Figure: 2 rows x 3 cols = 6 panels ----
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8.5),
                              sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    handles_for_legend = None
    labels_for_legend  = None

    for idx, sector in enumerate(SECTOR_ORDER):
        ax = axes_flat[idx]
        add_panel_label(ax, idx)
        curves = lag_curves.get(sector, {})

        # Visually distinguish the negative-control panel
        if sector == "Southern-ACC":
            ax.set_facecolor("#f7f7f7")

        ax.axhline(0, color="k", lw=0.5, alpha=0.6)
        ax.axvline(0, color="k", lw=0.5, alpha=0.6, ls=":")

        # OBS drawn last so it sits on top of model curves
        sources_ordered = sorted(curves.keys(), key=lambda s: (s == "OBS", s))
        models_in = [s for s in curves.keys() if s != "OBS"]

        peak_info_lines = []
        for source in sources_ordered:
            entry = curves[source]
            if len(entry) == 5:
                lags, rs, ci_lo, ci_hi, pval = entry
            else:
                lags, rs = entry
                nan_arr = np.full_like(rs, np.nan, dtype=float)
                ci_lo, ci_hi, pval = nan_arr, nan_arr.copy(), nan_arr.copy()

            if source == "OBS":
                color, lw, zord = "black", 2.4, 10
            else:
                color = PALETTE[models_in.index(source) % len(PALETTE)] \
                        if source in models_in else "gray"
                # Thinner model lines for many-model ensembles
                lw    = 1.4 if len(curves) <= 5 else 1.0
                zord  = 5

            # 95% bootstrap CI ribbon
            if np.any(np.isfinite(ci_lo)) and np.any(np.isfinite(ci_hi)):
                ax.fill_between(lags, ci_lo, ci_hi,
                                color=color, alpha=0.12, lw=0,
                                zorder=zord - 1)
            ax.plot(lags, rs, lw=lw, color=color, label=source,
                    alpha=0.9, zorder=zord)

            # Significance markers (p < alpha)
            sig_mask = np.isfinite(pval) & (pval < LAG_BOOT_ALPHA)
            if np.any(sig_mask):
                msz = 5.5 if source == "OBS" else 3.5
                ax.plot(lags[sig_mask], rs[sig_mask],
                        "o", ms=msz, mfc=color, mec="white",
                        mew=0.6, zorder=zord + 1)

            if np.any(np.isfinite(rs)):
                k_peak = int(lags[np.nanargmax(np.abs(rs))])
                r_peak = float(rs[np.nanargmax(np.abs(rs))])
                p_peak = (float(pval[np.nanargmax(np.abs(rs))])
                          if np.any(np.isfinite(pval)) else np.nan)
                if np.isfinite(p_peak):
                    line = (f"{source}: r={r_peak:+.2f} @ lag={k_peak:+d} "
                            f"(p={p_peak:.3f})")
                else:
                    line = f"{source}: r={r_peak:+.2f} @ lag={k_peak:+d}"
                # With many models the inset box overflows. Show OBS
                # only and log model peaks instead.
                if len(curves) <= 5 or source == "OBS":
                    peak_info_lines.append(line)
                else:
                    logger.info(f"    {line}")

        if sector == "Southern-ACC":
            title_str = "Southern-ACC  (open ACC band, 55-45 S; negative control)"
        else:
            title_str = sector
        ax.set_title(title_str, fontsize=11, fontweight="bold")

        ax.set_xlim(-LAG_MAX, LAG_MAX)
        ax.set_ylim(-1, 1)
        ax.set_xticks(np.arange(-LAG_MAX, LAG_MAX + 1, 4))
        ax.grid(alpha=0.25)

        if idx % ncols == 0:
            ax.set_ylabel("Pearson r", fontsize=10)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("Lag (months)  —  siconc leads → ", fontsize=9)

# [removed per-panel stat box]         if peak_info_lines:
# [removed per-panel stat box]             ax.text(0.02, 0.02, "\n".join(peak_info_lines),
# [removed per-panel stat box]                     transform=ax.transAxes, fontsize=6.5, va="bottom",
# [removed per-panel stat box]                     family="monospace",
# [removed per-panel stat box]                     bbox=dict(boxstyle="round,pad=0.25",
# [removed per-panel stat box]                               facecolor="white", edgecolor="gray",
# [removed per-panel stat box]                               alpha=0.85))

        if idx == 0:
            handles_for_legend, labels_for_legend = \
                ax.get_legend_handles_labels()

    if handles_for_legend is not None:
        fig.legend(handles_for_legend, labels_for_legend,
                   loc="lower center", ncol=min(len(handles_for_legend), 4),
                   fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 1.0])
    out_path = OUT_DIR / "fig11b_lag_corr_chl_siconc_sectors.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {out_path.name}")


# ============================================================
# Fig 11c: meridional (latitude-band) decomposition of the Southern
# Ocean Chl-a / sea-ice lag correlation.
#
# Where Fig 11b cuts the SIZ longitudinally into NSIDC sectors,
# Fig 11c cuts the Southern Ocean MERIDIONALLY into three contiguous
# latitude bands that map onto distinct dynamical regimes:
#
#   Subtropics  : 45-30 S  -- subtropical front, ice-free year round,
#                             no cryospheric memory expected. Acts as
#                             an upper-latitude negative control.
#   ACC         : 60-45 S  -- the open Antarctic Circumpolar Current,
#                             between the SIZ and the subtropics.
#                             Sporadic ice in some sectors but
#                             dominated by stratification and Ekman
#                             dynamics rather than cryospheric forcing.
#   High-lat    : 80-60 S  -- the cryospherically active band that
#                             contains the seasonal-ice zone and the
#                             marginal-ice-zone bloom. Capped at 80 S
#                             to match the headline Southern basin
#                             upper bound (no reliable obs poleward).
#
# Same methodology as Figs 11 and 11b -- deseasonalise, linearly
# detrend, z-score, Pearson r at lags -12..+12, sign convention
# "siconc leads Chl-a" for positive lag -- so any difference between
# Fig 11c and Fig 11/11b reflects geography only, not pipeline
# inconsistency. The contrast across the three meridional bands tests
# whether the Chl-a/sea-ice coupling is confined to the cryospherically
# active band, as physical reasoning would predict.
# ============================================================

# Meridional band definitions: (lat_min, lat_max, lon_min, lon_max).
# All circumpolar (0-360 lon). The boundaries are contiguous so that
# the three bands together cover the Southern Ocean box (-80, -30)
# without overlap. The upper bound matches the BASINS["Southern"]
# upper bound (-80) so the same polar-cap convention applies here.
SOUTHERN_MERIDIONAL_BANDS = {
    "Subtropics-SO":  (-45, -30,   0, 360),
    "ACC-band":       (-60, -45,   0, 360),
    "High-lat-SO":    (-80, -60,   0, 360),
}

# Display order: from low latitudes to high latitudes (poleward), so
# the eye follows the gradient from the ice-free subtropical control
# (a) through the ACC (b) to the cryospherically active high latitudes
# (c). The expected pattern is an emerging coupling as the band moves
# poleward into the SIZ.
MERIDIONAL_ORDER = ["Subtropics-SO", "ACC-band", "High-lat-SO"]


def make_fig11c_lag_siconc_meridional(
    obs_chl_ts: xr.DataArray,
    obs_siconc_ts: xr.DataArray | None,
    model_chl_ts: dict[str, xr.DataArray],
    model_siconc_ts: dict[str, xr.DataArray],
    common_grid: xr.DataArray,
) -> None:
    """Fig 11c: meridional decomposition of the Southern Ocean Chl-a /
    sea-ice lag correlation, three contiguous latitude bands.

    Three subplots (1x3): Subtropics, ACC, High-lat. The two
    non-cryospheric bands (Subtropics, ACC) are drawn against a light
    grey background so the eye can immediately separate the
    negative-control bands from the cryospherically active one.
    """
    logger.info("=== Fig 11c: meridional decomposition of Chl/siconc coupling ===")

    # ---- Build band masks on the common grid ----
    band_masks: dict[str, xr.DataArray] = {}
    for name, (lat_min, lat_max, lon_min, lon_max) in SOUTHERN_MERIDIONAL_BANDS.items():
        band_masks[name] = _build_sector_mask(
            common_grid, [(lat_min, lat_max, lon_min, lon_max)])

    for name in MERIDIONAL_ORDER:
        n_cells = int(band_masks[name].sum())
        logger.info(f"  {name:14s}  {n_cells:6d} cells")

    lag_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {
        b: {} for b in MERIDIONAL_ORDER
    }

    # ---- OBS ----
    if obs_siconc_ts is not None:
        try:
            logger.info("  [OBS  ] regridding Chl + siconc...")
            chl_re, sic_re = _prep_for_lag(
                obs_chl_ts, obs_siconc_ts, common_grid)
            for band in MERIDIONAL_ORDER:
                mask = band_masks[band]
                chl_b = _basin_mean_timeseries(chl_re, mask)
                sic_b = _basin_mean_timeseries(sic_re, mask)
                chl_z = _deseasonalise_zscore(chl_b)
                sic_z = _deseasonalise_zscore(sic_b)
                lags, rs, ci_lo, ci_hi, pval = _lag_correlation_with_bootstrap(chl_z, sic_z)
                lag_curves[band]["OBS"] = (lags, rs, ci_lo, ci_hi, pval)
                if np.any(np.isfinite(rs)):
                    k_peak = int(lags[np.nanargmax(np.abs(rs))])
                    r_peak = float(rs[np.nanargmax(np.abs(rs))])
                    logger.success(
                        f"  [OBS  ] {band:14s} peak r={r_peak:+.3f} "
                        f"@ lag={k_peak:+d} mo")
        except Exception as e:
            logger.warning(f"  OBS meridional lag-corr failed: {e}")
    else:
        logger.warning("  OBS siconc not available; skipping OBS curves")

    # ---- models ----
    for model in MODELS:
        if model not in model_chl_ts or model not in model_siconc_ts:
            continue
        try:
            logger.info(f"  [{model:9s}] regridding Chl + siconc...")
            chl_re, sic_re = _prep_for_lag(
                model_chl_ts[model], model_siconc_ts[model], common_grid)
            for band in MERIDIONAL_ORDER:
                mask = band_masks[band]
                chl_b = _basin_mean_timeseries(chl_re, mask)
                sic_b = _basin_mean_timeseries(sic_re, mask)
                chl_z = _deseasonalise_zscore(chl_b)
                sic_z = _deseasonalise_zscore(sic_b)
                lags, rs, ci_lo, ci_hi, pval = _lag_correlation_with_bootstrap(chl_z, sic_z)
                lag_curves[band][model] = (lags, rs, ci_lo, ci_hi, pval)
        except Exception as e:
            logger.warning(f"  [{model}] meridional lag-corr failed: {e}")

    if not any(lag_curves.values()):
        logger.warning("  Fig 11c SKIPPED: no curves built")
        return

    # ---- Figure: 1 row x 3 cols ----
    nrows, ncols = 1, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5.0),
                              sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    handles_for_legend = None
    labels_for_legend  = None

    for idx, band in enumerate(MERIDIONAL_ORDER):
        ax = axes_flat[idx]
        add_panel_label(ax, idx)
        curves = lag_curves.get(band, {})

        # Light grey background on the two non-cryospheric bands
        # marks them as negative-control panels.
        if band in ("Subtropics-SO", "ACC-band"):
            ax.set_facecolor("#f7f7f7")

        ax.axhline(0, color="k", lw=0.5, alpha=0.6)
        ax.axvline(0, color="k", lw=0.5, alpha=0.6, ls=":")

        sources_ordered = sorted(curves.keys(), key=lambda s: (s == "OBS", s))
        models_in = [s for s in curves.keys() if s != "OBS"]

        peak_info_lines = []
        for source in sources_ordered:
            entry = curves[source]
            if len(entry) == 5:
                lags, rs, ci_lo, ci_hi, pval = entry
            else:
                lags, rs = entry
                nan_arr = np.full_like(rs, np.nan, dtype=float)
                ci_lo, ci_hi, pval = nan_arr, nan_arr.copy(), nan_arr.copy()

            if source == "OBS":
                color, lw, zord = "black", 2.4, 10
            else:
                color = PALETTE[models_in.index(source) % len(PALETTE)] \
                        if source in models_in else "gray"
                # Thinner model lines for many-model ensembles
                lw    = 1.4 if len(curves) <= 5 else 1.0
                zord  = 5

            # 95% bootstrap CI ribbon
            if np.any(np.isfinite(ci_lo)) and np.any(np.isfinite(ci_hi)):
                ax.fill_between(lags, ci_lo, ci_hi,
                                color=color, alpha=0.12, lw=0,
                                zorder=zord - 1)
            ax.plot(lags, rs, lw=lw, color=color, label=source,
                    alpha=0.9, zorder=zord)

            # Significance markers
            sig_mask = np.isfinite(pval) & (pval < LAG_BOOT_ALPHA)
            if np.any(sig_mask):
                msz = 5.5 if source == "OBS" else 3.5
                ax.plot(lags[sig_mask], rs[sig_mask],
                        "o", ms=msz, mfc=color, mec="white",
                        mew=0.6, zorder=zord + 1)

            if np.any(np.isfinite(rs)):
                k_peak = int(lags[np.nanargmax(np.abs(rs))])
                r_peak = float(rs[np.nanargmax(np.abs(rs))])
                p_peak = (float(pval[np.nanargmax(np.abs(rs))])
                          if np.any(np.isfinite(pval)) else np.nan)
                if np.isfinite(p_peak):
                    line = (f"{source}: r={r_peak:+.2f} @ lag={k_peak:+d} "
                            f"(p={p_peak:.3f})")
                else:
                    line = f"{source}: r={r_peak:+.2f} @ lag={k_peak:+d}"
                # With many models the inset box overflows. Show OBS
                # only and log model peaks instead.
                if len(curves) <= 5 or source == "OBS":
                    peak_info_lines.append(line)
                else:
                    logger.info(f"    {line}")

        # Per-panel titles use the descriptive form (latitudes spelled
        # out) so the reader can identify the band without consulting
        # the caption.
        #title_map = {
        #    "Subtropics-SO": "Subtropics  (45-30 S; ice-free control)",
        #    "ACC-band":      "ACC band    (60-45 S; ice-poor control)",
        #    "High-lat-SO":   "High-lat    (80-60 S; cryospherically active)",
        #}
        #ax.set_title(title_map[band], fontsize=11, fontweight="bold")

        ax.set_xlim(-LAG_MAX, LAG_MAX)
        ax.set_ylim(-1, 1)
        ax.set_xticks(np.arange(-LAG_MAX, LAG_MAX + 1, 4))
        ax.grid(alpha=0.25)

        if idx % ncols == 0:
            ax.set_ylabel("Pearson r", fontsize=10)
        ax.set_xlabel("Lag (months)  —  siconc leads → ", fontsize=9)

        # [removed per-panel stat box]
        # if peak_info_lines:
        #     ax.text(0.02, 0.02, "\n".join(peak_info_lines),
        #             transform=ax.transAxes, fontsize=6.5, va="bottom",
        #             family="monospace",
        #             bbox=dict(boxstyle="round,pad=0.25",
        #                       facecolor="white", edgecolor="gray",
        #                       alpha=0.85))

        if idx == 0:
            handles_for_legend, labels_for_legend = \
                ax.get_legend_handles_labels()

    if handles_for_legend is not None:
        fig.legend(handles_for_legend, labels_for_legend,
                   loc="lower center", ncol=min(len(handles_for_legend), 4),
                   fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.06))

    fig.tight_layout(rect=[0, 0.05, 1, 1.0])
    out_path = OUT_DIR / "fig11c_lag_corr_chl_siconc_meridional.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {out_path.name}")


# ============================================================
# Fig 12: 12-month autocorrelation of basin-mean siconc anomalies.
#
# A single number per (source, basin). Tests directly whether the
# memory carrier itself (sea ice) retains cross-year information.
# Strong observed autocorrelation + weak modelled autocorrelation
# is the most direct possible evidence that models lose the memory
# that the Fig 9 SST→Chl lag implies.
# ============================================================

def _autocorr_at_lag(ts: xr.DataArray, lag: int) -> float:
    """Pearson autocorrelation at the given integer lag, after
    deseasonalising and z-scoring the input."""
    z = _deseasonalise_zscore(ts)
    v = z.values
    n = len(v)
    if n <= lag + 12:
        return float("nan")
    x = v[: n - lag]
    y = v[lag:]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 24:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def make_fig12_siconc_memory(
    obs_siconc_ts: xr.DataArray | None,
    model_siconc_ts: dict[str, xr.DataArray],
    common_grid: xr.DataArray,
    lag_months: int = 12,
) -> None:
    """Fig 12: bar chart of r(siconc(t), siconc(t-12)) per source per polar basin.

    This is the direct broken-memory test: if the observed value
    is large and the model values are small, the models lose the
    cross-year memory that nature carries — and that loss is a
    sufficient explanation for the Fig 11 / Fig 9 lag mismatch.
    """
    logger.info(f"=== Fig 12: siconc {lag_months}-month autocorrelation ===")

    all_masks = _build_masks(common_grid)
    basins = [b for b in SICONC_BASINS if b in all_masks]
    masks  = {b: all_masks[b] for b in basins}

    # results[source][basin] = autocorrelation value
    results: dict[str, dict[str, float]] = {}

    # ---- OBS ----
    if obs_siconc_ts is not None:
        try:
            obs_re = regrid_timeseries(
                obs_siconc_ts.sel(time=LAG_PERIOD), common_grid)
            results["OBS"] = {}
            for basin, mask in masks.items():
                bm = _basin_mean_timeseries(obs_re, mask)
                results["OBS"][basin] = _autocorr_at_lag(bm, lag_months)
                logger.success(
                    f"  [OBS  ] {basin:10s}  r(siconc, lag={lag_months}) = "
                    f"{results['OBS'][basin]:+.3f}")
        except Exception as e:
            logger.warning(f"  OBS siconc autocorrelation failed: {e}")

    # ---- models ----
    for model in MODELS:
        if model not in model_siconc_ts:
            continue
        try:
            sic_re = regrid_timeseries(
                model_siconc_ts[model].sel(time=LAG_PERIOD), common_grid)
            results[model] = {}
            for basin, mask in masks.items():
                bm = _basin_mean_timeseries(sic_re, mask)
                results[model][basin] = _autocorr_at_lag(bm, lag_months)
                logger.success(
                    f"  [{model:9s}] {basin:10s}  r(siconc, lag={lag_months}) = "
                    f"{results[model][basin]:+.3f}")
        except Exception as e:
            logger.warning(f"  [{model}] siconc autocorrelation failed: {e}")

    if not results:
        logger.warning("  Fig 12 SKIPPED: no autocorrelations computed")
        return

    # ---- write CSV table ----
    csv_path = OUT_DIR / "fig12_siconc_memory_table.csv"
    with open(csv_path, "w") as fp:
        fp.write("source,basin,r_lag12\n")
        for source, basin_results in results.items():
            for basin, r in basin_results.items():
                fp.write(f"{source},{basin},{r:.4f}\n")
    logger.success(f"  wrote {csv_path.name}")

    # ---- bar chart ----
    sources = list(results.keys())
    n_s = len(sources)
    n_b = len(basins)
    width = 0.78 / n_s
    x = np.arange(n_b)

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_b + 2), 5))

    for i, source in enumerate(sources):
        offset = (i - (n_s - 1) / 2) * width
        vals = [results[source].get(b, np.nan) for b in basins]
        if source == "OBS":
            color = "black"
        else:
            non_obs = [s for s in sources if s != "OBS"]
            color = PALETTE[non_obs.index(source) % len(PALETTE)] \
                    if source in non_obs else "gray"
        bars = ax.bar(x + offset, vals, width, label=source,
                      color=color, edgecolor="black", linewidth=0.4)
        for bar, v in zip(bars, vals):
            if not np.isfinite(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.02 if v >= 0 else -0.02), f"{v:+.2f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8, color=color, fontweight="bold")

    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(basins, fontsize=10)
    ax.set_ylabel(f"r(siconc(t), siconc(t−{lag_months} mo))", fontsize=10)
    ax.set_ylim(-0.4, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title(
        f"Fig 12 — {lag_months}-month autocorrelation of basin-mean "
        f"deseasonalised siconc anomalies\n"
        f"Tests whether the sea-ice memory carrier persists across "
        f"one year  |  period: {LAG_PERIOD.start}–{LAG_PERIOD.stop}",
        fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig12_siconc_memory.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote fig12_siconc_memory.png")


# ============================================================
# Supplementary Fig S1: same as Fig 12 but restricted to the
# *seasonal-ice zone*, defined as cells whose time-mean siconc
# fraction is between 0.05 and 0.95.
#
# Why: in Fig 12 the basin mean integrates over a region that
# includes both year-round open water (siconc ≈ 0, no variability)
# and perennial ice (siconc ≈ 1, also no variability). Both kinds
# of cells artificially inflate the apparent autocorrelation —
# a series that is constant at any value has perfect (r = 1)
# memory by definition. Restricting to the seasonal-ice zone
# (cells that actually vary year-to-year) tests whether the Fig 12
# signal reflects genuine cross-year persistence or just the
# dominance of static cells in the basin mean.
#
# The mask is computed *per source*, so each model is restricted
# to its own seasonal zone — important because model sea-ice
# extent varies and a fixed observational mask would be unfair.
# ============================================================

SEASONAL_ZONE_LO = 0.05
SEASONAL_ZONE_HI = 0.95


def _seasonal_ice_mask(siconc_ts: xr.DataArray) -> xr.DataArray:
    """Boolean mask: True where time-mean siconc ∈ [0.05, 0.95]."""
    mean = siconc_ts.mean("time", skipna=True)
    return (mean >= SEASONAL_ZONE_LO) & (mean <= SEASONAL_ZONE_HI)


def make_figS1_siconc_memory_seasonal_zone(
    obs_siconc_ts: xr.DataArray | None,
    model_siconc_ts: dict[str, xr.DataArray],
    common_grid: xr.DataArray,
    lag_months: int = 12,
) -> None:
    """Supplementary Fig S1: 12-month autocorrelation of basin-mean
    siconc anomalies, restricted to the seasonal-ice zone of each
    source. Robustness check for the Fig 12 main result.
    """
    logger.info(f"=== Fig S1: siconc {lag_months}-month autocorrelation, "
                f"seasonal-ice zone only ===")

    all_masks = _build_masks(common_grid)
    basins = [b for b in SICONC_BASINS if b in all_masks]
    basin_masks = {b: all_masks[b] for b in basins}

    results: dict[str, dict[str, float]] = {}
    n_cells: dict[str, dict[str, int]] = {}

    def _basin_mean_with_seasonal_zone(
        siconc_on_grid: xr.DataArray,
        basin_mask: xr.DataArray,
    ) -> xr.DataArray:
        """Basin-mean time series restricted to cells that are both
        inside the basin AND inside the seasonal-ice zone of this
        particular source."""
        zone_mask = _seasonal_ice_mask(siconc_on_grid)
        combined  = basin_mask & zone_mask
        return _basin_mean_timeseries(siconc_on_grid, combined), int(combined.sum())

    # ---- OBS ----
    if obs_siconc_ts is not None:
        try:
            obs_re = regrid_timeseries(
                obs_siconc_ts.sel(time=LAG_PERIOD), common_grid)
            results["OBS"] = {}
            n_cells["OBS"] = {}
            for basin, mask in basin_masks.items():
                bm, n = _basin_mean_with_seasonal_zone(obs_re, mask)
                results["OBS"][basin] = _autocorr_at_lag(bm, lag_months)
                n_cells["OBS"][basin] = n
                logger.success(
                    f"  [OBS  ] {basin:10s}  r={results['OBS'][basin]:+.3f}  "
                    f"(seasonal-zone cells: {n})")
        except Exception as e:
            logger.warning(f"  OBS seasonal-zone autocorrelation failed: {e}")

    # ---- models ----
    for model in MODELS:
        if model not in model_siconc_ts:
            continue
        try:
            sic_re = regrid_timeseries(
                model_siconc_ts[model].sel(time=LAG_PERIOD), common_grid)
            results[model] = {}
            n_cells[model] = {}
            for basin, mask in basin_masks.items():
                bm, n = _basin_mean_with_seasonal_zone(sic_re, mask)
                results[model][basin] = _autocorr_at_lag(bm, lag_months)
                n_cells[model][basin] = n
                logger.success(
                    f"  [{model:9s}] {basin:10s}  "
                    f"r={results[model][basin]:+.3f}  "
                    f"(seasonal-zone cells: {n})")
        except Exception as e:
            logger.warning(f"  [{model}] seasonal-zone autocorrelation failed: {e}")

    if not results:
        logger.warning("  Fig S1 SKIPPED: no autocorrelations computed")
        return

    # ---- CSV ----
    csv_path = OUT_DIR / "figS1_siconc_memory_seasonal_zone.csv"
    with open(csv_path, "w") as fp:
        fp.write("source,basin,r_lag12,n_seasonal_cells\n")
        for source, basin_results in results.items():
            for basin, r in basin_results.items():
                n = n_cells[source][basin]
                fp.write(f"{source},{basin},{r:.4f},{n}\n")
    logger.success(f"  wrote {csv_path.name}")

    # ---- bar chart ----
    sources = list(results.keys())
    n_s = len(sources)
    n_b = len(basins)
    width = 0.78 / n_s
    x = np.arange(n_b)

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_b + 2), 5))

    for i, source in enumerate(sources):
        offset = (i - (n_s - 1) / 2) * width
        vals = [results[source].get(b, np.nan) for b in basins]
        if source == "OBS":
            color = "black"
        else:
            non_obs = [s for s in sources if s != "OBS"]
            color = PALETTE[non_obs.index(source) % len(PALETTE)] \
                    if source in non_obs else "gray"
        bars = ax.bar(x + offset, vals, width, label=source,
                      color=color, edgecolor="black", linewidth=0.4)
        for bar, v, basin in zip(bars, vals, basins):
            if not np.isfinite(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (0.02 if v >= 0 else -0.02), f"{v:+.2f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8, color=color, fontweight="bold")
            # Annotate per-source seasonal-cell counts under each bar
            n = n_cells[source][basin]
            ax.text(bar.get_x() + bar.get_width() / 2, -0.36,
                    f"N={n}", ha="center", va="top",
                    fontsize=6, color=color, alpha=0.7)

    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(basins, fontsize=10)
    ax.set_ylabel(f"r(siconc(t), siconc(t−{lag_months} mo))", fontsize=10)
    ax.set_ylim(-0.4, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    #ax.set_title(
    #    f"Fig S1 — {lag_months}-month autocorrelation of basin-mean "
    #    f"siconc anomalies\n"
    #    f"Restricted to the seasonal-ice zone of each source "
    #    f"({SEASONAL_ZONE_LO:.2f} ≤ mean siconc ≤ {SEASONAL_ZONE_HI:.2f}).  "
    #    f"Robustness check for Fig 12.\n"
    #    f"Period: {LAG_PERIOD.start}–{LAG_PERIOD.stop}",
    #    fontsize=11)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figS1_siconc_memory_seasonal_zone.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote figS1_siconc_memory_seasonal_zone.png")


# ============================================================
# Supplementary Fig S2: piControl internal-variability envelope
# for the 12-month sea-ice autocorrelation.
#
# Fig 12 reports a single autocorrelation value per model from one
# historical realisation. With ~200 strongly autocorrelated months
# the sampling uncertainty of that statistic is large, so a single
# value cannot by itself separate a structural model bias from a
# draw of internal variability. Fig S2 addresses this directly.
#
# For each model the piControl run (no historical forcing, typically
# many centuries long) is cut into overlapping 17-year segments — the
# same length as the 1998-2014 historical analysis window, so the
# segment statistic carries the same sampling uncertainty as the real
# diagnostic. The 12-month autocorrelation is recomputed on every
# segment with the identical deseasonalise -> detrend -> z-score
# pipeline used for Fig 12. The spread of those values is the
# internal-variability envelope.
#
# Interpretation: if the observed HadISST value lies OUTSIDE a model's
# piControl envelope, the historical mismatch is unlikely to be
# explained by internal variability alone, and the persistence
# underestimate can be read as a structural model property. If it lies
# INSIDE, a single historical member genuinely cannot distinguish
# structural bias from internal variability — which is itself a
# quantified, defensible statement of the limitation.
#
# Caveat: piControl has no historical forcing, so its mean sea-ice
# state can differ from the historical run's, and siconc autocorrelation
# depends on the mean ice state. The seasonal-ice-zone restriction
# (SEASONAL_ZONE_LO/HI), computed per segment, is applied to keep the
# comparison consistent with Fig S1.
# ============================================================

PICONTROL_SEGMENT_YEARS = 17        # matches the 1998-2014 window
PICONTROL_STEP_YEARS    = 1         # 1-year slide -> maximal overlap

# Cap on how many years of each contiguous piControl run are used.
# An internal-variability envelope does not need the full 1000-1200 yr
# of a control run: 400 yr already yields ~380 overlapping 17-yr
# segments. Capping here cuts the (expensive) regrid cost ~3x for the
# long MPI / CESM2 controls with no meaningful loss of envelope width.
PICONTROL_MAX_YEARS = 400


_DATERANGE_RE = re.compile(r"_(\d{6})-(\d{6})\.nc$")


def _parse_file_daterange(path: Path) -> tuple[int, int] | None:
    """Extract (start_YYYYMM, end_YYYYMM) as ints from a CMIP6
    filename, or None if the name does not match."""
    m = _DATERANGE_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _months_between(ym0: int, ym1: int) -> int:
    """Number of months from one YYYYMM to another (ym1 - ym0)."""
    y0, m0 = ym0 // 100, ym0 % 100
    y1, m1 = ym1 // 100, ym1 % 100
    return (y1 - y0) * 12 + (m1 - m0)


def _latest_version_dir(grid_dir: Path) -> Path | None:
    """Newest vYYYYMMDD directory under .../SImon/<var>/<grid>/.
    CMIP6 version dirs sort lexically by date, so max() is correct."""
    versions = [d for d in grid_dir.iterdir()
                if d.is_dir() and d.name.startswith("v")]
    return max(versions, default=None)


def _picontrol_siconc_paths(model: str) -> list[Path]:
    """Locate piControl sea-ice files for one model in the DKRZ pool.

    Handles the real diversity of the pool, confirmed by inspection:
      - the variable may be `siconc` (ocean grid) or `siconca`
        (atmosphere grid) -- accept whichever exists,
      - the grid label may be `gn` or `gr`,
      - several version directories may exist -> take the latest.

    Returns a date-sorted list of files from the single chosen
    variant / variable / grid / version. The siconc-vs-siconca choice
    is logged so it can be noted in the Fig S2 caption.
    """
    if model not in INSTITUTIONS:
        logger.warning(f"  [{model}] no institution mapping for piControl")
        return []
    base = CMIP6_ROOT / INSTITUTIONS[model] / model / "piControl"
    if not base.is_dir():
        logger.warning(f"  [{model}] no piControl directory: {base}")
        return []

    for variant_dir in sorted(base.iterdir()):
        simon = variant_dir / "SImon"
        if not simon.is_dir():
            continue
        for var_name in ("siconc", "siconca"):       # ocean grid, then atmos grid
            var_dir = simon / var_name
            if not var_dir.is_dir():
                continue
            for grid_dir in sorted(var_dir.iterdir()):  # gn or gr
                if not grid_dir.is_dir():
                    continue
                vdir = _latest_version_dir(grid_dir)
                if vdir is None:
                    continue
                files = sorted(vdir.glob("*.nc"))
                if files:
                    logger.info(f"  [{model}] piControl {var_name} "
                                f"({grid_dir.name}, {vdir.name}): "
                                f"{len(files)} file(s)")
                    dated = [(f, _parse_file_daterange(f)) for f in files]
                    dated = [(f, d) for f, d in dated if d is not None]
                    dated.sort(key=lambda fd: fd[1][0])
                    return [f for f, _ in dated]
    logger.warning(f"  [{model}] no piControl siconc/siconca files found")
    return []


def _contiguous_file_groups(paths: list[Path]) -> list[list[Path]]:
    """Split a date-sorted file list into time-contiguous groups.

    A new group starts wherever the next file does not begin exactly
    one month after the previous file ends. This is essential because
    the DKRZ piControl holdings arrive as scattered century chunks
    with multi-century gaps; sliding a 17-yr window blindly across a
    gap would yield a meaningless autocorrelation.
    """
    groups: list[list[Path]] = []
    current: list[Path] = []
    prev_end: int | None = None
    for path in paths:
        rng = _parse_file_daterange(path)
        if rng is None:
            continue
        start, end = rng
        if prev_end is None:
            current = [path]
        elif _months_between(prev_end, start) == 1:
            current.append(path)
        else:
            if current:
                groups.append(current)
            current = [path]
        prev_end = end
    if current:
        groups.append(current)
    return groups


def load_picontrol_siconc_groups(model: str) -> list[xr.DataArray]:
    """Open piControl sea ice for one model as a list of time-CONTIGUOUS
    DataArrays (one per gap-free run). Each is a fraction in [0, 1] on
    the model's native grid. Returns [] if nothing is available.

    Uses combine='nested', concat_dim='time': the files are already
    date-sorted, and nested concat never inspects the spatial
    coordinates -- this avoids the 'coordinate y has no index' error
    that combine='by_coords' throws on regridded (gr) sea-ice grids.
    """
    paths = _picontrol_siconc_paths(model)
    if not paths:
        return []
    groups = _contiguous_file_groups(paths)
    logger.info(f"  [{model}] piControl: {len(paths)} file(s) -> "
                f"{len(groups)} contiguous run(s)")
    out: list[xr.DataArray] = []
    for gi, group in enumerate(groups):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = xr.open_mfdataset([str(p) for p in group],
                                       combine="nested", concat_dim="time",
                                       use_cftime=True)
            var = next((v for v in ("siconc", "siconca")
                        if v in ds.data_vars), None)
            if var is None:
                var = _pick_siconc_var(ds)
            da = ds[var].squeeze()
            da = standardize(da)
            if float(np.nanmax(da.values)) > 1.5:     # percent -> fraction
                da = da / 100.0
            da = da.where((da >= 0) & (da <= 1))

            # --- cap run length (speed) ---
            # Trim each contiguous run to at most PICONTROL_MAX_YEARS.
            # 400 yr still yields ~380 overlapping 17-yr segments; the
            # full 1000-1200 yr controls would triple the regrid cost
            # for no meaningful gain in envelope width.
            max_months = PICONTROL_MAX_YEARS * 12
            if da.sizes.get("time", 0) > max_months:
                da = da.isel(time=slice(0, max_months))

            # --- synthetic time axis (calendar-overflow fix) ---
            # piControl years are arbitrary model years, and the long
            # MPI/CESM2 controls run past 2262, the pandas datetime64
            # ceiling. Any downstream pandas conversion then throws
            # "Out of bounds timestamp". The diagnostics only need
            # month-of-year and ordering, so replace the time axis
            # with a safe synthetic monthly index on a noleap calendar
            # starting at year 1. This makes the run calendar-agnostic.
            n_t = da.sizes.get("time", 0)
            try:                                  # newer xarray
                synthetic = xr.date_range(
                    start="0001-01-01", periods=n_t, freq="MS",
                    calendar="noleap", use_cftime=True)
            except (AttributeError, TypeError):   # older xarray
                synthetic = xr.cftime_range(
                    start="0001-01-01", periods=n_t, freq="MS",
                    calendar="noleap")
            da = da.assign_coords(time=synthetic)

            n_months = da.sizes.get("time", 0)
            logger.success(f"    run {gi+1}: {n_months} months "
                            f"(~{n_months // 12} yr, capped at "
                            f"{PICONTROL_MAX_YEARS})")
            out.append(da)
        except Exception as e:
            logger.warning(f"  [{model}] piControl run {gi+1} "
                            f"failed to open: {e}")
    return out


def _picontrol_segment_autocorrelations(
    model: str,
    common_grid: xr.DataArray,
    basin_masks: dict[str, xr.DataArray],
    lag_months: int,
) -> dict[str, list[float]]:
    """For one model: load piControl sea ice as contiguous runs, slide
    a 17-yr window WITHIN each run, and compute the detrended 12-month
    siconc autocorrelation on the seasonal-ice zone for every basin.
    Segments never cross a gap.

    Returns {basin: [autocorrelation values across all segments]}.
    """
    runs = load_picontrol_siconc_groups(model)
    out: dict[str, list[float]] = {b: [] for b in basin_masks}
    if not runs:
        return out

    seg_len = PICONTROL_SEGMENT_YEARS * 12
    step    = PICONTROL_STEP_YEARS * 12
    total_segments = 0

    for run in runs:
        n_months = run.sizes.get("time", 0)
        if n_months < seg_len:
            continue                                  # run too short
        try:
            run_re = regrid_timeseries(run, common_grid)
        except Exception as e:
            logger.warning(f"  [{model}] piControl run regrid failed: {e}")
            continue
        for s0 in range(0, n_months - seg_len + 1, step):
            seg = run_re.isel(time=slice(s0, s0 + seg_len))
            # the seasonal-ice zone depends only on the segment, not
            # the basin -- compute it once per segment, not per basin
            zone = _seasonal_ice_mask(seg)
            for basin, bmask in basin_masks.items():
                combined = bmask & zone
                if int(combined.sum()) < 1:
                    continue
                bm = _basin_mean_timeseries(seg, combined)
                r  = _autocorr_at_lag(bm, lag_months)  # detrended pipeline
                if np.isfinite(r):
                    out[basin].append(r)
            total_segments += 1

    logger.success(f"  [{model}] {total_segments} valid "
                    f"{PICONTROL_SEGMENT_YEARS}-yr segment(s) across "
                    f"{len(runs)} contiguous run(s)")
    for basin, vals in out.items():
        logger.info(f"    {basin:13s}: {len(vals)} estimate(s)")
    return out


def make_figS2_picontrol_envelope(
    obs_siconc_ts: xr.DataArray | None,
    common_grid: xr.DataArray,
    lag_months: int = 12,
) -> None:
    """Supplementary Fig S2: piControl internal-variability envelope of
    the 12-month sea-ice autocorrelation.

    One violin per model = the distribution of the Fig 12 statistic
    over overlapping 17-yr piControl segments. The observed HadISST
    value (computed with the same detrended pipeline) is drawn as a
    horizontal reference line. A companion CSV records the 2.5/50/97.5
    percentiles and whether the observed value falls inside each
    model's envelope.
    """
    logger.info("=== Fig S2: piControl internal-variability envelope ===")
    if common_grid is None:
        logger.warning("  Fig S2 SKIPPED: no common grid (ORAS5 not loaded)")
        return

    all_masks = _build_masks(common_grid)
    basins = [b for b in SICONC_BASINS if b in all_masks]
    masks  = {b: all_masks[b] for b in basins}

    # ---- observed reference, detrended seasonal-zone pipeline ----
    obs_vals: dict[str, float] = {}
    if obs_siconc_ts is not None:
        try:
            obs_re = regrid_timeseries(
                obs_siconc_ts.sel(time=LAG_PERIOD), common_grid)
            for basin, mask in masks.items():
                zone     = _seasonal_ice_mask(obs_re)
                combined = mask & zone
                bm = _basin_mean_timeseries(obs_re, combined)
                obs_vals[basin] = _autocorr_at_lag(bm, lag_months)
                logger.success(f"  [OBS] {basin:13s} detrended r = "
                                f"{obs_vals[basin]:+.3f}")
        except Exception as e:
            logger.warning(f"  OBS detrended autocorrelation failed: {e}")

    # ---- per-model piControl segment distributions ----
    # dist[basin][model] = list of segment autocorrelations
    dist: dict[str, dict[str, list[float]]] = {b: {} for b in basins}
    for model in MODELS:
        try:
            seg = _picontrol_segment_autocorrelations(
                model, common_grid, masks, lag_months)
            for basin in basins:
                dist[basin][model] = seg.get(basin, [])
        except Exception as e:
            logger.warning(f"  [{model}] piControl segmenting failed: {e}")

    if not any(dist[b] for b in basins):
        logger.warning("  Fig S2 SKIPPED: no piControl distributions built")
        return

    # ---- CSV: percentiles + observed, per model per basin ----
    csv_path = OUT_DIR / "figS2_picontrol_envelope_table.csv"
    with open(csv_path, "w") as fp:
        fp.write("basin,model,n_segments,p2.5,p50,p97.5,obs,"
                 "obs_inside_envelope\n")
        for basin in basins:
            for model, vals in dist[basin].items():
                if not vals:
                    continue
                a = np.asarray(vals)
                lo, mid, hi = np.percentile(a, [2.5, 50, 97.5])
                obs = obs_vals.get(basin, np.nan)
                inside = (lo <= obs <= hi) if np.isfinite(obs) else ""
                fp.write(f"{basin},{model},{len(vals)},{lo:.4f},"
                         f"{mid:.4f},{hi:.4f},{obs:.4f},{inside}\n")
    logger.success(f"  wrote {csv_path.name}")

    # ---- figure: one panel per sea-ice basin ----
    n_panels = len(basins)
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(5.2 * n_panels, 5.2),
                             squeeze=False)
    axes = axes[0]

    for ax, basin in zip(axes, basins):
        model_order = [m for m in MODELS if dist[basin].get(m)]
        data = [np.asarray(dist[basin][m]) for m in model_order]
        if not data:
            ax.set_visible(False)
            continue

        positions = np.arange(len(model_order)) + 1
        vp = ax.violinplot(data, positions=positions, widths=0.75,
                           showextrema=False, showmedians=False)
        for i, body in enumerate(vp["bodies"]):
            body.set_facecolor(PALETTE[i % len(PALETTE)])
            body.set_edgecolor("black")
            body.set_alpha(0.55)

        # 95% interval whisker + median tick per model
        for pos, vals in zip(positions, data):
            lo, mid, hi = np.percentile(vals, [2.5, 50, 97.5])
            ax.plot([pos, pos], [lo, hi], color="black", lw=1.2, zorder=3)
            ax.plot([pos - 0.18, pos + 0.18], [mid, mid],
                    color="black", lw=1.6, zorder=3)

        # observed value as a horizontal reference line
        obs = obs_vals.get(basin, np.nan)
        if np.isfinite(obs):
            ax.axhline(obs, color="crimson", lw=1.8, ls="--", zorder=2,
                       label=f"HadISST observed ({obs:+.2f})")
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        ax.set_xticks(positions)
        ax.set_xticklabels(model_order, rotation=20, ha="right", fontsize=9)
        ax.set_title(basin, fontsize=12, fontweight="bold")
        ax.set_ylabel(f"r(siconc(t), siconc(t−{lag_months} mo))",
                      fontsize=10)
        ax.axhline(0, color="gray", lw=0.6)
        ax.set_ylim(-0.6, 1.0)
        ax.grid(axis="y", alpha=0.3)

    #fig.suptitle(
    #    "Fig S2 — piControl internal-variability envelope of the "
    #    f"{lag_months}-month sea-ice autocorrelation\n"
    #    f"Violins: statistic over overlapping "
    #    f"{PICONTROL_SEGMENT_YEARS}-yr piControl segments  |  "
    #    f"dashed line: observed (HadISST, "
    #    f"{LAG_PERIOD.start}–{LAG_PERIOD.stop})\n"
    #    f"Observed value outside a model's envelope -> historical "
    #    f"mismatch unlikely to be internal variability alone",
    #    fontsize=11, y=1.04)
    #fig.tight_layout()
    #fig.savefig(OUT_DIR / "figS2_picontrol_envelope.png",
    #            dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote figS2_picontrol_envelope.png")


# ============================================================
# Ensemble-member audit (standalone utility)
#
# Lists, per model, the historical (or piControl) variant labels in
# the DKRZ pool that have ALL variables the diagnostic needs. The
# intersection across variables is the figure that matters: a model
# may publish many `tos` members but only a few `chl` members.
#
# Run from the command line:
#   python example_chl_corrected.py --audit
#   python example_chl_corrected.py --audit piControl
# ============================================================

_AUDIT_REQUIRED = [
    ("Omon",  "chl"),
    ("Omon",  "mlotst"),
    ("Omon",  "tos"),
    ("Amon",  "clt"),
    ("SImon", "siconc"),
]


def _audit_variants_for(model: str, experiment: str,
                        table: str, var: str) -> set[str]:
    """Set of variant labels available for one model/experiment/table/var."""
    if model not in INSTITUTIONS:
        return set()
    base = CMIP6_ROOT / INSTITUTIONS[model] / model / experiment
    if not base.is_dir():
        return set()
    found = set()
    for variant_dir in base.iterdir():
        if not variant_dir.is_dir():
            continue
        var_path = variant_dir / table / var
        if var_path.is_dir() and any(var_path.rglob("*.nc")):
            found.add(variant_dir.name)
    return found


def audit_ensemble_members(experiment: str = "historical") -> dict[str, set]:
    """Print, per model, the variant labels with ALL required variables.

    Returns {model: set_of_complete_variants}.
    """
    logger.info(f"=== Ensemble-member audit: experiment='{experiment}' ===")
    complete: dict[str, set] = {}
    for model in INSTITUTIONS:
        per_var: dict[str, set] = {}
        for table, var in _AUDIT_REQUIRED:
            per_var[f"{table}/{var}"] = _audit_variants_for(
                model, experiment, table, var)
        sets = list(per_var.values())
        inter = set.intersection(*sets) if all(sets) else set()
        complete[model] = inter
        logger.info(f"  {model}")
        for key, s in per_var.items():
            logger.info(f"      {key:14s}: {len(s):3d} variant(s)")
        if inter:
            shown = ", ".join(sorted(inter)[:12])
            extra = "" if len(inter) <= 12 else f"  (+{len(inter)-12} more)"
            logger.success(f"      COMPLETE (all vars): {len(inter):3d}"
                            f"  -> {shown}{extra}")
        else:
            logger.warning("      COMPLETE (all vars):   0  "
                            "-> no variant has every variable")
    logger.info("  ---- summary ----")
    for model, inter in complete.items():
        logger.info(f"  {model:18s}  usable members: {len(inter)}")
    return complete


# ============================================================
# Fig limitation: surface rsds bias + integrated dfe spread +
#                 scatter of inter-model spread Δlimiter vs ΔChl
# ============================================================
# This is a 12-panel figure (5 + 5 + 2) addressing whether the
# bottom-up biogeochemical limiters (light and iron) explain the
# inter-model spread in surface Chl-a that the physical-coupling
# analysis (Section 3.5) could not.
#
# Row 1: model surface rsds bias against ISCCP-FH (surface SW).
#        Falls back to (model - ensemble mean) if the obs is absent.
# Row 2: depth-integrated dfe(0-100m) as % anomaly from the
#        five-model ensemble mean. There is no dfe obs reference;
#        ensemble-mean anomaly is the honest baseline.
# Row 3: inter-model std at each grid cell. Per-cell scatter of
#        std(ΔdfeBias) vs std(ΔChlBias)  and
#        std(ΔrsdsBias) vs std(ΔChlBias). A positive correlation
#        means cells where the models disagree more on the
#        limiter are also cells where they disagree more on Chl.
# ============================================================

def make_fig_limitation(
    model_clim: dict[str, xr.DataArray],
    obs_clim: xr.DataArray,
    dfe_clim: dict[str, xr.DataArray],
    rsds_clim: dict[str, xr.DataArray],
    rsds_obs_clim: xr.DataArray | None,
) -> None:
    """Build the 3-row, 12-panel limitation-factor figure."""
    from scipy.stats import spearmanr

    logger.info("Building Fig limitation (rsds + dfe + scatter)...")

    if not dfe_clim and not rsds_clim:
        logger.warning("  no dfe or rsds data; skipping")
        return

    # Use a consistent model ordering for both rows
    models = [m for m in MODELS if m in dfe_clim or m in rsds_clim]
    n_models = len(models)
    if n_models == 0:
        logger.warning("  no models with limiter data; skipping")
        return

    # Build the figure: 3 rows. Row 1: n_models rsds panels.
    # Row 2: n_models dfe panels. Row 3: 2 scatter panels.
    # We use a GridSpec so row 3 can have a different number of cols.
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(4.0 * n_models, 11.5))
    gs = GridSpec(
        3, n_models, figure=fig,
        height_ratios=[1.0, 1.0, 1.1],
        hspace=0.32, wspace=0.12,
    )

    # --- helper to put a map panel ---
    def _map_panel(ax, da, vmin, vmax, cmap, title, units_label):
        im = ax.pcolormesh(
            da.lon, da.lat, da.values,
            vmin=vmin, vmax=vmax, cmap=cmap,
            transform=ccrs.PlateCarree(), shading="auto",
        )
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        ax.coastlines(linewidth=0.4)
        ax.set_title(title, fontsize=10)
        return im

    # =====================================================
    # Row 1: rsds bias maps
    # =====================================================
    if rsds_clim:
        # Decide reference: surface-SW obs (ISCCP-FH) if present, else ensemble mean
        if rsds_obs_clim is not None:
            # Regrid the surface-SW obs to the chl obs grid (common grid)
            ref_rsds = regrid_to_obs(rsds_obs_clim, obs_clim)
            ref_label = RSDS_OBS_NAME
        else:
            # Compute the multi-model ensemble mean on the chl grid
            rsds_on_chl = {
                m: regrid_to_obs(d, obs_clim) for m, d in rsds_clim.items()
            }
            stack = xr.concat(list(rsds_on_chl.values()), dim="model")
            ref_rsds = stack.mean(dim="model")
            ref_label = "ensemble mean"

        rsds_bias = {}
        for m in models:
            if m not in rsds_clim:
                continue
            m_re = regrid_to_obs(rsds_clim[m], obs_clim)
            rsds_bias[m] = m_re - ref_rsds

        for col, model in enumerate(models):
            ax = fig.add_subplot(gs[0, col], projection=ccrs.Robinson(
                central_longitude=180))
            if model in rsds_bias:
                da = rsds_bias[model]
                im = _map_panel(
                    ax, da, vmin=-40, vmax=40, cmap="RdBu_r",
                    title=f"{model}\nrsds bias vs {ref_label}",
                    units_label="W m$^{-2}$"
                )
                _w = np.cos(np.deg2rad(da.lat))
                mean_bias = float(da.weighted(_w).mean(("lat", "lon")).values)
                rmse = float(np.sqrt(
                    (da ** 2).weighted(_w).mean(("lat", "lon"))).values)
                ax.text(0.02, 0.04,
                        f"mean={mean_bias:+.1f}  rmse={rmse:.1f}",
                        transform=ax.transAxes, fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", alpha=0.85))
                if col == n_models - 1:
                    cbar = fig.colorbar(
                        im, ax=ax, orientation="vertical",
                        fraction=0.04, pad=0.04, extend="both"
                    )
                    cbar.set_label("rsds bias (W m$^{-2}$)", fontsize=8)
            else:
                ax.set_title(f"{model}\n(no rsds)", fontsize=9)
            add_panel_label(ax, col)

    # =====================================================
    # Row 2: dfe relative anomaly from ensemble mean
    # =====================================================
    if dfe_clim:
        dfe_on_chl = {
            m: regrid_to_obs(d, obs_clim) for m, d in dfe_clim.items()
        }
        stack = xr.concat(list(dfe_on_chl.values()), dim="model")
        ens_mean_dfe = stack.mean(dim="model")
        # Relative anomaly (%) — avoids the huge dynamic range of dfe
        dfe_rel = {}
        for m, d in dfe_on_chl.items():
            denom = ens_mean_dfe.where(ens_mean_dfe > 1e-6)
            dfe_rel[m] = (d - ens_mean_dfe) / denom * 100.0

        for col, model in enumerate(models):
            ax = fig.add_subplot(gs[1, col], projection=ccrs.Robinson(
                central_longitude=180))
            if model in dfe_rel:
                da = dfe_rel[model]
                im = _map_panel(
                    ax, da, vmin=-100, vmax=100, cmap="PuOr_r",
                    title=f"{model}\ndfe(0-100m) % anomaly",
                    units_label="%"
                )
                if col == n_models - 1:
                    cbar = fig.colorbar(
                        im, ax=ax, orientation="vertical",
                        fraction=0.04, pad=0.04, extend="both"
                    )
                    cbar.set_label(
                        "dfe anomaly vs ensemble mean (%)", fontsize=8)
            else:
                ax.set_title(f"{model}\n(no dfe)", fontsize=9)
            add_panel_label(ax, n_models + col)

    # =====================================================
    # Row 3: scatter of inter-model spread Δlimiter vs ΔChl
    # =====================================================
    # Build the Chl-bias spread map: at each cell, std across the
    # 5 model Chl biases against ESA-CCI. Same for dfe and rsds
    # (std across the model anomalies from ensemble mean / ISCCP-FH).
    chl_bias_stack_list = []
    for m in models:
        if m not in model_clim:
            continue
        bias = model_clim[m] - obs_clim
        chl_bias_stack_list.append(bias)
    if not chl_bias_stack_list:
        logger.warning("  no Chl bias data; skipping row 3")
    else:
        chl_bias_std = xr.concat(chl_bias_stack_list, dim="model").std(
            dim="model")

        # --- scatter: dfe spread vs Chl spread ---
        if dfe_clim:
            dfe_anom_list = [
                v.mean(dim="time") if "time" in v.dims else v
                for v in dfe_rel.values()
            ]
            dfe_std = xr.concat(dfe_anom_list, dim="model").std(dim="model")

            ax = fig.add_subplot(gs[2, 0 : max(1, n_models // 2)])
            x = chl_bias_std.values.ravel()
            y = dfe_std.values.ravel()
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if x.size > 50:
                ax.hexbin(x, y, gridsize=40, cmap="viridis",
                          bins="log", mincnt=1)
                rho, pval = spearmanr(x, y)
                ax.text(
                    0.03, 0.97,
                    f"$\\rho$ = {rho:+.3f}\np = {pval:.2g}\nN = {ok.sum():,}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", edgecolor="black",
                              alpha=0.9))
                logger.success(
                    f"  scatter dfe: ρ(σ_dfe,σ_Chl) = {rho:+.3f}, "
                    f"p = {pval:.2g}")
            ax.set_xlabel("Inter-model std of Chl-$a$ bias [mg m$^{-3}$]")
            ax.set_ylabel("Inter-model std of dfe(0-100m) anomaly [%]")
            ax.set_title("(k)  dfe spread vs Chl spread",
                         fontsize=10, fontweight="bold")
            ax.grid(alpha=0.3)

        # --- scatter: rsds spread vs Chl spread ---
        if rsds_clim:
            rsds_anom_list = [
                v.mean(dim="time") if "time" in v.dims else v
                for v in rsds_bias.values()
            ]
            rsds_std = xr.concat(rsds_anom_list, dim="model").std(dim="model")

            ax = fig.add_subplot(gs[2, max(1, n_models // 2):])
            x = chl_bias_std.values.ravel()
            y = rsds_std.values.ravel()
            ok = np.isfinite(x) & np.isfinite(y)
            x, y = x[ok], y[ok]
            if x.size > 50:
                ax.hexbin(x, y, gridsize=40, cmap="viridis",
                          bins="log", mincnt=1)
                rho, pval = spearmanr(x, y)
                ax.text(
                    0.03, 0.97,
                    f"$\\rho$ = {rho:+.3f}\np = {pval:.2g}\nN = {ok.sum():,}",
                    transform=ax.transAxes, fontsize=9, va="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", edgecolor="black",
                              alpha=0.9))
                logger.success(
                    f"  scatter rsds: ρ(σ_rsds,σ_Chl) = {rho:+.3f}, "
                    f"p = {pval:.2g}")
            ax.set_xlabel("Inter-model std of Chl-$a$ bias [mg m$^{-3}$]")
            ax.set_ylabel("Inter-model std of rsds bias [W m$^{-2}$]")
            ax.set_title("(l)  rsds spread vs Chl spread",
                         fontsize=10, fontweight="bold")
            ax.grid(alpha=0.3)

    fig.suptitle(
        f"Limiting factors: rsds bias (row 1), dfe(0-100m) anomaly "
        f"(row 2), inter-model spread vs Chl spread (row 3)\n"
        f"Period: {PERIOD}",
        fontsize=11, y=0.995)

    out = OUT_DIR / "fig_limitation.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {out.name}")


# ============================================================
# Light (rsds) diagnostics: bias map + polar coupling
# ============================================================
# Two figures driven by surface downwelling shortwave ("the light"):
#   fig_light_bias_isccp.png       per-model rsds bias vs ISCCP-FH
#   fig_light_polar_coupling.png   lag corr of rsds anomalies vs Chl-a,
#                                  polar basins (single column)
# The sea-ice driver is no longer a column here; it is shown separately
# as a model-minus-HadISST bias map (fig_siconc_bias.png, built in
# Phase E).
# Both reuse the existing loaders, regridders, masks and the Section 2.6
# deseasonalise/detrend/z-score lag-correlation machinery.

# Polar basins for the coupling figure. These names must exist in BASINS.
# "Arctic" (60-80 N) and "Southern-SIZ" (70-55 S) are the two genuine
# seasonal-ice zones where a light / ice / productivity coupling is
# physically meaningful. Add "Southern" (-80..-30) or the ice-free
# "Southern-ACC" (negative control) if desired.
POLAR_LIGHT_BASINS = ["Arctic", "Southern-SIZ"]

# Driver columns of the coupling figure (rsds is the driver / x leads).
# Sea ice ("siconc") was removed from this figure; it is now shown as a
# separate model-minus-HadISST bias map (see fig_siconc_bias.png). The
# siconc series is still loaded and used internally by
# _source_polar_light_curves to build the ocean mask, but is no longer
# plotted as a column here.
LIGHT_COL_DRIVERS = ("Chl",)


def _month_key(ts: xr.DataArray) -> xr.DataArray:
    """Replace the time coordinate with an integer (year*12 + month-1).

    rsds and siconc are loaded with use_cftime=True (cftime axis) while
    obs chl / obs siconc / ISCCP-FH are datetime64. xr.align cannot join
    a CFTimeIndex against a DatetimeIndex, so cross-variable correlation
    must align on a calendar-agnostic key. All series are already snapped
    to the first of the month, so a plain year-month integer is
    unambiguous and lets _lag_correlation_with_bootstrap align correctly
    regardless of the underlying time type or calendar.
    """
    import cftime
    import pandas as pd
    keys = []
    for t in np.asarray(ts.time.values):
        if isinstance(t, cftime.datetime):
            y, m = t.year, t.month
        else:
            p = pd.Timestamp(t)
            y, m = p.year, p.month
        keys.append(y * 12 + (m - 1))
    return ts.assign_coords(time=("time", np.asarray(keys, dtype=int)))


def _aw_stats(da2d: xr.DataArray) -> tuple[float, float]:
    """cos(lat)-weighted mean and RMSE of a (lat, lon) bias field."""
    w = np.cos(np.deg2rad(da2d.lat))
    mean = float(da2d.weighted(w).mean(("lat", "lon")))
    rmse = float(np.sqrt((da2d ** 2).weighted(w).mean(("lat", "lon"))))
    return mean, rmse


def make_light_bias_map(
    obs_rsds_ts: xr.DataArray,
    model_rsds_ts: dict[str, xr.DataArray],
    vlim: float = 40.0,
) -> None:
    """Per-model surface-SW (rsds) bias vs the surface-SW obs (ISCCP-FH).

    This is the standalone "bias map of the light": the same quantity as
    row 1 of make_fig_limitation, but on its own canvas with
    area-weighted mean and RMSE so the summary numbers are not
    pole-dragged.
    """
    logger.info(f"=== Light bias map: model rsds vs {RSDS_OBS_NAME} ===")
    models = [m for m in MODELS if m in model_rsds_ts]
    if not models:
        logger.warning("  no model rsds available; skipping bias map")
        return

    common_grid = xr.DataArray(
        np.zeros((obs_rsds_ts.sizes["lat"], obs_rsds_ts.sizes["lon"])),
        dims=("lat", "lon"),
        coords={"lat": obs_rsds_ts.lat, "lon": obs_rsds_ts.lon},
    )
    obs_clim = regrid_to_obs(obs_rsds_ts.mean("time", keep_attrs=True),
                             common_grid)

    n = len(models)
    fig = plt.figure(figsize=(3.6 * n, 4.2))
    im = None
    for col, model in enumerate(models):
        ax = fig.add_subplot(1, n, col + 1,
                             projection=ccrs.Robinson(central_longitude=180))
        clim = regrid_to_obs(
            model_rsds_ts[model].mean("time", keep_attrs=True), common_grid)
        bias = clim - obs_clim
        im = ax.pcolormesh(
            bias.lon, bias.lat, bias.values,
            vmin=-vlim, vmax=vlim, cmap="RdBu_r",
            transform=ccrs.PlateCarree(), shading="auto",
        )
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        ax.coastlines(linewidth=0.4)
        ax.set_title(f"{model}\nrsds bias vs {RSDS_OBS_NAME}", fontsize=10)
        mean_bias, rmse = _aw_stats(bias)
        ax.text(0.02, 0.04, f"mean={mean_bias:+.1f}  rmse={rmse:.1f}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="white", alpha=0.85))
        add_panel_label(ax, col)

    if im is not None:
        cbar = fig.colorbar(im, ax=fig.axes, orientation="horizontal",
                            fraction=0.045, pad=0.08, extend="both")
        cbar.set_label(r"rsds bias (W m$^{-2}$)", fontsize=9)

    fig.suptitle(
        f"Surface shortwave (light) bias vs {RSDS_OBS_NAME}   |   "
        f"area-weighted mean & RMSE   |   period: {PERIOD}",
        fontsize=11, y=1.02,
    )
    fig.savefig(OUT_DIR / "fig_light_bias_isccp.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote fig_light_bias_isccp.png")


def _light_ocean_mask(sic_re: xr.DataArray) -> xr.DataArray:
    """Ocean = any cell where regridded siconc is ever finite.

    HadISST / CMIP6 siconc is finite over ocean (incl. ice-free ocean,
    where it is 0) and NaN over land, so this is a clean polar land-sea
    mask that keeps the rsds basin mean ocean-only and comparable to the
    (ocean-only) Chl and siconc fields.
    """
    return sic_re.notnull().any("time")


def _source_polar_light_curves(
    chl_ts: xr.DataArray,
    rsds_ts: xr.DataArray,
    sic_ts: xr.DataArray,
    masks: dict[str, xr.DataArray],
    common_grid: xr.DataArray,
) -> dict[str, dict[str, tuple]]:
    """For one source, return {basin: {"Chl": curve, "siconc": curve}}.

    Each curve is (lags, r, ci_lo, ci_hi, pvalue) with the convention
    that positive lag means rsds (light) LEADS the target.
    """
    # Regridding is the hard prerequisite: if it fails the whole source
    # cannot be processed, so let it propagate to the caller (which logs
    # the traceback and which inputs were present).
    chl_re  = regrid_timeseries(chl_ts.sel(time=LAG_PERIOD),  common_grid)
    rsds_re = regrid_timeseries(rsds_ts.sel(time=LAG_PERIOD), common_grid)
    sic_re  = regrid_timeseries(sic_ts.sel(time=LAG_PERIOD),  common_grid)
    ocean   = _light_ocean_mask(sic_re)

    lags_axis = np.arange(-LAG_MAX, LAG_MAX + 1)

    def _nan_curve() -> tuple:
        nan = np.full(lags_axis.shape, np.nan, dtype=float)
        return (lags_axis, nan.copy(), nan.copy(), nan.copy(), nan.copy())

    out: dict[str, dict[str, tuple]] = {}
    for basin, mask in masks.items():
        # Per-basin resilience: a failure in one polar basin (e.g. a
        # source with no finite Chl under perennial ice there) must not
        # discard the source's other basins. Emit a NaN curve instead.
        try:
            rsds_b = _basin_mean_timeseries(rsds_re, mask & ocean)
            chl_b  = _basin_mean_timeseries(chl_re,  mask)
            sic_b  = _basin_mean_timeseries(sic_re,  mask)

            rsds_z = _month_key(_deseasonalise_zscore(rsds_b))
            chl_z  = _month_key(_deseasonalise_zscore(chl_b))
            sic_z  = _month_key(_deseasonalise_zscore(sic_b))

            # _lag_correlation_with_bootstrap(target, driver): positive lag
            # = driver leads target. Driver = rsds (light) in both columns.
            out[basin] = {
                "Chl":    _lag_correlation_with_bootstrap(chl_z, rsds_z),
                "siconc": _lag_correlation_with_bootstrap(sic_z, rsds_z),
            }
        except Exception as e:
            logger.warning(f"    basin {basin} failed: {e}")
            out[basin] = {"Chl": _nan_curve(), "siconc": _nan_curve()}
    return out


def _draw_light_lag_panel(ax, panel_curves: dict[str, tuple],
                          models_in: list[str],
                          ylabel: str | None) -> tuple:
    """Draw OBS + model lag curves into one panel (Fig 11 style)."""
    ax.axhline(0, color="k", lw=0.5, alpha=0.6)
    ax.axvline(0, color="k", lw=0.5, alpha=0.6, ls=":")

    sources = sorted(panel_curves.keys(), key=lambda s: (s == "OBS", s))
    peak_lines = []
    for source in sources:
        lags, rs, ci_lo, ci_hi, pval = panel_curves[source]
        if source == "OBS":
            color, lw, zord = "black", 2.4, 10
        else:
            color = (PALETTE[models_in.index(source) % len(PALETTE)]
                     if source in models_in else "gray")
            lw, zord = 1.4, 5

        if np.any(np.isfinite(ci_lo)) and np.any(np.isfinite(ci_hi)):
            ax.fill_between(lags, ci_lo, ci_hi, color=color,
                            alpha=0.12, lw=0, zorder=zord - 1)
        ax.plot(lags, rs, lw=lw, color=color, label=source,
                alpha=0.9, zorder=zord)

        sig = np.isfinite(pval) & (pval < LAG_BOOT_ALPHA)
        if np.any(sig):
            ax.plot(lags[sig], rs[sig], "o",
                    ms=5.5 if source == "OBS" else 3.5,
                    mfc=color, mec="white", mew=0.6, zorder=zord + 1)

        if np.any(np.isfinite(rs)):
            j = int(np.nanargmax(np.abs(rs)))
            k_peak, r_peak = int(lags[j]), float(rs[j])
            p_peak = float(pval[j]) if np.any(np.isfinite(pval)) else np.nan
            line = (f"{source}: r={r_peak:+.2f} @ {k_peak:+d}"
                    + (f" (p={p_peak:.3f})" if np.isfinite(p_peak) else ""))
            # With many models the inset box overflows; show OBS only and
            # log the model peaks instead (same as Fig 11).
            if len(sources) <= 4 or source == "OBS":
                peak_lines.append(line)
            else:
                logger.info(f"    {line}")

    ax.set_xlim(-LAG_MAX, LAG_MAX)
    ax.set_ylim(-1, 1)
    ax.set_xticks(np.arange(-LAG_MAX, LAG_MAX + 1, 4))
    ax.grid(alpha=0.25)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    # Per-panel peak-stat box removed (consistent with Figs 9-12); the
    # peak r / lag / p-value lines are still written to the run log.
    # if peak_lines:
    #     ax.text(0.02, 0.02, "\n".join(peak_lines), transform=ax.transAxes,
    #             fontsize=6.5, va="bottom", family="monospace",
    #             bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
    #                       edgecolor="gray", alpha=0.85))
    return ax.get_legend_handles_labels()


def make_fig_light_polar_coupling(
    obs_chl_ts: xr.DataArray,
    obs_rsds_ts: xr.DataArray | None,
    obs_siconc_ts: xr.DataArray | None,
    model_chl_ts: dict[str, xr.DataArray],
    model_rsds_ts: dict[str, xr.DataArray],
    model_siconc_ts: dict[str, xr.DataArray],
    common_grid: xr.DataArray,
) -> None:
    """Lag correlation of polar rsds (light) anomalies vs Chl-a and siconc.

    One row per polar basin, two columns (light vs Chl, light vs sea ice).
    OBS in black plus the models, with 95% block-bootstrap CI ribbons and
    filled markers where p < LAG_BOOT_ALPHA. Positive lag = light leads.
    """
    logger.info("=== Fig light-polar: rsds vs Chl & siconc lag correlation ===")

    all_masks = _build_masks(common_grid)
    basins = [b for b in POLAR_LIGHT_BASINS if b in all_masks]
    if not basins:
        logger.warning(f"  none of {POLAR_LIGHT_BASINS} in BASINS; skipping")
        return
    masks = {b: all_masks[b] for b in basins}

    # curves[basin][driver][source] = (lags, r, ci_lo, ci_hi, pval)
    curves: dict[str, dict[str, dict[str, tuple]]] = {
        b: {d: {} for d in LIGHT_COL_DRIVERS} for b in basins
    }

    # ---- OBS ----
    if obs_rsds_ts is not None and obs_siconc_ts is not None:
        try:
            logger.info("  [OBS  ] computing polar light couplings...")
            src = _source_polar_light_curves(
                obs_chl_ts, obs_rsds_ts, obs_siconc_ts, masks, common_grid)
            for b in basins:
                for d in LIGHT_COL_DRIVERS:
                    curves[b][d]["OBS"] = src[b][d]
        except Exception as e:
            logger.warning(f"  OBS light-coupling failed: {e}")
    else:
        logger.warning("  OBS rsds and/or siconc missing; no OBS curve")

    # ---- models ----
    for model in MODELS:
        if (model not in model_chl_ts or model not in model_rsds_ts
                or model not in model_siconc_ts):
            continue
        try:
            logger.info(f"  [{model:9s}] computing polar light couplings...")
            src = _source_polar_light_curves(
                model_chl_ts[model], model_rsds_ts[model],
                model_siconc_ts[model], masks, common_grid)
            for b in basins:
                for d in LIGHT_COL_DRIVERS:
                    curves[b][d][model] = src[b][d]
        except Exception as e:
            import traceback
            logger.warning(f"  [{model}] light-coupling failed: {e}  "
                           f"(chl={model in model_chl_ts}, "
                           f"rsds={model in model_rsds_ts}, "
                           f"siconc={model in model_siconc_ts})")
            logger.opt(colors=False).info(traceback.format_exc())

    # Report which models were dropped and why-class (missing input vs error)
    drawn = {m for b in basins for d in LIGHT_COL_DRIVERS for m in curves[b][d]
             if m != "OBS"}
    for model in MODELS:
        if model not in drawn:
            missing = [k for k, present in
                       (("chl", model in model_chl_ts),
                        ("rsds", model in model_rsds_ts),
                        ("siconc", model in model_siconc_ts)) if not present]
            reason = (f"missing input(s): {', '.join(missing)}" if missing
                      else "all inputs present -> failed during computation "
                           "(see traceback above)")
            logger.warning(f"  [{model}] absent from light-coupling fig: {reason}")

    if not any(curves[b][d] for b in basins for d in LIGHT_COL_DRIVERS):
        logger.warning("  Fig light-polar SKIPPED: no curves built")
        return

    models_in = [m for m in MODELS if any(
        m in curves[b][d] for b in basins for d in LIGHT_COL_DRIVERS)]

    n_rows = len(basins)
    n_cols = len(LIGHT_COL_DRIVERS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6.8 * n_cols, 4.6 * n_rows),
                             squeeze=False, sharex=True)
    col_title = {
        "Chl":    "light (rsds)  vs  Chl-a",
        "siconc": "light (rsds)  vs  sea ice (siconc)",
    }
    handles = labels = None
    panel = 0
    for r, basin in enumerate(basins):
        for c, driver in enumerate(LIGHT_COL_DRIVERS):
            ax = axes[r][c]
            add_panel_label(ax, panel)
            panel += 1
            ylabel = "Pearson r" if c == 0 else None
            h, l = _draw_light_lag_panel(
                ax, curves[basin][driver], models_in, ylabel)
            if h:
                handles, labels = h, l
            if r == 0:
                ax.set_title(col_title[driver], fontsize=11, fontweight="bold")
            if r == n_rows - 1:
                ax.set_xlabel("Lag (months)   —   light leads -> ", fontsize=9)
            ax.text(0.98, 0.97, basin, transform=ax.transAxes,
                    ha="right", va="top", fontsize=10, fontweight="bold",
                    color="#333333")

    if handles is not None:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(handles), 6), fontsize=9,
                   framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Polar light coupling - rsds anomalies vs Chl-a\n"
        f"OBS: {RSDS_OBS_NAME} rsds, ESA-CCI Chl  |  "
        f"period {LAG_PERIOD.start}-{LAG_PERIOD.stop}  |  deseasonalised, "
        f"detrended, z-scored  |  ribbons: 95% block-bootstrap CI; "
        f"filled markers: p < {LAG_BOOT_ALPHA:.2f}",
        fontsize=11, y=1.0,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(OUT_DIR / "fig_light_polar_coupling.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote fig_light_polar_coupling.png")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    # ---- CLI shortcut: ensemble-member audit, then exit ----
    # Usage:  python example_chl_corrected.py --audit [historical|piControl]
    # Lists which variant labels in the DKRZ pool have every variable
    # the diagnostic needs. Does not run the figure pipeline.
    if len(sys.argv) > 1 and sys.argv[1] == "--audit":
        _exp = sys.argv[2] if len(sys.argv) > 2 else "historical"
        audit_ensemble_members(experiment=_exp)
        sys.exit(0)

    # =========================================================
    # PHASE A — mean state + seasonal cycle (Figs 1-4)
    # =========================================================
    logger.info("=== PHASE A: mean state + seasonal cycle ===")

    logger.info("Loading observations...")
    obs_full = load_obs_chl()
    obs_clim = obs_full.mean("time", keep_attrs=True)
    logger.info(f"  obs grid: {obs_clim.shape}  "
                f"max={float(np.nanmax(obs_clim.values)):.3f}")

    logger.info("Loading models...")
    model_clim: dict[str, xr.DataArray] = {}
    model_seasonal: dict[str, xr.DataArray] = {}
    model_chl_ts: dict[str, xr.DataArray] = {}   # native monthly TS for Figs 9, 10
    for model in MODELS:
        try:
            da = load_model_chl(model)
            model_chl_ts[model] = da
            clim_native = da.mean("time", keep_attrs=True)
            model_clim[model] = regrid_to_obs(clim_native, obs_clim)
            try:
                months = da.groupby("time.month").mean("time")
                regridded = []
                for m in months.month.values:
                    re = regrid_to_obs(months.sel(month=m), obs_clim)
                    regridded.append(re.expand_dims(month=[m]))
                model_seasonal[model] = xr.concat(regridded, dim="month")
            except Exception as e:
                logger.warning(f"  [{model}] seasonal regrid failed: {e}")
            logger.success(f"  loaded {model}")
        except FileNotFoundError as e:
            logger.warning(f"  skipped {model}: {e}")
        except Exception as e:
            logger.warning(f"  [{model}] failed: {e}")

    if not model_clim:
        raise RuntimeError("No model Chl NetCDFs found")

    # ---- Load dfe and rsds for the limitation analysis ----
    # We now keep the FULL monthly rsds (light) series, not just the
    # climatology, because the polar light-coupling figure needs it.
    dfe_clim: dict[str, xr.DataArray] = {}
    rsds_clim: dict[str, xr.DataArray] = {}
    model_rsds_ts: dict[str, xr.DataArray] = {}    # full monthly series
    for model in MODELS:
        try:
            d = load_model_dfe(model, z_max=100.0)
            if d is not None:
                dfe_clim[model] = d.mean("time", keep_attrs=True)
        except Exception as e:
            logger.warning(f"  [{model}] dfe load failed: {e}")
        try:
            r = load_model_rsds(model)
            if r is not None:
                model_rsds_ts[model] = r
                rsds_clim[model] = r.mean("time", keep_attrs=True)
        except Exception as e:
            logger.warning(f"  [{model}] rsds load failed: {e}")
    rsds_obs_clim = None
    obs_rsds_ts = None
    try:
        obs_rsds_ts = load_obs_rsds()
        if obs_rsds_ts is not None:
            rsds_obs_clim = obs_rsds_ts.mean("time", keep_attrs=True)
    except Exception as e:
        logger.warning(f"  {RSDS_OBS_NAME} rsds load failed: {e}")
    logger.info(
        f"  limitation data loaded: dfe for {len(dfe_clim)} models, "
        f"rsds for {len(rsds_clim)} models, "
        f"{RSDS_OBS_NAME} {'present' if rsds_obs_clim is not None else 'absent'}"
    )

    # ---- Standalone light (rsds) bias map vs the surface-SW obs ----
    if model_rsds_ts and obs_rsds_ts is not None:
        try:
            make_light_bias_map(obs_rsds_ts, model_rsds_ts)
        except Exception as e:
            import traceback
            logger.warning(f"  light bias map crashed: {e}")
            logger.opt(colors=False).warning(traceback.format_exc())

    make_fig1_and_fig2_combined(obs_clim, model_clim)
    if model_seasonal:
        make_fig3_hovmoller(
        obs_full=obs_full,
        model_seasonal=model_seasonal,
        obs_clim=obs_clim,
        )
    else:
        logger.warning("  Fig 3 Hovmöller SKIPPED: no model seasonal climatology")
    if model_seasonal:
        make_fig4(obs_full, model_seasonal)
    if dfe_clim or rsds_clim:
        try:
            make_fig_limitation(
                model_clim=model_clim,
                obs_clim=obs_clim,
                dfe_clim=dfe_clim,
                rsds_clim=rsds_clim,
                rsds_obs_clim=rsds_obs_clim,
            )
        except Exception as e:
            import traceback
            logger.warning(f"  Fig limitation crashed: {e}")
            logger.opt(colors=False).warning(traceback.format_exc())

    # =========================================================
    # PHASE B — physical driver biases (Figs 5-7)
    # =========================================================
    logger.info("=== PHASE B: driver biases (MLD, SST, clt) ===")

    # ---- Fig 5: MLD bias ----
    obs_mld      = None
    obs_mld_clim = None
    obs_mld_ts: xr.DataArray | None = None   # for Figs 9, 10
    common_grid  = None
    mld_bias: dict[str, xr.DataArray] = {}
    model_mld_clim: dict[str, xr.DataArray] = {}   # for Fig 9 (legacy)
    model_mld_ts:   dict[str, xr.DataArray] = {}   # native time series for Fig 10
    try:
        obs_mld = load_obs_mld()
        obs_mld_ts = obs_mld                       # keep raw monthly series
        logger.info(f"  ORAS5 MLD: {obs_mld.shape}")
        common_grid = xr.DataArray(
            np.zeros((obs_mld.sizes["lat"], obs_mld.sizes["lon"])),
            dims=("lat", "lon"),
            coords={"lat": obs_mld.lat, "lon": obs_mld.lon},
        )
        obs_mld_clim = obs_mld.mean("time", keep_attrs=True)
        for model in MODELS:
            mld_native = load_model_mld(model)
            if mld_native is None:
                continue
            try:
                mld_re = regrid_timeseries(mld_native, common_grid)
                model_mld_ts[model] = mld_re        # keep monthly on common grid
                mld_clim_on_oras = mld_re.mean("time")
                model_mld_clim[model] = mld_clim_on_oras
                mld_bias[model] = mld_clim_on_oras - obs_mld_clim
                logger.success(f"  [{model}] MLD bias computed")
            except Exception as e:
                logger.warning(f"  [{model}] MLD bias failed: {e}")
        if mld_bias:
           make_driver_bias_with_obs(bias_dict=mld_bias, obs_clim=obs_mld_clim,obs_title="ORAS5 (obs)", obs_cmap="Blues_r",obs_label=r"MLD (m)", obs_vmin=0, obs_vmax=300,bias_vmin=-50, bias_vmax=50,bias_label=r"Bias (m)", filename="fig5_mld_bias.png", stat_fmt="mean={mean:+.1f} m  RMSE={rmse:.1f} m",  ) 
           
    except Exception as e:
        logger.warning(f"MLD section failed: {e}")

    # ---- Fig 6: SST bias ----
    obs_sst_clim = None
    obs_sst_ts: xr.DataArray | None = None    # for Fig 9
    sst_bias: dict[str, xr.DataArray] = {}
    model_sst_clim: dict[str, xr.DataArray] = {}   # for Fig 9 (legacy)
    model_sst_ts:   dict[str, xr.DataArray] = {}   # native time series for Fig 9
    try:
        obs_sst      = load_obs_sst()
        obs_sst_ts   = obs_sst                     # keep raw monthly series
        obs_sst_clim = obs_sst.mean("time", keep_attrs=True)
        for model in MODELS:
            sst_native = load_model_sst(model)
            if sst_native is None:
                continue
            try:
                # Keep native monthly time series for lag-corr; regrid to
                # common ORAS5 1° grid later inside Fig 9 (avoid double work
                # here — Fig 9 will regrid all sources together).
                model_sst_ts[model] = sst_native
                sst_clim = sst_native.mean("time", keep_attrs=True)
                sst_re   = regrid_to_obs(sst_clim, obs_sst_clim)
                model_sst_clim[model] = sst_re
                sst_bias[model] = sst_re - obs_sst_clim
                logger.success(f"  [{model}] SST bias computed")
            except Exception as e:
                logger.warning(f"  [{model}] SST bias failed: {e}")
        if sst_bias:
            make_driver_bias_with_obs(
            bias_dict=sst_bias,
            obs_clim=obs_sst_clim,
            obs_title="NOAA-ERSSTv3b (obs)",
            obs_cmap="RdYlBu_r",
            obs_label=r"SST (°C)",
            obs_vmin=-2, obs_vmax=32,
            bias_vmin=-3, bias_vmax=3,
            bias_label=r"Bias (°C)",
            filename="fig6_sst_bias.png",
            stat_fmt="mean={mean:+.2f} °C  RMSE={rmse:.2f} °C",)
    except Exception as e:
        logger.warning(f"SST section failed: {e}")

    # ---- Fig 7: cloud cover bias ----
    try:
        obs_clt      = load_obs_clt()
        obs_clt_clim = obs_clt.mean("time", keep_attrs=True)
        clt_bias: dict[str, xr.DataArray] = {}
        for model in MODELS:
            clt_native = load_model_clt(model)
            if clt_native is None:
                continue
            try:
                clt_clim = clt_native.mean("time", keep_attrs=True)
                clt_re   = regrid_to_obs(clt_clim, obs_clt_clim)
                clt_bias[model] = clt_re - obs_clt_clim
                logger.success(f"  [{model}] clt bias computed")
            except Exception as e:
                logger.warning(f"  [{model}] clt bias failed: {e}")
        if clt_bias:
            make_bias_map_2x2(
                clt_bias, obs_clt_clim,
                vmin=-20, vmax=20,
                unit_label="%",
                cmap="RdBu_r",
                fig_num=7,
                title="Cloud cover bias: model − ESA-CCI AVHRR",
                filename="fig7_clt_bias.png",
                stat_fmt="mean={mean:.2f} %  RMSE={rmse:.2f} %",
            )
    except Exception as e:
        logger.warning(f"Cloud section failed: {e}")

    # =========================================================
    # PHASE C — MLD-Chl coupling diagnostics (Figs 8-9)
    # =========================================================
    logger.info("=== PHASE C: MLD-Chl coupling diagnostics ===")

    # ---- Fig 8: Hovmöller r(MLD anomaly, Chl bias) on (lat, month) ----
    if obs_mld is not None and common_grid is not None:
        hov_maps: dict[str, xr.DataArray] = {}
        for model in MODELS:
            mld_native = load_model_mld(model)
            if mld_native is None:
                continue
            try:
                logger.info(f"  [{model}] computing Hovmöller MLD-Chl correlation...")
                mld_re     = regrid_timeseries(mld_native, common_grid)
                chl_native = load_model_chl(model)
                chl_re     = regrid_timeseries(chl_native, common_grid)
                obs_chl_re = regrid_timeseries(obs_full, common_grid)
                hov = compute_hovmoller_mld_chl_correlation(
                    mld_re, chl_re, obs_chl_re)
                hov_maps[model] = hov
                logger.success(f"  [{model}] Hovmöller done")
            except Exception as e:
                logger.warning(f"  [{model}] Hovmöller failed: {e}")
        if hov_maps:
            make_fig8(hov_maps)
    else:
        logger.warning("  Fig 8 SKIPPED: no ORAS5 MLD")

    # ---- Fig 8b: basin-mean spatial pattern correlation -----------------
    # r(Chl-a, MLD) and r(Chl-a, SST) per basin per source. OBS uses
    # ESA-CCI Chl vs ORAS5 MLD (panel a) and vs NOAA-ERSSTv3b SST
    # (panel b); model bars use each model's own Chl and driver. The
    # corresponding climatologies are already in scope here as
    # obs_clim, obs_mld_clim, obs_sst_clim, model_clim, model_mld_clim,
    # and model_sst_clim. No figure-level title.
    try:
        make_fig8b_pattern_corr(
            obs_chl_clim=obs_clim,
            obs_mld_clim=obs_mld_clim,
            obs_sst_clim=obs_sst_clim,
            model_chl_clim=model_clim,
            model_mld_clim=model_mld_clim,
            model_sst_clim=model_sst_clim,
        )
    except Exception as e:
        import traceback
        logger.warning(f"  Fig 8b crashed: {e}")
        logger.opt(colors=False).warning(traceback.format_exc())

    # ---- Fig 9 (Chl vs SST) and Fig 10 (Chl vs MLD): lag correlation ----
    # Basin-mean monthly anomalies, deseasonalised + z-scored, then
    # Pearson r at lags -12…+12 months. OBS curves use ESA-CCI Chl
    # against NOAA-ERSSTv3b SST / ORAS5 MLD; model curves use each
    # model's own Chl and driver. Six subplots, one per basin.
    if common_grid is None:
        logger.warning("  Figs 9/10 SKIPPED: no common grid (ORAS5 not loaded)")
    else:
        logger.info(f"  Fig 9 inputs: obs_chl=yes  obs_sst="
                    f"{'yes' if obs_sst_ts is not None else 'NO'}  "
                    f"model_chl={len(model_chl_ts)}  "
                    f"model_sst={len(model_sst_ts)}")
        if model_chl_ts and (obs_sst_ts is not None or model_sst_ts):
            try:
                make_fig9_lag_sst(
                    obs_chl_ts=obs_full,
                    obs_sst_ts=obs_sst_ts,
                    model_chl_ts=model_chl_ts,
                    model_sst_ts=model_sst_ts,
                    common_grid=common_grid,
                )
            except Exception as e:
                import traceback
                logger.warning(f"  Fig 9 crashed: {e}")
                logger.opt(colors=False).warning(traceback.format_exc())
        else:
            logger.warning(
                f"  Fig 9 SKIPPED: chl_ts={len(model_chl_ts)}, "
                f"sst_ts={len(model_sst_ts)}, obs_sst={obs_sst_ts is not None}"
            )

        logger.info(f"  Fig 10 inputs: obs_chl=yes  obs_mld="
                    f"{'yes' if obs_mld_ts is not None else 'NO'}  "
                    f"model_chl={len(model_chl_ts)}  "
                    f"model_mld={len(model_mld_ts)}")
        if model_chl_ts and (obs_mld_ts is not None or model_mld_ts):
            try:
                make_fig10_lag_mld(
                    obs_chl_ts=obs_full,
                    obs_mld_ts=obs_mld_ts,
                    model_chl_ts=model_chl_ts,
                    model_mld_ts=model_mld_ts,
                    common_grid=common_grid,
                )
            except Exception as e:
                import traceback
                logger.warning(f"  Fig 10 crashed: {e}")
                logger.opt(colors=False).warning(traceback.format_exc())
        else:
            logger.warning(
                f"  Fig 10 SKIPPED: chl_ts={len(model_chl_ts)}, "
                f"mld_ts={len(model_mld_ts)}, obs_mld={obs_mld_ts is not None}"
            )

    # =========================================================
    # PHASE E — sea-ice memory diagnostics (Figs 11-12)
    # =========================================================
    # Tests whether the +11-month Arctic SST→Chl lag found in
    # Fig 9 is sea-ice-mediated, and whether models lose this
    # memory because their sea-ice itself loses cross-year
    # persistence. Skipped silently if siconc is not available
    # for either obs or any model.
    logger.info("=== PHASE E: sea-ice memory diagnostics ===")

    if common_grid is None:
        logger.warning("  Phase E SKIPPED: no common grid (ORAS5 not loaded)")
    else:
        # Load obs siconc once; loader returns None if file missing
        obs_siconc_ts = load_obs_siconc()

        # Load model siconc for each available model
        model_siconc_ts: dict[str, xr.DataArray] = {}
        for model in MODELS:
            sic = load_model_siconc(model)
            if sic is not None:
                model_siconc_ts[model] = sic

        logger.info(f"  Phase E inputs: obs_siconc="
                    f"{'yes' if obs_siconc_ts is not None else 'NO'}  "
                    f"model_siconc={len(model_siconc_ts)}")

        # ---- Sea-ice bias maps: model − HadISST, polar caps ----
        # Time-mean siconc bias against the observational reference,
        # expressed in percent sea-ice concentration. Shown as two
        # polar-stereographic figures: an Arctic cap (50-90 N) for the
        # northern comparison and a Southern Ocean cap (50-90 S) covering
        # the seasonal-ice zone. siconc is held internally as a fraction
        # (0-1); we multiply by 100 so the maps and the per-cap mean/RMSE
        # insets read in concentration percent.
        if obs_siconc_ts is not None and model_siconc_ts:
            try:
                obs_sic_clim = obs_siconc_ts.mean("time", keep_attrs=True) * 100.0
                siconc_bias: dict[str, xr.DataArray] = {}
                for model in MODELS:
                    if model not in model_siconc_ts:
                        continue
                    try:
                        m_clim = model_siconc_ts[model].mean(
                            "time", keep_attrs=True) * 100.0
                        m_re = regrid_to_obs(m_clim, obs_sic_clim)
                        siconc_bias[model] = m_re - obs_sic_clim
                        logger.success(f"  [{model}] siconc bias computed")
                    except Exception as e:
                        logger.warning(f"  [{model}] siconc bias failed: {e}")
                if siconc_bias:
                    make_polar_bias_map(
                        siconc_bias, obs_sic_clim, pole="north",
                        vmin=-50, vmax=50,
                        unit_label="% conc.",
                        cmap="RdBu_r",
                        title="Arctic sea-ice concentration bias: "
                              "model − HadISST",
                        filename="fig_siconc_bias_arctic.png",
                        stat_fmt="mean={mean:+.1f} %  RMSE={rmse:.1f} %",
                    )
                    make_polar_bias_map(
                        siconc_bias, obs_sic_clim, pole="south",
                        vmin=-50, vmax=50,
                        unit_label="% conc.",
                        cmap="RdBu_r",
                        title="Southern Ocean sea-ice concentration bias: "
                              "model − HadISST",
                        filename="fig_siconc_bias_southern.png",
                        stat_fmt="mean={mean:+.1f} %  RMSE={rmse:.1f} %",
                    )
            except Exception as e:
                import traceback
                logger.warning(f"  Sea-ice bias maps crashed: {e}")
                logger.opt(colors=False).warning(traceback.format_exc())
        else:
            logger.warning(
                f"  Sea-ice bias maps SKIPPED: obs_siconc="
                f"{obs_siconc_ts is not None}, model_siconc={len(model_siconc_ts)}"
            )

        # ---- Polar light coupling: rsds vs Chl-a AND rsds vs siconc ----
        # Lag correlation of light (rsds) anomalies against Chl-a and
        # sea-ice anomalies in the polar basins. Reuses the common ORAS5
        # grid and the Section 2.6 deseasonalise/detrend/z-score pipeline.
        if (model_rsds_ts and model_siconc_ts and model_chl_ts):
            try:
                make_fig_light_polar_coupling(
                    obs_chl_ts=obs_full,
                    obs_rsds_ts=obs_rsds_ts,
                    obs_siconc_ts=obs_siconc_ts,
                    model_chl_ts=model_chl_ts,
                    model_rsds_ts=model_rsds_ts,
                    model_siconc_ts=model_siconc_ts,
                    common_grid=common_grid,
                )
            except Exception as e:
                import traceback
                logger.warning(f"  Fig light-polar crashed: {e}")
                logger.opt(colors=False).warning(traceback.format_exc())
        else:
            logger.warning(
                f"  Fig light-polar SKIPPED: rsds_ts={len(model_rsds_ts)}, "
                f"chl_ts={len(model_chl_ts)}, siconc_ts={len(model_siconc_ts)}"
            )

        # ---- Fig 11: lag corr Chl vs siconc (polar basins only) ----
        if model_chl_ts and (obs_siconc_ts is not None or model_siconc_ts):
            try:
                make_fig11_lag_siconc(
                    obs_chl_ts=obs_full,
                    obs_siconc_ts=obs_siconc_ts,
                    model_chl_ts=model_chl_ts,
                    model_siconc_ts=model_siconc_ts,
                    common_grid=common_grid,
                )
            except Exception as e:
                import traceback
                logger.warning(f"  Fig 11 crashed: {e}")
                logger.opt(colors=False).warning(traceback.format_exc())
        else:
            logger.warning(
                f"  Fig 11 SKIPPED: chl_ts={len(model_chl_ts)}, "
                f"siconc_ts={len(model_siconc_ts)}, "
                f"obs_siconc={obs_siconc_ts is not None}"
            )

        # ---- Fig 11b: longitudinal sector decomposition (SO) ----
        # Resolves the Fig 11 Southern-SIZ panel into 5 circumpolar
        # NSIDC-style sectors (Weddell, Indian, West Pacific, Ross,
        # Bellingshausen-Amundsen) plus a Southern-ACC negative-control
        # panel. Distinguishes uniform sector-level coupling failure
        # from compensating sign errors between sectors.
        if model_chl_ts and (obs_siconc_ts is not None or model_siconc_ts):
            try:
                make_fig11b_lag_siconc_sectors(
                    obs_chl_ts=obs_full,
                    obs_siconc_ts=obs_siconc_ts,
                    model_chl_ts=model_chl_ts,
                    model_siconc_ts=model_siconc_ts,
                    common_grid=common_grid,
                )
            except Exception as e:
                import traceback
                logger.warning(f"  Fig 11b crashed: {e}")
                logger.opt(colors=False).warning(traceback.format_exc())
        else:
            logger.warning(
                f"  Fig 11b SKIPPED: chl_ts={len(model_chl_ts)}, "
                f"siconc_ts={len(model_siconc_ts)}, "
                f"obs_siconc={obs_siconc_ts is not None}"
            )

        # ---- Fig 11c: meridional (latitude-band) decomposition (SO) ----
        # Cuts the Southern Ocean meridionally into three contiguous
        # latitude bands (Subtropics 45-30 S, ACC 60-45 S, High-lat
        # 80-60 S) to test whether the Chl-a/sea-ice coupling is
        # confined to the cryospherically active band as physical
        # reasoning predicts. The Subtropics and ACC bands act as
        # negative controls.
        if model_chl_ts and (obs_siconc_ts is not None or model_siconc_ts):
            try:
                make_fig11c_lag_siconc_meridional(
                    obs_chl_ts=obs_full,
                    obs_siconc_ts=obs_siconc_ts,
                    model_chl_ts=model_chl_ts,
                    model_siconc_ts=model_siconc_ts,
                    common_grid=common_grid,
                )
            except Exception as e:
                import traceback
                logger.warning(f"  Fig 11c crashed: {e}")
                logger.opt(colors=False).warning(traceback.format_exc())
        else:
            logger.warning(
                f"  Fig 11c SKIPPED: chl_ts={len(model_chl_ts)}, "
                f"siconc_ts={len(model_siconc_ts)}, "
                f"obs_siconc={obs_siconc_ts is not None}"
            )

        # ---- Fig 12: 12-month autocorrelation of siconc ----
        if obs_siconc_ts is not None or model_siconc_ts:
            try:
                make_fig12_siconc_memory(
                    obs_siconc_ts=obs_siconc_ts,
                    model_siconc_ts=model_siconc_ts,
                    common_grid=common_grid,
                    lag_months=12,
                )
            except Exception as e:
                import traceback
                logger.warning(f"  Fig 12 crashed: {e}")
                logger.opt(colors=False).warning(traceback.format_exc())
        else:
            logger.warning(
                f"  Fig 12 SKIPPED: siconc_ts={len(model_siconc_ts)}, "
                f"obs_siconc={obs_siconc_ts is not None}"
            )

        # ---- Fig S1: same as Fig 12 but restricted to seasonal-ice zone ----
        # Robustness check: is the Fig 12 basin-mean signal driven by
        # cells that actually vary year-to-year, or by static (always
        # ice-free or always ice-covered) cells inflating r?
        if obs_siconc_ts is not None or model_siconc_ts:
            try:
                make_figS1_siconc_memory_seasonal_zone(
                    obs_siconc_ts=obs_siconc_ts,
                    model_siconc_ts=model_siconc_ts,
                    common_grid=common_grid,
                    lag_months=12,
                )
            except Exception as e:
                import traceback
                logger.warning(f"  Fig S1 crashed: {e}")
                logger.opt(colors=False).warning(traceback.format_exc())
        else:
            logger.warning(
                f"  Fig S1 SKIPPED: siconc_ts={len(model_siconc_ts)}, "
                f"obs_siconc={obs_siconc_ts is not None}"
            )

        # ---- Fig S2: piControl internal-variability envelope ----
        # Recomputes the Fig 12 statistic on overlapping 17-yr
        # segments of each model's piControl run, to show how much of
        # the historical sea-ice-persistence spread is internal
        # variability. Reads piControl siconc straight from the DKRZ
        # pool; skips silently per model if piControl is unavailable.
        try:
            make_figS2_picontrol_envelope(
                obs_siconc_ts=obs_siconc_ts,
                common_grid=common_grid,
                lag_months=12,
            )
        except Exception as e:
            import traceback
            logger.warning(f"  Fig S2 crashed: {e}")
            logger.opt(colors=False).warning(traceback.format_exc())

    # =========================================================
    # Done
    # =========================================================
    logger.info(f"All figures in {OUT_DIR}")
    logger.info("Summary of figures produced:")
    for f in sorted(OUT_DIR.glob("fig*.png")):
        logger.info(f"  {f.name}")
