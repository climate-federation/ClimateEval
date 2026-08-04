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
        subplot_kw={"projection": ccrs.Robinson(central_longitude=0)},
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

    _lon2d, _lat2d = np.meshgrid(
        np.array(obs_clim.lon, dtype=float),
        np.array(obs_clim.lat, dtype=float),
    )
    _obs_vals = np.array(obs_clim.values, dtype=float)
    im_obs = ax0.pcolormesh(
        _lon2d, _lat2d, _obs_vals,
        vmin=obs_vmin, vmax=obs_vmax, cmap=obs_cmap,
        transform=ccrs.PlateCarree(),
    )
    # land mask: chl is NaN over land — no explicit feature needed
    ax0.grid(linestyle=":", linewidth=0.3, alpha=0.5)
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

        _blon2d, _blat2d = np.meshgrid(
            np.array(obs_clim.lon, dtype=float),
            np.array(obs_clim.lat, dtype=float),
        )
        _bias_vals = np.array(bias.values, dtype=float)
        im_bias = ax.pcolormesh(
            _blon2d, _blat2d, _bias_vals,
            norm=norm_bias, cmap="RdBu_r",
            transform=ccrs.PlateCarree(),
        )
        # land mask: chl is NaN over land — no explicit feature needed

        ax.grid(linestyle=":", linewidth=0.3, alpha=0.4)

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
        subplot_kw={"projection": ccrs.Robinson(central_longitude=0)},
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
    )  # land NaN in chl — no land mask needed
    ax0.grid(linestyle=":", linewidth=0.3, alpha=0.5)
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

        _blon2d, _blat2d = np.meshgrid(
            np.array(obs_clim.lon, dtype=float),
            np.array(obs_clim.lat, dtype=float),
        )
        _bias_vals = np.array(bias.values, dtype=float)
        im_bias = ax.pcolormesh(
            _blon2d, _blat2d, _bias_vals,
            norm=norm_bias, cmap="RdBu_r",
            transform=ccrs.PlateCarree(),
        )  # land NaN in chl — no land mask needed
        ax.grid(linestyle=":", linewidth=0.3, alpha=0.4)

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
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        try:

            ax.coastlines(linewidth=0.4)

        except Exception:

            pass
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


def make_polar_bias_map_combined(
    bias_dict: dict[str, xr.DataArray],
    obs_grid: xr.DataArray,
    vmin: float = -50.0,
    vmax: float = 50.0,
    unit_label: str = "% conc.",
    cmap: str = "RdBu_r",
    stat_fmt: str = "mean={mean:+.1f}%  RMSE={rmse:.1f}%",
    obs_edge_threshold: float = 15.0,
    filename: str = "fig5c_siconc_bias_combined.png",
    dpi: int = 180,
) -> None:
    """Two-row x N_models polar bias map (Arctic top, Southern Ocean bottom).

    Replaces the two separate make_polar_bias_map calls (one per pole) with
    a single combined figure. Layout: 2 rows x len(MODELS) columns. Each
    panel is a circular polar-stereographic cap. The observed 15% ice-edge
    contour (HadISST) is overlaid in black on every panel. A single shared
    horizontal colorbar sits below the bottom row.

    Parameters
    ----------
    bias_dict:
        {model_name: (lat, lon) DataArray of siconc bias in percent}.
        Models are plotted left-to-right in MODELS order.
    obs_grid:
        HadISST time-mean siconc climatology in percent, used for the
        ice-edge contour and cap-mean statistics.
    obs_edge_threshold:
        Percent concentration for the observed ice-edge contour (NSIDC
        standard = 15 %).
    """
    import matplotlib.path as mpath
    from cartopy.util import add_cyclic_point

    logger.info("Building combined polar bias map (Arctic + Southern Ocean)...")

    # Preserve MODELS order so columns match other multi-model figures.
    models = [m for m in MODELS if m in bias_dict]
    n = len(models)
    if n == 0:
        logger.warning(f"  no data; skipping {filename}")
        return

    lat = obs_grid.lat.values
    lon = obs_grid.lon.values

    # Row definitions: row 0 = Arctic, row 1 = Southern Ocean.
    poles = [
        {
            "label":   "Arctic",
            "proj":    ccrs.NorthPolarStereo(),
            "extent":  [-180, 180, 50, 90],
            "cap_sel": lat >= 50,
        },
        {
            "label":   "Southern Ocean",
            "proj":    ccrs.SouthPolarStereo(),
            "extent":  [-180, 180, -90, -50],
            "cap_sel": lat <= -50,
        },
    ]

    # Figure geometry: each column ~3.5 in wide, two rows + colorbar strip.
    fig_w = max(3.5 * n, 10)
    fig_h = 7.6
    fig   = plt.figure(figsize=(fig_w, fig_h))

    # Circular clip path shared across all panels.
    theta  = np.linspace(0, 2 * np.pi, 200)
    circle = mpath.Path(
        np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + [0.5, 0.5]
    )

    norm    = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    last_im = None

    # Row y-positions in figure fraction (bottom edge of each row).
    # A thin strip at the very bottom is reserved for the colorbar.
    row_bottom = [0.52, 0.04]
    row_h      = 0.44
    col_w      = 1.0 / n

    for row_idx, pole in enumerate(poles):
        row_axes = []
        for col_idx, model in enumerate(models):
            bias = bias_dict[model]

            rect = [col_idx * col_w, row_bottom[row_idx], col_w, row_h]
            ax   = fig.add_axes(rect, projection=pole["proj"])
            ax.set_extent(pole["extent"], ccrs.PlateCarree())
            ax.set_boundary(circle, transform=ax.transAxes)

            # ---- bias field ----
            data_c, lon_c = add_cyclic_point(bias.values, coord=lon)
            im = ax.pcolormesh(
                lon_c, lat, data_c,
                norm=norm, cmap=cmap,
                transform=ccrs.PlateCarree(),
                rasterized=True,
            )
            last_im = im

            # ---- observed ice-edge contour ----
            obs_c, lon_cc = add_cyclic_point(obs_grid.values, coord=lon)
            try:
                ax.contour(
                    lon_cc, lat, obs_c,
                    levels=[obs_edge_threshold],
                    colors="black", linewidths=1.0,
                    transform=ccrs.PlateCarree(),
                )
            except Exception:
                pass

            # ---- geography ----
            ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
            try:
                ax.coastlines(linewidth=0.4, zorder=3)
            except Exception:
                pass
            ax.gridlines(linestyle=":", linewidth=0.3, alpha=0.4)

            # ---- model name title (top row only) ----
            if row_idx == 0:
                ax.set_title(model, fontsize=14, pad=3)

            # ---- cap mean/RMSE inset ----
            cap_vals = bias.values[pole["cap_sel"], :]
            v = cap_vals[np.isfinite(cap_vals)]
            if v.size:
                mean_b = float(np.mean(v))
                rmse_b = float(np.sqrt(np.mean(v ** 2)))
                ax.text(
                    0.03, 0.03,
                    stat_fmt.format(mean=mean_b, rmse=rmse_b),
                    transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor="#555555",
                              alpha=0.90),
                )

            # ---- panel letter ----
            letter = chr(ord("a") + row_idx * n + col_idx)
            ax.text(
                0.01, 0.97, f"({letter})",
                transform=ax.transAxes, fontsize=11,
                va="top", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15",
                          facecolor="white", alpha=0.7, edgecolor="none"),
            )

            row_axes.append(ax)

        # ---- row label (rotated, left of leftmost panel) ----
        row_axes[0].text(
            -0.06, 0.5, pole["label"],
            transform=row_axes[0].transAxes,
            fontsize=13, fontweight="bold",
            ha="right", va="center", rotation=90,
        )

    # ---- shared horizontal colorbar ----
    # bottom=-0.04 pushes it below the panels; bbox_inches="tight" captures it.
    if last_im is not None:
        cbar_ax = fig.add_axes([0.15, -0.04, 0.70, 0.022])
        cbar    = fig.colorbar(last_im, cax=cbar_ax,
                               orientation="horizontal", extend="both")
        cbar.set_label(f"Sea-ice concentration bias [{unit_label}]",
                       fontsize=16)
        cbar.ax.tick_params(labelsize=18)

    fig.savefig(OUT_DIR / filename, dpi=dpi, bbox_inches="tight")
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



# ============================================================
# CLIMATE INDEX COMPUTATION FROM OBS SST
# Insert this block immediately after _lag_correlation_with_bootstrap()
# and before make_fig9_lag_sst()
# ============================================================

# Geographic boxes for the three indices — all on the common 1° grid
# with longitudes in [0, 360].
_AMO_BOX      = (0.0,  70.0, 280.0, 360.0)   # N. Atlantic 0-70N, 280-360E
_GLOBAL_BOX   = (-90.0, 90.0, 0.0, 360.0)    # global mean (removed from AMO)
_NINO34_BOX   = (-5.0,  5.0, 190.0, 240.0)   # Niño 3.4: 5S-5N, 190-240E
_PDO_BOX      = (20.0, 70.0, 110.0, 260.0)   # N. Pacific 20-70N, 110-260E


def _box_mean(
    sst_re: xr.DataArray,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> xr.DataArray:
    """cos(lat)-weighted area mean of a (time, lat, lon) field in a box.

    lon_min and lon_max are in [0, 360]. Handles wrap-around correctly
    when lon_min > lon_max, though none of the index boxes require it.
    Returns a 1-D (time,) DataArray.
    """
    lat = sst_re.lat
    lon = sst_re.lon
    lat2d, lon2d = xr.broadcast(lat, lon)

    in_lat = (lat2d >= lat_min) & (lat2d <= lat_max)
    if lon_min <= lon_max:
        in_lon = (lon2d >= lon_min) & (lon2d <= lon_max)
    else:
        in_lon = (lon2d >= lon_min) | (lon2d <= lon_max)
    mask = in_lat & in_lon

    w = np.cos(np.deg2rad(sst_re.lat)) * xr.ones_like(sst_re.lon)
    w = w.where(mask)
    return sst_re.where(mask).weighted(w.fillna(0)).mean(
        dim=("lat", "lon"), skipna=True
    )


def _compute_climate_indices(
    obs_sst_re: xr.DataArray,
) -> dict[str, xr.DataArray]:
    """Compute AMO, Niño 3.4, and PDO indices from the regridded obs SST.

    All three are computed from the same NOAA-ERSSTv3b field already on
    the common 1° grid, so no external download or file is needed.

    AMO
    ---
    Area-weighted mean SST anomaly over the North Atlantic (0°–70°N,
    280°–360°E) with the global-mean SST anomaly subtracted to remove
    the externally forced warming trend (Trenberth & Shea, 2006).
    Both the basin mean and the global mean are computed from the raw
    monthly SST; their difference is the AMO index. The index is then
    deseasonalised, linearly detrended, and standardised with the
    same pipeline used for all other anomaly series (Section 2.6).

    Niño 3.4
    --------
    Area-weighted mean SST anomaly over 5°S–5°N, 190°–240°E. The
    anomaly is computed by subtracting the monthly climatology of the
    same region, then detrending and standardising.

    PDO
    ---
    Leading EOF of monthly North Pacific (20°–70°N, 110°–260°E) SST
    anomaly field after removing the global-mean SST anomaly. The
    spatial EOF is computed from the full 1997–2014 period; the index
    is the corresponding principal component (PC1), sign-corrected so
    that positive PDO corresponds to a warm eastern North Pacific
    (standard convention: PC1 is multiplied by -1 if its area-weighted
    mean in the eastern North Pacific 20–50°N, 200–260°E is negative).

    Returns
    -------
    dict with keys "AMO", "NINO34", "PDO", each a 1-D (time,)
    DataArray that is deseasonalised, linearly detrended, and
    standardised to unit variance — ready for _regress_out().
    """
    from numpy.linalg import svd

    logger.info("  Computing climate indices from obs SST...")

    # ----------------------------------------------------------------
    # Step 0: raw monthly SST on common grid, already trimmed to period
    # ----------------------------------------------------------------
    sst = obs_sst_re   # (time, lat, lon)

    # ----------------------------------------------------------------
    # Global mean SST anomaly (used by both AMO and PDO)
    # ----------------------------------------------------------------
    gm_ts = _box_mean(sst, *_GLOBAL_BOX)            # (time,) raw SST
    gm_clim = gm_ts.groupby("time.month").mean("time")
    gm_anom = (gm_ts.groupby("time.month") - gm_clim)  # deseasonalised

    # ----------------------------------------------------------------
    # AMO: N. Atlantic mean SST anomaly minus global-mean SST anomaly
    # ----------------------------------------------------------------
    atl_ts   = _box_mean(sst, *_AMO_BOX)
    atl_clim = atl_ts.groupby("time.month").mean("time")
    atl_anom = atl_ts.groupby("time.month") - atl_clim
    amo_raw  = atl_anom - gm_anom                    # remove global trend
    amo_idx  = _deseasonalise_zscore(amo_raw)        # detrend + zscore
    logger.success(
        f"  AMO index computed: {int(amo_idx.size)} months, "
        f"std={float(amo_idx.std()):.3f}"
    )

    # ----------------------------------------------------------------
    # Niño 3.4: equatorial Pacific box mean SST anomaly
    # ----------------------------------------------------------------
    n34_ts   = _box_mean(sst, *_NINO34_BOX)
    n34_idx  = _deseasonalise_zscore(n34_ts)
    logger.success(
        f"  Niño 3.4 index computed: {int(n34_idx.size)} months, "
        f"std={float(n34_idx.std()):.3f}"
    )

    # ----------------------------------------------------------------
    # PDO: leading EOF of N. Pacific SST anomaly (global mean removed)
    # ----------------------------------------------------------------
    # 1. Extract N. Pacific box
    lat  = sst.lat.values
    lon  = sst.lon.values
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")

    in_lat = (lat2d >= _PDO_BOX[0]) & (lat2d <= _PDO_BOX[1])
    in_lon = (lon2d >= _PDO_BOX[2]) & (lon2d <= _PDO_BOX[3])
    box_mask = in_lat & in_lon                        # (nlat, nlon) bool

    # 2. SST anomaly in the box: remove global mean first, then
    #    deseasonalise within the box
    sst_no_gm = sst - gm_anom                        # remove global trend
    sst_clim  = sst_no_gm.groupby("time.month").mean("time")
    sst_anom  = sst_no_gm.groupby("time.month") - sst_clim  # (time,lat,lon)

    # 3. Build data matrix X: (n_time, n_valid_cells), cos(lat) weighted
    vals = sst_anom.values                            # (T, nlat, nlon)
    T    = vals.shape[0]
    w2d  = np.cos(np.deg2rad(lat2d))                 # (nlat, nlon)

    # Flatten space; keep only cells inside the box with no NaNs
    X_flat = vals.reshape(T, -1)                     # (T, nlat*nlon)
    w_flat = w2d.ravel()
    m_flat = box_mask.ravel()

    # Valid: inside box AND no NaN in any time step
    valid  = m_flat & np.all(np.isfinite(X_flat), axis=0)
    if valid.sum() < 10:
        logger.warning("  PDO: too few valid cells; returning NaN index")
        pdo_idx = xr.full_like(amo_idx, np.nan)
        pdo_idx.name = "PDO"
        return {"AMO": amo_idx, "NINO34": n34_idx, "PDO": pdo_idx}

    X_sub = X_flat[:, valid]                         # (T, n_valid)
    w_sub = w_flat[valid]                            # (n_valid,)

    # 4. Weight the data matrix by sqrt(cos(lat)) for area-correct EOF
    W     = np.sqrt(w_sub)[np.newaxis, :]            # (1, n_valid)
    Xw    = X_sub * W                                # (T, n_valid)

    # 5. Thin SVD: U (T, k), S (k,), Vt (k, n_valid)
    #    PC1 = U[:, 0] * S[0]  (time series of leading mode)
    U, S, Vt = svd(Xw, full_matrices=False)
    pc1 = U[:, 0] * S[0]                            # (T,)

    # 6. Sign convention: positive PDO = warm eastern N. Pacific.
    #    Check sign against eastern North Pacific (20-50N, 200-260E).
    ep_lat = (lat2d >= 20) & (lat2d <= 50)
    ep_lon = (lon2d >= 200) & (lon2d <= 260)
    ep_valid = ep_lat & ep_lon & box_mask
    ep_flat  = ep_valid.ravel()[valid]
    if ep_flat.any():
        # Mean EOF loading in eastern N. Pacific (unweighted for sign)
        eof1_ep = Vt[0, ep_flat].mean()
        if eof1_ep > 0:
            # Positive EOF loading in eastern N. Pacific → standard convention
            # requires the PC to be negative when east is warm after the
            # sign flip (loading positive = PC positive = warm east = +PDO).
            # Keep as-is.
            pass
        else:
            # Flip so that positive PC1 = warm eastern North Pacific
            pc1 = -pc1

    pdo_raw = xr.DataArray(
        pc1,
        coords={"time": sst_anom.time},
        dims=["time"],
        name="PDO",
    )
    pdo_idx = _deseasonalise_zscore(pdo_raw)
    logger.success(
        f"  PDO index computed: {int(pdo_idx.size)} months, "
        f"std={float(pdo_idx.std()):.3f}"
    )

    return {"AMO": amo_idx, "NINO34": n34_idx, "PDO": pdo_idx}


def _regress_out(
    target: xr.DataArray,
    predictor: xr.DataArray,
) -> xr.DataArray:
    """Remove the linear influence of `predictor` from `target` via OLS.

    Both series must be pre-processed (deseasonalised, detrended,
    standardised). Alignment is done by matching year*100+month integer
    keys (calendar-agnostic) rather than coordinate values, so that
    cftime model axes and pandas-Timestamp OBS axes are handled
    correctly. Returns the OLS residual target - beta*predictor. The
    residual is NOT re-standardised so its variance is directly
    comparable to the raw series.
    """
    ym_t = _ym_key(target)
    ym_p = _ym_key(predictor)
    common = np.intersect1d(ym_t, ym_p)
    if common.size < 12:
        logger.warning(
            f"  _regress_out: only {common.size} common (year,month) "
            f"steps; returning target unchanged"
        )
        return target

    t_idx = np.argsort(ym_t)[np.searchsorted(np.sort(ym_t), common)]
    p_idx = np.argsort(ym_p)[np.searchsorted(np.sort(ym_p), common)]

    t_arr = np.asarray(target.values,    dtype=float)
    p_arr = np.asarray(predictor.values, dtype=float)
    t_sub = t_arr[t_idx]
    p_sub = p_arr[p_idx]

    ok = np.isfinite(t_sub) & np.isfinite(p_sub)
    if ok.sum() < 12:
        logger.warning(
            "  _regress_out: <12 finite pairs; returning target unchanged"
        )
        return target

    beta = float(np.dot(p_sub[ok], t_sub[ok]) / np.dot(p_sub[ok], p_sub[ok]))

    # Apply to the full target length
    p_aligned = np.full(t_arr.size, np.nan, dtype=float)
    p_aligned[t_idx] = p_sub
    resid = t_arr - beta * p_aligned
    resid[~np.isfinite(t_arr)] = np.nan
    no_pred = ~np.isfinite(p_aligned)
    resid[no_pred] = t_arr[no_pred]
    return target.copy(data=resid)


# ============================================================
# Supplemental figure: time series of the three climate indices
# ============================================================

def make_figS_climate_indices(
    indices: dict[str, xr.DataArray],
) -> None:
    """Fig S_indices: time series of AMO, Niño 3.4, and PDO indices.

    Three stacked panels, one per index.  Each panel shows the
    standardised monthly index (thin grey line) and a 12-month
    running mean (coloured bold line).  A horizontal zero line and
    ±1σ dashed lines are drawn for reference.  Positive phases are
    shaded light red, negative phases light blue.

    Saved to OUT_DIR/figS_climate_indices.png.
    """
    import matplotlib.dates as mdates

    logger.info("=== Fig S_indices: climate index time series ===")

    if not indices:
        logger.warning("  No indices available; skipping figS_climate_indices")
        return

    index_meta = {
        "AMO":    {"color": "#CC3333",
                   "label": "AMO index\n(N. Atl. SST anom. − global mean SST anom.)",
                   "panel": "(a)"},
        "NINO34": {"color": "#3366CC",
                   "label": "Niño 3.4 index\n(equatorial Pacific SST anom.)",
                   "panel": "(b)"},
        "PDO":    {"color": "#228833",
                   "label": "PDO index\n(N. Pacific leading EOF)",
                   "panel": "(c)"},
    }

    names = [n for n in ("AMO", "NINO34", "PDO") if n in indices]
    n_panels = len(names)
    if n_panels == 0:
        logger.warning("  No computed indices; skipping")
        return

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(12, 3.2 * n_panels),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for ax, name in zip(axes_flat, names):
        idx  = indices[name]
        meta = index_meta[name]
        time = idx.time.values

        # Convert cftime/datetime64 to matplotlib float dates
        try:
            import pandas as pd
            t_plot = pd.DatetimeIndex(
                [pd.Timestamp(str(t)) for t in time]
            )
        except Exception:
            t_plot = np.arange(len(time))

        vals = np.asarray(idx.values, dtype=float)

        # 12-month running mean
        from pandas import Series
        smooth = Series(vals).rolling(12, center=True,
                                      min_periods=6).mean().values

        # Shading: positive/negative phases
        ax.fill_between(
            t_plot, vals, 0,
            where=(vals >= 0),
            color="#FFCCCC", alpha=0.6, linewidth=0,
        )
        ax.fill_between(
            t_plot, vals, 0,
            where=(vals < 0),
            color="#CCE0FF", alpha=0.6, linewidth=0,
        )

        # Raw monthly index
        ax.plot(t_plot, vals, color="grey", lw=0.8,
                alpha=0.7, label="Monthly")

        # 12-month running mean
        ax.plot(t_plot, smooth, color=meta["color"], lw=2.0,
                label="12-month running mean")

        # Reference lines
        ax.axhline(0,   color="k",             lw=0.8)
        ax.axhline(+1,  color="k", lw=0.6, ls="--", alpha=0.4)
        ax.axhline(-1,  color="k", lw=0.6, ls="--", alpha=0.4)

        ax.set_ylabel("Standardised index", fontsize=9)
        ax.set_ylim(-3.5, 3.5)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")

        # Panel label + title
        ax.text(0.01, 0.97, meta["panel"],
                transform=ax.transAxes, fontsize=11,
                fontweight="bold", va="top")
        ax.set_title(meta["label"], fontsize=9, loc="left", pad=2)

    # x-axis formatting on bottom panel
    try:
        axes_flat[-1].xaxis.set_major_locator(mdates.YearLocator(2))
        axes_flat[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(axes_flat[-1].xaxis.get_majorticklabels(), rotation=45)
    except Exception:
        pass
    axes_flat[-1].set_xlabel("Year", fontsize=10)

    fig.suptitle(
        "Supplemental — Climate indices computed from NOAA-ERSSTv3b "
        "(1998–2014, deseasonalised, detrended, standardised)\n"
        "Used for partial-correlation analysis of the North Atlantic "
        "SST–Chl-$a$ lag structure (Section 2 / Fig S_amo)",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    out_path = OUT_DIR / "figS_climate_indices.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {out_path.name}")


# ============================================================
# Combined supplemental: climate index time series + lag correlation
# sensitivity, one row per index, no raw panel
# ============================================================

# ============================================================
# Analysis 1: Bloom onset vs MLD shoaling scatter (per basin)
# ============================================================

def _bloom_mld_timing(
    chl_ts: xr.DataArray,
    mld_ts: xr.DataArray,
    mask:   xr.DataArray,
    common_grid: xr.DataArray,
    period: slice = LAG_PERIOD,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each year compute bloom onset month and MLD shoaling month.

    Bloom onset: month of Chl-a annual maximum in the basin mean.
    MLD shoaling completion: month of MLD annual minimum (shallowest).

    Both are computed from the RAW (not deseasonalised) monthly
    climatology-plus-anomaly series so that intra-annual timing is
    preserved. A linear trend is removed from the monthly annual-max
    and annual-min series before returning so that any long-term
    shift in timing does not inflate the interannual scatter.

    Returns
    -------
    years      : (n_years,) int
    bloom_month: (n_years,) float — month of Chl-a peak (1-12)
    shoal_month: (n_years,) float — month of MLD minimum (1-12)
    """
    chl_re = regrid_timeseries(chl_ts.sel(time=period), common_grid)
    mld_re = regrid_timeseries(mld_ts.sel(time=period), common_grid)

    chl_b = _basin_mean_timeseries(chl_re, mask)
    mld_b = _basin_mean_timeseries(mld_re, mask)

    # Align on year-month keys
    ym_c = _ym_key(chl_b)
    ym_m = _ym_key(mld_b)
    common = np.intersect1d(ym_c, ym_m)
    c_idx = np.argsort(ym_c)[np.searchsorted(np.sort(ym_c), common)]
    m_idx = np.argsort(ym_m)[np.searchsorted(np.sort(ym_m), common)]

    chl_vals = np.asarray(chl_b.values, dtype=float)[c_idx]
    mld_vals = np.asarray(mld_b.values, dtype=float)[m_idx]
    ym_common = ym_c[c_idx]                  # sorted year*100+month keys

    years_all  = ym_common // 100
    months_all = ym_common % 100
    unique_years = np.unique(years_all)

    bloom_months = []
    shoal_months = []
    valid_years  = []

    for yr in unique_years:
        sel = years_all == yr
        if sel.sum() < 10:            # need most of the year
            continue
        mo  = months_all[sel]
        chl = chl_vals[sel]
        mld = mld_vals[sel]

        if not (np.any(np.isfinite(chl)) and np.any(np.isfinite(mld))):
            continue

        # Month of Chl-a maximum
        ok_c = np.isfinite(chl)
        bloom_mo = float(mo[ok_c][np.argmax(chl[ok_c])])

        # Month of MLD minimum (shallowest = stratified)
        ok_m = np.isfinite(mld)
        shoal_mo = float(mo[ok_m][np.argmin(mld[ok_m])])

        bloom_months.append(bloom_mo)
        shoal_months.append(shoal_mo)
        valid_years.append(yr)

    if len(valid_years) < 3:
        return (np.array([]), np.array([]), np.array([]))

    bm = np.array(bloom_months,  dtype=float)
    sm = np.array(shoal_months,  dtype=float)
    yr = np.array(valid_years,   dtype=int)

    # Detrend both timing series so long-term drift does not
    # inflate the year-to-year scatter
    for arr in (bm, sm):
        ok = np.isfinite(arr)
        if ok.sum() >= 3:
            t = np.arange(len(arr), dtype=float)
            sl, ic = np.polyfit(t[ok], arr[ok], 1)
            arr -= sl * t + ic

    return yr, bm, sm


def make_figS_bloom_mld_timing(
    obs_chl_ts:   xr.DataArray,
    obs_mld_ts:   "xr.DataArray | None",
    model_chl_ts: dict[str, xr.DataArray],
    model_mld_ts: dict[str, xr.DataArray],
    common_grid:  xr.DataArray,
    period:       slice = LAG_PERIOD,
) -> None:
    """Fig S_bloom_timing: scatter of bloom onset vs MLD shoaling month.

    Six panels (one per basin). Each panel shows detrended bloom onset
    month (y-axis) vs detrended MLD shoaling month (x-axis), one point
    per year. OBS in black; each model in its PALETTE colour. A linear
    regression line is drawn for OBS and each model. The slope of the
    OBS regression tests whether early-shoaling years produce early
    blooms (slope ~ 1, critical-depth control) or whether the bloom
    is decoupled from MLD shoaling (slope ~ 0).

    Output: figS_bloom_mld_timing.png
    """
    from scipy import stats as _stats

    logger.info("=== Fig S_bloom_mld_timing: bloom onset vs MLD shoaling ===")

    if obs_mld_ts is None and not model_mld_ts:
        logger.warning("  No MLD data; skipping figS_bloom_mld_timing")
        return

    masks       = _build_masks(common_grid)
    basin_order = [b for b in MEANSTATE_BASINS if b in masks]
    models_list = [m for m in MODELS if m in model_chl_ts]
    ncols = 3
    nrows = int(np.ceil(len(basin_order) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5.5 * ncols, 4.8 * nrows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for b_idx, basin in enumerate(basin_order):
        ax   = axes_flat[b_idx]
        mask = masks[basin]

        # ---- OBS ----
        obs_yr = obs_bm = obs_sm = None
        if obs_mld_ts is not None:
            try:
                obs_yr, obs_bm, obs_sm = _bloom_mld_timing(
                    obs_chl_ts, obs_mld_ts, mask, common_grid, period
                )
                if obs_yr.size > 2:
                    ax.scatter(obs_sm, obs_bm, color="black", s=55,
                               zorder=10, label="OBS", alpha=0.85)
                    # OBS regression line
                    sl, ic, r, p, _ = _stats.linregress(obs_sm, obs_bm)
                    x_fit = np.linspace(obs_sm.min(), obs_sm.max(), 50)
                    ax.plot(x_fit, sl * x_fit + ic, color="black",
                            lw=2.0, zorder=9)
                    ax.text(
                        0.03, 0.97,
                        f"OBS: slope={sl:.2f}, r={r:.2f}, p={p:.3f}",
                        transform=ax.transAxes, fontsize=7.5,
                        va="top", family="monospace",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white", edgecolor="grey",
                                  alpha=0.85),
                    )
            except Exception as e:
                logger.warning(f"  OBS {basin} timing failed: {e}")

        # ---- Models ----
        for m_idx, model in enumerate(models_list):
            if model not in model_mld_ts:
                continue
            try:
                yr, bm, sm = _bloom_mld_timing(
                    model_chl_ts[model], model_mld_ts[model],
                    mask, common_grid, period
                )
                if yr.size < 3:
                    continue
                color = PALETTE[m_idx % len(PALETTE)]
                ax.scatter(sm, bm, color=color, s=30, alpha=0.55,
                           zorder=5, label=model)
                sl, ic, r, p, _ = _stats.linregress(sm, bm)
                x_fit = np.linspace(sm.min(), sm.max(), 50)
                ax.plot(x_fit, sl * x_fit + ic, color=color,
                        lw=1.2, alpha=0.8, zorder=4)
            except Exception as e:
                logger.warning(f"  [{model}] {basin} timing failed: {e}")

        # Diagonal reference: slope = 1 (perfect bloom-MLD tracking)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        mn   = min(xlim[0], ylim[0])
        mx   = max(xlim[1], ylim[1])
        ax.plot([mn, mx], [mn, mx], "k--", lw=0.8, alpha=0.4,
                label="slope=1 (perfect tracking)")

        ax.set_xlabel("MLD shoaling month (detrended)", fontsize=9)
        ax.set_ylabel("Bloom onset month (detrended)", fontsize=9)
        ax.set_title(basin, fontsize=10, fontweight="bold")
        ax.grid(alpha=0.25)

        letter = chr(ord("a") + b_idx)
        ax.text(0.97, 0.03, f"({letter})",
                transform=ax.transAxes, fontsize=10,
                fontweight="bold", ha="right", va="bottom")

    # Hide unused panels
    for ax in axes_flat[len(basin_order):]:
        ax.set_visible(False)

    # Shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc="lower center", ncol=min(len(handles), 5),
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Supplemental -- Bloom onset month vs MLD shoaling month "
        "(detrended, one point per year)\n"
        "x-axis: month of annual MLD minimum (shallowest mixed layer); "
        "y-axis: month of annual Chl-a maximum.\n"
        "Dashed diagonal: slope=1 (bloom tracks shoaling perfectly). "
        "Regression line + statistics shown for OBS.",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()
    out_path = OUT_DIR / "figS_bloom_mld_timing.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {out_path.name}")


# ============================================================
# Analysis 2: Composite Chl-a anomaly maps around MLD extremes
# ============================================================

def _mld_extreme_years(
    mld_ts:  xr.DataArray,
    mask:    xr.DataArray,
    common_grid: xr.DataArray,
    period:  slice = LAG_PERIOD,
    quantile: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify years with anomalously deep / shallow basin-mean MLD.

    The basin-mean annual-mean MLD anomaly is computed (deseasonalised,
    detrended). Years in the top `quantile` fraction are "deep" years;
    years in the bottom `quantile` fraction are "shallow" years.

    Returns (deep_years, shallow_years) as integer arrays.
    """
    mld_re = regrid_timeseries(mld_ts.sel(time=period), common_grid)
    mld_b  = _basin_mean_timeseries(mld_re, mask)
    mld_z  = _deseasonalise_zscore(mld_b)

    # Annual mean of the detrended anomaly
    ym   = _ym_key(mld_z)
    yrs  = ym // 100
    vals = np.asarray(mld_z.values, dtype=float)
    unique_yrs = np.unique(yrs)

    ann_mean = {}
    for yr in unique_yrs:
        sel = yrs == yr
        v   = vals[sel]
        if np.any(np.isfinite(v)):
            ann_mean[yr] = float(np.nanmean(v))

    if len(ann_mean) < 4:
        return np.array([], dtype=int), np.array([], dtype=int)

    yr_arr  = np.array(list(ann_mean.keys()),   dtype=int)
    val_arr = np.array(list(ann_mean.values()), dtype=float)

    thresh_hi = np.quantile(val_arr, 1 - quantile)
    thresh_lo = np.quantile(val_arr,     quantile)

    deep_yrs    = yr_arr[val_arr >= thresh_hi]
    shallow_yrs = yr_arr[val_arr <= thresh_lo]
    return deep_yrs, shallow_yrs


def _composite_chl_map(
    chl_ts:   xr.DataArray,
    years:    np.ndarray,
    lags_mo:  list[int],
    common_grid: xr.DataArray,
    period:   slice = LAG_PERIOD,
) -> dict[int, xr.DataArray]:
    """Composite Chl-a anomaly map at specified lags after extreme years.

    For each lag k (months after January of the extreme year), compute
    the mean of the deseasonalised, detrended Chl-a anomaly field
    across all extreme years.

    Returns {lag: (lat, lon) DataArray of composite anomaly}.
    """
    if years.size == 0:
        return {}

    chl_re  = regrid_timeseries(chl_ts.sel(time=period), common_grid)
    # Deseasonalise and detrend cell-by-cell
    chl_z   = _pointwise_deseasonalise_zscore(chl_re)   # (time, lat, lon)

    ym_vals = _ym_key(chl_z)
    chl_arr = np.asarray(chl_z.values, dtype=float)

    composites: dict[int, xr.DataArray] = {}
    for lag in lags_mo:
        slices = []
        for yr in years:
            # Target: lag months after January of the extreme year
            # lag=0  → January of that year
            # lag=3  → April of that year
            # lag=6  → July of that year
            target_yr  = yr + (lag // 12)
            target_mo  = 1 + (lag % 12)
            if target_mo > 12:
                target_mo  -= 12
                target_yr  += 1
            target_key = target_yr * 100 + target_mo
            t_idx = np.where(ym_vals == target_key)[0]
            if t_idx.size == 0:
                continue
            slices.append(chl_arr[t_idx[0], :, :])

        if not slices:
            continue
        comp = np.nanmean(np.stack(slices, axis=0), axis=0)
        composites[lag] = xr.DataArray(
            comp,
            dims=("lat", "lon"),
            coords={"lat": chl_z.lat, "lon": chl_z.lon},
            name=f"chl_composite_lag{lag:+d}",
        )
    return composites


def make_figS_mld_composite(
    obs_chl_ts:   xr.DataArray,
    obs_mld_ts:   "xr.DataArray | None",
    model_chl_ts: dict[str, xr.DataArray],
    model_mld_ts: dict[str, xr.DataArray],
    common_grid:  xr.DataArray,
    period:       slice = LAG_PERIOD,
    lags_mo:      list[int] = [0, 3, 6],
    quantile:     float = 0.25,
) -> None:
    """Fig S_mld_composite: Chl-a anomaly maps after MLD extreme years.

    For each basin and each source (OBS + models), identifies years
    with anomalously deep (top quartile) and shallow (bottom quartile)
    basin-mean MLD. Composites the global Chl-a anomaly field at
    lag 0, +3, and +6 months after those extreme years.

    Layout: one figure per basin. Within each figure:
      Rows    = sources (OBS first, then models)
      Columns = deep-MLD composites at lag 0/+3/+6 months,
                then shallow-MLD composites at lag 0/+3/+6 months
      Total   = n_sources rows x 6 columns

    A diverging contourf colourmap (RdBu_r) is used so blue = anomalously
    low Chl-a, red = anomalously high Chl-a. The deep/shallow split is
    shown side-by-side to make the response asymmetry visible.

    Output files: figS_mld_composite_{basin}.png (one per basin).
    """
    logger.info("=== Fig S_mld_composite: Chl-a composites at MLD extremes ===")

    if obs_mld_ts is None and not model_mld_ts:
        logger.warning("  No MLD data; skipping figS_mld_composite")
        return

    masks       = _build_masks(common_grid)
    basin_order = [b for b in MEANSTATE_BASINS if b in masks]
    models_list = [m for m in MODELS if m in model_chl_ts]

    # Lag labels for column headers
    lag_labels  = [f"lag {l:+d} mo" for l in lags_mo]
    col_headers = (
        [f"Deep MLD\n{ll}" for ll in lag_labels]
        + [f"Shallow MLD\n{ll}" for ll in lag_labels]
    )
    ncols = len(col_headers)   # 6

    # Sources: OBS first then models
    sources: list[tuple[str,
                        xr.DataArray,
                        "xr.DataArray | None"]] = []
    if obs_mld_ts is not None:
        sources.append(("OBS", obs_chl_ts, obs_mld_ts))
    for model in models_list:
        if model in model_mld_ts:
            sources.append((model,
                            model_chl_ts[model],
                            model_mld_ts[model]))

    if not sources:
        logger.warning("  No valid sources; skipping")
        return

    nrows = len(sources)
    vmax  = 0.5   # standardised anomaly colour scale

    for basin in basin_order:
        logger.info(f"  Basin: {basin}...")
        mask = masks[basin]

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(3.2 * ncols, 3.0 * nrows),
            subplot_kw={"projection": ccrs.Robinson(central_longitude=0)},
            squeeze=False,
        )
        fig.subplots_adjust(
            left=0.04, right=0.88,
            bottom=0.06, top=0.90,
            hspace=0.08, wspace=0.04,
        )

        levels = np.linspace(-vmax, vmax, 13)
        last_cf = None

        for r_idx, (src_label, chl_ts, mld_ts) in enumerate(sources):
            # Identify extreme years for this source and basin
            try:
                deep_yrs, shallow_yrs = _mld_extreme_years(
                    mld_ts, mask, common_grid, period, quantile
                )
            except Exception as e:
                logger.warning(
                    f"  [{src_label}] {basin} extreme years failed: {e}"
                )
                for c in range(ncols):
                    axes[r_idx, c].set_visible(False)
                continue

            # Composites for deep and shallow years
            try:
                deep_comp    = _composite_chl_map(
                    chl_ts, deep_yrs,    lags_mo, common_grid, period
                )
                shallow_comp = _composite_chl_map(
                    chl_ts, shallow_yrs, lags_mo, common_grid, period
                )
            except Exception as e:
                logger.warning(
                    f"  [{src_label}] {basin} composites failed: {e}"
                )
                for c in range(ncols):
                    axes[r_idx, c].set_visible(False)
                continue

            # Plot each column
            for c_idx, (comp_dict, lag) in enumerate(
                [(deep_comp, l) for l in lags_mo]
                + [(shallow_comp, l) for l in lags_mo]
            ):
                ax = axes[r_idx, c_idx]
                comp = comp_dict.get(lag)

                if comp is None:
                    ax.set_visible(False)
                    continue

                cf = ax.contourf(
                    comp.lon, comp.lat, comp.values,
                    levels=levels,
                    cmap="RdBu_r",
                    extend="both",
                    transform=ccrs.PlateCarree(),
                )
                ax.contour(
                    comp.lon, comp.lat, comp.values,
                    levels=[0],
                    colors="k", linewidths=0.5,
                    transform=ccrs.PlateCarree(),
                )
                ax.add_feature(
                    cfeature.LAND,
                    facecolor="lightgray", zorder=4,
                )
                try:
                    ax.coastlines(linewidth=0.3, zorder=5)
                except Exception:
                    pass
                last_cf = cf

                # Column header (top row only)
                if r_idx == 0:
                    ax.set_title(col_headers[c_idx],
                                 fontsize=8, pad=3)

                # Row label (source name, left of first column)
                if c_idx == 0:
                    ax.text(
                        -0.04, 0.5, src_label,
                        transform=ax.transAxes,
                        fontsize=7.5, fontweight="bold",
                        ha="right", va="center", rotation=90,
                    )

                # Deep/shallow divider: thick left border on col 3
                if c_idx == len(lags_mo):
                    ax.spines["left"].set_linewidth(2.0)
                    ax.spines["left"].set_color("dimgrey")

        # Shared colourbar
        if last_cf is not None:
            cax = fig.add_axes([0.90, 0.15, 0.015, 0.70])
            cb  = fig.colorbar(last_cf, cax=cax, extend="both")
            cb.set_label(
                "Chl-a anomaly (standardised)", fontsize=9
            )
            cb.ax.tick_params(labelsize=8)

        # Divider line between deep and shallow columns
        # (drawn as a vertical line in figure coordinates)
        mid_x = (axes[0, len(lags_mo) - 1].get_position().x1
                  + axes[0, len(lags_mo)].get_position().x0) / 2
        fig.add_artist(
            plt.matplotlib.lines.Line2D(
                [mid_x, mid_x], [0.06, 0.90],
                transform=fig.transFigure,
                color="dimgrey", lw=1.5, ls="--",
            )
        )
        fig.text(
            mid_x - 0.10, 0.915,
            f"Deep MLD years (top {int(quantile*100)}th pct)",
            fontsize=9, fontweight="bold", ha="center",
        )
        fig.text(
            mid_x + 0.10, 0.915,
            f"Shallow MLD years (bottom {int(quantile*100)}th pct)",
            fontsize=9, fontweight="bold", ha="center",
        )
        fig.suptitle(
            f"{basin}: Chl-a anomaly composites following "
            f"anomalously deep / shallow MLD years\n"
            f"Composites at lag 0, +3, +6 months after January of "
            f"the extreme-MLD year. "
            f"Anomalies deseasonalised, linearly detrended, "
            f"standardised. Top/bottom {int(quantile*100)}th percentile.",
            fontsize=9, y=0.975,
        )
        out_path = OUT_DIR / f"figS_mld_composite_{basin}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"  wrote {out_path.name}")


def make_figS_mld_climate_combined(
    indices:       dict[str, xr.DataArray],
    obs_chl_ts:    xr.DataArray,
    obs_mld_ts:    "xr.DataArray | None",
    model_chl_ts:  dict[str, xr.DataArray],
    model_mld_ts:  dict[str, xr.DataArray],
    common_grid:   xr.DataArray,
    period:        slice = LAG_PERIOD,
) -> None:
    """Combined supplemental: MLD index TS + six-basin lag correlation.

    Identical structure to make_figS_climate_combined but with MLD
    replacing SST as the physical driver.  Three rows (one per climate
    index: AMO, Nino 3.4, PDO).  Each row has:
      Left  - standardised monthly index time series with 12-month
               running mean, positive/negative shading, +/-1 sigma lines.
      Right - all six basins (3 columns x 2 rows nested grid) showing
               the MLD-Chl-a lag correlation with that index removed from
               both MLD and Chl-a before correlating (partial correlation).

    All anomaly series are deseasonalised, linearly detrended, and
    standardised to unit variance (Section 2.6) before correlation.

    Output: figS_mld_climate_combined.png
    """
    import matplotlib.dates as mdates
    import matplotlib.gridspec as gridspec
    from pandas import Series, DatetimeIndex, Timestamp

    logger.info("=== Fig S_mld_climate_combined: MLD index TS + lag corr ===")

    if not indices:
        logger.warning("  No indices; skipping figS_mld_climate_combined")
        return
    if obs_mld_ts is None and not model_mld_ts:
        logger.warning("  No MLD data; skipping figS_mld_climate_combined")
        return

    index_meta = {
        "AMO":    {"color": "#CC3333",
                   "short": "AMO",
                   "label": "AMO index\n(N. Atl. SST minus global mean SST)"},
        "NINO34": {"color": "#3366CC",
                   "short": "Nino 3.4",
                   "label": "Nino 3.4 index\n(equatorial Pacific SST)"},
        "PDO":    {"color": "#228833",
                   "short": "PDO",
                   "label": "PDO index\n(N. Pacific leading EOF)"},
    }

    names = [n for n in ("AMO", "NINO34", "PDO") if n in indices]
    if not names:
        logger.warning("  No recognised indices; skipping")
        return

    nrows        = len(names)
    models_list  = [m for m in MODELS if m in model_chl_ts]
    masks        = _build_masks(common_grid)
    basin_order  = [b for b in MEANSTATE_BASINS if b in masks]

    # ----------------------------------------------------------------
    # Pre-compute MLD-Chl lag curves for all basins, one per index
    # ----------------------------------------------------------------
    def _mld_lag_curves(idx_name: str) -> dict[str, dict[str, tuple]]:
        """All-basin MLD-Chl lag curves with idx_name removed."""
        idx = indices[idx_name]
        curves: dict[str, dict[str, tuple]] = {b: {} for b in basin_order}

        # OBS
        if obs_mld_ts is not None:
            try:
                chl_re = regrid_timeseries(
                    obs_chl_ts.sel(time=period), common_grid)
                mld_re = regrid_timeseries(
                    obs_mld_ts.sel(time=period), common_grid)
                for basin, mask in masks.items():
                    if basin not in basin_order:
                        continue
                    chl_b = _basin_mean_timeseries(chl_re, mask)
                    mld_b = _basin_mean_timeseries(mld_re, mask)
                    chl_z = _deseasonalise_zscore(chl_b)
                    mld_z = _deseasonalise_zscore(mld_b)
                    chl_z = _regress_out(chl_z, idx)
                    mld_z = _regress_out(mld_z, idx)
                    curves[basin]["OBS"] = _lag_correlation_with_bootstrap(
                        chl_z, mld_z)
            except Exception as e:
                logger.warning(f"  OBS MLD ({idx_name}) failed: {e}")

        # Models
        for model in MODELS:
            if model not in model_chl_ts or model not in model_mld_ts:
                continue
            try:
                chl_re = regrid_timeseries(
                    model_chl_ts[model].sel(time=period), common_grid)
                mld_re = regrid_timeseries(
                    model_mld_ts[model].sel(time=period), common_grid)
                for basin, mask in masks.items():
                    if basin not in basin_order:
                        continue
                    chl_b = _basin_mean_timeseries(chl_re, mask)
                    mld_b = _basin_mean_timeseries(mld_re, mask)
                    chl_z = _deseasonalise_zscore(chl_b)
                    mld_z = _deseasonalise_zscore(mld_b)
                    chl_z = _regress_out(chl_z, idx)
                    mld_z = _regress_out(mld_z, idx)
                    curves[basin][model] = _lag_correlation_with_bootstrap(
                        chl_z, mld_z)
            except Exception as e:
                logger.warning(f"  [{model}] MLD ({idx_name}) failed: {e}")

        return curves

    logger.info("  Computing MLD lag curves for each index...")
    all_lag_curves = {n: _mld_lag_curves(n) for n in names}

    # ----------------------------------------------------------------
    # Build figure: nrows rows, outer 2 columns (TS | basin grid)
    # ----------------------------------------------------------------
    # Each row height must accommodate a 2-row basin sub-grid.
    row_h   = 5.5   # inches per row
    fig_h   = row_h * nrows + 0.8
    fig     = plt.figure(figsize=(20, fig_h))

    outer_gs = gridspec.GridSpec(
        nrows, 2,
        figure=fig,
        width_ratios=[1, 3.0],
        hspace=0.45,
        wspace=0.12,
        left=0.05, right=0.97,
        top=0.95, bottom=0.07,
    )

    for row, name in enumerate(names):
        meta       = index_meta[name]
        idx        = indices[name]
        lag_curves = all_lag_curves[name]

        # ── LEFT: index time series ───────────────────────────────
        ax_ts = fig.add_subplot(outer_gs[row, 0])

        time = idx.time.values
        try:
            t_plot = DatetimeIndex([Timestamp(str(t)) for t in time])
        except Exception:
            t_plot = np.arange(len(time))

        vals   = np.asarray(idx.values, dtype=float)
        smooth = Series(vals).rolling(12, center=True,
                                      min_periods=6).mean().values

        ax_ts.fill_between(t_plot, vals, 0,
                           where=(vals >= 0),
                           color="#FFCCCC", alpha=0.65, linewidth=0)
        ax_ts.fill_between(t_plot, vals, 0,
                           where=(vals < 0),
                           color="#CCE0FF", alpha=0.65, linewidth=0)
        ax_ts.plot(t_plot, vals,   color="grey",        lw=0.8, alpha=0.7,
                   label="Monthly")
        ax_ts.plot(t_plot, smooth, color=meta["color"], lw=2.2,
                   label="12-month mean")
        ax_ts.axhline(0,  color="k", lw=0.9)
        ax_ts.axhline(+1, color="k", lw=0.55, ls="--", alpha=0.4)
        ax_ts.axhline(-1, color="k", lw=0.55, ls="--", alpha=0.4)
        ax_ts.set_ylim(-3.2, 3.2)
        ax_ts.set_ylabel("Standardised index", fontsize=8)
        ax_ts.grid(alpha=0.2)
        ax_ts.legend(fontsize=7, loc="upper right", framealpha=0.8)

        if row < nrows - 1:
            ax_ts.set_xticklabels([])
        else:
            try:
                ax_ts.xaxis.set_major_locator(mdates.YearLocator(2))
                ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                import matplotlib.pyplot as _plt
                _plt.setp(ax_ts.xaxis.get_majorticklabels(),
                          rotation=35, ha="right")
            except Exception:
                pass
            ax_ts.set_xlabel("Year", fontsize=8.5)

        letter = chr(ord("a") + row * 2)
        ax_ts.text(0.02, 0.97, f"({letter})",
                   transform=ax_ts.transAxes, fontsize=10,
                   fontweight="bold", va="top")
        ax_ts.set_title(meta["label"], fontsize=8, loc="left", pad=3)

        # ── RIGHT: nested 3x2 basin sub-grid ─────────────────────
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 3,
            subplot_spec=outer_gs[row, 1],
            hspace=0.38, wspace=0.18,
        )

        n_basins = len(basin_order)
        for b_idx, basin in enumerate(basin_order):
            br = b_idx // 3
            bc = b_idx % 3
            ax = fig.add_subplot(inner_gs[br, bc])

            curves = lag_curves.get(basin, {})
            ax.axhline(0, color="k", lw=0.5, alpha=0.6)
            ax.axvline(0, color="k", lw=0.5, alpha=0.6, ls=":")

            sources = sorted(curves.keys(),
                             key=lambda s: (s == "OBS", s))
            for source in sources:
                entry = curves[source]
                if len(entry) == 5:
                    lags, rs, ci_lo, ci_hi, pval = entry
                else:
                    lags, rs = entry
                    nan_arr = np.full_like(rs, np.nan, dtype=float)
                    ci_lo = ci_hi = pval = nan_arr

                if source == "OBS":
                    color, lw, zord = "black", 2.2, 10
                else:
                    color = (PALETTE[models_list.index(source) % len(PALETTE)]
                             if source in models_list else "grey")
                    lw, zord = 1.0, 5

                if (np.any(np.isfinite(ci_lo))
                        and np.any(np.isfinite(ci_hi))):
                    ax.fill_between(lags, ci_lo, ci_hi,
                                    color=color, alpha=0.10,
                                    lw=0, zorder=zord - 1)
                ax.plot(lags, rs, ls="-", lw=lw, color=color,
                        alpha=0.9, zorder=zord,
                        label=source)

                sig = np.isfinite(pval) & (pval < LAG_BOOT_ALPHA)
                if np.any(sig):
                    msz = 5.0 if source == "OBS" else 3.0
                    ax.plot(lags[sig], rs[sig],
                            "o", ms=msz, mfc=color, mec="white",
                            mew=0.5, zorder=zord + 1)

                # OBS peak annotation
                if source == "OBS" and np.any(np.isfinite(rs)):
                    k_pk = int(lags[np.nanargmax(np.abs(rs))])
                    r_pk = float(rs[np.nanargmax(np.abs(rs))])
                    p_pk = (float(pval[np.nanargmax(np.abs(rs))])
                            if np.any(np.isfinite(pval)) else np.nan)
                    txt = (f"r={r_pk:+.2f}@{k_pk:+d}"
                           + (f" p={p_pk:.2f}"
                              if np.isfinite(p_pk) else ""))
                    ax.text(0.03, 0.04, txt,
                            transform=ax.transAxes, fontsize=6,
                            va="bottom", family="monospace",
                            bbox=dict(boxstyle="round,pad=0.2",
                                      facecolor="white",
                                      edgecolor="grey", alpha=0.8))

            ax.set_xlim(-LAG_MAX, LAG_MAX)
            ax.set_ylim(-1, 1)
            ax.set_xticks(np.arange(-LAG_MAX, LAG_MAX + 1, 6))
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)
            ax.set_title(basin, fontsize=8, pad=2)

            # y-label on leftmost column only
            if bc == 0:
                ax.set_ylabel("Pearson r", fontsize=7.5)
            # x-label on bottom row only
            if br == 1:
                ax.set_xlabel("Lag (mo), MLD leads ->", fontsize=7)

            # Sub-panel letter (continues from ax_ts letter)
            # ax_ts uses letter a/c/e; basin panels use b1-b6, d1-d6, f1-f6
            outer_letter = chr(ord("b") + row * 2)
            ax.text(0.02, 0.97,
                    f"({outer_letter}{b_idx + 1})",
                    transform=ax.transAxes, fontsize=7.5,
                    fontweight="bold", va="top",
                    bbox=dict(boxstyle="round,pad=0.1",
                              facecolor="white", alpha=0.7,
                              edgecolor="none"))

        # Row subtitle over the basin grid
        right_ax = fig.add_subplot(outer_gs[row, 1])
        right_ax.set_visible(False)
        right_ax.set_title(
            f"MLD-Chl-a lag correlation  ({meta['short']} removed "
            f"from both fields)",
            fontsize=8.5, pad=6,
        )

    # Shared legend (take handles from first basin panel of first row)
    # Find the first basin axes by walking fig.axes
    basin_axes = [ax for ax in fig.axes
                  if ax.get_title() in basin_order]
    if basin_axes:
        handles, labels = basin_axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="lower center",
            ncol=min(len(handles), 5),
            fontsize=8, framealpha=0.9,
            bbox_to_anchor=(0.58, -0.01),
        )

    fig.suptitle(
        "Supplemental -- MLD index time series and effect on "
        "the MLD-Chl-a lag correlation (all six basins)\n"
        "Left: standardised index (NOAA-ERSSTv3b, 1998-2014, "
        "deseasonalised, detrended).  "
        "Right: partial r(MLD, Chl | index), "
        "index removed from both fields.\n"
        "Ribbons: 95% block-bootstrap CI;  "
        "filled markers: p < 0.05;  "
        "all anomalies deseasonalised, linearly detrended, "
        "standardised (unit variance).",
        fontsize=8.5, y=0.99,
    )

    out_path = OUT_DIR / "figS_mld_climate_combined.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {out_path.name}")


def make_figS_climate_combined(
    indices:       dict[str, xr.DataArray],
    obs_chl_ts:    xr.DataArray,
    obs_sst_ts:    xr.DataArray | None,
    model_chl_ts:  dict[str, xr.DataArray],
    model_sst_ts:  dict[str, xr.DataArray],
    common_grid:   xr.DataArray,
    period:        slice = LAG_PERIOD,
) -> None:
    """Combined supplemental figure: index time series + lag correlation.

    Three rows, one per climate index (AMO, Niño 3.4, PDO).
    Each row has two panels:
      Left  — standardised monthly index time series with 12-month
               running mean, positive/negative shading, ±1σ lines.
      Right — North Atlantic SST-Chl-a lag correlation with that
               index removed from both Chl-a and SST (partial
               correlation), OBS black curve + all 7 models coloured,
               95% block-bootstrap CI ribbons, significance markers.

    The raw panel is omitted — it appears as panel (a) of Fig 9 in
    the main text. The purpose of this figure is to show (a) what
    each index looks like and (b) how the Atlantic lag-correlation
    changes when it is removed, side by side.

    Output: figS_climate_modes_combined.png
    """
    import matplotlib.dates as mdates
    from pandas import Series, DatetimeIndex, Timestamp

    logger.info("=== Fig S_climate_combined: index TS + lag correlation ===")

    if not indices:
        logger.warning("  No indices; skipping figS_climate_combined")
        return

    index_meta = {
        "AMO":    {"color": "#CC3333",
                   "short": "AMO",
                   "label": "AMO index\n(N. Atl. SST − global mean SST)"},
        "NINO34": {"color": "#3366CC",
                   "short": "Niño 3.4",
                   "label": "Niño 3.4 index\n(equatorial Pacific SST)"},
        "PDO":    {"color": "#228833",
                   "short": "PDO",
                   "label": "PDO index\n(N. Pacific leading EOF)"},
    }

    names = [n for n in ("AMO", "NINO34", "PDO") if n in indices]
    if not names:
        logger.warning("  No recognised indices; skipping")
        return

    nrows  = len(names)
    models_list = [m for m in MODELS if m in model_chl_ts]

    # ── Pre-compute Atlantic lag curves for each index removed ──────────
    masks = _build_masks(common_grid)
    atl_mask = masks.get("Atlantic")

    def _atl_lag_curve(idx_name: str) -> dict[str, tuple]:
        """North Atlantic SST-Chl lag correlation with idx_name removed."""
        curves: dict[str, tuple] = {}
        idx = indices[idx_name]

        # OBS
        if obs_sst_ts is not None:
            try:
                chl_re = regrid_timeseries(
                    obs_chl_ts.sel(time=period), common_grid
                )
                sst_re = regrid_timeseries(
                    obs_sst_ts.sel(time=period), common_grid
                )
                chl_b = _basin_mean_timeseries(chl_re, atl_mask)
                sst_b = _basin_mean_timeseries(sst_re, atl_mask)
                chl_z = _deseasonalise_zscore(chl_b)
                sst_z = _deseasonalise_zscore(sst_b)
                chl_z = _regress_out(chl_z, idx)
                sst_z = _regress_out(sst_z, idx)
                curves["OBS"] = _lag_correlation_with_bootstrap(chl_z, sst_z)
            except Exception as e:
                logger.warning(f"  OBS Atlantic ({idx_name}) failed: {e}")

        # Models
        for model in MODELS:
            if model not in model_chl_ts or model not in model_sst_ts:
                continue
            try:
                chl_re = regrid_timeseries(
                    model_chl_ts[model].sel(time=period), common_grid
                )
                sst_re = regrid_timeseries(
                    model_sst_ts[model].sel(time=period), common_grid
                )
                chl_b = _basin_mean_timeseries(chl_re, atl_mask)
                sst_b = _basin_mean_timeseries(sst_re, atl_mask)
                chl_z = _deseasonalise_zscore(chl_b)
                sst_z = _deseasonalise_zscore(sst_b)
                chl_z = _regress_out(chl_z, idx)
                sst_z = _regress_out(sst_z, idx)
                curves[model] = _lag_correlation_with_bootstrap(chl_z, sst_z)
            except Exception as e:
                logger.warning(f"  [{model}] Atlantic ({idx_name}) failed: {e}")

        return curves

    logger.info("  Computing Atlantic lag curves for each index...")
    lag_curves = {n: _atl_lag_curve(n) for n in names}

    # ── Build figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 4.2 * nrows))

    # Two-column layout: left col narrower (index TS), right col wider (lag)
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        nrows, 2,
        figure=fig,
        width_ratios=[1, 1.6],
        hspace=0.38,
        wspace=0.22,
        left=0.06, right=0.97,
        top=0.94, bottom=0.07,
    )

    for row, name in enumerate(names):
        meta   = index_meta[name]
        idx    = indices[name]
        curves = lag_curves[name]

        # ── LEFT: index time series ──────────────────────────────────
        ax_ts = fig.add_subplot(gs[row, 0])

        time = idx.time.values
        try:
            t_plot = DatetimeIndex([Timestamp(str(t)) for t in time])
        except Exception:
            t_plot = np.arange(len(time))

        vals   = np.asarray(idx.values, dtype=float)
        smooth = Series(vals).rolling(12, center=True,
                                      min_periods=6).mean().values

        ax_ts.fill_between(t_plot, vals, 0,
                           where=(vals >= 0),
                           color="#FFCCCC", alpha=0.65, linewidth=0)
        ax_ts.fill_between(t_plot, vals, 0,
                           where=(vals < 0),
                           color="#CCE0FF", alpha=0.65, linewidth=0)
        ax_ts.plot(t_plot, vals, color="grey", lw=0.8, alpha=0.7,
                   label="Monthly")
        ax_ts.plot(t_plot, smooth, color=meta["color"], lw=2.2,
                   label="12-month mean")
        ax_ts.axhline(0,  color="k", lw=0.9)
        ax_ts.axhline(+1, color="k", lw=0.55, ls="--", alpha=0.4)
        ax_ts.axhline(-1, color="k", lw=0.55, ls="--", alpha=0.4)
        ax_ts.set_ylim(-3.2, 3.2)
        ax_ts.set_ylabel("Standardised index", fontsize=8.5)
        ax_ts.grid(alpha=0.2)
        ax_ts.legend(fontsize=7.5, loc="upper right", framealpha=0.8)

        # x-axis: only label the bottom row
        if row < nrows - 1:
            ax_ts.set_xticklabels([])
        else:
            try:
                ax_ts.xaxis.set_major_locator(mdates.YearLocator(2))
                ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
                plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=35,
                         ha="right")
            except Exception:
                pass
            ax_ts.set_xlabel("Year", fontsize=9)

        # Panel letter and title
        letter = chr(ord("a") + row * 2)
        ax_ts.text(0.02, 0.97, f"({letter})",
                   transform=ax_ts.transAxes, fontsize=10,
                   fontweight="bold", va="top")
        ax_ts.set_title(meta["label"], fontsize=8.5, loc="left", pad=3)

        # ── RIGHT: lag correlation with this index removed ───────────
        ax_lag = fig.add_subplot(gs[row, 1])

        ax_lag.axhline(0, color="k", lw=0.5, alpha=0.6)
        ax_lag.axvline(0, color="k", lw=0.5, alpha=0.6, ls=":")

        sources = sorted(curves.keys(), key=lambda s: (s == "OBS", s))
        for source in sources:
            entry = curves[source]
            if len(entry) == 5:
                lags, rs, ci_lo, ci_hi, pval = entry
            else:
                lags, rs = entry
                nan_arr = np.full_like(rs, np.nan, dtype=float)
                ci_lo = ci_hi = pval = nan_arr

            if source == "OBS":
                color, lw, zord = "black", 2.4, 10
            else:
                color = (
                    PALETTE[models_list.index(source) % len(PALETTE)]
                    if source in models_list else "grey"
                )
                lw, zord = 1.1, 5

            if np.any(np.isfinite(ci_lo)) and np.any(np.isfinite(ci_hi)):
                ax_lag.fill_between(lags, ci_lo, ci_hi,
                                    color=color, alpha=0.10, lw=0,
                                    zorder=zord - 1)
            ax_lag.plot(lags, rs, ls="-", lw=lw, color=color,
                        label=source, alpha=0.9, zorder=zord)

            sig = np.isfinite(pval) & (pval < LAG_BOOT_ALPHA)
            if np.any(sig):
                msz = 5.5 if source == "OBS" else 3.5
                ax_lag.plot(lags[sig], rs[sig],
                            "o", ms=msz, mfc=color, mec="white",
                            mew=0.6, zorder=zord + 1)

            # OBS peak annotation
            if source == "OBS" and np.any(np.isfinite(rs)):
                k_pk = int(lags[np.nanargmax(np.abs(rs))])
                r_pk = float(rs[np.nanargmax(np.abs(rs))])
                p_pk = (float(pval[np.nanargmax(np.abs(rs))])
                        if np.any(np.isfinite(pval)) else np.nan)
                txt = (f"OBS: r={r_pk:+.2f} @ lag={k_pk:+d}"
                       + (f"\n(p={p_pk:.3f})" if np.isfinite(p_pk) else ""))
                ax_lag.text(
                    0.02, 0.05, txt,
                    transform=ax_lag.transAxes, fontsize=7.5,
                    va="bottom", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor="grey",
                              alpha=0.85),
                )

        ax_lag.set_xlim(-LAG_MAX, LAG_MAX)
        ax_lag.set_ylim(-1, 1)
        ax_lag.set_xticks(np.arange(-LAG_MAX, LAG_MAX + 1, 4))
        ax_lag.grid(alpha=0.25)
        ax_lag.set_ylabel("Pearson r", fontsize=8.5)
        if row == nrows - 1:
            ax_lag.set_xlabel("Lag (months)  —  SST leads →", fontsize=9)

        # Panel letter and title
        letter_lag = chr(ord("a") + row * 2 + 1)
        ax_lag.text(0.02, 0.97, f"({letter_lag})",
                    transform=ax_lag.transAxes, fontsize=10,
                    fontweight="bold", va="top")
        ax_lag.set_title(
            f"N. Atlantic SST–Chl-$a$ lag correlation\n"
            f"({meta['short']} removed from both fields)",
            fontsize=8.5, loc="left", pad=3,
        )

    # Shared legend below all panels
    handles, labels = fig.axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=min(len(handles), 5),
        fontsize=8, framealpha=0.9,
        bbox_to_anchor=(0.55, -0.01),
    )

    fig.suptitle(
        "Supplemental — Climate index time series and their effect on the "
        "North Atlantic SST–Chl-$a$ lag correlation\n"
        "Left: standardised index (NOAA-ERSSTv3b, 1998–2014, "
        "deseasonalised, detrended).  "
        "Right: partial $r$(SST, Chl $|$ index), "
        "index removed from both fields before correlation.\n"
        "Ribbons: 95% block-bootstrap CI;  filled markers: $p < 0.05$.",
        fontsize=8.5, y=0.99,
    )

    out_path = OUT_DIR / "figS_climate_modes_combined.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {out_path.name}")


# ============================================================
# Fig 9 variant: SST-Chl lag correlation with indices removed
# Produces TWO files:
#   fig9_lag_corr_chl_sst.png         — raw (no index removed)
#   fig9b_lag_corr_chl_sst_amo_removed.png — AMO+ENSO+PDO removed
# And one supplemental 2x2 Atlantic sensitivity panel:
#   figS_climate_modes_atlantic.png   — raw vs each index removed
# ============================================================

def make_fig9_lag_sst_with_modes(
    obs_chl_ts:    xr.DataArray,
    obs_sst_ts:    xr.DataArray | None,
    model_chl_ts:  dict[str, xr.DataArray],
    model_sst_ts:  dict[str, xr.DataArray],
    common_grid:   xr.DataArray,
    indices:       dict[str, xr.DataArray],
) -> None:
    """Produce Fig 9 (raw) and Fig 9b (indices removed), plus the
    supplemental Atlantic sensitivity panel.

    Fig 9  — fig9_lag_corr_chl_sst.png
        Identical to the original make_fig9_lag_sst output: six-basin
        lag correlation with no index removed.  Kept as the main-text
        figure so the raw coupling is always shown.

    Fig 9b — fig9b_lag_corr_chl_sst_amo_removed.png
        Same six-basin layout but with AMO, Niño 3.4, and PDO all
        regressed out of both the Chl-a and SST basin-mean series before
        correlation.  For models, the OBS-derived indices are used (the
        standard approach when model internal variability need not match
        the observed modes phase-by-phase).

    Fig S_amo — figS_climate_modes_atlantic.png  (supplemental)
        2×2 panel restricted to the North Atlantic:
          (a) raw
          (b) AMO removed
          (c) Niño 3.4 removed
          (d) PDO removed
        Allows the reader to see the contribution of each mode
        independently.
    """
    logger.info(
        "=== Fig 9 + Fig 9b + Fig S_amo: SST-Chl lag (raw & modes removed) ==="
    )
    masks = _build_masks(common_grid)

    # ------------------------------------------------------------------
    # Helper: build lag curves for all basins, optionally removing indices
    # ------------------------------------------------------------------
    def _build_lag_curves(
        remove: list[str],
    ) -> dict[str, dict[str, tuple]]:
        """Compute basin-mean SST-Chl lag curves.

        remove: list of index names (subset of indices keys) to regress
                out of both Chl and SST before correlation.
        """
        lag_curves: dict[str, dict[str, tuple]] = {b: {} for b in masks}

        # ---- OBS ----
        if obs_sst_ts is not None:
            try:
                chl_re, sst_re = _prep_for_lag(
                    obs_chl_ts, obs_sst_ts, common_grid
                )
                for basin, mask in masks.items():
                    chl_b = _basin_mean_timeseries(chl_re, mask)
                    sst_b = _basin_mean_timeseries(sst_re, mask)
                    chl_z = _deseasonalise_zscore(chl_b)
                    sst_z = _deseasonalise_zscore(sst_b)
                    for idx_name in remove:
                        if idx_name in indices:
                            chl_z = _regress_out(chl_z, indices[idx_name])
                            sst_z = _regress_out(sst_z, indices[idx_name])
                    result = _lag_correlation_with_bootstrap(chl_z, sst_z)
                    lag_curves[basin]["OBS"] = result
                    lags, rs = result[0], result[1]
                    if np.any(np.isfinite(rs)):
                        k = int(lags[np.nanargmax(np.abs(rs))])
                        r = float(rs[np.nanargmax(np.abs(rs))])
                        p = float(result[4][np.nanargmax(np.abs(rs))])
                        tag = (f"remove={remove}" if remove else "raw")
                        logger.success(
                            f"  [OBS  ] {basin:10s} ({tag}): "
                            f"peak r={r:+.3f} @ lag={k:+d} (p={p:.3f})"
                        )
            except Exception as e:
                import traceback
                logger.warning(f"  OBS lag-corr failed: {e}")
                logger.warning(traceback.format_exc().replace("<", "\\<"))

        # ---- models ----
        for model in MODELS:
            if model not in model_chl_ts or model not in model_sst_ts:
                continue
            try:
                chl_re, sst_re = _prep_for_lag(
                    model_chl_ts[model], model_sst_ts[model], common_grid
                )
                for basin, mask in masks.items():
                    chl_b = _basin_mean_timeseries(chl_re, mask)
                    sst_b = _basin_mean_timeseries(sst_re, mask)
                    chl_z = _deseasonalise_zscore(chl_b)
                    sst_z = _deseasonalise_zscore(sst_b)
                    # For models: use the OBS-derived indices
                    # (models need not reproduce the observed phase of AMO/ENSO/PDO)
                    for idx_name in remove:
                        if idx_name in indices:
                            chl_z = _regress_out(chl_z, indices[idx_name])
                            sst_z = _regress_out(sst_z, indices[idx_name])
                    lag_curves[basin][model] = _lag_correlation_with_bootstrap(
                        chl_z, sst_z
                    )
            except Exception as e:
                logger.warning(f"  [{model}] lag-corr failed: {e}")

        return lag_curves

    # ------------------------------------------------------------------
    # Fig 9 — raw (no removal)
    # ------------------------------------------------------------------
    raw_curves = _build_lag_curves(remove=[])
    _make_lag_figure(
        raw_curves,
        obs_label="ESA-CCI Chl vs NOAA-ERSSTv3b SST",
        driver_name="SST",
        title="Lag correlation: basin-mean Chl-a vs SST anomalies",
        filename="fig9_lag_corr_chl_sst.png",
        fig_num="9",
    )

    if not indices:
        logger.warning(
            "  No climate indices computed; skipping Fig 9b and Fig S_amo"
        )
        return

    # ------------------------------------------------------------------
    # Fig 9b — all three indices removed simultaneously
    # ------------------------------------------------------------------
    all_names = [n for n in ("AMO", "NINO34", "PDO") if n in indices]
    removed_curves = _build_lag_curves(remove=all_names)
    label_removed = " + ".join(all_names)
    _make_lag_figure(
        removed_curves,
        obs_label=(
            f"ESA-CCI Chl vs NOAA-ERSSTv3b SST  "
            f"({label_removed} removed)"
        ),
        driver_name="SST",
        title=(
            f"Lag correlation: Chl-a vs SST anomalies "
            f"({label_removed} removed)"
        ),
        filename="fig9b_lag_corr_chl_sst_amo_removed.png",
        fig_num="9b",
    )

    # ------------------------------------------------------------------
    # Fig S_amo — North Atlantic only, 2×2: raw + one index removed per panel
    # ------------------------------------------------------------------
    logger.info("  Building Fig S_amo (Atlantic sensitivity, 2x2)...")

    atl_mask = masks.get("Atlantic")
    if atl_mask is None:
        logger.warning("  Atlantic mask not found; skipping Fig S_amo")
        return

    # Four panels: raw, then one index removed per panel
    panel_specs: list[tuple[str, list[str]]] = [
        ("Raw (no index removed)",    []),
        ("AMO removed",               ["AMO"]),
        ("Niño 3.4 (ENSO) removed",  ["NINO34"]),
        ("PDO removed",               ["PDO"]),
    ]

    # Pre-compute Atlantic curves for each panel spec
    panel_curves: list[dict[str, tuple]] = []
    for panel_title, remove in panel_specs:
        atl: dict[str, tuple] = {}

        # OBS
        if obs_sst_ts is not None:
            try:
                chl_re, sst_re = _prep_for_lag(
                    obs_chl_ts, obs_sst_ts, common_grid
                )
                chl_b = _basin_mean_timeseries(chl_re, atl_mask)
                sst_b = _basin_mean_timeseries(sst_re, atl_mask)
                chl_z = _deseasonalise_zscore(chl_b)
                sst_z = _deseasonalise_zscore(sst_b)
                for idx_name in remove:
                    if idx_name in indices:
                        chl_z = _regress_out(chl_z, indices[idx_name])
                        sst_z = _regress_out(sst_z, indices[idx_name])
                atl["OBS"] = _lag_correlation_with_bootstrap(chl_z, sst_z)
            except Exception as e:
                logger.warning(f"  OBS Atlantic ({panel_title}) failed: {e}")

        # Models
        for model in MODELS:
            if model not in model_chl_ts or model not in model_sst_ts:
                continue
            try:
                chl_re, sst_re = _prep_for_lag(
                    model_chl_ts[model], model_sst_ts[model], common_grid
                )
                chl_b = _basin_mean_timeseries(chl_re, atl_mask)
                sst_b = _basin_mean_timeseries(sst_re, atl_mask)
                chl_z = _deseasonalise_zscore(chl_b)
                sst_z = _deseasonalise_zscore(sst_b)
                for idx_name in remove:
                    if idx_name in indices:
                        chl_z = _regress_out(chl_z, indices[idx_name])
                        sst_z = _regress_out(sst_z, indices[idx_name])
                atl[model] = _lag_correlation_with_bootstrap(chl_z, sst_z)
            except Exception as e:
                logger.warning(f"  [{model}] Atlantic ({panel_title}) failed: {e}")

        panel_curves.append(atl)

    # Plot 2×2
    fig, axes = plt.subplots(
        2, 2, figsize=(12, 8),
        sharex=True, sharey=True,
    )
    axes_flat = axes.flatten()
    models_in = [m for m in MODELS if m in model_chl_ts]

    for pi, (ax, (panel_title, _)) in enumerate(
        zip(axes_flat, panel_specs)
    ):
        curves = panel_curves[pi]
        ax.axhline(0, color="k", lw=0.5, alpha=0.6)
        ax.axvline(0, color="k", lw=0.5, alpha=0.6, ls=":")

        sources = sorted(curves.keys(), key=lambda s: (s == "OBS", s))
        for source in sources:
            entry = curves[source]
            if len(entry) == 5:
                lags, rs, ci_lo, ci_hi, pval = entry
            else:
                lags, rs = entry
                nan_arr  = np.full_like(rs, np.nan, dtype=float)
                ci_lo = ci_hi = pval = nan_arr

            if source == "OBS":
                color, lw, zord = "black", 2.4, 10
            else:
                color = (
                    PALETTE[models_in.index(source) % len(PALETTE)]
                    if source in models_in else "grey"
                )
                lw, zord = 1.2, 5

            if np.any(np.isfinite(ci_lo)) and np.any(np.isfinite(ci_hi)):
                ax.fill_between(
                    lags, ci_lo, ci_hi,
                    color=color, alpha=0.12, lw=0, zorder=zord - 1,
                )
            ax.plot(lags, rs, ls="-", lw=lw, color=color,
                    label=source, alpha=0.9, zorder=zord)

            sig = np.isfinite(pval) & (pval < LAG_BOOT_ALPHA)
            if np.any(sig):
                msz = 5.5 if source == "OBS" else 3.5
                ax.plot(
                    lags[sig], rs[sig],
                    "o", ms=msz, mfc=color, mec="white",
                    mew=0.6, zorder=zord + 1,
                )

            # OBS peak annotation
            if source == "OBS" and np.any(np.isfinite(rs)):
                k_pk = int(lags[np.nanargmax(np.abs(rs))])
                r_pk = float(rs[np.nanargmax(np.abs(rs))])
                p_pk = (float(pval[np.nanargmax(np.abs(rs))])
                        if np.any(np.isfinite(pval)) else np.nan)
                txt = (
                    f"OBS: r={r_pk:+.2f} @ lag={k_pk:+d}"
                    + (f"\n(p={p_pk:.3f})" if np.isfinite(p_pk) else "")
                )
                ax.text(
                    0.02, 0.05, txt,
                    transform=ax.transAxes, fontsize=7.5,
                    va="bottom", family="monospace",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor="grey",
                              alpha=0.85),
                )

        # Panel label
        letter = string.ascii_lowercase[pi]
        ax.text(
            0.02, 0.98, f"({letter})",
            transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top",
        )
        ax.set_title(panel_title, fontsize=10)
        ax.set_xlim(-LAG_MAX, LAG_MAX)
        ax.set_ylim(-1, 1)
        ax.set_xticks(np.arange(-LAG_MAX, LAG_MAX + 1, 4))
        ax.grid(alpha=0.25)
        if pi % 2 == 0:
            ax.set_ylabel("Pearson r", fontsize=10)
        if pi >= 2:
            ax.set_xlabel("Lag (months)  —  SST leads →", fontsize=9)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=min(len(handles), 5),
        fontsize=8, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "North Atlantic SST–Chl-$a$ lag correlation: "
        "sensitivity to climate mode removal\n"
        "OBS: ESA-CCI Chl vs NOAA-ERSSTv3b SST; "
        "ribbons: 95% block-bootstrap CI; filled markers: $p<0.05$",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    out_path = OUT_DIR / "figS_climate_modes_atlantic.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {out_path.name}")

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


# ============================================================
# Bloom phenology + sea-ice retreat timing analysis
# ============================================================

def _bloom_doy_field(
    chl_ts: xr.DataArray,
    common_grid: xr.DataArray,
    period: slice = LAG_PERIOD,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute mean and std of bloom peak day-of-year at every grid cell.

    For each year and each (lat, lon) cell, the bloom peak is defined
    as the calendar month with the highest Chl-a value in the RAW
    (not deseasonalised) monthly series.  The month index is converted
    to an approximate day-of-year (mid-month: Jan=15, Feb=46, ...).

    Only cells with at least 5 valid years are returned; others are NaN.

    Returns
    -------
    mean_doy : (lat, lon) — mean bloom peak DOY across years
    std_doy  : (lat, lon) — interannual std of bloom peak DOY
    """
    chl_re = regrid_timeseries(chl_ts.sel(time=period), common_grid)
    arr    = np.asarray(chl_re.values, dtype=float)   # (T, nlat, nlon)

    ym     = _ym_key(chl_re)
    years  = ym // 100
    months = ym % 100
    unique_yrs = np.unique(years)

    T, nlat, nlon = arr.shape
    # Mid-month DOY lookup (non-leap year)
    _MID_DOY = np.array([0, 15, 46, 74, 105, 135,
                          166, 196, 227, 258, 288, 319, 349])

    doy_by_year = np.full((len(unique_yrs), nlat, nlon), np.nan)

    for yi, yr in enumerate(unique_yrs):
        sel = years == yr
        if sel.sum() < 6:          # need most months of the year
            continue
        mo_sel  = months[sel]
        arr_sel = arr[sel, :, :]   # (n_months, nlat, nlon)

        ok_cell = np.any(np.isfinite(arr_sel), axis=0)  # (nlat,nlon)
        peak_mo = np.full((nlat, nlon), np.nan)

        for j in range(nlat):
            for i in range(nlon):
                if not ok_cell[j, i]:
                    continue
                vals = arr_sel[:, j, i]
                fin  = np.isfinite(vals)
                # Lower threshold to 3 so OBS polar cells (fewer
                # valid months due to polar night) are not excluded
                if fin.sum() < 3:
                    continue
                peak_idx     = np.argmax(vals[fin])
                peak_mo[j,i] = float(mo_sel[fin][peak_idx])

        # Safe DOY lookup: only where peak_mo is a valid month (1-12)
        valid_peak = np.isfinite(peak_mo) & (peak_mo >= 1) & (peak_mo <= 12)
        doy_by_year[yi] = np.where(
            valid_peak,
            _MID_DOY[np.where(valid_peak,
                               peak_mo.astype(int), 1)],
            np.nan,
        )

    n_valid  = np.sum(np.isfinite(doy_by_year), axis=0)
    mean_doy = np.where(n_valid >= 3,
                        np.nanmean(doy_by_year, axis=0), np.nan)
    std_doy  = np.where(n_valid >= 3,
                        np.nanstd( doy_by_year, axis=0), np.nan)

    coords = {"lat": chl_re.lat, "lon": chl_re.lon}
    return (
        xr.DataArray(mean_doy, dims=("lat","lon"), coords=coords,
                     name="mean_bloom_doy"),
        xr.DataArray(std_doy,  dims=("lat","lon"), coords=coords,
                     name="std_bloom_doy"),
    )


def _seaice_retreat_doy_field(
    sic_ts: xr.DataArray,
    common_grid: xr.DataArray,
    period: slice = LAG_PERIOD,
    threshold: float = 0.15,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Mean and std of sea-ice retreat day-of-year at every grid cell.

    Retreat DOY is the first month in the calendar year when siconc
    drops below `threshold` (default 0.15 = 15%, standard NSIDC extent
    threshold).  Only cells that are seasonally ice-covered (mean siconc
    > threshold in at least one month) are included; perennially open or
    perennially ice-covered cells are masked.

    Returns mean_doy, std_doy — both (lat, lon) DataArrays.
    """
    sic_re = regrid_timeseries(sic_ts.sel(time=period), common_grid)
    arr    = np.asarray(sic_re.values, dtype=float)

    ym     = _ym_key(sic_re)
    years  = ym // 100
    months = ym % 100
    unique_yrs = np.unique(years)

    T, nlat, nlon = arr.shape
    _MID_DOY = np.array([0, 15, 46, 74, 105, 135,
                          166, 196, 227, 258, 288, 319, 349])

    doy_by_year = np.full((len(unique_yrs), nlat, nlon), np.nan)

    for yi, yr in enumerate(unique_yrs):
        sel    = years == yr
        mo_sel = months[sel]
        a_sel  = arr[sel, :, :]

        for j in range(nlat):
            for i in range(nlon):
                col = a_sel[:, j, i]
                fin = np.isfinite(col)
                if fin.sum() < 6:
                    continue
                # Find first month below threshold in this year
                below = (col < threshold) & fin
                if not np.any(below):
                    continue   # stays ice-covered all year
                if np.all(below):
                    continue   # ice-free all year
                first_idx = int(np.argmax(below))
                doy_by_year[yi, j, i] = float(
                    _MID_DOY[int(mo_sel[first_idx])]
                )

    n_valid  = np.sum(np.isfinite(doy_by_year), axis=0)
    mean_doy = np.where(n_valid >= 5,
                        np.nanmean(doy_by_year, axis=0), np.nan)
    std_doy  = np.where(n_valid >= 5,
                        np.nanstd( doy_by_year, axis=0), np.nan)

    coords = {"lat": sic_re.lat, "lon": sic_re.lon}
    return (
        xr.DataArray(mean_doy, dims=("lat","lon"), coords=coords,
                     name="mean_retreat_doy"),
        xr.DataArray(std_doy,  dims=("lat","lon"), coords=coords,
                     name="std_retreat_doy"),
    )


def _bloom_retreat_regression(
    chl_ts:     xr.DataArray,
    sic_ts:     xr.DataArray,
    common_grid: xr.DataArray,
    period:     slice = LAG_PERIOD,
    threshold:  float = 0.15,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Regress bloom peak DOY anomaly onto sea-ice retreat DOY anomaly.

    At each (lat, lon) cell, computes the OLS slope (days of bloom
    delay per day of late ice retreat), the R^2, and the p-value
    (two-sided t-test on the slope) across years.

    Only cells with >= 5 valid year pairs are included.

    Returns slope, r2, pval — all (lat, lon) DataArrays.
    """
    from scipy import stats as _stats

    chl_re = regrid_timeseries(chl_ts.sel(time=period), common_grid)
    sic_re = regrid_timeseries(sic_ts.sel(time=period), common_grid)

    ym_c = _ym_key(chl_re);  yrs_c = ym_c // 100; mo_c = ym_c % 100
    ym_s = _ym_key(sic_re);  yrs_s = ym_s // 100; mo_s = ym_s % 100

    arr_c = np.asarray(chl_re.values, dtype=float)
    arr_s = np.asarray(sic_re.values, dtype=float)

    _MID_DOY = np.array([0, 15, 46, 74, 105, 135,
                          166, 196, 227, 258, 288, 319, 349])

    common_yrs = np.intersect1d(np.unique(yrs_c), np.unique(yrs_s))
    T, nlat, nlon = arr_c.shape

    bloom_doy   = np.full((len(common_yrs), nlat, nlon), np.nan)
    retreat_doy = np.full((len(common_yrs), nlat, nlon), np.nan)

    for yi, yr in enumerate(common_yrs):
        # Bloom DOY
        sel_c  = yrs_c == yr
        mo_sel_c = mo_c[sel_c]
        a_sel_c  = arr_c[sel_c]
        for j in range(nlat):
            for i in range(nlon):
                col = a_sel_c[:, j, i]
                fin = np.isfinite(col)
                if fin.sum() < 3:
                    continue
                pk = int(mo_sel_c[fin][np.argmax(col[fin])])
                if 1 <= pk <= 12:
                    bloom_doy[yi, j, i] = float(_MID_DOY[pk])

        # Retreat DOY
        sel_s    = yrs_s == yr
        mo_sel_s = mo_s[sel_s]
        a_sel_s  = arr_s[sel_s]
        for j in range(nlat):
            for i in range(nlon):
                col = a_sel_s[:, j, i]
                fin = np.isfinite(col)
                if fin.sum() < 3:
                    continue
                below = (col < threshold) & fin
                if not np.any(below) or np.all(below):
                    continue
                first = int(np.argmax(below))
                mo_first = int(mo_sel_s[first])
                if 1 <= mo_first <= 12:
                    retreat_doy[yi, j, i] = float(
                        _MID_DOY[mo_first]
                    )

    # OLS regression cell by cell
    slope_arr = np.full((nlat, nlon), np.nan)
    r2_arr    = np.full((nlat, nlon), np.nan)
    pval_arr  = np.full((nlat, nlon), np.nan)

    for j in range(nlat):
        for i in range(nlon):
            b = bloom_doy[:, j, i]
            r = retreat_doy[:, j, i]
            ok = np.isfinite(b) & np.isfinite(r)
            if ok.sum() < 3:
                continue
            sl, ic, rval, pv, _ = _stats.linregress(r[ok], b[ok])
            slope_arr[j, i] = sl
            r2_arr[j, i]    = rval ** 2
            pval_arr[j, i]  = pv

    coords = {"lat": chl_re.lat, "lon": chl_re.lon}
    return (
        xr.DataArray(slope_arr, dims=("lat","lon"), coords=coords,
                     name="slope"),
        xr.DataArray(r2_arr,   dims=("lat","lon"), coords=coords,
                     name="r2"),
        xr.DataArray(pval_arr, dims=("lat","lon"), coords=coords,
                     name="pval"),
    )


def _interp_time_gaps(
    ts: xr.DataArray,
    max_gap: int = 3,
) -> xr.DataArray:
    """Linearly interpolate NaN gaps along the time axis, per cell.

    Only fills gaps of <= max_gap consecutive NaN months. Longer gaps
    (e.g. full polar night of 6 months) are left as NaN to avoid
    unreliable extrapolation. This is applied to OBS Chl-a before
    deseasonalisation so that the monthly climatology benefits from
    the interpolated values as well.

    Parameters
    ----------
    ts      : (time, lat, lon) DataArray
    max_gap : maximum number of consecutive NaN months to fill (default 3)
    """
    arr   = np.asarray(ts.values, dtype=float).copy()
    T, nlat, nlon = arr.shape

    for j in range(nlat):
        for i in range(nlon):
            col = arr[:, j, i]
            if np.all(np.isfinite(col)) or np.all(~np.isfinite(col)):
                continue   # nothing to do
            # Find indices of finite values
            fin_idx = np.where(np.isfinite(col))[0]
            if fin_idx.size < 2:
                continue
            # Walk through NaN runs and fill short gaps
            t = 0
            while t < T:
                if np.isfinite(col[t]):
                    t += 1
                    continue
                # Start of a NaN run
                run_start = t
                while t < T and not np.isfinite(col[t]):
                    t += 1
                run_end = t   # first finite after gap (exclusive)
                gap_len = run_end - run_start
                if gap_len > max_gap:
                    continue   # too long — leave as NaN
                # Need valid values on both sides
                if run_start == 0 or run_end >= T:
                    continue   # at edge — cannot interpolate
                v0 = col[run_start - 1]
                v1 = col[run_end]
                if not (np.isfinite(v0) and np.isfinite(v1)):
                    continue
                # Linear interpolation
                for k in range(gap_len):
                    frac = (k + 1) / (gap_len + 1)
                    col[run_start + k] = v0 + frac * (v1 - v0)
            arr[:, j, i] = col

    return ts.copy(data=arr)


def _deseas_detrend_zscore(ts: xr.DataArray) -> xr.DataArray:
    """Deseasonalise, linearly detrend, and z-score a (time,lat,lon) field.

    std is computed only over finite values per cell so that polar cells
    with all-NaN winters (polar night) are not masked out — their
    summer-only std is non-zero and valid. Cells where every value is NaN
    are left as NaN throughout.

    Uses numpy-based monthly climatology (via _ym_key) rather than
    xarray groupby so that cftime calendars (noleap, 360-day, etc.)
    are handled correctly without requiring a pandas-compatible time axis.
    """
    arr  = np.asarray(ts.values, dtype=float)
    T    = arr.shape[0]
    # Extract month index (1-12) from time axis — calendar-agnostic
    ym   = _ym_key(ts)
    mo   = (ym % 100).astype(int)   # shape (T,)

    # Monthly climatology: mean over all time steps sharing the same month
    clim_arr = np.full((12,) + arr.shape[1:], np.nan)
    for m in range(1, 13):
        sel = (mo == m)
        if sel.any():
            clim_arr[m-1] = np.nanmean(arr[sel], axis=0)

    # Subtract climatology
    anom_arr = arr.copy()
    for m in range(1, 13):
        sel = (mo == m)
        if sel.any():
            anom_arr[sel] -= clim_arr[m-1]
    anom_arr[~np.isfinite(arr)] = np.nan

    vals  = anom_arr
    t_c   = np.arange(T, dtype=float)
    t_c  -= t_c.mean()
    num   = np.nansum(t_c[:,None,None] * vals, axis=0)
    denom = np.nansum(t_c**2)
    slope = num / denom if denom != 0 else np.zeros_like(num)
    det   = vals - slope[None,:,:] * t_c[:,None,None]
    det[~np.isfinite(vals)] = np.nan
    std   = np.nanstd(det, axis=0, ddof=0)
    ok    = np.isfinite(std) & (std > 0)
    nrm   = np.where(ok[None,:,:],
                     det / np.where(ok, std, 1.0)[None,:,:],
                     np.nan)
    # Return as DataArray preserving original coordinates
    return ts.copy(data=nrm)


def make_figS_seaice_chl_spatial_corr(
    obs_chl_ts:      xr.DataArray,
    obs_siconc_ts:   xr.DataArray | None,
    model_chl_ts:    dict[str, xr.DataArray],
    model_siconc_ts: dict[str, xr.DataArray],
    common_grid:     xr.DataArray,
    period:          slice = LAG_PERIOD,
) -> None:
    """Pointwise siconc-Chl-a correlation maps with beta contours.

    Produces two figures:
      figS_seaice_chl_corr_Arctic.png
      figS_seaice_chl_corr_Southern.png

    Each figure has three panels:
      (a) OBS   — filled contourf of r(siconc, Chl-a)
      (b) Ensemble mean r
      (c) Ensemble std of r (inter-model spread)

    Dashed contour lines overlaid on (a) and (b) show the OLS
    regression coefficient beta (Chl-a anomaly per unit siconc
    anomaly), with negative beta dashed and positive solid.

    Lag used:
      Arctic       — lag 0  (siconc and Chl-a same month)
      Southern SIZ — lag +3 (siconc leads Chl-a by 3 months)

    Both siconc and Chl-a anomalies are deseasonalised, linearly
    detrended, and standardised to unit variance per cell before
    correlation, using _pointwise_deseasonalise_zscore.
    """
    from scipy import stats as _stats

    logger.info(
        "=== Fig S_seaice_chl_corr: pointwise siconc-Chl-a "
        "correlation maps ==="
    )

    if obs_siconc_ts is None and not model_siconc_ts:
        logger.warning(
            "  No siconc data; skipping figS_seaice_chl_corr"
        )
        return

    regions = {
        "Arctic": {
            "lat_min":  45, "lat_max":  90,
            "lag":       0,
            "proj":     ccrs.NorthPolarStereo(),
            "extent":   [-180, 180, 43, 90],
        },
        "Southern": {
            "lat_min": -78, "lat_max": -45,
            "lag":       3,
            "proj":     ccrs.SouthPolarStereo(),
            "extent":   [-180, 180, -78, -43],
        },
    }

    lats = common_grid.lat.values
    lons = common_grid.lon.values
    lat2d, _ = np.meshgrid(lats, lons, indexing="ij")
    models_list = [m for m in MODELS
                   if m in model_chl_ts and m in model_siconc_ts]

    def _compute_r_beta(
        sic_ts: xr.DataArray,
        chl_ts: xr.DataArray,
        lag:    int,
        reg_mask: np.ndarray,
        interpolate_chl: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise Pearson r and beta at specified lag.

        Both fields preprocessed per cell (deseasonalise, detrend,
        z-score). At lag k > 0 siconc leads Chl-a.

        Returns r_map, beta_map — both (nlat, nlon), NaN outside mask.
        """
        sic_re = regrid_timeseries(
            sic_ts.sel(time=period), common_grid)
        chl_re = regrid_timeseries(
            chl_ts.sel(time=period), common_grid)

        # Interpolate short NaN gaps in OBS Chl-a (polar night)
        if interpolate_chl:
            chl_re = _interp_time_gaps(chl_re, max_gap=3)

        sic_z = _deseas_detrend_zscore(sic_re)
        chl_z = _deseas_detrend_zscore(chl_re)

        # Align on year-month keys
        ym_s = _ym_key(sic_z)
        ym_c = _ym_key(chl_z)
        common_ym = np.intersect1d(ym_s, ym_c)
        s_idx = np.argsort(ym_s)[
            np.searchsorted(np.sort(ym_s), common_ym)]
        c_idx = np.argsort(ym_c)[
            np.searchsorted(np.sort(ym_c), common_ym)]

        S = np.asarray(sic_z.values, dtype=float)[s_idx]  # (T,nlat,nlon)
        C = np.asarray(chl_z.values, dtype=float)[c_idx]
        T = S.shape[0]

        # Apply lag: siconc at t, Chl-a at t+lag
        if lag > 0 and lag < T:
            S_lag = S[:T - lag]
            C_lag = C[lag:]
        else:
            S_lag = S
            C_lag = C

        nlat, nlon = S_lag.shape[1], S_lag.shape[2]
        r_map    = np.full((nlat, nlon), np.nan)
        beta_map = np.full((nlat, nlon), np.nan)

        for j in range(nlat):
            for i in range(nlon):
                if not reg_mask[j, i]:
                    continue
                s = S_lag[:, j, i]
                c = C_lag[:, j, i]
                ok = np.isfinite(s) & np.isfinite(c)
                if ok.sum() < 10:
                    continue
                r, _ = _stats.pearsonr(s[ok], c[ok])
                # OLS beta (s already z-scored, so beta = r * std(c)/std(s)
                # but since both are z-scored, beta ≈ r)
                beta = float(
                    np.dot(s[ok] - s[ok].mean(),
                           c[ok] - c[ok].mean())
                    / np.dot(s[ok] - s[ok].mean(),
                             s[ok] - s[ok].mean())
                )
                r_map[j, i]    = r
                beta_map[j, i] = beta

        return r_map, beta_map

    LAGS = [0, 1, 2, 3]   # siconc leads Chl-a by 0-3 months
    r_levels = np.linspace(-0.45, 0.45, 15)
    r_cmap   = "RdBu_r"

    for region_name, reg in regions.items():
        lat_min  = reg["lat_min"]
        lat_max  = reg["lat_max"]
        proj     = reg["proj"]
        extent   = reg["extent"]
        reg_mask = (lat2d >= lat_min) & (lat2d <= lat_max)

        logger.info(f"  {region_name}: computing r maps for lags {LAGS}...")

        # Pre-compute r maps for OBS and models at each lag
        # obs_r_by_lag[lag]  = (nlat,nlon) array
        # ens_r_by_lag[lag]  = (nlat,nlon) ensemble mean
        obs_r_by_lag: dict[int, np.ndarray] = {}
        ens_r_by_lag: dict[int, np.ndarray] = {}

        for lag in LAGS:
            # OBS
            if obs_siconc_ts is not None:
                try:
                    obs_r, _ = _compute_r_beta(
                        obs_siconc_ts, obs_chl_ts,
                        lag, reg_mask, interpolate_chl=True)
                    obs_r_by_lag[lag] = obs_r
                    logger.success(
                        f"  OBS {region_name} lag{lag:+d}: "
                        f"median r={np.nanmedian(obs_r):.3f}")
                except Exception as e:
                    logger.warning(
                        f"  OBS {region_name} lag{lag:+d} failed: {e}")

            # Models
            model_r_list = []
            for model in models_list:
                try:
                    r_m, _ = _compute_r_beta(
                        model_siconc_ts[model],
                        model_chl_ts[model],
                        lag, reg_mask)
                    model_r_list.append(r_m)
                except Exception as e:
                    logger.warning(
                        f"  [{model}] {region_name} lag{lag:+d} failed: {e}")
            if model_r_list:
                ens_r_by_lag[lag] = np.nanmean(
                    np.stack(model_r_list, 0), 0)

        # ── Figure: 2 rows (OBS | ensemble mean) x 4 cols (lags) ──
        n_lags = len(LAGS)
        fig, axes = plt.subplots(
            2, n_lags,
            figsize=(5.5 * n_lags, 11),
            subplot_kw={"projection": proj},
            squeeze=False,
        )
        fig.subplots_adjust(
            left=0.04, right=0.88,
            bottom=0.04, top=0.92,
            hspace=0.10, wspace=0.06,
        )

        last_cf = None
        row_labels = [
            "OBS\n(ESA-CCI Chl / HadISST siconc)",
            f"Ensemble mean\n({len(models_list)} models)",
        ]

        for row, (r_by_lag, row_lbl) in enumerate([
            (obs_r_by_lag, row_labels[0]),
            (ens_r_by_lag, row_labels[1]),
        ]):
            for col, lag in enumerate(LAGS):
                ax = axes[row, col]
                data = r_by_lag.get(lag)

                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.add_feature(cfeature.LAND,
                               facecolor="lightgray", zorder=4)
                try:
                    ax.coastlines(linewidth=0.4, zorder=5)
                except Exception:
                    pass
                ax.gridlines(linestyle=":", linewidth=0.3, alpha=0.4)

                if data is None or not np.any(np.isfinite(data)):
                    ax.text(0.5, 0.5, "no data",
                            transform=ax.transAxes,
                            ha="center", va="center",
                            fontsize=10, color="grey")
                else:
                    cf = ax.contourf(
                        lons, lats, data,
                        levels=r_levels, cmap=r_cmap,
                        extend="both",
                        transform=ccrs.PlateCarree())
                    ax.contour(
                        lons, lats, data,
                        levels=[0], colors="k",
                        linewidths=0.6,
                        transform=ccrs.PlateCarree())
                    last_cf = cf

                # Column headers (top row only)
                if row == 0:
                    lag_lbl = ("simultaneous"
                               if lag == 0
                               else f"siconc leads by {lag} month"
                               + ("s" if lag > 1 else ""))
                    ax.set_title(
                        f"Lag {lag:+d}\n({lag_lbl})",
                        fontsize=11, fontweight="bold", pad=4)

                # Row label (leftmost column only)
                if col == 0:
                    ax.text(
                        -0.06, 0.5, row_lbl,
                        transform=ax.transAxes,
                        fontsize=9, fontweight="bold",
                        ha="right", va="center", rotation=90)

                # Panel letter
                letter = chr(ord("a") + row * n_lags + col)
                ax.text(
                    0.02, 0.97, f"({letter})",
                    transform=ax.transAxes, fontsize=10,
                    fontweight="bold", va="top",
                    bbox=dict(facecolor="white", alpha=0.75,
                              edgecolor="none"))

        # Shared colourbar
        if last_cf is not None:
            cax = fig.add_axes([0.90, 0.12, 0.015, 0.74])
            cb  = fig.colorbar(last_cf, cax=cax, extend="both")
            cb.set_label("Pearson r", fontsize=12)
            cb.ax.tick_params(labelsize=11)
            cb.set_ticks([-0.4, -0.2, 0, 0.2, 0.4])

        # No suptitle — figure is self-explanatory from panel labels

        fname = f"figS_seaice_chl_corr_{region_name}.png"
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"  wrote {fname}")




def make_figS_bloom_phenology(
    obs_chl_ts:      xr.DataArray,
    obs_siconc_ts:   xr.DataArray | None,
    model_chl_ts:    dict[str, xr.DataArray],
    model_siconc_ts: dict[str, xr.DataArray],
    common_grid:     xr.DataArray,
    period:          slice = LAG_PERIOD,
) -> None:
    """Bloom phenology and sea-ice retreat analysis for polar regions.

    For each polar region (Arctic, Southern Ocean) produces one figure
    with layout:
        Rows    = metric (3 rows):
          Row 1 — Mean bloom peak DOY
          Row 2 — Std of bloom peak DOY (interannual variability)
          Row 3 — OLS slope: bloom DOY anomaly ~ sea-ice retreat DOY anomaly
                  (days of bloom delay per day of late ice retreat)
        Columns = sources: OBS first, then models in MODELS order

    Cells with significant regression (p < 0.05) are stippled in row 3.

    Output files:
        figS_bloom_phenology_Arctic.png
        figS_bloom_phenology_Southern.png
    """
    logger.info("=== Fig S_bloom_phenology: bloom timing + sea-ice retreat ===")

    if obs_siconc_ts is None and not model_siconc_ts:
        logger.warning(
            "  No siconc data; regression row will be empty")

    regions = {
        "Arctic": {
            "lat_min":  50, "lat_max":  90,
            "proj":     ccrs.NorthPolarStereo(),
            "extent":   [-180, 180, 48, 90],
        },
        "Southern": {
            "lat_min": -78, "lat_max": -45,
            "proj":    ccrs.SouthPolarStereo(),
            "extent":  [-180, 180, -78, -43],
        },
    }

    lats = common_grid.lat.values
    lons = common_grid.lon.values
    lat2d, _ = np.meshgrid(lats, lons, indexing="ij")

    # Sources: OBS first, then models that have both Chl and siconc
    sources_chl: list[tuple[str, xr.DataArray]] = [("OBS", obs_chl_ts)]
    sources_sic: dict[str, xr.DataArray | None] = {"OBS": obs_siconc_ts}
    models_avail = [m for m in MODELS if m in model_chl_ts]
    for model in models_avail:
        sources_chl.append((model, model_chl_ts[model]))
        sources_sic[model] = model_siconc_ts.get(model)

    n_src  = len(sources_chl)
    n_rows = 3

    # Row metadata
    row_meta = [
        {"title": "Mean bloom peak DOY",
         "cmap":  "twilight_shifted",
         "vmin":  1, "vmax": 365,
         "label": "Day of year"},
        {"title": "Interannual std of bloom peak DOY",
         "cmap":  "YlOrRd",
         "vmin":  0, "vmax": 90,
         "label": "Std (days)"},
        {"title": "Slope: bloom DOY ~ sea-ice retreat DOY\n"
                  "(days bloom delay per day late retreat)",
         "cmap":  "RdBu_r",
         "vmin": -2, "vmax": 2,
         "label": "Slope (days/day)"},
    ]

    for region_name, reg in regions.items():
        logger.info(f"  Region: {region_name}...")
        lat_min, lat_max = reg["lat_min"], reg["lat_max"]
        proj   = reg["proj"]
        extent = reg["extent"]
        reg_mask = (lat2d >= lat_min) & (lat2d <= lat_max)

        # Pre-compute all maps for this region
        mean_doy_maps:    dict[str, xr.DataArray] = {}
        std_doy_maps:     dict[str, xr.DataArray] = {}
        slope_maps:       dict[str, xr.DataArray] = {}
        r2_maps:          dict[str, xr.DataArray] = {}
        pval_maps:        dict[str, xr.DataArray] = {}

        for src_label, chl_ts in sources_chl:
            logger.info(f"    [{src_label}] computing bloom DOY...")
            try:
                md, sd = _bloom_doy_field(chl_ts, common_grid, period)
                # Mask to region
                mean_doy_maps[src_label] = md.where(
                    xr.DataArray(reg_mask,
                                 dims=("lat","lon"),
                                 coords={"lat": md.lat,
                                         "lon": md.lon}))
                std_doy_maps[src_label]  = sd.where(
                    xr.DataArray(reg_mask,
                                 dims=("lat","lon"),
                                 coords={"lat": sd.lat,
                                         "lon": sd.lon}))
            except Exception as e:
                logger.warning(
                    f"    [{src_label}] bloom DOY failed: {e}")

            sic_ts = sources_sic.get(src_label)
            if sic_ts is not None:
                logger.info(
                    f"    [{src_label}] computing retreat regression...")
                try:
                    sl, r2, pv = _bloom_retreat_regression(
                        chl_ts, sic_ts, common_grid, period)
                    slope_maps[src_label] = sl.where(
                        xr.DataArray(reg_mask,
                                     dims=("lat","lon"),
                                     coords={"lat": sl.lat,
                                             "lon": sl.lon}))
                    r2_maps[src_label]    = r2.where(
                        xr.DataArray(reg_mask,
                                     dims=("lat","lon"),
                                     coords={"lat": r2.lat,
                                             "lon": r2.lon}))
                    pval_maps[src_label]  = pv.where(
                        xr.DataArray(reg_mask,
                                     dims=("lat","lon"),
                                     coords={"lat": pv.lat,
                                             "lon": pv.lon}))
                    logger.success(
                        f"    [{src_label}] done")
                except Exception as e:
                    logger.warning(
                        f"    [{src_label}] regression failed: {e}")

        # ── Draw figure ──────────────────────────────────────────────
        fig, axes = plt.subplots(
            n_rows, n_src,
            figsize=(3.8 * n_src, 4.5 * n_rows),
            subplot_kw={"projection": proj},
            squeeze=False,
        )
        fig.subplots_adjust(
            left=0.04, right=0.88,
            bottom=0.04, top=0.92,
            hspace=0.18, wspace=0.06,
        )

        # Store last mappable per row for shared colourbar
        last_im = [None, None, None]

        for col, (src_label, _) in enumerate(sources_chl):
            maps_by_row = [
                mean_doy_maps.get(src_label),
                std_doy_maps.get(src_label),
                slope_maps.get(src_label),
            ]
            pval_r3 = pval_maps.get(src_label)

            for row, (data, meta) in enumerate(
                zip(maps_by_row, row_meta)
            ):
                ax = axes[row, col]

                if data is None:
                    ax.text(0.5, 0.5, "no data",
                            transform=ax.transAxes,
                            ha="center", va="center",
                            fontsize=8, color="grey")
                    ax.set_extent(extent, crs=ccrs.PlateCarree())
                    ax.add_feature(
                        cfeature.LAND,
                        facecolor="lightgray", zorder=4)
                    try:
                        ax.coastlines(linewidth=0.35, zorder=5)
                    except Exception:
                        pass
                    if row == 0:
                        ax.set_title(src_label, fontsize=9,
                                     fontweight="bold")
                    continue

                vals = data.values
                levels = np.linspace(meta["vmin"], meta["vmax"], 13)
                cf = ax.contourf(
                    lons, lats, vals,
                    levels=levels,
                    cmap=meta["cmap"],
                    extend="both",
                    transform=ccrs.PlateCarree(),
                )
                last_im[row] = cf

                # Significance stippling for regression row
                if row == 2 and pval_r3 is not None:
                    pv = pval_r3.values
                    sig = np.isfinite(pv) & (pv < 0.05)
                    lon2d_p, lat2d_p = np.meshgrid(lons, lats)
                    ax.plot(
                        lon2d_p[sig], lat2d_p[sig],
                        "k.", ms=0.6, alpha=0.45,
                        transform=ccrs.PlateCarree(),
                        rasterized=True,
                    )

                ax.set_extent(extent, crs=ccrs.PlateCarree())
                ax.add_feature(
                    cfeature.LAND,
                    facecolor="lightgray", zorder=4)
                try:
                    ax.coastlines(linewidth=0.35, zorder=5)
                except Exception:
                    pass

                # Column title (source name, top row only)
                if row == 0:
                    ax.set_title(src_label, fontsize=9,
                                 fontweight="bold", pad=4)

                # Row label (leftmost column only)
                if col == 0:
                    ax.text(
                        -0.06, 0.5,
                        meta["title"],
                        transform=ax.transAxes,
                        fontsize=8, fontweight="bold",
                        ha="right", va="center",
                        rotation=90,
                    )

                # Panel letter
                # Simple sequential letter per column (a=OBS, b=first model...)
                letter = chr(ord("a") + col)
                ax.text(
                    0.02, 0.97, f"({letter})",
                    transform=ax.transAxes, fontsize=8,
                    fontweight="bold", va="top",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", alpha=0.75,
                              edgecolor="none"),
                )

        # Shared colourbars (one per row, right side)
        cbar_x     = 0.895
        cbar_width = 0.012
        row_tops   = [0.92 - r * (0.88 / n_rows) for r in range(n_rows)]
        row_height = 0.88 / n_rows

        for row, (im, meta) in enumerate(zip(last_im, row_meta)):
            if im is None:
                continue
            cax = fig.add_axes([
                cbar_x,
                row_tops[row] - row_height + 0.03,
                cbar_width,
                row_height - 0.06,
            ])
            cb = fig.colorbar(im, cax=cax, extend="both")
            cb.set_label(meta["label"], fontsize=8)
            cb.ax.tick_params(labelsize=7)

        n_models_with_sic = sum(
            1 for m in models_avail if m in model_siconc_ts)
        fig.suptitle(
            f"Bloom phenology and sea-ice retreat timing — "
            f"{region_name}\n"
            f"Row 1: Mean bloom peak day-of-year (Chl-$a$ annual "
            f"maximum, 1998–2014).  "
            f"Row 2: Interannual std of bloom peak DOY.\n"
            f"Row 3: OLS slope of bloom peak DOY anomaly regressed "
            f"onto sea-ice retreat DOY anomaly (day when siconc < "
            f"15\\%).  Stippling: $p < 0.05$.  "
            f"{n_models_with_sic} of {len(MODELS)} models have "
            f"siconc available.",
            fontsize=8.5, y=0.975,
        )

        fname = f"figS_bloom_phenology_{region_name}.png"
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"  wrote {fname}")



# ============================================================
# EOF helper: compute EOF/PC/variance spectrum
# ============================================================

def _compute_eof(
    field_z: xr.DataArray,
    region_mask: np.ndarray,
    n_eof: int = 5,
) -> dict:
    """Compute EOFs of a (time, lat, lon) standardised anomaly field.

    Parameters
    ----------
    field_z    : (time, lat, lon) — deseasonalised, detrended,
                 standardised per cell (unit variance).
    region_mask: (lat, lon) bool — True = include in EOF.
    n_eof      : number of EOFs to retain.

    Returns dict with keys:
      eofs      : (n_eof, nlat, nlon) — spatial patterns (NaN outside mask)
      pcs       : (n_eof, T)          — principal component time series
      var_frac  : (n_eof,)            — fraction of variance explained
      lats, lons: coordinate arrays
    """
    from numpy.linalg import svd as _svd

    arr  = np.asarray(field_z.values, dtype=float)  # (T, nlat, nlon)
    T, nlat, nlon = arr.shape
    lats = field_z.lat.values
    lons = field_z.lon.values

    # cos(lat) weights for area-correct EOF
    w2d   = np.cos(np.deg2rad(lats))[:, np.newaxis] * np.ones((1, nlon))
    # Valid: inside region mask AND at least T/3 finite time steps
    # (not all-finite, to allow seasonal ice-zone cells with summer NaN)
    n_finite = np.sum(np.isfinite(arr), axis=0)
    valid = region_mask & (n_finite >= max(10, T // 3))

    if valid.sum() < 10:
        return {}

    w_flat = w2d.ravel()[valid.ravel()]
    X_flat = arr.reshape(T, -1)[:, valid.ravel()]   # (T, n_valid)

    # Fill remaining NaN with 0 before SVD (zero-mean anomalies,
    # so NaN → 0 is equivalent to excluding those time steps from
    # the covariance contribution without distorting the pattern)
    X_flat = np.where(np.isfinite(X_flat), X_flat, 0.0)

    # Weight the data by sqrt(cos(lat))
    Xw = X_flat * np.sqrt(w_flat)[np.newaxis, :]    # (T, n_valid)

    # Thin SVD
    k    = min(n_eof, T - 1, Xw.shape[1] - 1)
    U, S, Vt = _svd(Xw, full_matrices=False)

    var_frac = S[:k] ** 2 / np.sum(S ** 2)

    # Reconstruct full spatial patterns
    eofs_out = np.full((k, nlat, nlon), np.nan)
    pcs_out  = np.full((k, T),          np.nan)
    for i in range(k):
        pattern = np.full(nlat * nlon, np.nan)
        pattern[valid.ravel()] = Vt[i] / np.sqrt(w_flat)
        eofs_out[i] = pattern.reshape(nlat, nlon)
        pcs_out[i]  = U[:, i] * S[i]

    return {
        "eofs":     eofs_out,
        "pcs":      pcs_out,
        "var_frac": var_frac[:k],
        "lats":     lats,
        "lons":     lons,
        "S":        S,
    }


# ============================================================
# Figure A: Sea-ice EOF analysis
# ============================================================

def make_figS_seaice_eof(
    obs_siconc_ts:    xr.DataArray | None,
    model_siconc_ts:  dict[str, xr.DataArray],
    common_grid:      xr.DataArray,
    period:           slice = LAG_PERIOD,
    n_eof:            int   = 5,
) -> None:
    """Fig S_seaice_eof: EOF analysis of sea-ice concentration anomalies.

    Two polar regions (Arctic: 60-90N, Southern Ocean SIZ: 50-75S),
    computed separately. For each region:
      Col 1 — EOF1 spatial pattern (contourf, polar stereographic)
      Col 2 — PC1 time series (OBS black, model ensemble grey)
      Col 3 — Variance spectrum / scree plot (% variance per EOF,
               OBS bars, model ensemble mean line + spread)

    Two rows (Arctic top, Southern Ocean bottom).
    OBS and model ensemble overlaid on PC1 and scree panels.

    Output: figS_seaice_eof.png
    """
    logger.info("=== Fig S_seaice_eof: sea-ice EOF analysis ===")

    if obs_siconc_ts is None and not model_siconc_ts:
        logger.warning("  No siconc data; skipping figS_seaice_eof")
        return

    regions = {
        "Arctic":   {"lat_min":  60, "lat_max":  90,
                     "proj": ccrs.NorthPolarStereo(),
                     "extent": [-180, 180, 55, 90]},
        "Southern": {"lat_min": -75, "lat_max": -50,
                     "proj": ccrs.SouthPolarStereo(),
                     "extent": [-180, 180, -75, -45]},
    }

    fig = plt.figure(figsize=(16, 10))
    import matplotlib.gridspec as _gs
    outer = _gs.GridSpec(2, 3, figure=fig,
                         hspace=0.38, wspace=0.28,
                         left=0.05, right=0.97,
                         bottom=0.08, top=0.90)

    models_list = list(model_siconc_ts.keys())

    for r_idx, (region_name, reg) in enumerate(regions.items()):
        lat_min, lat_max = reg["lat_min"], reg["lat_max"]

        # Region mask on common grid
        lats = common_grid.lat.values
        lons = common_grid.lon.values
        lat2d, _ = np.meshgrid(lats, lons, indexing="ij")
        region_mask = (lat2d >= lat_min) & (lat2d <= lat_max)

        # Compute OBS EOF
        obs_eof = {}
        if obs_siconc_ts is not None:
            try:
                sic_re = regrid_timeseries(
                    obs_siconc_ts.sel(time=period), common_grid)
                sic_z  = _deseas_detrend_zscore(sic_re)
                obs_eof = _compute_eof(sic_z, region_mask, n_eof)
                logger.success(
                    f"  OBS {region_name} siconc EOF: "
                    f"var1={obs_eof.get('var_frac', [np.nan])[0]*100:.1f}%"
                )
            except Exception as e:
                logger.warning(
                    f"  OBS {region_name} siconc EOF failed: {e}")

        # Compute model EOFs
        model_eofs = []
        for model in models_list:
            try:
                sic_re = regrid_timeseries(
                    model_siconc_ts[model].sel(time=period), common_grid)
                sic_z  = _deseas_detrend_zscore(sic_re)
                eof_m  = _compute_eof(sic_z, region_mask, n_eof)
                if eof_m:
                    model_eofs.append(eof_m)
            except Exception as e:
                logger.warning(
                    f"  [{model}] {region_name} siconc EOF failed: {e}")

        # ── Col 0: EOF1 spatial pattern (OBS) ───────────────────────
        ax_map = fig.add_subplot(outer[r_idx, 0],
                                 projection=reg["proj"])
        if obs_eof and "eofs" in obs_eof:
            eof1 = obs_eof["eofs"][0]
            vmax = np.nanpercentile(np.abs(eof1), 95)
            cf = ax_map.contourf(
                lons, lats, eof1,
                levels=np.linspace(-vmax, vmax, 13),
                cmap="RdBu_r", extend="both",
                transform=ccrs.PlateCarree(),
            )
            plt.colorbar(cf, ax=ax_map, shrink=0.7,
                         label="EOF1 loading")
        ax_map.set_extent(reg["extent"], crs=ccrs.PlateCarree())
        ax_map.add_feature(cfeature.LAND, facecolor="lightgray",
                           zorder=4)
        try:
            ax_map.coastlines(linewidth=0.4, zorder=5)
        except Exception:
            pass
        v1 = obs_eof.get("var_frac", [np.nan])[0] * 100 \
            if obs_eof else np.nan
        ax_map.set_title(
            f"{region_name} — OBS EOF1\n"
            f"({v1:.1f}% variance)",
            fontsize=9,
        )
        letter = chr(ord("a") + r_idx * 3)
        ax_map.text(0.02, 0.97, f"({letter})",
                    transform=ax_map.transAxes, fontsize=9,
                    fontweight="bold", va="top",
                    bbox=dict(facecolor="white", alpha=0.7,
                              edgecolor="none"))

        # ── Col 1: PC1 time series ───────────────────────────────────
        ax_pc = fig.add_subplot(outer[r_idx, 1])
        import pandas as pd

        # Model PCs (grey)
        for eof_m in model_eofs:
            if "pcs" not in eof_m:
                continue
            pc = eof_m["pcs"][0]
            T  = len(pc)
            ax_pc.plot(np.arange(T), pc, color="grey",
                       lw=0.7, alpha=0.4)

        # OBS PC (black)
        if obs_eof and "pcs" in obs_eof:
            pc_obs = obs_eof["pcs"][0]
            T      = len(pc_obs)
            ax_pc.plot(np.arange(T), pc_obs, color="black",
                       lw=1.8, label="OBS")
            ax_pc.axhline(0, color="k", lw=0.5, alpha=0.5)

        ax_pc.set_xlabel("Month index (1998–2014)", fontsize=8)
        ax_pc.set_ylabel("PC1 (standardised)", fontsize=8)
        ax_pc.set_title(f"{region_name} — PC1 time series",
                        fontsize=9)
        ax_pc.grid(alpha=0.2)
        ax_pc.legend(fontsize=8)
        letter = chr(ord("a") + r_idx * 3 + 1)
        ax_pc.text(0.02, 0.97, f"({letter})",
                   transform=ax_pc.transAxes, fontsize=9,
                   fontweight="bold", va="top")

        # ── Col 2: Variance spectrum (scree plot) ────────────────────
        ax_sc = fig.add_subplot(outer[r_idx, 2])
        x_pos = np.arange(1, n_eof + 1)

        # Model spread
        if model_eofs:
            m_varfrac = np.array([e["var_frac"] * 100
                                  for e in model_eofs
                                  if "var_frac" in e])
            if m_varfrac.ndim == 2 and m_varfrac.shape[1] >= n_eof:
                e_mn  = m_varfrac.mean(axis=0)[:n_eof]
                e_min = m_varfrac.min(axis=0)[:n_eof]
                e_max = m_varfrac.max(axis=0)[:n_eof]
                ax_sc.fill_between(x_pos, e_min, e_max,
                                   color="steelblue", alpha=0.25,
                                   label="Model spread")
                ax_sc.plot(x_pos, e_mn, color="steelblue",
                           lw=1.5, ls="--", label="Ensemble mean")

        # OBS bars
        if obs_eof and "var_frac" in obs_eof:
            vf = obs_eof["var_frac"] * 100
            ax_sc.bar(x_pos[:len(vf)], vf[:n_eof],
                      color="black", alpha=0.7, width=0.5,
                      label="OBS")

        ax_sc.set_xlabel("EOF number", fontsize=8)
        ax_sc.set_ylabel("Variance explained (%)", fontsize=8)
        ax_sc.set_title(f"{region_name} — Variance spectrum",
                        fontsize=9)
        ax_sc.set_xticks(x_pos)
        ax_sc.legend(fontsize=7)
        ax_sc.grid(alpha=0.2, axis="y")
        letter = chr(ord("a") + r_idx * 3 + 2)
        ax_sc.text(0.02, 0.97, f"({letter})",
                   transform=ax_sc.transAxes, fontsize=9,
                   fontweight="bold", va="top")

    fig.suptitle(
        "Sea-ice concentration EOF analysis — Arctic (top) and "
        "Southern Ocean SIZ (bottom)\n"
        "Input: deseasonalised, linearly detrended, standardised "
        "per cell (unit variance).  "
        "Area weights: cos(lat).  Period: 1998–2014.\n"
        "Col 1: OBS EOF1 spatial pattern.  "
        "Col 2: PC1 time series (black: OBS, grey: models).  "
        "Col 3: Variance spectrum (bars: OBS, line+shading: ensemble).",
        fontsize=8.5, y=0.97,
    )
    fig.savefig(OUT_DIR / "figS_seaice_eof.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote figS_seaice_eof.png")


# ============================================================
# Figure B: Chl-a EOF analysis (same layout as sea-ice)
# ============================================================

def make_figS_chl_eof(
    obs_chl_ts:   xr.DataArray,
    model_chl_ts: dict[str, xr.DataArray],
    common_grid:  xr.DataArray,
    period:       slice = LAG_PERIOD,
    n_eof:        int   = 5,
) -> None:
    """Fig S_chl_eof: EOF analysis of Chl-a anomalies in polar regions.

    Same layout as figS_seaice_eof: Arctic (top), Southern Ocean (bottom),
    three columns (EOF1 map | PC1 time series | variance spectrum).

    Output: figS_chl_eof.png
    """
    logger.info("=== Fig S_chl_eof: Chl-a EOF analysis ===")

    regions = {
        "Arctic":   {"lat_min":  60, "lat_max":  90,
                     "proj": ccrs.NorthPolarStereo(),
                     "extent": [-180, 180, 55, 90]},
        "Southern": {"lat_min": -75, "lat_max": -50,
                     "proj": ccrs.SouthPolarStereo(),
                     "extent": [-180, 180, -75, -45]},
    }

    fig = plt.figure(figsize=(16, 10))
    import matplotlib.gridspec as _gs
    outer = _gs.GridSpec(2, 3, figure=fig,
                         hspace=0.38, wspace=0.28,
                         left=0.05, right=0.97,
                         bottom=0.08, top=0.90)

    models_list = [m for m in MODELS if m in model_chl_ts]
    lats = common_grid.lat.values
    lons = common_grid.lon.values
    lat2d, _ = np.meshgrid(lats, lons, indexing="ij")

    for r_idx, (region_name, reg) in enumerate(regions.items()):
        lat_min, lat_max = reg["lat_min"], reg["lat_max"]
        region_mask = (lat2d >= lat_min) & (lat2d <= lat_max)

        # OBS EOF
        obs_eof = {}
        try:
            chl_re = regrid_timeseries(
                obs_chl_ts.sel(time=period), common_grid)
            chl_z  = _deseas_detrend_zscore(chl_re)
            obs_eof = _compute_eof(chl_z, region_mask, n_eof)
            logger.success(
                f"  OBS {region_name} Chl EOF: "
                f"var1={obs_eof.get('var_frac',[np.nan])[0]*100:.1f}%"
            )
        except Exception as e:
            logger.warning(
                f"  OBS {region_name} Chl EOF failed: {e}")

        # Model EOFs
        model_eofs = []
        for model in models_list:
            try:
                chl_re = regrid_timeseries(
                    model_chl_ts[model].sel(time=period), common_grid)
                chl_z  = _deseas_detrend_zscore(chl_re)
                eof_m  = _compute_eof(chl_z, region_mask, n_eof)
                if eof_m:
                    model_eofs.append(eof_m)
            except Exception as e:
                logger.warning(
                    f"  [{model}] {region_name} Chl EOF failed: {e}")

        # Col 0: EOF1 map
        ax_map = fig.add_subplot(outer[r_idx, 0],
                                 projection=reg["proj"])
        if obs_eof and "eofs" in obs_eof:
            eof1 = obs_eof["eofs"][0]
            vmax = np.nanpercentile(np.abs(eof1), 95)
            cf = ax_map.contourf(
                lons, lats, eof1,
                levels=np.linspace(-vmax, vmax, 13),
                cmap="RdBu_r", extend="both",
                transform=ccrs.PlateCarree(),
            )
            plt.colorbar(cf, ax=ax_map, shrink=0.7,
                         label="EOF1 loading")
        ax_map.set_extent(reg["extent"], crs=ccrs.PlateCarree())
        ax_map.add_feature(cfeature.LAND, facecolor="lightgray",
                           zorder=4)
        try:
            ax_map.coastlines(linewidth=0.4, zorder=5)
        except Exception:
            pass
        v1 = obs_eof.get("var_frac", [np.nan])[0] * 100 \
            if obs_eof else np.nan
        ax_map.set_title(
            f"{region_name} — OBS Chl-$a$ EOF1\n({v1:.1f}% variance)",
            fontsize=9)
        letter = chr(ord("a") + r_idx * 3)
        ax_map.text(0.02, 0.97, f"({letter})",
                    transform=ax_map.transAxes, fontsize=9,
                    fontweight="bold", va="top",
                    bbox=dict(facecolor="white", alpha=0.7,
                              edgecolor="none"))

        # Col 1: PC1 time series
        ax_pc = fig.add_subplot(outer[r_idx, 1])
        for eof_m in model_eofs:
            if "pcs" not in eof_m:
                continue
            ax_pc.plot(np.arange(len(eof_m["pcs"][0])),
                       eof_m["pcs"][0],
                       color="grey", lw=0.7, alpha=0.4)
        if obs_eof and "pcs" in obs_eof:
            pc_obs = obs_eof["pcs"][0]
            ax_pc.plot(np.arange(len(pc_obs)), pc_obs,
                       color="black", lw=1.8, label="OBS")
            ax_pc.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax_pc.set_xlabel("Month index (1998–2014)", fontsize=8)
        ax_pc.set_ylabel("PC1 (standardised)", fontsize=8)
        ax_pc.set_title(f"{region_name} — Chl-$a$ PC1", fontsize=9)
        ax_pc.grid(alpha=0.2)
        ax_pc.legend(fontsize=8)
        letter = chr(ord("a") + r_idx * 3 + 1)
        ax_pc.text(0.02, 0.97, f"({letter})",
                   transform=ax_pc.transAxes, fontsize=9,
                   fontweight="bold", va="top")

        # Col 2: Variance spectrum
        ax_sc = fig.add_subplot(outer[r_idx, 2])
        x_pos = np.arange(1, n_eof + 1)
        if model_eofs:
            m_varfrac = np.array([e["var_frac"] * 100
                                  for e in model_eofs
                                  if "var_frac" in e])
            if m_varfrac.ndim == 2 and m_varfrac.shape[1] >= n_eof:
                e_mn  = m_varfrac.mean(axis=0)[:n_eof]
                e_min = m_varfrac.min(axis=0)[:n_eof]
                e_max = m_varfrac.max(axis=0)[:n_eof]
                ax_sc.fill_between(x_pos, e_min, e_max,
                                   color="steelblue", alpha=0.25,
                                   label="Model spread")
                ax_sc.plot(x_pos, e_mn, color="steelblue",
                           lw=1.5, ls="--", label="Ensemble mean")
        if obs_eof and "var_frac" in obs_eof:
            vf = obs_eof["var_frac"] * 100
            ax_sc.bar(x_pos[:len(vf)], vf[:n_eof],
                      color="black", alpha=0.7, width=0.5,
                      label="OBS")
        ax_sc.set_xlabel("EOF number", fontsize=8)
        ax_sc.set_ylabel("Variance explained (%)", fontsize=8)
        ax_sc.set_title(f"{region_name} — Variance spectrum",
                        fontsize=9)
        ax_sc.set_xticks(x_pos)
        ax_sc.legend(fontsize=7)
        ax_sc.grid(alpha=0.2, axis="y")
        letter = chr(ord("a") + r_idx * 3 + 2)
        ax_sc.text(0.02, 0.97, f"({letter})",
                   transform=ax_sc.transAxes, fontsize=9,
                   fontweight="bold", va="top")

    fig.suptitle(
        "Chl-$a$ anomaly EOF analysis — Arctic (top) and "
        "Southern Ocean SIZ (bottom)\n"
        "Input: deseasonalised, linearly detrended, standardised "
        "per cell.  Area weights: cos(lat).  Period: 1998–2014.\n"
        "Col 1: OBS EOF1 spatial pattern.  "
        "Col 2: PC1 time series.  "
        "Col 3: Variance spectrum.",
        fontsize=8.5, y=0.97,
    )
    fig.savefig(OUT_DIR / "figS_chl_eof.png",
                dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote figS_chl_eof.png")


# ============================================================
# Figure C: Explained variance map
# (sea-ice PC1 regressed onto Chl-a anomaly field)
# ============================================================

def make_figS_seaice_chl_regression(
    obs_chl_ts:      xr.DataArray,
    obs_siconc_ts:   xr.DataArray | None,
    model_chl_ts:    dict[str, xr.DataArray],
    model_siconc_ts: dict[str, xr.DataArray],
    common_grid:     xr.DataArray,
    period:          slice = LAG_PERIOD,
) -> None:
    """Fig S_seaice_chl_regression: Chl-a variance explained by sea-ice PC1.

    For each polar region (Arctic, Southern Ocean SIZ):
      1. Compute sea-ice EOF1 -> PC1 time series
      2. At every Chl-a grid cell, regress PC1 onto the local
         Chl-a anomaly time series -> R^2 (fraction of variance
         explained) and regression coefficient beta
      3. Plot R^2 map and beta map for OBS and ensemble mean

    Layout: 2 rows (Arctic, Southern Ocean) x 4 cols
      (a) OBS R^2   (b) OBS beta   (c) Ens-mean R^2   (d) Ens-mean beta

    Stippling marks cells where p < 0.05 (F-test for R^2).

    Output: figS_seaice_chl_regression.png
    """
    from scipy import stats as _stats

    logger.info(
        "=== Fig S_seaice_chl_regression: Chl-a variance "
        "explained by sea-ice PC1 ==="
    )

    if obs_siconc_ts is None and not model_siconc_ts:
        logger.warning("  No siconc; skipping figS_seaice_chl_regression")
        return

    regions = {
        "Arctic": {
            "sic_lat": (50,  90), "chl_lat": (50,  90),
            "proj":    ccrs.NorthPolarStereo(),
            "extent":  [-180, 180, 48, 90],
        },
        "Southern": {
            "sic_lat": (-78, -45), "chl_lat": (-78, -45),
            "proj":    ccrs.SouthPolarStereo(),
            "extent":  [-180, 180, -78, -43],
        },
    }

    lats = common_grid.lat.values
    lons = common_grid.lon.values
    lat2d, _ = np.meshgrid(lats, lons, indexing="ij")
    models_list = [m for m in MODELS
                   if m in model_chl_ts and m in model_siconc_ts]

    def _regress_pcs_onto_chl(
        pcs:      np.ndarray,   # (n_pcs, T) — multiple PCs
        chl_z:    xr.DataArray, # (T, nlat, nlon)
        var_frac: np.ndarray,   # (n_pcs,) — fraction explained per PC
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Multiple regression of leading sea-ice PCs onto Chl-a.

        Uses enough PCs to explain >= 50% of sea-ice variance.
        Reports total R^2 from the multiple regression, the beta
        coefficient of PC1 (dominant mode), and the F-test p-value
        for the full model (k predictors).

        Returns R^2 map, beta_PC1 map, p-value map — all (nlat, nlon).
        """
        # Select PCs until cumulative variance >= 50%
        cum_var = np.cumsum(var_frac)
        n_pcs   = int(np.searchsorted(cum_var, 0.50)) + 1
        n_pcs   = min(n_pcs, pcs.shape[0])
        logger.info(
            f"    Multiple regression: using {n_pcs} PCs "
            f"(cumulative variance = {cum_var[n_pcs-1]*100:.1f}%)"
        )

        C    = np.asarray(chl_z.values, dtype=float)  # (T, nlat, nlon)
        T    = pcs.shape[1]
        nlat, nlon = C.shape[1], C.shape[2]
        r2   = np.full((nlat, nlon), np.nan)
        bet  = np.full((nlat, nlon), np.nan)
        pv   = np.full((nlat, nlon), np.nan)

        X_full = pcs[:n_pcs].T  # (T, n_pcs)
        x_ok   = np.all(np.isfinite(X_full), axis=1)  # (T,)
        X_v    = X_full[x_ok]        # (T_v, n_pcs)
        C_v    = C[x_ok]             # (T_v, nlat, nlon)
        T_v    = X_v.shape[0]

        # Reshape for vectorised OLS
        C_flat  = C_v.reshape(T_v, -1)                # (T_v, ncells)
        n_fin   = np.sum(np.isfinite(C_flat), axis=0) # (ncells,)
        valid   = n_fin >= n_pcs + 2

        # NaN → 0 for OLS (zero-mean anomaly; NaN months contribute 0)
        C_nz = np.where(np.isfinite(C_flat), C_flat, 0.0)
        X_c  = X_v - X_v.mean(axis=0, keepdims=True)  # (T_v, n_pcs)
        XtX  = X_c.T @ X_c                            # (n_pcs, n_pcs)
        XtC  = X_c.T @ C_nz                           # (n_pcs, ncells)
        try:
            beta_flat = np.linalg.solve(XtX, XtC)     # (n_pcs, ncells)
        except np.linalg.LinAlgError:
            beta_flat = np.linalg.lstsq(XtX, XtC, rcond=None)[0]

        C_hat  = X_c @ beta_flat                       # (T_v, ncells)
        C_mean = np.nanmean(C_flat, axis=0, keepdims=True)
        C_dm   = C_nz - C_mean
        ss_res = np.sum((C_dm - C_hat)**2, axis=0)
        ss_tot = np.sum(C_dm**2,            axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            r2_flat = np.where(ss_tot > 0, 1.0 - ss_res/ss_tot, np.nan)
        r2_flat = np.clip(r2_flat, 0.0, 1.0)

        k = n_pcs
        with np.errstate(divide="ignore", invalid="ignore"):
            F_stat = ((r2_flat / k) /
                      ((1.0 - r2_flat) / np.maximum(n_fin - k - 1, 1)))
        pv_flat = np.where(
            valid & np.isfinite(F_stat) & (F_stat > 0),
            1.0 - _stats.f.cdf(
                np.where(F_stat > 0, F_stat, 0.0),
                k, np.maximum(n_fin - k - 1, 1)),
            np.nan,
        )

        r2[...]  = np.where(valid, r2_flat,      np.nan).reshape(nlat, nlon)
        pv[...]  = np.where(valid, pv_flat,      np.nan).reshape(nlat, nlon)
        bet[...] = np.where(valid, beta_flat[0], np.nan).reshape(nlat, nlon)
        return r2, bet, pv

    # Only OBS + ensemble mean (2 cols per metric = 4 cols total)
    ncols = 4
    region_list = list(regions.keys())
    nrows = len(region_list)

    import matplotlib.gridspec as _gs2
    fig = plt.figure(figsize=(5.5 * ncols, 5.5 * nrows))
    gs  = _gs2.GridSpec(nrows, ncols, figure=fig,
                        hspace=0.25, wspace=0.08,
                        left=0.04, right=0.96,
                        bottom=0.06, top=0.90)

    # Create axes with correct projection per row
    axes = np.empty((nrows, ncols), dtype=object)
    for r_i, rname in enumerate(region_list):
        proj_r = regions[rname]["proj"]
        for c_i in range(ncols):
            axes[r_i, c_i] = fig.add_subplot(gs[r_i, c_i],
                                              projection=proj_r)

    col_titles = [
        "OBS — $R^2$ (sea-ice PC1 → Chl-a)",
        "OBS — $\\beta$ (regression coeff.)",
        "Ensemble mean — $R^2$",
        "Ensemble mean — $\\beta$",
    ]
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=9, fontweight="bold", pad=4)

    last_r2 = last_beta = None

    for r_idx, (region_name, reg) in enumerate(regions.items()):
        sic_lat  = reg["sic_lat"]
        chl_lat  = reg["chl_lat"]
        proj     = reg["proj"]
        extent   = reg["extent"]

        sic_mask = ((lat2d >= sic_lat[0]) & (lat2d <= sic_lat[1]))
        chl_mask = ((lat2d >= chl_lat[0]) & (lat2d <= chl_lat[1]))

        # ---- OBS ----
        obs_r2 = obs_beta = obs_pv = None
        if obs_siconc_ts is not None:
            try:
                sic_re = regrid_timeseries(
                    obs_siconc_ts.sel(time=period), common_grid)
                sic_z  = _deseas_detrend_zscore(sic_re)
                # Compute enough EOFs to explain 50% of variance
                eof_s  = _compute_eof(sic_z, sic_mask, n_eof=10)
                if eof_s and "pcs" in eof_s:
                    pcs_obs      = eof_s["pcs"]        # (n_eof, T)
                    var_frac_obs = eof_s["var_frac"]   # (n_eof,)
                    cum_var = np.cumsum(var_frac_obs)
                    logger.info(
                        f"  OBS {region_name}: EOF ok, "
                        f"n_eof={pcs_obs.shape[0]}, "
                        f"var_frac={var_frac_obs[:5]*100}, "
                        f"cum50%_at_pc={np.searchsorted(cum_var,0.5)+1}"
                    )
                    print(f"[REGRESSION DEBUG] OBS {region_name}: "
                          f"sic_z shape={sic_z.shape}, "
                          f"n_valid_eof={eof_s.get('n_valid','?')}",
                          flush=True)
                    chl_re  = regrid_timeseries(
                        obs_chl_ts.sel(time=period), common_grid)
                    # Interpolate OBS polar night gaps before preprocessing
                    chl_re  = _interp_time_gaps(chl_re, max_gap=3)
                    chl_z   = _deseas_detrend_zscore(chl_re)
                    # Align time
                    ym_p = _ym_key(sic_re)
                    ym_c = _ym_key(chl_re)
                    common_ym = np.intersect1d(ym_p, ym_c)
                    logger.info(
                        f"  OBS {region_name}: "
                        f"common_ym={common_ym.size} months, "
                        f"chl_z shape={chl_z.shape}"
                    )
                    p_idx = np.argsort(ym_p)[
                        np.searchsorted(np.sort(ym_p), common_ym)]
                    c_idx = np.argsort(ym_c)[
                        np.searchsorted(np.sort(ym_c), common_ym)]
                    pcs_aligned  = pcs_obs[:, p_idx]   # (n_eof, T_c)
                    chl_aligned  = chl_z[c_idx, :, :]
                    logger.info(
                        f"  OBS {region_name}: starting regression "
                        f"({chl_aligned.shape[1]}x{chl_aligned.shape[2]} cells)..."
                    )
                    obs_r2, obs_beta, obs_pv = _regress_pcs_onto_chl(
                        pcs_aligned, chl_aligned, var_frac_obs)
                    obs_r2[~chl_mask]   = np.nan
                    obs_beta[~chl_mask] = np.nan
                    obs_pv[~chl_mask]   = np.nan
                    logger.success(
                        f"  OBS {region_name}: median R2="
                        f"{np.nanmedian(obs_r2):.3f}"
                    )
            except Exception as e:
                import traceback as _tb
                logger.warning(
                    f"  OBS {region_name} regression failed: {e}")
                logger.warning(_tb.format_exc().replace("<", "\\<"))

        # ---- Models ----
        model_r2_list   = []
        model_beta_list = []
        for model in models_list:
            try:
                sic_re = regrid_timeseries(
                    model_siconc_ts[model].sel(time=period), common_grid)
                sic_z  = _deseas_detrend_zscore(sic_re)
                eof_s  = _compute_eof(sic_z, sic_mask, n_eof=10)
                if not eof_s or "pcs" not in eof_s:
                    continue
                pcs_m      = eof_s["pcs"]        # (n_eof, T)
                var_frac_m = eof_s["var_frac"]
                chl_re = regrid_timeseries(
                    model_chl_ts[model].sel(time=period), common_grid)
                chl_z  = _deseas_detrend_zscore(chl_re)
                ym_p   = _ym_key(sic_re)
                ym_c   = _ym_key(chl_re)
                common_ym = np.intersect1d(ym_p, ym_c)
                p_idx  = np.argsort(ym_p)[
                    np.searchsorted(np.sort(ym_p), common_ym)]
                c_idx  = np.argsort(ym_c)[
                    np.searchsorted(np.sort(ym_c), common_ym)]
                pcs_aligned  = pcs_m[:, p_idx]
                chl_aligned  = chl_z[c_idx, :, :]
                r2_m, beta_m, _ = _regress_pcs_onto_chl(
                    pcs_aligned, chl_aligned, var_frac_m)
                r2_m[~chl_mask]   = np.nan
                beta_m[~chl_mask] = np.nan
                model_r2_list.append(r2_m)
                model_beta_list.append(beta_m)
            except Exception as e:
                logger.warning(
                    f"  [{model}] {region_name} regression failed: {e}")

        ens_r2   = np.nanmean(np.stack(model_r2_list,   axis=0), axis=0) \
            if model_r2_list   else None
        ens_beta = np.nanmean(np.stack(model_beta_list, axis=0), axis=0) \
            if model_beta_list else None

        # ---- Plot with shared colourbars and correct projections ----
        # Compute robust colour scale from data percentiles
        all_r2   = [x for x in [obs_r2, ens_r2]   if x is not None]
        all_beta = [x for x in [obs_beta, ens_beta] if x is not None]
        r2_vmax = float(np.nanpercentile(
            np.concatenate([a.ravel()[np.isfinite(a.ravel())]
                            for a in all_r2]), 95)) if all_r2 else 0.3
        r2_vmax = min(r2_vmax, 0.4)
        bv = float(np.nanpercentile(
            np.abs(np.concatenate([a.ravel()[np.isfinite(a.ravel())]
                                   for a in all_beta])), 95
        )) if all_beta else 0.3
        bv = min(bv, 0.5)

        r2_levels   = np.linspace(0,   r2_vmax, 11)
        beta_levels = np.linspace(-bv,  bv,      13)

        def _draw_reg_panel(ax, data, pval, levels, cmap,
                            title, letter):
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND,
                           facecolor="lightgray", zorder=4)
            try:
                ax.coastlines(linewidth=0.4, zorder=5)
            except Exception:
                pass
            ax.gridlines(linestyle=":", linewidth=0.3, alpha=0.4)
            ax.set_title(title, fontsize=11,
                         fontweight="bold", pad=5)
            ax.text(0.02, 0.97, f"({letter})",
                    transform=ax.transAxes, fontsize=11,
                    fontweight="bold", va="top",
                    bbox=dict(facecolor="white", alpha=0.7,
                              edgecolor="none"))
            if data is None:
                ax.text(0.5, 0.5, "no data",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        fontsize=11, color="grey")
                return None
            cf = ax.contourf(
                lons, lats, data,
                levels=levels, cmap=cmap,
                extend="both",
                transform=ccrs.PlateCarree(),
            )
            if cmap == "RdBu_r":
                ax.contour(
                    lons, lats, data,
                    levels=[0], colors="k",
                    linewidths=0.6,
                    transform=ccrs.PlateCarree(),
                )
            # Significance: light grey overlay on non-significant cells
            if pval is not None:
                nonsig = np.where(
                    np.isfinite(pval) & (pval >= 0.05),
                    1.0, np.nan)
                if np.any(np.isfinite(nonsig)):
                    ax.contourf(
                        lons, lats, nonsig,
                        levels=[0.5, 1.5],
                        colors=["white"],
                        alpha=0.45,
                        transform=ccrs.PlateCarree(),
                    )
            return cf

        letters_row = (
            ["a","b","c","d"] if r_idx == 0
            else ["e","f","g","h"]
        )
        cf_r2_obs = _draw_reg_panel(
            axes[r_idx, 0], obs_r2, obs_pv,
            r2_levels, "YlOrRd",
            "OBS — $R^2$", letters_row[0],
        )
        cf_b_obs  = _draw_reg_panel(
            axes[r_idx, 1], obs_beta, obs_pv,
            beta_levels, "RdBu_r",
            "OBS — $\\beta$", letters_row[1],
        )
        cf_r2_ens = _draw_reg_panel(
            axes[r_idx, 2], ens_r2, None,
            r2_levels, "YlOrRd",
            f"Ensemble mean — $R^2$ ({len(model_r2_list)} models)",
            letters_row[2],
        )
        cf_b_ens  = _draw_reg_panel(
            axes[r_idx, 3], ens_beta, None,
            beta_levels, "RdBu_r",
            f"Ensemble mean — $\\beta$ ({len(model_beta_list)} models)",
            letters_row[3],
        )
        if cf_r2_obs is not None: last_r2   = cf_r2_obs
        if cf_b_obs  is not None: last_beta = cf_b_obs

        # Row label
        fig.text(
            0.005,
            0.75 - r_idx * 0.50,
            region_name,
            fontsize=11, fontweight="bold",
            va="center", rotation=90,
        )

    fig.suptitle(
        "Chl-$a$ variance explained by leading sea-ice modes — "
        "Arctic (top) and Southern Ocean (bottom)\n"
        "$R^2$: fraction of local Chl-$a$ variance explained by "
        "multiple regression on leading sea-ice PCs "
        "(enough PCs to explain $\\geq$50\\% of sea-ice variance).  "
        "$\\beta$: PC1 regression coefficient.  "
        "Shading: $p \\geq 0.05$ (F-test).  "
        "Cols 3-4: ensemble mean over available CMIP6 models.",
        fontsize=8.5, y=0.98,
    )
    fig.savefig(OUT_DIR / "figS_seaice_chl_regression.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote figS_seaice_chl_regression.png")




# ============================================================
# Seasonal MLD-Chl-a cross-lag helper functions
# (module-level so accessible from both make_fig10_enhanced
# and the combined figure drawing block)
# ============================================================

_SEASONS_DEF    = {"DJF": [12,1,2], "MAM": [3,4,5],
                   "JJA": [6,7,8],  "SON": [9,10,11]}
_SEASON_LIST    = ["DJF", "MAM", "JJA", "SON"]
_N_LAG_SEAS     = 4


def _seasonal_mean_series(
    ts: xr.DataArray,
    months: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Annual mean of ts restricted to the given months.

    Returns (years, values) sorted by year. ts must already be
    deseasonalised and detrended (z-score applied per season inside
    _seasonal_cross_lag_matrix for comparability across seasons).
    """
    ym   = _ym_key(ts)
    yrs  = ym // 100
    mo   = ym % 100
    vals = np.asarray(ts.values, dtype=float)

    in_season  = np.isin(mo, months)
    unique_yrs = np.unique(yrs[in_season])
    ann = []
    for yr in unique_yrs:
        sel = in_season & (yrs == yr)
        v   = vals[sel]
        if np.any(np.isfinite(v)):
            ann.append((yr, float(np.nanmean(v))))
    if not ann:
        return np.array([]), np.array([])
    ann_arr = np.array(ann)
    return ann_arr[:, 0].astype(int), ann_arr[:, 1]


def _seasonal_cross_lag_matrix(
    chl_z: xr.DataArray,
    mld_z: xr.DataArray,
) -> tuple[np.ndarray, np.ndarray]:
    """4 x 4 matrix of Pearson r and p-values.

    Rows    = MLD season (DJF, MAM, JJA, SON).
    Columns = lag in seasons (0, 1, 2, 3).
    At (row, col): correlate annual MLD anomaly of season s_mld with
    annual Chl-a anomaly of season s_chl = (s_mld_idx + lag) % 4.
    Returns r_mat (4,4) and p_mat (4,4).
    """
    from scipy import stats as _stats

    r_mat = np.full((4, 4), np.nan)
    p_mat = np.full((4, 4), np.nan)

    for s_idx, s_mld in enumerate(_SEASON_LIST):
        yr_m, v_m = _seasonal_mean_series(mld_z, _SEASONS_DEF[s_mld])
        if yr_m.size < 5:
            continue
        std_m = np.nanstd(v_m)
        if std_m > 0:
            v_m = (v_m - np.nanmean(v_m)) / std_m

        for lag in range(_N_LAG_SEAS):
            s_chl_idx = (s_idx + lag) % 4
            s_chl     = _SEASON_LIST[s_chl_idx]
            yr_c, v_c = _seasonal_mean_series(chl_z, _SEASONS_DEF[s_chl])
            if yr_c.size < 5:
                continue
            std_c = np.nanstd(v_c)
            if std_c > 0:
                v_c = (v_c - np.nanmean(v_c)) / std_c

            common_yrs = np.intersect1d(yr_m, yr_c)
            if common_yrs.size < 5:
                continue
            vm = v_m[np.isin(yr_m, common_yrs)]
            vc = v_c[np.isin(yr_c, common_yrs)]
            ok = np.isfinite(vm) & np.isfinite(vc)
            if ok.sum() < 5:
                continue

            r, p = _stats.pearsonr(vm[ok], vc[ok])
            r_mat[s_idx, lag] = r
            p_mat[s_idx, lag] = p

    return r_mat, p_mat


def _draw_seasonal_heatmap(
    ax,
    r_mat: np.ndarray,
    p_mat: np.ndarray,
    title: str,
    show_ylab: bool = True,
) -> object:
    """Draw a 4x4 seasonal cross-lag heatmap on ax.

    Rows = MLD season (DJF/MAM/JJA/SON).
    Columns = lag in seasons (0-3).
    x-axis labels show lag number and the Chl-a season for each
    MLD row as [DJF_chl/MAM_chl/JJA_chl/SON_chl].
    Significant cells (p<0.05) have a bold black border and bold text.
    Returns the mappable for a shared colourbar.
    """
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    norm = TwoSlopeNorm(vmin=-0.8, vcenter=0.0, vmax=0.8)
    im   = ax.imshow(r_mat, cmap="RdBu_r", norm=norm,
                     aspect="auto", origin="upper")

    for si in range(4):
        for li in range(4):
            r = r_mat[si, li]
            p = p_mat[si, li]
            if not np.isfinite(r):
                ax.text(li, si, "—", ha="center", va="center",
                        fontsize=14, color="grey")
                continue
            col = "white" if abs(r) > 0.45 else "black"
            ax.text(li, si, f"{r:+.2f}",
                    ha="center", va="center",
                    fontsize=14, color=col,
                    fontweight="bold"
                    if (np.isfinite(p) and p < 0.05) else "normal")
            if np.isfinite(p) and p < 0.05:
                rect = Rectangle(
                    (li - 0.5, si - 0.5), 1, 1,
                    linewidth=2.0, edgecolor="black",
                    facecolor="none", zorder=5,
                )
                ax.add_patch(rect)

    # x-axis: clean lag labels
    ax.set_xticks(range(_N_LAG_SEAS))
    ax.set_xticklabels(
        ["lag 0\n(same)", "lag 1\n(+1)", "lag 2\n(+2)", "lag 3\n(+3)"],
        fontsize=13, ha="center",
    )
    ax.set_xlabel("MLD → Chl-a lag (seasons)", fontsize=13)

    if show_ylab:
        ax.set_yticks(range(4))
        ax.set_yticklabels(_SEASON_LIST, fontsize=13, fontweight="bold")
        ax.set_ylabel("MLD season", fontsize=13)
    else:
        ax.set_yticks(range(4))
        ax.set_yticklabels([])

    ax.set_title(title, fontsize=14, fontweight="bold", pad=4)
    return im


def make_fig10_enhanced(
    obs_chl_ts:   xr.DataArray,
    obs_mld_ts:   "xr.DataArray | None",
    model_chl_ts: dict[str, xr.DataArray],
    model_mld_ts: dict[str, xr.DataArray],
    common_grid:  xr.DataArray,
    period:       slice = LAG_PERIOD,
) -> None:
    """Three-panel enhanced MLD-Chl-a lag analysis.

    Panel A (fig10a_lag_all_models.png)
        Six-basin lag correlation with ALL 7 models shown individually
        as thin coloured lines, ensemble mean as a thick coloured line,
        ensemble spread (min-max) as a shaded band, and OBS as a thick
        black line with significance markers (filled circles, p<0.05)
        and 95% bootstrap CI ribbon.

    Panel B (fig10b_seasonal_lag.png)
        For the two scientifically interesting basins (Indian and
        Arctic), the lag correlation is repeated separately for DJF,
        MAM, JJA, SON anomalies. Only months belonging to each season
        are selected from the deseasonalised, detrended, standardised
        series before computing the correlation. Four coloured curves
        per basin (OBS only) reveal which season drives the full-record
        signal seen in Fig 10.

    Panel C (fig10c_peak_lag_summary.png)
        Taylor-diagram-style scatter summary. One point per basin.
        x-axis: OBS peak lag (months, positive = MLD leads Chl-a).
        y-axis: ensemble-mean model peak lag.
        Point size: proportional to |OBS peak r|.
        Point colour: sign of OBS peak r (red = positive, blue = negative).
        Diagonal reference: y=x (models agree with OBS).
        Vertical and horizontal lines at lag=0 divide the plot into
        quadrants.
    """
    logger.info("=== Fig 10 enhanced: Panel A / B / C ===")
    masks       = _build_masks(common_grid)
    basin_order = [b for b in MEANSTATE_BASINS if b in masks]
    models_list = [m for m in MODELS
                   if m in model_chl_ts and m in model_mld_ts]

    # ------------------------------------------------------------------
    # Step 1: compute lag curves for all sources (OBS + all models)
    # ------------------------------------------------------------------
    logger.info("  Step 1: computing basin-mean MLD-Chl lag curves...")
    all_curves: dict[str, dict[str, tuple]] = {b: {} for b in basin_order}

    # OBS
    if obs_mld_ts is not None:
        try:
            chl_re, mld_re = _prep_for_lag(
                obs_chl_ts, obs_mld_ts, common_grid, period)
            for basin, mask in masks.items():
                if basin not in basin_order:
                    continue
                chl_b = _basin_mean_timeseries(chl_re, mask)
                mld_b = _basin_mean_timeseries(mld_re, mask)
                chl_z = _deseasonalise_zscore(chl_b)
                mld_z = _deseasonalise_zscore(mld_b)
                all_curves[basin]["OBS"] = \
                    _lag_correlation_with_bootstrap(chl_z, mld_z)
        except Exception as e:
            logger.warning(f"  OBS MLD lag failed: {e}")

    # Models
    for model in models_list:
        try:
            chl_re, mld_re = _prep_for_lag(
                model_chl_ts[model], model_mld_ts[model],
                common_grid, period)
            for basin, mask in masks.items():
                if basin not in basin_order:
                    continue
                chl_b = _basin_mean_timeseries(chl_re, mask)
                mld_b = _basin_mean_timeseries(mld_re, mask)
                chl_z = _deseasonalise_zscore(chl_b)
                mld_z = _deseasonalise_zscore(mld_b)
                all_curves[basin][model] = \
                    _lag_correlation_with_bootstrap(chl_z, mld_z)
        except Exception as e:
            logger.warning(f"  [{model}] MLD lag failed: {e}")

    # ------------------------------------------------------------------
    # PANEL A: six basins, all models + ensemble stats + OBS
    # ------------------------------------------------------------------
    logger.info("  Panel A: all-model six-basin figure...")
    ncols = 3
    nrows = int(np.ceil(len(basin_order) / ncols))
    fig_a, axes_a = plt.subplots(
        nrows, ncols,
        figsize=(14, 4.5 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    fig_a.subplots_adjust(hspace=0.32, wspace=0.12,
                          left=0.06, right=0.97,
                          bottom=0.10, top=0.93)

    for b_idx, basin in enumerate(basin_order):
        ax     = axes_a[b_idx // ncols, b_idx % ncols]
        curves = all_curves[basin]

        ax.axhline(0, color="k", lw=0.5, alpha=0.6)
        ax.axvline(0, color="k", lw=0.5, alpha=0.6, ls=":")

        # Collect model r arrays for ensemble stats
        lags_ref = None
        model_rs: list[np.ndarray] = []

        for model in models_list:
            if model not in curves:
                continue
            entry = curves[model]
            lags, rs = entry[0], entry[1]
            if lags_ref is None:
                lags_ref = lags
            # Individual thin line (low alpha)
            color = PALETTE[models_list.index(model) % len(PALETTE)]
            ax.plot(lags, rs, lw=0.8, color=color, alpha=0.45,
                    zorder=4, label=model)
            model_rs.append(rs)

        # Ensemble spread and mean
        if model_rs and lags_ref is not None:
            arr   = np.stack(model_rs, axis=0)   # (n_models, n_lags)
            e_min = np.nanmin(arr, axis=0)
            e_max = np.nanmax(arr, axis=0)
            e_mn  = np.nanmean(arr, axis=0)
            ax.fill_between(lags_ref, e_min, e_max,
                            color="steelblue", alpha=0.15,
                            lw=0, zorder=3, label="Ensemble spread")
            ax.plot(lags_ref, e_mn, color="steelblue", lw=2.0,
                    ls="--", zorder=6, alpha=0.9,
                    label="Ensemble mean")

        # OBS: thick black, CI ribbon, significance markers
        if "OBS" in curves:
            entry = curves["OBS"]
            if len(entry) == 5:
                lags, rs, ci_lo, ci_hi, pval = entry
            else:
                lags, rs = entry
                nan_arr = np.full_like(rs, np.nan)
                ci_lo = ci_hi = pval = nan_arr

            if np.any(np.isfinite(ci_lo)):
                ax.fill_between(lags, ci_lo, ci_hi,
                                color="black", alpha=0.12,
                                lw=0, zorder=7)
            ax.plot(lags, rs, color="black", lw=2.4,
                    zorder=10, label="OBS")
            sig = np.isfinite(pval) & (pval < LAG_BOOT_ALPHA)
            if np.any(sig):
                ax.plot(lags[sig], rs[sig], "o", ms=5.5,
                        mfc="black", mec="white", mew=0.6,
                        zorder=11)
            # Peak annotation (OBS only)
            if np.any(np.isfinite(rs)):
                k_pk = int(lags[np.nanargmax(np.abs(rs))])
                r_pk = float(rs[np.nanargmax(np.abs(rs))])
                p_pk = (float(pval[np.nanargmax(np.abs(rs))])
                        if np.any(np.isfinite(pval)) else np.nan)
                txt = (f"OBS: r={r_pk:+.2f} @ lag={k_pk:+d}"
                       + (f"\np={p_pk:.3f}"
                          if np.isfinite(p_pk) else ""))
                ax.text(0.02, 0.04, txt,
                        transform=ax.transAxes, fontsize=7,
                        va="bottom", family="monospace",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor="white",
                                  edgecolor="grey", alpha=0.85))

        ax.set_title(basin, fontsize=11, fontweight="bold")
        ax.set_xlim(-LAG_MAX, LAG_MAX)
        ax.set_ylim(-1, 1)
        ax.set_xticks(np.arange(-LAG_MAX, LAG_MAX + 1, 4))
        ax.grid(alpha=0.22)
        if b_idx % ncols == 0:
            ax.set_ylabel("Pearson r", fontsize=10)
        if b_idx // ncols == nrows - 1:
            ax.set_xlabel("Lag (months)  —  MLD leads →", fontsize=9)

        letter = chr(ord("a") + b_idx)
        ax.text(0.02, 0.97, f"({letter})",
                transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top")

    for idx in range(len(basin_order), nrows * ncols):
        axes_a.flatten()[idx].set_visible(False)

    handles, labels = axes_a[0, 0].get_legend_handles_labels()
    fig_a.legend(handles, labels,
                 loc="lower center",
                 ncol=min(len(handles), 5),
                 fontsize=8, framealpha=0.9,
                 bbox_to_anchor=(0.5, -0.01))
    fig_a.suptitle(
        "Fig 10a — MLD–Chl-$a$ lag correlation: all models + ensemble stats\n"
        "Thin lines: individual models;  dashed: ensemble mean;  "
        "blue shading: ensemble spread (min–max).\n"
        "Black: OBS (ESA-CCI Chl vs ORAS5 MLD);  "
        "ribbon: 95% block-bootstrap CI;  filled markers: $p<0.05$.",
        fontsize=9, y=0.99,
    )
    fig_a.savefig(OUT_DIR / "fig10a_lag_all_models.png",
                  dpi=180, bbox_inches="tight")
    plt.close(fig_a)
    logger.success("  wrote fig10a_lag_all_models.png")

    # ------------------------------------------------------------------
    # PANEL B: seasonal-mean cross-lag heatmap, merged with Panel A
    # into a single combined figure (3 rows x 6 cols):
    #   Row 1 = lag-correlation curves (Panel A content)
    #   Row 2 = OBS seasonal heatmaps
    #   Row 3 = Ensemble mean seasonal heatmaps
    # Also saved as standalone fig10b_seasonal_lag.png for the paper.
    # ------------------------------------------------------------------
    logger.info("  Panel B: seasonal-mean cross-lag heatmap (all basins)...")

    # Use module-level season constants
    SEASONS     = _SEASONS_DEF
    SEASON_LIST = _SEASON_LIST
    N_LAG_SEAS  = _N_LAG_SEAS

    all_basins = basin_order

    # Pre-compute obs and model matrices
    obs_mats:   dict[str, tuple] = {}
    model_mats: dict[str, list]  = {}
    ens_mats:   dict[str, tuple] = {}

    chl_re_obs = mld_re_obs = None
    if obs_mld_ts is not None:
        try:
            chl_re_obs, mld_re_obs = _prep_for_lag(
                obs_chl_ts, obs_mld_ts, common_grid, period)
        except Exception as e:
            logger.warning(f"  Panel B OBS regrid failed: {e}")

    model_re: dict[str, tuple] = {}
    for model in models_list:
        try:
            cr, mr = _prep_for_lag(
                model_chl_ts[model], model_mld_ts[model],
                common_grid, period)
            model_re[model] = (cr, mr)
        except Exception as e:
            logger.warning(f"  Panel B [{model}] regrid failed: {e}")

    for basin in all_basins:
        mask = masks[basin]
        if chl_re_obs is not None and mld_re_obs is not None:
            try:
                chl_b = _basin_mean_timeseries(chl_re_obs, mask)
                mld_b = _basin_mean_timeseries(mld_re_obs, mask)
                chl_z = _deseasonalise_zscore(chl_b)
                mld_z = _deseasonalise_zscore(mld_b)
                r, p  = _seasonal_cross_lag_matrix(chl_z, mld_z)
                obs_mats[basin] = (r, p)
            except Exception as e:
                logger.warning(f"  Panel B OBS {basin} failed: {e}")

        model_r_list = []
        for model, (cr, mr) in model_re.items():
            try:
                chl_b = _basin_mean_timeseries(cr, mask)
                mld_b = _basin_mean_timeseries(mr, mask)
                chl_z = _deseasonalise_zscore(chl_b)
                mld_z = _deseasonalise_zscore(mld_b)
                rm, _ = _seasonal_cross_lag_matrix(chl_z, mld_z)
                model_r_list.append(rm)
            except Exception as e:
                logger.warning(
                    f"  Panel B [{model}] {basin} failed: {e}")
        model_mats[basin] = model_r_list

        if model_r_list:
            from scipy import stats as _stats
            arr   = np.stack(model_r_list, axis=0)
            ens_r = np.nanmean(arr, axis=0)
            ens_p = np.full((4, 4), np.nan)
            for si in range(4):
                for li in range(4):
                    v  = arr[:, si, li]
                    ok = np.isfinite(v)
                    if ok.sum() >= 3:
                        _, p = _stats.ttest_1samp(v[ok], 0.0)
                        ens_p[si, li] = p
            ens_mats[basin] = (ens_r, ens_p)

    n_models_used = len(model_re)

    # ------------------------------------------------------------------
    # Combined figure: clean explicit-axes layout
    #   Row 1 (top 40%):  lag-correlation curves, 6 basins
    #   Row 2 (mid 28%):  OBS seasonal heatmaps, Indian + Arctic only
    #   Row 3 (bot 28%):  Ensemble mean seasonal heatmaps
    # Uses figure-fraction coordinates for precise alignment.
    # ------------------------------------------------------------------
    import matplotlib.gridspec as _gs

    # Figure dimensions
    fig_w   = 26.0   # inches
    fig_h   = 18.0
    fig_comb = plt.figure(figsize=(fig_w, fig_h))

    ncols_c      = len(all_basins)           # 6
    heatmap_basins = [b for b in ["Indian", "Arctic"] if b in masks]
    indian_col   = all_basins.index("Indian") if "Indian" in all_basins else 3
    arctic_col   = all_basins.index("Arctic") if "Arctic" in all_basins else 5

    # Layout constants (figure fractions)
    LEFT  = 0.07
    RIGHT = 0.97
    TOP   = 0.93
    BOT   = 0.04
    W     = RIGHT - LEFT          # total plot width
    col_w = W / ncols_c           # width per column

    # Row boundaries (bottom of each row)
    ROW1_TOP = TOP
    ROW1_BOT = 0.56
    ROW2_TOP = 0.52
    ROW2_BOT = 0.28
    ROW3_TOP = 0.24
    ROW3_BOT = BOT
    ROW_H1   = ROW1_TOP - ROW1_BOT
    ROW_H2   = ROW2_TOP - ROW2_BOT
    ROW_H3   = ROW3_TOP - ROW3_BOT

    # Heat-column width (wider than curve columns — only 2 heatmaps)
    HEAT_W   = min(0.30, W * 0.28)   # width of each heatmap panel
    HEAT_GAP = 0.06                   # gap between the two heatmaps
    # Centre the two heatmaps in the figure
    heat_total = 2 * HEAT_W + HEAT_GAP
    heat_left  = LEFT + (W - heat_total) / 2
    heat_positions = {
        "Indian": heat_left,
        "Arctic": heat_left + HEAT_W + HEAT_GAP,
    }

    # ── Row 1: six lag-correlation curve panels ──────────────────────
    outer = _gs.GridSpec(
        1, ncols_c,
        figure=fig_comb,
        left=LEFT, right=RIGHT,
        bottom=ROW1_BOT, top=ROW1_TOP,
        wspace=0.12,
    )

    # Row 1: lag-correlation curves (reproduce Panel A per basin)
    for b_idx, basin in enumerate(all_basins):
        ax = fig_comb.add_subplot(outer[0, b_idx])
        curves = all_curves[basin]

        ax.axhline(0, color="k", lw=0.5, alpha=0.6)
        ax.axvline(0, color="k", lw=0.5, alpha=0.6, ls=":")

        lags_ref  = None
        model_rs: list[np.ndarray] = []
        for model in models_list:
            if model not in curves:
                continue
            entry = curves[model]
            lags, rs = entry[0], entry[1]
            if lags_ref is None:
                lags_ref = lags
            color = PALETTE[models_list.index(model) % len(PALETTE)]
            ax.plot(lags, rs, lw=0.8, color=color, alpha=0.45, zorder=4)
            model_rs.append(rs)

        if model_rs and lags_ref is not None:
            arr   = np.stack(model_rs, axis=0)
            e_mn  = np.nanmean(arr, axis=0)
            e_min = np.nanmin(arr, axis=0)
            e_max = np.nanmax(arr, axis=0)
            ax.fill_between(lags_ref, e_min, e_max,
                            color="steelblue", alpha=0.15, lw=0)
            ax.plot(lags_ref, e_mn, color="steelblue",
                    lw=1.8, ls="--", alpha=0.9)

        if "OBS" in curves:
            entry = curves["OBS"]
            lags, rs = entry[0], entry[1]
            ci_lo = entry[2] if len(entry) > 2 else np.full_like(rs, np.nan)
            ci_hi = entry[3] if len(entry) > 3 else np.full_like(rs, np.nan)
            pval  = entry[4] if len(entry) > 4 else np.full_like(rs, np.nan)
            if np.any(np.isfinite(ci_lo)):
                ax.fill_between(lags, ci_lo, ci_hi,
                                color="black", alpha=0.12, lw=0)
            ax.plot(lags, rs, color="black", lw=2.2, zorder=10)
            sig = np.isfinite(pval) & (pval < LAG_BOOT_ALPHA)
            if np.any(sig):
                ax.plot(lags[sig], rs[sig], "o", ms=4.5,
                        mfc="black", mec="white", mew=0.5, zorder=11)

        ax.set_xlim(-LAG_MAX, LAG_MAX)
        ax.set_ylim(-1, 1)
        ax.set_xticks(np.arange(-LAG_MAX, LAG_MAX + 1, 6))
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        if b_idx == 0:
            # Leftmost panel: large visible y-axis numbers
            ax.tick_params(axis="y", labelsize=16, width=1.5, length=6)
            ax.tick_params(axis="x", labelsize=16)
            ax.set_ylabel("Pearson r", fontsize=16, fontweight="bold")
            ax.text(-0.30, 0.5, "Lag correlation\n(monthly)",
                    transform=ax.transAxes, fontsize=12,
                    fontweight="bold", ha="center", va="center",
                    rotation=90, color="dimgrey")
        else:
            # All other panels: hide y-axis tick labels
            ax.tick_params(axis="y", labelleft=False, length=3)
            ax.tick_params(axis="x", labelsize=16)
            ax.set_ylabel("")
        ax.grid(alpha=0.2)
        ax.set_title(basin, fontsize=16, fontweight="bold")
        ax.set_xlabel("Lag (months)", fontsize=16)

        letter = chr(ord("a") + b_idx)
        ax.text(0.02, 0.97, f"({letter})",
                transform=ax.transAxes, fontsize=14,
                fontweight="bold", va="top")

    # ── Rows 2-3: peak-lag grouped bar charts ────────────────────────────
    # For each MLD season (DJF/MAM/JJA/SON), find the lag with the
    # highest |r| and show that single r value as a bar.
    # The best Chl-a season is annotated on each bar.
    # OBS (black) and ensemble mean (steelblue) bars side by side.

    # (no divider line between rows)

    def _peak_lag_bars(r_mat: np.ndarray) -> tuple[list, list, list]:
        """For each MLD season row, find the lag with max |r|.

        Returns (peak_r, peak_chl_season, peak_lag_idx) lists of
        length 4 (one per MLD season).
        """
        peak_r, peak_chl, peak_lag_idx = [], [], []
        for si in range(4):
            row = r_mat[si, :]
            fin = np.isfinite(row)
            if not fin.any():
                peak_r.append(np.nan)
                peak_chl.append("—")
                peak_lag_idx.append(0)
                continue
            best_li = int(np.nanargmax(np.abs(row)))
            peak_r.append(float(row[best_li]))
            # Chl-a season at this lag
            peak_chl.append(_SEASON_LIST[(si + best_li) % 4])
            peak_lag_idx.append(best_li)
        return peak_r, peak_chl, peak_lag_idx

    # Combined row 2+3 → one bar chart per basin (rows merged)
    # Layout: side-by-side bar chart, 4 groups (MLD seasons),
    # 2 bars per group (OBS | ensemble mean).
    # Placed in the ROW2/ROW3 space, one axes per basin.
    bar_w      = 0.30    # width of each bar
    bar_gap    = 0.08    # gap between OBS and ens bars in same group
    group_gap  = 0.6     # gap between MLD season groups
    colors_bar = {"OBS": "black", "Ensemble mean": "steelblue"}
    bar_row_h  = ROW2_TOP - ROW3_BOT   # combined height for both rows

    for heat_idx, basin in enumerate(heatmap_basins):
        x_left = heat_positions[basin]
        ax = fig_comb.add_axes(
            [x_left, ROW3_BOT + 0.02,
             HEAT_W,  bar_row_h - 0.05]
        )

        x_centers = np.arange(4) * (1 + group_gap)  # group centres

        for src_idx, (src_label, mats_dict, bar_color) in enumerate([
            ("OBS",           obs_mats, "black"),
            ("Ensemble mean", ens_mats, "steelblue"),
        ]):
            if basin not in mats_dict:
                continue
            r_mat, p_mat = mats_dict[basin]
            peak_r, peak_chl, _ = _peak_lag_bars(r_mat)

            x_pos = x_centers + (src_idx - 0.5) * (bar_w + bar_gap)
            for gi, (r_val, chl_s) in enumerate(zip(peak_r, peak_chl)):
                if not np.isfinite(r_val):
                    continue
                bar_col = bar_color
                ax.bar(
                    x_pos[gi], r_val,
                    width=bar_w,
                    color=bar_col,
                    alpha=0.80 if src_label == "OBS" else 0.55,
                    edgecolor="white", linewidth=0.5,
                    label=src_label if gi == 0 else "",
                )
                # Annotate with best Chl-a season name
                y_txt = r_val + (0.03 if r_val >= 0 else -0.06)
                ax.text(
                    x_pos[gi], y_txt, chl_s,
                    ha="center", va="bottom" if r_val >= 0 else "top",
                    fontsize=10, fontweight="bold",
                    color=bar_col,
                )

        # Zero line
        ax.axhline(0, color="k", lw=0.8, alpha=0.6)

        # x-axis: MLD season labels
        ax.set_xticks(x_centers)
        ax.set_xticklabels(
            [f"MLD:\n{s}" for s in _SEASON_LIST],
            fontsize=12, fontweight="bold",
        )
        ax.set_xlabel("MLD season", fontsize=12)

        # y-axis: only on leftmost basin
        if heat_idx == 0:
            ax.set_ylabel("Peak Pearson r", fontsize=12)
            ax.tick_params(axis="y", labelsize=12)
        else:
            ax.tick_params(axis="y", labelleft=False)

        ax.set_ylim(-1, 1)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.grid(alpha=0.2, axis="y")

        # Basin title
        # Panel letter for bar chart row: (g) Indian, (h) Arctic
        bar_letter = chr(ord("g") + heat_idx)
        ax.text(0.02, 0.98, f"({bar_letter})",
                transform=ax.transAxes, fontsize=14,
                fontweight="bold", va="top")
        ax.set_title(
            f"{basin}\n(bar label = best Chl-$a$ season at peak |r|)",
            fontsize=13, fontweight="bold", pad=6,
        )

        # Legend on first basin only
        if heat_idx == 0:
            ax.legend(fontsize=11, loc="lower left",
                      framealpha=0.9)

        # Significance annotation: asterisk on OBS bars where p < 0.05
        if basin in obs_mats:
            r_mat, p_mat = obs_mats[basin]
            peak_r_obs, _, peak_li_obs = _peak_lag_bars(r_mat)
            x_obs = x_centers - 0.5 * (bar_w + bar_gap)
            for gi, (r_val, li) in enumerate(zip(peak_r_obs, peak_li_obs)):
                pv = p_mat[gi, li] if np.isfinite(p_mat[gi, li]) else 1.0
                if pv < 0.05 and np.isfinite(r_val):
                    y_star = r_val + (0.10 if r_val >= 0 else -0.12)
                    ax.text(
                        x_obs[gi], y_star, "*",
                        ha="center", va="center",
                        fontsize=16, color="black",
                    )

    # Legend for row 1 curves
    from matplotlib.lines import Line2D as _L2D
    from matplotlib.patches import Patch as _Patch
    legend_els = [
        _L2D([0],[0], color="black", lw=2.5,
             label="OBS Pearson r"),
        _L2D([0],[0], color="grey", lw=6, alpha=0.5,
             label="OBS 95% bootstrap CI (grey shading)"),
        _L2D([0],[0], color="steelblue", lw=2.0, ls="--",
             label="Ensemble mean"),
        _L2D([0],[0], color="steelblue", lw=6, alpha=0.4,
             label="Ensemble spread min-max (blue shading)"),
    ]
    first_ax = fig_comb.axes[0]
    first_ax.legend(
        handles=legend_els,
        loc="lower left",
        fontsize=10, framealpha=0.95,
        edgecolor="grey",
        handlelength=3.0,
        bbox_to_anchor=(0.0, -0.42),
        borderaxespad=0,
    )
    # No suptitle — figure is self-explanatory from panel labels
    fig_comb.savefig(OUT_DIR / "fig10_combined.png",
                     dpi=150, bbox_inches="tight")
    plt.close(fig_comb)
    logger.success("  wrote fig10_combined.png")

    # Also save standalone seasonal heatmap figure (for supplemental)
    nrows_b = 2
    ncols_b = len(all_basins)
    fig_b, axes_b = plt.subplots(
        nrows_b, ncols_b,
        figsize=(4.2 * ncols_b, 5.0 * nrows_b),
        squeeze=False,
    )
    fig_b.subplots_adjust(
        left=0.06, right=0.91,
        bottom=0.08, top=0.88,
        hspace=0.30, wspace=0.18,
    )
    last_im = None
    for b_idx, basin in enumerate(all_basins):
        ax0 = axes_b[0, b_idx]
        if basin in obs_mats:
            r, p = obs_mats[basin]
            im = _draw_seasonal_heatmap(
                ax0, r, p, title=basin, show_ylab=(b_idx == 0))
            last_im = im
        else:
            ax0.set_visible(False)
        ax1 = axes_b[1, b_idx]
        if basin in ens_mats:
            r, p = ens_mats[basin]
            im = _draw_seasonal_heatmap(
                ax1, r, p, title="", show_ylab=(b_idx == 0))
            last_im = im
        else:
            ax1.set_visible(False)

    if last_im is not None:
        cax = fig_b.add_axes([0.925, 0.15, 0.015, 0.68])
        cb  = fig_b.colorbar(last_im, cax=cax, extend="both")
        cb.set_label("Pearson r", fontsize=9)
        cb.ax.tick_params(labelsize=8)
    fig_b.suptitle(
        "Fig 10b — Seasonal MLD–Chl-$a$ cross-lag heatmap: all basins\n"
        "Row 1: OBS. Row 2: Ensemble mean "
        f"({n_models_used} of {len(MODELS)} models).\n"
        "Bold border: p < 0.05.  "
        "Column labels [X/Y/Z/W] = Chl-a season per MLD row "
        "(DJF/MAM/JJA/SON).",
        fontsize=8.5, y=0.97,
    )
    fig_b.savefig(OUT_DIR / "fig10b_seasonal_lag.png",
                  dpi=180, bbox_inches="tight")
    plt.close(fig_b)
    logger.success("  wrote fig10b_seasonal_lag.png")


    # ------------------------------------------------------------------
    # PANEL C: peak-lag summary scatter (Taylor-diagram style)
    # ------------------------------------------------------------------
    logger.info("  Panel C: peak-lag summary scatter...")

    fig_c, ax_c = plt.subplots(1, 1, figsize=(7, 6.5))

    obs_peak_r:   dict[str, float] = {}
    obs_peak_lag: dict[str, float] = {}
    ens_peak_lag: dict[str, float] = {}
    ens_peak_r:   dict[str, float] = {}

    for basin in basin_order:
        curves = all_curves[basin]

        # OBS peak
        if "OBS" in curves:
            lags, rs = curves["OBS"][0], curves["OBS"][1]
            if np.any(np.isfinite(rs)):
                k  = int(lags[np.nanargmax(np.abs(rs))])
                r  = float(rs[np.nanargmax(np.abs(rs))])
                obs_peak_lag[basin] = float(k)
                obs_peak_r[basin]   = r

        # Ensemble mean peak lag
        model_peak_lags = []
        model_peak_rs   = []
        for model in models_list:
            if model not in curves:
                continue
            lags_m, rs_m = curves[model][0], curves[model][1]
            if np.any(np.isfinite(rs_m)):
                k_m = int(lags_m[np.nanargmax(np.abs(rs_m))])
                r_m = float(rs_m[np.nanargmax(np.abs(rs_m))])
                model_peak_lags.append(float(k_m))
                model_peak_rs.append(r_m)

        if model_peak_lags:
            ens_peak_lag[basin] = float(np.mean(model_peak_lags))
            ens_peak_r[basin]   = float(np.mean(model_peak_rs))

    # Plot one point per basin
    all_lags = (
        list(obs_peak_lag.values()) + list(ens_peak_lag.values())
    )
    ax_range = max(abs(min(all_lags)), abs(max(all_lags))) + 2 if all_lags \
        else LAG_MAX

    ax_c.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax_c.axvline(0, color="k", lw=0.6, alpha=0.5)
    # y=x reference line
    lim = ax_range + 1
    ax_c.plot([-lim, lim], [-lim, lim], "k--", lw=1.0,
              alpha=0.4, label="Model = OBS")

    # Quadrant labels
    for xt, yt, txt in [
        ( lim * 0.55,  lim * 0.85, "MLD leads\nmodel & OBS"),
        (-lim * 0.85,  lim * 0.85, "MLD leads OBS\nChl leads model"),
        (-lim * 0.85, -lim * 0.85, "Chl leads\nmodel & OBS"),
        ( lim * 0.55, -lim * 0.85, "MLD leads model\nChl leads OBS"),
    ]:
        ax_c.text(xt, yt, txt, fontsize=7, color="grey",
                  ha="center", va="center", style="italic")

    for basin in basin_order:
        if basin not in obs_peak_lag or basin not in ens_peak_lag:
            continue
        x    = obs_peak_lag[basin]
        y    = ens_peak_lag[basin]
        r    = obs_peak_r[basin]
        size = max(40, abs(r) * 400)
        color = "#CC3333" if r > 0 else "#3366CC"

        ax_c.scatter(x, y, s=size, color=color,
                     edgecolors="white", linewidths=0.8,
                     zorder=8, alpha=0.85)
        ax_c.annotate(
            basin,
            xy=(x, y),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8.5, fontweight="bold",
        )

    # Legend for colour meaning
    from matplotlib.lines import Line2D
    legend_els = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#CC3333", markersize=10,
               label="OBS r > 0 (deep MLD → more Chl)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#3366CC", markersize=10,
               label="OBS r < 0 (deep MLD → less Chl)"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="grey", markersize=6,
               label="Point size ∝ |OBS peak r|"),
        Line2D([0], [0], color="k", ls="--", lw=1.0,
               label="Model = OBS (y = x)"),
    ]
    ax_c.legend(handles=legend_els, fontsize=8,
                loc="upper left", framealpha=0.9)

    ax_c.set_xlim(-lim, lim)
    ax_c.set_ylim(-lim, lim)
    ax_c.set_xlabel("OBS peak lag (months, + = MLD leads Chl-a)",
                    fontsize=10)
    ax_c.set_ylabel("Ensemble-mean model peak lag (months)",
                    fontsize=10)
    ax_c.set_title(
        "Fig 10c — MLD–Chl-$a$ peak-lag summary\n"
        "Each point = one basin.  Size ∝ |OBS peak r|.  "
        "Colour = sign of OBS coupling.\n"
        "Points on the dashed diagonal: models agree with OBS on lag.",
        fontsize=9,
    )
    ax_c.grid(alpha=0.25)

    fig_c.tight_layout()
    fig_c.savefig(OUT_DIR / "fig10c_peak_lag_summary.png",
                  dpi=180, bbox_inches="tight")
    plt.close(fig_c)
    logger.success("  wrote fig10c_peak_lag_summary.png")




def make_fig10_lag_mld_with_modes(
    obs_chl_ts:   xr.DataArray,
    obs_mld_ts:   "xr.DataArray | None",
    model_chl_ts: dict[str, xr.DataArray],
    model_mld_ts: dict[str, xr.DataArray],
    common_grid:  xr.DataArray,
    indices:      dict[str, xr.DataArray] | None = None,
    period:       slice = LAG_PERIOD,
) -> None:
    """Produce Fig 10 (raw MLD-Chl lag) and Fig 10b (modes removed).

    Fig 10  -- fig10_lag_corr_chl_mld.png
        Six-basin lag correlation between MLD and Chl-a anomalies,
        no index removed. Identical to make_fig10_lag_mld output.

    Fig 10b -- fig10b_lag_corr_chl_mld_amo_removed.png
        Same six-basin layout with AMO, Nino 3.4, and PDO all
        regressed out of both Chl-a and MLD anomalies before
        correlation. Directly comparable to Fig 9b for SST.

    All anomaly series are deseasonalised, linearly detrended, and
    standardised to unit variance (Section 2.6) before correlation.
    For models, the OBS-derived indices are used (standard practice
    when model internal variability need not match observed phase).
    """
    logger.info(
        "=== Fig 10 + Fig 10b: MLD-Chl lag (raw & modes removed) ==="
    )
    masks = _build_masks(common_grid)

    # ------------------------------------------------------------------
    # Helper: build lag curves for all basins, optionally removing indices
    # ------------------------------------------------------------------
    def _build_mld_lag_curves(
        remove: list[str],
    ) -> dict[str, dict[str, tuple]]:
        lag_curves: dict[str, dict[str, tuple]] = {b: {} for b in masks}

        # OBS
        if obs_mld_ts is not None:
            try:
                chl_re, mld_re = _prep_for_lag(
                    obs_chl_ts, obs_mld_ts, common_grid
                )
                for basin, mask in masks.items():
                    chl_b = _basin_mean_timeseries(chl_re, mask)
                    mld_b = _basin_mean_timeseries(mld_re, mask)
                    chl_z = _deseasonalise_zscore(chl_b)
                    mld_z = _deseasonalise_zscore(mld_b)
                    for idx_name in remove:
                        if indices and idx_name in indices:
                            chl_z = _regress_out(chl_z, indices[idx_name])
                            mld_z = _regress_out(mld_z, indices[idx_name])
                    result = _lag_correlation_with_bootstrap(chl_z, mld_z)
                    lag_curves[basin]["OBS"] = result
                    lags, rs = result[0], result[1]
                    if np.any(np.isfinite(rs)):
                        k = int(lags[np.nanargmax(np.abs(rs))])
                        r = float(rs[np.nanargmax(np.abs(rs))])
                        p = float(result[4][np.nanargmax(np.abs(rs))])
                        tag = f"remove={remove}" if remove else "raw"
                        logger.success(
                            f"  [OBS  ] {basin:10s} ({tag}): "
                            f"peak r={r:+.3f} @ lag={k:+d} (p={p:.3f})"
                        )
            except Exception as e:
                import traceback as _tb
                logger.warning(f"  OBS MLD lag-corr failed: {e}")
                logger.warning(_tb.format_exc().replace("<", "\\<"))
        else:
            logger.warning("  OBS MLD not available; skipping OBS curve")

        # Models
        for model in MODELS:
            if model not in model_chl_ts or model not in model_mld_ts:
                continue
            try:
                chl_re, mld_re = _prep_for_lag(
                    model_chl_ts[model], model_mld_ts[model], common_grid
                )
                for basin, mask in masks.items():
                    chl_b = _basin_mean_timeseries(chl_re, mask)
                    mld_b = _basin_mean_timeseries(mld_re, mask)
                    chl_z = _deseasonalise_zscore(chl_b)
                    mld_z = _deseasonalise_zscore(mld_b)
                    for idx_name in remove:
                        if indices and idx_name in indices:
                            chl_z = _regress_out(chl_z, indices[idx_name])
                            mld_z = _regress_out(mld_z, indices[idx_name])
                    lag_curves[basin][model] = _lag_correlation_with_bootstrap(
                        chl_z, mld_z
                    )
            except Exception as e:
                logger.warning(f"  [{model}] MLD lag-corr failed: {e}")

        return lag_curves

    # ------------------------------------------------------------------
    # Fig 10 -- raw (no removal)
    # ------------------------------------------------------------------
    raw_curves = _build_mld_lag_curves(remove=[])
    _make_lag_figure(
        raw_curves,
        obs_label="ESA-CCI Chl vs ORAS5 MLD",
        driver_name="MLD",
        title="Lag correlation: basin-mean Chl-a vs MLD anomalies",
        filename="fig10_lag_corr_chl_mld.png",
        fig_num="10",
    )

    if not indices:
        logger.warning(
            "  No climate indices; skipping Fig 10b"
        )
        return

    # ------------------------------------------------------------------
    # Fig 10b -- all three indices removed simultaneously
    # ------------------------------------------------------------------
    all_names = [n for n in ("AMO", "NINO34", "PDO") if n in indices]
    removed_curves = _build_mld_lag_curves(remove=all_names)
    label_removed  = " + ".join(all_names)
    _make_lag_figure(
        removed_curves,
        obs_label=(
            f"ESA-CCI Chl vs ORAS5 MLD  ({label_removed} removed)"
        ),
        driver_name="MLD",
        title=(
            f"Lag correlation: Chl-a vs MLD anomalies "
            f"({label_removed} removed)"
        ),
        filename="fig10b_lag_corr_chl_mld_amo_removed.png",
        fig_num="10b",
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
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        try:

            ax.coastlines(linewidth=0.4)

        except Exception:

            pass
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
            transform=ccrs.PlateCarree(),
        )
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        try:

            ax.coastlines(linewidth=0.4)

        except Exception:

            pass
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
            logger.info(traceback.format_exc().replace("<", "\\<"))

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
# Supplemental: pointwise peak-lag correlation maps
# ============================================================
# For each physical driver (SST, MLD, siconc) and each source
# (OBS + 7 models), compute at every (lat, lon) grid cell the
# Pearson r between the local driver anomaly time series and the
# local Chl-a anomaly time series across all lags -12…+12 months,
# then retain:
#   peak_r   : r value at the lag of maximum |r|
#   peak_lag : the lag (months) at which |r| is maximum
# The result is two global maps per source (peak_r and peak_lag)
# laid out as rows = sources (OBS first, then models) and two
# column groups (peak_r left, peak_lag right).
#
# For SST a second paired figure is produced with AMO+ENSO+PDO
# removed cell-by-cell before correlation, to be placed alongside
# the raw SST figure in the supplemental.
# ============================================================

def _pointwise_peak_lag(
    drv_z: xr.DataArray,
    chl_z: xr.DataArray,
    lag_max: int = LAG_MAX,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Compute pointwise peak-lag Pearson r between driver and Chl-a.

    Both inputs must be (time, lat, lon) DataArrays on the same grid,
    already deseasonalised, linearly detrended, and standardised
    (output of _deseasonalise_zscore applied cell-by-cell — see
    _pointwise_deseasonalise_zscore below).

    At each (lat, lon) cell the Pearson r is computed at every lag
    k ∈ [-lag_max, +lag_max] months.  The lag with the largest |r|
    is retained.

    Sign convention: positive lag = driver leads Chl-a.

    Returns
    -------
    peak_r   : (lat, lon) DataArray — r at the peak lag
    peak_lag : (lat, lon) DataArray — lag (months) of peak |r|
    """
    drv_z, chl_z = xr.align(drv_z, chl_z, join="inner")
    D = drv_z.values   # (T, nlat, nlon)
    C = chl_z.values
    T, nlat, nlon = D.shape

    lags = np.arange(-lag_max, lag_max + 1)
    peak_r   = np.full((nlat, nlon), np.nan, dtype=float)
    peak_lag = np.full((nlat, nlon), np.nan, dtype=float)

    for j in range(nlat):
        for i in range(nlon):
            d = D[:, j, i]
            c = C[:, j, i]
            best_r   = np.nan
            best_lag = np.nan
            best_abs = -1.0
            for k in lags:
                if k >= 0:
                    x = d[:T - k] if k > 0 else d
                    y = c[k:]     if k > 0 else c
                else:
                    kk = -k
                    x = d[kk:]
                    y = c[:T - kk]
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() < 24:
                    continue
                r = float(np.corrcoef(x[ok], y[ok])[0, 1])
                if abs(r) > best_abs:
                    best_abs = abs(r)
                    best_r   = r
                    best_lag = float(k)
            peak_r[j, i]   = best_r
            peak_lag[j, i] = best_lag

    coords = {"lat": drv_z.lat, "lon": drv_z.lon}
    return (
        xr.DataArray(peak_r,   dims=("lat", "lon"), coords=coords,
                     name="peak_r"),
        xr.DataArray(peak_lag, dims=("lat", "lon"), coords=coords,
                     name="peak_lag"),
    )


def _pointwise_deseasonalise_zscore(
    ts: xr.DataArray,
) -> xr.DataArray:
    """Apply deseasonalise → linear detrend → z-score at every grid cell.

    Input : (time, lat, lon) DataArray on the common grid.
    Output: same shape, each cell standardised independently.

    The same three-step pipeline used for basin-mean series
    (Section 2.6 of the manuscript) is applied here cell-by-cell so
    that the pointwise correlation maps are directly comparable to the
    basin-mean lag-correlation curves of Figs 9-11.
    """
    ts = snap_to_month_start(ts)
    # 1. Deseasonalise: subtract monthly climatology at each cell
    clim = ts.groupby("time.month").mean("time")
    anom = ts.groupby("time.month") - clim

    # 2. Linear detrend: remove least-squares line at each cell
    #    Vectorised over space using numpy broadcasting.
    vals = np.asarray(anom.values, dtype=float)   # (T, nlat, nlon)
    T = vals.shape[0]
    t_idx = np.arange(T, dtype=float)
    t_c   = t_idx - t_idx.mean()                  # centre for numerical stability

    # OLS slope at each cell (intercept not needed — centred t)
    num   = np.nansum(t_c[:, None, None] * vals, axis=0)   # (nlat, nlon)
    denom = np.nansum(t_c ** 2)                            # scalar
    slope = num / denom if denom != 0 else np.zeros_like(num)
    trend = slope[None, :, :] * t_c[:, None, None]
    detrended = vals - trend
    detrended[~np.isfinite(vals)] = np.nan

    # 3. Standardise: divide by std at each cell (time axis)
    std = np.nanstd(detrended, axis=0, ddof=0)            # (nlat, nlon)
    std[std == 0] = np.nan
    normed = detrended / std[None, :, :]

    return anom.copy(data=normed)


def _ym_key(da: xr.DataArray) -> np.ndarray:
    """Return integer year*100+month for each time step of da.

    Works for both cftime and numpy datetime64 / pandas Timestamp
    time coordinates.  This is the root cause of the all-NaN right
    panels: xr.align(join="inner") matches on coordinate *values*,
    but model axes use cftime objects and the OBS-derived index uses
    pandas Timestamps, so the inner join finds zero matches.
    Matching on year*100+month integers is calendar-agnostic.
    """
    times = da.time.values
    try:
        # cftime objects have .year / .month attributes
        years  = np.array([int(t.year)  for t in times], dtype=np.int32)
        months = np.array([int(t.month) for t in times], dtype=np.int32)
    except AttributeError:
        import pandas as _pd
        dti    = _pd.DatetimeIndex(times)
        years  = dti.year.to_numpy(dtype=np.int32)
        months = dti.month.to_numpy(dtype=np.int32)
    return years * 100 + months


def _pointwise_regress_out(
    target: xr.DataArray,
    predictor_1d: xr.DataArray,
) -> xr.DataArray:
    """Remove the linear influence of a 1-D index from a 3-D field.

    target       : (time, lat, lon) — already deseasonalised/detrended/z-scored
    predictor_1d : (time,)          — the climate index, same preprocessing

    Alignment is done by matching year*100+month integer keys rather
    than coordinate objects, so cftime model axes and pandas-Timestamp
    OBS axes are handled correctly (xr.align on coordinate values fails
    when the two calendars produce non-identical objects, causing the
    right-column panels to appear as all-NaN).

    At each cell, fits beta by OLS and returns target - beta*predictor.
    Vectorised over the spatial dimensions.
    """
    ym_t = _ym_key(target)        # (T_t,)  int keys for target
    ym_p = _ym_key(predictor_1d)  # (T_p,)  int keys for predictor

    # Common keys present in both, in sorted (chronological) order
    common = np.intersect1d(ym_t, ym_p)
    if common.size < 12:
        logger.warning(
            f"  _pointwise_regress_out: only {common.size} common "
            f"(year,month) steps; returning target unchanged"
        )
        return target

    # Integer-position indices in each array for the common keys
    t_idx = np.searchsorted(np.sort(ym_t), common)
    t_idx = np.argsort(ym_t)[np.searchsorted(np.sort(ym_t), common)]
    p_idx = np.argsort(ym_p)[np.searchsorted(np.sort(ym_p), common)]

    T_full = np.asarray(target.values,       dtype=float)  # (T_t, nlat, nlon)
    p_full = np.asarray(predictor_1d.values, dtype=float)  # (T_p,)

    T_sub  = T_full[t_idx, :, :]   # (T_c, nlat, nlon)
    p_sub  = p_full[p_idx]          # (T_c,)

    ok = np.isfinite(p_sub)
    if ok.sum() < 12:
        logger.warning(
            "  _pointwise_regress_out: <12 finite predictor values; "
            "returning target unchanged"
        )
        return target

    p_ok = p_sub[ok]
    T_ok = T_sub[ok, :, :]

    # OLS beta at every cell simultaneously (no intercept — both series
    # are already zero-mean after the detrend step)
    pp   = float(np.dot(p_ok, p_ok))
    beta = np.einsum("t,tjk->jk", p_ok, T_ok) / pp        # (nlat, nlon)

    # Build a full-length predictor aligned to the target time axis
    # (NaN at time steps not in the common window so those residuals
    # fall back to the original target value)
    p_aligned = np.full(T_full.shape[0], np.nan, dtype=float)
    p_aligned[t_idx] = p_sub

    resid = T_full - beta[None, :, :] * p_aligned[:, None, None]
    # Restore NaN where the original field was NaN
    resid[~np.isfinite(T_full)] = np.nan
    # Where predictor was NaN (no common key), keep original value
    no_pred = ~np.isfinite(p_aligned)
    resid[no_pred, :, :] = T_full[no_pred, :, :]

    return target.copy(data=resid)


def make_figS_seaice_chl_polar_corr(
    obs_chl_ts:      xr.DataArray,
    obs_siconc_ts:   xr.DataArray | None,
    model_chl_ts:    dict[str, xr.DataArray],
    model_siconc_ts: dict[str, xr.DataArray],
    common_grid:     xr.DataArray,
    period:          slice = LAG_PERIOD,
) -> None:
    """Pointwise siconc-Chl-a lag-0 correlation maps, both poles.

    Layout: N_sources rows x 2 columns
      Col 1 — Southern Ocean (SouthPolarStereo, 50-80S)
      Col 2 — Arctic         (NorthPolarStereo, 60-90N)

    Each cell: filled contourf of Pearson r at lag 0 between
    deseasonalised, detrended, standardised siconc and Chl-a
    anomalies.  Thin black zero contour overlaid.
    Significance hatching (////) on non-significant cells (p >= 0.05,
    N_eff t-test).

    Output: figS_seaice_chl_polar_corr.png
    """
    from scipy import stats as _stats

    logger.info(
        "=== Fig S_seaice_chl_polar_corr: lag-0 siconc-Chl-a "
        "correlation, both poles ==="
    )

    # Sources: OBS first then models
    sources: list[tuple[str, xr.DataArray | None, xr.DataArray]] = []
    if obs_siconc_ts is not None:
        sources.append(("OBS", obs_siconc_ts, obs_chl_ts))
    for model in MODELS:
        if model in model_siconc_ts and model in model_chl_ts:
            sources.append((model,
                            model_siconc_ts[model],
                            model_chl_ts[model]))

    if not sources:
        logger.warning(
            "  No sources with both siconc and chl; skipping")
        return

    n_src = len(sources)
    lats  = common_grid.lat.values
    lons  = common_grid.lon.values
    lat2d, _ = np.meshgrid(lats, lons, indexing="ij")

    regions = {
        "Southern": {
            "lat_min": -78, "lat_max": -45,
            "proj":    ccrs.SouthPolarStereo(),
            "extent":  [-180, 180, -78, -43],
        },
        "Arctic": {
            "lat_min":  50, "lat_max":  90,
            "proj":     ccrs.NorthPolarStereo(),
            "extent":   [-180, 180, 48, 90],
        },
    }
    region_list = list(regions.keys())   # ["Southern", "Arctic"]

    def _r_at_lag0(
        sic_ts: xr.DataArray,
        chl_ts: xr.DataArray,
        reg_mask: np.ndarray,
        interpolate_chl: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise Pearson r and p-value (N_eff t-test) at lag 0.

        interpolate_chl : if True, linearly fill short NaN gaps
            (<=3 months) in the raw Chl-a field before preprocessing.
            Should be True for OBS (satellite gaps from polar night /
            cloud cover) and False for models (no missing data).
        """
        sic_re = regrid_timeseries(
            sic_ts.sel(time=period), common_grid)
        chl_re = regrid_timeseries(
            chl_ts.sel(time=period), common_grid)
        # Interpolate short gaps in OBS Chl-a before deseasonalisation
        if interpolate_chl:
            chl_re = _interp_time_gaps(chl_re, max_gap=3)
        # Use _deseas_detrend_zscore (finite-only std) so that polar
        # cells with winter NaN gaps are not masked out
        sic_z  = _deseas_detrend_zscore(sic_re)
        chl_z  = _deseas_detrend_zscore(chl_re)

        ym_s = _ym_key(sic_z)
        ym_c = _ym_key(chl_z)
        common_ym = np.intersect1d(ym_s, ym_c)
        si = np.argsort(ym_s)[np.searchsorted(np.sort(ym_s), common_ym)]
        ci = np.argsort(ym_c)[np.searchsorted(np.sort(ym_c), common_ym)]

        S = np.asarray(sic_z.values, dtype=float)[si]
        C = np.asarray(chl_z.values, dtype=float)[ci]
        nlat, nlon = S.shape[1], S.shape[2]

        # Coverage mask: require >= 24 valid paired (siconc, Chl-a)
        # months. This is the actual constraint used in the correlation
        # loop. The old 102-month threshold was applied to the
        # deseasonalised Chl-a which has NaN in polar winter months
        # (no climatology), giving an artificially low count that
        # excluded almost all OBS cells. The minimum of 24 paired
        # finite months (approximately 1.5 years of summer data across
        # 17 years of record) is sufficient for a meaningful r estimate
        # with the N_eff correction applied.
        # MIN_PAIRED: minimum finite paired months required.
        # 24 = approximately 1.4 years of summer data across 17 years,
        # sufficient for a stable Pearson r estimate.
        # MIN_NEFF: minimum effective sample size after autocorrelation
        # correction — prevents showing correlations from highly
        # autocorrelated short series where N_eff collapses to <5.
        MIN_PAIRED = 24
        MIN_NEFF   = 8
        r_map  = np.full((nlat, nlon), np.nan)
        pv_map = np.full((nlat, nlon), np.nan)

        for j in range(nlat):
            for i in range(nlon):
                if not reg_mask[j, i]:
                    continue
                s = S[:, j, i]
                c = C[:, j, i]
                ok = np.isfinite(s) & np.isfinite(c)
                n  = ok.sum()
                if n < MIN_PAIRED:
                    continue
                r = float(np.corrcoef(s[ok], c[ok])[0, 1])
                # N_eff (Bretherton 1999): account for temporal
                # autocorrelation reducing effective sample size
                r1s = float(np.corrcoef(s[ok][:-1], s[ok][1:])[0,1])
                r1c = float(np.corrcoef(c[ok][:-1], c[ok][1:])[0,1])
                d   = 1.0 + r1s * r1c
                neff = int(n * (1 - r1s*r1c) / d if d > 0 else n)
                neff = max(3, neff)
                # Skip cells where N_eff is too small for a
                # meaningful significance test
                if neff < MIN_NEFF:
                    continue
                if abs(r) < 1.0:
                    t  = r * np.sqrt(neff-2) / np.sqrt(1 - r**2)
                    pv = 2.0 * float(_stats.t.sf(abs(t), df=neff-2))
                else:
                    pv = 0.0
                r_map[j, i]  = r
                pv_map[j, i] = pv
        return r_map, pv_map

    # Pre-compute maps for every source x region
    r_maps:  dict[str, dict[str, np.ndarray]] = {}
    pv_maps: dict[str, dict[str, np.ndarray]] = {}

    for src_label, sic_ts, chl_ts in sources:
        r_maps[src_label]  = {}
        pv_maps[src_label] = {}
        for rname, reg in regions.items():
            reg_mask = ((lat2d >= reg["lat_min"]) &
                        (lat2d <= reg["lat_max"]))
            logger.info(f"  [{src_label}] {rname} lag-0 corr...")
            try:
                r_m, pv_m = _r_at_lag0(sic_ts, chl_ts, reg_mask,
                                        interpolate_chl=(src_label == "OBS"))
                # Mask outside region
                r_m[ ~reg_mask] = np.nan
                pv_m[~reg_mask] = np.nan
                r_maps[src_label][rname]  = r_m
                pv_maps[src_label][rname] = pv_m
                logger.success(
                    f"  [{src_label}] {rname}: "
                    f"median r={np.nanmedian(r_m):.3f}"
                )
            except Exception as e:
                import traceback as _tb
                logger.warning(
                    f"  [{src_label}] {rname} failed: {e}")
                logger.warning(
                    _tb.format_exc().replace("<", "\\<"))

    # ── Drawing helper ───────────────────────────────────────────────
    r_levels = np.linspace(-0.7, 0.7, 15)
    r_cmap   = "RdBu_r"

    def _draw_corr_ax(ax, rname, src_label, letter):
        reg  = regions[rname]
        r_m  = r_maps[src_label].get(rname)
        pv_m = pv_maps[src_label].get(rname)
        ax.set_extent(reg["extent"], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=4)
        try:
            ax.coastlines(linewidth=0.4, zorder=5)
        except Exception:
            pass
        ax.gridlines(linestyle=":", linewidth=0.3, alpha=0.4)
        ax.text(0.02, 0.97, f"({letter})",
                transform=ax.transAxes, fontsize=10,
                fontweight="bold", va="top",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        if r_m is None or not np.any(np.isfinite(r_m)):
            ax.text(0.5, 0.5, "no data",
                    transform=ax.transAxes, ha="center",
                    va="center", fontsize=9, color="grey")
            return None
        cf = ax.contourf(lons, lats, r_m, levels=r_levels,
                         cmap=r_cmap, extend="both",
                         transform=ccrs.PlateCarree())
        ax.contour(lons, lats, r_m, levels=[0], colors="k",
                   linewidths=0.6, transform=ccrs.PlateCarree())
        if pv_m is not None:
            nonsig = np.where(np.isfinite(pv_m) & (pv_m >= 0.05),
                              1.0, np.nan)
            if np.any(np.isfinite(nonsig)):
                ax.contourf(lons, lats, nonsig,
                            levels=[0.5, 1.5],
                            colors=["white"], alpha=0.45,
                            transform=ccrs.PlateCarree())
        return cf

    def _shared_r_cbar(fig, cf):
        cax = fig.add_axes([0.92, 0.12, 0.015, 0.74])
        cb  = fig.colorbar(cf, cax=cax, extend="both")
        cb.set_label("Pearson r (lag 0)", fontsize=16)
        cb.ax.tick_params(labelsize=16)
        cb.set_ticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])

    base_note = (
        "Filled: r (RdBu_r, +-0.7).  White overlay: p>=0.05 (Neff t-test).  "
        "Black contour: r=0.  "
        "Both fields deseasonalised, detrended, standardised per cell. "
        "Period: 1998-2014."
    )

    # ── Figure set 1: OBS + individual models (one per region) ──────
    for rname in region_list:
        proj    = regions[rname]["proj"]
        n_panels = len(sources)       # OBS + up to 7 models
        ncols_i  = 2
        nrows_i  = int(np.ceil(n_panels / ncols_i))

        fig_i, axes_i = plt.subplots(
            nrows_i, ncols_i,
            figsize=(9.5, 4.6 * nrows_i),
            subplot_kw={"projection": proj},
            squeeze=False,
        )
        fig_i.subplots_adjust(
            left=0.04, right=0.88,
            bottom=0.04, top=0.90,
            hspace=0.20, wspace=0.06,
        )
        last_cf = None
        for idx, (src_label, _, __) in enumerate(sources):
            ax = axes_i[idx // ncols_i, idx % ncols_i]
            cf = _draw_corr_ax(ax, rname, src_label,
                               chr(ord("a") + idx))
            if cf is not None:
                last_cf = cf
            ax.set_title(src_label, fontsize=11,
                         fontweight="bold", pad=4)
        for idx in range(len(sources), nrows_i * ncols_i):
            axes_i[idx // ncols_i, idx % ncols_i].set_visible(False)
        if last_cf is not None:
            _shared_r_cbar(fig_i, last_cf)
        fig_i.suptitle(
            f"{rname} -- OBS and individual model siconc-Chl-a "
            f"correlation at lag 0\n{base_note}",
            fontsize=8.5, y=0.97,
        )
        fname = f"figS_seaice_chl_polar_indiv_{rname}.png"
        fig_i.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig_i)
        logger.success(f"  wrote {fname}")

    # ── Figure set 2: Ensemble mean + std (one per region) ──────────
    for rname in region_list:
        proj = regions[rname]["proj"]
        ext  = regions[rname]["extent"]

        model_rs = [
            r_maps[sl][rname]
            for sl, _, __ in sources
            if sl != "OBS" and rname in r_maps.get(sl, {})
            and r_maps[sl][rname] is not None
        ]
        n_m = len(model_rs)
        if n_m == 0:
            logger.warning(f"  No model maps for {rname}; skip ens")
            continue

        stack    = np.stack(model_rs, axis=0)
        ens_mean = np.nanmean(stack, axis=0)
        ens_std  = np.nanstd( stack, axis=0)

        fig_e, axes_e = plt.subplots(
            1, 2,
            figsize=(12, 6.5),
            subplot_kw={"projection": proj},
            squeeze=False,
        )
        fig_e.subplots_adjust(
            left=0.04, right=0.88,
            bottom=0.06, top=0.88,
            wspace=0.06,
        )
        # Panel (a): ensemble mean
        ax_mn = axes_e[0, 0]
        ax_mn.set_extent(ext, crs=ccrs.PlateCarree())
        ax_mn.add_feature(cfeature.LAND,
                          facecolor="lightgray", zorder=4)
        try:
            ax_mn.coastlines(linewidth=0.4, zorder=5)
        except Exception:
            pass
        ax_mn.gridlines(linestyle=":", linewidth=0.3, alpha=0.4)
        cf_mn = ax_mn.contourf(lons, lats, ens_mean,
                               levels=r_levels, cmap=r_cmap,
                               extend="both",
                               transform=ccrs.PlateCarree())
        ax_mn.contour(lons, lats, ens_mean,
                      levels=[0], colors="k", linewidths=0.6,
                      transform=ccrs.PlateCarree())
        ax_mn.set_title(f"(a) Ensemble mean ({n_m} models)",
                        fontsize=12, fontweight="bold", pad=5)

        # Panel (b): ensemble std
        ax_sd = axes_e[0, 1]
        ax_sd.set_extent(ext, crs=ccrs.PlateCarree())
        ax_sd.add_feature(cfeature.LAND,
                          facecolor="lightgray", zorder=4)
        try:
            ax_sd.coastlines(linewidth=0.4, zorder=5)
        except Exception:
            pass
        ax_sd.gridlines(linestyle=":", linewidth=0.3, alpha=0.4)
        std_max = min(float(np.nanpercentile(ens_std, 95)), 0.5)
        cf_sd = ax_sd.contourf(lons, lats, ens_std,
                               levels=np.linspace(0, std_max, 11),
                               cmap="YlOrRd", extend="max",
                               transform=ccrs.PlateCarree())
        ax_sd.set_title(f"(b) Ensemble spread ({n_m} models)",
                        fontsize=12, fontweight="bold", pad=5)

        # Colourbars
        cax1 = fig_e.add_axes([0.90, 0.15, 0.013, 0.70])
        cb1  = fig_e.colorbar(cf_mn, cax=cax1, extend="both")
        cb1.set_label("Pearson r (mean)", fontsize=16)
        cb1.ax.tick_params(labelsize=16)
        cb1.set_ticks([-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6])

        cax2 = fig_e.add_axes([0.955, 0.15, 0.013, 0.70])
        cb2  = fig_e.colorbar(cf_sd, cax=cax2, extend="max")
        cb2.set_label("Std of r", fontsize=16)
        cb2.ax.tick_params(labelsize=16)

        fig_e.suptitle(
            f"{rname} -- Ensemble mean and spread of siconc-Chl-a "
            f"Pearson r at lag 0\n"
            f"Panel (a): mean r across {n_m} CMIP6 models.  "
            f"Panel (b): inter-model std.  {base_note}",
            fontsize=8.5, y=0.97,
        )
        fname = f"figS_seaice_chl_polar_ens_{rname}.png"
        fig_e.savefig(OUT_DIR / fname, dpi=150,
                      bbox_inches="tight")
        plt.close(fig_e)
        logger.success(f"  wrote {fname}")

    logger.success("  wrote figS_seaice_chl_polar_corr.png")


def make_figS_spatial_peak_lag(
    obs_chl_ts:     xr.DataArray,
    obs_sst_ts:     xr.DataArray | None,
    obs_mld_ts:     xr.DataArray | None,
    obs_siconc_ts:  xr.DataArray | None,
    model_chl_ts:   dict[str, xr.DataArray],
    model_sst_ts:   dict[str, xr.DataArray],
    model_mld_ts:   dict[str, xr.DataArray],
    model_siconc_ts: dict[str, xr.DataArray],
    common_grid:    xr.DataArray,
    indices:        dict[str, xr.DataArray] | None = None,
    period:         slice = LAG_PERIOD,
) -> None:
    """Supplemental: pointwise peak-lag correlation maps for SST, MLD, siconc.

    For each driver produces one figure with rows = sources (OBS first,
    then models in MODELS order) and two columns:
      left  — peak Pearson r  (diverging colourmap, ±0.7)
      right — lag at peak |r| (sequential colourmap, −12…+12 months)

    Files produced
    --------------
    figS_spatial_peak_lag_sst.png      — SST raw
    figS_spatial_peak_lag_sst_amo.png  — SST with AMO+ENSO+PDO removed
                                         (only if `indices` is non-empty)
    figS_spatial_peak_lag_mld.png      — MLD
    figS_spatial_peak_lag_siconc.png   — sea-ice concentration

    Note on computation time
    ------------------------
    The inner loop over (nlat, nlon) cells is pure Python; on a 1°×1°
    global grid (180×360 = 64 800 cells) with 25 lags this takes
    ~2–4 minutes per source on a single core.  With 8 sources (OBS +
    7 models) per driver the total runtime is ~30–60 minutes.  On
    Levante this is best submitted as a batch job rather than run
    interactively.  The function logs progress every 10 latitudes so
    the user can monitor throughput.
    """
    logger.info("=== Fig S_spatial_peak_lag: pointwise peak-lag maps ===")

    # ------------------------------------------------------------------
    # Inner helper: regrid both fields, deseasonalise/detrend/zscore,
    # optionally remove indices, then call _pointwise_peak_lag.
    # Returns (peak_r, peak_lag) or (None, None) on failure.
    # ------------------------------------------------------------------
    def _compute_maps(
        drv_ts: xr.DataArray,
        chl_ts: xr.DataArray,
        remove_indices: list[str],
        label: str,
    ) -> tuple[xr.DataArray | None, xr.DataArray | None]:
        try:
            logger.info(f"    [{label}] regridding...")
            drv_re = regrid_timeseries(drv_ts.sel(time=period), common_grid)
            chl_re = regrid_timeseries(chl_ts.sel(time=period), common_grid)

            logger.info(f"    [{label}] deseasonalising/detrending/z-scoring...")
            drv_z = _pointwise_deseasonalise_zscore(drv_re)
            chl_z = _pointwise_deseasonalise_zscore(chl_re)

            if remove_indices and indices:
                for idx_name in remove_indices:
                    if idx_name in indices:
                        logger.info(
                            f"    [{label}] removing {idx_name}..."
                        )
                        drv_z = _pointwise_regress_out(drv_z, indices[idx_name])
                        chl_z = _pointwise_regress_out(chl_z, indices[idx_name])

            logger.info(f"    [{label}] computing pointwise peak-lag...")
            # Log progress every 10 latitudes inside _pointwise_peak_lag
            # by monkey-patching is impractical here; instead we rely on
            # the outer per-source logger.info above.
            pr, pl = _pointwise_peak_lag(drv_z, chl_z)
            logger.success(f"    [{label}] done")
            return pr, pl
        except Exception as e:
            import traceback as _tb
            logger.warning(f"    [{label}] failed: {e}")
            logger.warning(_tb.format_exc().replace("<", "\\<"))
            return None, None

    # ------------------------------------------------------------------
    # Inner helper: draw one complete figure given a dict of maps.
    # maps: {source_label: (peak_r, peak_lag)} — OBS first.
    # ------------------------------------------------------------------
    def _draw_peak_lag_figure(
        maps: dict[str, tuple[xr.DataArray, xr.DataArray]],
        driver_name: str,
        filename: str,
        extra_subtitle: str = "",
    ) -> None:
        sources = list(maps.keys())
        n_src   = len(sources)
        if n_src == 0:
            logger.warning(f"  No maps for {driver_name}; skipping {filename}")
            return

        ncols = 2          # left = peak_r, right = peak_lag
        nrows = n_src

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(13, 3.5 * nrows),
            subplot_kw={"projection": ccrs.Robinson(central_longitude=0)},
            squeeze=False,
        )
        fig.subplots_adjust(
            left=0.04, right=0.88,
            bottom=0.06, top=0.96,
            hspace=0.12, wspace=0.04,
        )

        from matplotlib.colors import TwoSlopeNorm, BoundaryNorm
        import matplotlib.cm as mcm

        # Colour scales
        r_norm    = TwoSlopeNorm(vmin=-0.7, vcenter=0.0, vmax=0.7)
        r_cmap    = "RdBu_r"
        lag_bounds = np.arange(-LAG_MAX - 0.5, LAG_MAX + 1.5, 1.0)
        lag_norm   = BoundaryNorm(lag_bounds, ncolors=256)
        lag_cmap   = "coolwarm"

        im_r = im_lag = None

        for row_idx, src in enumerate(sources):
            pr, pl = maps[src]
            ax_r   = axes[row_idx, 0]
            ax_lag = axes[row_idx, 1]

            for ax, data, norm, cmap, col_title in [
                (ax_r,   pr, r_norm,   r_cmap,  "Peak |r|"),
                (ax_lag, pl, lag_norm, lag_cmap, "Lag at peak (months)"),
            ]:
                im = ax.pcolormesh(
                    data.lon, data.lat, data.values,
                    norm=norm, cmap=cmap,
                    transform=ccrs.PlateCarree(),
                    rasterized=True,
                )
                ax.add_feature(cfeature.LAND,
                               facecolor="lightgray", zorder=2)
                try:
                    ax.coastlines(linewidth=0.3, zorder=3)
                except Exception:
                    pass
                ax.gridlines(linestyle=":", linewidth=0.2, alpha=0.4)

                if row_idx == 0:
                    ax.set_title(col_title, fontsize=10, fontweight="bold")

                if ax is ax_r:
                    im_r = im
                else:
                    im_lag = im

            # Row label: source name on the left of the peak_r panel
            ax_r.text(
                -0.04, 0.5,
                src,
                transform=ax_r.transAxes,
                fontsize=8, fontweight="bold",
                ha="right", va="center", rotation=90,
            )

            # Panel letters: (a)(b) first row, (c)(d) second, …
            for col_idx, ax in enumerate([ax_r, ax_lag]):
                letter = chr(ord("a") + row_idx * 2 + col_idx)
                ax.text(
                    0.01, 0.97, f"({letter})",
                    transform=ax.transAxes, fontsize=9,
                    fontweight="bold", va="top",
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="white", alpha=0.7,
                              edgecolor="none"),
                )

        # Shared colourbars on the right
        if im_r is not None:
            cax_r = fig.add_axes([0.89, 0.55, 0.012, 0.38])
            cb_r  = fig.colorbar(im_r, cax=cax_r, extend="both")
            cb_r.set_label("Peak Pearson r", fontsize=9)
            cb_r.ax.tick_params(labelsize=8)

        if im_lag is not None:
            cax_l = fig.add_axes([0.89, 0.08, 0.012, 0.38])
            cb_l  = fig.colorbar(im_lag, cax=cax_l, extend="neither")
            cb_l.set_label("Lag (months, + = driver leads)", fontsize=9)
            cb_l.set_ticks(np.arange(-LAG_MAX, LAG_MAX + 1, 3))
            cb_l.ax.tick_params(labelsize=8)

        subtitle = (
            f"Pointwise peak-lag Pearson r between {driver_name} anomalies "
            f"and Chl-$a$ anomalies\n"
            f"Period: {period.start}–{period.stop}; "
            f"deseasonalised, linearly detrended, standardised per cell; "
            f"positive lag = {driver_name} leads Chl-$a$"
        )
        if extra_subtitle:
            subtitle += f"\n{extra_subtitle}"
        fig.suptitle(subtitle, fontsize=8, y=0.995)

        out_path = OUT_DIR / filename
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"  wrote {out_path.name}")

    # ------------------------------------------------------------------
    # SST maps (raw)
    # ------------------------------------------------------------------
    logger.info("  === SST peak-lag maps (raw) ===")
    sst_maps_raw: dict[str, tuple] = {}
    if obs_sst_ts is not None:
        pr, pl = _compute_maps(obs_sst_ts, obs_chl_ts, [], "OBS")
        if pr is not None:
            sst_maps_raw["OBS"] = (pr, pl)
    for model in MODELS:
        if model in model_sst_ts and model in model_chl_ts:
            pr, pl = _compute_maps(
                model_sst_ts[model], model_chl_ts[model], [], model
            )
            if pr is not None:
                sst_maps_raw[model] = (pr, pl)
    _draw_peak_lag_figure(
        sst_maps_raw, "SST",
        filename="figS_spatial_peak_lag_sst.png",
    )

    # ------------------------------------------------------------------
    # SST maps (AMO + ENSO + PDO removed)
    # ------------------------------------------------------------------
    if indices:
        logger.info("  === SST peak-lag maps (AMO+ENSO+PDO removed) ===")
        all_idx = [n for n in ("AMO", "NINO34", "PDO") if n in indices]
        sst_maps_amo: dict[str, tuple] = {}
        if obs_sst_ts is not None:
            pr, pl = _compute_maps(
                obs_sst_ts, obs_chl_ts, all_idx, "OBS"
            )
            if pr is not None:
                sst_maps_amo["OBS"] = (pr, pl)
        for model in MODELS:
            if model in model_sst_ts and model in model_chl_ts:
                pr, pl = _compute_maps(
                    model_sst_ts[model], model_chl_ts[model],
                    all_idx, model
                )
                if pr is not None:
                    sst_maps_amo[model] = (pr, pl)
        _draw_peak_lag_figure(
            sst_maps_amo, "SST",
            filename="figS_spatial_peak_lag_sst_amo.png",
            extra_subtitle=(
                f"AMO + Niño 3.4 + PDO regressed out cell-by-cell "
                f"before correlation"
            ),
        )

    # ------------------------------------------------------------------
    # MLD maps
    # ------------------------------------------------------------------
    logger.info("  === MLD peak-lag maps ===")
    mld_maps: dict[str, tuple] = {}
    if obs_mld_ts is not None:
        pr, pl = _compute_maps(obs_mld_ts, obs_chl_ts, [], "OBS")
        if pr is not None:
            mld_maps["OBS"] = (pr, pl)
    for model in MODELS:
        if model in model_mld_ts and model in model_chl_ts:
            pr, pl = _compute_maps(
                model_mld_ts[model], model_chl_ts[model], [], model
            )
            if pr is not None:
                mld_maps[model] = (pr, pl)
    _draw_peak_lag_figure(
        mld_maps, "MLD",
        filename="figS_spatial_peak_lag_mld.png",
    )

    # ------------------------------------------------------------------
    # siconc: polar correlation maps (dedicated function)
    # ------------------------------------------------------------------
    # Replaced by make_figS_seaice_chl_polar_corr which shows
    # lag-0 pointwise r in polar stereographic projections,
    # one column per pole, one row per source.
    make_figS_seaice_chl_polar_corr(
        obs_chl_ts=obs_chl_ts,
        obs_siconc_ts=obs_siconc_ts,
        model_chl_ts=model_chl_ts,
        model_siconc_ts=model_siconc_ts,
        common_grid=common_grid,
    )


# ============================================================
# Supplemental: regional contourf SST-Chl-a correlation maps
# raw vs climate-index removed
# ============================================================
# Three paired figures (left = raw, right = index removed):
#   figS_regional_corr_atlantic.png  — N. Atlantic, AMO removed
#   figS_regional_corr_pacific.png   — Pacific,     ENSO removed
#   figS_regional_corr_so.png        — Southern Ocean, ENSO removed
#
# Each panel shows the pointwise Pearson r between the local
# SST anomaly and the local Chl-a anomaly at lag 0 (simultaneous
# correlation) as a contourf map with significance stippling
# (|r| > r_crit at p < 0.05 based on N_eff).  Both fields are
# deseasonalised, linearly detrended, and standardised per cell
# before correlation, using _pointwise_deseasonalise_zscore.
# ============================================================

# Regional extent definitions: (lat_min, lat_max, lon_min, lon_max)
# in PlateCarree coordinates (lon in 0-360).
_REGION_EXTENTS = {
    "Atlantic":     (  0,  70, 280, 360),   # North Atlantic
    "Pacific":      (-60,  60, 130, 280),   # Full Pacific
    "Southern":     (-80, -30,   0, 360),   # Southern Ocean
}

# For cartopy set_extent: (lon_min, lon_max, lat_min, lat_max)
_CARTOPY_EXTENTS = {
    "Atlantic":     (-80,   0,   0,  70),
    "Pacific":      (130, 280, -60,  60),
    "Southern":     (  0, 360, -80, -30),
}

# Central longitude for each region's projection
_CENTRAL_LON = {
    "Atlantic": -40,
    "Pacific":  200,
    "Southern":   0,
}


def _pointwise_lag_corr(
    drv_z: xr.DataArray,
    chl_z: xr.DataArray,
    lag: int = 0,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Pointwise Pearson r and significance mask at a specified lag.

    Both inputs are (time, lat, lon) DataArrays, already
    deseasonalised, detrended, and standardised per cell.

    Sign convention: positive lag = driver leads Chl-a.
    At lag k > 0 the driver is shifted forward by k months relative
    to Chl-a (driver[:-k] vs chl[k:]).
    At lag k < 0 Chl-a leads the driver (driver[|k|:] vs chl[:-|k|]).

    Parameters
    ----------
    drv_z : (time, lat, lon) — driver anomaly field
    chl_z : (time, lat, lon) — Chl-a anomaly field
    lag   : int — lag in months (default 0 = simultaneous)

    Returns
    -------
    r_map   : (lat, lon) — Pearson r at the specified lag
    sig_map : (lat, lon) bool — True where p < 0.05 based on N_eff
              (Bretherton et al. 1999, eq. 1 of the manuscript).
    """
    from scipy import stats as _stats

    # Align by year-month key (calendar-agnostic)
    ym_d = _ym_key(drv_z)
    ym_c = _ym_key(chl_z)
    common = np.intersect1d(ym_d, ym_c)
    if common.size < 24:
        coords = {"lat": drv_z.lat, "lon": drv_z.lon}
        nlat, nlon = len(drv_z.lat), len(drv_z.lon)
        return (
            xr.DataArray(np.full((nlat, nlon), np.nan),
                         dims=("lat", "lon"), coords=coords),
            xr.DataArray(np.zeros((nlat, nlon), dtype=bool),
                         dims=("lat", "lon"), coords=coords),
        )

    d_idx = np.argsort(ym_d)[np.searchsorted(np.sort(ym_d), common)]
    c_idx = np.argsort(ym_c)[np.searchsorted(np.sort(ym_c), common)]

    D_full = np.asarray(drv_z.values, dtype=float)   # (T, nlat, nlon)
    C_full = np.asarray(chl_z.values, dtype=float)

    D = D_full[d_idx, :, :]   # (T_c, nlat, nlon) — common time steps
    C = C_full[c_idx, :, :]
    T, nlat, nlon = D.shape

    # Apply lag: positive lag = driver leads Chl-a
    k = int(lag)
    if k > 0:
        if k >= T:
            D_lag = D[:0, :, :]
            C_lag = C[:0, :, :]
        else:
            D_lag = D[:T - k, :, :]   # driver at time t
            C_lag = C[k:,     :, :]   # chl    at time t + k
    elif k < 0:
        kk = -k
        if kk >= T:
            D_lag = D[:0, :, :]
            C_lag = C[:0, :, :]
        else:
            D_lag = D[kk:,     :, :]  # driver at time t + |k|
            C_lag = C[:T - kk, :, :]  # chl    at time t
    else:
        D_lag = D
        C_lag = C

    T_lag = D_lag.shape[0]
    r_arr   = np.full((nlat, nlon), np.nan, dtype=float)
    sig_arr = np.zeros((nlat, nlon), dtype=bool)

    for j in range(nlat):
        for i in range(nlon):
            d = D_lag[:, j, i]
            c = C_lag[:, j, i]
            ok = np.isfinite(d) & np.isfinite(c)
            n  = ok.sum()
            if n < 24:
                continue
            r = float(np.corrcoef(d[ok], c[ok])[0, 1])
            r_arr[j, i] = r

            # N_eff via lag-1 autocorrelations (Bretherton 1999)
            r1d = float(np.corrcoef(d[ok][:-1], d[ok][1:])[0, 1])
            r1c = float(np.corrcoef(c[ok][:-1], c[ok][1:])[0, 1])
            denom = 1.0 + r1d * r1c
            n_eff = max(3, int(n * (1.0 - r1d * r1c) / denom
                               if denom > 0 else n))

            if abs(r) < 1.0:
                t_stat = r * np.sqrt(n_eff - 2) / np.sqrt(1 - r ** 2)
                p_val  = 2.0 * float(
                    _stats.t.sf(abs(t_stat), df=n_eff - 2)
                )
                sig_arr[j, i] = p_val < 0.05

    coords = {"lat": drv_z.lat, "lon": drv_z.lon}
    return (
        xr.DataArray(r_arr,   dims=("lat", "lon"), coords=coords,
                     name=f"r_lag{k:+d}"),
        xr.DataArray(sig_arr, dims=("lat", "lon"), coords=coords,
                     name="significant"),
    )


def _extract_region(
    da: xr.DataArray,
    lat_min: float, lat_max: float,
    lon_min: float, lon_max: float,
) -> xr.DataArray:
    """Slice a (time, lat, lon) or (lat, lon) field to a regional box.

    Handles the wrap-around case (lon_min > lon_max) correctly by
    concatenating the two halves.
    """
    if lon_min <= lon_max:
        return da.sel(
            lat=slice(lat_min, lat_max),
            lon=slice(lon_min, lon_max),
        )
    else:
        left  = da.sel(lat=slice(lat_min, lat_max),
                       lon=slice(lon_min, 360))
        right = da.sel(lat=slice(lat_min, lat_max),
                       lon=slice(0, lon_max))
        return xr.concat([left, right], dim="lon")


def _draw_contourf_panel(
    ax,
    r_map: xr.DataArray,
    sig_map: xr.DataArray,
    title: str,
    region: str,
    vmax: float = 0.6,
    n_levels: int = 13,
) -> object:
    """Draw one contourf correlation panel on a cartopy axes.

    Returns the mappable for the shared colourbar.
    """
    import matplotlib.ticker as mticker

    lon  = r_map.lon.values
    lat  = r_map.lat.values
    rval = r_map.values

    levels = np.linspace(-vmax, vmax, n_levels)
    cmap   = plt.cm.RdBu_r

    cf = ax.contourf(
        lon, lat, rval,
        levels=levels,
        cmap=cmap,
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    # Black contour at r = 0
    ax.contour(
        lon, lat, rval,
        levels=[0],
        colors="k",
        linewidths=0.6,
        transform=ccrs.PlateCarree(),
    )

    # Stippling: dots where |r| is significant at p < 0.05
    sig = sig_map.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    sig_lons = lon2d[sig]
    sig_lats = lat2d[sig]
    if sig_lons.size > 0:
        ax.plot(
            sig_lons, sig_lats,
            "k.", ms=0.8, alpha=0.5,
            transform=ccrs.PlateCarree(),
            rasterized=True,
        )

    # Coastlines and land
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=4)
    try:
        ax.coastlines(linewidth=0.5, zorder=5)
    except Exception:
        pass

    # Grid lines
    gl = ax.gridlines(
        draw_labels=True,
        linestyle=":",
        linewidth=0.3,
        alpha=0.5,
        x_inline=False,
        y_inline=False,
    )
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 7}
    gl.ylabel_style = {"size": 7}

    # Set regional extent
    ext = _CARTOPY_EXTENTS[region]
    try:
        ax.set_extent(
            [ext[0], ext[1], ext[2], ext[3]],
            crs=ccrs.PlateCarree(),
        )
    except Exception:
        pass

    ax.set_title(title, fontsize=9, pad=3)
    return cf


def make_figS_regional_contourf_corr(
    obs_chl_ts:    xr.DataArray,
    obs_sst_ts:    xr.DataArray | None,
    model_chl_ts:  dict[str, xr.DataArray],
    model_sst_ts:  dict[str, xr.DataArray],
    common_grid:   xr.DataArray,
    indices:       dict[str, xr.DataArray] | None = None,
    period:        slice = LAG_PERIOD,
) -> None:
    """Three regional paired contourf figures: raw vs index removed.

    For each region the figure has two columns (raw | index removed)
    and nrows = 1 (OBS) + n_models rows.

    North Atlantic — AMO removed
    Pacific        — ENSO (Niño 3.4) removed
    Southern Ocean — ENSO (Niño 3.4) removed

    The correlation shown is the simultaneous (lag 0) pointwise
    Pearson r between local SST anomaly and local Chl-a anomaly.
    Significance stippling marks cells where |r| exceeds the p < 0.05
    threshold based on N_eff (Bretherton et al. 1999).

    Output files
    ------------
    figS_regional_corr_atlantic.png
    figS_regional_corr_pacific.png
    figS_regional_corr_so.png
    """
    logger.info(
        "=== Fig S_regional_corr: regional contourf SST-Chl-a maps ==="
    )

    if obs_sst_ts is None and not model_sst_ts:
        logger.warning(
            "  Fig S_regional_corr SKIPPED: no SST data available"
        )
        return

    idx = indices or {}

    # Map each region to the index name to remove
    region_index = {
        "Atlantic": "AMO",
        "Pacific":  "NINO34",
        "Southern": "NINO34",
    }
    region_title = {
        "Atlantic": "North Atlantic",
        "Pacific":  "Pacific",
        "Southern": "Southern Ocean",
    }
    index_label = {
        "AMO":    "AMO removed",
        "NINO34": "Niño 3.4 (ENSO) removed",
    }
    # Observed peak lags from Fig 9 (basin-mean lag-correlation curves).
    # The spatial maps are evaluated at the same lag so that the
    # spatial figure is directly comparable to the basin-mean result.
    # Sign convention: negative = Chl leads driver (same as Fig 9).
    #   Atlantic : Chl leads SST by 8 months  → lag = -8
    #   Pacific  : no significant obs peak; use lag 0 as reference
    #   Southern : sea-ice / ENSO signal peaks near lag +3
    #              (driver leads Chl); use lag = +3
    region_peak_lag = {
        "Atlantic": -8,
        "Pacific":   0,
        "Southern": +3,
    }
    region_filename = {
        "Atlantic": "figS_regional_corr_atlantic.png",
        "Pacific":  "figS_regional_corr_pacific.png",
        "Southern": "figS_regional_corr_so.png",
    }

    # Sources: OBS first, then models in order
    sources_all: list[tuple[str, xr.DataArray, xr.DataArray]] = []
    if obs_sst_ts is not None:
        sources_all.append(("OBS", obs_sst_ts, obs_chl_ts))
    for model in MODELS:
        if model in model_sst_ts and model in model_chl_ts:
            sources_all.append(
                (model, model_sst_ts[model], model_chl_ts[model])
            )

    if not sources_all:
        logger.warning("  No sources available; skipping")
        return

    n_src = len(sources_all)

    # ------------------------------------------------------------------
    # Pre-compute regridded, preprocessed fields for all sources once
    # (reused across the three region figures)
    # ------------------------------------------------------------------
    logger.info("  Pre-processing SST and Chl-a fields for all sources...")
    preprocessed: dict[str, dict] = {}   # {source: {raw, amo, nino34}}
    for src_label, sst_ts, chl_ts in sources_all:
        try:
            sst_re = regrid_timeseries(
                sst_ts.sel(time=period), common_grid
            )
            chl_re = regrid_timeseries(
                chl_ts.sel(time=period), common_grid
            )
            sst_z  = _pointwise_deseasonalise_zscore(sst_re)
            chl_z  = _pointwise_deseasonalise_zscore(chl_re)

            entry: dict = {"sst_raw": sst_z, "chl_raw": chl_z}

            # Partial correlation: remove the index from BOTH SST and
            # Chl-a before correlating. Removing from SST only leaves
            # the shared index-driven variance in Chl-a, so the
            # right-column panels look nearly identical to the raw ones.
            # Removing from both isolates the SST-Chl-a coupling that
            # is orthogonal to the index — i.e. the partial correlation
            # r(SST, Chl | index).
            if "AMO" in idx:
                entry["sst_amo"] = _pointwise_regress_out(
                    sst_z, idx["AMO"]
                )
                entry["chl_amo"] = _pointwise_regress_out(
                    chl_z, idx["AMO"]
                )
            # ENSO-removed SST and Chl-a
            if "NINO34" in idx:
                entry["sst_nino34"] = _pointwise_regress_out(
                    sst_z, idx["NINO34"]
                )
                entry["chl_nino34"] = _pointwise_regress_out(
                    chl_z, idx["NINO34"]
                )

            preprocessed[src_label] = entry
            logger.success(f"  [{src_label}] preprocessed")
        except Exception as e:
            import traceback as _tb
            logger.warning(f"  [{src_label}] preprocessing failed: {e}")
            logger.warning(_tb.format_exc().replace("<", "\\<"))

    if not preprocessed:
        logger.warning("  All sources failed preprocessing; skipping")
        return

    # ------------------------------------------------------------------
    # Loop over regions and produce one figure per region
    # ------------------------------------------------------------------
    for region, idx_name in region_index.items():
        logger.info(f"  Drawing {region_title[region]} figure...")

        lat_min, lat_max, lon_min, lon_max = _REGION_EXTENTS[region]
        sst_key_removed = (
            "sst_amo"    if idx_name == "AMO"    else
            "sst_nino34" if idx_name == "NINO34" else None
        )
        chl_key_removed = (
            "chl_amo"    if idx_name == "AMO"    else
            "chl_nino34" if idx_name == "NINO34" else None
        )
        has_removed = (
            sst_key_removed is not None
            and any(
                sst_key_removed in preprocessed.get(s, {})
                for s, _, _ in sources_all
            )
        )

        # Three columns: raw | partial (index removed from both) | difference
        ncols = 3 if has_removed else 1
        nrows = sum(1 for s, _, _ in sources_all if s in preprocessed)
        if nrows == 0:
            logger.warning(f"  {region}: no preprocessed sources; skipping")
            continue

        proj = ccrs.PlateCarree(
            central_longitude=_CENTRAL_LON[region]
        )
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 3.5 * nrows),
            subplot_kw={"projection": proj},
            squeeze=False,
        )
        fig.subplots_adjust(
            left=0.06, right=0.90,
            bottom=0.06, top=0.94,
            hspace=0.15, wspace=0.06,
        )

        # Column headers
        idx_removed_label = index_label.get(idx_name, idx_name)
        col_titles = [
            "Raw r(SST, Chl)",
            f"Partial r(SST, Chl | {idx_name})",
            f"Difference (raw − partial)",
        ]

        row_idx = 0
        last_cf = last_cf_diff = None
        for src_label, _, _ in sources_all:
            if src_label not in preprocessed:
                continue
            entry = preprocessed[src_label]
            sst_z_raw = entry["sst_raw"]
            chl_z_raw = entry["chl_raw"]

            # Slice raw fields to region
            sst_reg_raw = _extract_region(
                sst_z_raw, lat_min, lat_max, lon_min, lon_max
            )
            chl_reg_raw = _extract_region(
                chl_z_raw, lat_min, lat_max, lon_min, lon_max
            )

            # --- raw correlation at the observed peak lag for this region ---
            r_raw_map = sig_raw_map = None
            peak_lag  = region_peak_lag[region]
            try:
                r_raw_map, sig_raw_map = _pointwise_lag_corr(
                    sst_reg_raw, chl_reg_raw, lag=peak_lag
                )
                cf = _draw_contourf_panel(
                    axes[row_idx, 0], r_raw_map, sig_raw_map,
                    title=col_titles[0] if row_idx == 0 else src_label,
                    region=region,
                )
                last_cf = cf
            except Exception as e:
                logger.warning(
                    f"  [{src_label}] {region} raw corr failed: {e}"
                )

            # --- partial correlation: index removed from BOTH fields ---
            if has_removed and ncols == 3:
                sst_z_rem = entry.get(sst_key_removed)
                chl_z_rem = entry.get(chl_key_removed)
                if sst_z_rem is not None and chl_z_rem is not None:
                    try:
                        sst_reg_rem = _extract_region(
                            sst_z_rem, lat_min, lat_max,
                            lon_min, lon_max
                        )
                        chl_reg_rem = _extract_region(
                            chl_z_rem, lat_min, lat_max,
                            lon_min, lon_max
                        )
                        r_rem, sig_rem = _pointwise_lag_corr(
                            sst_reg_rem, chl_reg_rem, lag=peak_lag
                        )
                        cf2 = _draw_contourf_panel(
                            axes[row_idx, 1], r_rem, sig_rem,
                            title=col_titles[1] if row_idx == 0 else "",
                            region=region,
                        )
                        last_cf = cf2

                        # Difference panel: raw r minus partial r
                        # Shows the spatial footprint of the index
                        # contribution. Uses a tighter colour scale
                        # (±0.4) since differences are smaller.
                        if r_raw_map is not None:
                            r_diff = r_raw_map.copy(
                                data=r_raw_map.values - r_rem.values
                            )
                            sig_diff = sig_raw_map  # use raw significance
                            cf3 = _draw_contourf_panel(
                                axes[row_idx, 2], r_diff, sig_diff,
                                title=col_titles[2] if row_idx == 0 else "",
                                region=region,
                                vmax=0.4,
                                n_levels=9,
                            )
                            last_cf_diff = cf3
                    except Exception as e:
                        logger.warning(
                            f"  [{src_label}] {region} partial corr "
                            f"failed: {e}"
                        )
                else:
                    axes[row_idx, 1].set_visible(False)
                    if ncols == 3:
                        axes[row_idx, 2].set_visible(False)

            # Row label (source name) on the left
            axes[row_idx, 0].text(
                -0.08, 0.5,
                src_label,
                transform=axes[row_idx, 0].transAxes,
                fontsize=8, fontweight="bold",
                ha="right", va="center", rotation=90,
            )

            # Panel letter
            for col in range(ncols):
                letter = chr(ord("a") + row_idx * ncols + col)
                axes[row_idx, col].text(
                    0.02, 0.97, f"({letter})",
                    transform=axes[row_idx, col].transAxes,
                    fontsize=9, fontweight="bold", va="top",
                    bbox=dict(
                        boxstyle="round,pad=0.15",
                        facecolor="white", alpha=0.75,
                        edgecolor="none",
                    ),
                )

            row_idx += 1

        # Two shared colourbars: one for raw/partial (cols 0-1), one
        # for the difference column (col 2, tighter ±0.4 scale)
        if last_cf is not None:
            cax1 = fig.add_axes([0.91, 0.35, 0.013, 0.55])
            cb1  = fig.colorbar(last_cf, cax=cax1, extend="both")
            cb1.set_label(
                f"Pearson r (lag {region_peak_lag[region]:+d} months)",
                fontsize=9,
            )
            cb1.ax.tick_params(labelsize=8)
        if last_cf_diff is not None:
            cax2 = fig.add_axes([0.91, 0.06, 0.013, 0.24])
            cb2  = fig.colorbar(last_cf_diff, cax=cax2, extend="both")
            cb2.set_label(r"$\Delta$r (raw 2212 partial)", fontsize=8)
            cb2.ax.tick_params(labelsize=7)

        idx_desc = index_label.get(idx_name, idx_name)
        lag_str = f"lag {region_peak_lag[region]:+d} months"
        fig.suptitle(
            f"{region_title[region]}: pointwise SST–Chl-$a$ correlation "
            f"at {lag_str} (observed peak lag from Fig 9)\n"
            f"Col 1: raw $r$(SST, Chl);  "
            f"Col 2: partial $r$(SST, Chl $|$ {idx_name}) — "
            f"{idx_desc} removed from both fields;  "
            f"Col 3: difference (raw − partial)\n"
            f"Stippling: $p < 0.05$ ($N_{{\\rm eff}}$, "
            f"Bretherton et al. 1999);  "
            f"period {period.start}–{period.stop}",
            fontsize=9, y=0.995,
        )

        out_path = OUT_DIR / region_filename[region]
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"  wrote {out_path.name}")


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
            logger.warning(traceback.format_exc().replace("<", "\\<"))

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
            logger.warning(traceback.format_exc().replace("<", "\\<"))

    # =========================================================
    # PHASE B — physical driver biases (Figs 5-7)
    # =========================================================
    logger.info("=== PHASE B: driver biases (MLD, SST, clt) ===")

    # ---- Common 1x1 grid: defined unconditionally ----
    # The lag-correlation figures (Figs 9-11, S_amo, S_spatial, S_regional)
    # all depend on common_grid.  Previously it was built from ORAS5 MLD
    # inside a try/except, so a missing ORAS5 file silently skipped every
    # correlation figure.  We now define it unconditionally from a standard
    # 1deg x 1deg global grid; ORAS5 is loaded separately for Fig 5 and
    # its MLD coordinates are verified to match (they always do on Levante
    # since ORAS5 is regridded to 1deg before archiving).
    _lat1 = np.arange(-89.5, 90.5, 1.0)
    _lon1 = np.arange(  0.5, 360.5, 1.0)
    common_grid = xr.DataArray(
        np.zeros((len(_lat1), len(_lon1))),
        dims=("lat", "lon"),
        coords={"lat": _lat1, "lon": _lon1},
    )
    logger.info(
        f"  common_grid set: {len(_lat1)} lat x {len(_lon1)} lon "
        f"(1deg global, unconditional)"
    )

    # ---- Fig 5: MLD bias ----
    obs_mld      = None
    obs_mld_clim = None
    obs_mld_ts: xr.DataArray | None = None   # for Figs 9, 10
    mld_bias: dict[str, xr.DataArray] = {}
    model_mld_clim: dict[str, xr.DataArray] = {}   # for Fig 9 (legacy)
    model_mld_ts:   dict[str, xr.DataArray] = {}   # native time series for Fig 10
    # Load obs MLD independently — a missing ORAS5 file must not
    # prevent model_mld_ts from being populated, since the lag-
    # correlation and new MLD analyses only need model fields.
    try:
        obs_mld    = load_obs_mld()
        obs_mld_ts = obs_mld
        logger.info(f"  ORAS5 MLD: {obs_mld.shape}")
        obs_mld_clim = obs_mld.mean("time", keep_attrs=True)
    except Exception as e:
        logger.warning(f"  ORAS5 MLD load failed: {e} — OBS MLD curves will be absent")

    # Load model MLD independently of obs — always runs
    for model in MODELS:
        try:
            mld_native = load_model_mld(model)
            if mld_native is None:
                continue
            mld_re = regrid_timeseries(mld_native, common_grid)
            model_mld_ts[model] = mld_re
            mld_clim_on_oras = mld_re.mean("time")
            model_mld_clim[model] = mld_clim_on_oras
            if obs_mld_clim is not None:
                mld_bias[model] = mld_clim_on_oras - obs_mld_clim
            logger.success(f"  [{model}] MLD loaded ({model_mld_ts[model].sizes})")
        except Exception as e:
            logger.warning(f"  [{model}] MLD load/regrid failed: {e}")
    logger.info(f'DEBUG fig5: mld_bias has {len(mld_bias)} models, obs_mld_clim is {"set" if obs_mld_clim is not None else "None"}')
    if mld_bias:
        try:
            make_driver_bias_with_obs(
                bias_dict=mld_bias,
                obs_clim=obs_mld_clim,
                obs_title="ORAS5 (obs)",
                obs_cmap="Blues_r",
                obs_label=r"MLD (m)",
                obs_vmin=0, obs_vmax=300,
                bias_vmin=-50, bias_vmax=50,
                bias_label=r"Bias (m)",
                filename="fig5_mld_bias.png",
                stat_fmt="mean={mean:+.1f} m  RMSE={rmse:.1f} m",
            )
        except Exception as e:
            import traceback
            logger.warning(f"  fig5 MLD bias plot failed: {e}")
            logger.warning(traceback.format_exc().replace("<", "\\<"))

    # ---- Fig 6: SST bias ----
    obs_sst_clim = None
    obs_sst_ts: xr.DataArray | None = None    # for Fig 9
    sst_bias: dict[str, xr.DataArray] = {}
    model_sst_clim: dict[str, xr.DataArray] = {}   # for Fig 9 (legacy)
    model_sst_ts:   dict[str, xr.DataArray] = {}   # native time series for Fig 9
    # Load obs SST independently so a missing ERSSTv3b file does
    # not prevent model_sst_ts from being populated.
    try:
        obs_sst      = load_obs_sst()
        obs_sst_ts   = obs_sst
        obs_sst_clim = obs_sst.mean("time", keep_attrs=True)
        logger.info(f"  NOAA-ERSSTv3b SST: {obs_sst.shape}")
    except Exception as e:
        logger.warning(
            f"  NOAA-ERSSTv3b SST load failed: {e} — OBS SST curves absent"
        )

    # Load model SST independently — always runs
    for model in MODELS:
        try:
            sst_native = load_model_sst(model)
            if sst_native is None:
                continue
            model_sst_ts[model] = sst_native
            sst_clim = sst_native.mean("time", keep_attrs=True)
            if obs_sst_clim is not None:
                sst_re = regrid_to_obs(sst_clim, obs_sst_clim)
                model_sst_clim[model] = sst_re
                sst_bias[model] = sst_re - obs_sst_clim
            logger.success(f"  [{model}] SST loaded")
        except Exception as e:
            logger.warning(f"  [{model}] SST load/regrid failed: {e}")
    logger.info(f'DEBUG fig6: sst_bias has {len(sst_bias)} models, obs_sst_clim is {"set" if obs_sst_clim is not None else "None"}')
    if sst_bias:
        try:
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
                stat_fmt="mean={mean:+.2f} °C  RMSE={rmse:.2f} °C",
            )
        except Exception as e:
            import traceback
            logger.warning(f"  fig6 SST bias plot failed: {e}")
            logger.warning(traceback.format_exc().replace("<", "\\<"))

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
        logger.warning(traceback.format_exc().replace("<", "\\<"))

    # ---- Fig 9 (Chl vs SST) and Fig 10 (Chl vs MLD): lag correlation ----
    # Basin-mean monthly anomalies, deseasonalised + z-scored, then
    # Pearson r at lags -12…+12 months. OBS curves use ESA-CCI Chl
    # against NOAA-ERSSTv3b SST / ORAS5 MLD; model curves use each
    # model's own Chl and driver. Six subplots, one per basin.
    # common_grid is now defined unconditionally so this block always runs.
    logger.info(
        f"  Fig 9/10 inputs: "
        f"obs_sst={'YES' if obs_sst_ts is not None else 'MISSING'} | "
        f"obs_mld={'YES' if obs_mld_ts is not None else 'MISSING'} | "
        f"model_chl={len(model_chl_ts)} models | "
        f"model_sst={len(model_sst_ts)} models | "
        f"model_mld={len(model_mld_ts)} models"
    )
    if common_grid is None:
        logger.warning("  Figs 9/10 SKIPPED: common_grid is None (should not happen)")
    else:
        logger.info(f"  Fig 9 inputs: obs_chl=yes  obs_sst="
                    f"{'yes' if obs_sst_ts is not None else 'NO'}  "
                    f"model_chl={len(model_chl_ts)}  "
                    f"model_sst={len(model_sst_ts)}")
        if model_chl_ts and (obs_sst_ts is not None or model_sst_ts):
            try:
                # ---- compute climate indices from obs SST ----
                # Indices are computed once here and reused by:
                #   Fig 9b (all-basins, modes removed)
                #   Fig S_indices (time-series supplemental)
                #   Fig S_amo (Atlantic sensitivity 2x2 supplemental)
                # If obs_sst_ts is None the dict will be empty and the
                # mode-removal figures are silently skipped downstream.
                climate_indices: dict[str, xr.DataArray] = {}
                if obs_sst_ts is not None:
                    try:
                        obs_sst_re = regrid_timeseries(
                            obs_sst_ts.sel(time=LAG_PERIOD), common_grid
                        )
                        climate_indices = _compute_climate_indices(obs_sst_re)
                    except Exception as _e:
                        import traceback as _tb
                        logger.warning(
                            f"  Climate index computation failed: {_e}"
                        )
                        logger.warning(
                            _tb.format_exc().replace("<", "\\<")
                        )

                # ---- Fig S_climate_combined: index TS + lag sensitivity ----
                # Single combined supplemental figure replacing the
                # separate figS_climate_indices and figS_climate_modes_atlantic.
                # Three rows (one per index): left = index time series,
                # right = N. Atlantic lag correlation with that index removed.
                if climate_indices:
                    try:
                        make_figS_climate_combined(
                            indices=climate_indices,
                            obs_chl_ts=obs_full,
                            obs_sst_ts=obs_sst_ts,
                            model_chl_ts=model_chl_ts,
                            model_sst_ts=model_sst_ts,
                            common_grid=common_grid,
                        )
                    except Exception as e:
                        import traceback
                        logger.warning(f"  Fig S_climate_combined crashed: {e}")
                        logger.warning(
                            traceback.format_exc().replace("<", "\\<")
                        )

                # ---- Fig 9 (raw) + Fig 9b (modes removed) ----
                make_fig9_lag_sst_with_modes(
                    obs_chl_ts=obs_full,
                    obs_sst_ts=obs_sst_ts,
                    model_chl_ts=model_chl_ts,
                    model_sst_ts=model_sst_ts,
                    common_grid=common_grid,
                    indices=climate_indices,
                )
            except Exception as e:
                import traceback
                logger.warning(f"  Fig 9 / climate modes crashed: {e}")
                logger.warning(traceback.format_exc().replace("<", "\\<"))
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
                make_fig10_lag_mld_with_modes(
                    obs_chl_ts=obs_full,
                    obs_mld_ts=obs_mld_ts,
                    model_chl_ts=model_chl_ts,
                    model_mld_ts=model_mld_ts,
                    common_grid=common_grid,
                    indices=climate_indices if "climate_indices" in dir()
                            else {},
                )
            except Exception as e:
                import traceback
                logger.warning(f"  Fig 10 crashed: {e}")
                logger.warning(traceback.format_exc().replace("<", "\\<"))
        else:
            logger.warning(
                f"  Fig 10 SKIPPED: chl_ts={len(model_chl_ts)}, "
                f"mld_ts={len(model_mld_ts)}, obs_mld={obs_mld_ts is not None}"
            )

        # ---- Fig 10 enhanced: Panel A / B / C ----
        print(f"[DEBUG fig10_enhanced] model_chl={len(model_chl_ts)} "
              f"obs_mld={'YES' if obs_mld_ts is not None else 'NO'} "
              f"model_mld={len(model_mld_ts)}", flush=True)
        logger.info(
            f"  Fig10_enhanced gate: model_chl={len(model_chl_ts)} "
            f"obs_mld={'YES' if obs_mld_ts is not None else 'NO'} "
            f"model_mld={len(model_mld_ts)}"
        )
        if model_chl_ts:
            try:
                make_fig10_enhanced(
                    obs_chl_ts=obs_full,
                    obs_mld_ts=obs_mld_ts,
                    model_chl_ts=model_chl_ts,
                    model_mld_ts=model_mld_ts,
                    common_grid=common_grid,
                )
            except Exception as e:
                import traceback
                logger.warning(f"  Fig 10 enhanced crashed: {e}")
                logger.warning(traceback.format_exc().replace("<", "\\<"))
                print(traceback.format_exc(), flush=True)
        else:
            logger.warning("  Fig 10 enhanced SKIPPED: no model_chl_ts")

        # ---- Fig S_bloom_mld_timing: bloom onset vs MLD shoaling scatter ----
        print(f"[DEBUG bloom_timing] running", flush=True)
        logger.info(
            f"  Fig S_bloom_timing gate: "
            f"model_chl={len(model_chl_ts)} "
            f"obs_mld={'YES' if obs_mld_ts is not None else 'NO'} "
            f"model_mld={len(model_mld_ts)}"
        )
        if model_chl_ts:
            try:
                make_figS_bloom_mld_timing(
                    obs_chl_ts=obs_full,
                    obs_mld_ts=obs_mld_ts,
                    model_chl_ts=model_chl_ts,
                    model_mld_ts=model_mld_ts,
                    common_grid=common_grid,
                )
            except Exception as e:
                import traceback
                logger.warning(f"  Fig S_bloom_timing crashed: {e}")
                logger.warning(traceback.format_exc().replace("<", "\\<"))
        else:
            logger.warning("  Fig S_bloom_timing SKIPPED: missing MLD data")

        if climate_indices and model_chl_ts and (
            obs_mld_ts is not None or model_mld_ts
        ):
            try:
                make_figS_mld_climate_combined(
                    indices=climate_indices,
                    obs_chl_ts=obs_full,
                    obs_mld_ts=obs_mld_ts,
                    model_chl_ts=model_chl_ts,
                    model_mld_ts=model_mld_ts,
                    common_grid=common_grid,
                )
            except Exception as e:
                import traceback
                logger.warning(f"  Fig S_mld_combined crashed: {e}")
                logger.warning(traceback.format_exc().replace("<", "\\<"))
        else:
            logger.warning(
                "  Fig S_mld_combined SKIPPED: missing indices or MLD data"
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

    # common_grid is always defined; Phase E always runs
    if False:
        pass  # placeholder to keep else indentation valid
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
                    # Single combined figure: Arctic (top row) +
                    # Southern Ocean (bottom row), 2 x N_models panels.
                    make_polar_bias_map_combined(
                        siconc_bias, obs_sic_clim,
                        vmin=-50, vmax=50,
                        unit_label="% conc.",
                        cmap="RdBu_r",
                        stat_fmt="mean={mean:+.1f}%  RMSE={rmse:.1f}%",
                        obs_edge_threshold=15.0,
                        filename="fig5c_siconc_bias_combined.png",
                    )
            except Exception as e:
                import traceback
                logger.warning(f"  Sea-ice bias maps crashed: {e}")
                logger.warning(traceback.format_exc().replace("<", "\\<"))
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
                logger.warning(traceback.format_exc().replace("<", "\\<"))
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
                logger.warning(traceback.format_exc().replace("<", "\\<"))
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
                logger.warning(traceback.format_exc().replace("<", "\\<"))
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
                logger.warning(traceback.format_exc().replace("<", "\\<"))
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
                logger.warning(traceback.format_exc().replace("<", "\\<"))
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
                logger.warning(traceback.format_exc().replace("<", "\\<"))
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
            logger.warning(traceback.format_exc().replace("<", "\\<"))


    # =========================================================
    # PHASE F — pointwise peak-lag spatial maps (supplemental)
    # =========================================================
    # Produces figS_spatial_peak_lag_{sst,sst_amo,mld,siconc}.png.
    # These figures show at each (lat,lon) cell the Pearson r and the
    # lag at which |r| is maximum, for each driver vs Chl-a. They
    # complement the basin-mean lag curves of Figs 9-11 by revealing
    # the spatial structure of the coupling and the geographic
    # heterogeneity that the basin average masks.
    #
    # Runtime note: the pointwise loop over 64 800 cells × 25 lags is
    # ~2-4 min per source. With 8 sources × 4 figures this is
    # ~1-2 hours total. Submit as a batch job on Levante if needed.
    logger.info("=== PHASE F: pointwise peak-lag spatial maps ===")
    if common_grid is not None:
        try:
            make_figS_spatial_peak_lag(
                obs_chl_ts=obs_full,
                obs_sst_ts=obs_sst_ts,
                obs_mld_ts=obs_mld_ts,
                obs_siconc_ts=obs_siconc_ts,
                model_chl_ts=model_chl_ts,
                model_sst_ts=model_sst_ts,
                model_mld_ts=model_mld_ts,
                model_siconc_ts=model_siconc_ts,
                common_grid=common_grid,
                indices=climate_indices if "climate_indices" in dir() else {},
            )
        except Exception as e:
            import traceback
            logger.warning(f"  Phase F crashed: {e}")
            logger.warning(traceback.format_exc().replace("<", "\\<"))
    else:
        logger.warning("  Phase F SKIPPED: no common grid")

    # ---- Pointwise siconc-Chl-a spatial correlation maps ----
    if common_grid is not None and model_chl_ts:
        try:
            make_figS_seaice_chl_spatial_corr(
                obs_chl_ts=obs_full,
                obs_siconc_ts=obs_siconc_ts,
                model_chl_ts=model_chl_ts,
                model_siconc_ts=model_siconc_ts,
                common_grid=common_grid,
            )
        except Exception as e:
            import traceback
            logger.warning(
                f"  Fig S_seaice_chl_corr crashed: {e}")
            logger.warning(
                traceback.format_exc().replace("<", "\\<"))
            print(traceback.format_exc(), flush=True)
    else:
        logger.warning("  Fig S_seaice_chl_corr SKIPPED")

    # ---- Bloom phenology + sea-ice retreat analysis ----
    if common_grid is not None and model_chl_ts:
        try:
            make_figS_bloom_phenology(
                obs_chl_ts=obs_full,
                obs_siconc_ts=obs_siconc_ts,
                model_chl_ts=model_chl_ts,
                model_siconc_ts=model_siconc_ts,
                common_grid=common_grid,
            )
        except Exception as e:
            import traceback
            logger.warning(f"  Fig S_bloom_phenology crashed: {e}")
            logger.warning(
                traceback.format_exc().replace("<", "\\<"))
            print(traceback.format_exc(), flush=True)
    else:
        logger.warning("  Fig S_bloom_phenology SKIPPED")

    # ---- EOF analyses and explained variance maps ----
    # Fig S_seaice_eof: sea-ice EOF1 pattern, PC1, variance spectrum
    if common_grid is not None:
        try:
            make_figS_seaice_eof(
                obs_siconc_ts=obs_siconc_ts,
                model_siconc_ts=model_siconc_ts,
                common_grid=common_grid,
            )
        except Exception as e:
            import traceback
            logger.warning(f"  Fig S_seaice_eof crashed: {e}")
            logger.warning(traceback.format_exc().replace("<", "\\<"))

    # Fig S_chl_eof: Chl-a EOF1 pattern, PC1, variance spectrum
    if common_grid is not None and model_chl_ts:
        try:
            make_figS_chl_eof(
                obs_chl_ts=obs_full,
                model_chl_ts=model_chl_ts,
                common_grid=common_grid,
            )
        except Exception as e:
            import traceback
            logger.warning(f"  Fig S_chl_eof crashed: {e}")
            logger.warning(traceback.format_exc().replace("<", "\\<"))

    # Fig S_seaice_chl_regression: Chl-a variance explained by sea-ice PC1
    if common_grid is not None and model_chl_ts:
        try:
            make_figS_seaice_chl_regression(
                obs_chl_ts=obs_full,
                obs_siconc_ts=obs_siconc_ts,
                model_chl_ts=model_chl_ts,
                model_siconc_ts=model_siconc_ts,
                common_grid=common_grid,
            )
        except Exception as e:
            import traceback
            logger.warning(
                f"  Fig S_seaice_chl_regression crashed: {e}")
            logger.warning(traceback.format_exc().replace("<", "\\<"))

    # ---- Fig S_regional_corr: regional contourf SST-Chl-a maps ----
    if common_grid is not None:
        try:
            make_figS_regional_contourf_corr(
                obs_chl_ts=obs_full,
                obs_sst_ts=obs_sst_ts,
                model_chl_ts=model_chl_ts,
                model_sst_ts=model_sst_ts,
                common_grid=common_grid,
                indices=climate_indices if "climate_indices" in dir() else {},
            )
        except Exception as e:
            import traceback
            logger.warning(f"  Fig S_regional_corr crashed: {e}")
            logger.warning(traceback.format_exc().replace("<", "\\<"))
    else:
        logger.warning("  Fig S_regional_corr SKIPPED: no common grid")

    # =========================================================
    # Done
    # =========================================================
    logger.info(f"All figures in {OUT_DIR}")
    logger.info("Summary of figures produced:")
    for f in sorted(OUT_DIR.glob("fig*.png")):
        logger.info(f"  {f.name}")
