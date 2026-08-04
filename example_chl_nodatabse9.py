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

import os
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
# Output directory. With CHL_MASK_TO_OBS=1 the model fields are blanked
# wherever the ocean-colour retrieval is missing in that month, which
# changes every figure the models appear in. Those results go to a
# separate directory rather than overwriting the unmasked ones: the
# point of running both is to compare them, and a masked figure sitting
# under the same filename as an unmasked one is indistinguishable from
# it once the run has finished.
_MASK_TO_OBS_ENV = os.environ.get("CHL_MASK_TO_OBS", "0").strip().lower()
OUT_DIR   = Path.cwd() / (
    "paper_figures_maskedobs"
    if _MASK_TO_OBS_ENV in ("1", "true", "yes", "on")
    else "paper_figures")
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
# All diagnostics (mean state, lag correlations, sea-ice bias maps) use
# all seven where the required field is available. The iron-limitation
# figure that depended on dfe was removed in v7, so the missing dfe in
# ACCESS-ESM1-5 (WOMBAT), CanESM5 (CMOC) and CESM2 no longer matters.

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
SICONC_OBS_NAME = "HadISST"

# ------------------------------------------------------------
# Second observational sea-ice product: ESA-CCI SEAICE
#
# Confirmed from the file header (reSICCI3LF, product_version 3.0,
# SSM/I + SSMIS, 1991-2020, EPSG:6931 Lambert azimuthal equal-area):
#   variable  siconc, units "%", _FillValue 1e20
#   grid      864 x 864 on (j, i), lat(j,i) and lon(j,i) 2-D
#   NH extent reaches down to 16.6 N (geospatial_lat_min)
#   time      DAILY (time_coverage_resolution = P1D, 365 steps per
#             yearly file) despite the SImon token in the filename
#
# Independent satellite reference for the pointwise siconc-Chl-a lag
# correlation maps. Four structural differences from HadISST matter
# and are handled by the loader below:
#
#   1. Hemisphere-split files. The archive stores NH and SH separately
#      ("...-12.5kmEASE2-fv3.0-NH_SImon_siconc_YYYYMM-YYYYMM.nc" and
#      the matching -SH file), one file per year. Both hemispheres are
#      loaded and merged onto the global 1 deg common grid, with each
#      file's coverage clipped to its own hemisphere so the polar grid
#      cannot bleed across the equator.
#
#   2. Native EASE2 projection. Unlike HadISST's 1 deg regular grid,
#      the 12.5 km EASE2 grid is projected, so lat/lon are 2-D
#      auxiliary coordinates over (x, y). Nearest-neighbour regridding
#      in lat/lon space degenerates near the pole (all meridians
#      converge), so _regrid_curvilinear_to_grid below builds the
#      neighbour index on the unit sphere in 3-D Cartesian coordinates
#      instead. The index is built ONCE and reused for every month,
#      which matters: the naive per-timestep scipy.griddata path in
#      regrid_to_obs would rebuild a ~750k-point tree 216 times per
#      hemisphere.
#
#   3. Daily storage. The files are daily, so they are aggregated to
#      monthly means on the native grid before regridding. This is not
#      optional: _ym_key maps every day of a month to one year-month
#      integer, so the alignment in _compute_r_beta would otherwise
#      pick a single arbitrary day per month rather than the mean.
#      Doing it before the regrid also cuts ~6600 regrid applications
#      per hemisphere down to ~216.
#
#   4. Percent units. siconc is stored as "%", converted to fraction
#      from the declared units attribute to match HadISST and the
#      CMIP6 models.
#
# The loader tolerates a regular 1-D lat/lon file too, in case the
# pool copy has already been regridded — it falls back to the standard
# interpolation path in that case.
# ------------------------------------------------------------

SICONC_OBS2_NAME = "ESACCI-SEAICE"
SICONC_OBS2_DIR  = Path(
    "/work/bd0854/DATA/ESMValTool2/OBS/Tier2/ESACCI-SEAICE"
)
# Hemisphere tag is substituted in; leading wildcard absorbs the
# processing-level/sensor/grid/version string, which differs between
# the NH and SH products.
SICONC_OBS2_GLOB = "OBS6_ESACCI-SEAICE_sat_*{hemi}_SImon_siconc_*.nc"
# Cell-area file for the same grid, one per hemisphere. Used to
# area-weight the 12.5 km -> 1 deg aggregation. This is not a detail:
# a 1 deg cell contains ~50 source cells at 50 deg latitude and ~27 at
# 70 deg, so nearest-neighbour sampling would keep one of them and
# discard the rest. In the marginal ice zone a single 12.5 km cell
# swings between 0 and 1 as the edge migrates, while the true 1 deg
# mean varies smoothly, so subsampling injects noise directly into the
# quantity being correlated. Aggregating with area weights is both the
# physically correct operator and far less noisy.
SICONC_OBS2_AREA_GLOB = "OBS6_ESACCI-SEAICE_sat_*{hemi}_Ofx_areacello.nc"
# Minimum fraction of a target cell's geometric area that must be
# covered by valid source cells for that cell to be retained. Guards
# against a 1 deg cell at the edge of the product footprint being
# defined by two or three source cells.
SICONC_OBS2_MIN_COVER = 0.5
# Fallback only: nearest-neighbour acceptance radius, used if the
# areacello file cannot be read. The 1 deg target cell is ~111 km
# across, the source is 12.5 km, so any target inside the footprint
# has a source point far closer than this; the threshold exists only
# to leave cells outside the footprint as NaN rather than filling them
# from the nearest edge point.
SICONC_OBS2_MAX_DIST_KM = 120.0

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


def _get_2d_lat_lon(da: xr.DataArray) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (lat2d, lon2d) matching the trailing two dims of `da`.

    Returns None if the field is on a regular 1-D lat/lon grid, in
    which case the caller should use the ordinary interpolation path.
    """
    lat_name = next((n for n in ("lat", "latitude", "nav_lat", "TLAT")
                     if n in da.coords), None)
    lon_name = next((n for n in ("lon", "longitude", "nav_lon", "TLON")
                     if n in da.coords), None)
    if lat_name is None or lon_name is None:
        return None
    lat = np.asarray(da[lat_name].values)
    lon = np.asarray(da[lon_name].values)
    if lat.ndim == 1 and lon.ndim == 1 and \
            lat_name in da.dims and lon_name in da.dims:
        return None                      # regular grid
    shp = da.shape[-2:]
    if lat.ndim == 1 and lat.size == shp[0]:
        lat = np.broadcast_to(lat[:, None], shp).copy()
    if lon.ndim == 1 and lon.size == shp[1]:
        lon = np.broadcast_to(lon[None, :], shp).copy()
    if lat.shape != shp or lon.shape != shp:
        return None
    return lat, lon


def _build_sphere_nn_index(
    src_lat2d: np.ndarray,
    src_lon2d: np.ndarray,
    tgt_lat:   np.ndarray,
    tgt_lon:   np.ndarray,
    max_dist_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbour index from a curvilinear source to a regular grid.

    Neighbours are found in 3-D Cartesian coordinates on the unit
    sphere, not in (lat, lon) degrees. This matters at high latitude:
    in degree space two points at 85 N separated by 180 deg of
    longitude look 180 units apart when they are physically ~100 km
    apart, so a lat/lon KD-tree mis-assigns neighbours exactly in the
    region these figures are about.

    Returns (src_flat_idx, valid) both of length nlat*nlon, where
    `valid` is False for target cells with no source point inside
    max_dist_km (i.e. outside the product footprint).
    """
    from scipy.spatial import cKDTree
    R_EARTH_KM = 6371.0

    finite = np.isfinite(src_lat2d) & np.isfinite(src_lon2d)
    if not finite.any():
        raise ValueError("source grid has no finite lat/lon")

    slat = np.deg2rad(src_lat2d[finite])
    slon = np.deg2rad(src_lon2d[finite])
    src_xyz = np.column_stack([
        np.cos(slat) * np.cos(slon),
        np.cos(slat) * np.sin(slon),
        np.sin(slat),
    ])
    tree = cKDTree(src_xyz)

    tlon2d, tlat2d = np.meshgrid(tgt_lon, tgt_lat)
    tlat = np.deg2rad(tlat2d.ravel())
    tlon = np.deg2rad(tlon2d.ravel())
    tgt_xyz = np.column_stack([
        np.cos(tlat) * np.cos(tlon),
        np.cos(tlat) * np.sin(tlon),
        np.sin(tlat),
    ])

    dist, idx = tree.query(tgt_xyz, k=1)
    # Great-circle distance d maps to chord 2*sin(d / 2R) on the unit
    # sphere; compare in chord space to avoid an arcsin per point.
    chord_max = 2.0 * np.sin(max_dist_km / (2.0 * R_EARTH_KM))
    valid = dist <= chord_max
    src_flat_idx = np.flatnonzero(finite.ravel())[idx]
    return src_flat_idx, valid


def _target_cell_area_m2(tgt_lat: np.ndarray,
                         tgt_lon: np.ndarray) -> np.ndarray:
    """Geometric area (m2) of each cell of a regular lat/lon grid.

    A = R^2 * dlon_rad * (sin(lat_north) - sin(lat_south)).
    Used as the denominator of the coverage test, so that "half the
    cell is covered" means half its true spherical area, not half its
    cell count.
    """
    R_M = 6371000.0
    dlat = float(np.abs(np.diff(tgt_lat)).mean())
    dlon = float(np.abs(np.diff(tgt_lon)).mean())
    lat_s = np.deg2rad(tgt_lat - dlat / 2.0)
    lat_n = np.deg2rad(tgt_lat + dlat / 2.0)
    band = (R_M ** 2) * np.deg2rad(dlon) * (np.sin(lat_n) - np.sin(lat_s))
    return np.broadcast_to(np.abs(band)[:, None],
                           (tgt_lat.size, tgt_lon.size)).copy()


def _to_monthly_mean(da: xr.DataArray, label: str = "") -> xr.DataArray:
    """Collapse a sub-monthly time axis to monthly means.

    The ESA-CCI files carry an SImon-style name but are stored at
    daily resolution (time_coverage_resolution = P1D, 365 steps per
    yearly file). Nothing downstream would raise on that, which is
    exactly the danger: _ym_key maps every day of a month to the same
    year*100+month integer, and the searchsorted alignment inside
    _compute_r_beta then picks the FIRST matching entry per month. The
    correlation would silently be computed from one arbitrary day per
    month instead of the monthly mean, against monthly Chl-a — noisier
    and not the quantity the manuscript describes.

    Grouping on the year-month integer rather than using
    .resample("MS") keeps this calendar-agnostic, and the returned
    time axis is plain datetime64 month-starts so it matches the
    HadISST and ESA-CCI Chl-a axes without any cftime mixing.
    """
    import pandas as pd

    if "time" not in da.dims or da.sizes["time"] < 2:
        return da

    ym = _ym_key(da)
    n_months = np.unique(ym).size
    if n_months == da.sizes["time"]:
        # Already one step per month — just normalise the stamps.
        stamps = pd.to_datetime(
            [f"{v // 100:04d}-{v % 100:02d}-01" for v in ym])
        return da.assign_coords(time=stamps)

    steps_per_month = da.sizes["time"] / max(n_months, 1)
    logger.info(
        f"    {label}: time axis is sub-monthly "
        f"({da.sizes['time']} steps over {n_months} months, "
        f"~{steps_per_month:.1f} per month) — aggregating to "
        f"monthly means")

    units = da.attrs.get("units")
    out = (da.assign_coords(_ym=("time", ym))
             .groupby("_ym").mean("time", skipna=True))
    ym_vals = np.asarray(out["_ym"].values)
    stamps = pd.to_datetime(
        [f"{int(v) // 100:04d}-{int(v) % 100:02d}-01" for v in ym_vals])
    out = (out.rename({"_ym": "time"})
              .assign_coords(time=stamps)
              .sortby("time"))
    if units is not None:
        out.attrs["units"] = units
    return out


def _load_esacci_areacello(hemi: str,
                           expected_shape: tuple[int, int]
                           ) -> np.ndarray | None:
    """Load the ESA-CCI cell-area field for one hemisphere.

    Returns a 2-D array matching the siconc grid, or None if the file
    is absent or its shape disagrees (in which case the caller falls
    back to nearest-neighbour).
    """
    paths = sorted(
        SICONC_OBS2_DIR.glob(SICONC_OBS2_AREA_GLOB.format(hemi=hemi)))
    if not paths:
        logger.warning(
            f"  {SICONC_OBS2_NAME} {hemi}: no areacello matching "
            f"{SICONC_OBS2_AREA_GLOB.format(hemi=hemi)}")
        return None
    try:
        ds = xr.open_dataset(str(paths[-1]))
        name = next((n for n in ("areacello", "areacella", "area", "cell_area")
                     if n in ds.data_vars), None)
        if name is None:
            name = list(ds.data_vars)[0]
        area = np.asarray(ds[name].squeeze().values, dtype=float)
    except Exception as e:
        logger.warning(
            f"  {SICONC_OBS2_NAME} {hemi}: areacello read failed: {e}")
        return None

    if area.shape != expected_shape:
        logger.warning(
            f"  {SICONC_OBS2_NAME} {hemi}: areacello shape {area.shape} "
            f"!= siconc grid {expected_shape}; ignoring areacello")
        return None

    # Treat non-positive / non-finite as no cell (land, off-grid).
    area = np.where(np.isfinite(area) & (area > 0), area, np.nan)
    finite = np.isfinite(area)
    logger.info(
        f"  {SICONC_OBS2_NAME} {hemi}: areacello '{name}' "
        f"{paths[-1].name}, {int(finite.sum())} valid cells, "
        f"mean={np.nanmean(area) / 1e6:.2f} km2, "
        f"range {np.nanmin(area) / 1e6:.2f}-{np.nanmax(area) / 1e6:.2f} km2")
    return area


def _regrid_curvilinear_area_weighted(
    da:        xr.DataArray,
    target:    xr.DataArray,
    area2d:    np.ndarray,
    src_lat2d: np.ndarray,
    src_lon2d: np.ndarray,
    min_cover: float = SICONC_OBS2_MIN_COVER,
    lat_clip:  tuple[float, float] | None = None,
) -> xr.DataArray:
    """Area-weighted aggregation of a fine curvilinear field onto a
    regular lat/lon grid.

    Each source cell is binned into the target cell containing its
    centre, and the target value is the area-weighted mean of the valid
    source cells in that bin:

        value_target = sum(area_i * v_i) / sum(area_i)

    over source cells i with finite v_i. The bin index is computed once
    from the static grid, so each timestep costs two np.bincount passes.

    Cells whose contributing source area falls below `min_cover` times
    their geometric area are set to NaN — this both trims the ragged
    edge of the product footprint and prevents a partially ice-masked
    cell from being represented by a handful of source cells.
    """
    tgt_lat = np.asarray(target.lat.values)
    tgt_lon = np.asarray(target.lon.values)
    nlat, nlon = tgt_lat.size, tgt_lon.size

    dlat = float(np.abs(np.diff(tgt_lat)).mean())
    dlon = float(np.abs(np.diff(tgt_lon)).mean())
    lat0 = float(tgt_lat.min()) - dlat / 2.0
    lon0 = float(tgt_lon.min()) - dlon / 2.0

    slat = src_lat2d.ravel()
    slon = src_lon2d.ravel() % 360.0
    sarea = area2d.ravel()

    good = np.isfinite(slat) & np.isfinite(slon) & np.isfinite(sarea)
    if lat_clip is not None:
        good &= (slat >= lat_clip[0]) & (slat <= lat_clip[1])

    # Replace non-finite coordinates with in-range sentinels before the
    # integer cast: casting NaN to int64 is undefined behaviour and
    # raises a RuntimeWarning. These entries are excluded by `good`
    # regardless, so the sentinel value never reaches the output.
    slat_f = np.where(np.isfinite(slat), slat, lat0)
    slon_f = np.where(np.isfinite(slon), slon, lon0)
    j = np.floor((slat_f - lat0) / dlat).astype(np.int64)
    i = np.floor((slon_f - lon0) / dlon).astype(np.int64)
    # Longitude is periodic, so WRAP rather than clip. Clipping is
    # correct only when the target runs 0..360 (as common_grid does
    # here): on a -180..180 grid, lon0 = -180 while source longitudes
    # are taken mod 360, so indices land in [180, 540) and clipping
    # would squeeze the entire eastern hemisphere into the last
    # column. Wrapping is right for both conventions.
    i = np.mod(i, nlon)
    good &= (j >= 0) & (j < nlat)

    src_sel = np.flatnonzero(good)
    bin_idx = (j[src_sel] * nlon + i[src_sel]).astype(np.int64)
    w = sarea[src_sel]
    n_bins = nlat * nlon

    # Denominator for the coverage test: geometric target-cell area.
    tgt_area = _target_cell_area_m2(tgt_lat, tgt_lon).ravel()

    logger.info(
        f"    area-weighted binning: {src_sel.size} source cells -> "
        f"{np.unique(bin_idx).size} target cells "
        f"(median {src_sel.size / max(np.unique(bin_idx).size, 1):.1f} "
        f"source cells per target cell)")

    times = da.time.values
    out = np.full((len(times), n_bins), np.nan, dtype=np.float32)
    n_src_cells = int(np.prod(da.shape[-2:]))
    reported = False

    for k in range(len(times)):
        slab = np.asarray(da.isel(time=k).values, dtype=float).ravel()
        if slab.size != n_src_cells:
            raise ValueError(
                f"timestep {k} has {slab.size} cells, expected "
                f"{n_src_cells}")
        v = slab[src_sel]
        ok = np.isfinite(v)
        wv = np.where(ok, w, 0.0)
        num = np.bincount(bin_idx, weights=wv * np.where(ok, v, 0.0),
                          minlength=n_bins)
        den = np.bincount(bin_idx, weights=wv, minlength=n_bins)
        cover = den / np.where(tgt_area > 0, tgt_area, np.nan)
        keep = (den > 0) & (cover >= min_cover)
        vals = np.full(n_bins, np.nan)
        vals[keep] = num[keep] / den[keep]
        out[k] = vals
        if not reported:
            logger.info(
                f"    first timestep: {int(keep.sum())} target cells "
                f"pass the {min_cover:.0%} coverage test")
            reported = True

    result = xr.DataArray(
        out.reshape(len(times), nlat, nlon),
        dims=("time", "lat", "lon"),
        coords={"time": da.time, "lat": target.lat, "lon": target.lon},
    )
    return snap_to_month_start(result)


def _regrid_curvilinear_to_grid(
    da:          xr.DataArray,
    target:      xr.DataArray,
    area2d:      np.ndarray | None = None,
    max_dist_km: float = SICONC_OBS2_MAX_DIST_KM,
    lat_clip:    tuple[float, float] | None = None,
    src_latlon:  tuple[np.ndarray, np.ndarray] | None = None,
) -> xr.DataArray:
    """Regrid a (time, y, x) curvilinear field onto target's 1-D lat/lon.

    Preferred path is area-weighted aggregation using `area2d`
    (areacello), which averages all ~30-50 source cells that fall in
    each target cell. If areacello is unavailable, falls back to
    nearest-neighbour on the sphere, which is a subsample rather than
    an average and is noticeably noisier in the marginal ice zone —
    acceptable as a degraded mode, not as the default.

    The neighbour/bin index is computed once from the static source
    grid and reused for every timestep. `lat_clip` restricts the output
    to a latitude band, used to stop a hemispheric polar grid from
    filling cells on the far side of the equator.
    """
    latlon = src_latlon if src_latlon is not None else _get_2d_lat_lon(da)
    if latlon is None:
        # Already a regular grid — ordinary path.
        out = regrid_timeseries(standardize(da), target)
        if lat_clip is not None:
            out = out.where((out.lat >= lat_clip[0])
                            & (out.lat <= lat_clip[1]))
        return out

    src_lat2d, src_lon2d = latlon

    if area2d is not None and area2d.shape == src_lat2d.shape:
        return _regrid_curvilinear_area_weighted(
            da, target, area2d, src_lat2d, src_lon2d,
            lat_clip=lat_clip)

    logger.warning(
        "    no usable areacello — falling back to nearest-neighbour "
        "(subsamples ~1 of 30 source cells; expect a noisier field)")

    tgt_lat = np.asarray(target.lat.values)
    tgt_lon = np.asarray(target.lon.values)
    nlat, nlon = tgt_lat.size, tgt_lon.size

    src_idx, valid = _build_sphere_nn_index(
        src_lat2d, src_lon2d % 360.0, tgt_lat, tgt_lon % 360.0,
        max_dist_km=max_dist_km)

    if lat_clip is not None:
        lat2d_t = np.broadcast_to(tgt_lat[:, None], (nlat, nlon)).ravel()
        valid = valid & (lat2d_t >= lat_clip[0]) & (lat2d_t <= lat_clip[1])

    logger.info(
        f"    NN index built: {int(valid.sum())}/{valid.size} target "
        f"cells inside footprint"
    )

    if da.ndim != 3:
        raise ValueError(
            f"expected (time, y, x) after squeeze, got dims {da.dims}; "
            f"a stray singleton dimension usually means the CMORized "
            f"file carries an extra coordinate that .squeeze() missed"
        )
    n_src_cells = int(np.prod(da.shape[-2:]))

    times = da.time.values
    out = np.full((len(times), nlat * nlon), np.nan, dtype=np.float32)
    for k, t in enumerate(times):
        slab = np.asarray(da.isel(time=k).values, dtype=np.float32).ravel()
        if slab.size != n_src_cells:
            raise ValueError(
                f"timestep {k} has {slab.size} cells, expected "
                f"{n_src_cells}")
        vals = slab[src_idx]
        out[k, valid] = vals[valid]

    result = xr.DataArray(
        out.reshape(len(times), nlat, nlon),
        dims=("time", "lat", "lon"),
        coords={"time": da.time, "lat": target.lat, "lon": target.lon},
    )
    return snap_to_month_start(result)


def _esacci_file_in_range(path: Path, y0: int, y1: int) -> bool:
    """Keep only files whose YYYYMM-YYYYMM stamp overlaps [y0, y1]."""
    m = re.search(r"_(\d{6})-(\d{6})\.nc$", path.name)
    if not m:
        return True                      # unparseable — let xarray decide
    f0, f1 = int(m.group(1)[:4]), int(m.group(2)[:4])
    return not (f1 < y0 or f0 > y1)


def load_obs_siconc_esacci(
    common_grid: xr.DataArray,
    year_range:  tuple[int, int] = (1997, 2014),
) -> xr.DataArray | None:
    """Load ESA-CCI SEAICE siconc, both hemispheres, on the 1 deg grid.

    Returns a (time, lat, lon) fraction field covering 1997-2014, or
    None if the product is unavailable. NH and SH are read separately
    (the archive splits them), regridded independently, then merged;
    each hemisphere is clipped to its own side of the equator.
    """
    if not SICONC_OBS2_DIR.exists():
        logger.warning(
            f"  {SICONC_OBS2_NAME} directory not found: {SICONC_OBS2_DIR}")
        return None

    y0, y1 = year_range
    hemis = {"NH": (0.0, 90.0), "SH": (-90.0, 0.0)}
    parts: list[xr.DataArray] = []

    for hemi, clip in hemis.items():
        paths = sorted(
            SICONC_OBS2_DIR.glob(SICONC_OBS2_GLOB.format(hemi=hemi)))
        paths = [p for p in paths if _esacci_file_in_range(p, y0, y1)]
        if not paths:
            logger.warning(
                f"  {SICONC_OBS2_NAME} {hemi}: no files in "
                f"{y0}-{y1} matching "
                f"{SICONC_OBS2_GLOB.format(hemi=hemi)}")
            continue
        logger.info(
            f"  {SICONC_OBS2_NAME} {hemi}: {len(paths)} files "
            f"({paths[0].name} ... {paths[-1].name})")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # combine="nested" rather than "by_coords": these files
                # carry 2-D lat/lon plus lat_bnds/lon_bnds over
                # (j, i, bnds_4), and by_coords tries to align every
                # such variable across the 18 files, which raises on
                # any floating-point discrepancy in the grid
                # definition. The files are one per year and already
                # sorted, so concatenating along time is both correct
                # and cheaper. compat/join="override" takes the grid
                # from the first file instead of comparing.
                # chunks= forces a dask-backed array: these are daily
                # 864x864 fields, ~20 GB over 1997-2014, which must
                # not be read eagerly.
                ds = xr.open_mfdataset(
                    [str(p) for p in paths],
                    combine="nested",
                    concat_dim="time",
                    data_vars="minimal",
                    coords="minimal",
                    compat="override",
                    join="override",
                    chunks={"time": 31},
                    use_cftime=True,
                    drop_variables=("lat_bnds", "lon_bnds", "time_bnds",
                                    "i_bnds", "j_bnds"),
                )
            var = _pick_siconc_var(ds)
            da  = ds[var].sel(time=slice(str(y0), str(y1))).squeeze()
            if da.sizes.get("time", 0) == 0:
                logger.warning(
                    f"  {SICONC_OBS2_NAME} {hemi}: no timesteps in "
                    f"{y0}-{y1}")
                continue
            logger.info(
                f"    var='{var}' units='{da.attrs.get('units', '?')}' "
                f"native shape {tuple(da.shape)}")

            # Capture the 2-D lat/lon BEFORE aggregating. Older xarray
            # versions drop non-dimension coordinates through a
            # groupby reduction; if that happens the field silently
            # looks like a regular grid, falls into the interpolation
            # branch, and fails there with a confusing error.
            src_latlon = _get_2d_lat_lon(da)
            logger.info(
                f"    grid: "
                f"{'curvilinear 2-D lat/lon' if src_latlon is not None else 'regular 1-D lat/lon'}"
            )

            # Daily -> monthly BEFORE regridding: aggregating first
            # turns ~6600 regrid applications into ~216 per hemisphere.
            da = _to_monthly_mean(da, label=f"{SICONC_OBS2_NAME} {hemi}")
            logger.info(
                f"    after monthly aggregation: {tuple(da.shape)}")

            # Percent -> fraction, driven by the declared units where
            # present (the CMORized files say units = "%") and falling
            # back to a range check otherwise.
            units = str(da.attrs.get("units", "")).strip().lower()
            if units in ("%", "percent", "percentage"):
                logger.info(
                    f"    units='{units}' — converting percent to fraction")
                da = da / 100.0
            elif units not in ("1", "fraction", ""):
                logger.warning(
                    f"    unrecognised units '{units}' — will fall back "
                    f"to a range check after merging")

            area2d = _load_esacci_areacello(hemi, tuple(da.shape[-2:]))
            re_da = _regrid_curvilinear_to_grid(
                da, common_grid, area2d=area2d, lat_clip=clip,
                src_latlon=src_latlon)
            parts.append(re_da)
            logger.success(
                f"  {SICONC_OBS2_NAME} {hemi} regridded: "
                f"{re_da.sizes.get('time')} months")
        except Exception as e:
            import traceback
            logger.warning(
                f"  {SICONC_OBS2_NAME} {hemi} load failed: {e}")
            logger.warning(traceback.format_exc().replace("<", "\\<"))

    if not parts:
        logger.warning(f"  {SICONC_OBS2_NAME}: nothing loaded")
        return None

    if len(parts) == 1:
        merged = parts[0]
    else:
        # Hemispheres are disjoint by construction, so combine_first
        # after an outer time align simply stitches them together.
        a, b = xr.align(parts[0], parts[1], join="outer")
        merged = a.combine_first(b)

    # Safety net only — the per-hemisphere step above already converts
    # using the declared units. This catches a file whose units
    # attribute is missing or non-standard.
    mx = float(np.nanmax(merged.values)) if merged.size else np.nan
    if np.isfinite(mx) and mx > 1.5:
        logger.warning(
            f"  {SICONC_OBS2_NAME}: max={mx:.1f} after unit handling — "
            f"applying percent-to-fraction fallback")
        merged = merged / 100.0
    merged = merged.where((merged >= 0) & (merged <= 1))
    merged.attrs["units"] = "fraction"
    merged.attrs["source"] = SICONC_OBS2_NAME

    frac_cov = float(np.isfinite(merged.values).mean())
    logger.success(
        f"  {SICONC_OBS2_NAME} ready: {merged.sizes.get('time')} months, "
        f"{frac_cov * 100:.1f}% of grid cells have data")
    return merged


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
    cbar_obs.set_label(obs_label, fontsize=15)
    cbar_obs.ax.tick_params(labelsize=14)

    if im_bias is not None:
        ax_cbar_bias = fig.add_axes([0.55, cbar_y, 0.40, cbar_height])
        cbar_bias    = fig.colorbar(
            im_bias, cax=ax_cbar_bias,
            orientation="horizontal", extend="both",
        )
        cbar_bias.set_label(bias_label, fontsize=15)
        cbar_bias.ax.tick_params(labelsize=14)

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
        re_slab = regrid_to_obs(slab, obs_da)
        out_list.append(re_slab.expand_dims(time=[t]))
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
    cbar_obs.set_label(CHL_LABEL, fontsize=15)
    cbar_obs.ax.tick_params(labelsize=14)

    if im_bias is not None:
        ax_cbar_bias = fig.add_axes([0.55, cbar_y, 0.40, cbar_height])
        cbar_bias    = fig.colorbar(
            im_bias, cax=ax_cbar_bias,
            orientation="horizontal", extend="both",
        )
        cbar_bias.set_label(f"Bias ({CHL_LABEL})", fontsize=13)
        cbar_bias.ax.tick_params(labelsize=14)

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
    legible over busy pcolormesh or line content.

    Indices beyond 25 continue as (aa), (ab), ... rather than wrapping.
    The previous `index % 26` silently reused letters, which is harmless
    while every figure has fewer than 27 panels but produces two panels
    labelled (a) in the per-model correlation figure, where 8 sources x
    5 basins gives 40."""
    if index < 26:
        letter = string.ascii_lowercase[index]
    else:
        first = string.ascii_lowercase[(index // 26) - 1]
        letter = first + string.ascii_lowercase[index % 26]
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
    model_chl_full: dict[str, xr.DataArray] | None = None,
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
    logger.info(
        "Building Fig 3 (seasonality of interannual Chl-a variability)...")

    if model_chl_full is None:
        logger.warning(
            "  Fig 3 needs the full model Chl-a series (model_chl_full) "
            "to compute interannual variability; skipping")
        return
    models   = [m for m in MODELS if m in model_chl_full]
    n_models = len(models)
    nrows, ncols = 4, 2

    def _seasonal_amplitude(ts: xr.DataArray,
                            key: str | None = None) -> xr.DataArray:
        """Seasonal distribution of INTERANNUAL Chl-a variability.

        Steps, in this order:
          1. per grid cell, deseasonalise (remove the monthly
             climatology), linearly detrend, and standardise to unit
             variance over the full record (_deseas_detrend_zscore);
          2. group the resulting anomalies by calendar month and take
             the standard deviation ACROSS YEARS, giving the typical
             anomaly amplitude in each month;
          3. average zonally.

        Order matters. Deseasonalising first is what makes this an
        interannual quantity: the previous version of this figure
        plotted the seasonal cycle itself (monthly climatology minus
        annual mean), which is not an anomaly and duplicated Fig. 2.
        Standardising per cell makes the result dimensionless and
        comparable between latitudes and between models, which differ
        in mean Chl-a by up to a factor of three; a value of 1
        indicates a month whose interannual spread equals that cell's
        record-length standard deviation.

        Averaging zonally LAST is also deliberate: the amplitude is
        computed per cell and then averaged, so that anomalies of
        opposite sign at the same latitude do not cancel.
        """
        # Model fields arrive on their native ocean grids, where `lat`
        # is a 2-D auxiliary coordinate; the zonal mean and the sortby
        # below both need 1-D lat, so regrid to the observational grid
        # first. Routed through the shared cache so the result is
        # reused by the later phases rather than recomputed.
        if key is not None:
            z = _regrid_z_cached(ts, obs_clim, LAG_PERIOD, key)
        else:
            z = _deseas_detrend_zscore(ts.sel(time=LAG_PERIOD))

        mo = (_ym_key(z) % 100).astype(int)
        z  = z.assign_coords(_mon=("time", mo))
        amp = z.groupby("_mon").std("time", skipna=True)  # (12, lat, lon)
        amp = amp.rename({"_mon": "month"})
        if amp["lat"].ndim == 1:
            amp = amp.sortby("lat")
        return amp.mean("lon", skipna=True).transpose("lat", "month")

    obs_anom = _seasonal_amplitude(obs_full)

    model_anom: dict[str, xr.DataArray] = {}
    for model in models:
        try:
            model_anom[model] = _seasonal_amplitude(
                model_chl_full[model], key=f"chl::{model}")
        except Exception as e:
            logger.warning(f"  [{model}] Fig 3 amplitude failed: {e}")
    models = [m for m in models if m in model_anom]
    if not models:
        logger.warning("  Fig 3 SKIPPED: no model amplitude fields")
        return

    # ----------------------------------------------------------------
    # Colour scale limits
    # ----------------------------------------------------------------
    # The amplitude is a standard deviation, so it is non-negative:
    # a sequential scale, shared by every panel, so that observations
    # and models are directly comparable. The previous version used a
    # diverging scale with a separate range for the model panels, which
    # were differences rather than the model fields themselves.
    _pool = [obs_anom.values] + [model_anom[m].values for m in models]
    _pool = np.concatenate([v[np.isfinite(v)] for v in _pool])
    amp_vmax = float(np.nanpercentile(_pool, 98)) if _pool.size else 1.5
    amp_vmax = max(amp_vmax, 0.2)
    obs_vlim = bias_vlim = amp_vmax

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
        cmap="viridis",
        vmin=0.0, vmax=obs_vlim,
        shading="auto",
    )
    ax0.set_title("ESA-CCI (observations)", fontsize=10,
                  fontweight="bold")
    ax0.set_ylabel("Latitude (°)", fontsize=12)
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
    # Left-column colourbars carry no label: every panel shares one
    # scale, so repeating the text four times down the left side adds
    # nothing and crowds the axes. The right-hand column keeps it.
    cb_obs.ax.tick_params(labelsize=14)

    # ----------------------------------------------------------------
    # Panels (b)-(h): each MODEL's own variability amplitude, on the
    # same scale as the observations. Previously these panels showed
    # model minus obs, which the manuscript text described as if it
    # were the model field itself — on a difference panel "near-zero"
    # means agreement, not weak variability.
    # ----------------------------------------------------------------
    im_bias_last = None
    for idx, model in enumerate(models):
        ax  = axes_flat[idx + 1]
        add_panel_label(ax, idx + 1)

        fld = model_anom[model]               # (lat, month)

        im_bias_last = ax.pcolormesh(
            months, lats, fld.values,
            cmap="viridis",
            vmin=0.0, vmax=amp_vmax,
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
            ax.set_ylabel("Latitude (°)", fontsize=12)

        # Colourbar on each panel
        cb = fig.colorbar(
            im_bias_last, ax=ax,
            orientation="vertical", pad=0.02,
            fraction=0.046, extend="both",
        )
        if col != 0:
            cb.set_label("Interannual Chl-$a$ variability\n(s.d. of standardised anomalies)", fontsize=13)
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
        cbar.set_label(f"Bias [{unit_label}]", fontsize=15)

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
        cbar.set_label("Pearson r  [MLD anomaly vs Chl-a bias]", fontsize=15)

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

# Minimum fraction of months a cell must be observed in to enter the
# fixed footprint used by the basin-mean lag diagnostics. Both the
# basin-mean curves and the pointwise maps are restricted to these
# cells, so the two figures describe the same geography. 0.9 retains
# the open ocean and drops cells ocean colour only sees seasonally;
# lowering it recovers polar area at the cost of admitting cells whose
# record is dominated by one part of the year.
LAG_FOOTPRINT_MIN_FRAC = 0.9
# Lags used by the sea-ice figures, in months, with siconc
# LEADING Chl-a. Module-level because both the correlation maps
# and the regression maps must use the identical list for their
# panels to be comparable column by column.
LAGS = [0, 1, 2, 3]

# ------------------------------------------------------------
# Season definitions for the seasonal cross-lag diagnostic
# (make_fig10_enhanced, Panels B and C).
#
# These were referenced in three places -- _seasonal_cross_lag_matrix
# and make_fig10_enhanced, whose comment reads "Use module-level
# season constants" -- but never actually defined, so Panel B always
# died with "name '_SEASONS_DEF' is not defined". Panel A was written
# before the failure, which is why fig10a existed while fig10b and
# fig10c did not.
#
# Months are given in calendar order; DJF spans the year boundary and
# _seasonal_mean_series groups by the calendar year of each month, so
# a December is grouped with the January and February that PRECEDE it
# rather than those that follow. That is the conventional choice for a
# 17-year record and keeps every season the same length.
# ------------------------------------------------------------
_SEASONS_DEF = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}
_SEASON_LIST = ["DJF", "MAM", "JJA", "SON"]
# Number of seasonal lags scanned in the cross-lag matrix: 0 to 3
# seasons, i.e. up to one full year of lead.
_N_LAG_SEAS = 4

# Number of sea-ice EOF modes retained before the 50%-variance cut is
# applied. Must be comfortably larger than the number of modes actually
# needed, otherwise the cut is never reached and the regression quietly
# uses every available mode. See _prepare_source.
# Polar regions shared by every sea-ice x Chl-a figure. Defined once
# so the correlation, regression and gain maps cannot drift apart:
# they had, with the Arctic running 45-90N in the correlation figure
# and 50-90N in the regression, which made the two families
# non-comparable panel by panel despite identical layout.
SEAICE_REGIONS = {
    "Arctic": {
        "lat_min":  45, "lat_max":  90,
        "proj":     ccrs.NorthPolarStereo(),
        "extent":   [-180, 180, 43, 90],
    },
    "Southern": {
        "lat_min": -78, "lat_max": -45,
        "proj":     ccrs.SouthPolarStereo(),
        "extent":   [-180, 180, -78, -43],
    },
}

EOF_N_MODES = 40
# Fraction of sea-ice variance the retained PCs must explain. This is
# the single knob that drives everything downstream: the target sets
# the predictor count n_pcs, which sets the beta sample-size floor
# (MIN_OBS_PER_PREDICTOR_BETA x n_pcs), which sets how much of the
# seasonal-ice zone gets masked. Measured on the Southern Ocean:
#     target   n_pcs   beta floor (4x)
#       30%      10          40 months
#       40%      15          60 months
#       50%      17-22       68-88 months
# Per-cell z-scoring whitens the field, so no small set of modes
# dominates and a high target is expensive. 0.50 is defensible but
# masks much of the ice zone; 0.30-0.40 keeps far more coverage and
# gives a more stable beta, at the cost of describing less of the ice
# variability. Whatever is chosen must match the manuscript text.
EOF_VAR_TARGET = 0.40

# Minimum valid observations per regression predictor. With k
# predictors and only ~k observations the fit is saturated and the
# PC1 coefficient is dominated by noise regardless of the true
# relationship (measured: |beta| = 0.62 at n/k = 1.2 with true
# beta = 0; 0.12 at n/k = 3.5). 4 is a conventional floor.
MIN_OBS_PER_PREDICTOR = 2
# Beta (the PC1 coefficient) is far more fragile than R^2 and needs a
# stricter floor. Adjusted R^2 penalises for sample size and stays
# usable at n/k = 2, but the individual coefficient does not: at
# n/k = 2 the null distribution of |beta| has p99 = 0.70 against
# ~0.19 at n/k = 12. Those cells set the colour scale and hide the
# real structure everywhere else.
MIN_OBS_PER_PREDICTOR_BETA = 3
# Which lag's sample-size / conditioning mask defines the beta
# footprint for ALL lag panels. Using one reference lag keeps the four
# columns comparable; +3 is chosen because it has the widest coverage
# in the Southern Ocean seasonal-ice zone.
BETA_MASK_LAG = 3
# Minimum valid months for the SIMPLE PC1 regression that produces
# beta. With a single predictor the sample requirement collapses:
# 10 months already gives a stable slope, against ~3*n_pcs = 45 for
# the multiple regression that produces R^2.
BETA_MIN_OBS_SIMPLE = 10

# Speckle suppression for the R2-gain map. Isolated single cells at 1
# degree in the seasonal-ice zone are noise riding on a real signal; a
# cell is kept only if it exceeds GAIN_SPECKLE_MIN and at least
# GAIN_MIN_NEIGHBOURS of its 8 neighbours do too.
GAIN_SPECKLE_MIN   = 0.05
GAIN_MIN_NEIGHBOURS = 4
# The PCs are orthogonal over the full 204-month record but NOT over
# the subset of months a gappy cell retains — a cell keeping only
# summer breaks the orthogonality and inflates every coefficient.
# Cells whose restricted design matrix exceeds this condition number
# have their beta masked (R^2 is retained: it is a property of the fit
# as a whole and is far less sensitive to collinearity).
MAX_DESIGN_COND = 30.0

# ------------------------------------------------------------
# Stage selection — CHL_STAGES environment variable
#
# The full run is ~6.5 h, dominated by a few blocks (measured):
#     figS_spatial_peak_lag_*        ~155 min
#     Fig 9 / Fig 10 lag correlation  ~65 min
#     Fig 11 / Fig 12 siconc memory   ~49 min
#     sea-ice corr + R2/beta + EOF    ~67 min
# Iterating on one figure family should not pay for the rest.
#
# Usage:
#     CHL_STAGES=all      (default) everything, as before
#     CHL_STAGES=seaice   only the sea-ice x Chl-a figures
#     CHL_STAGES=A,B,C    phase letters
#     CHL_STAGES=seaice,bloom,eof   individual stage tags
#
# Stage tags:
#     A B C E F   phase letters (coarse)
#     light       rsds bias + polar light coupling
#     drivers     Fig 5/6/7 bias maps
#     coupling    Fig 8/8b/9/9b/10 MLD-Chl and SST lag correlation
#     memory      Fig 11/11b/11c/12/S1 siconc memory
#     peaklag     figS_spatial_peak_lag_*        <-- the 2.5 h one
#     seaice      pointwise siconc-Chl-a corr + R2/beta maps
#     polar       figS_seaice_chl_polar_*
#     bloom       figS_bloom_phenology_*
#     eof         figS_seaice_eof + figS_chl_eof
#     regional    figS_regional_corr_*
#
# Data loads are gated too: selecting `seaice` skips the MLD, SST,
# cloud and rsds reads entirely, not just their figures.
# ------------------------------------------------------------
_STAGE_ALL = {
    "A", "B", "C", "E", "F",
    "light", "drivers", "coupling", "memory", "peaklag",
    "seaice", "polar", "bloom", "eof", "regional", "ptcorr",
}
_STAGE_PRESETS = {
    "seaice": {"E", "F", "seaice", "eof"},
    "quick":  {"E", "F", "seaice"},
    # Phase A only: mean state (Fig 1-2), anomaly amplitude (Fig 3),
    # seasonal cycle (Fig 4). Skips every driver, coupling, memory and
    # polar diagnostic, so it needs only the Chl-a fields.
    "meanstate": {"A"},
    # Pointwise driver maps need the same SST/MLD loads as the
    # basin-mean coupling diagnostics.
    "ptcorr": {"ptcorr", "coupling"},
}

def _parse_stages() -> set[str]:
    raw = os.environ.get("CHL_STAGES", "").strip()
    if not raw or raw.lower() in ("all", "full"):
        return set(_STAGE_ALL)
    out: set[str] = set()
    for tok in re.split(r"[,\s]+", raw):
        if not tok:
            continue
        if tok.lower() in _STAGE_PRESETS:
            out |= _STAGE_PRESETS[tok.lower()]
        elif tok in _STAGE_ALL:
            out.add(tok)
        elif tok.upper() in _STAGE_ALL:
            out.add(tok.upper())
        else:
            print(f"[CHL_STAGES] unknown stage '{tok}' ignored; "
                  f"valid: {sorted(_STAGE_ALL)} "
                  f"or presets {sorted(_STAGE_PRESETS)}", flush=True)
    return out or set(_STAGE_ALL)

RUN_STAGES = _parse_stages()

def stage(*tags: str) -> bool:
    """True if any of `tags` is enabled for this run."""
    return any(t in RUN_STAGES for t in tags)

# Data-load gates, derived from the enabled stages.
NEED_RSDS = stage("A", "light")
NEED_CLT  = stage("B", "drivers")
NEED_MLD  = stage("B", "C", "drivers", "coupling", "peaklag", "ptcorr")
NEED_SST  = stage("B", "C", "drivers", "coupling", "peaklag", "regional",
                  "ptcorr")
NEED_SEASONAL = stage("A")



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


_BASIN_MEAN_LEGACY_WARNED: set[str] = set()


def _basin_mean_timeseries(
    da: xr.DataArray,
    mask: xr.DataArray,
) -> xr.DataArray:
    """cos(lat)-weighted basin-mean monthly time series. LEGACY.

    Input must have dims (time, lat, lon) on the same grid as `mask`.
    Returns a 1-D DataArray indexed by time.

    Superseded by `_basin_mean_static`, which averages per-cell anomalies
    over a fixed footprint. This version averages RAW values with
    skipna=True, so the set of cells contributing changes from month to
    month with cloud, sun angle and sea ice, and each absent cell shifts
    the mean by its own climatological offset. The resulting series
    carries a sampling signal that co-varies with SST and is absent from
    the pointwise maps, which is one reason the two diagnostics
    disagreed.

    It is kept because the MLD and sea-ice figures have not been migrated
    yet, and removing it outright would break them silently. To make sure
    no path stays on it unnoticed, every distinct call site is logged
    once with its file and line: a figure that still uses the old method
    now says so in the run log rather than looking like it was updated.
    """
    import inspect
    try:
        fr = inspect.stack()[1]
        site = f"{fr.function}:{fr.lineno}"
    except Exception:
        site = "unknown"
    if site not in _BASIN_MEAN_LEGACY_WARNED:
        _BASIN_MEAN_LEGACY_WARNED.add(site)
        logger.warning(
            f"  _basin_mean_timeseries (legacy varying footprint) called "
            f"from {site} - this path has NOT been migrated to the fixed "
            f"footprint used by Figs 9/9b; its curves are not directly "
            f"comparable with the pointwise maps")

    da = da.where(mask)
    w  = np.cos(np.deg2rad(da.lat)) * xr.ones_like(da.lon)
    w  = w.where(mask)
    return da.weighted(w.fillna(0)).mean(dim=("lat", "lon"), skipna=True)


def _static_valid_mask(*fields: xr.DataArray,
                       min_frac: float = 0.9) -> xr.DataArray:
    """Cells finite in at least `min_frac` of months in EVERY field.

    `_basin_mean_timeseries` averages with skipna=True, so its footprint
    changes from month to month: whichever cells ocean colour happened to
    see that month are the ones that enter the average. The pointwise
    diagnostic instead uses complete-case cells with n >= 36. The two
    figures therefore describe different sets of pixels, and because the
    ESA-CCI footprint breathes with cloud, sun angle and sea ice, the
    varying-footprint average carries a sampling signal that is itself
    correlated with SST — a spurious contribution present in one
    diagnostic and absent from the other.

    Applying this mask to both paths removes that difference at source.
    min_frac=0.9 keeps the open ocean and drops seasonally-observed polar
    cells; relax towards ~0.6 if the Arctic and Southern basins lose too
    much area to be meaningful, but use the SAME value on both paths.
    """
    mask = None
    for f in fields:
        if f is None:
            continue
        frac = np.isfinite(f).mean(dim="time")
        m = frac >= min_frac
        mask = m if mask is None else (mask & m)
    if mask is None:
        raise ValueError("_static_valid_mask: no fields supplied")
    return mask


def _basin_mean_static(
    da: xr.DataArray,
    basin_mask: xr.DataArray,
    valid_mask: xr.DataArray,
    min_weight_frac: float = 0.75,
) -> xr.DataArray:
    """cos(lat)-weighted basin mean of an ANOMALY field over fixed cells.

    `da` must already be deseasonalised and detrended PER CELL. This is
    not a stylistic preference, it is what makes the fixed footprint work:

      - Averaging raw values over a footprint that loses a few cells in
        some months lets each absent cell shift the basin mean by its
        own climatological offset, which re-introduces exactly the
        sampling signal the fixed footprint exists to remove. Per-cell
        anomalies have expectation zero, so an absent cell biases
        nothing.
      - Several of the BASINS boxes (Global, Atlantic, Indian) straddle
        the equator. Removing the seasonal cycle from the basin mean, as
        the previous code did, removes only the residue left after the
        northern and southern cycles have partly cancelled; the surviving
        anomalies are then not the area-average of the local anomalies
        and are not comparable with the pointwise maps. Removing the
        cycle per cell first avoids the cancellation entirely.

    A month is returned as NaN unless cells carrying at least
    `min_weight_frac` of the basin's cos(lat) weight are present. The
    default of 0.75 tolerates the scattered cloud gaps that survive
    `_interp_time_gaps` while still rejecting months in which a large
    contiguous region — a polar cap in winter, say — is absent. Setting
    it near 1.0 sounds safer but is not usable in practice: with even a
    few percent of cells missing at random, almost no month clears the
    bar and the whole series is blanked.
    """
    sel = basin_mask & valid_mask
    da  = da.where(sel)
    w   = np.cos(np.deg2rad(da.lat)) * xr.ones_like(da.lon)
    w   = w.where(sel).fillna(0.0)

    num = (da.fillna(0.0) * w).sum(dim=("lat", "lon"))
    den = w.where(np.isfinite(da)).fillna(0.0).sum(dim=("lat", "lon"))
    out = num / den.where(den > 0)

    wtot = float(w.sum())
    if wtot > 0:
        out = out.where(den >= min_weight_frac * wtot)
    return out


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
    interp_gaps: bool = True,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Regrid two time series to the common grid and clip to period.

    `interp_gaps` fills Chl-a gaps of up to three consecutive months, the
    same treatment `_regrid_z_cached(..., interp_gaps=True)` applies on
    the pointwise path. It defaults to True so the basin-mean and
    pointwise diagnostics start from identical data; the previous
    default of no interpolation meant Fig 9 and Fig S_pointwise_corr were
    computed on different versions of the ocean-colour record, which is
    one of the reasons they disagreed.

    The driver is not interpolated: SST and MLD products are gap-free
    over the analysis period, and interpolating a field that has no gaps
    is a no-op that only costs time.
    """
    chl_re = regrid_timeseries(chl_ts.sel(time=period), common_grid)
    drv_re = regrid_timeseries(drv_ts.sel(time=period), common_grid)
    if interp_gaps:
        chl_re = _interp_time_gaps(chl_re, max_gap=3)
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


# ============================================================
# Reconciling the basin-mean and pointwise lag diagnostics
# ============================================================
#
# The basin-mean curve (Fig 9) and the pointwise map
# (Fig S_pointwise_corr) are not two estimates of one quantity. With
# w_i = cos(lat_i), sigma_Xi the residual std of field X at cell i and
# r_ij(k) the cross-correlation between driver at i and Chl-a at j,
#
#   r_basin(k) =
#         sum_ij w_i w_j sigma_Di sigma_Cj r_ij(k)
#       -------------------------------------------------------------
#       sqrt(sum_ij w_i w_j sigma_Di sigma_Dj rDD_ij)
#     * sqrt(sum_ij w_i w_j sigma_Ci sigma_Cj rCC_ij)
#
# whereas the map shows only the diagonal r_ii(k), each cell already
# normalised by its own variance. Two distinct effects separate them:
#
#   1. VARIANCE WEIGHTING. The basin average is dominated by
#      high-variance cells (subpolar blooms, equatorial upwelling) and
#      barely hears the low-variance subtropical gyres that occupy most
#      of the map area.
#   2. SPATIAL COHERENCE. The off-diagonal terms i != j — Chl-a at one
#      place covarying with SST at another — enter the basin mean and
#      are absent from the map entirely. These are the teleconnections:
#      ENSO, the AMO, the PDO.
#
# The functions below evaluate the diagonal-only counterpart of the
# formula above, so the gap between the two figures can be attributed
# to (1) and (2) quantitatively rather than described in words.

def _lagged_pair(X: np.ndarray, Y: np.ndarray, k: int):
    """Slice two arrays so that positive k means the DRIVER leads.

    r(k) = corr(driver(t), chl(t + k)). This is the single definition
    used by `_lag_correlation`, `_pointwise_lag_corr` and
    `_ptcorr_r_and_p`; those three arrive at it with different argument
    orders, so `_assert_lag_convention` below pins it down with a test
    rather than trusting three comments to stay in agreement.
    """
    T = X.shape[0]
    if k > 0:
        return X[: T - k], Y[k:]
    if k < 0:
        kk = -k
        return X[kk:], Y[: T - kk]
    return X, Y


def _assert_lag_convention() -> None:
    """Fail loudly if the sign convention is ever inverted.

    Build a series in which Chl-a is a delayed copy of the driver by
    exactly three months. The peak must then land at lag +3 (driver
    leads), not -3. Cheap enough to run at import; a silent sign flip
    here would reverse the physical interpretation of every lag figure
    in the paper.
    """
    rng = np.random.default_rng(0)
    n = 240
    drv = rng.normal(size=n)
    chl = np.roll(drv, 3)
    chl[:3] = np.nan
    lags = np.arange(-6, 7)
    rr = []
    for k in lags:
        x, y = _lagged_pair(drv, chl, int(k))
        ok = np.isfinite(x) & np.isfinite(y)
        rr.append(np.corrcoef(x[ok], y[ok])[0, 1])
    k_peak = int(lags[int(np.nanargmax(np.abs(rr)))])
    if k_peak != 3:
        raise AssertionError(
            f"lag convention broken: driver-leads-by-3 test peaked at "
            f"lag {k_peak:+d}, expected +3")


_assert_lag_convention()


def _pointwise_r_at_lag(D: np.ndarray, C: np.ndarray, k: int,
                        min_n: int = 36) -> np.ndarray:
    """Per-cell Pearson r at lag k, complete-case, vectorised.

    Matches `_ptcorr_r_and_p` numerically but skips the significance
    machinery, which the decomposition does not need.
    """
    X, Y = _lagged_pair(D, C, int(k))
    ok = np.isfinite(X) & np.isfinite(Y)
    n  = ok.sum(axis=0)
    Xz = np.where(ok, X, 0.0)
    Yz = np.where(ok, Y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = Xz.sum(axis=0) / np.maximum(n, 1)
        my = Yz.sum(axis=0) / np.maximum(n, 1)
        dx = np.where(ok, X - mx[None], 0.0)
        dy = np.where(ok, Y - my[None], 0.0)
        num = (dx * dy).sum(axis=0)
        den = np.sqrt((dx ** 2).sum(axis=0) * (dy ** 2).sum(axis=0))
        r = np.where((den > 0) & (n >= min_n), num / den, np.nan)
    return r


def lag_curve_decomposition(
    drv_z:      xr.DataArray,
    chl_z:      xr.DataArray,
    sigma_drv:  xr.DataArray,
    sigma_chl:  xr.DataArray,
    basin_mask: xr.DataArray,
    valid_mask: xr.DataArray,
    lags:       np.ndarray,
) -> dict[str, np.ndarray]:
    """Three lag curves that together explain the basin-mean/map gap.

    All inputs must already be deseasonalised, detrended and z-scored
    PER CELL, with sigma_* the pre-standardisation residual std of the
    same cells — i.e. the two outputs of
    `_deseas_detrend_zscore(..., return_sigma=True)`.

    Returns
    -------
    r_area : sum_i w_i r_ii(k) / sum_i w_i
             the plain area-weighted mean of the pointwise map, which is
             what the eye reads off Fig S_pointwise_corr.
    r_diag : the correlation the basin mean would give if every
             cross-cell covariance vanished,
               sum_i w_i^2 sD_i sC_i r_ii
               / sqrt(sum_i w_i^2 sD_i^2 * sum_i w_i^2 sC_i^2)
             It differs from r_area ONLY through variance weighting.
    r_full : the actual basin-mean correlation over the same fixed
             footprint — the black curve of Fig 9.

    Reading the result:
      r_area -> r_diag   is the variance-weighting contribution
      r_diag -> r_full   is the spatial-coherence contribution
    If r_diag already lands close to r_full, the disagreement between
    the two figures is a pure weighting artefact and one sentence
    disposes of it. If it does not, the basin mean is being carried by
    large-scale coupled modes, which is a physical result and belongs
    with the ENSO/AMO partial-correlation analysis rather than in a
    methods caveat.
    """
    sel = (basin_mask & valid_mask).values
    lat = drv_z.lat.values
    w2d = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, drv_z.sizes["lon"]))
    w   = np.where(sel, w2d, 0.0)

    sD = np.where(sel, np.asarray(sigma_drv.values, dtype=float), np.nan)
    sC = np.where(sel, np.asarray(sigma_chl.values, dtype=float), np.nan)

    D = np.asarray(drv_z.values, dtype=float)
    C = np.asarray(chl_z.values, dtype=float)

    # Basin-mean series of the PHYSICAL anomalies: undo the per-cell
    # standardisation by multiplying the z-scores back by sigma, so the
    # average is of anomalies in native units and reproduces the Fig 9
    # curve rather than an average of z-scores.
    Dm = _basin_mean_static(drv_z * sigma_drv, basin_mask, valid_mask).values
    Cm = _basin_mean_static(chl_z * sigma_chl, basin_mask, valid_mask).values

    out = {key: np.full(np.shape(lags), np.nan)
           for key in ("r_area", "r_diag", "r_full")}

    for i, k in enumerate(lags):
        k = int(k)
        r_ii = _pointwise_r_at_lag(D, C, k)

        good = (np.isfinite(r_ii) & (w > 0)
                & np.isfinite(sD) & np.isfinite(sC))
        if good.sum() >= 10:
            wg, rg = w[good], r_ii[good]
            sDg, sCg = sD[good], sC[good]

            out["r_area"][i] = float(np.sum(wg * rg) / np.sum(wg))

            num = np.sum(wg ** 2 * sDg * sCg * rg)
            den = np.sqrt(np.sum(wg ** 2 * sDg ** 2)
                          * np.sum(wg ** 2 * sCg ** 2))
            out["r_diag"][i] = float(num / den) if den > 0 else np.nan

        x, y = _lagged_pair(Dm, Cm, k)
        okm = np.isfinite(x) & np.isfinite(y)
        if okm.sum() >= 24:
            out["r_full"][i] = float(np.corrcoef(x[okm], y[okm])[0, 1])

    return out


# ============================================================
# Shared basin-mean lag-curve construction
# ============================================================
#
# There used to be two independent implementations of "compute the
# basin-mean lag curves": one in `make_fig9_lag_sst` and one in the
# nested `_build_lag_curves` of `make_fig9_lag_sst_with_modes`. Only the
# second is reachable from main(), so a fix applied to the first was
# silently inert while the figures kept being produced by the second.
# Everything now routes through `_basin_lag_curves` below and the two
# entry points are thin wrappers, so the two cannot drift apart again.

_LAG_FIELD_CACHE: dict[str, dict] = {}


def _lag_fields_cached(
    chl_ts:      xr.DataArray,
    drv_ts:      xr.DataArray,
    common_grid: xr.DataArray,
    key:         str,
    period:      slice = LAG_PERIOD,
    want_z:      bool = False,
) -> dict:
    """Regrid, gap-fill and per-cell deseasonalise/detrend a Chl/driver pair.

    `make_fig9_lag_sst_with_modes` calls the curve builder five times
    (raw, all modes removed, then AMO/NINO34/PDO individually) and the
    regridding plus per-cell preprocessing is identical every time, so
    without this cache the same work is redone five times for eight
    sources.

    Arrays are stored as float32: the pair of anomaly fields for one
    source on a 1 deg grid over 204 months is ~105 MB in float64 and
    half that in float32, and the correlations are computed in float64
    from these anyway. `want_z` additionally retains the standardised
    fields and their per-cell sigma, which only the decomposition needs
    and therefore only OBS requests.
    """
    # `want_z` is deliberately NOT part of the key. Fig 9 asks for the
    # z-fields (it draws the decomposition) while Fig 9b does not, and
    # keying on it would make OBS go through the whole regrid and
    # per-cell detrend a second time for the same period and grid. The
    # entry is instead upgraded in place the first time the extra fields
    # are asked for.
    full_key = (
        f"{key}|{_da_fingerprint(chl_ts)}|{_da_fingerprint(drv_ts)}"
        f"|{tuple(common_grid.sizes.values())}"
        f"|{getattr(period, 'start', None)}-{getattr(period, 'stop', None)}"
    )
    hit = _LAG_FIELD_CACHE.get(full_key)
    if hit is not None and (not want_z or "chl_z" in hit):
        return hit

    chl_re, drv_re = _prep_for_lag(chl_ts, drv_ts, common_grid, period)
    chl_re, drv_re = xr.align(chl_re, drv_re, join="inner")

    chl_z, sig_c = _deseas_detrend_zscore(chl_re, return_sigma=True)
    drv_z, sig_d = _deseas_detrend_zscore(drv_re, return_sigma=True)

    def _f32(da):
        return da.astype("float32")

    out = {
        # per-cell anomalies in native units: the seasonal cycle and the
        # trend are removed BEFORE any spatial averaging
        "chl_anom": _f32(chl_z * sig_c),
        "drv_anom": _f32(drv_z * sig_d),
        # finite pattern of the regridded input, for the footprint test
        "chl_finite_frac": np.isfinite(chl_re).mean(dim="time"),
        "drv_finite_frac": np.isfinite(drv_re).mean(dim="time"),
    }
    if want_z:
        out.update({"chl_z": _f32(chl_z), "drv_z": _f32(drv_z),
                    "sig_c": sig_c, "sig_d": sig_d})

    _LAG_FIELD_CACHE[full_key] = out
    nbytes = sum(
        v.nbytes for entry in _LAG_FIELD_CACHE.values()
        for v in entry.values() if hasattr(v, "nbytes"))
    logger.info(
        f"    lag-field cache MISS '{full_key}' "
        f"({len(_LAG_FIELD_CACHE)} entries, {nbytes / 1e9:.2f} GB)")
    return out


def _footprint_from_fracs(*fracs, min_frac: float = 0.9) -> xr.DataArray:
    """Fixed footprint from precomputed finite-fraction maps."""
    mask = None
    for f in fracs:
        m = f >= min_frac
        mask = m if mask is None else (mask & m)
    return mask


def _basin_lag_curves(
    obs_chl_ts:   xr.DataArray,
    obs_drv_ts:   xr.DataArray | None,
    model_chl_ts: dict[str, xr.DataArray],
    model_drv_ts: dict[str, xr.DataArray],
    common_grid:  xr.DataArray,
    masks:        dict[str, xr.DataArray],
    driver:       str,
    remove:       list[str] | None = None,
    indices:      dict[str, xr.DataArray] | None = None,
    with_decomposition: bool = False,
    period:       slice = LAG_PERIOD,
    min_frac:     float = LAG_FOOTPRINT_MIN_FRAC,
) -> tuple[dict[str, dict[str, tuple]], dict[str, dict[str, np.ndarray]]]:
    """Basin-mean lag curves for OBS and every model, on a fixed footprint.

    The single implementation behind Figs 9, 9b, 10 and 11. Three
    properties distinguish it from the code it replaces:

      1. FIXED FOOTPRINT. Cells must be observed in at least `min_frac`
         of months, and the footprint derived from OBS is imposed on the
         models too. The previous `_basin_mean_timeseries` averaged with
         skipna=True over whatever cells each month happened to have, so
         the series carried a sampling signal that co-varied with the
         driver and was absent from the pointwise maps.
      2. ANOMALY-FIRST AVERAGING. The seasonal cycle and linear trend are
         removed per cell before the area average, not from the average
         afterwards. Global, Atlantic and Indian all straddle the
         equator, so the basin-mean seasonal cycle was previously the
         residue left after the two hemispheres partly cancelled.
      3. OPTIONAL DECOMPOSITION against the pointwise maps, returned as
         the second element and computed for OBS only.

    `remove` lists climate indices to regress out of both series before
    correlating; `indices` supplies them. The decomposition is only
    meaningful for the raw curves, since the pointwise maps it is
    compared against have no index removed, so it is skipped whenever
    `remove` is non-empty.
    """
    remove  = remove or []
    indices = indices or {}
    lag_curves: dict[str, dict[str, tuple]] = {b: {} for b in masks}
    aux_curves: dict[str, dict[str, np.ndarray]] = {}
    valid_mask: xr.DataArray | None = None

    want_z = bool(with_decomposition and not remove)

    def _curves_for_source(entry: dict, vm: xr.DataArray, label: str) -> None:
        for basin, mask in masks.items():
            sel = mask & vm
            if int(sel.sum()) < 10:
                if label == "OBS":
                    logger.warning(
                        f"  [OBS  ] {basin:10s}: <10 cells in the fixed "
                        f"footprint; skipping")
                continue
            chl_b = _basin_mean_static(entry["chl_anom"], mask, vm)
            drv_b = _basin_mean_static(entry["drv_anom"], mask, vm)
            if int(np.isfinite(chl_b.values).sum()) < 24:
                if label == "OBS":
                    logger.warning(
                        f"  [OBS  ] {basin:10s}: <24 months survive the "
                        f"min-weight test; consider lowering "
                        f"LAG_FOOTPRINT_MIN_FRAC (now {min_frac})")
                continue
            chl_z = _deseasonalise_zscore(chl_b)
            drv_z = _deseasonalise_zscore(drv_b)
            for idx_name in remove:
                if idx_name in indices:
                    chl_z = _regress_out(chl_z, indices[idx_name])
                    drv_z = _regress_out(drv_z, indices[idx_name])
            lag_curves[basin][label] = \
                _lag_correlation_with_bootstrap(chl_z, drv_z)

    # ---- OBS: defines the footprint ----
    if obs_drv_ts is not None:
        try:
            obs = _lag_fields_cached(
                obs_chl_ts, obs_drv_ts, common_grid,
                f"{driver}::OBS", period, want_z=want_z)
            valid_mask = _footprint_from_fracs(
                obs["chl_finite_frac"], obs["drv_finite_frac"],
                min_frac=min_frac)
            logger.info(
                f"  [OBS  ] fixed footprint: {int(valid_mask.sum())} cells "
                f"({100 * float(valid_mask.mean()):.1f}% of grid) at "
                f"min_frac={min_frac}")
            _curves_for_source(obs, valid_mask, "OBS")

            tag = f"remove={remove}" if remove else "raw"
            for basin in masks:
                if "OBS" not in lag_curves[basin]:
                    continue
                lags, rs = lag_curves[basin]["OBS"][0], lag_curves[basin]["OBS"][1]
                if not np.any(np.isfinite(rs)):
                    continue
                j = int(np.nanargmax(np.abs(rs)))
                logger.success(
                    f"  [OBS  ] {basin:10s} ({tag}): peak r={float(rs[j]):+.3f} "
                    f"@ lag={int(lags[j]):+d} "
                    f"(p={float(lag_curves[basin]['OBS'][4][j]):.3f})")

            if want_z:
                for basin, mask in masks.items():
                    if "OBS" not in lag_curves[basin]:
                        continue
                    try:
                        lags = lag_curves[basin]["OBS"][0]
                        dec = lag_curve_decomposition(
                            obs["drv_z"].astype("float64"),
                            obs["chl_z"].astype("float64"),
                            obs["sig_d"], obs["sig_c"],
                            mask, valid_mask, lags)
                        dec["lags"] = lags
                        aux_curves[basin] = dec

                        i0 = int(np.flatnonzero(lags == 0)[0])
                        ra, rd, rf = (dec["r_area"][i0], dec["r_diag"][i0],
                                      dec["r_full"][i0])
                        if np.all(np.isfinite([ra, rd, rf])):
                            logger.info(
                                f"  [OBS  ] {basin:10s} lag 0: map mean "
                                f"{ra:+.3f} -> var-weighted {rd:+.3f} -> "
                                f"basin mean {rf:+.3f}  [weighting "
                                f"{rd - ra:+.3f}, coherence {rf - rd:+.3f}]")
                        r_plot = lag_curves[basin]["OBS"][1][i0]
                        if (np.isfinite(rf) and np.isfinite(r_plot)
                                and abs(rf - r_plot) > 0.05):
                            logger.warning(
                                f"  [OBS  ] {basin:10s}: decomposition "
                                f"r_full {rf:+.3f} disagrees with the "
                                f"plotted OBS r {r_plot:+.3f} at lag 0 - "
                                f"the two are no longer computed on the "
                                f"same footprint")
                    except Exception as e:
                        logger.warning(
                            f"  [OBS  ] {basin} decomposition failed: {e}")
        except Exception as e:
            import traceback
            logger.warning(f"  OBS lag-corr failed: {e}")
            logger.warning(traceback.format_exc().replace("<", "\\<"))
    else:
        logger.warning(f"  OBS {driver} not available; skipping OBS curve")

    # ---- models: inherit the observational footprint ----
    for model in MODELS:
        if model not in model_chl_ts or model not in model_drv_ts:
            continue
        try:
            ent = _lag_fields_cached(
                model_chl_ts[model], model_drv_ts[model], common_grid,
                f"{driver}::{model}", period, want_z=False)
            vm = valid_mask
            if vm is None:
                vm = _footprint_from_fracs(
                    ent["chl_finite_frac"], ent["drv_finite_frac"],
                    min_frac=min_frac)
            _curves_for_source(ent, vm, model)
        except Exception as e:
            logger.warning(f"  [{model}] lag-corr failed: {e}")

    return lag_curves, aux_curves


def _make_lag_figure(
    lag_curves: dict[str, dict[str, tuple]],
    obs_label: str,
    driver_name: str,
    title: str,
    filename: str,
    fig_num: str,
    aux_curves: dict[str, dict[str, np.ndarray]] | None = None,
) -> None:
    """Plot lag-correlation curves as 6 subplots (one per basin).

    lag_curves: {basin: {source_name: (lags, r, ci_lo, ci_hi, pval)}}
                where source_name is "OBS" or a model name and the
                last four entries come from
                _lag_correlation_with_bootstrap. Backward
                compatibility: a 2-tuple (lags, r) is accepted but
                in that case no CI band or significance marker is
                drawn for the curve.

    aux_curves: optional {basin: {"r_area": arr, "r_diag": arr,
                "lags": arr}} from `lag_curve_decomposition`. When
                supplied, the area-weighted mean of the pointwise map is
                drawn as a thin dashed grey line and the
                variance-weighted diagonal as a dotted one. Plotting
                them next to the black OBS curve makes Fig 9 and
                Fig S_pointwise_corr legible against each other in a
                single glance, so a reader does not mistake the offset
                between the two figures for an inconsistency.

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

        # Decomposition overlay: what the pointwise map gives when read
        # as a plain area average (dashed), and what the basin mean
        # would give with the cross-cell covariances switched off
        # (dotted). The vertical distance from the dashed line to the
        # dotted one is the variance-weighting contribution; from the
        # dotted line to the black OBS curve is the spatial-coherence
        # contribution.
        if aux_curves and basin in aux_curves:
            aux    = aux_curves[basin]
            a_lags = aux.get("lags")
            if a_lags is not None:
                r_area = aux.get("r_area")
                r_diag = aux.get("r_diag")
                if r_area is not None and np.any(np.isfinite(r_area)):
                    ax.plot(a_lags, r_area, ls="--", lw=1.3,
                            color="0.35", alpha=0.95, zorder=9,
                            label="OBS: area mean of pointwise r")
                if r_diag is not None and np.any(np.isfinite(r_diag)):
                    ax.plot(a_lags, r_diag, ls=":", lw=1.5,
                            color="0.35", alpha=0.95, zorder=9,
                            label="OBS: variance-weighted, diagonal only")

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
                "Chl-a anomaly (standardised)", fontsize=15
            )
            cb.ax.tick_params(labelsize=14)

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
        """North Atlantic SST-Chl lag correlation with idx_name removed.

        Delegates to `_basin_lag_curves` restricted to the Atlantic mask.
        This was a fourth copy of the basin-mean computation, and it also
        skipped `_prep_for_lag` entirely — calling `regrid_timeseries`
        directly, so it never received the gap interpolation the other
        paths use. Since this panel carries the AMO partial-correlation
        result, having it on different preprocessing from Fig 9b was the
        least defensible of the four.
        """
        try:
            curves, _ = _basin_lag_curves(
                obs_chl_ts, obs_sst_ts, model_chl_ts, model_sst_ts,
                common_grid, {"Atlantic": atl_mask}, driver="SST",
                remove=[idx_name], indices=indices,
                with_decomposition=False, period=period,
            )
            return curves.get("Atlantic", {})
        except Exception as e:
            logger.warning(f"  Atlantic ({idx_name}) failed: {e}")
            return {}

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

def _write_lag_peak_table(
    lag_curves: dict[str, dict[str, tuple]],
    driver:     str,
    tag:        str,
) -> None:
    """Tabulate peak lag correlations for every basin and source.

    Two numbers are reported per model, because they answer different
    questions and the distinction matters for the manuscript:

      r_own / lag_own  each model's OWN peak and where it falls. A
                       model can have a strong correlation at the
                       wrong lag, which is a different failure from
                       having no correlation at all.
      r_at_obs         each model's correlation AT THE OBSERVED peak
                       lag. This is the like-for-like comparison and
                       is the number to quote when saying a model does
                       or does not reproduce the observed coupling.

    Writes a CSV next to the figures and echoes a formatted table to
    the log so the values can be read straight into the text.
    """
    rows = []
    for basin, curves in lag_curves.items():
        if "OBS" not in curves:
            continue
        lags, r_obs, lo, hi, p_obs = curves["OBS"]
        lags = np.asarray(lags)
        k_obs = int(np.nanargmax(np.abs(r_obs)))
        lag_obs = int(lags[k_obs])

        rows.append({
            "basin": basin, "source": "OBS",
            "r_own": float(r_obs[k_obs]), "lag_own": lag_obs,
            "p_own": float(p_obs[k_obs]),
            "ci_lo": float(lo[k_obs]), "ci_hi": float(hi[k_obs]),
            "r_at_obs_lag": float(r_obs[k_obs]),
            "p_at_obs_lag": float(p_obs[k_obs]),
        })

        for src, entry in curves.items():
            if src == "OBS":
                continue
            l_m, r_m, lo_m, hi_m, p_m = entry
            k_m = int(np.nanargmax(np.abs(r_m)))
            j = int(np.argmin(np.abs(np.asarray(l_m) - lag_obs)))
            rows.append({
                "basin": basin, "source": src,
                "r_own": float(r_m[k_m]), "lag_own": int(l_m[k_m]),
                "p_own": float(p_m[k_m]),
                "ci_lo": float(lo_m[k_m]), "ci_hi": float(hi_m[k_m]),
                "r_at_obs_lag": float(r_m[j]),
                "p_at_obs_lag": float(p_m[j]),
            })

    if not rows:
        return

    import csv
    fname = OUT_DIR / f"table_lagpeak_{driver}{tag}.csv"
    with open(fname, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.success(f"  wrote {fname.name}")

    logger.info(
        f"  === Peak lag correlations: Chl-a vs {driver}{tag} ===")
    logger.info(
        f"  {'basin':<12s} {'source':<18s} "
        f"{'r_own':>7s} {'lag':>5s} {'p':>7s} "
        f"{'95% CI':>16s} {'r@obs_lag':>10s} {'p@obs':>7s}")
    cur = None
    for r in rows:
        if r["basin"] != cur:
            cur = r["basin"]
            logger.info("  " + "-" * 84)
        star = " *" if r["p_own"] < 0.05 else "  "
        logger.info(
            f"  {r['basin']:<12s} {r['source']:<18s} "
            f"{r['r_own']:+7.3f} {r['lag_own']:+5d} "
            f"{r['p_own']:7.3f}{star}"
            f"[{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] "
            f"{r['r_at_obs_lag']:+10.3f} {r['p_at_obs_lag']:7.3f}")


def _write_lag_decomposition_table(
    aux_curves: dict[str, dict[str, np.ndarray]],
    driver:     str,
    tag:        str = "",
) -> None:
    """Tabulate the basin-mean / pointwise-map attribution per basin.

    For each basin and each lag the table carries three correlations:

      r_area   area-weighted mean of the pointwise map — the number a
               reader estimates by eye from Fig S_pointwise_corr
      r_diag   the same quantity after variance weighting, i.e. the
               basin-mean correlation with all cross-cell covariance
               removed
      r_full   the plotted basin-mean correlation

    and the two differences that partition the gap between the figures:
    `d_weighting` = r_diag - r_area and `d_coherence` = r_full - r_diag.
    Whichever dominates determines how the discrepancy is described in
    the text — a weighting artefact belongs in the methods, whereas a
    coherence-dominated gap is a statement about large-scale coupled
    modes and belongs in the results.
    """
    if not aux_curves:
        return

    rows = []
    for basin, dec in aux_curves.items():
        lags = np.asarray(dec.get("lags", []))
        if lags.size == 0:
            continue
        for i, k in enumerate(lags):
            ra = float(dec["r_area"][i])
            rd = float(dec["r_diag"][i])
            rf = float(dec["r_full"][i])
            rows.append({
                "basin": basin, "lag": int(k),
                "r_area": ra, "r_diag": rd, "r_full": rf,
                "d_weighting": rd - ra,
                "d_coherence": rf - rd,
            })

    if not rows:
        return

    import csv
    fname = OUT_DIR / f"table_lagdecomp_{driver}{tag}.csv"
    with open(fname, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.success(f"  wrote {fname.name}")

    logger.info(
        f"  === Basin-mean vs pointwise attribution: Chl-a vs "
        f"{driver}{tag} (lag 0 and each basin's OBS peak) ===")
    logger.info(
        f"  {'basin':<12s} {'lag':>5s} {'r_area':>8s} {'r_diag':>8s} "
        f"{'r_full':>8s} {'weight':>8s} {'coher':>8s}  dominant")
    logger.info("  " + "-" * 74)
    for basin, dec in aux_curves.items():
        lags = np.asarray(dec.get("lags", []))
        if lags.size == 0:
            continue
        show = [int(np.flatnonzero(lags == 0)[0])] if np.any(lags == 0) else []
        rf_all = dec["r_full"]
        if np.any(np.isfinite(rf_all)):
            k_pk = int(np.nanargmax(np.abs(rf_all)))
            if k_pk not in show:
                show.append(k_pk)
        for i in show:
            ra, rd, rf = (dec["r_area"][i], dec["r_diag"][i],
                          dec["r_full"][i])
            if not np.all(np.isfinite([ra, rd, rf])):
                continue
            dw, dc = rd - ra, rf - rd
            dom = "weighting" if abs(dw) >= abs(dc) else "coherence"
            logger.info(
                f"  {basin:<12s} {int(lags[i]):+5d} {ra:+8.3f} {rd:+8.3f} "
                f"{rf:+8.3f} {dw:+8.3f} {dc:+8.3f}  {dom}")


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
    ) -> tuple[dict[str, dict[str, tuple]], dict[str, dict[str, np.ndarray]]]:
        """Compute basin-mean SST-Chl lag curves.

        remove: list of index names (subset of indices keys) to regress
                out of both Chl and SST before correlation.

        Thin wrapper over `_basin_lag_curves`. This function used to hold
        its own copy of the whole computation, which is how it came to
        differ from `make_fig9_lag_sst`: a fix applied there never
        reached the figures, because main() calls this path and not that
        one. Both now share one implementation.
        """
        return _basin_lag_curves(
            obs_chl_ts, obs_sst_ts, model_chl_ts, model_sst_ts,
            common_grid, masks, driver="SST",
            remove=remove, indices=indices,
            with_decomposition=True,
        )

    # ------------------------------------------------------------------
    # Fig 9 — raw (no removal)
    # ------------------------------------------------------------------
    raw_curves, raw_aux = _build_lag_curves(remove=[])
    _write_lag_peak_table(raw_curves, "SST", "")
    _write_lag_decomposition_table(raw_aux, driver="SST")
    _make_lag_figure(
        raw_curves,
        obs_label="ESA-CCI Chl vs NOAA-ERSSTv3b SST",
        driver_name="SST",
        title="Lag correlation: basin-mean Chl-a vs SST anomalies",
        filename="fig9_lag_corr_chl_sst.png",
        fig_num="9",
        aux_curves=raw_aux or None,
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
    # No decomposition overlay here: the pointwise maps it would be
    # compared against have no index removed, so the two would not be
    # like for like.
    removed_curves, _ = _build_lag_curves(remove=all_names)
    label_removed = " + ".join(all_names)
    _write_lag_peak_table(
        removed_curves, "SST", f"_minus_{'_'.join(all_names)}")
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

    # Pre-compute Atlantic curves for each panel spec. This was a third
    # copy of the same computation; it now shares `_basin_lag_curves`
    # with Figs 9 and 9b, restricted to the Atlantic mask, so all three
    # panels of the paper rest on identical preprocessing.
    atl_only = {"Atlantic": atl_mask}
    panel_curves: list[dict[str, tuple]] = []
    for panel_title, remove in panel_specs:
        try:
            curves, _ = _basin_lag_curves(
                obs_chl_ts, obs_sst_ts, model_chl_ts, model_sst_ts,
                common_grid, atl_only, driver="SST",
                remove=remove, indices=indices,
                with_decomposition=False,
            )
            panel_curves.append(curves.get("Atlantic", {}))
        except Exception as e:
            logger.warning(f"  Atlantic ({panel_title}) failed: {e}")
            panel_curves.append({})

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
    """Fig 9 without the climate-mode panels: thin wrapper.

    Retained for callers that want the raw six-basin figure on its own.
    main() uses `make_fig9_lag_sst_with_modes` instead, which produces
    this figure plus Fig 9b and Fig S_amo.

    This function previously carried its own copy of the basin-mean
    computation. That copy diverged from the one actually used, so a
    correction applied here had no effect on any published figure while
    appearing to. It now delegates to `_basin_lag_curves` like every
    other entry point, and holds no numerics of its own.
    """
    logger.info("=== Fig 9: lag correlation Chl vs SST (basin-mean) ===")
    masks = _build_masks(common_grid)

    lag_curves, aux_curves = _basin_lag_curves(
        obs_chl_ts, obs_sst_ts, model_chl_ts, model_sst_ts,
        common_grid, masks, driver="SST",
        with_decomposition=True,
    )

    _make_lag_figure(
        lag_curves,
        obs_label="ESA-CCI Chl vs NOAA-ERSSTv3b SST",
        driver_name="SST",
        title="Lag correlation: basin-mean Chl-a vs SST anomalies",
        filename="fig9_lag_corr_chl_sst.png",
        fig_num="9",
        aux_curves=aux_curves or None,
    )
    _write_lag_decomposition_table(aux_curves, driver="SST")

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

    Implementation note. This used to be a Python double loop over every
    (lat, lon) cell, which cost tens of seconds per call on a global 1°
    grid. That was tolerable while only the pointwise figures used it,
    but `_prep_for_lag` now applies the same interpolation so the
    basin-mean and pointwise diagnostics start from identical data, and
    at ~20 call sites times seven models the loop would have dominated
    the run. The vectorised form below is numerically identical: for
    each time step it locates the nearest finite value before and after
    along the time axis, fills only interior gaps no longer than
    max_gap, and never extrapolates past the first or last observation.
    """
    arr = np.asarray(ts.values, dtype=float).copy()
    T   = arr.shape[0]
    flat = arr.reshape(T, -1)                  # (T, ncell)
    fin  = np.isfinite(flat)

    idx = np.arange(T)[:, None]

    # Index of the nearest finite sample at or before each time step
    prev_idx = np.maximum.accumulate(np.where(fin, idx, -1), axis=0)
    # ... and at or after (accumulate the reversed array)
    nxt_idx = np.flip(
        np.minimum.accumulate(np.flip(np.where(fin, idx, T), axis=0), axis=0),
        axis=0,
    )

    has_both = (prev_idx >= 0) & (nxt_idx < T)      # interior only
    gap_len  = nxt_idx - prev_idx - 1               # NaNs between them
    fillable = (~fin) & has_both & (gap_len <= max_gap) & (gap_len > 0)

    if fillable.any():
        cols = np.broadcast_to(np.arange(flat.shape[1])[None, :], flat.shape)
        p = np.where(has_both, prev_idx, 0)
        q = np.where(has_both, nxt_idx, 0)
        v0 = flat[p, cols]
        v1 = flat[q, cols]
        span = np.where(q > p, q - p, 1)
        frac = (idx - p) / span
        interp = v0 + frac * (v1 - v0)
        flat[fillable] = interp[fillable]

    return ts.copy(data=flat.reshape(arr.shape))


def _deseas_detrend_zscore(
    ts: xr.DataArray,
    return_sigma: bool = False,
):
    """Deseasonalise, linearly detrend, and z-score a (time,lat,lon) field.

    std is computed only over finite values per cell so that polar cells
    with all-NaN winters (polar night) are not masked out — their
    summer-only std is non-zero and valid. Cells where every value is NaN
    are left as NaN throughout.

    Uses numpy-based monthly climatology (via _ym_key) rather than
    xarray groupby so that cftime calendars (noleap, 360-day, etc.)
    are handled correctly without requiring a pandas-compatible time axis.

    DETRENDING (corrected). The previous implementation computed the
    trend slope as

        num   = nansum(t_c * vals, axis=0)
        denom = nansum(t_c**2)            # over ALL T, every cell
        slope = num / denom

    with t_c centred once over the full time axis. That is only correct
    for cells with a complete record, and two things went wrong wherever
    data were missing:

      (a) the denominator counted months the cell did not actually have,
          so sum(t^2) was too large and the slope was biased towards
          zero — i.e. gappy cells were UNDER-detrended, and gappiness is
          worst exactly in the polar and high-cloud regions where the
          Chl-a and SST trends are strongest;
      (b) t_c was centred globally rather than over each cell's own
          finite subset. When missingness is seasonally clustered
          (polar night, persistent cloud) sum(t_c) over the valid months
          is not zero, so the cell's residual mean leaks into the slope.

    Both are repaired by solving the least-squares line per cell over its
    own finite mask, which is what the 1-D `_detrend_linear` (used for
    the basin-mean series) already did via np.polyfit. The two paths are
    now consistent, which matters because Figs 9-12 compare a basin-mean
    diagnostic against a pointwise one and any difference in the
    detrending shows up as an apparent disagreement between them.

    Cells with fewer than 24 finite months are left undetrended rather
    than fitted on a handful of points; their std is still computed and
    they are still standardised.

    With return_sigma=True the pre-standardisation residual std is
    returned alongside the z-scored field. `lag_curve_decomposition`
    needs it to reconstruct the physical (un-standardised) anomalies
    from the z-scored field.
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
            with np.errstate(invalid="ignore"):
                clim_arr[m-1] = np.nanmean(arr[sel], axis=0)

    # Subtract climatology
    anom_arr = arr.copy()
    for m in range(1, 13):
        sel = (mo == m)
        if sel.any():
            anom_arr[sel] -= clim_arr[m-1]
    anom_arr[~np.isfinite(arr)] = np.nan

    # --- per-cell linear detrend over each cell's own finite subset ---
    ok = np.isfinite(anom_arr)                     # (T, nlat, nlon)
    n  = ok.sum(axis=0).astype(float)              # (nlat, nlon)

    t    = np.arange(T, dtype=float)[:, None, None]
    t_ok = np.where(ok, t, 0.0)
    y_ok = np.where(ok, anom_arr, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        t_bar = t_ok.sum(axis=0) / np.maximum(n, 1.0)
        y_bar = y_ok.sum(axis=0) / np.maximum(n, 1.0)

        dt = np.where(ok, t - t_bar[None], 0.0)
        dy = np.where(ok, anom_arr - y_bar[None], 0.0)

        sxy   = (dt * dy).sum(axis=0)
        sxx   = (dt * dt).sum(axis=0)              # per-cell, finite-only
        slope = np.where(sxx > 0, sxy / np.maximum(sxx, 1e-12), 0.0)

        det = anom_arr - slope[None] * dt
    det[~ok] = np.nan

    # Too few points to fit a trend: keep the plain anomaly.
    thin = n < 24
    if thin.any():
        det[:, thin] = anom_arr[:, thin]

    with np.errstate(invalid="ignore"):
        std = np.nanstd(det, axis=0, ddof=0)
    good = np.isfinite(std) & (std > 0)
    nrm  = np.where(good[None, :, :],
                    det / np.where(good, std, 1.0)[None, :, :],
                    np.nan)

    out = ts.copy(data=nrm)
    if not return_sigma:
        return out

    sig_da = xr.DataArray(
        np.where(good, std, np.nan),
        dims=("lat", "lon"),
        coords={"lat": ts.lat, "lon": ts.lon},
        name="sigma",
    )
    return out, sig_da


# =====================================================================
# Pointwise (grid-point) lag correlation between Chl-a and a physical
# driver. Global analogue of make_figS_seaice_chl_spatial_corr, which
# is restricted to the polar caps.
# =====================================================================

# Lags shown by the pointwise driver maps, in months, with the DRIVER
# leading Chl-a at positive lag. Negative lags are included because the
# basin-mean diagnostic finds the observed North Atlantic SST-Chl-a
# peak at lag -8 (Chl-a leading), which a positive-only set of columns
# could not show.
PTCORR_LAGS = [-8, -4, 0, 4]

# False-discovery-rate level for the pointwise significance test. A
# global 1 deg map has ~4.5e4 ocean cells, so at a nominal p<0.05 some
# 2000 cells would be flagged by chance alone; stippling those would
# misrepresent noise as signal. The Benjamini-Hochberg procedure below
# controls the expected proportion of false positives among the cells
# actually flagged.
PTCORR_FDR_Q = 0.10


def _effective_n(x: np.ndarray, y: np.ndarray, axis: int = 0) -> np.ndarray:
    """Effective sample size after Bretherton et al. (1999).

    N_eff = N (1 - r1x r1y) / (1 + r1x r1y), with r1 the lag-1
    autocorrelation of each series. Monthly ocean anomalies are
    strongly autocorrelated, so the nominal N = 204 months badly
    overstates the information content; using it directly would make
    the pointwise significance test far too liberal. This is the same
    correction the manuscript applies to the basin-mean series.
    """
    n = np.sum(np.isfinite(x) & np.isfinite(y), axis=axis)

    def _r1(a):
        a0 = a[:-1]
        a1 = a[1:]
        ok = np.isfinite(a0) & np.isfinite(a1)
        cnt = ok.sum(axis=axis)
        a0 = np.where(ok, a0, 0.0)
        a1 = np.where(ok, a1, 0.0)
        m0 = np.where(cnt > 0, a0.sum(axis=axis) / np.maximum(cnt, 1), 0.0)
        m1 = np.where(cnt > 0, a1.sum(axis=axis) / np.maximum(cnt, 1), 0.0)
        d0 = np.where(ok, a0 - m0, 0.0)
        d1 = np.where(ok, a1 - m1, 0.0)
        num = (d0 * d1).sum(axis=axis)
        den = np.sqrt((d0 ** 2).sum(axis=axis) * (d1 ** 2).sum(axis=axis))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, 0.0)

    r1x = np.clip(_r1(x), -0.99, 0.99)
    r1y = np.clip(_r1(y), -0.99, 0.99)
    prod = r1x * r1y
    with np.errstate(invalid="ignore", divide="ignore"):
        neff = n * (1.0 - prod) / (1.0 + prod)
    # Never claim more information than the record actually contains.
    return np.clip(neff, 3.0, n)


def _fdr_mask(pvals: np.ndarray, q: float = PTCORR_FDR_Q) -> np.ndarray:
    """Benjamini-Hochberg step-up procedure over all finite cells."""
    flat = pvals.ravel()
    ok = np.isfinite(flat)
    out = np.zeros_like(flat, dtype=bool)
    if ok.sum() == 0:
        return out.reshape(pvals.shape)
    p = flat[ok]
    order = np.argsort(p)
    ps = p[order]
    m = ps.size
    thresh = q * np.arange(1, m + 1) / m
    passed = ps <= thresh
    if passed.any():
        k = np.max(np.flatnonzero(passed))
        keep = np.zeros(m, dtype=bool)
        keep[order[: k + 1]] = True
        out[np.flatnonzero(ok)] = keep
    return out.reshape(pvals.shape)


def _ptcorr_r_and_p(
    drv_z: xr.DataArray,
    chl_z: xr.DataArray,
    lag:   int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell Pearson r at one lag, with an Neff-adjusted p-value.

    Distinct from _pointwise_lag_corr, which serves the regional
    contourf figure: that one returns a boolean mask already
    thresholded at p < 0.05, whereas the Benjamini-Hochberg correction
    used here needs the p-values themselves. The argument order
    (driver, Chl-a) deliberately matches _pointwise_lag_corr so the
    two cannot be confused at a call site.

    Sign convention matches _lag_correlation exactly: positive lag
    means the DRIVER leads Chl-a. Complete-case per cell, so seasonal
    gaps in ocean colour are handled the same way as elsewhere.
    """
    from scipy import stats as _st

    ym_c = _ym_key(chl_z)
    ym_d = _ym_key(drv_z)
    common = np.intersect1d(ym_c, ym_d)
    if common.size < 36:
        raise ValueError(f"only {common.size} common months")
    ic = np.argsort(ym_c)[np.searchsorted(np.sort(ym_c), common)]
    id_ = np.argsort(ym_d)[np.searchsorted(np.sort(ym_d), common)]

    C = np.asarray(chl_z.values, dtype=float)[ic]
    D = np.asarray(drv_z.values, dtype=float)[id_]
    T = C.shape[0]

    # Identical fields are not a physical result, they mean the same
    # data reached both arguments. A cache-key collision did exactly
    # that once, and produced a uniformly saturated map that looked
    # superficially plausible. Check before lagging, where the two
    # fields are directly comparable.
    if C.shape == D.shape and np.allclose(np.nan_to_num(C),
                                          np.nan_to_num(D), atol=1e-8):
        raise ValueError(
            "driver and Chl-a are the same field - check the cache keys")

    k = int(lag)
    if k > 0:
        X, Y = D[: T - k], C[k:]
    elif k < 0:
        kk = -k
        X, Y = D[kk:], C[: T - kk]
    else:
        X, Y = D, C

    if X.shape != Y.shape:
        raise ValueError(
            f"driver {X.shape} and Chl-a {Y.shape} disagree after "
            f"lagging; they are probably on different grids")

    ok = np.isfinite(X) & np.isfinite(Y)
    n  = ok.sum(axis=0)
    Xz = np.where(ok, X, 0.0)
    Yz = np.where(ok, Y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = Xz.sum(axis=0) / np.maximum(n, 1)
        my = Yz.sum(axis=0) / np.maximum(n, 1)
        dx = np.where(ok, X - mx[None], 0.0)
        dy = np.where(ok, Y - my[None], 0.0)
        num = (dx * dy).sum(axis=0)
        den = np.sqrt((dx ** 2).sum(axis=0) * (dy ** 2).sum(axis=0))
        r = np.where((den > 0) & (n >= 36), num / den, np.nan)

    neff = _effective_n(np.where(ok, X, np.nan), np.where(ok, Y, np.nan))
    dof  = np.maximum(neff - 2.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        tstat = r * np.sqrt(dof / np.maximum(1.0 - r ** 2, 1e-12))
    p = np.where(np.isfinite(r),
                 2.0 * _st.t.sf(np.abs(tstat), dof), np.nan)
    return r, p


# =====================================================================
# Spatially filtered, basin-resolved lag-0 correlation maps
# =====================================================================
#
# Replaces the global pointwise maps (make_figS_pointwise_driver_corr).
# Three things were wrong with those:
#
#   1. ATTENUATION. Each 1 deg anomaly carries small-scale variability
#      that is uncorrelated between Chl-a and the driver (mesoscale,
#      retrieval noise, cloud-gap artefacts). Correlating two noisy
#      fields biases r toward zero. With N_eff ~110 the null sd of r at
#      a single cell is ~0.095, so most of the colour range of those
#      maps sat inside the noise distribution.
#   2. INCOMPARABLE ROWS. The ensemble mean of seven r-fields has its
#      noise reduced by ~sqrt(7) relative to the single observational
#      realisation, but shared a colour scale with it. The observations
#      were guaranteed to look patchier and stronger than the models
#      whatever the underlying physics.
#   3. NO SEASONAL INFORMATION. A single annual correlation hides that
#      the sign of the Chl-a/SST relationship reverses between the
#      light-limited and nutrient-limited seasons in mid latitudes.
#
# The replacement applies a spatial filter before correlating, which
# removes the uncorrelated small scales and recovers most of the true
# large-scale coupling, and shows the models individually rather than
# only as a mean.

# ---------------------------------------------------------------------
# Significance marking convention, used by EVERY correlation figure here
# ---------------------------------------------------------------------
# Stippling (dots) ALWAYS means "this correlation is significant". It
# never means anything else, and nothing else is ever marked.
#
#   OBS panels      dots where the correlation passes a
#                   Benjamini-Hochberg FDR test at q = PTCORR_FDR_Q on
#                   the autocorrelation-adjusted effective sample size.
#   Per-model panels dots on the same test applied to that model.
#   Ensemble panels dots where at least two thirds of the models with
#                   data at that cell are individually significant AND
#                   share the sign of the ensemble mean.
#
# The ensemble criterion is deliberately phrased in the positive. An
# earlier version hatched the ensemble where models DISAGREED, so marks
# meant "reliable" in the top row and "unreliable" in the row directly
# beneath it - the same figure carrying two opposite visual grammars.
# Marking robustness rather than its absence loses no information: the
# unmarked area of an ensemble panel is exactly the area where the
# models fail to agree significantly.

# Full width of the spatial box filter, in degrees. Applied to the
# standardised anomalies of BOTH fields before correlating. 6 deg is a
# compromise: wide enough to average away the mesoscale and most
# retrieval noise, narrow enough to keep the subpolar/subtropical
# contrast that motivates the analysis.
SPATIAL_FILTER_DEG = 6.0

# Minimum fraction of a filter window that must be valid ocean for the
# filtered value to be kept. Cells whose 6 deg box is mostly land or
# permanently unobserved are set to NaN rather than being filled from
# one or two surviving neighbours.
SPATIAL_FILTER_MIN_VALID = 0.20

# Basins shown, one column per basin, in this order.
#
# "Global" is deliberately excluded. As a panel it is just a global map,
# which is the thing these figures replace, and it is not independent of
# the other columns: it is their union, so it adds no information while
# taking the widest slot in the row. Worse, a global panel mixes basins
# whose correlations have opposite sign, so it reads as weak coupling
# wherever the hemispheres or basins disagree - the same cancellation
# that motivated resolving the analysis by basin in the first place.
# The basin masks themselves are unchanged; only the plotted set differs.
CORR_BASINS = [b for b in MEANSTATE_BASINS if b != "Global"]

# Minimum number of years contributing to a single-calendar-month
# correlation in the Hovmoller figure.
SEASONAL_MIN_YEARS = 10


def _running_sum(a: np.ndarray, half: int, axis: int,
                 periodic: bool) -> np.ndarray:
    """Sliding-window sum of width 2*half+1 along `axis`.

    Uses a cumulative sum, so cost is independent of the window width.
    `periodic` wraps the window around the axis, which is what longitude
    needs and latitude must not have.
    """
    n = a.shape[axis]
    if half <= 0:
        return a.copy()
    if periodic and 2 * half + 1 >= n:
        # Window covers the whole circle: every point gets the total.
        tot = a.sum(axis=axis, keepdims=True)
        return np.broadcast_to(tot, a.shape).copy()
    h = half if periodic else min(half, n - 1)

    pad = [(0, 0)] * a.ndim
    pad[axis] = (h, h)
    ap = np.pad(a, pad, mode=("wrap" if periodic else "constant"))

    zshape = list(ap.shape)
    zshape[axis] = 1
    c = np.concatenate(
        [np.zeros(zshape, dtype=ap.dtype), np.cumsum(ap, axis=axis)],
        axis=axis,
    )
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(2 * h + 1, 2 * h + 1 + n)
    lo[axis] = slice(0, n)
    return c[tuple(hi)] - c[tuple(lo)]


def _spatial_filter(
    da:            xr.DataArray,
    width_deg:     float = SPATIAL_FILTER_DEG,
    isotropic:     bool = True,
    min_valid:     float = SPATIAL_FILTER_MIN_VALID,
    restrict_to_input: bool = True,
) -> xr.DataArray:
    """NaN-aware, area-weighted box filter on a (time, lat, lon) field.

    The window is `width_deg` across in latitude. With `isotropic=True`
    the longitude window is widened by 1/cos(lat) so the filter covers
    roughly the same physical distance at every latitude; without it a
    6 deg box at 70 N would span only ~2 deg of great-circle distance in
    longitude and the filter would be far weaker in the polar basins
    than in the tropics, which is exactly where the noise problem is
    worst. The widening is capped at half the grid so the window never
    exceeds the full circle.

    Missing values are handled by weighting: the filtered value is the
    area-weighted mean over the valid cells inside the window. A cell
    whose window is less than `min_valid` valid ocean is returned NaN,
    so coastal and permanently unobserved cells are not filled from one
    or two distant neighbours.

    `restrict_to_input` returns NaN at any cell that never had data in
    the input, and it matters more than it sounds. Because the filter
    fills any window that is at least `min_valid` valid, it otherwise
    writes finite values two to three cells inland from every coastline
    - a 6 deg window centred on a coastal land cell still catches plenty
    of ocean. Two things go wrong without the restriction. The visible
    one is that significance and model-disagreement marks appear over
    land, since `np.isfinite` stops being a usable proxy for "ocean".
    The substantive one is that those coastal cells carry correlations
    computed from anomalies averaged ACROSS the land boundary, mixing
    shelf and open-ocean signal in a way no reader would expect from a
    map that looks like it stops at the coast.

    Time gaps are still filled from spatial neighbours, which is the
    point of the filter; only cells that are missing at EVERY time step
    are removed, so a cloudy month is interpolated while a continent is
    not invented.

    Longitude wraps; latitude does not.
    """
    lat = np.asarray(da.lat.values, dtype=float)
    lon = np.asarray(da.lon.values, dtype=float)
    nlat, nlon = lat.size, lon.size
    dlat = float(np.abs(np.median(np.diff(lat)))) if nlat > 1 else 1.0
    dlon = float(np.abs(np.median(np.diff(lon)))) if nlon > 1 else 1.0

    arr = np.asarray(da.values, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"_spatial_filter expects (time,lat,lon), got {arr.shape}")

    valid = np.isfinite(arr)
    w1d = np.cos(np.deg2rad(np.clip(lat, -89.9, 89.9)))
    w = np.broadcast_to(w1d[None, :, None], arr.shape)

    num = np.where(valid, arr, 0.0) * w      # weighted values
    den = np.where(valid, 1.0, 0.0) * w      # weights present
    cnt = np.where(valid, 1.0, 0.0)          # unweighted count

    half_lat = int(round(width_deg / 2.0 / dlat))

    # --- longitude pass (periodic; window may vary with latitude) ---
    if isotropic:
        cosl = np.cos(np.deg2rad(np.clip(lat, -89.9, 89.9)))
        cosl = np.maximum(cosl, 1e-3)
        half_lon_row = np.minimum(
            np.round(width_deg / 2.0 / dlon / cosl).astype(int),
            max(nlon // 2, 1),
        )
    else:
        half_lon_row = np.full(nlat, int(round(width_deg / 2.0 / dlon)))

    nom_lon = np.zeros(nlat)
    for h in np.unique(half_lon_row):
        rows = np.flatnonzero(half_lon_row == h)
        nom_lon[rows] = min(2 * int(h) + 1, nlon)
        sl = (slice(None), rows, slice(None))
        num[sl] = _running_sum(num[sl], int(h), axis=2, periodic=True)
        den[sl] = _running_sum(den[sl], int(h), axis=2, periodic=True)
        cnt[sl] = _running_sum(cnt[sl], int(h), axis=2, periodic=True)

    # --- latitude pass (not periodic) ---
    num = _running_sum(num, half_lat, axis=1, periodic=False)
    den = _running_sum(den, half_lat, axis=1, periodic=False)
    cnt = _running_sum(cnt, half_lat, axis=1, periodic=False)

    # Nominal window size per latitude, accounting for truncation at the
    # poles, so the min_valid test is not unfairly strict at the edges.
    rows_lat = np.arange(nlat)
    lo = np.maximum(rows_lat - half_lat, 0)
    hi = np.minimum(rows_lat + half_lat, nlat - 1)
    nom = (hi - lo + 1).astype(float) * nom_lon
    nom = np.maximum(nom, 1.0)[None, :, None]

    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where((den > 0) & (cnt >= min_valid * nom),
                       num / np.where(den > 0, den, 1.0), np.nan)

    if restrict_to_input:
        ever = valid.any(axis=0)                     # (nlat, nlon)
        out = np.where(ever[None, :, :], out, np.nan)
    return da.copy(data=out)


def _restandardise(da: xr.DataArray) -> xr.DataArray:
    """Rescale each cell to unit variance in time, preserving NaN.

    Applied after `_spatial_filter`, which reduces the variance of every
    cell by an amount that depends on how much of its local variability
    was small-scale. Without this step the filtered field would carry a
    spatially varying amplitude that has nothing to do with the physics.
    Correlation is scale-invariant so this does not change any r, but it
    keeps the intermediate fields interpretable and lets the same arrays
    be reused for regression diagnostics.
    """
    arr = np.asarray(da.values, dtype=float)
    with np.errstate(invalid="ignore"):
        sd = np.nanstd(arr, axis=0, ddof=0)
    ok = np.isfinite(sd) & (sd > 0)
    out = np.where(ok[None], arr / np.where(ok, sd, 1.0)[None], np.nan)
    return da.copy(data=out)


def _da_fingerprint(da: xr.DataArray) -> str:
    """Cheap content fingerprint of a DataArray, for cache keys.

    The caches in this module are keyed by a human-written name such as
    "chl::OBS". That is fine while one field per name reaches them, but
    it fails silently the moment two different fields share a name - the
    second caller gets the first caller's data and the figure looks
    plausible. This module already carries a scar from that failure mode:
    `_ptcorr_r_and_p` raises if the driver and Chl-a arrays turn out to be
    identical, because a cache-key collision once produced a uniformly
    saturated map that passed visual inspection.

    Including a fingerprint of the contents makes a collision impossible
    rather than unlikely. A strided subsample of at most a few thousand
    values is enough to separate any two real fields, and costs far less
    than the regrid the cache exists to avoid.
    """
    v = np.asarray(da.values)
    flat = v.reshape(-1)
    step = max(1, flat.size // 4096)
    sub = np.asarray(flat[::step], dtype="float64")
    sub = np.where(np.isfinite(sub), sub, 0.0)
    n_fin = int(np.isfinite(flat[::step]).sum())
    h = hash((v.shape, n_fin, float(sub.sum()), float(np.abs(sub).max())
              if sub.size else 0.0))
    return f"{v.shape}:{h & 0xFFFFFFFF:08x}"


_FILT_CACHE: dict[str, xr.DataArray] = {}


def _filtered_z(
    ts:          xr.DataArray,
    common_grid: xr.DataArray,
    period:      slice,
    key:         str,
    width_deg:   float = SPATIAL_FILTER_DEG,
    interp_gaps: bool = False,
) -> xr.DataArray:
    """Regrid, deseasonalise, detrend, z-score, spatially filter, rescale.

    The order matters. Standardising BEFORE the filter means each cell
    contributes equally to its neighbourhood average; filtering the raw
    anomalies instead would let a single high-variance cell (a coastal
    bloom, say) dominate a whole 6 deg box, which is the same weighting
    pathology that separates the basin-mean curves from the pointwise
    maps, reproduced at small scale.
    """
    full_key = (f"{key}|{_da_fingerprint(ts)}"
                f"|{width_deg}|{tuple(common_grid.sizes.values())}"
                f"|{getattr(period,'start',None)}-{getattr(period,'stop',None)}"
                f"|gap{int(bool(interp_gaps))}"
                f"|obsmask{int(bool(MASK_TO_OBS))}")
    hit = _FILT_CACHE.get(full_key)
    if hit is not None:
        return hit

    re_da = regrid_timeseries(ts.sel(time=period), common_grid)
    if interp_gaps:
        re_da = _interp_time_gaps(re_da, max_gap=3)
    if key.startswith("chl::OBS"):
        _register_obs_sampling_mask(re_da)
    re_da = _apply_obs_sampling_mask(re_da, key)
    z = _deseas_detrend_zscore(re_da)
    z = _spatial_filter(z, width_deg=width_deg)
    z = _restandardise(z).astype("float32")

    _FILT_CACHE[full_key] = z
    nbytes = sum(v.nbytes for v in _FILT_CACHE.values())
    logger.info(f"    filtered-field cache MISS '{full_key}' "
                f"({len(_FILT_CACHE)} entries, {nbytes/1e9:.2f} GB)")
    return z


def _corr_map_lag0(drv_z: xr.DataArray, chl_z: xr.DataArray):
    """Per-cell lag-0 Pearson r and Neff-adjusted p. Thin wrapper.

    Delegates to `_ptcorr_r_and_p` so the significance treatment stays
    identical to the rest of the paper.
    """
    return _ptcorr_r_and_p(drv_z, chl_z, 0)


def _corr_by_month(
    drv_z: xr.DataArray,
    chl_z: xr.DataArray,
    min_years: int = SEASONAL_MIN_YEARS,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell correlation computed separately for each calendar month.

    For calendar month m the correlation uses only the month-m values,
    one per year, so each estimate rests on ~17 samples over 1998-2014.
    A single such correlation is very uncertain (sd ~0.27), which is why
    the figure that consumes this zonally averages before plotting: over
    a basin's worth of independent 6 deg longitudes the noise falls by
    roughly an order of magnitude.

    The inputs are already deseasonalised, so a month-m anomaly is a
    departure from the month-m climatology and the correlation measures
    interannual covariation within that month, not the seasonal cycle.

    Returns (r, n) each shaped (12, nlat, nlon), month index 0 = January.
    """
    ym_c = _ym_key(chl_z)
    ym_d = _ym_key(drv_z)
    common = np.intersect1d(ym_c, ym_d)
    if common.size < 24:
        raise ValueError(f"only {common.size} common months")
    ic = np.argsort(ym_c)[np.searchsorted(np.sort(ym_c), common)]
    id_ = np.argsort(ym_d)[np.searchsorted(np.sort(ym_d), common)]

    C = np.asarray(chl_z.values, dtype=float)[ic]
    D = np.asarray(drv_z.values, dtype=float)[id_]
    mo = (common % 100).astype(int)

    nlat, nlon = C.shape[1], C.shape[2]
    r_out = np.full((12, nlat, nlon), np.nan)
    n_out = np.zeros((12, nlat, nlon), dtype=int)

    for m in range(1, 13):
        sel = mo == m
        if sel.sum() < min_years:
            continue
        X, Y = D[sel], C[sel]
        ok = np.isfinite(X) & np.isfinite(Y)
        n = ok.sum(axis=0)
        Xz = np.where(ok, X, 0.0)
        Yz = np.where(ok, Y, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = Xz.sum(axis=0) / np.maximum(n, 1)
            my = Yz.sum(axis=0) / np.maximum(n, 1)
            dx = np.where(ok, X - mx[None], 0.0)
            dy = np.where(ok, Y - my[None], 0.0)
            num = (dx * dy).sum(axis=0)
            den = np.sqrt((dx ** 2).sum(axis=0) * (dy ** 2).sum(axis=0))
            r = np.where((den > 0) & (n >= min_years), num / den, np.nan)
        r_out[m - 1] = r
        n_out[m - 1] = n
    return r_out, n_out


def _basin_box(basin: str) -> tuple[float, float, float, float]:
    """(lat_min, lat_max, lon_min, lon_max) for a basin, from BASINS."""
    return BASINS[basin]


def _basin_axes_kw(basin: str) -> dict:
    """Projection for a basin panel.

    The two circumpolar basins get polar stereographic axes; everything
    else is PlateCarree centred on the basin. Mixing projections in one
    row is deliberate: a Southern Ocean box is 360 deg wide and 50 deg
    tall, so in PlateCarree it would be an unreadable letterbox next to
    the Atlantic.
    """
    lat_min, lat_max, lon_min, lon_max = _basin_box(basin)
    if basin == "Arctic":
        return {"projection": ccrs.NorthPolarStereo()}
    if basin in ("Southern", "Southern-SIZ", "Southern-ACC"):
        return {"projection": ccrs.SouthPolarStereo()}
    clon = (lon_min + lon_max) / 2.0
    if lon_min > lon_max:
        clon = ((lon_min + lon_max + 360) / 2.0) % 360
    return {"projection": ccrs.PlateCarree(central_longitude=clon)}


def _basin_aspect(basin: str) -> float:
    """Width-to-height ratio of a basin panel.

    Used as a GridSpec width_ratio so that every panel in a row shares
    the same height while keeping its true geographic aspect. Without
    this the row mixes a 2:1 global panel with a 0.5:1 Atlantic one and
    matplotlib letterboxes them to different heights, which leaves the
    column titles misaligned and wastes most of the figure area.
    Circumpolar panels are drawn on polar stereographic axes and are
    square by construction.
    """
    if basin in ("Arctic", "Southern", "Southern-SIZ", "Southern-ACC"):
        return 1.0
    lat_min, lat_max, lon_min, lon_max = _basin_box(basin)
    lon_span = (lon_max - lon_min) % 360 or 360.0
    lat_span = max(lat_max - lat_min, 1.0)
    return float(np.clip(lon_span / lat_span, 0.45, 2.6))


def _add_cyclic(field2d: np.ndarray, lons: np.ndarray):
    """Append the first longitude column at +360 to close the circle.

    contourf leaves an uncoloured wedge between the last and first
    longitude of a circumpolar panel, which on polar stereographic axes
    appears as a white radial line running from the pole to the rim.

    Implemented in numpy rather than via cartopy.util.add_cyclic_point.
    The cartopy call was wrapped in a bare except that returned the
    field unchanged, so any failure - an import problem, an unevenly
    spaced coordinate - reinstated the seam silently and the figure
    looked merely imperfect rather than broken. This version cannot
    fail, and warns if the grid spacing is irregular enough that the
    appended column would not sit flush against the first.
    """
    lons = np.asarray(lons, dtype=float)
    if lons.size < 2:
        return field2d, lons
    d = np.diff(lons)
    step = float(np.median(d))
    if step <= 0 or np.nanmax(np.abs(d - step)) > 0.5 * abs(step):
        logger.warning(
            "  _add_cyclic: longitude spacing is irregular; the "
            "circumpolar seam may not close cleanly")
    gap = (lons[0] + 360.0) - lons[-1]
    if abs(gap - step) > 0.5 * abs(step):
        logger.warning(
            f"  _add_cyclic: wrap gap {gap:.3f} deg differs from the grid "
            f"step {step:.3f} deg; longitudes may not span a full circle")
    out = np.concatenate([field2d, field2d[:, :1]], axis=1)
    lo = np.append(lons, lons[0] + 360.0)
    return out, lo


def _basin_map_axes(fig, nrows: int, basins: list) -> list:
    """GridSpec of map axes, one column per basin, equal panel heights."""
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(
        nrows, len(basins), figure=fig,
        width_ratios=[_basin_aspect(b) for b in basins],
        hspace=0.12, wspace=0.08,
        left=0.055, right=0.895, bottom=0.05, top=0.90,
    )
    return [[fig.add_subplot(gs[r, c], **_basin_axes_kw(b))
             for c, b in enumerate(basins)] for r in range(nrows)]


def _setup_basin_ax(ax, basin: str) -> None:
    """Apply extent, coastlines and (for polar panels) a circular frame."""
    lat_min, lat_max, lon_min, lon_max = _basin_box(basin)
    try:
        if basin == "Arctic":
            ax.set_extent([-180, 180, lat_min, 90], ccrs.PlateCarree())
            _circular_boundary(ax)
        elif basin in ("Southern", "Southern-SIZ", "Southern-ACC"):
            ax.set_extent([-180, 180, -90, lat_max], ccrs.PlateCarree())
            _circular_boundary(ax)
        else:
            lo0 = lon_min - 360 if lon_min > 180 else lon_min
            lo1 = lon_max - 360 if lon_max > 180 else lon_max
            if lo1 <= lo0:
                lo1 += 360
            ax.set_extent([lo0, lo1, lat_min, lat_max], ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=4)
        ax.coastlines(linewidth=0.3, zorder=5)
    except Exception:
        pass


def _circular_boundary(ax) -> None:
    """Clip a polar-stereographic axes to a circle."""
    try:
        import matplotlib.path as mpath
        theta = np.linspace(0, 2 * np.pi, 200)
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * 0.5 + 0.5)
        ax.set_boundary(circle, transform=ax.transAxes)
    except Exception:
        pass


def _draw_basin_corr_panel(ax, basin, r2d, lats, lons, levels, cmap,
                           sig=None):
    """Filled contours of r over one basin, stippled where significant.

    `sig` is the only overlay. See the convention note at the top of this
    section: dots mean significant, everywhere, in every figure.
    """
    _setup_basin_ax(ax, basin)
    lat_min, lat_max, lon_min, lon_max = _basin_box(basin)
    if ((lon_max - lon_min) % 360) == 0:          # circumpolar: close the seam
        if sig is not None:
            sig, _ = _add_cyclic(np.asarray(sig, dtype=float), lons)
            sig = sig > 0.5
        r2d, lons = _add_cyclic(r2d, lons)
    cf = ax.contourf(lons, lats, r2d, levels=levels, cmap=cmap,
                     extend="both", transform=ccrs.PlateCarree())
    try:
        ax.contour(lons, lats, r2d, levels=[0], colors="k",
                   linewidths=0.4, transform=ccrs.PlateCarree())
    except Exception:
        pass

    # Marks describe the correlation, so they are drawn only where one
    # exists, and beneath the land polygons (zorder 4) so that anything
    # straddling a coastline is covered rather than floating over land.
    if sig is not None:
        sig = np.asarray(sig, dtype=bool) & np.isfinite(r2d)
        if np.any(sig):
            lon2d, lat2d = np.meshgrid(lons, lats)
            ax.plot(lon2d[sig], lat2d[sig], "k.", ms=0.5, alpha=0.45,
                    transform=ccrs.PlateCarree(), zorder=3,
                    linestyle="none")
    return cf


def _basin_mask_2d(basin: str, common_grid: xr.DataArray) -> np.ndarray:
    """Boolean (lat, lon) mask for a basin on the common grid."""
    return _build_masks(common_grid)[basin].values


def make_figS_basin_corr_obs_ens(
    obs_chl_ts:   xr.DataArray,
    obs_drv_ts:   xr.DataArray | None,
    model_chl_ts: dict[str, xr.DataArray],
    model_drv_ts: dict[str, xr.DataArray],
    common_grid:  xr.DataArray,
    driver:       str,
    driver_label: str,
    period:       slice = LAG_PERIOD,
    width_deg:    float = SPATIAL_FILTER_DEG,
    remove:       tuple = (),
    obs_sst_for_idx:   xr.DataArray | None = None,
    model_sst_for_idx: dict[str, xr.DataArray] | None = None,
) -> dict:
    """Lag-0 Chl-a/driver correlation by basin: OBS row, ensemble row.

    With `remove` non-empty (e.g. ("AMO", "NINO34", "PDO")) each source's
    own climate modes are regressed out of both fields AFTER the spatial
    filter. Each source uses indices computed from its OWN SST, because
    a model's ENSO and AMO are not phase-locked to the observed ones and
    removing the observed index from a model would remove essentially
    nothing.

    Two rows by the basins in CORR_BASINS (five: Global is excluded, see
    the note there). Both fields are deseasonalised, detrended,
    standardised per cell and then smoothed with a `width_deg` box filter
    before correlating, which removes the small-scale variance that is
    uncorrelated between the two fields and would otherwise bias every r
    toward zero.

    Both rows are stippled, and in both rows the dots mean the same
    thing: significant. See the convention note at the top of this
    section. The observational row uses a Benjamini-Hochberg
    false-discovery-rate test at q = PTCORR_FDR_Q on the
    autocorrelation-adjusted effective sample size; the ensemble row
    marks cells where at least two thirds of the models WITH DATA THERE
    are individually significant and carry the sign of the ensemble
    mean.

    Returns {model: (r, significance mask)} so that the per-model figure
    and the ensemble criterion can reuse them without recomputing.
    """
    suffix, mode_lbl = _mode_tag(remove)
    logger.info(f"=== Fig S_basincorr ({driver}{suffix}): {width_deg:.0f} "
                f"deg filtered lag-0 correlation by basin - {mode_lbl} ===")
    if obs_drv_ts is None and not model_drv_ts:
        logger.warning(f"  No {driver} data; skipping")
        return {}

    lats = common_grid.lat.values
    lons = common_grid.lon.values
    models_list = [m for m in MODELS
                   if m in model_chl_ts and m in model_drv_ts]

    model_sst_for_idx = model_sst_for_idx or {}
    obs_idx = (_indices_for_source(obs_sst_for_idx, common_grid, period,
                                   "idx::OBS") if remove else {})
    if remove:
        logger.info(f"  removing {list(remove)} from each source "
                    f"(own indices, after filtering)")

    chl_obs = _filtered_z_modes(obs_chl_ts, common_grid, period, "chl::OBS",
                                width_deg, interp_gaps=True,
                                remove=remove, indices=obs_idx)

    obs_r = obs_sig = None
    if obs_drv_ts is not None:
        try:
            drv_obs = _filtered_z_modes(
                obs_drv_ts, common_grid, period,
                f"{driver.lower()}::OBS", width_deg,
                remove=remove, indices=obs_idx)
            r, p = _corr_map_lag0(drv_obs, chl_obs)
            obs_r, obs_sig = r, _fdr_mask(p)
            logger.success(
                f"  OBS: median r={np.nanmedian(r):+.3f}, "
                f"|r| p95={np.nanpercentile(np.abs(r),95):.3f}, "
                f"{100*obs_sig.sum()/max(np.isfinite(r).sum(),1):.1f}% "
                f"of cells significant after FDR")
        except Exception as e:
            logger.warning(f"  OBS {driver} correlation failed: {e}")

    # Keep each model's own significance mask, not only its r. The
    # per-model figure needs it, and the ensemble robustness criterion is
    # defined by how many models are individually significant, which
    # cannot be recovered from the mean.
    model_packs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model in models_list:
        try:
            m_idx = (_indices_for_source(
                model_sst_for_idx.get(model), common_grid, period,
                f"idx::{model}") if remove else {})
            cz = _filtered_z_modes(model_chl_ts[model], common_grid, period,
                                   f"chl::{model}", width_deg,
                                   remove=remove, indices=m_idx)
            dz = _filtered_z_modes(model_drv_ts[model], common_grid, period,
                                   f"{driver.lower()}::{model}", width_deg,
                                   remove=remove, indices=m_idx)
            r_m, p_m = _corr_map_lag0(dz, cz)
            model_packs[model] = (r_m, _fdr_mask(p_m))
        except Exception as e:
            logger.warning(f"  [{model}] {driver} correlation failed: {e}")

    if obs_r is None and not model_packs:
        logger.warning(f"  Fig S_basincorr ({driver}): nothing to plot")
        return {}

    ens_r = ens_sig = None
    if model_packs:
        names_m = [m for m in models_list if m in model_packs]
        stack = np.stack([model_packs[m][0] for m in names_m], 0)
        sigst = np.stack([model_packs[m][1] for m in names_m], 0)
        ens_r = np.nanmean(stack, 0)
        n_contrib = stack.shape[0]

        # Robustness, phrased positively so that dots mean the same
        # thing here as in the observational row: a cell is marked when
        # at least two thirds of the models WITH DATA THERE are each
        # individually significant and carry the sign of the ensemble
        # mean. Counting models per cell rather than using the ensemble
        # size matters - over land every model is NaN, and partially
        # covered cells would otherwise be scored against a quorum they
        # cannot reach.
        n_fin = np.sum(np.isfinite(stack), axis=0)
        with np.errstate(invalid="ignore"):
            same_sign = np.sign(stack) == np.sign(ens_r)[None]
        n_ok = np.sum(same_sign & sigst & np.isfinite(stack), axis=0)
        ens_sig = ((n_ok >= np.ceil(2.0 * n_fin / 3.0)) & (n_fin >= 3)
                   & np.isfinite(ens_r))
        frac = 100 * ens_sig.sum() / max(int(np.isfinite(ens_r).sum()), 1)
        logger.info(
            f"  ensemble: {n_contrib} of {len(models_list)} models; "
            f"{frac:.0f}% of ocean cells robust (>=2/3 of models "
            f"individually significant and of the ensemble-mean sign)")

    pool = np.concatenate([
        v.ravel()[np.isfinite(v.ravel())]
        for v in ([obs_r] if obs_r is not None else [])
        + ([ens_r] if ens_r is not None else [])])
    vlim = float(np.clip(np.nanpercentile(np.abs(pool), 98), 0.15, 0.8))
    levels = np.linspace(-vlim, vlim, 13)
    cmap = plt.cm.RdBu_r

    nb = len(CORR_BASINS)
    tot_w = sum(_basin_aspect(b) for b in CORR_BASINS)
    fig = plt.figure(figsize=(2.6 * tot_w + 2.0, 6.6))
    axes = _basin_map_axes(fig, 2, CORR_BASINS)

    last_cf = None
    rows = [(obs_r, obs_sig, f"OBS\n(ESA-CCI Chl / {driver_label})"),
            (ens_r, ens_sig,
             f"Ensemble mean\n({_ens_label({0: len(model_packs)}, models_list)})")]

    for row, (field, sig, row_lbl) in enumerate(rows):
        for col, basin in enumerate(CORR_BASINS):
            ax = axes[row][col]
            if field is None:
                ax.axis("off")
                continue
            bm = _basin_mask_2d(basin, common_grid)
            r2d = np.where(bm, field, np.nan)
            s2d = (np.where(bm, sig, False) if sig is not None else None)
            last_cf = _draw_basin_corr_panel(
                ax, basin, r2d, lats, lons, levels, cmap, s2d)
            add_panel_label(ax, row * nb + col, fontsize=10)
            if row == 0:
                ax.set_title(basin, fontsize=11, fontweight="bold")
            if col == 0:
                ax.text(-0.08, 0.5, row_lbl, transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=9,
                        fontweight="bold")

    if last_cf is not None:
        cax = fig.add_axes([0.915, 0.15, 0.013, 0.7])
        cb = fig.colorbar(last_cf, cax=cax)
        cb.set_label(f"Pearson r (Chl-a vs {driver}), lag 0, "
                     f"{width_deg:.0f}$^\\circ$ filtered", fontsize=10)

    # The label goes in the figure, not only in the filename: four of
    # these panels-of-panels look nearly identical side by side, and a
    # reader who has printed one cannot recover which mode was removed.
    fig.suptitle(
        f"Chl-a vs {driver} correlation at lag 0, "
        f"{width_deg:.0f}$^\\circ$ filtered  |  {mode_lbl}",
        fontsize=12.5, fontweight="bold", y=0.975)
    fname = f"figS_basincorr_{driver}{suffix}.png"
    fig.savefig(OUT_DIR / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {fname}  ({mode_lbl}; stippling = significant)")
    return model_packs


def make_figS_basin_corr_per_model(
    model_packs: dict[str, tuple[np.ndarray, np.ndarray]],
    common_grid: xr.DataArray,
    driver:      str,
    obs_pack:    tuple[np.ndarray, np.ndarray] | None = None,
    width_deg:   float = SPATIAL_FILTER_DEG,
    remove:      tuple = (),
) -> None:
    """Same layout, one row per model, each stippled where significant.

    The companion to `make_figS_basin_corr_obs_ens`. Where that figure's
    ensemble row is unmarked, this one shows why: whether the models are
    individually weak, or individually strong but split on sign.

    Each row carries its own FDR test, so the stippling means exactly
    what it means in every other panel of the paper.
    """
    if not model_packs:
        logger.warning(f"  Fig S_basincorr_models ({driver}): nothing to plot")
        return
    suffix, mode_lbl = _mode_tag(remove)
    logger.info(f"=== Fig S_basincorr_models ({driver}{suffix}) - "
                f"{mode_lbl} ===")

    lats = common_grid.lat.values
    lons = common_grid.lon.values
    packs = dict(model_packs)
    names = [m for m in MODELS if m in packs]
    if obs_pack is not None:
        packs["OBS"] = obs_pack
        names = ["OBS"] + names

    pool = np.concatenate([
        packs[m][0].ravel()[np.isfinite(packs[m][0].ravel())] for m in names])
    vlim = float(np.clip(np.nanpercentile(np.abs(pool), 98), 0.15, 0.8))
    levels = np.linspace(-vlim, vlim, 13)
    cmap = plt.cm.RdBu_r

    nb, nr = len(CORR_BASINS), len(names)
    tot_w = sum(_basin_aspect(b) for b in CORR_BASINS)
    fig = plt.figure(figsize=(2.6 * tot_w + 2.0, 2.4 * nr + 0.8))
    grid = _basin_map_axes(fig, nr, CORR_BASINS)
    last_cf = None
    for row, name in enumerate(names):
        r_f, s_f = packs[name]
        for col, basin in enumerate(CORR_BASINS):
            ax = grid[row][col]
            bm = _basin_mask_2d(basin, common_grid)
            r2d = np.where(bm, r_f, np.nan)
            s2d = (np.where(bm, s_f, False) if s_f is not None else None)
            last_cf = _draw_basin_corr_panel(
                ax, basin, r2d, lats, lons, levels, cmap, s2d)
            add_panel_label(ax, row * nb + col, fontsize=9)
            if row == 0:
                ax.set_title(basin, fontsize=11, fontweight="bold")
            if col == 0:
                ax.text(-0.10, 0.5, name, transform=ax.transAxes,
                        rotation=90, va="center", ha="center",
                        fontsize=9, fontweight="bold")

    if last_cf is not None:
        cax = fig.add_axes([0.915, 0.2, 0.012, 0.6])
        cb = fig.colorbar(last_cf, cax=cax)
        cb.set_label(f"Pearson r (Chl-a vs {driver}), lag 0, "
                     f"{width_deg:.0f}$^\\circ$ filtered", fontsize=10)

    fig.suptitle(
        f"Chl-a vs {driver} correlation at lag 0 by model, "
        f"{width_deg:.0f}$^\\circ$ filtered  |  {mode_lbl}",
        fontsize=12.5, fontweight="bold", y=0.995)
    fname = f"figS_basincorr_models_{driver}{suffix}.png"
    fig.savefig(OUT_DIR / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {fname}  ({mode_lbl}; stippling = significant)")


def make_figS_seasonal_hovmoller(
    obs_chl_ts:   xr.DataArray,
    obs_drv_ts:   xr.DataArray | None,
    model_chl_ts: dict[str, xr.DataArray],
    model_drv_ts: dict[str, xr.DataArray],
    common_grid:  xr.DataArray,
    driver:       str,
    driver_label: str,
    period:       slice = LAG_PERIOD,
    width_deg:    float = SPATIAL_FILTER_DEG,
) -> None:
    """Seasonal dependence of the lag-0 correlation: latitude vs month.

    For each calendar month the correlation is computed from that month's
    values only, one per year, then averaged zonally within each basin.
    Two rows (OBS, ensemble mean) by six basins; month on the horizontal
    axis, latitude on the vertical.

    This is the diagnostic the annual-mean maps cannot give. In mid
    latitudes the Chl-a response to warming reverses sign through the
    year - warming relieves light limitation in late winter and
    reinforces nutrient limitation in summer - so an annual correlation
    averages two opposing regimes toward zero and reports "no coupling"
    where the coupling is in fact strong and seasonally reversing.

    A single-month, single-cell correlation from ~17 years has a null sd
    near 0.27, far too noisy to map. Zonal averaging is what makes it
    usable: over a basin's independent 6 deg longitudes the sampling
    noise drops by roughly sqrt(n_indep). Latitudes where fewer than
    three independent longitudes survive are left blank, and the contour
    at the approximate 95% level is drawn so the reader can see which
    features clear the noise.
    """
    logger.info(f"=== Fig S_seasonal_hov ({driver}): seasonal correlation ===")
    if obs_drv_ts is None and not model_drv_ts:
        logger.warning(f"  No {driver} data; skipping")
        return

    lats = common_grid.lat.values
    models_list = [m for m in MODELS
                   if m in model_chl_ts and m in model_drv_ts]

    def _zonal(r_m: np.ndarray, n_m: np.ndarray, bmask: np.ndarray):
        """Zonal mean of the monthly r within a basin, plus an
        approximate 95% threshold from the number of independent
        longitudes at each latitude."""
        rr = np.where(bmask[None], r_m, np.nan)
        with np.errstate(invalid="ignore"):
            zm = np.nanmean(rr, axis=2)                   # (12, nlat)
        ncell = np.sum(np.isfinite(rr), axis=2)           # (12, nlat)
        # 6 deg filter -> neighbouring cells are not independent
        dlon = float(np.abs(np.median(np.diff(common_grid.lon.values))))
        per_indep = max(width_deg / dlon, 1.0)
        n_indep = np.maximum(ncell / per_indep, 1.0)
        n_yr = np.maximum(np.nanmax(n_m, axis=2), 3)
        se = 1.0 / np.sqrt(np.maximum(n_yr - 3, 1)) / np.sqrt(n_indep)
        thr = 1.96 * se
        zm = np.where(n_indep >= 3, zm, np.nan)
        return zm, thr

    def _fields_for(chl_ts, drv_ts, keyc, keyd):
        cz = _filtered_z(chl_ts, common_grid, period, keyc, width_deg,
                         interp_gaps=(keyc.endswith("OBS")))
        dz = _filtered_z(drv_ts, common_grid, period, keyd, width_deg)
        return _corr_by_month(dz, cz)

    obs_pack = None
    if obs_drv_ts is not None:
        try:
            obs_pack = _fields_for(obs_chl_ts, obs_drv_ts,
                                   "chl::OBS", f"{driver.lower()}::OBS")
        except Exception as e:
            logger.warning(f"  OBS seasonal correlation failed: {e}")

    stack_r, stack_n = [], []
    for model in models_list:
        try:
            r_m, n_m = _fields_for(model_chl_ts[model], model_drv_ts[model],
                                   f"chl::{model}", f"{driver.lower()}::{model}")
            stack_r.append(r_m)
            stack_n.append(n_m)
        except Exception as e:
            logger.warning(f"  [{model}] seasonal correlation failed: {e}")
    ens_pack = None
    if stack_r:
        ens_pack = (np.nanmean(np.stack(stack_r, 0), 0),
                    np.nanmax(np.stack(stack_n, 0), 0))
        logger.info(f"  ensemble seasonal: {len(stack_r)} of "
                    f"{len(models_list)} models contributed")

    if obs_pack is None and ens_pack is None:
        logger.warning(f"  Fig S_seasonal_hov ({driver}): nothing to plot")
        return

    pool = np.concatenate([
        p[0].ravel()[np.isfinite(p[0].ravel())]
        for p in (obs_pack, ens_pack) if p is not None])
    vlim = float(np.clip(np.nanpercentile(np.abs(pool), 98), 0.15, 0.8))
    levels = np.linspace(-vlim, vlim, 13)

    # Per-model zonal means, so the ensemble row can be hatched where the
    # models disagree on sign rather than carrying a significance test
    # that does not apply to a multi-model mean.
    per_model_zm: list[dict] = []
    for r_m, n_m in zip(stack_r, stack_n):
        d = {}
        for basin in CORR_BASINS:
            try:
                d[basin] = _zonal(r_m, n_m, _basin_mask_2d(basin, common_grid))[0]
            except Exception:
                pass
        per_model_zm.append(d)

    nb = len(CORR_BASINS)
    fig, axes = plt.subplots(2, nb, figsize=(2.7 * nb, 6.6),
                             squeeze=False, sharex=True)
    months = np.arange(1, 13)
    last = None
    rows = [(obs_pack, f"OBS\n(ESA-CCI Chl / {driver_label})"),
            (ens_pack, f"Ensemble mean\n({len(stack_r)} of "
                       f"{len(models_list)} models)")]

    for row, (pack, row_lbl) in enumerate(rows):
        for col, basin in enumerate(CORR_BASINS):
            ax = axes[row][col]
            if pack is None:
                ax.axis("off")
                continue
            bmask = _basin_mask_2d(basin, common_grid)
            zm, thr = _zonal(pack[0], pack[1], bmask)
            rows_ok = np.flatnonzero(np.any(np.isfinite(zm), axis=0))
            if rows_ok.size == 0:
                ax.axis("off")
                continue
            y = lats[rows_ok]
            Z = zm[:, rows_ok].T                       # (nlat_ok, 12)
            last = ax.contourf(months, y, Z, levels=levels,
                               cmap=plt.cm.RdBu_r, extend="both")
            add_panel_label(ax, row * nb + col, fontsize=10)

            # Same convention as the maps: dots mean significant.
            # An earlier version hatched what was NOT trustworthy, which
            # inverted the meaning of a mark relative to the map figures
            # sitting beside it in the same supplement.
            try:
                ax.contour(months, y, Z, levels=[0], colors="k",
                           linewidths=0.4)
                if row == 0:
                    # OBS: |r| clears the approximate 95% zonal-mean
                    # noise level for that latitude.
                    ok = (np.abs(Z) > thr[:, rows_ok].T) & np.isfinite(Z)
                elif per_model_zm:
                    # Ensemble: at least two thirds of the models with
                    # data at that (month, latitude) share the sign of
                    # the ensemble mean.
                    Zs = np.stack([mz[basin][:, rows_ok].T
                                   for mz in per_model_zm
                                   if basin in mz], 0)
                    n_fin = np.sum(np.isfinite(Zs), axis=0)
                    with np.errstate(invalid="ignore"):
                        same = np.sign(Zs) == np.sign(Z)[None]
                    n_ok = np.sum(same & np.isfinite(Zs), axis=0)
                    ok = ((n_ok >= np.ceil(2.0 * n_fin / 3.0))
                          & (n_fin >= 3) & np.isfinite(Z))
                else:
                    ok = None
                if ok is not None and np.any(ok):
                    ax.contourf(months, y, ok.astype(float),
                                levels=[0.5, 1.5], colors="none",
                                hatches=["...."], zorder=3)
            except Exception:
                pass

            ax.set_xticks([1, 4, 7, 10])
            ax.set_xticklabels(["J", "A", "J", "O"])
            ax.grid(alpha=0.2, lw=0.4)
            if row == 0:
                ax.set_title(basin, fontsize=11, fontweight="bold")
            if row == 1:
                ax.set_xlabel("Month", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{row_lbl}\n\nLatitude", fontsize=8,
                              fontweight="bold")

    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.08, top=0.92,
                        hspace=0.18, wspace=0.22)
    if last is not None:
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        cb = fig.colorbar(last, cax=cax)
        cb.set_label(
            f"Zonal-mean Pearson r (Chl-a vs {driver}) by calendar month, "
            f"{width_deg:.0f}$^\\circ$ filtered", fontsize=9)

    fname = f"figS_seasonal_corr_{driver}.png"
    fig.savefig(OUT_DIR / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(
        f"  wrote {fname}  (stippling = significant: OBS row |r| above the "
        f"approximate 95% zonal-mean noise level; ensemble row >=2/3 of "
        f"models sharing the ensemble-mean sign)")


# ---------------------------------------------------------------------
# Climate-mode removal for the filtered correlation fields
# ---------------------------------------------------------------------

# Display names and filename slugs for the climate modes. The figures
# used to carry a generic "modes removed" label, which told the reader
# that something had been regressed out but not what, and produced one
# combined figure where three separate ones are far more informative:
# ENSO, the PDO and the AMO act on different basins and different time
# scales, and removing them together confounds their contributions.
_MODE_DISPLAY = {"AMO": "AMO", "NINO34": "Nino 3.4 (ENSO)", "PDO": "PDO"}
_MODE_SLUG    = {"AMO": "noAMO", "NINO34": "noENSO",         "PDO": "noPDO"}


def _mode_tag(remove) -> tuple[str, str]:
    """(filename suffix, in-figure label) for a set of removed modes."""
    remove = tuple(remove or ())
    if not remove:
        return "", "no climate mode removed"
    slugs = [_MODE_SLUG.get(m, f"no{m}") for m in remove]
    disp = [_MODE_DISPLAY.get(m, m) for m in remove]
    label = (disp[0] if len(disp) == 1
             else ", ".join(disp[:-1]) + f" and {disp[-1]}")
    return "_" + "-".join(slugs), f"{label} regressed out"


def _regress_out_field(
    da:    xr.DataArray,
    index: xr.DataArray,
) -> xr.DataArray:
    """Remove the linear influence of a 1-D index from a 3-D field.

    The grid-point analogue of `_regress_out`. Times are matched on
    year*100+month integer keys rather than coordinate values, so cftime
    model axes and pandas OBS axes align correctly.

    ORDER RELATIVE TO THE SPATIAL FILTER. This is applied AFTER
    `_spatial_filter`, not before, and the reason is estimation quality
    rather than correctness. The two operations commute exactly when the
    valid mask is constant in time: the regression coefficient of a
    filtered field equals the weighted mean of its neighbours'
    coefficients, so F[R[x]] = R[F[x]] to machine precision. With Chl-a
    cloud gaps the filter weights vary in time and the identity holds
    only to ~0.3% of the field standard deviation, still negligible.

    What is not negligible is how well beta is estimated. At 1 deg the
    per-cell series is noise-dominated, and a beta estimated from it
    carries large error; subtracting beta_hat * index then removes a
    partly random component and injects noise into the residual. From
    the 6 deg filtered series the same estimate is far tighter - in a
    synthetic test, corr(beta_hat, beta_true) rose from 0.855 to 0.992
    and the scatter about the truth fell fivefold. So the field is
    filtered first and the mode removed second.
    """
    ym_t = _ym_key(da)
    ym_i = _ym_key(index)
    common = np.intersect1d(ym_t, ym_i)
    if common.size < 24:
        logger.warning(
            f"  _regress_out_field: only {common.size} common months; "
            f"returning the field unchanged")
        return da

    t_idx = np.argsort(ym_t)[np.searchsorted(np.sort(ym_t), common)]
    i_idx = np.argsort(ym_i)[np.searchsorted(np.sort(ym_i), common)]

    arr = np.asarray(da.values, dtype=float)[t_idx]
    idx = np.asarray(index.values, dtype=float)[i_idx]

    ok_i = np.isfinite(idx)
    if ok_i.sum() < 24:
        return da
    idx = np.where(ok_i, idx, 0.0)

    ok = np.isfinite(arr) & ok_i[:, None, None]
    n = ok.sum(axis=0)

    # Centre both the field and the index over each cell's own finite
    # subset, for the same reason the detrending does: when missingness
    # is seasonally clustered the index mean over the available months
    # is not the index mean overall, and the offset would leak into beta.
    A = np.where(ok, arr, 0.0)
    I = np.where(ok, idx[:, None, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        a_bar = A.sum(axis=0) / np.maximum(n, 1)
        i_bar = I.sum(axis=0) / np.maximum(n, 1)
        da_ = np.where(ok, arr - a_bar[None], 0.0)
        di_ = np.where(ok, idx[:, None, None] - i_bar[None], 0.0)
        sxy = (da_ * di_).sum(axis=0)
        sxx = (di_ * di_).sum(axis=0)
        beta = np.where(sxx > 0, sxy / np.maximum(sxx, 1e-12), 0.0)
        res = arr - beta[None] * (idx[:, None, None] - i_bar[None])
    res[~ok] = np.nan

    # Report how much variance the index actually accounts for. Without
    # this the only evidence that a mode was removed is that the figure
    # changed, and for a mode with little variance in the analysis window
    # the figure barely changes - which is indistinguishable by eye from
    # the removal having silently failed.
    #
    # Expect this to be small for the AMO and modest for the PDO over a
    # 17-year window: their periods are 60-80 and 20-60 years, so within
    # 1998-2014 they are close to linear ramps and the per-cell detrend
    # has already taken most of what they had. ENSO, at 2-7 years, is the
    # only one fully resolved by the record.
    with np.errstate(invalid="ignore", divide="ignore"):
        syy = (da_ * da_).sum(axis=0)
        r2 = np.where(syy > 0, (sxy ** 2) / np.maximum(sxx * syy, 1e-12),
                      np.nan)
    lat_w = np.cos(np.deg2rad(np.asarray(da.lat.values, dtype=float)))
    w = lat_w[:, None] * np.ones((1, r2.shape[1]))
    good = np.isfinite(r2) & (n >= 24)
    if good.any():
        r2_bar = float(np.sum(w[good] * r2[good]) / np.sum(w[good]))
        logger.info(
            f"    regressed out an index: area-mean R2 = {r2_bar:.3f} "
            f"({100*r2_bar:.1f}% of local variance), "
            f"index sd = {float(np.nanstd(idx[ok_i])):.3g}, "
            f"max local R2 = {float(np.nanmax(r2[good])):.3f}")
        if r2_bar < 0.01:
            logger.warning(
                "    that index explains under 1% of local variance in "
                "this window - the 'removed' figure will look almost "
                "identical to the raw one, and legitimately so")

    out = da.isel(time=xr.DataArray(t_idx, dims="time"))
    return out.copy(data=res)


_INDEX_CACHE: dict[str, dict] = {}


def _indices_for_source(
    sst_ts:      xr.DataArray | None,
    common_grid: xr.DataArray,
    period:      slice,
    key:         str,
) -> dict:
    """AMO, Nino 3.4 and PDO computed from a source's OWN SST field.

    Each model gets its own indices rather than inheriting the observed
    ones, and this is not a detail. A free-running CMIP6 historical
    simulation has its own internal ENSO, AMO and PDO with phases that
    are unrelated to the real ones. Regressing the OBSERVED index out of
    a model field therefore removes almost nothing - beta is close to
    zero by construction - so the "modes removed" model panels would be
    indistinguishable from the raw ones while the observations lost most
    of their variance. The comparison would be asymmetric in a way that
    looks like a physical result and is purely a methodological artefact.

    Computing each model's indices from its own SST removes each source's
    own modes, which is the like-for-like comparison the figure claims to
    make. It does mean the removed patterns differ between sources, which
    belongs in the caption.
    """
    if sst_ts is None:
        return {}
    full = f"{key}|{_da_fingerprint(sst_ts)}|{tuple(common_grid.sizes.values())}"
    hit = _INDEX_CACHE.get(full)
    if hit is not None:
        return hit
    try:
        sst_re = regrid_timeseries(sst_ts.sel(time=period), common_grid)
        out = _compute_climate_indices(sst_re)
    except Exception as e:
        logger.warning(f"  climate indices for {key} failed: {e}")
        out = {}
    _INDEX_CACHE[full] = out
    return out


def _filtered_z_modes(
    ts:          xr.DataArray,
    common_grid: xr.DataArray,
    period:      slice,
    key:         str,
    width_deg:   float = SPATIAL_FILTER_DEG,
    interp_gaps: bool = False,
    remove:      tuple = (),
    indices:     dict | None = None,
) -> xr.DataArray:
    """`_filtered_z` with climate-mode removal applied after filtering."""
    z = _filtered_z(ts, common_grid, period, key, width_deg, interp_gaps)
    if not remove or not indices:
        return z
    names = [n for n in remove if n in indices]
    if not names:
        logger.warning(f"  {key}: none of {list(remove)} available; "
                       f"no mode removed")
        return z
    for nm in names:
        z = _regress_out_field(z, indices[nm])
    return _restandardise(z).astype("float32")


# ---------------------------------------------------------------------
# Lag curves built as the area mean of the filtered correlation maps
# ---------------------------------------------------------------------

def make_fig_lag_areamean(
    obs_chl_ts:       xr.DataArray,
    obs_drv_ts:       xr.DataArray | None,
    model_chl_ts:     dict[str, xr.DataArray],
    model_drv_ts:     dict[str, xr.DataArray],
    common_grid:      xr.DataArray,
    driver:           str,
    driver_label:     str,
    remove:           tuple = (),
    obs_sst_for_idx:  xr.DataArray | None = None,
    model_sst_for_idx: dict[str, xr.DataArray] | None = None,
    period:           slice = LAG_PERIOD,
    width_deg:        float = SPATIAL_FILTER_DEG,
    lag_max:          int = LAG_MAX,
) -> None:
    """Lag curves formed by AREA-AVERAGING the filtered correlation maps.

    Replaces the previous basin-mean lag figure, which correlated two
    basin-mean time series. That construction answered a different
    question from the maps beside it, and the gap between the two had
    three separate causes: attenuation of the pointwise correlations by
    small-scale noise, variance weighting of the basin average toward a
    few high-variance regions, and the off-diagonal Chl-a/driver
    covariance that only the basin mean sees.

    Here the correlation is computed at every grid point and every lag on
    the same 6 deg filtered fields used by the map figures, and only then
    averaged over the basin with cos(lat) weights. The curve is therefore
    the exact lag-resolved counterpart of the maps: the value at lag 0
    is the area mean of the corresponding map panel, by construction. No
    teleconnection term enters, and no region dominates by virtue of
    having large variance.

    Shading is the interquartile range of the pointwise correlations
    within the basin, so it shows how much the sign and strength vary
    across the basin rather than sampling uncertainty. A basin whose
    curve sits near zero with a wide band contains strong coupling of
    both signs; one near zero with a narrow band genuinely has none.
    That distinction is invisible in a single averaged time series and
    is the main thing this figure adds.

    Markers are filled at lags where at least half the basin area is
    individually significant after FDR, consistent with the convention
    that stippling means significant everywhere else in this section.
    """
    suffix, mode_lbl = _mode_tag(remove)
    lbl = f" [{mode_lbl}]"
    logger.info(f"=== Fig lag area-mean ({driver}{suffix}): "
                f"{width_deg:.0f} deg filtered, per basin - {mode_lbl} ===")
    if obs_drv_ts is None and not model_drv_ts:
        logger.warning(f"  No {driver} data; skipping")
        return

    lags = np.arange(-lag_max, lag_max + 1)
    models_list = [m for m in MODELS
                   if m in model_chl_ts and m in model_drv_ts]
    model_sst_for_idx = model_sst_for_idx or {}

    lat = common_grid.lat.values
    w2d = (np.cos(np.deg2rad(lat))[:, None]
           * np.ones((1, common_grid.sizes["lon"])))
    masks = {b: _basin_mask_2d(b, common_grid) for b in CORR_BASINS}

    curves: dict[str, dict[str, np.ndarray]] = {b: {} for b in CORR_BASINS}
    bands:  dict[str, tuple] = {}
    sigfr:  dict[str, np.ndarray] = {}

    def _one_source(chl_ts, drv_ts, sst_ts, label, keyc, keyd, is_obs):
        idx = (_indices_for_source(sst_ts, common_grid, period,
                                   f"idx::{label}") if remove else {})
        cz = _filtered_z_modes(chl_ts, common_grid, period, keyc, width_deg,
                               interp_gaps=is_obs, remove=remove, indices=idx)
        dz = _filtered_z_modes(drv_ts, common_grid, period, keyd, width_deg,
                               remove=remove, indices=idx)
        per_basin = {b: np.full(lags.size, np.nan) for b in CORR_BASINS}
        per_band = {b: (np.full(lags.size, np.nan),
                        np.full(lags.size, np.nan)) for b in CORR_BASINS}
        per_sig = {b: np.full(lags.size, np.nan) for b in CORR_BASINS}

        for li, k in enumerate(lags):
            if is_obs:
                r, p = _ptcorr_r_and_p(dz, cz, int(k))
                sig = _fdr_mask(p)
            else:
                # Only the observational curve carries significance
                # markers, so the models skip the effective-sample-size
                # and FDR machinery entirely. With 25 lags x 7 models x
                # 4 mode variants x 2 drivers that is 1400 p-value fields
                # never plotted.
                r = _pointwise_r_at_lag(dz.values.astype(float),
                                        cz.values.astype(float), int(k))
                sig = np.zeros_like(r, dtype=bool)
            fin = np.isfinite(r)
            for b in CORR_BASINS:
                sel = masks[b] & fin
                if sel.sum() < 10:
                    continue
                w = w2d[sel]
                rv = r[sel]
                per_basin[b][li] = float(np.sum(w * rv) / np.sum(w))
                per_band[b][0][li] = float(np.percentile(rv, 25))
                per_band[b][1][li] = float(np.percentile(rv, 75))
                per_sig[b][li] = float(np.sum(w * sig[sel]) / np.sum(w))
        return per_basin, per_band, per_sig

    # ---- OBS ----
    if obs_drv_ts is not None:
        try:
            pb, bd, sg = _one_source(
                obs_chl_ts, obs_drv_ts, obs_sst_for_idx, "OBS",
                "chl::OBS", f"{driver.lower()}::OBS", True)
            for b in CORR_BASINS:
                curves[b]["OBS"] = pb[b]
                bands[b] = bd[b]
                sigfr[b] = sg[b]
                if np.any(np.isfinite(pb[b])):
                    j = int(np.nanargmax(np.abs(pb[b])))
                    logger.success(
                        f"  [OBS  ] {b:9s}{lbl}: peak area-mean r="
                        f"{pb[b][j]:+.3f} @ lag={int(lags[j]):+d} "
                        f"({100*sg[b][j]:.0f}% of area significant)")
        except Exception as e:
            import traceback
            logger.warning(f"  OBS {driver} area-mean lag curve failed: {e}")
            logger.warning(traceback.format_exc().replace("<", "\\<"))

    # ---- models ----
    for model in models_list:
        try:
            pb, _, _ = _one_source(
                model_chl_ts[model], model_drv_ts[model],
                model_sst_for_idx.get(model), model,
                f"chl::{model}", f"{driver.lower()}::{model}", False)
            for b in CORR_BASINS:
                curves[b][model] = pb[b]
        except Exception as e:
            logger.warning(f"  [{model}] area-mean lag curve failed: {e}")

    if not any(curves[b] for b in CORR_BASINS):
        logger.warning(f"  Fig lag area-mean ({driver}): nothing to plot")
        return

    nb = len(CORR_BASINS)
    ncols = min(3, nb)
    nrows = int(np.ceil(nb / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.4 * nrows),
                             squeeze=False, sharex=True, sharey=True)
    handles: dict[str, object] = {}

    for i, basin in enumerate(CORR_BASINS):
        ax = axes[i // ncols][i % ncols]
        ax.axhline(0, color="k", lw=0.5, alpha=0.6)
        ax.axvline(0, color="k", lw=0.5, alpha=0.6, ls=":")

        if basin in bands:
            lo, hi = bands[basin]
            if np.any(np.isfinite(lo)):
                ax.fill_between(lags, lo, hi, color="0.5", alpha=0.18, lw=0,
                                zorder=1,
                                label="OBS interquartile range across basin")

        for src, rs in curves[basin].items():
            if src == "OBS":
                continue
            c = PALETTE[models_list.index(src) % len(PALETTE)] \
                if src in models_list else "0.6"
            h, = ax.plot(lags, rs, lw=1.1, color=c, alpha=0.85, zorder=2)
            handles.setdefault(src, h)

        if "OBS" in curves[basin]:
            rs = curves[basin]["OBS"]
            h, = ax.plot(lags, rs, lw=2.6, color="black", zorder=5)
            handles.setdefault("OBS", h)
            fr = sigfr.get(basin)
            if fr is not None:
                m = np.isfinite(rs) & np.isfinite(fr) & (fr >= 0.5)
                if np.any(m):
                    ax.plot(lags[m], rs[m], "o", ms=4.5, mfc="black",
                            mec="white", mew=0.6, zorder=6, linestyle="none")

        ax.set_title(basin, fontsize=11, fontweight="bold")
        ax.set_xlim(-lag_max, lag_max)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xticks(np.arange(-lag_max, lag_max + 1, 4))
        ax.grid(alpha=0.25)
        add_panel_label(ax, i, fontsize=11)
        if i % ncols == 0:
            ax.set_ylabel("Area-mean Pearson r", fontsize=10)
        if i // ncols == nrows - 1:
            ax.set_xlabel(f"Lag (months)  -  {driver} leads ->", fontsize=10)

    for j in range(nb, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    order = (["OBS"] if "OBS" in handles else []) + \
            [m for m in models_list if m in handles]
    if order:
        fig.legend([handles[k] for k in order], order,
                   loc="lower center", ncol=min(5, len(order)),
                   frameon=True, fontsize=9,
                   bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Area mean of the {width_deg:.0f}$^\\circ$-filtered grid-point "
        f"Chl-a / {driver} correlation  |  {mode_lbl}",
        fontsize=12, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    fname = f"fig9_lag_areamean_{driver}{suffix}.png"
    fig.savefig(OUT_DIR / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {fname}  (filled markers: >=50% of basin "
                   f"area individually significant)")


def _ens_label(ens_n: dict[int, int], models_list: list) -> str:
    """Row label reporting how many models the ensemble mean really used.

    Reports a single count when every lag used the same models, and the
    range otherwise, so a panel built from a partial ensemble cannot be
    mistaken for the full one.
    """
    counts = sorted(set(ens_n.values())) if ens_n else []
    if not counts:
        return f"0 of {len(models_list)} models"
    if len(counts) == 1:
        n = counts[0]
        return (f"{n} models" if n == len(models_list)
                else f"{n} of {len(models_list)} models")
    return f"{counts[0]}-{counts[-1]} of {len(models_list)} models"


def make_figS_pointwise_driver_corr(
    obs_chl_ts:   xr.DataArray,
    obs_drv_ts:   xr.DataArray | None,
    model_chl_ts: dict[str, xr.DataArray],
    model_drv_ts: dict[str, xr.DataArray],
    common_grid:  xr.DataArray,
    driver:       str,
    driver_label: str,
    period:       slice = LAG_PERIOD,
    lags:         list[int] | None = None,
) -> None:
    """Pointwise Chl-a/driver lag correlation maps, global domain.

    The basin-mean diagnostic of Section 3.5 collapses each basin to a
    single number per lag, which hides two things: whether the basin
    signal comes from a coherent region or from a few cells, and
    whether cancellation between sub-regions of opposite sign is
    suppressing the basin mean. This figure resolves both.

    Layout matches the sea-ice maps: 2 rows (OBS, ensemble mean of the
    per-model r fields) x len(lags) columns. Positive lag means the
    driver leads Chl-a, the same convention used throughout.

    Significance is assessed per cell with a t-test on an
    autocorrelation-adjusted effective sample size (Bretherton et al.
    1999) and then corrected for multiple comparisons across the map
    by the Benjamini-Hochberg procedure at q = PTCORR_FDR_Q. Cells
    surviving that correction are stippled. Without the correction a
    global map would flag thousands of cells by chance; without the
    Neff adjustment the test would be far too liberal given the strong
    month-to-month persistence of ocean anomalies.

    Output: figS_pointwise_corr_{driver}.png
    """
    lags = PTCORR_LAGS if lags is None else lags
    logger.info(
        f"=== Fig S_pointwise_corr ({driver}): grid-point Chl-a/"
        f"{driver} lag correlation ===")

    if obs_drv_ts is None and not model_drv_ts:
        logger.warning(f"  No {driver} data; skipping")
        return

    lats = common_grid.lat.values
    lons = common_grid.lon.values
    models_list = [m for m in MODELS
                   if m in model_chl_ts and m in model_drv_ts]

    chl_obs_z = _regrid_z_cached(
        obs_chl_ts, common_grid, period, "chl::OBS", interp_gaps=True)

    obs_r: dict[int, np.ndarray] = {}
    obs_p: dict[int, np.ndarray] = {}
    if obs_drv_ts is not None:
        drv_obs_z = _regrid_z_cached(
            obs_drv_ts, common_grid, period, f"{driver.lower()}::OBS")
        for lag in lags:
            try:
                r, p = _ptcorr_r_and_p(drv_obs_z, chl_obs_z, lag)
                obs_r[lag] = r
                obs_p[lag] = p
                sig = _fdr_mask(p)
                frac_extreme = float(
                    np.nanmean(np.abs(r) > 0.9)) if np.isfinite(r).any() else 0.0
                if frac_extreme > 0.5:
                    logger.warning(
                        f"  OBS {driver} lag{lag:+d}: {100*frac_extreme:.0f}% "
                        f"of cells have |r|>0.9 - this is not a physical "
                        f"field, suspect a data or cache problem")
                logger.success(
                    f"  OBS {driver} lag{lag:+d}: median r="
                    f"{np.nanmedian(r):+.3f}, "
                    f"|r| p95={np.nanpercentile(np.abs(r), 95):.3f}, "
                    f"{100 * sig.sum() / max(np.isfinite(r).sum(), 1):.1f}% "
                    f"of cells significant after FDR")
            except Exception as e:
                logger.warning(f"  OBS {driver} lag{lag:+d} failed: {e}")

    ens_r: dict[int, np.ndarray] = {}
    # Number of models that actually contributed at each lag. The row
    # label used to be built from len(models_list), i.e. the number of
    # CANDIDATE models, so a panel could read "7 models" while being a
    # mean over three if four had raised. That makes two runs of the
    # figure silently incomparable.
    ens_n: dict[int, int] = {}
    for lag in lags:
        stack = []
        for model in models_list:
            try:
                cz = _regrid_z_cached(
                    model_chl_ts[model], common_grid, period,
                    f"chl::{model}")
                dz = _regrid_z_cached(
                    model_drv_ts[model], common_grid, period,
                    f"{driver.lower()}::{model}")
                r_m, _ = _ptcorr_r_and_p(dz, cz, lag)
                stack.append(r_m)
            except Exception as e:
                logger.warning(
                    f"  [{model}] {driver} lag{lag:+d} failed: {e}")
        if stack:
            ens_r[lag] = np.nanmean(np.stack(stack, 0), 0)
            ens_n[lag] = len(stack)
            logger.info(
                f"  ensemble {driver} lag{lag:+d}: {len(stack)} of "
                f"{len(models_list)} models contributed")
            if len(stack) < len(models_list):
                logger.warning(
                    f"  ensemble {driver} lag{lag:+d} is a mean over "
                    f"{len(stack)} models, NOT {len(models_list)} - the "
                    f"panel is labelled with the true count, but any "
                    f"comparison with another run must check that the "
                    f"same models contributed")
        else:
            ens_n[lag] = 0
            logger.warning(
                f"  ensemble {driver} lag{lag:+d}: NO model contributed "
                f"(of {len(models_list)} candidates) - the per-model "
                f"warnings above give the reason")

    if not obs_r and not ens_r:
        logger.warning(f"  Fig S_pointwise_corr ({driver}): nothing to plot")
        return

    # Shared symmetric scale across every panel so the columns and the
    # two rows are directly comparable.
    pool = np.concatenate([
        v.ravel()[np.isfinite(v.ravel())]
        for v in list(obs_r.values()) + list(ens_r.values())]) \
        if (obs_r or ens_r) else np.array([0.3])
    vlim = float(min(np.nanpercentile(np.abs(pool), 99), 0.8))
    vlim = max(vlim, 0.1)
    levels = np.linspace(-vlim, vlim, 13)

    n_lags = len(lags)
    fig, axes = plt.subplots(
        2, n_lags, figsize=(5.2 * n_lags, 6.4),
        subplot_kw={"projection": ccrs.Robinson(central_longitude=200)},
        squeeze=False,
    )
    fig.subplots_adjust(left=0.03, right=0.88, bottom=0.06, top=0.90,
                        hspace=0.08, wspace=0.05)
    last_cf = None

    rows = [
        (obs_r, f"OBS\n(ESA-CCI Chl / {driver_label})", True),
        (ens_r, f"Ensemble mean\n({_ens_label(ens_n, models_list)})", False),
    ]

    for row, (r_by_lag, row_lbl, show_sig) in enumerate(rows):
        for col, lag in enumerate(lags):
            ax = axes[row][col]
            try:
                ax.add_feature(cfeature.LAND, facecolor="lightgray",
                               zorder=4)
                ax.coastlines(linewidth=0.3, zorder=5)
            except Exception:
                pass

            data = r_by_lag.get(lag)
            if data is not None:
                try:
                    cf = ax.contourf(
                        lons, lats, data, levels=levels, cmap="RdBu_r",
                        extend="both", transform=ccrs.PlateCarree())
                    last_cf = cf
                except Exception as e:
                    logger.warning(
                        f"    contourf failed {driver} lag{lag:+d}: {e}")
                    cf = None

                if show_sig and lag in obs_p:
                    sig = _fdr_mask(obs_p[lag]) & np.isfinite(data)
                    if sig.any():
                        try:
                            ax.contourf(
                                lons, lats,
                                np.where(sig, 1.0, np.nan),
                                levels=[0.5, 1.5], colors="none",
                                hatches=["...."],
                                transform=ccrs.PlateCarree(), zorder=3)
                        except Exception:
                            pass
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="grey")

            if row == 0:
                if lag == 0:
                    sub = "simultaneous"
                elif lag > 0:
                    sub = (f"{driver} leads by {lag} "
                           f"month{'s' if lag > 1 else ''}")
                else:
                    sub = (f"Chl-$a$ leads by {-lag} "
                           f"month{'s' if -lag > 1 else ''}")
                ax.set_title(f"Lag {lag:+d}\n({sub})",
                             fontsize=11, fontweight="bold", pad=4)
            if col == 0:
                ax.text(-0.04, 0.5, row_lbl, transform=ax.transAxes,
                        fontsize=9, fontweight="bold", ha="right",
                        va="center", rotation=90)
            letter = chr(ord("a") + row * n_lags + col)
            ax.text(0.01, 0.97, f"({letter})", transform=ax.transAxes,
                    fontsize=10, fontweight="bold", va="top",
                    bbox=dict(facecolor="white", alpha=0.75,
                              edgecolor="none"))

    if last_cf is not None:
        cax = fig.add_axes([0.90, 0.15, 0.014, 0.68])
        cb = fig.colorbar(last_cf, cax=cax, extend="both")
        cb.set_label(f"Pearson r (Chl-$a$ vs {driver})", fontsize=14)
        cb.ax.tick_params(labelsize=14)

    fname = f"figS_pointwise_corr_{driver}.png"
    try:
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        logger.success(f"  wrote {fname}")
    except Exception as e:
        logger.warning(f"  {fname}: tight save failed ({e}); retrying")
        fig.savefig(OUT_DIR / fname, dpi=150)
        logger.success(f"  wrote {fname} (untrimmed)")
    plt.close(fig)


# ---------------------------------------------------------------------
# Seasonal stratification of the sea-ice / Chl-a correlation
# ---------------------------------------------------------------------

# Seasons for the sea-ice/Chl-a stratification, by hemisphere.
#
# Spring and summer are kept as separate three-month windows because
# they are where the bloom and the ocean-colour coverage both are, and
# because they answer different questions: at a +3 month lag a spring
# bloom pairs with peak winter ice, whereas a summer bloom pairs with
# the retreat itself.
#
# Autumn and winter are MERGED into a single six-month window. Kept
# apart, the winter panel is close to unusable: ocean colour does not
# exist through polar night, so a JJA window in the Southern Ocean
# retains almost nothing poleward of the ice edge. Folding the autumn
# months in roughly doubles the available month-pairs and, more
# importantly, contributes the months that still carry light at high
# southern latitudes. The result is a genuinely observable non-growing
# season panel rather than a mostly blank one, at the cost of a window
# twice as wide as the other two - so its correlations average over a
# broader range of ice states and should be read as a baseline against
# which the spring and summer panels are compared, not as a season in
# the same sense.
_SEASON_MONTHS = {
    "Southern": {
        "spring":     (9, 10, 11),          # austral spring
        "summer":     (12, 1, 2),           # austral summer
        "fallwinter": (3, 4, 5, 6, 7, 8),   # austral autumn + winter
    },
    "Arctic": {
        "spring":     (3, 4, 5),            # boreal spring
        "summer":     (6, 7, 8),            # boreal summer
        "fallwinter": (9, 10, 11, 12, 1, 2),
    },
}
_SEASON_LABEL = {
    "Southern": {
        "spring":     "austral spring (SON)",
        "summer":     "austral summer (DJF)",
        "fallwinter": "austral autumn-winter (MAM-JJA)",
    },
    "Arctic": {
        "spring":     "boreal spring (MAM)",
        "summer":     "boreal summer (JJA)",
        "fallwinter": "boreal autumn-winter (SON-DJF)",
    },
}

# A three-month window over 17 years gives at most 51 month-pairs (102
# for the merged autumn-winter window), and fewer where ocean colour is
# missing. Below this many the correlation is not interpretable and the
# cell is masked. This is a far smaller sample than the unstratified
# diagnostic, so the seasonal figures should be read for the sign and
# spatial coherence of their structure rather than for the precise
# magnitude of any single cell -- and the spring and summer panels rest
# on half the sample of the autumn-winter one, so their noise floors
# differ.
SEASONAL_PTCORR_MIN_N = 20


def _ptcorr_r_and_p_months(
    drv_z:  xr.DataArray,
    chl_z:  xr.DataArray,
    lag:    int,
    months: tuple,
    min_n:  int = SEASONAL_PTCORR_MIN_N,
):
    """Pointwise correlation restricted to selected calendar months.

    `months` selects the CHL-A months, that is the response side, and
    the driver is then taken at the requested lag WITHOUT any further
    restriction. This is deliberate and is what makes the diagnostic
    mean what the seasonal-ice-zone mechanism claims. At lag +3 with
    Chl-a in December and January the driver is drawn from September and
    October, which is peak Antarctic ice extent: the pairing is
    precisely "bloom now against the ice that preceded it". Restricting
    both sides to the same six months instead would discard exactly the
    winter-ice-to-summer-bloom pairs the analysis exists to test.

    The anomalies entering here are already deseasonalised, so the
    correlation measures interannual covariation within the selected
    months rather than any part of the seasonal cycle.

    Significance uses the same Bretherton effective sample size as the
    full-record diagnostic. Note that subsetting to alternate halves of
    the year removes some of the month-to-month persistence, so N_eff is
    a larger fraction of N here than in the unstratified case, but N
    itself is roughly six times smaller.
    """
    ym_c = _ym_key(chl_z)
    ym_d = _ym_key(drv_z)
    common = np.intersect1d(ym_c, ym_d)
    if common.size < min_n:
        raise ValueError(f"only {common.size} common months")
    ic = np.argsort(ym_c)[np.searchsorted(np.sort(ym_c), common)]
    idr = np.argsort(ym_d)[np.searchsorted(np.sort(ym_d), common)]

    C = np.asarray(chl_z.values, dtype=float)[ic]
    D = np.asarray(drv_z.values, dtype=float)[idr]
    mo = (common % 100).astype(int)
    T = C.shape[0]

    # Positive lag: driver leads. Chl-a index t is paired with driver
    # index t-lag, and the SEASON TEST IS APPLIED TO t.
    if lag > 0:
        c_idx = np.arange(lag, T)
        d_idx = c_idx - lag
    elif lag < 0:
        c_idx = np.arange(0, T + lag)
        d_idx = c_idx - lag
    else:
        c_idx = np.arange(T)
        d_idx = c_idx

    keep = np.isin(mo[c_idx], months)
    c_idx, d_idx = c_idx[keep], d_idx[keep]
    if c_idx.size < min_n:
        shape = C.shape[1:]
        return np.full(shape, np.nan), np.full(shape, np.nan)

    Y = C[c_idx]
    X = D[d_idx]
    ok = np.isfinite(X) & np.isfinite(Y)
    n = ok.sum(axis=0)

    Xz = np.where(ok, X, 0.0)
    Yz = np.where(ok, Y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = Xz.sum(axis=0) / np.maximum(n, 1)
        my = Yz.sum(axis=0) / np.maximum(n, 1)
        dx = np.where(ok, X - mx[None], 0.0)
        dy = np.where(ok, Y - my[None], 0.0)
        num = (dx * dy).sum(axis=0)
        den = np.sqrt((dx ** 2).sum(axis=0) * (dy ** 2).sum(axis=0))
        r = np.where((den > 0) & (n >= min_n), num / den, np.nan)

    neff = _effective_n(X, Y, axis=0)
    neff = np.minimum(neff, n.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        dof = np.maximum(neff - 2.0, 1.0)
        tstat = r * np.sqrt(dof / np.maximum(1.0 - r ** 2, 1e-12))
        from scipy import stats as _st
        p = 2.0 * _st.t.sf(np.abs(tstat), dof)
    p = np.where(np.isfinite(r), p, np.nan)
    return r, p


# Lags spanned by the seasonal sea-ice lag curve. Extended to a full year
# because the winter gap in ocean colour constrains only the RESPONSE:
# sea ice is observed year-round, so fixing Chl-a in the months the
# satellite can see and scanning the lag reaches winter ice without ever
# needing winter Chl-a. At lag +4 to +8 a DJF bloom is being paired with
# June-October ice. Lag +12 returns to the same calendar months one year
# earlier and is a useful internal check: a peak there would indicate
# aliasing of the annual cycle rather than a memory pathway.
SEAICE_LAGCURVE_LAGS = tuple(range(0, 13))

# Zonal bands for the seasonal lag curve, as (lat_min, lat_max).
#
# Latitude rather than longitude is the natural stratification for this
# diagnostic. The quantities that set the memory pathway - ice-season
# duration, retreat date, and how long the water column has been exposed
# to light before the bloom - all vary first and foremost with distance
# from the continent, because that is what determines how far the ice
# edge has to travel to reach a cell. A band at 57 S is ice-covered for
# a few weeks a year; one at 73 S may not clear at all in some years.
# Those are different physical regimes, and averaging them together is
# what flattens the circumpolar curve.
#
# The bands also separate the sign structure visible in the maps: the
# coherent positive band sits along the outer seasonal-ice zone while
# values closer to the continent are frequently negative, so a
# circumpolar average cancels them against each other.
#
# 45-55 S is ice-free year round and is retained as a negative control:
# it has no cryospheric memory to carry, so a flat curve there is the
# correct answer and confirms that structure in the ice-covered bands is
# not an artefact of the seasonal subsetting.
SEAICE_ZONAL_BANDS = {
    "45-55S (ACC, ice-free)": (-55, -45),
    "55-60S (outer SIZ)":     (-60, -55),
    "60-65S (mid SIZ)":       (-65, -60),
    "65-70S (inner SIZ)":     (-70, -65),
    "70-78S (coastal)":       (-78, -70),
}


def make_fig_seaice_chl_lagcurve_seasonal(
    obs_chl_ts:      xr.DataArray,
    obs_siconc_ts:   xr.DataArray | None,
    model_chl_ts:    dict[str, xr.DataArray],
    model_siconc_ts: dict[str, xr.DataArray],
    common_grid:     xr.DataArray,
    period:          slice = LAG_PERIOD,
    obs_siconc_name: str = SICONC_OBS_NAME,
    out_tag:         str = "",
    region_name:     str = "Southern",
    by_band:         bool = True,
) -> None:
    """Area-mean sea-ice/Chl-a correlation against lag, by season and zonal band.

    Answers the two questions the seasonal maps raise but cannot resolve
    on their own:

      1. Does the bloom respond differently depending on WHEN in the
         growing season it occurs? Each ROW is a different Chl-a window,
         so comparing rows answers this.
      2. Does ice at different times of year matter differently, winter
         ice included? The lag axis runs to twelve months, and because
         the Chl-a window is held fixed while the lag moves only the ice
         term, a curve is a clean scan through the ice year for a single
         bloom season.

    The second question is the one worth being explicit about. Ocean
    colour has no winter data in the seasonal-ice zone, which appears to
    rule out any statement about winter. It does not: the gap is on the
    response side only, and sea-ice concentration is observed all year.
    What cannot be tested is winter CHLOROPHYLL coupling, which is not a
    meaningful quantity because there is no bloom in darkness. What can
    be tested, and is tested here, is whether the summer bloom carries
    memory of the preceding winter's ice. The upper axis of each panel
    gives the ice month reached at each lag so this can be read directly.

    The curve is the area-weighted mean of the grid-point correlation
    over cells where it is defined, which is the seasonal-ice zone by
    construction: north of the maximum ice edge concentration is
    identically zero and has no variance, and under the perennial pack
    there is no ocean colour. Shading is the interquartile range of the
    pointwise correlations across the averaging region, so it shows how
    much the relationship varies in space rather than sampling
    uncertainty. A marker is filled where at least half the region's
    area is individually significant after FDR correction.

    WHY ZONAL BANDS. Averaged over the whole 45-78 S band the observed
    curve peaks near r = 0.10 with an interquartile spread of about
    +-0.15 and reaches significance at no lag, while the maps show local
    values beyond +-0.4. The circumpolar-and-all-latitudes mean is not
    measuring a weak coupling; it is averaging a coupling of opposite
    sign at different distances from the continent toward zero, the same
    cancellation that separates a basin-mean correlation from a map of
    pointwise ones. Resolving by latitude is therefore a precondition
    for the diagnostic meaning anything, not a refinement of it.

    Columns run from the ice-free Antarctic Circumpolar Current band to
    the coastal zone, and the rightmost column repeats the undivided
    average so the cost of not stratifying is visible in the same
    figure. See SEAICE_ZONAL_BANDS for the physical rationale.
    """
    logger.info(
        f"=== Fig seaice lag curve{out_tag} ({region_name}): "
        f"area-mean r vs lag, by season ===")
    if obs_siconc_ts is None and not model_siconc_ts:
        logger.warning("  No siconc data; skipping")
        return
    if region_name not in SEAICE_REGIONS:
        logger.warning(f"  unknown region {region_name}; skipping")
        return

    reg = SEAICE_REGIONS[region_name]
    seasons = _SEASON_MONTHS[region_name]
    lags = np.array(SEAICE_LAGCURVE_LAGS)

    lats = common_grid.lat.values
    lons = common_grid.lon.values
    in_reg = (lats >= reg["lat_min"]) & (lats <= reg["lat_max"])
    whole = np.zeros((lats.size, lons.size), dtype=bool)
    whole[in_reg, :] = True
    w2d = (np.cos(np.deg2rad(lats))[:, None]
           * np.ones((1, lons.size)))

    # Averaging regions: zonal bands when by_band, otherwise the single
    # circumpolar band.
    if by_band:
        areas: dict[str, np.ndarray] = {}
        for bname, (lo_lat, hi_lat) in SEAICE_ZONAL_BANDS.items():
            m = np.zeros_like(whole)
            m[(lats >= lo_lat) & (lats < hi_lat), :] = True
            if m.sum():
                areas[bname] = m
        areas["all (45-78S)"] = whole
    else:
        areas = {"Circumpolar": whole}

    models_list = [m for m in MODELS
                   if m in model_chl_ts and m in model_siconc_ts]

    chl_obs = _regrid_z_cached(obs_chl_ts, common_grid, period, "chl::OBS",
                               interp_gaps=True)
    ice_obs = (None if obs_siconc_ts is None else
               _regrid_z_cached(obs_siconc_ts, common_grid, period,
                                f"siconc::OBS{out_tag}"))

    def _curve(ice_z, chl_z, months, want_sig, area_mask):
        """Area-mean r, IQR and significant-area fraction at each lag."""
        mean = np.full(lags.size, np.nan)
        lo = np.full(lags.size, np.nan)
        hi = np.full(lags.size, np.nan)
        frac = np.full(lags.size, np.nan)
        for i, lag in enumerate(lags):
            try:
                r, p = _ptcorr_r_and_p_months(ice_z, chl_z, int(lag), months)
            except Exception:
                continue
            sel = area_mask & np.isfinite(r)
            if sel.sum() < 20:
                continue
            w = w2d[sel]
            rv = r[sel]
            mean[i] = float(np.sum(w * rv) / np.sum(w))
            lo[i] = float(np.percentile(rv, 25))
            hi[i] = float(np.percentile(rv, 75))
            if want_sig:
                sig = _fdr_mask(np.where(area_mask, p, np.nan))
                frac[i] = float(np.sum(w * sig[sel]) / np.sum(w))
        return mean, lo, hi, frac

    obs_c: dict[tuple, tuple] = {}
    mod_c: dict[tuple, dict[str, np.ndarray]] = {}
    for season, months in seasons.items():
        for aname, amask in areas.items():
            key = (season, aname)
            mod_c[key] = {}
            if ice_obs is not None:
                try:
                    obs_c[key] = _curve(ice_obs, chl_obs, months, True, amask)
                    m = obs_c[key][0]
                    if np.any(np.isfinite(m)):
                        j = int(np.nanargmax(np.abs(m)))
                        logger.success(
                            f"  [OBS] {season:10s} {aname:24s}: peak "
                            f"r={m[j]:+.3f} @ lag +{int(lags[j])} mo "
                            f"({100*obs_c[key][3][j]:.0f}% of the area "
                            f"significant)")
                except Exception as e:
                    logger.warning(f"  OBS {season}/{aname} failed: {e}")
            for mdl in models_list:
                try:
                    cz = _regrid_z_cached(model_chl_ts[mdl], common_grid,
                                          period, f"chl::{mdl}")
                    iz = _regrid_z_cached(model_siconc_ts[mdl], common_grid,
                                          period, f"siconc::{mdl}")
                    mod_c[key][mdl] = _curve(iz, cz, months, False, amask)[0]
                except Exception as e:
                    logger.warning(f"  [{mdl}] {season}/{aname} failed: {e}")

    if not obs_c and not any(mod_c.values()):
        logger.warning("  nothing to plot")
        return

    _MON = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    srow = [x for x in ("spring", "summer", "fallwinter") if x in seasons]
    acol = list(areas.keys())
    nrow, ncol = len(srow), len(acol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.1 * nrow),
                             squeeze=False, sharey=True, sharex=True)
    handles: dict[str, object] = {}

    for ri, season in enumerate(srow):
        # Circular mean, not the median of the sorted list. DJF sorts to
        # [1, 2, 12] and its middle element is February, which is off by
        # one month and mislabels every tick of the summer row; the same
        # failure shifts the Arctic autumn-winter window. Months are
        # angles on a circle, so the centre has to be computed as one.
        _ang = np.deg2rad([(m - 1) * 30.0 for m in seasons[season]])
        _mu = np.arctan2(np.sin(_ang).mean(), np.cos(_ang).mean())
        mid = int(round((np.rad2deg(_mu) % 360.0) / 30.0)) % 12 + 1

        for ci, aname in enumerate(acol):
            ax = axes[ri][ci]
            ax.axhline(0, color="k", lw=0.5, alpha=0.6)
            key = (season, aname)

            if key in obs_c:
                mean, lo, hi, frac = obs_c[key]
                if np.any(np.isfinite(lo)):
                    ax.fill_between(lags, lo, hi, color="0.5", alpha=0.18,
                                    lw=0, zorder=1)
                h, = ax.plot(lags, mean, lw=2.4, color="black", zorder=5)
                handles.setdefault("OBS", h)
                m = np.isfinite(mean) & np.isfinite(frac) & (frac >= 0.5)
                if np.any(m):
                    ax.plot(lags[m], mean[m], "o", ms=4.5, mfc="black",
                            mec="white", mew=0.6, zorder=6, linestyle="none")

            for mdl, cur in mod_c.get(key, {}).items():
                c = PALETTE[models_list.index(mdl) % len(PALETTE)]
                h, = ax.plot(lags, cur, lw=1.0, color=c, alpha=0.8, zorder=2)
                handles.setdefault(mdl, h)

            ax.set_xlim(lags[0], lags[-1])
            ax.set_xticks(lags[::3])
            ax.grid(alpha=0.25)
            add_panel_label(ax, ri * ncol + ci, fontsize=9)
            if ri == 0:
                ax.set_title(aname, fontsize=10.5, fontweight="bold")
                top = ax.twiny()
                top.set_xlim(ax.get_xlim())
                top.set_xticks(lags[::3])
                top.set_xticklabels(
                    [_MON[((mid - int(l) - 1) % 12) + 1] for l in lags[::3]],
                    fontsize=7.5)
            if ri == nrow - 1:
                ax.set_xlabel("Lag (mo) - ice leads", fontsize=9)
            if ci == 0:
                ax.set_ylabel(
                    f"{_SEASON_LABEL[region_name][season]}\n\n"
                    f"Area-mean r", fontsize=8.5, fontweight="bold")

    ord_leg = (["OBS"] if "OBS" in handles else []) + \
              [m for m in models_list if m in handles]
    if ord_leg:
        fig.legend([handles[k] for k in ord_leg], ord_leg,
                   loc="lower center", ncol=min(8, len(ord_leg)),
                   frameon=True, fontsize=8.5, bbox_to_anchor=(0.5, -0.045))

    fig.suptitle(
        f"{region_name} Ocean: sea-ice / Chl-a coupling by bloom season, "
        f"latitude band and ice lead time, 1998-2014",
        fontsize=12.5, fontweight="bold", y=1.005)
    fig.tight_layout()

    fname = f"figS_seaice_chl_lagcurve{out_tag}_{region_name}.png"
    fig.savefig(OUT_DIR / fname, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success(
        f"  wrote {fname}  (filled markers: >=50% of the zone "
        f"individually significant)")


def make_figS_seaice_chl_seasonal_corr(
    obs_chl_ts:       xr.DataArray,
    obs_siconc_ts:    xr.DataArray | None,
    model_chl_ts:     dict[str, xr.DataArray],
    model_siconc_ts:  dict[str, xr.DataArray],
    common_grid:      xr.DataArray,
    period:           slice = LAG_PERIOD,
    obs_siconc_name:  str = SICONC_OBS_NAME,
    out_tag:          str = "",
    lags:             tuple = (0, 1, 2, 3),
) -> None:
    """Sea-ice/Chl-a lag correlation stratified by half of the year.

    The seasonal counterpart of `make_figS_seaice_chl_spatial_corr`,
    restricted to the Southern Ocean and split into one figure per
    season:
      figS_seaice_chl_seasonal{out_tag}_Southern_spring.png
      figS_seaice_chl_seasonal{out_tag}_Southern_summer.png
      figS_seaice_chl_seasonal{out_tag}_Southern_fallwinter.png

    Each figure is two rows by four columns: observations above the
    ensemble mean, columns being lags 0 to +3 months with sea ice
    leading. This matches the layout of the all-months figure, so the
    three can be compared panel by panel, and places each observed row
    directly above the model row it is meant to be read against.

    The season is applied to the Chl-a months, so that a warm-season
    panel at lag +3 pairs a December-January bloom with September-October
    ice. See `_ptcorr_r_and_p_months`.

    WHY THE WINDOWS ARE UNEQUAL. Spring and summer are three months
    each; autumn and winter are merged into six. Ocean colour requires
    sunlight, so a winter-only window retains almost nothing poleward of
    the Antarctic ice edge, and a panel that is blank because the
    instrument cannot see is easily misread as a panel that is blank
    because there is no coupling. Merging the autumn months in supplies
    both the sample size and the months that still carry light at high
    southern latitudes.

    Coverage remains uneven even so, and the routine reports the
    fraction of the region retained in each panel. A blank area should
    always be read against that number.
    """
    logger.info(
        f"=== Fig S_seaice_chl_seasonal{out_tag}: seasonal siconc-Chl-a "
        f"correlation ===")
    if obs_siconc_ts is None and not model_siconc_ts:
        logger.warning(f"  No siconc data ({obs_siconc_name}); skipping")
        return

    lats = common_grid.lat.values
    lons = common_grid.lon.values
    models_list = [m for m in MODELS
                   if m in model_chl_ts and m in model_siconc_ts]

    chl_obs = _regrid_z_cached(obs_chl_ts, common_grid, period,
                               "chl::OBS", interp_gaps=True)
    ice_obs = (None if obs_siconc_ts is None else
               _regrid_z_cached(obs_siconc_ts, common_grid, period,
                                f"siconc::OBS{out_tag}"))

    # Southern Ocean only. The Arctic seasonal split is not run: the
    # observed sea-ice/Chl-a coupling that motivates this figure is a
    # Southern Ocean result, the Arctic headline lag is zero so the lag
    # columns carry little information there, and the Arctic growing
    # season is short enough that a further split leaves too few
    # month-pairs to interpret.
    for region_name in ("Southern",):
        if region_name not in SEAICE_REGIONS:
            continue
        reg = SEAICE_REGIONS[region_name]
        seasons = _SEASON_MONTHS[region_name]
        in_reg = ((lats >= reg["lat_min"]) & (lats <= reg["lat_max"]))
        reg_mask = np.zeros((lats.size, lons.size), dtype=bool)
        reg_mask[in_reg, :] = True

        panels: dict[tuple, np.ndarray] = {}
        sigs:   dict[tuple, np.ndarray] = {}
        cover:  dict[tuple, float] = {}

        for season, months in seasons.items():
            # ---- OBS ----
            if ice_obs is not None:
                for lag in lags:
                    try:
                        r, p = _ptcorr_r_and_p_months(
                            ice_obs, chl_obs, int(lag), months)
                        r = np.where(reg_mask, r, np.nan)
                        panels[("OBS", season, lag)] = r
                        sigs[("OBS", season, lag)] = _fdr_mask(
                            np.where(reg_mask, p, np.nan))
                        cover[("OBS", season, lag)] = float(
                            np.isfinite(r).sum() / max(reg_mask.sum(), 1))
                    except Exception as e:
                        logger.warning(
                            f"  [{region_name}/{season}/lag{lag}] OBS "
                            f"failed: {e}")
            # ---- ensemble ----
            for lag in lags:
                stack = []
                for m in models_list:
                    try:
                        cz = _regrid_z_cached(model_chl_ts[m], common_grid,
                                              period, f"chl::{m}")
                        iz = _regrid_z_cached(model_siconc_ts[m], common_grid,
                                              period, f"siconc::{m}")
                        r_m, _ = _ptcorr_r_and_p_months(
                            iz, cz, int(lag), months)
                        stack.append(np.where(reg_mask, r_m, np.nan))
                    except Exception as e:
                        logger.warning(
                            f"  [{m}/{region_name}/{season}/lag{lag}] "
                            f"failed: {e}")
                if stack:
                    st = np.stack(stack, 0)
                    ens = np.nanmean(st, 0)
                    panels[("ENS", season, lag)] = ens
                    n_fin = np.sum(np.isfinite(st), axis=0)
                    with np.errstate(invalid="ignore"):
                        same = np.sign(st) == np.sign(ens)[None]
                    n_ok = np.sum(same & np.isfinite(st), axis=0)
                    sigs[("ENS", season, lag)] = (
                        (n_ok >= np.ceil(2.0 * n_fin / 3.0))
                        & (n_fin >= 3) & np.isfinite(ens))
                    cover[("ENS", season, lag)] = float(
                        np.isfinite(ens).sum() / max(reg_mask.sum(), 1))

        if not panels:
            logger.warning(f"  {region_name}: nothing to plot")
            continue

        for season in seasons:
            c = cover.get(("OBS", season, lags[0]))
            if c is not None:
                lvl = ("USABLE" if c > 0.30 else
                       "SPARSE" if c > 0.10 else "ESSENTIALLY EMPTY")
                logger.info(
                    f"  {region_name} {season}: ocean-colour coverage "
                    f"{100*c:.0f}% of the region  [{lvl}]")
                if c <= 0.10:
                    logger.warning(
                        f"  {region_name} {season}: coverage is too low to "
                        f"interpret; this is polar night, not an absence "
                        f"of coupling")

        pool = np.concatenate([
            v.ravel()[np.isfinite(v.ravel())] for v in panels.values()])
        if pool.size == 0:
            continue
        vlim = float(np.clip(np.nanpercentile(np.abs(pool), 98), 0.15, 0.8))
        levels = np.linspace(-vlim, vlim, 13)

        # One figure per growing-season half, each with OBS directly
        # above its ensemble counterpart. Splitting this way rather than
        # stacking all four rows keeps the layout identical to the
        # all-months figure, so the two can be compared panel by panel,
        # and puts each observed row adjacent to the model row it is
        # meant to be read against.
        for season in seasons:
            rows = [("OBS", season), ("ENS", season)]
            if not any((src, season, lg) in panels
                       for src, _ in rows for lg in lags):
                logger.warning(
                    f"  {region_name} {season}: no panels; skipping figure")
                continue

            nrow, ncol = len(rows), len(lags)
            fig = plt.figure(figsize=(3.1 * ncol + 2.0, 3.1 * nrow + 1.0))
            last = None
            for ri, (src, seas) in enumerate(rows):
                for ci, lag in enumerate(lags):
                    ax = fig.add_subplot(nrow, ncol, ri * ncol + ci + 1,
                                         projection=reg["proj"])
                    try:
                        ax.set_extent(reg["extent"], ccrs.PlateCarree())
                        _circular_boundary(ax)
                        ax.add_feature(cfeature.LAND, facecolor="lightgray",
                                       zorder=4)
                        ax.coastlines(linewidth=0.3, zorder=5)
                    except Exception:
                        pass
                    key = (src, seas, lag)
                    if key in panels:
                        fld, lo2 = _add_cyclic(panels[key], lons)
                        last = ax.contourf(
                            lo2, lats, fld, levels=levels,
                            cmap=plt.cm.RdBu_r, extend="both",
                            transform=ccrs.PlateCarree())
                        sg = sigs.get(key)
                        if sg is not None:
                            sg2, _ = _add_cyclic(sg.astype(float), lons)
                            sg2 = (sg2 > 0.5) & np.isfinite(fld)
                            if np.any(sg2):
                                lo2d, la2d = np.meshgrid(lo2, lats)
                                ax.plot(lo2d[sg2], la2d[sg2], "k.", ms=0.5,
                                        alpha=0.45, zorder=3,
                                        linestyle="none",
                                        transform=ccrs.PlateCarree())
                    else:
                        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                                ha="center", va="center", fontsize=9,
                                color="0.4")
                    add_panel_label(ax, ri * ncol + ci, fontsize=9)
                    if ri == 0:
                        ax.set_title(f"Lag +{lag} mo", fontsize=11,
                                     fontweight="bold")
                    if ci == 0:
                        lab = ("OBS" if src == "OBS"
                               else f"Ensemble ({len(models_list)} models)")
                        ax.text(-0.10, 0.5, lab, transform=ax.transAxes,
                                rotation=90, va="center", ha="center",
                                fontsize=9, fontweight="bold")

            fig.subplots_adjust(left=0.06, right=0.90, bottom=0.04,
                                top=0.90, hspace=0.08, wspace=0.06)
            if last is not None:
                cax = fig.add_axes([0.915, 0.2, 0.012, 0.6])
                cb = fig.colorbar(last, cax=cax)
                cb.set_label(
                    f"Pearson r (Chl-a vs sea-ice concentration), "
                    f"{obs_siconc_name} ice", fontsize=10)
            fig.suptitle(
                f"{region_name} Ocean: sea-ice / Chl-a lag correlation, "
                f"{_SEASON_LABEL[region_name][season]}  |  sea ice leads",
                fontsize=12.5, fontweight="bold", y=0.965)

            fname = (f"figS_seaice_chl_seasonal{out_tag}_"
                     f"{region_name}_{season}.png")
            fig.savefig(OUT_DIR / fname, dpi=180, bbox_inches="tight")
            plt.close(fig)
            logger.success(
                f"  wrote {fname}  "
                f"({_SEASON_LABEL[region_name][season]}; "
                f"stippling = significant)")


def make_figS_seaice_chl_spatial_corr(
    obs_chl_ts:       xr.DataArray,
    obs_siconc_ts:    xr.DataArray | None,
    model_chl_ts:     dict[str, xr.DataArray],
    model_siconc_ts:  dict[str, xr.DataArray],
    common_grid:      xr.DataArray,
    period:           slice = LAG_PERIOD,
    obs_siconc_name:  str = SICONC_OBS_NAME,
    out_tag:          str = "",
) -> dict[str, dict[int, np.ndarray]]:
    """Pointwise siconc-Chl-a lag correlation maps for ONE ice product.

    Produces two figures per call:
      figS_seaice_chl_corr{out_tag}_Arctic.png
      figS_seaice_chl_corr{out_tag}_Southern.png

    Columns are lags 0 to +3 months (siconc leading Chl-a). Rows are:
      1. OBS, ESA-CCI Chl vs the supplied `obs_siconc_ts`
      2. Ensemble mean r over the available models

    `obs_siconc_name` only sets the row label and log prefix;
    `out_tag` is inserted into the filename so that calling this once
    per ice product writes separate files rather than overwriting.
    The intended use is two calls — HadISST (out_tag="") and ESA-CCI
    (out_tag="_ESACCI") — so the two observational products can be
    compared side by side as independent figures.

    Returns {region_name: {lag: r_map}} for the OBS row, so the caller
    can quantify how far the two ice products agree without recomputing
    anything.

    Both siconc and Chl-a anomalies are deseasonalised, linearly
    detrended, and standardised to unit variance per cell before
    correlation, via _deseas_detrend_zscore.
    """
    from scipy import stats as _stats

    logger.info(
        "=== Fig S_seaice_chl_corr: pointwise siconc-Chl-a "
        "correlation maps ==="
    )

    if obs_siconc_ts is None and not model_siconc_ts:
        logger.warning(
            f"  No siconc data ({obs_siconc_name}); "
            f"skipping figS_seaice_chl_corr{out_tag}"
        )
        return {}
    logger.info(
        f"  ice product: {obs_siconc_name}  "
        f"(obs {'present' if obs_siconc_ts is not None else 'ABSENT'}, "
        f"{len(model_siconc_ts)} models)  out_tag='{out_tag}'"
    )
    obs_r_all: dict[str, dict[int, np.ndarray]] = {}

    # Domain shared with the regression figure via SEAICE_REGIONS, so
    # the two figure families cover exactly the same cells and can be
    # compared panel by panel. "lag" is the headline lag for this
    # region and is local to this figure.
    _headline_lag = {"Arctic": 0, "Southern": 3}
    regions = {
        name: {**r, "lag": _headline_lag.get(name, 0)}
        for name, r in SEAICE_REGIONS.items()
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
        sic_key: str = "",
        chl_key: str = "",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pointwise Pearson r and beta at specified lag.

        Both fields preprocessed per cell (deseasonalise, detrend,
        z-score). At lag k > 0 siconc leads Chl-a.

        The regrid + z-score depends only on the source field, not on
        the lag or the region, but this function is called once per
        lag per source per region — 64 times per ice product. Without
        memoisation that is 64 repetitions of a 204-timestep regrid
        for each field, which dominated the runtime of this figure.
        Passing sic_key/chl_key routes it through the shared cache;
        empty keys fall back to computing in place.

        Returns r_map, beta_map — both (nlat, nlon), NaN outside mask.
        """
        if sic_key:
            sic_z = _regrid_z_cached(sic_ts, common_grid, period, sic_key)
        else:
            sic_z = _deseas_detrend_zscore(
                regrid_timeseries(sic_ts.sel(time=period), common_grid))
        if chl_key:
            chl_z = _regrid_z_cached(chl_ts, common_grid, period, chl_key,
                                     interp_gaps=interpolate_chl)
        else:
            chl_re = regrid_timeseries(
                chl_ts.sel(time=period), common_grid)
            if interpolate_chl:
                chl_re = _interp_time_gaps(chl_re, max_gap=3)
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
        obs_r_by_lag:  dict[int, np.ndarray] = {}
        ens_r_by_lag:  dict[int, np.ndarray] = {}

        for lag in LAGS:
            if obs_siconc_ts is not None:
                try:
                    obs_r, _ = _compute_r_beta(
                        obs_siconc_ts, obs_chl_ts,
                        lag, reg_mask, interpolate_chl=True,
                        sic_key=f"sic::{obs_siconc_name}",
                        chl_key="chl::OBS")
                    obs_r_by_lag[lag] = obs_r
                    logger.success(
                        f"  {obs_siconc_name} {region_name} lag{lag:+d}: "
                        f"median r={np.nanmedian(obs_r):.3f}")
                except Exception as e:
                    logger.warning(
                        f"  {obs_siconc_name} {region_name} "
                        f"lag{lag:+d} failed: {e}")

            # Models
            model_r_list = []
            for model in models_list:
                try:
                    r_m, _ = _compute_r_beta(
                        model_siconc_ts[model],
                        model_chl_ts[model],
                        lag, reg_mask,
                        sic_key=f"sic::{model}",
                        chl_key=f"chl::{model}")
                    model_r_list.append(r_m)
                except Exception as e:
                    logger.warning(
                        f"  [{model}] {region_name} lag{lag:+d} failed: {e}")
            if model_r_list:
                ens_r_by_lag[lag] = np.nanmean(
                    np.stack(model_r_list, 0), 0)

        obs_r_all[region_name] = dict(obs_r_by_lag)

        # ── Figure: 2 rows (OBS | ensemble mean) x 4 cols (lags) ──
        rows_spec: list[tuple[dict[int, np.ndarray], str]] = [
            (obs_r_by_lag,
             f"OBS\n(ESA-CCI Chl / {obs_siconc_name} siconc)"),
            (ens_r_by_lag,
             f"Ensemble mean\n({len(models_list)} models)"),
        ]

        n_lags = len(LAGS)
        n_rows = len(rows_spec)
        fig, axes = plt.subplots(
            n_rows, n_lags,
            figsize=(5.5 * n_lags, 5.5 * n_rows),
            subplot_kw={"projection": proj},
            squeeze=False,
        )
        fig.subplots_adjust(
            left=0.04, right=0.88,
            bottom=0.04, top=0.92,
            hspace=0.10, wspace=0.06,
        )

        last_cf = None

        for row, (r_by_lag, row_lbl) in enumerate(rows_spec):
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
            cb.set_label("Pearson r", fontsize=15)
            cb.ax.tick_params(labelsize=14)
            cb.set_ticks([-0.4, -0.2, 0, 0.2, 0.4])

        # No suptitle — figure is self-explanatory from panel labels

        fname = f"figS_seaice_chl_corr{out_tag}_{region_name}.png"
        fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"  wrote {fname}")

    return obs_r_all




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
            cb.set_label(meta["label"], fontsize=15)
            cb.ax.tick_params(labelsize=14)

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

_REGRID_Z_CACHE: dict[str, xr.DataArray] = {}


# =====================================================================
# Month-by-month observational sampling mask
# =====================================================================
#
# Ocean-colour retrievals are missing wherever the water is cloudy,
# ice-covered or too dimly lit, and that pattern changes every month.
# Models have no such gaps. Correlating a model over all 204 months
# against observations over whatever subset survived at each cell
# compares two different samples.
#
# The problem is not mainly the difference in sample size. The missing
# months are not a random subset: they are systematically the cloudy,
# low-sun, ice-covered ones. At a mid-latitude cell the observed
# correlation is therefore weighted toward summer while the model's
# spans the whole year, and because the Chl-a/SST relationship reverses
# sign between the light-limited and nutrient-limited regimes, an
# apparent model-observation disagreement can be nothing more than a
# difference in which months were compared.
#
# Applying the mask BEFORE deseasonalising is essential. The monthly
# climatology, the linear trend and hence every anomaly must all be
# built from the same calendar sampling as the observations; masking
# afterwards would leave a climatology computed from months the
# satellite never saw.
#
# Set CHL_MASK_TO_OBS=1 to enable. Output then goes to a separate
# directory so the two versions can be compared side by side rather
# than overwriting one another.

MASK_TO_OBS = _MASK_TO_OBS_ENV in ("1", "true", "yes", "on")

# Filled by the loader once the observational Chl-a field has been
# regridded to the common grid; keyed by grid shape so a run that uses
# more than one target grid cannot silently apply the wrong mask.
_OBS_SAMPLING_MASK: dict[tuple, xr.DataArray] = {}


def _register_obs_sampling_mask(obs_re: xr.DataArray) -> None:
    """Record where and when the observational field has data."""
    key = (tuple(obs_re.sizes[d] for d in ("lat", "lon")),
           int(obs_re.sizes["time"]))
    _OBS_SAMPLING_MASK[key] = np.isfinite(obs_re)
    frac = float(np.isfinite(obs_re).mean())
    logger.info(
        f"    registered observational sampling mask "
        f"{key}: {100*frac:.1f}% of cell-months observed")


def _apply_obs_sampling_mask(re_da: xr.DataArray, key: str) -> xr.DataArray:
    """Blank a regridded model field wherever the observations are absent.

    A no-op for the observational fields themselves, and a no-op when no
    mask has been registered for this grid and record length, so the
    function is safe to call unconditionally.
    """
    if not MASK_TO_OBS or "::OBS" in key:
        return re_da
    mkey = (tuple(re_da.sizes[d] for d in ("lat", "lon")),
            int(re_da.sizes["time"]))
    m = _OBS_SAMPLING_MASK.get(mkey)
    if m is None:
        logger.warning(
            f"    {key}: CHL_MASK_TO_OBS is set but no observational "
            f"mask is registered for grid {mkey}; field left unmasked")
        return re_da
    before = float(np.isfinite(re_da).mean())
    out = re_da.where(m.values)
    after = float(np.isfinite(out).mean())
    logger.info(
        f"    {key}: masked to the observational sampling "
        f"({100*before:.0f}% -> {100*after:.0f}% of cell-months retained)")
    return out


def _regrid_z_cached(
    da:          xr.DataArray,
    common_grid: xr.DataArray,
    period:      slice,
    key:         str,
    interp_gaps: bool = False,
) -> xr.DataArray:
    """Regrid + deseasonalise + detrend + z-score, memoised by `key`.

    regrid_timeseries walks all ~204 months one at a time, so it
    dominates the cost of the sea-ice figures. The result depends only
    on the source field, not on which observational ice product it is
    being compared against — but the correlation, regression and EOF
    figures are each called once per ice product, so every model was
    being regridded twice for identical output. That doubling is what
    pushed the sea-ice stage past a 3 h wall limit once ESA-CCI began
    loading successfully.

    Keys must identify the source uniquely, e.g. "sic::CanESM5",
    "chl::OBS", "sic::HadISST".
    """
    # The cache key must encode everything the result depends on, not
    # just the source field. Fig 3 regrids the model Chl-a onto
    # `obs_clim` while the coupling figures use `common_grid`, and the
    # gap-interpolation flag differs between callers; keying on the
    # source name alone meant whichever figure ran first silently
    # decided the grid and the gap treatment for every later one. A
    # mismatch then surfaces downstream as a shape error, or worse, as
    # a correlation computed against a field on the wrong grid.
    full_key = (
        f"{key}|{tuple(common_grid.sizes.values())}"
        f"|{float(common_grid.lat.values[0]):.3f}"
        f"|{getattr(period, 'start', None)}-{getattr(period, 'stop', None)}"
        f"|gap{int(bool(interp_gaps))}"
        f"|obsmask{int(bool(MASK_TO_OBS))}"
    )
    hit = _REGRID_Z_CACHE.get(full_key)
    if hit is not None:
        return hit
    re_da = regrid_timeseries(da.sel(time=period), common_grid)
    if interp_gaps:
        re_da = _interp_time_gaps(re_da, max_gap=3)
    if key.startswith("chl::OBS"):
        _register_obs_sampling_mask(re_da)
    re_da = _apply_obs_sampling_mask(re_da, key)
    z = _deseas_detrend_zscore(re_da)
    _REGRID_Z_CACHE[full_key] = z
    nbytes = sum(v.nbytes for v in _REGRID_Z_CACHE.values())
    logger.info(
        f"    regrid cache MISS '{full_key}' -> computed "
        f"shape={tuple(z.shape)} "
        f"({len(_REGRID_Z_CACHE)} entries, {nbytes / 1e9:.2f} GB)")
    return z


_EOF1_TEMPLATES: dict[str, np.ndarray] = {}


def _pointwise_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-cell Pearson r between two (T, nlat, nlon) anomaly fields.

    Complete-case per cell, so cells with missing Chl-a months are
    handled the same way as the regression. Returns (nlat, nlon).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    ok = np.isfinite(A) & np.isfinite(B)
    n  = ok.sum(axis=0)
    Az = np.where(ok, A, 0.0)
    Bz = np.where(ok, B, 0.0)
    sa = Az.sum(axis=0)
    sb = Bz.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = sa / n
        mb = sb / n
        da = np.where(ok, A - ma[None], 0.0)
        db = np.where(ok, B - mb[None], 0.0)
        num = (da * db).sum(axis=0)
        den = np.sqrt((da ** 2).sum(axis=0) * (db ** 2).sum(axis=0))
        r = np.where((den > 0) & (n >= 12), num / den, np.nan)
    return r


def _apply_eof_sign_convention(
    eof_s:        dict,
    field_z:      xr.DataArray,
    region_mask:  np.ndarray,
    label:        str = "",
    template_key: str | None = None,
    is_template:  bool = False,
) -> dict:
    """Fix the arbitrary sign of each EOF/PC pair.

    An eigendecomposition determines EOF_k and PC_k only up to a
    common sign: (-EOF_k, -PC_k) reconstructs the field identically.
    Left alone, each model lands on whichever polarity the solver
    happened to produce, so an ensemble mean of the PC1 regression
    coefficient averages coefficients of arbitrary sign and collapses
    toward zero however strong the individual responses are. R^2 is
    invariant to this and stays high, which is the tell-tale
    signature: high ensemble R^2 alongside near-zero ensemble beta.

    Convention applied here:

      PC1 — flipped so that PC1 correlates POSITIVELY with the
            cos(lat)-weighted regional mean sea-ice anomaly. PC1 then
            means "more ice than usual across the region", so a
            positive beta means "more ice -> more Chl-a" in every
            model and in the observations, and the ensemble mean is
            a meaningful average of like-signed quantities.

      PC_k, k>=2 — higher modes are typically dipoles with near-zero
            regional mean, so the area-mean reference is degenerate
            for them. They get a deterministic rule instead: the
            loading of largest magnitude is made positive. This does
            not affect R^2 (invariant) or the plotted beta (PC1 only),
            but it makes repeat runs reproducible.

    Returns a new dict; the input is not modified.
    """
    if not eof_s or "pcs" not in eof_s or "eofs" not in eof_s:
        return eof_s

    pcs  = np.array(eof_s["pcs"],  copy=True)   # (k, T)
    eofs = np.array(eof_s["eofs"], copy=True)   # (k, nlat, nlon)

    # Reference index: cos(lat)-weighted regional mean anomaly.
    lats = np.asarray(field_z.lat.values)
    nlon = field_z.sizes["lon"]
    w2d  = np.broadcast_to(
        np.cos(np.deg2rad(lats))[:, None], (lats.size, nlon)).copy()
    w2d  = np.where(region_mask, w2d, 0.0)

    X = np.asarray(field_z.values, dtype=float)          # (T, nlat, nlon)
    fin = np.isfinite(X)
    num = np.nansum(np.where(fin, X, 0.0) * w2d[None], axis=(1, 2))
    den = np.sum(np.where(fin, w2d[None], 0.0),        axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = np.where(den > 0, num / den, np.nan)       # (T,)

    # --- PC1 sign ---
    #
    # Preferred rule: align EOF1 to a common spatial TEMPLATE, taken
    # from the reference observational product for this region. The
    # earlier rule oriented PC1 to correlate positively with the
    # cos(lat)-weighted regional MEAN ice anomaly, which works in the
    # Arctic (r = 0.19-0.86) but fails in the Southern Ocean, where
    # EOF1 is a dipole with near-zero regional mean (r = 0.03-0.13).
    # At that magnitude the flip is a coin toss, and HadISST and
    # ESA-CCI landed on opposite polarities — producing beta maps of
    # opposite sign from the same chlorophyll field.
    #
    # Pattern alignment is well posed for a dipole: the spatial
    # correlation between two EOF1 maps is large in magnitude whether
    # or not they share a polarity, so its SIGN is a reliable
    # orientation. Every source (both obs products and all seven
    # models) is aligned to the same template, which is what makes the
    # ensemble mean and the two obs rows comparable.
    flipped = []
    p0 = pcs[0]
    r_ref = np.nan
    template = _EOF1_TEMPLATES.get(template_key) if template_key else None
    method = "regional-mean"

    if template is not None and template.shape == eofs[0].shape:
        a = eofs[0].ravel()
        b = template.ravel()
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 20 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
            r_ref = float(np.corrcoef(a[ok], b[ok])[0, 1])
            method = f"template '{template_key}'"
            if r_ref < 0:
                pcs[0]  = -pcs[0]
                eofs[0] = -eofs[0]
                flipped.append(1)
                r_ref = -r_ref

    if method == "regional-mean":
        # No usable template (this source IS the template, or the
        # template is missing): fall back to the regional-mean rule,
        # then to a deterministic one.
        ok = np.isfinite(p0) & np.isfinite(ref)
        if ok.sum() >= 12 and np.std(p0[ok]) > 0 and np.std(ref[ok]) > 0:
            r_ref = float(np.corrcoef(p0[ok], ref[ok])[0, 1])
            if r_ref < 0:
                pcs[0]  = -pcs[0]
                eofs[0] = -eofs[0]
                flipped.append(1)
                r_ref = -r_ref
        else:
            e0 = eofs[0]
            j = np.nanargmax(np.abs(np.where(np.isfinite(e0), e0, 0.0)))
            if e0.ravel()[j] < 0:
                pcs[0]  = -pcs[0]
                eofs[0] = -eofs[0]
                flipped.append(1)
            method = "deterministic"

    # --- Higher modes: largest-magnitude loading made positive ---
    for k in range(1, eofs.shape[0]):
        ek = eofs[k]
        flat = np.where(np.isfinite(ek), ek, 0.0).ravel()
        if not np.any(flat):
            continue
        if flat[np.argmax(np.abs(flat))] < 0:
            pcs[k]  = -pcs[k]
            eofs[k] = -eofs[k]
            flipped.append(k + 1)

    # Store the template AFTER its own polarity has been fixed, so
    # every later source is aligned to an already-oriented pattern.
    if is_template and template_key:
        _EOF1_TEMPLATES[template_key] = np.array(eofs[0], copy=True)
        if label:
            logger.info(
                f"    {label}: stored EOF1 template '{template_key}'")

    if label:
        logger.info(
            f"    {label}: EOF sign via {method}, r={r_ref:+.3f}"
            f"{' after flip' if 1 in flipped else ''}; "
            f"flipped PCs {flipped if flipped else 'none'}")

    out = dict(eof_s)
    out["pcs"]  = pcs
    out["eofs"] = eofs
    return out


def make_figS_seaice_eof(
    obs_siconc_ts:    xr.DataArray | None,
    model_siconc_ts:  dict[str, xr.DataArray],
    common_grid:      xr.DataArray,
    period:           slice = LAG_PERIOD,
    n_eof:            int   = 5,
    obs_siconc_name:  str   = SICONC_OBS_NAME,
    out_tag:          str   = "",
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

    `obs_siconc_name` sets the OBS labels and `out_tag` separates the
    filenames, so calling this once per ice product gives a matched
    pair. This is worth doing rather than trusting one product: the
    leading EOFs computed here are the predictors used by
    make_figS_seaice_chl_regression, so if the two products disagree
    on the ice modes themselves, that propagates directly into the
    R^2 and beta maps and has to be diagnosed here rather than there.
    Compare EOF1 patterns, PC1 series and the variance spectra —
    especially the number of EOFs needed to reach 50% of variance,
    which sets the predictor count in the regression.

    Output: figS_seaice_eof{out_tag}.png
    """
    logger.info(
        f"=== Fig S_seaice_eof ({obs_siconc_name}): "
        f"sea-ice EOF analysis ===")

    if obs_siconc_ts is None and not model_siconc_ts:
        logger.warning(
            f"  No siconc data ({obs_siconc_name}); "
            f"skipping figS_seaice_eof{out_tag}")
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
                obs_eof = _apply_eof_sign_convention(
                    obs_eof, sic_z, region_mask,
                    label=f"OBS {region_name} ({obs_siconc_name})",
                    template_key=f"eof::{region_name}",
                    is_template=(obs_siconc_name == SICONC_OBS_NAME))
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
                    eof_m = _apply_eof_sign_convention(
                        eof_m, sic_z, region_mask,
                        label=f"[{model}] {region_name}",
                        template_key=f"eof::{region_name}",
                        is_template=False)
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
            f"{region_name} — OBS EOF1 ({obs_siconc_name})\n"
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
                       lw=1.8, label=f"OBS ({obs_siconc_name})")
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
                      label=f"OBS ({obs_siconc_name})")

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
        f"Sea-ice concentration EOF analysis ({obs_siconc_name} obs) "
        f"— Arctic (top) and Southern Ocean SIZ (bottom)\n"
        "Input: deseasonalised, linearly detrended, standardised "
        "per cell (unit variance).  "
        "Area weights: cos(lat).  Period: 1998–2014.\n"
        "Col 1: OBS EOF1 spatial pattern.  "
        "Col 2: PC1 time series (black: OBS, grey: models).  "
        "Col 3: Variance spectrum (bars: OBS, line+shading: ensemble).",
        fontsize=8.5, y=0.97,
    )
    fname = f"figS_seaice_eof{out_tag}.png"
    fig.savefig(OUT_DIR / fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  wrote {fname}")


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
    obs_siconc_name: str   = SICONC_OBS_NAME,
    out_tag:         str   = "",
) -> None:
    """Chl-a variance explained by leading sea-ice modes, resolved by lag.

    Companion to make_figS_seaice_chl_spatial_corr, and deliberately
    built to the same layout so the two can be read side by side:
    2 rows (OBS | ensemble mean) x 4 columns (lags 0 to +3 months).
    The lag convention is identical — sea ice at month t against
    Chl-a at month t+lag, so positive lag means ice leads.

    The two diagnostics answer different questions about the same
    pairing, which is why they are now separate figures rather than
    separate columns of one figure:

      correlation maps  — does the LOCAL ice anomaly covary with the
                          local Chl-a anomaly?
      regression maps   — how much local Chl-a variance is explained
                          by the LARGE-SCALE ice modes (leading EOFs
                          of the regional ice field)?

    A cell can have weak local correlation but high R^2 if it responds
    to basin-scale ice variability rather than to the ice directly
    overhead, so the pair separates local coupling from remote or
    mode-driven control.

    Writes four files per call:
      figS_seaice_chl_R2{out_tag}_{Arctic,Southern}.png
      figS_seaice_chl_beta{out_tag}_{Arctic,Southern}.png

    R^2 is the fraction of local Chl-a variance explained by multiple
    regression on enough leading sea-ice PCs to cover >=50% of ice
    variance; beta is the PC1 coefficient. Cells with p >= 0.05
    (F-test on the full model) are washed out. `obs_siconc_name` sets
    the row label and `out_tag` separates the output filenames, so
    calling this once per ice product gives directly comparable pairs.

    The EOF decomposition does not depend on lag, so it is computed
    once per region and source and reused across all four lags — only
    the regression is repeated.
    """
    from scipy import stats as _stats

    logger.info(
        f"=== Fig S_seaice_chl regression ({obs_siconc_name}): "
        f"Chl-a variance explained by sea-ice modes, by lag ==="
    )

    if obs_siconc_ts is None and not model_siconc_ts:
        logger.warning(
            f"  No siconc ({obs_siconc_name}); skipping regression figures")
        return

    # Shared with the correlation figure via SEAICE_REGIONS so the two
    # families cover exactly the same domain and can be compared panel
    # by panel.
    regions = {
        name: {
            "sic_lat": (r["lat_min"], r["lat_max"]),
            "chl_lat": (r["lat_min"], r["lat_max"]),
            "proj":    r["proj"],
            "extent":  r["extent"],
        }
        for name, r in SEAICE_REGIONS.items()
    }

    lats = common_grid.lat.values
    lons = common_grid.lon.values
    lat2d, _ = np.meshgrid(lats, lons, indexing="ij")
    models_list = [m for m in MODELS
                   if m in model_chl_ts and m in model_siconc_ts]

    def _regress_pcs_onto_chl(
        pcs:      np.ndarray,   # (n_pcs, T)
        chl_arr:  np.ndarray,   # (T, nlat, nlon)
        var_frac: np.ndarray,   # (n_pcs,)
        quiet:    bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Multiple regression of leading sea-ice PCs onto Chl-a.

        Uses enough PCs to explain >= 50% of sea-ice variance.
        Returns R^2, beta_PC1 and the F-test p-value, each
        (nlat, nlon). Identical maths to the previous version; it now
        takes a plain array so the caller can slice it for each lag.
        """
        cum_var = np.cumsum(var_frac)
        n_pcs   = int(np.searchsorted(cum_var, EOF_VAR_TARGET)) + 1
        n_pcs   = min(n_pcs, pcs.shape[0])
        if not quiet:
            logger.info(
                f"    multiple regression: {n_pcs} PCs "
                f"(cumulative variance {cum_var[n_pcs-1]*100:.1f}%)")

        C = np.asarray(chl_arr, dtype=float)
        nlat, nlon = C.shape[1], C.shape[2]
        r2  = np.full((nlat, nlon), np.nan)
        bet = np.full((nlat, nlon), np.nan)
        pv  = np.full((nlat, nlon), np.nan)

        X_full = pcs[:n_pcs].T                        # (T, n_pcs)
        x_ok   = np.all(np.isfinite(X_full), axis=1)
        X_v    = X_full[x_ok]
        C_v    = C[x_ok]
        T_v    = X_v.shape[0]
        if T_v < n_pcs + 3:
            return r2, bet, pv

        C_flat = C_v.reshape(T_v, -1)
        M      = np.isfinite(C_flat)
        n_fin  = np.sum(M, axis=0)

        # Minimum valid months per cell. The old guard (n_pcs + 2) was
        # far too lax once the 50%-variance criterion started selecting
        # ~17 predictors: with 20 months and 17 predictors the fit is
        # near-saturated and beta_PC1 has mean |beta| = 0.62 even when
        # the true coefficient is exactly zero (95th pct 1.76). That is
        # what produced uniformly saturated OBS panels in the Southern
        # Ocean, where winter gaps leave few valid months. Requiring
        # ~4 observations per predictor brings the null |beta| down to
        # ~0.12 and makes the map interpretable.
        min_obs = max(MIN_OBS_PER_PREDICTOR * n_pcs, n_pcs + 20)
        valid   = n_fin >= min_obs
        if not quiet:
            logger.info(
                f"    sample-size guard: n_pcs={n_pcs}, requiring "
                f">={min_obs} valid months "
                f"({int(valid.sum())} of {int((n_fin > 0).sum())} "
                f"wet cells pass)")

        # Complete-case regression, grouped by missing-data pattern.
        #
        # The previous implementation substituted 0.0 for missing Chl-a
        # months before the solve. Because the anomalies are z-scored,
        # the cell mean is ~0, so a missing month added nothing to
        # ss_tot but contributed C_hat^2 to ss_res — deflating R^2 in
        # proportion to the gap fraction (measured on synthetic data:
        # -0.10 at 40% gaps, -0.15 at 60%). Ocean-colour Chl-a is
        # missing through polar night and under ice, well beyond the
        # 3-month interpolation window, while model Chl-a has no gaps
        # at all. The bias therefore fell almost entirely on the
        # observations, and precisely in the seasonal-ice zone.
        #
        # Cells sharing a missing-data pattern share a design matrix,
        # so they are solved together: gaps are set by polar night and
        # ice cover, which vary smoothly in space, giving far fewer
        # distinct patterns than cells.
        ncell      = C_flat.shape[1]
        r2_flat    = np.full(ncell, np.nan)
        beta1_flat = np.full(ncell, np.nan)
        n_used     = np.full(ncell, np.nan)
        cond_used  = np.full(ncell, np.nan)

        # Only cells with enough valid months can be solved at all;
        # everything else is masked out downstream anyway. Restricting
        # here keeps the pattern-grouping off the ~3/4 of the global
        # grid that lies outside the polar region.
        use = np.flatnonzero(valid)
        if use.size:
            packed  = np.packbits(
                np.ascontiguousarray(M[:, use].T), axis=1)
            _, inv  = np.unique(packed, axis=0, return_inverse=True)
            inv     = np.asarray(inv).ravel()
            n_groups = int(inv.max()) + 1

            # Bucket the cells by group in ONE pass. The obvious
            # np.flatnonzero(inv == g) inside the loop is O(n_groups x
            # ncell): ocean-colour gaps are per-pixel and irregular, so
            # the Southern Ocean produces tens of thousands of distinct
            # patterns and that product runs into billions of
            # operations per lag per model. Sorting once and slicing is
            # linear.
            order  = np.argsort(inv, kind="stable")
            inv_s  = inv[order]
            bounds = np.searchsorted(inv_s, np.arange(n_groups + 1))

            if not quiet:
                logger.info(
                    f"    complete-case regression: {use.size} usable "
                    f"cells in {n_groups} distinct missing-data patterns")

            for g in range(n_groups):
                sel = use[order[bounds[g]:bounds[g + 1]]]
                if sel.size == 0:
                    continue
                rows = M[:, sel[0]]
                n_g  = int(rows.sum())
                if n_g < min_obs:
                    continue
                Xg = X_v[rows]
                Xg = Xg - Xg.mean(axis=0, keepdims=True)
                Cg = C_flat[np.ix_(rows, sel)]
                Cg = Cg - Cg.mean(axis=0, keepdims=True)
                XtXg = Xg.T @ Xg
                XtCg = Xg.T @ Cg
                try:
                    bg = np.linalg.solve(XtXg, XtCg)
                except np.linalg.LinAlgError:
                    bg = np.linalg.lstsq(XtXg, XtCg, rcond=None)[0]
                resid = Cg - Xg @ bg
                ssr_g = np.sum(resid ** 2, axis=0)
                sst_g = np.sum(Cg ** 2,    axis=0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    r2_g = np.where(sst_g > 0, 1.0 - ssr_g / sst_g, np.nan)
                # Adjusted R^2. Complete-case regression removed the
                # deflation caused by zero-filling, but left the
                # opposite confound: with fewer valid months the same
                # n_pcs predictors explain more variance by chance
                # (E[R^2] ~ k/(n-1) under the null). Observations have
                # far fewer valid months than the gap-free models in
                # the ice zone, so raw R^2 would now favour the
                # observations for a purely statistical reason.
                dof = n_g - n_pcs - 1
                if dof > 0:
                    r2_g = 1.0 - (1.0 - r2_g) * (n_g - 1) / dof
                r2_flat[sel]    = np.clip(r2_g, 0.0, 1.0)
                beta1_flat[sel] = bg[0]
                n_used[sel]     = n_g
                cond_used[sel]  = float(np.linalg.cond(XtXg))

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

        # ---- Beta from a SIMPLE regression on PC1 alone ----
        #
        # R^2 above is the multiple regression on all n_pcs retained
        # modes: "how much Chl-a variance do the ice modes collectively
        # explain". Beta is a different question — "how does Chl-a
        # respond to the leading ice mode" — and taking it as the
        # partial coefficient from that same 15-predictor fit was the
        # root of most of this figure's trouble. A partial coefficient
        # is the response to PC1 holding the other 14 modes fixed,
        # which is not the quantity the text describes, and it inherits
        # the multiple regression's sample-size cost (45 months) and
        # its collinearity problem once complete-case selection breaks
        # PC orthogonality.
        #
        # Fitting beta separately, on PC1 only, restores the textbook
        # interpretation (Chl-a z-score per 1 s.d. of PC1), drops the
        # sample requirement from ~3*n_pcs to ~10 months, and makes the
        # conditioning test vacuous: with one predictor there is
        # nothing to be collinear with. Since both series are z-scored
        # and PC1 is unit variance, this beta equals the pointwise
        # correlation between Chl-a and PC1.
        pc1 = X_v[:, 0]
        beta1_simple = np.full(ncell, np.nan)
        n_simple     = np.full(ncell, np.nan)
        p1c = pc1 - pc1.mean()
        for g in range(n_groups):
            sel = use[order[bounds[g]:bounds[g + 1]]]
            if sel.size == 0:
                continue
            rows = M[:, sel[0]]
            n_g  = int(rows.sum())
            if n_g < BETA_MIN_OBS_SIMPLE:
                continue
            xg = pc1[rows]
            xg = xg - xg.mean()
            sxx = float(xg @ xg)
            if sxx <= 0:
                continue
            Cg = C_flat[np.ix_(rows, sel)]
            Cg = Cg - Cg.mean(axis=0, keepdims=True)
            beta1_simple[sel] = (xg @ Cg) / sxx
            n_simple[sel]     = n_g
        beta1_flat = beta1_simple

        beta_min = BETA_MIN_OBS_SIMPLE
        beta_ok  = np.isfinite(beta1_flat) & (n_simple >= beta_min)
        if not quiet:
            fin = np.isfinite(n_used)
            if fin.any():
                logger.info(
                    f"    beta (simple regression on PC1): "
                    f"{int(beta_ok.sum())} cells retained, requiring "
                    f">={beta_min} months "
                    f"(R2 multiple regression retained "
                    f"{int(valid.sum())} at >={min_obs} months); "
                    f"multiple-fit cond median="
                    f"{np.nanmedian(cond_used):.1f} "
                    f"p99={np.nanpercentile(cond_used, 99):.1f}")
                bb = beta1_flat[beta_ok]
                if bb.size:
                    logger.info(
                        f"    beta after mask: p50="
                        f"{np.nanpercentile(np.abs(bb), 50):.3f} "
                        f"p99={np.nanpercentile(np.abs(bb), 99):.3f} "
                        f"max={np.nanmax(np.abs(bb)):.3f}")

        # Beta is returned UNMASKED (only the shared `valid` test is
        # applied, as for R^2); the beta-specific mask is returned
        # alongside so the caller can choose one reference lag's
        # footprint for every panel. Masking here instead would make
        # that impossible: once a cell is NaN it cannot be restored,
        # so a wider reference mask would have no effect.
        r2[...]  = np.where(valid, r2_flat,    np.nan).reshape(nlat, nlon)
        pv[...]  = np.where(valid, pv_flat,    np.nan).reshape(nlat, nlon)
        bet[...] = np.where(valid, beta1_flat, np.nan).reshape(nlat, nlon)
        bmask    = beta_ok.reshape(nlat, nlon)
        return r2, bet, pv, bmask

    def _prepare_source(
        sic_ts: xr.DataArray,
        chl_ts: xr.DataArray,
        sic_mask: np.ndarray,
        label: str,
        interp_chl: bool,
        sic_key: str,
        chl_key: str,
        template_key: str,
        is_template: bool,
    ):
        """EOF the ice field and align PCs with Chl-a on common months.

        Returns (pcs_aligned, chl_aligned, var_frac) or None. Done once
        per source; the lag is applied afterwards by slicing, exactly
        as _compute_r_beta does, so both figures share one convention.
        """
        # Cached: the regrid + z-score is identical for both
        # observational ice products; the region mask only enters
        # inside _compute_eof.
        sic_z  = _regrid_z_cached(sic_ts, common_grid, period, sic_key)
        # n_eof must exceed the number of modes needed to reach the 50%
        # target, or the criterion is silently unmet. Per-cell z-scoring
        # whitens the field, spreading variance across many modes: with
        # n_eof=10 every source reported "50% at PC11", i.e. searchsorted
        # ran off the end and the regression used all 10 modes while
        # explaining LESS than 50% of ice variance.
        eof_s  = _compute_eof(sic_z, sic_mask, n_eof=EOF_N_MODES)
        if not eof_s or "pcs" not in eof_s:
            logger.warning(f"  {label}: EOF failed")
            return None
        # Fix the arbitrary EOF sign BEFORE the regression, so that
        # beta_PC1 is comparable across models and the ensemble mean
        # is an average of like-signed coefficients.
        eof_s = _apply_eof_sign_convention(
            eof_s, sic_z, sic_mask, label=label,
            template_key=template_key, is_template=is_template)

        # Normalise each PC to unit variance before the regression.
        #
        # _compute_eof returns pcs = U[:,i] * S[i], so PC amplitude
        # scales with the singular value and therefore differs between
        # every model and the observations. beta is the coefficient of
        # a unit-variance Chl-a anomaly on that PC, so beta ~ 1/S1 and
        # its units are "Chl-a z-score per unit of THIS source's PC1" —
        # not a common unit, and not averageable across models.
        #
        # The bias is not random. A spatially smoother, more coherent
        # ice field concentrates variance in mode 1, raising S1 and
        # lowering beta. CMIP6 ice fields are smoother than the
        # observed field, so raw beta makes every model look weaker
        # than observations even when the underlying response is
        # identical (verified on synthetic fields: 1.7x apparent
        # weakening from coherence alone).
        #
        # After normalising, beta is "Chl-a z-score change per 1 s.d.
        # of PC1" — dimensionless, comparable across sources, and
        # averageable. R^2 is scale-invariant and unchanged.
        pcs_raw = eof_s["pcs"]
        sd = np.nanstd(pcs_raw, axis=1, keepdims=True)
        eof_s = dict(eof_s)
        eof_s["pcs"] = pcs_raw / np.where(sd > 0, sd, 1.0)
        logger.info(
            f"    {label}: PC1 s.d. {float(sd[0, 0]):.3f} -> normalised "
            f"to unit variance (beta now per 1 s.d. of PC1)")

        pcs      = eof_s["pcs"]
        var_frac = eof_s["var_frac"]

        chl_z = _regrid_z_cached(chl_ts, common_grid, period, chl_key,
                                 interp_gaps=interp_chl)

        # z-scoring preserves the time axis, so the year-month keys are
        # the same as they would be on the pre-anomaly fields.
        ym_p = _ym_key(sic_z)
        ym_c = _ym_key(chl_z)
        common_ym = np.intersect1d(ym_p, ym_c)
        if common_ym.size < 24:
            logger.warning(
                f"  {label}: only {common_ym.size} common months")
            return None
        p_idx = np.argsort(ym_p)[np.searchsorted(np.sort(ym_p), common_ym)]
        c_idx = np.argsort(ym_c)[np.searchsorted(np.sort(ym_c), common_ym)]

        cum = np.cumsum(var_frac)
        k50 = int(np.searchsorted(cum, EOF_VAR_TARGET)) + 1
        if k50 > cum.size:
            logger.warning(
                f"  {label}: 50% of ice variance NOT reached within "
                f"{cum.size} modes (cumulative {cum[-1]*100:.1f}%). "
                f"The regression will use all {cum.size} modes and the "
                f"50% criterion is not met — raise EOF_N_MODES.")
        else:
            logger.info(
                f"  {label}: EOF ok, {common_ym.size} common months, "
                f"50% of ice variance at PC{k50} "
                f"(PC1 alone {var_frac[0]*100:.1f}%, "
                f"{cum.size} modes = {cum[-1]*100:.1f}%)")
        # sic_a is the aligned LOCAL ice anomaly, returned so the
        # caller can compute the pointwise correlation r and compare
        # r^2 against the mode-based R^2. That comparison is the test
        # of whether the regression adds anything the correlation maps
        # do not already show.
        return (pcs[:, p_idx],
                np.asarray(chl_z.values, dtype=float)[c_idx],
                var_frac,
                np.asarray(sic_z.values, dtype=float)[p_idx])

    for region_name, reg in regions.items():
        try:
            sic_lat = reg["sic_lat"]
            chl_lat = reg["chl_lat"]
            proj    = reg["proj"]
            extent  = reg["extent"]

            sic_mask = ((lat2d >= sic_lat[0]) & (lat2d <= sic_lat[1]))
            chl_mask = ((lat2d >= chl_lat[0]) & (lat2d <= chl_lat[1]))

            logger.info(f"  --- {region_name} ({obs_siconc_name}) ---")

            # ---- OBS: EOF once, then regress at each lag ----
            obs_by_lag: dict[int, tuple] = {}
            obs_r2_local: dict[int, np.ndarray] = {}
            obs_bmask: dict[int, np.ndarray] = {}
            if obs_siconc_ts is not None:
                try:
                    prep = _prepare_source(
                        obs_siconc_ts, obs_chl_ts, sic_mask,
                        f"OBS {region_name}", interp_chl=True,
                        sic_key=f"sic::{obs_siconc_name}",
                        chl_key="chl::OBS",
                        template_key=f"regr::{region_name}",
                        is_template=(obs_siconc_name == SICONC_OBS_NAME))
                    if prep is not None:
                        pcs_a, chl_a, vf, sic_a = prep
                        T = min(pcs_a.shape[1], chl_a.shape[0])
                        for lag in LAGS:
                            if lag >= T:
                                continue
                            p_l = pcs_a[:, :T - lag] if lag > 0 else pcs_a[:, :T]
                            c_l = chl_a[lag:T]       if lag > 0 else chl_a[:T]
                            r2, bt, pv, bmask = _regress_pcs_onto_chl(
                                p_l, c_l, vf, quiet=(lag != 0))
                            r2[~chl_mask] = np.nan
                            bt[~chl_mask] = np.nan
                            pv[~chl_mask] = np.nan
                            bmask = bmask & chl_mask
                            obs_by_lag[lag]  = (r2, bt, pv)
                            obs_bmask[lag]   = bmask

                            # --- Does the regression add anything the
                            # correlation map does not already show?
                            # Compare the mode-based R^2 against the
                            # squared POINTWISE correlation of local ice
                            # with local Chl-a at the same lag.
                            #   R^2 ~= r^2  -> regression is recovering
                            #                  local coupling, redundant
                            #   R^2 >> r^2  -> that cell responds to
                            #                  BASIN-SCALE ice
                            #                  variability, which the
                            #                  correlation map cannot see
                            s_l = sic_a[:T - lag] if lag > 0 else sic_a[:T]
                            r_loc = _pointwise_corr(s_l, c_l)
                            r_loc[~chl_mask] = np.nan
                            r2_loc = r_loc ** 2
                            obs_r2_local[lag] = r2_loc
                            gain = r2 - r2_loc
                            both = np.isfinite(r2) & np.isfinite(r2_loc)
                            if both.sum() > 20:
                                sp = float(np.corrcoef(
                                    r2[both], r2_loc[both])[0, 1])
                                logger.info(
                                    f"  OBS {region_name} lag{lag:+d} "
                                    f"REDUNDANCY CHECK: median R2="
                                    f"{np.nanmedian(r2[both]):.3f} vs "
                                    f"median r^2_local="
                                    f"{np.nanmedian(r2_loc[both]):.3f}; "
                                    f"median gain={np.nanmedian(gain[both]):+.3f}; "
                                    f"spatial corr(R2, r^2)={sp:+.3f}; "
                                    f"{100 * np.mean(gain[both] > 0.05):.0f}% "
                                    f"of cells gain >0.05")
                            logger.success(
                                f"  OBS {region_name} lag{lag:+d}: "
                                f"median R2={np.nanmedian(r2):.3f}, "
                                f"sig cells={int(np.nansum(pv < 0.05))}")
                except Exception as e:
                    import traceback as _tb
                    logger.warning(
                        f"  OBS {region_name} regression failed: {e}")
                    logger.warning(_tb.format_exc().replace("<", "\\<"))

            # ---- Models: EOF once per model, then regress at each lag ----
            model_r2:    dict[int, list[np.ndarray]] = {L: [] for L in LAGS}
            model_beta:  dict[int, list[np.ndarray]] = {L: [] for L in LAGS}
            model_bmask: dict[int, list[np.ndarray]] = {L: [] for L in LAGS}
            for model in models_list:
                try:
                    prep = _prepare_source(
                        model_siconc_ts[model], model_chl_ts[model], sic_mask,
                        f"[{model}] {region_name}", interp_chl=False,
                        sic_key=f"sic::{model}",
                        chl_key=f"chl::{model}",
                        template_key=f"regr::{region_name}",
                        is_template=False)
                    if prep is None:
                        continue
                    pcs_a, chl_a, vf, _sic_a = prep
                    T = min(pcs_a.shape[1], chl_a.shape[0])
                    for lag in LAGS:
                        if lag >= T:
                            continue
                        p_l = pcs_a[:, :T - lag] if lag > 0 else pcs_a[:, :T]
                        c_l = chl_a[lag:T]       if lag > 0 else chl_a[:T]
                        r2_m, bt_m, _, bmask_m = _regress_pcs_onto_chl(
                            p_l, c_l, vf, quiet=True)
                        r2_m[~chl_mask] = np.nan
                        bt_m[~chl_mask] = np.nan
                        model_r2[lag].append(r2_m)
                        model_beta[lag].append(bt_m)
                        model_bmask[lag].append(bmask_m & chl_mask)
                except Exception as e:
                    logger.warning(
                        f"  [{model}] {region_name} regression failed: {e}")

            ens_r2   = {L: (np.nanmean(np.stack(v, 0), 0) if v else None)
                        for L, v in model_r2.items()}
            ens_beta = {L: (np.nanmean(np.stack(v, 0), 0) if v else None)
                        for L, v in model_beta.items()}
            n_models = max((len(v) for v in model_r2.values()), default=0)

            # ---- Common mask across lags, taken from BETA_MASK_LAG ----
            # The sample-size and conditioning tests are evaluated on
            # the lag-specific subset of months: at lag L the fit pairs
            # pcs[:, :T-L] with chl[L:T], so both the number of valid
            # months and their SEASONAL composition change with lag
            # (ice months are summer at lag 0, spring at lag +3).
            # Cells near either threshold flipped in and out between
            # columns, giving each panel a different coastline of
            # missing data and making the columns non-comparable.
            #
            # One reference lag now sets the footprint for all four
            # panels. This is the widest sensible choice when that lag
            # has the best coverage, but note the trade-off: cells that
            # pass at the reference lag but fail at another are still
            # shown there, so a few cells in the other columns are
            # under-determined relative to the stated criterion. The
            # alternative (intersecting all lags) is stricter but
            # discards cells that are perfectly well sampled at the
            # lag being plotted.
            def _mask_from_lag(dct, getter, ref_lag):
                a = getter(dct, ref_lag)
                if a is None:
                    for L in LAGS:                     # fall back
                        a = getter(dct, L)
                        if a is not None:
                            break
                return None if a is None else np.asarray(a, dtype=bool)

            obs_common = _mask_from_lag(
                obs_bmask, lambda d, L: d.get(L), BETA_MASK_LAG)
            if obs_common is not None:
                n_before = int(np.mean([int(m.sum())
                                        for m in obs_bmask.values()]))
                for L in list(obs_by_lag):
                    r2_L, bt_L, pv_L = obs_by_lag[L]
                    obs_by_lag[L] = (r2_L,
                                     np.where(obs_common, bt_L, np.nan),
                                     pv_L)
                logger.info(
                    f"  {region_name} beta mask taken from lag"
                    f"{BETA_MASK_LAG:+d} and applied to all lags: "
                    f"{int(obs_common.sum())} cells retained "
                    f"(per-lag mean was {n_before})")

            # For the ensemble, require the reference lag's mask to
            # hold in a majority of models.
            ens_bmask = {L: (np.mean(np.stack(v, 0), 0) >= 0.5 if v else None)
                         for L, v in model_bmask.items()}
            ens_common = _mask_from_lag(ens_bmask, lambda d, L: d.get(L),
                                        BETA_MASK_LAG)
            if ens_common is not None:
                ens_beta = {L: (np.where(ens_common, a, np.nan)
                                if a is not None else None)
                            for L, a in ens_beta.items()}

            # ---- Sign-cancellation diagnostic ----
            # If EOF signs were still arbitrary, averaging beta across
            # models would cancel: mean(|beta_m|) would greatly exceed
            # |mean_m(beta_m)|. With the sign convention applied the two
            # should be comparable wherever the models agree. A ratio near
            # 1 means coherent response; a ratio >> 1 means either
            # genuine model disagreement or a convention that failed.
            for lag in LAGS:
                stack = model_beta.get(lag, [])
                if len(stack) < 2:
                    continue
                B = np.stack(stack, 0)                     # (n_models, ...)
                mean_abs = np.nanmean(np.abs(B))
                abs_mean = np.abs(np.nanmean(B, axis=0))
                abs_mean = np.nanmean(abs_mean)
                ratio = (mean_abs / abs_mean) if abs_mean > 0 else np.inf
                # per-model sign agreement on PC1 beta
                sgn = np.sign(B)
                with np.errstate(invalid="ignore"):
                    agree = np.nanmean(np.abs(np.nanmean(sgn, axis=0)))
                logger.info(
                    f"  {region_name} lag{lag:+d} beta cancellation check: "
                    f"mean|beta|={mean_abs:.4f}  |mean beta|={abs_mean:.4f}  "
                    f"ratio={ratio:.2f}  mean|sign agreement|={agree:.2f} "
                    f"(1.0 = all {n_models} models same sign)")
                if ratio > 3.0:
                    logger.warning(
                        f"  {region_name} lag{lag:+d}: beta still largely "
                        f"cancels across models (ratio {ratio:.1f}). Either "
                        f"the models genuinely disagree on the sign of the "
                        f"ice-Chl response, or the PC1 sign reference is "
                        f"degenerate for this region — check the "
                        f"corr(PC1, regional mean ice anomaly) values above.")

            # ---- Shared colour scales across all lags of this region ----
            r2_pool = [obs_by_lag[L][0] for L in obs_by_lag] + \
                      [a for a in ens_r2.values() if a is not None]
            b_pool  = [obs_by_lag[L][1] for L in obs_by_lag] + \
                      [a for a in ens_beta.values() if a is not None]

            def _p95(arrs, absolute=False, default=0.3, cap=0.5,
                     pct=99.0):
                """Robust upper limit for the colour scale.

                The scale is pooled over all four lags on purpose, so
                the columns stay comparable. With the 95th percentile
                that meant whichever lag carried the largest amplitude
                saturated: the ESA-CCI Southern lag+0 and lag+1 OBS
                panels went uniformly to the top of the scale. A higher
                percentile and a looser cap still clip genuine outlier
                cells but stop clipping the bulk of a whole panel.
                """
                vals = [a.ravel()[np.isfinite(a.ravel())] for a in arrs]
                vals = [v for v in vals if v.size]
                if not vals:
                    return default
                cat = np.concatenate(vals)
                cat = np.abs(cat) if absolute else cat
                hi  = float(min(np.percentile(cat, pct), cap))
                logger.info(
                    f"    colour scale: p{pct:.0f}="
                    f"{np.percentile(cat, pct):.3f}  max="
                    f"{np.nanmax(cat):.3f}  cap={cap}  -> limit {hi:.3f} "
                    f"({100 * np.mean(cat > hi):.1f}% of cells clipped)")
                return hi

            r2_vmax = _p95(r2_pool, default=0.3, cap=0.8, pct=99.0)
            # beta is now standardised (Chl-a z-score per 1 s.d. of PC1),
            # so it is bounded near +-1 rather than the ~+-0.05 of the old
            # unnormalised units. Cap accordingly or the polar panels clip.
            bv      = _p95(b_pool, absolute=True, default=0.3,
                           cap=1.0, pct=98.0)
            r2_levels   = np.linspace(0, max(r2_vmax, 0.05), 11)
            beta_levels = np.linspace(-max(bv, 0.05), max(bv, 0.05), 13)

            # ---- One figure per metric ----
            specs = [
                ("R2", "$R^2$", r2_levels, "YlOrRd",
                 lambda L: (obs_by_lag[L][0] if L in obs_by_lag else None),
                 lambda L: ens_r2.get(L),
                 "Fraction of local Chl-$a$ variance explained"),
                ("beta", r"$\beta$", beta_levels, "RdBu_r",
                 lambda L: (obs_by_lag[L][1] if L in obs_by_lag else None),
                 lambda L: ens_beta.get(L),
                 "PC1 regression coefficient"),
            ]

            # Redundancy map: variance explained by the BASIN-SCALE ice modes
            # beyond what the LOCAL ice correlation already captures. Near
            # zero everywhere means the regression tells the same story as the
            # correlation maps and does not earn a separate figure; positive
            # values mark cells whose Chl-a responds to remote ice variability.
            if obs_by_lag and obs_r2_local:
                _gp = [obs_by_lag[L][0] - obs_r2_local[L]
                       for L in obs_by_lag if L in obs_r2_local]
                if _gp:
                    # Sequential 0 -> max, not diverging: the gain is
                    # essentially never negative, so half a diverging
                    # bar was unused and the dynamic range wasted.
                    _gv = _p95(_gp, absolute=True, default=0.2,
                               cap=1.0, pct=99.0)
                    _gl = np.linspace(0.0, max(_gv, 0.05), 11)

                    def _gain_map(L):
                        if L not in obs_by_lag or L not in obs_r2_local:
                            return None
                        g = obs_by_lag[L][0] - obs_r2_local[L]
                        # Suppress single-cell speckle: keep only cells
                        # whose 8 neighbours also show a gain, so what
                        # survives is spatially coherent rather than
                        # per-cell noise riding on a real signal.
                        pos = np.isfinite(g) & (g > GAIN_SPECKLE_MIN)
                        nb = np.zeros_like(pos, dtype=int)
                        for di in (-1, 0, 1):
                            for dj in (-1, 0, 1):
                                if di == 0 and dj == 0:
                                    continue
                                nb += np.roll(np.roll(pos, di, axis=0),
                                              dj, axis=1).astype(int)
                        keep = pos & (nb >= GAIN_MIN_NEIGHBOURS)
                        return np.where(keep, g, np.nan)

                    specs.append(
                        ("R2gain", r"$R^2-r^2_{local}$", _gl, "YlOrBr",
                         _gain_map,
                         lambda L: None,
                         "Additional Chl-$a$ variance explained by "
                         "basin-scale sea-ice modes beyond local ice"))

            for key, sym, levels, cmap, get_obs, get_ens, cb_label in specs:
                n_lags = len(LAGS)
                # The gain diagnostic is observational only — there is
                # no ensemble counterpart — so it gets a single row
                # rather than four empty panels.
                n_rows = 1 if key == "R2gain" else 2
                fig, axes = plt.subplots(
                    n_rows, n_lags,
                    figsize=(5.5 * n_lags, 5.5 * n_rows + 0.5),
                    subplot_kw={"projection": proj},
                    squeeze=False,
                )
                fig.subplots_adjust(
                    left=0.04, right=0.88,
                    bottom=0.04, top=0.92,
                    hspace=0.10, wspace=0.06,
                )
                last_cf = None

                rows = [
                    (get_obs, f"OBS\n(ESA-CCI Chl / {obs_siconc_name} siconc)",
                     True),
                    (get_ens, f"Ensemble mean\n({n_models} models)", False),
                ][:n_rows]

                for row, (getter, row_lbl, show_sig) in enumerate(rows):
                    for col, lag in enumerate(LAGS):
                        ax = axes[row][col]
                        # cartopy 0.25 / shapely 2.1 raise GEOS
                        # "UnsupportedOperationException: getX called on
                        # empty Point" while clipping geometry to a
                        # polar-stereographic extent. It is a rendering
                        # failure, not a data problem, and it killed the
                        # Southern Ocean figure after the Arctic one had
                        # already been written. Each decoration is
                        # guarded separately so a failure costs only
                        # that element, not the panel or the figure.
                        try:
                            ax.set_extent(extent, crs=ccrs.PlateCarree())
                        except Exception as _ge:
                            logger.warning(
                                f"    set_extent failed on "
                                f"{region_name} lag{lag:+d}: {_ge}")
                        try:
                            ax.add_feature(cfeature.LAND,
                                           facecolor="lightgray", zorder=4)
                        except Exception as _ge:
                            logger.warning(
                                f"    LAND feature failed on "
                                f"{region_name} lag{lag:+d}: {_ge}")
                        try:
                            ax.coastlines(linewidth=0.4, zorder=5)
                        except Exception:
                            pass
                        try:
                            ax.gridlines(linestyle=":", linewidth=0.3,
                                         alpha=0.4)
                        except Exception as _ge:
                            logger.warning(
                                f"    gridlines failed on "
                                f"{region_name} lag{lag:+d}: {_ge}")

                        data = getter(lag)
                        if data is not None:
                            # cartopy 0.25 contours in lat/lon space and
                            # then projects every resulting path point by
                            # point to set the axes data limits
                            # (geoaxes.contourf -> get_datalim ->
                            # _project_point). Where a contour collapses
                            # against the map boundary GEOS returns an
                            # empty Point and raises
                            # "getX called on empty Point". The Southern
                            # extent (-78 to -43 on SouthPolarStereo)
                            # produces such contours; the Arctic one does
                            # not, which is why only the Southern
                            # regression figures died.
                            #
                            # transform_first=True projects the
                            # COORDINATES first and contours in projected
                            # space, so no path geometry is projected and
                            # the failure cannot occur. It is also
                            # markedly faster. It needs 2-D coordinate
                            # arrays, hence the meshgrid.
                            cf = None
                            try:
                                cf = ax.contourf(
                                    lons, lats, data,
                                    levels=levels, cmap=cmap,
                                    extend="both",
                                    transform=ccrs.PlateCarree())
                            except Exception as _ce:
                                logger.warning(
                                    f"    contourf failed on "
                                    f"{region_name} lag{lag:+d} "
                                    f"({_ce}); retrying with "
                                    f"transform_first=True")
                                try:
                                    _lon2d, _lat2d = np.meshgrid(lons, lats)
                                    cf = ax.contourf(
                                        _lon2d, _lat2d, data,
                                        levels=levels, cmap=cmap,
                                        extend="both",
                                        transform=ccrs.PlateCarree(),
                                        transform_first=True)
                                except Exception as _ce2:
                                    logger.warning(
                                        f"    contourf retry also failed "
                                        f"on {region_name} "
                                        f"lag{lag:+d}: {_ce2}")
                            if cf is not None:
                                last_cf = cf
                            # Overlays guarded separately: losing the
                            # zero contour or the significance wash is
                            # cosmetic, losing the filled field is not.
                            if cf is not None and cmap == "RdBu_r":
                                try:
                                    ax.contour(
                                        lons, lats, data, levels=[0],
                                        colors="k", linewidths=0.6,
                                        transform=ccrs.PlateCarree())
                                except Exception as _ge:
                                    logger.warning(
                                        f"    zero contour failed on "
                                        f"{region_name} lag{lag:+d}: {_ge}")
                            if show_sig and lag in obs_by_lag:
                                pv = obs_by_lag[lag][2]
                                nonsig = np.where(
                                    np.isfinite(pv) & (pv >= 0.05), 1.0, np.nan)
                                if np.any(np.isfinite(nonsig)):
                                    try:
                                        ax.contourf(
                                            lons, lats, nonsig,
                                            levels=[0.5, 1.5],
                                            colors=["white"],
                                            alpha=0.45,
                                            transform=ccrs.PlateCarree())
                                    except Exception as _ge:
                                        logger.warning(
                                            f"    significance wash failed "
                                            f"on {region_name} "
                                            f"lag{lag:+d}: {_ge}")
                        else:
                            ax.text(0.5, 0.5, "no data",
                                    transform=ax.transAxes,
                                    ha="center", va="center",
                                    fontsize=11, color="grey")

                        if row == 0:
                            lag_lbl = ("simultaneous" if lag == 0
                                       else f"siconc leads by {lag} "
                                            f"month{'s' if lag > 1 else ''}")
                            ax.set_title(f"Lag {lag:+d}\n({lag_lbl})",
                                         fontsize=11, fontweight="bold", pad=4)
                        if col == 0:
                            ax.text(-0.06, 0.5, row_lbl,
                                    transform=ax.transAxes,
                                    fontsize=9, fontweight="bold",
                                    ha="right", va="center", rotation=90)
                        letter = chr(ord("a") + row * n_lags + col)
                        ax.text(0.02, 0.97, f"({letter})",
                                transform=ax.transAxes, fontsize=10,
                                fontweight="bold", va="top",
                                bbox=dict(facecolor="white", alpha=0.75,
                                          edgecolor="none"))

                if last_cf is not None:
                    cax = fig.add_axes([0.90, 0.12, 0.015, 0.74])
                    cb  = fig.colorbar(last_cf, cax=cax, extend="both")
                    cb.set_label(f"{sym} — {cb_label}", fontsize=15)
                    cb.ax.tick_params(labelsize=14)

                fname = f"figS_seaice_chl_{key}{out_tag}_{region_name}.png"
                # cartopy renders lazily: geometry clipping for a polar
                # stereographic axis happens at DRAW time, inside
                # savefig, not when contourf/gridlines are called. A
                # GEOS "getX called on empty Point" therefore escapes
                # the per-command guards above and surfaces here.
                # bbox_inches="tight" compounds it by forcing an extra
                # renderer pass to measure the artists, so retry
                # without it before giving up on the figure.
                try:
                    fig.savefig(OUT_DIR / fname, dpi=150,
                                bbox_inches="tight")
                    logger.success(f"  wrote {fname}")
                except Exception as _se:
                    logger.warning(
                        f"  {fname}: tight-bbox save failed ({_se}); "
                        f"retrying without bbox_inches")
                    try:
                        fig.savefig(OUT_DIR / fname, dpi=150)
                        logger.success(
                            f"  wrote {fname} (untrimmed margins)")
                    except Exception as _se2:
                        logger.warning(
                            f"  {fname}: save FAILED: {_se2}")
                plt.close(fig)



        except Exception as _e:
            # Keep the regions independent. Previously a
            # failure in one region propagated out of the
            # whole function, so a Southern Ocean error
            # discarded the Arctic figures that had already
            # been written and reported only a single
            # top-level warning with no location.
            import traceback as _tb
            logger.warning(
                f"  {region_name} ({obs_siconc_name}) regression FAILED: {_e}")
            logger.warning(
                _tb.format_exc().replace("<", "\\<"))
            print(f"[regression {region_name}] "
                  f"{_tb.format_exc()}", flush=True)
            continue
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
        cb.set_label("Pearson r", fontsize=15)
        cb.ax.tick_params(labelsize=14)
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
    _write_lag_peak_table(raw_curves, "MLD", "")
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
    _write_lag_peak_table(
        removed_curves, "MLD", f"_minus_{'_'.join(all_names)}")
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
            cb_r.set_label("Peak Pearson r", fontsize=15)
            cb_r.ax.tick_params(labelsize=14)

        if im_lag is not None:
            cax_l = fig.add_axes([0.89, 0.08, 0.012, 0.38])
            cb_l  = fig.colorbar(im_lag, cax=cax_l, extend="neither")
            cb_l.set_label("Lag (months, + = driver leads)", fontsize=9)
            cb_l.set_ticks(np.arange(-LAG_MAX, LAG_MAX + 1, 3))
            cb_l.ax.tick_params(labelsize=14)

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
                fontsize=15,
            )
            cb1.ax.tick_params(labelsize=14)
        if last_cf_diff is not None:
            cax2 = fig.add_axes([0.91, 0.06, 0.013, 0.24])
            cb2  = fig.colorbar(last_cf_diff, cax=cax2, extend="both")
            cb2.set_label(r"$\Delta$r (raw 2212 partial)", fontsize=8)
            cb2.ax.tick_params(labelsize=14)

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
    logger.info(f"=== STAGES ENABLED: {sorted(RUN_STAGES)} ===")
    if RUN_STAGES != set(_STAGE_ALL):
        logger.warning(
            "  PARTIAL RUN — figures for disabled stages will not be\n"
            "  written or updated. Existing PNGs from earlier runs are\n"
            "  left untouched on disk and may be stale.")
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
                if not NEED_SEASONAL:
                    raise RuntimeError("seasonal skipped (CHL_STAGES)")
                months = da.groupby("time.month").mean("time")
                regridded = []
                for m in months.month.values:
                    # NB: never name this `re` — this loop runs at module
                    # scope inside __main__, so it rebinds the stdlib `re`
                    # module to a DataArray and every later re.search()
                    # fails with "'DataArray' object has no attribute
                    # 'search'". That silently killed the ESA-CCI loader,
                    # which parses year stamps out of filenames.
                    re_m = regrid_to_obs(months.sel(month=m), obs_clim)
                    regridded.append(re_m.expand_dims(month=[m]))
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

    # ---- Load rsds (light) for the light-bias and polar-light figures ----
    # DISABLED (v7): the dfe / rsds limitation figure (fig_limitation.png)
    # was dropped — the inter-model spread hexbins carried no usable signal
    # (rho = -0.02 for rsds) and three of seven models have no dfe in the
    # CMIP6 archive on Levante, leaving empty panels. dfe is therefore no
    # longer loaded at all, and only the FULL monthly rsds series is kept,
    # since make_light_bias_map and make_fig_light_polar_coupling need it.
    model_rsds_ts: dict[str, xr.DataArray] = {}    # full monthly series
    for model in (MODELS if NEED_RSDS else []):
        try:
            r = load_model_rsds(model)
            if r is not None:
                model_rsds_ts[model] = r
        except Exception as e:
            logger.warning(f"  [{model}] rsds load failed: {e}")
    obs_rsds_ts = None
    try:
        obs_rsds_ts = load_obs_rsds() if NEED_RSDS else None
    except Exception as e:
        logger.warning(f"  {RSDS_OBS_NAME} rsds load failed: {e}")
    logger.info(
        f"  light data loaded: rsds for {len(model_rsds_ts)} models, "
        f"{RSDS_OBS_NAME} {'present' if obs_rsds_ts is not None else 'absent'}"
    )

    # ---- Standalone light (rsds) bias map vs the surface-SW obs ----
    # STAGE GATE: light
    if stage("light") and model_rsds_ts and obs_rsds_ts is not None:
        try:
            make_light_bias_map(obs_rsds_ts, model_rsds_ts)
        except Exception as e:
            import traceback
            logger.warning(f"  light bias map crashed: {e}")
            logger.warning(traceback.format_exc().replace("<", "\\<"))

    # STAGE GATE: A  (mean state, seasonality, anomaly amplitude)
    if stage("A"):
        make_fig1_and_fig2_combined(obs_clim, model_clim)
        if model_seasonal:
            make_fig3_hovmoller(
                obs_full=obs_full,
                model_seasonal=model_seasonal,
                obs_clim=obs_clim,
                model_chl_full=model_chl_ts,
            )
            make_fig4(obs_full, model_seasonal)
        else:
            logger.warning(
                "  Figs 3-4 SKIPPED: no model seasonal climatology")
    else:
        logger.info("  Phase A figures skipped (CHL_STAGES)")
    # DISABLED (v7): make_fig_limitation(...) — fig_limitation.png removed.
    # The function definition is kept further up in the file (unused) in case
    # the analysis is revived; nothing else calls it.

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
        obs_mld    = load_obs_mld() if NEED_MLD else None
        if obs_mld is None and not NEED_MLD:
            raise RuntimeError("MLD skipped (CHL_STAGES)")
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
        if not NEED_SST:
            raise RuntimeError("SST skipped (CHL_STAGES)")
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
        if not NEED_CLT:
            raise RuntimeError("cloud cover skipped (CHL_STAGES)")
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

        # DISABLED (v7): make_figS_bloom_mld_timing(...) —
        # figS_bloom_mld_timing.png removed. The year-to-year timing scatter
        # was not informative: no OBS regression reached significance in any
        # basin (p = 0.23-0.98), and the discrete month index produced
        # degenerate fits where the MLD-minimum month is nearly constant
        # (Arctic OBS slope ~ -1e13 from an x-range collapsed onto a single
        # value). The _bloom_mld_timing helper is kept above but is now
        # unused; nothing else calls it.

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

        # Second, independent obs sea-ice product (ESA-CCI). Regridded
        # to common_grid inside the loader because its native EASE2
        # projection needs the sphere-aware nearest-neighbour path, and
        # because doing it once here avoids repeating it for every lag
        # and region inside the correlation figure. Used by the
        # pointwise lag-correlation maps only; Figs 11/11b/11c/12 keep
        # HadISST as their single reference for now.
        obs2_siconc_ts = None
        try:
            obs2_siconc_ts = load_obs_siconc_esacci(common_grid)
        except Exception as e:
            import traceback
            logger.warning(f"  {SICONC_OBS2_NAME} load failed: {e}")
            logger.warning(traceback.format_exc().replace("<", "\\<"))

        # Load model siconc for each available model
        model_siconc_ts: dict[str, xr.DataArray] = {}
        for model in MODELS:
            sic = load_model_siconc(model)
            if sic is not None:
                model_siconc_ts[model] = sic

        logger.info(f"  Phase E inputs: obs_siconc="
                    f"{'yes' if obs_siconc_ts is not None else 'NO'}  "
                    f"obs2_siconc="
                    f"{'yes' if obs2_siconc_ts is not None else 'NO'}  "
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
        # STAGE GATE: memory  (Figs 11, 11b, 11c, 12, S1)
        if stage("memory") and model_chl_ts and (
                obs_siconc_ts is not None or model_siconc_ts):
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
    if common_grid is not None and stage("peaklag"):
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

    # STAGE GATE: ptcorr
    # Spatially filtered, basin-resolved lag-0 correlation maps, the
    # per-model version, the seasonal Hovmoller, and the area-mean lag
    # curves - each in a raw form and with AMO/Nino3.4/PDO removed.
    # These replace the former global pointwise maps and the basin-mean
    # lag figure; see the notes on those functions for why.
    if stage("ptcorr") and common_grid is not None and model_chl_ts:
        # One variant with nothing removed, then one per mode SEPARATELY.
        # Removing all three together confounds their contributions: they
        # act on different basins and different time scales, so a change
        # in the combined figure cannot be attributed to any of them.
        _VARIANTS = ((), ("AMO",), ("NINO34",), ("PDO",))
        for _drv, _lbl, _obs, _mod in (
            ("SST", "NOAA-ERSSTv3b SST", obs_sst_ts, model_sst_ts),
            ("MLD", "ORAS5 MLD",         obs_mld_ts, model_mld_ts),
        ):
            for _rm in _VARIANTS:
                _obs_pack = None
                _model_packs: dict = {}
                try:
                    _model_packs = make_figS_basin_corr_obs_ens(
                        obs_chl_ts=obs_full,
                        obs_drv_ts=_obs,
                        model_chl_ts=model_chl_ts,
                        model_drv_ts=_mod,
                        common_grid=common_grid,
                        driver=_drv,
                        driver_label=_lbl,
                        remove=_rm,
                        obs_sst_for_idx=obs_sst_ts,
                        model_sst_for_idx=model_sst_ts,
                    ) or {}
                    if _obs is not None:
                        try:
                            _oi = (_indices_for_source(
                                obs_sst_ts, common_grid, LAG_PERIOD,
                                "idx::OBS") if _rm else {})
                            _cz = _filtered_z_modes(
                                obs_full, common_grid, LAG_PERIOD, "chl::OBS",
                                SPATIAL_FILTER_DEG, interp_gaps=True,
                                remove=_rm, indices=_oi)
                            _dz = _filtered_z_modes(
                                _obs, common_grid, LAG_PERIOD,
                                f"{_drv.lower()}::OBS", SPATIAL_FILTER_DEG,
                                remove=_rm, indices=_oi)
                            _r, _p = _corr_map_lag0(_dz, _cz)
                            _obs_pack = (_r, _fdr_mask(_p))
                        except Exception:
                            _obs_pack = None
                except Exception as e:
                    import traceback
                    logger.warning(
                        f"  Fig S_basincorr ({_drv} {_rm}) crashed: {e}")
                    logger.warning(traceback.format_exc().replace("<", "\\<"))

                try:
                    make_figS_basin_corr_per_model(
                        model_packs=_model_packs,
                        common_grid=common_grid,
                        driver=_drv,
                        obs_pack=_obs_pack,
                        remove=_rm,
                    )
                except Exception as e:
                    import traceback
                    logger.warning(
                        f"  Fig S_basincorr_models ({_drv} {_rm}) "
                        f"crashed: {e}")
                    logger.warning(traceback.format_exc().replace("<", "\\<"))

                try:
                    make_fig_lag_areamean(
                        obs_chl_ts=obs_full,
                        obs_drv_ts=_obs,
                        model_chl_ts=model_chl_ts,
                        model_drv_ts=_mod,
                        common_grid=common_grid,
                        driver=_drv,
                        driver_label=_lbl,
                        remove=_rm,
                        obs_sst_for_idx=obs_sst_ts,
                        model_sst_for_idx=model_sst_ts,
                    )
                except Exception as e:
                    import traceback
                    logger.warning(
                        f"  Fig lag area-mean ({_drv} {_rm}) crashed: {e}")
                    logger.warning(traceback.format_exc().replace("<", "\\<"))

            # Seasonal Hovmoller: raw only. The seasonal breakdown already
            # separates the regimes that the mode removal is meant to
            # disentangle, and 17 years give only ~17 samples per calendar
            # month, too few to also absorb three regression predictors.
            try:
                make_figS_seasonal_hovmoller(
                    obs_chl_ts=obs_full,
                    obs_drv_ts=_obs,
                    model_chl_ts=model_chl_ts,
                    model_drv_ts=_mod,
                    common_grid=common_grid,
                    driver=_drv,
                    driver_label=_lbl,
                )
            except Exception as e:
                import traceback
                logger.warning(
                    f"  Fig S_seasonal_corr ({_drv}) crashed: {e}")
                logger.warning(traceback.format_exc().replace("<", "\\<"))

    # STAGE GATE: seaice
    if stage("seaice"):
        # ---- Pointwise siconc-Chl-a spatial correlation maps ----
        # Run once per observational ice product, writing separate figures:
        #   figS_seaice_chl_corr_{Arctic,Southern}.png         HadISST
        #   figS_seaice_chl_corr_ESACCI_{Arctic,Southern}.png  ESA-CCI
        # The model row is identical in both (models always use their own
        # siconc); only the OBS row changes, which is what makes the pair
        # a clean test of observational sensitivity.
        if common_grid is not None and model_chl_ts:
            obs_r_hadisst: dict[str, dict[int, np.ndarray]] = {}
            obs_r_esacci:  dict[str, dict[int, np.ndarray]] = {}
            try:
                obs_r_hadisst = make_figS_seaice_chl_spatial_corr(
                    obs_chl_ts=obs_full,
                    obs_siconc_ts=obs_siconc_ts,
                    model_chl_ts=model_chl_ts,
                    model_siconc_ts=model_siconc_ts,
                    common_grid=common_grid,
                    obs_siconc_name=SICONC_OBS_NAME,
                    out_tag="",
                )
            except Exception as e:
                import traceback
                logger.warning(
                    f"  Fig S_seaice_chl_corr ({SICONC_OBS_NAME}) "
                    f"crashed: {e}")
                logger.warning(
                    traceback.format_exc().replace("<", "\\<"))
                print(traceback.format_exc(), flush=True)

            # Seasonal stratification of the same diagnostic. Run for the
            # primary ice product only: the cold-season half is limited by
            # ocean-colour coverage rather than by the ice product, so
            # repeating it for ESA-CCI ice would not test anything new.
            try:
                make_figS_seaice_chl_seasonal_corr(
                    obs_chl_ts=obs_full,
                    obs_siconc_ts=obs_siconc_ts,
                    model_chl_ts=model_chl_ts,
                    model_siconc_ts=model_siconc_ts,
                    common_grid=common_grid,
                    obs_siconc_name=SICONC_OBS_NAME,
                    out_tag="",
                )
            except Exception as e:
                import traceback
                logger.warning(
                    f"  Fig S_seaice_chl_seasonal ({SICONC_OBS_NAME}) "
                    f"crashed: {e}")
                logger.warning(
                    traceback.format_exc().replace("<", "\\<"))

            # Companion lag curve: the same seasonal stratification with
            # the lag extended to a full year, which is what lets the
            # winter ice question be asked at all.
            try:
                make_fig_seaice_chl_lagcurve_seasonal(
                    obs_chl_ts=obs_full,
                    obs_siconc_ts=obs_siconc_ts,
                    model_chl_ts=model_chl_ts,
                    model_siconc_ts=model_siconc_ts,
                    common_grid=common_grid,
                    obs_siconc_name=SICONC_OBS_NAME,
                    out_tag="",
                )
            except Exception as e:
                import traceback
                logger.warning(
                    f"  Fig seaice lag curve ({SICONC_OBS_NAME}) "
                    f"crashed: {e}")
                logger.warning(
                    traceback.format_exc().replace("<", "\\<"))

            if obs2_siconc_ts is not None:
                try:
                    obs_r_esacci = make_figS_seaice_chl_spatial_corr(
                        obs_chl_ts=obs_full,
                        obs_siconc_ts=obs2_siconc_ts,
                        model_chl_ts=model_chl_ts,
                        model_siconc_ts=model_siconc_ts,
                        common_grid=common_grid,
                        obs_siconc_name=SICONC_OBS2_NAME,
                        out_tag="_ESACCI",
                    )
                except Exception as e:
                    import traceback
                    logger.warning(
                        f"  Fig S_seaice_chl_corr ({SICONC_OBS2_NAME}) "
                        f"crashed: {e}")
                    logger.warning(
                        traceback.format_exc().replace("<", "\\<"))
                    print(traceback.format_exc(), flush=True)
            else:
                logger.warning(
                    f"  {SICONC_OBS2_NAME} figure SKIPPED: product not loaded")

            # ---- Agreement between the two observational ice products ----
            # Spatial correlation of the two OBS r-maps, cell by cell, over
            # the region where both are defined. This is the number that
            # decides how the pair can be written up: high agreement means
            # the observed pattern is a robust target and any model-obs gap
            # is model error; low agreement means observational uncertainty
            # in the ice field is a competing explanation.
            if obs_r_hadisst and obs_r_esacci:
                logger.info(
                    f"  === {SICONC_OBS_NAME} vs {SICONC_OBS2_NAME} "
                    f"agreement ===")
                for region_name in obs_r_hadisst:
                    a_by_lag = obs_r_hadisst.get(region_name, {})
                    b_by_lag = obs_r_esacci.get(region_name, {})
                    for lag in sorted(set(a_by_lag) & set(b_by_lag)):
                        a, b = a_by_lag[lag], b_by_lag[lag]
                        if a.shape != b.shape:
                            continue
                        ok = np.isfinite(a) & np.isfinite(b)
                        if ok.sum() < 20:
                            continue
                        r_oo = float(np.corrcoef(a[ok], b[ok])[0, 1])
                        logger.info(
                            f"  {region_name} lag{lag:+d}: pattern "
                            f"r={r_oo:+.3f} over {int(ok.sum())} cells  "
                            f"(mean r {np.nanmean(a[ok]):+.3f} vs "
                            f"{np.nanmean(b[ok]):+.3f}; "
                            f"coverage {np.isfinite(a).sum()} vs "
                            f"{np.isfinite(b).sum()} cells)")
        else:
            logger.warning("  Fig S_seaice_chl_corr SKIPPED")

    # STAGE GATE: bloom
    if stage("bloom"):
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

    # STAGE GATE: eof
    if stage("eof"):
        # ---- EOF analyses and explained variance maps ----
        # Fig S_seaice_eof: sea-ice EOF1 pattern, PC1, variance spectrum
        if common_grid is not None:
            try:
                make_figS_seaice_eof(
                    obs_siconc_ts=obs_siconc_ts,
                    model_siconc_ts=model_siconc_ts,
                    common_grid=common_grid,
                    obs_siconc_name=SICONC_OBS_NAME,
                    out_tag="",
                )
            except Exception as e:
                import traceback
                logger.warning(
                    f"  Fig S_seaice_eof ({SICONC_OBS_NAME}) crashed: {e}")
                logger.warning(traceback.format_exc().replace("<", "\\<"))

            if obs2_siconc_ts is not None:
                try:
                    make_figS_seaice_eof(
                        obs_siconc_ts=obs2_siconc_ts,
                        model_siconc_ts=model_siconc_ts,
                        common_grid=common_grid,
                        obs_siconc_name=SICONC_OBS2_NAME,
                        out_tag="_ESACCI",
                    )
                except Exception as e:
                    import traceback
                    logger.warning(
                        f"  Fig S_seaice_eof ({SICONC_OBS2_NAME}) "
                        f"crashed: {e}")
                    logger.warning(
                        traceback.format_exc().replace("<", "\\<"))
            else:
                logger.warning(
                    f"  Fig S_seaice_eof for {SICONC_OBS2_NAME} SKIPPED: "
                    f"product not loaded")

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


    # STAGE GATE: seaice (regression maps)
    if stage("seaice"):
        # Fig S_seaice_chl_regression: Chl-a variance explained by sea-ice PC1
        if common_grid is not None and model_chl_ts:
            try:
                make_figS_seaice_chl_regression(
                    obs_chl_ts=obs_full,
                    obs_siconc_ts=obs_siconc_ts,
                    model_chl_ts=model_chl_ts,
                    model_siconc_ts=model_siconc_ts,
                    common_grid=common_grid,
                    obs_siconc_name=SICONC_OBS_NAME,
                    out_tag="",
                )
            except Exception as e:
                import traceback
                logger.warning(
                    f"  Fig S_seaice_chl_regression ({SICONC_OBS_NAME}) "
                    f"crashed: {e}")
                logger.warning(traceback.format_exc().replace("<", "\\<"))

            if obs2_siconc_ts is not None:
                try:
                    make_figS_seaice_chl_regression(
                        obs_chl_ts=obs_full,
                        obs_siconc_ts=obs2_siconc_ts,
                        model_chl_ts=model_chl_ts,
                        model_siconc_ts=model_siconc_ts,
                        common_grid=common_grid,
                        obs_siconc_name=SICONC_OBS2_NAME,
                        out_tag="_ESACCI",
                    )
                except Exception as e:
                    import traceback
                    logger.warning(
                        f"  Fig S_seaice_chl_regression ({SICONC_OBS2_NAME}) "
                        f"crashed: {e}")
                    logger.warning(
                        traceback.format_exc().replace("<", "\\<"))
            else:
                logger.warning(
                    f"  Regression figures for {SICONC_OBS2_NAME} SKIPPED: "
                    f"product not loaded")


    # STAGE GATE: regional
    if stage("regional"):
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
