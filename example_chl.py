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
"""

from __future__ import annotations

import sys
import warnings
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
    "GFDL-ESM4",
    "UKESM1-0-LL",
    "MPI-ESM-1-2-HAM",
    "CESM2",
]

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
    "Southern": (-90, -45,   0, 360),
    "Arctic":   ( 65,  90,   0, 360),
}

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
}


# ============================================================
# Unit handling -- convert Chl to mg m-3
# ============================================================

def to_mg_m3(da: xr.DataArray, units_str: str | None) -> xr.DataArray:
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
    """Snap all time coordinates to the first day of each month as datetime64."""
    import pandas as pd
    if "time" not in da.coords:
        return da
    times = da.time.values
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
        raise FileNotFoundError(f"No chl file for {model}")
    ds = xr.open_dataset(candidates[0], mask_and_scale=True)
    da = ds["chl"]
    units_str = da.attrs.get("units")
    for zname in ("lev", "olevel", "depth"):
        if zname in da.dims:
            da = da.isel({zname: 0}, drop=True)
        if zname in da.coords:
            da = da.drop_vars(zname, errors="ignore")
        for bnd in (f"{zname}_bnds", f"{zname}_bounds"):
            if bnd in da.coords:
                da = da.drop_vars(bnd, errors="ignore")
    da = da.where(np.abs(da) < 1.0)
    da = da.squeeze()
    da = standardize(da)
    return to_mg_m3(da, units_str)


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
# Panel labelling helper
# ============================================================

def _panel_tag(ax, index: int, *, projection: bool = False) -> None:
    """Add a lowercase (a), (b), ... tag to the top-left of a panel.

    index is 0-based. For cartopy GeoAxes pass projection=True so the
    tag is placed in axes-fraction coordinates rather than data coords.
    """
    label = f"({chr(ord('a') + index)})"
    ax.text(
        0.015, 0.985, label,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor="none", alpha=0.85),
        zorder=20,
    )


# ============================================================
# Fig 1: ESA-CCI Chl climatology
# ============================================================

def make_fig1(obs_clim: xr.DataArray) -> None:
    logger.info("Building Fig 1 (obs climatology)...")
    fig = plt.figure(figsize=(10, 5.5))
    ax  = plt.axes(projection=ccrs.PlateCarree())
    im = ax.pcolormesh(
        obs_clim.lon, obs_clim.lat, obs_clim.values,
        norm=LogNorm(vmin=LOG_VMIN, vmax=LOG_VMAX),
        cmap="YlGn", transform=ccrs.PlateCarree(), shading="auto",
    )
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.coastlines(linewidth=0.4)
    ax.gridlines(linestyle=":", linewidth=0.3, draw_labels=True,
                 alpha=0.5, x_inline=False, y_inline=False)
    cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                        fraction=0.05, pad=0.08, extend="both")
    cbar.set_label(CHL_LABEL)
    fig.savefig(OUT_DIR / "fig1_obs_climatology.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote fig1_obs_climatology.png")


# ============================================================
# Fig 2: Chl bias maps 2x2
# ============================================================

def make_fig2(model_clim: dict[str, xr.DataArray], obs_clim: xr.DataArray) -> None:
    logger.info("Building Fig 2 (Chl bias maps, 2x2)...")
    n     = len(model_clim)
    ncols = 2
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10, 8 * nrows // 2),
        subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False,
    )
    axes_flat = axes.flatten()
    norm = TwoSlopeNorm(vmin=-BIAS_LIMIT, vcenter=0, vmax=BIAS_LIMIT)
    last_im = None
    for idx, (ax, (model, m_re)) in enumerate(zip(axes_flat, model_clim.items())):
        bias = m_re - obs_clim
        rmse, r = rmse_and_r(m_re, obs_clim)
        im = ax.pcolormesh(
            obs_clim.lon, obs_clim.lat, bias.values,
            norm=norm, cmap="RdBu_r",
            transform=ccrs.PlateCarree(), shading="auto",
        )
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
        ax.coastlines(linewidth=0.4)
        ax.gridlines(linestyle=":", linewidth=0.3, draw_labels=True,
                     alpha=0.4, x_inline=False, y_inline=False,
                     xlabel_style={"size": 7}, ylabel_style={"size": 7})
        ax.set_title(f"Bias: {model} − CCI", fontsize=10)
        ax.text(0.02, 0.04, f"RMSE={rmse:.3f}  r={r:.3f}",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="black", alpha=0.9))
        _panel_tag(ax, idx)
        last_im = im
    for ax in axes_flat[n:]:
        ax.axis("off")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, orientation="horizontal",
                            fraction=0.04, pad=0.07, extend="both")
        cbar.set_label(f"Bias ({CHL_LABEL})")
    fig.savefig(OUT_DIR / "fig2_bias_maps.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote fig2_bias_maps.png")


# ============================================================
# Fig 3: basin-mean bias bars
# ============================================================

def make_fig3(model_clim: dict[str, xr.DataArray], obs_clim: xr.DataArray) -> None:
    logger.info("Building Fig 3 (basin bias bars)...")
    bias_table: dict[str, dict[str, float]] = {b: {} for b in BASINS}
    for model, m_re in model_clim.items():
        for basin, box in BASINS.items():
            o = basin_mean(obs_clim, *box)
            m = basin_mean(m_re, *box)
            bias_table[basin][model] = m - o

    basins = list(BASINS.keys())
    models = list(model_clim.keys())
    n_m = len(models)
    width = 0.8 / max(n_m, 1)
    x = np.arange(len(basins))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for i, model in enumerate(models):
        color = PALETTE[i % len(PALETTE)]
        vals  = [bias_table[b].get(model, np.nan) for b in basins]
        bars  = ax1.bar(x + i * width - 0.4 + width / 2, vals, width,
                        label=model, color=color, edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, vals):
            if not np.isfinite(v):
                continue
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     v + (0.01 if v >= 0 else -0.01), f"{v:.3f}",
                     ha="center", va="bottom" if v >= 0 else "top",
                     fontsize=7, color=color, fontweight="bold")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.6)
    ax1.set_xticks(x); ax1.set_xticklabels(basins)
    ax1.set_ylabel(r"Bias [mg m$^{-3}$]")
    ax1.set_title("Signed bias")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="best", fontsize=8, frameon=True)
    _panel_tag(ax1, 0)

    for i, model in enumerate(models):
        color = PALETTE[i % len(PALETTE)]
        vals  = [abs(bias_table[b].get(model, np.nan)) for b in basins]
        bars  = ax2.bar(x + i * width - 0.4 + width / 2, vals, width,
                        label=model, color=color, edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, vals):
            if not np.isfinite(v):
                continue
            ax2.text(bar.get_x() + bar.get_width() / 2, v * 1.05,
                     f"{v:.3f}", ha="center", va="bottom",
                     fontsize=7, color=color, fontweight="bold")
    ax2.set_yscale("log")
    ax2.set_xticks(x); ax2.set_xticklabels(basins)
    ax2.set_ylabel(r"|Bias| [mg m$^{-3}$]")
    ax2.set_title("Absolute bias (log)")
    ax2.grid(axis="y", alpha=0.3, which="both")
    ax2.legend(loc="best", fontsize=8, frameon=True)
    _panel_tag(ax2, 1)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_basin_bias.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote fig3_basin_bias.png")


# ============================================================
# Fig 4: seasonal cycle
# ============================================================

def make_fig4(obs_da: xr.DataArray,
              model_seasonal: dict[str, xr.DataArray]) -> None:
    logger.info("Building Fig 4 (seasonal cycle)...")
    fig, ax = plt.subplots(figsize=(8, 5))
    obs_sc  = obs_da.groupby("time.month").mean("time")
    weights = np.cos(np.deg2rad(obs_sc.lat))
    obs_global = obs_sc.weighted(weights).mean(dim=("lat", "lon"))
    ax.plot(obs_global.month, obs_global.values, color="black",
            linewidth=2.0, marker="o", label="ESA CCI-OC", zorder=10)
    for i, (model, sc) in enumerate(model_seasonal.items()):
        w  = np.cos(np.deg2rad(sc.lat))
        gm = sc.weighted(w).mean(dim=("lat", "lon"))
        ax.plot(gm.month, gm.values, color=PALETTE[i % len(PALETTE)],
                linewidth=1.4, marker="o", markersize=4, alpha=0.85, label=model)
    ax.set_xlabel("Month"); ax.set_ylabel(CHL_LABEL)
    ax.set_xticks(range(1, 13)); ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, frameon=True)
    fig.savefig(OUT_DIR / "fig4_seasonal_cycle.png", dpi=180, bbox_inches="tight")
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
    """Generic 2x2 bias map figure."""
    logger.info(f"Building Fig {fig_num} ({title})...")
    n = len(bias_dict)
    if n == 0:
        logger.warning(f"  no data; skipping Fig {fig_num}")
        return
    ncols = 2
    nrows = max(1, int(np.ceil(n / ncols)))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(10, 8 * nrows // 2),
        subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False,
    )
    axes_flat = axes.flatten()
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    last_im = None

    for idx, (ax, (model, bias)) in enumerate(zip(axes_flat, bias_dict.items())):
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
        _panel_tag(ax, idx)
        last_im = im

    for ax in axes_flat[n:]:
        ax.axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, orientation="horizontal",
                            fraction=0.04, pad=0.07, extend="both")
        cbar.set_label(f"Bias [{unit_label}]")

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

    for idx, (ax, (model, hov)) in enumerate(zip(axes_flat, hov_maps.items())):
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
        _panel_tag(ax, idx)
        last_im = im

    for ax in axes_flat[n:]:
        ax.axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes,
                            orientation="horizontal",
                            fraction=0.04, pad=0.12)
        cbar.set_label("Pearson r  [MLD anomaly vs Chl-a bias]", fontsize=11)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig8_hovmoller_mld_chl.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote fig8_hovmoller_mld_chl.png")


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


def _deseasonalise_zscore(ts: xr.DataArray) -> xr.DataArray:
    """Remove monthly climatology then standardise to unit variance."""
    ts = snap_to_month_start(ts)
    clim = ts.groupby("time.month").mean("time")
    anom = ts.groupby("time.month") - clim
    s = float(anom.std("time", skipna=True))
    if not np.isfinite(s) or s == 0:
        return anom * np.nan
    return anom / s


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


def _make_lag_figure(
    lag_curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    obs_label: str,
    driver_name: str,
    title: str,
    filename: str,
    fig_num: str,
) -> None:
    """Plot lag-correlation curves as 6 subplots (one per basin).

    lag_curves: {basin: {source_name: (lags, r_values)}}
                where source_name is "OBS" or a model name.
    """
    logger.info(f"Building Fig {fig_num} ({title})...")
    basins = list(BASINS.keys())
    n_b = len(basins)
    ncols = 3
    nrows = int(np.ceil(n_b / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.0 * nrows),
                              sharex=True, sharey=True, squeeze=False)
    axes_flat = axes.flatten()

    # Significance band: ±2/sqrt(N) for the deseasonalised series
    # N ≈ 17 years × 12 months = 204
    sig_band = 2.0 / np.sqrt(204)

    handles_for_legend = None
    labels_for_legend  = None

    for idx, basin in enumerate(basins):
        ax = axes_flat[idx]
        curves = lag_curves.get(basin, {})

        # Significance band as shaded grey
        ax.axhspan(-sig_band, sig_band, color="gray", alpha=0.12,
                   lw=0, zorder=0)
        ax.axhline(0, color="k", lw=0.5, alpha=0.6)
        ax.axvline(0, color="k", lw=0.5, alpha=0.6, ls=":")

        # Plot OBS first (black, thick), then models
        peak_info_lines = []
        for source, (lags, rs) in curves.items():
            if source == "OBS":
                ls    = "-"
                lw    = 2.4
                color = "black"
                zord  = 10
            else:
                ls    = "-"
                lw    = 1.4
                models_list = list(curves.keys())
                # Use the model's stable index for colour consistency
                color = PALETTE[
                    [s for s in models_list if s != "OBS"].index(source)
                    % len(PALETTE)
                ]
                zord  = 5
            ax.plot(lags, rs, ls=ls, lw=lw, color=color,
                    label=source, alpha=0.9, zorder=zord)

            # Record peak |r| for the annotation
            if np.any(np.isfinite(rs)):
                k_peak = int(lags[np.nanargmax(np.abs(rs))])
                r_peak = float(rs[np.nanargmax(np.abs(rs))])
                peak_info_lines.append(
                    f"{source}: r={r_peak:+.2f} @ lag={k_peak:+d}"
                )

        ax.set_title(basin, fontsize=11, fontweight="bold")
        ax.set_xlim(-LAG_MAX, LAG_MAX)
        ax.set_ylim(-1, 1)
        ax.set_xticks(np.arange(-LAG_MAX, LAG_MAX + 1, 4))
        ax.grid(alpha=0.25)
        _panel_tag(ax, idx)

        if idx % ncols == 0:
            ax.set_ylabel("Pearson r", fontsize=10)
        if idx // ncols == nrows - 1:
            ax.set_xlabel(f"Lag (months)  —  {driver_name} leads → ",
                          fontsize=9)

        # Small peak-r annotation box
        if peak_info_lines:
            ax.text(0.02, 0.02, "\n".join(peak_info_lines),
                    transform=ax.transAxes, fontsize=6.5, va="bottom",
                    family="monospace",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor="gray",
                              alpha=0.85))

        if idx == 0:
            handles_for_legend, labels_for_legend = ax.get_legend_handles_labels()

    # Hide any unused panels
    for idx in range(n_b, nrows * ncols):
        axes_flat[idx].axis("off")

    # Figure-level legend in the unused panel area or below
    if handles_for_legend is not None:
        fig.legend(handles_for_legend, labels_for_legend,
                   loc="lower center", ncol=min(len(handles_for_legend), 5),
                   fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
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
                lags, rs = _lag_correlation(chl_z, sst_z)
                lag_curves[basin]["OBS"] = (lags, rs)
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
                lags, rs = _lag_correlation(chl_z, sst_z)
                lag_curves[basin][model] = (lags, rs)
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
                lags, rs = _lag_correlation(chl_z, mld_z)
                lag_curves[basin]["OBS"] = (lags, rs)
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
                lags, rs = _lag_correlation(chl_z, mld_z)
                lag_curves[basin][model] = (lags, rs)
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

SICONC_BASINS = ("Arctic", "Southern")


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
                lags, rs = _lag_correlation(chl_z, sic_z)
                lag_curves[basin]["OBS"] = (lags, rs)
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
                lags, rs = _lag_correlation(chl_z, sic_z)
                lag_curves[basin][model] = (lags, rs)
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

    sig_band = 2.0 / np.sqrt(204)

    handles_for_legend = None
    labels_for_legend  = None

    for idx, basin in enumerate(basins):
        ax = axes_flat[idx]
        curves = lag_curves.get(basin, {})

        ax.axhspan(-sig_band, sig_band, color="gray", alpha=0.12,
                   lw=0, zorder=0)
        ax.axhline(0, color="k", lw=0.5, alpha=0.6)
        ax.axvline(0, color="k", lw=0.5, alpha=0.6, ls=":")

        peak_info_lines = []
        for source, (lags, rs) in curves.items():
            if source == "OBS":
                color, lw, zord = "black", 2.4, 10
            else:
                models_in = [s for s in curves.keys() if s != "OBS"]
                color = PALETTE[models_in.index(source) % len(PALETTE)] \
                        if source in models_in else "gray"
                lw, zord = 1.4, 5
            ax.plot(lags, rs, lw=lw, color=color, label=source,
                    alpha=0.9, zorder=zord)
            if np.any(np.isfinite(rs)):
                k_peak = int(lags[np.nanargmax(np.abs(rs))])
                r_peak = float(rs[np.nanargmax(np.abs(rs))])
                peak_info_lines.append(
                    f"{source}: r={r_peak:+.2f} @ lag={k_peak:+d}")

        ax.set_title(basin, fontsize=11, fontweight="bold")
        ax.set_xlim(-LAG_MAX, LAG_MAX)
        ax.set_ylim(-1, 1)
        ax.set_xticks(np.arange(-LAG_MAX, LAG_MAX + 1, 4))
        ax.grid(alpha=0.25)
        ax.set_xlabel("Lag (months)  —  siconc leads → ", fontsize=9)
        if idx == 0:
            ax.set_ylabel("Pearson r", fontsize=10)
        _panel_tag(ax, idx)

        if peak_info_lines:
            ax.text(0.02, 0.02, "\n".join(peak_info_lines),
                    transform=ax.transAxes, fontsize=6.5, va="bottom",
                    family="monospace",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="white", edgecolor="gray",
                              alpha=0.85))

        if idx == 0:
            handles_for_legend, labels_for_legend = \
                ax.get_legend_handles_labels()

    if handles_for_legend is not None:
        fig.legend(handles_for_legend, labels_for_legend,
                   loc="lower center", ncol=min(len(handles_for_legend), 5),
                   fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    fig.savefig(OUT_DIR / "fig11_lag_corr_chl_siconc.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote fig11_lag_corr_chl_siconc.png")


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

    fig.tight_layout()
    fig.savefig(OUT_DIR / "figS1_siconc_memory_seasonal_zone.png",
                dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.success("  wrote figS1_siconc_memory_seasonal_zone.png")



# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

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

    make_fig1(obs_clim)
    make_fig2(model_clim, obs_clim)
    make_fig3(model_clim, obs_clim)
    if model_seasonal:
        make_fig4(obs_full, model_seasonal)

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
            make_bias_map_2x2(
                mld_bias, obs_mld_clim,
                vmin=-50, vmax=50,
                unit_label="m",
                cmap="RdBu_r",
                fig_num=5,
                title="Mixed-layer depth bias: model − ORAS5",
                filename="fig5_mld_bias.png",
                stat_fmt="mean={mean:.1f} m  RMSE={rmse:.1f} m",
            )
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
            make_bias_map_2x2(
                sst_bias, obs_sst_clim,
                vmin=-3, vmax=3,
                unit_label="°C",
                cmap="RdBu_r",
                fig_num=6,
                title="SST bias: model − NOAA-ERSSTv3b",
                filename="fig6_sst_bias.png",
                stat_fmt="mean={mean:.2f} °C  RMSE={rmse:.2f} °C",
            )
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

    # =========================================================
    # Done
    # =========================================================
    logger.info(f"All figures in {OUT_DIR}")
    logger.info("Summary of figures produced:")
    for f in sorted(OUT_DIR.glob("fig*.png")):
        logger.info(f"  {f.name}")
