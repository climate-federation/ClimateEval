"""
plot_chl_evaluation.py
======================
Standalone plotting script for Chl-a model evaluation vs ESA-CCI.
Produces 4 figures:
  1. Bias maps (model - CCI) per model
  2. Seasonal cycle: models vs CCI (global + basins)
  3. Basin bias bar chart (Atlantic, Pacific, Indian, Southern, Arctic)
  4. Zonal mean (linear + log scale)

Usage:
    python plot_chl_evaluation.py

Reads data directly from CMIP6 files and CCI file.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

matplotlib.use("Agg")

# ============================================================
# Config
# ============================================================
YEAR_START = 1997
YEAR_END   = 2014
SCALE      = 1e6   # kg/m3 -> mg/m3
OUT_DIR    = Path.home() / "ClimateEval" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CCI_FILE = Path(
    "/work/bd0854/DATA/ESMValTool2/OBS/Tier2/ESACCI-OC/"
    "OBS6_ESACCI-OC_sat_fv5.0_Omon_chl_199709-202012.nc"
)

MODELS = {
    "GFDL-ESM4": {
        "path":    Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NOAA-GFDL/GFDL-ESM4/"
                        "historical/r1i1p1f1/Omon/chl/gr/v20190726"),
        "pattern": "chl_Omon_GFDL-ESM4_historical_r1i1p1f1_gr_*.nc",
        "color":   "#1f77b4",
    },
    "CESM2": {
        "path":    Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/NCAR/CESM2/"
                        "historical/r1i1p1f1/Omon/chl/gn"),
        "pattern": "chl_Omon_CESM2_historical_r1i1p1f1_gn_*.nc",
        "color":   "#d62728",
    },
    "MPI-ESM1-2-LR": {
        "path":    Path("/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/"
                        "historical/r22i1p1f1/Omon/chl/gn"),
        "pattern": "chl_Omon_MPI-ESM1-2-LR_historical_r22i1p1f1_gn_*.nc",
        "color":   "#2ca02c",
    },
}

BASINS = {
    "Atlantic": (-70,  70,  -80,   20),
    "Pacific":  (-60,  65, -180,  -70),
    "Indian":   (-70,  30,   20,  120),
    "Southern": (-90, -40, -180,  180),
    "Arctic":   ( 65,  90, -180,  180),
}

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
CCI_COLOR    = "darkorange"
CCI_LS       = "--"


# ============================================================
# Helpers
# ============================================================

def _norm_lons(lons):
    return np.where(lons > 180, lons - 360, lons)


def _fill_seam(data):
    """Fill NaN seam columns (CESM2 tripolar artifact)."""
    filled = data.copy()
    nrows, ncols = filled.shape
    for col in range(ncols):
        lc = (col - 1) % ncols
        rc = (col + 1) % ncols
        seam = np.isnan(filled[:, col]) & ~np.isnan(filled[:, lc]) & ~np.isnan(filled[:, rc])
        if seam.any():
            filled[seam, col] = (filled[seam, lc] + filled[seam, rc]) / 2.0
    return filled


def load_surface_mean(model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load time-mean surface Chl-a (mg/m3)."""
    cfg   = MODELS[model_name]
    paths = sorted(cfg["path"].glob(cfg["pattern"]))
    if not paths:
        paths = sorted(cfg["path"].parent.glob(cfg["pattern"]))
    if not paths:
        print(f"WARNING: no files for {model_name}")
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_mfdataset([str(p) for p in paths],
                                combine="by_coords", use_cftime=True)

    da = ds["chl"].sel(time=slice(str(YEAR_START), str(YEAR_END)))
    for dim in da.dims:
        if dim in ("lev", "depth", "olevel", "lev_partial"):
            da = da.isel({dim: 0}); break

    data = da.mean("time").values * SCALE
    lats = ds["lat"].values
    lons = _norm_lons(ds["lon"].values)
    ds.close()

    if data.shape == (len(lons), len(lats)):
        data = data.T
    data = _fill_seam(data)
    print(f"{model_name}: surface mean shape={data.shape} "
          f"min={np.nanmin(data):.4f} max={np.nanmax(data):.4f}")
    return data, lats, lons


def load_monthly_clim(model_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load monthly climatology (12, lat, lon) in mg/m3."""
    cfg   = MODELS[model_name]
    paths = sorted(cfg["path"].glob(cfg["pattern"]))
    if not paths:
        paths = sorted(cfg["path"].parent.glob(cfg["pattern"]))
    if not paths:
        return None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = xr.open_mfdataset([str(p) for p in paths],
                                combine="by_coords", use_cftime=True)

    da = ds["chl"].sel(time=slice(str(YEAR_START), str(YEAR_END)))
    for dim in da.dims:
        if dim in ("lev", "depth", "olevel", "lev_partial"):
            da = da.isel({dim: 0}); break

    clim = (da * SCALE).groupby("time.month").mean("time")
    lats = ds["lat"].values
    lons = _norm_lons(ds["lon"].values)
    ds.close()

    data = clim.values  # (12, lat, lon)
    print(f"{model_name}: monthly clim shape={data.shape}")
    return data, lats, lons


def load_cci_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds   = xr.open_dataset(str(CCI_FILE))
    da   = ds["chl"].sel(time=slice(str(YEAR_START), str(YEAR_END)))
    if "lev" in da.dims or "depth" in da.dims:
        da = da.isel({"lev" if "lev" in da.dims else "depth": 0})
    data = da.mean("time").values * SCALE
    lats = ds["lat"].values
    lons = _norm_lons(ds["lon"].values)
    ds.close()
    return data, lats, lons


def load_cci_monthly() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds   = xr.open_dataset(str(CCI_FILE))
    da   = ds["chl"].sel(time=slice(str(YEAR_START), str(YEAR_END)))
    if "lev" in da.dims or "depth" in da.dims:
        da = da.isel({"lev" if "lev" in da.dims else "depth": 0})
    clim = (da * SCALE).groupby("time.month").mean("time")
    lats = ds["lat"].values
    lons = _norm_lons(ds["lon"].values)
    ds.close()
    return clim.values, lats, lons


def regrid(src, src_lats, src_lons, tgt_lats, tgt_lons):
    si     = np.argsort(src_lons)
    interp = RegularGridInterpolator(
        (src_lats, src_lons[si]), src[:, si],
        method="linear", bounds_error=False, fill_value=np.nan)
    lo2d, la2d = np.meshgrid(tgt_lons, tgt_lats)
    return interp(np.column_stack([la2d.ravel(),
                                   lo2d.ravel()])).reshape(la2d.shape)


def basin_wmean(data, lats, lons, basin_name):
    la0, la1, lo0, lo1 = BASINS[basin_name]
    lat2d = np.meshgrid(lons, lats)[1]
    cos2d = np.cos(np.deg2rad(lat2d))
    lat_ok = (lats >= la0) & (lats <= la1)
    lon_ok = ((lons >= lo0) & (lons <= lo1) if lo0 < lo1
              else (lons >= lo0) | (lons <= lo1))
    ix  = np.ix_(lat_ok, lon_ok)
    sub = data[ix].ravel()
    w   = cos2d[ix].ravel()
    fin = np.isfinite(sub)
    return float(np.average(sub[fin], weights=w[fin])) if fin.sum() > 0 else np.nan


def basin_wmean_monthly(clim, lats, lons, basin_name):
    """Returns (12,) basin-mean seasonal cycle."""
    result = np.full(12, np.nan)
    for m in range(12):
        result[m] = basin_wmean(clim[m], lats, lons, basin_name)
    return result


def map_panel(ax, lons, lats, data, norm, cmap, title, left_labels=True):
    im = ax.pcolormesh(lons, lats, data, norm=norm, cmap=cmap,
                       transform=ccrs.PlateCarree(),
                       rasterized=True, shading="auto")
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=2)
    ax.coastlines(resolution="110m", lw=0.6, color="black", zorder=3)
    gl = ax.gridlines(draw_labels=True, lw=0.3,
                      color="gray", alpha=0.5, linestyle="--")
    gl.top_labels   = False
    gl.right_labels = False
    gl.left_labels  = left_labels
    gl.xlabel_style = {"fontsize": 10}
    gl.ylabel_style = {"fontsize": 10}
    ax.set_global()
    ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
    return im


def save_fig(fig, name):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================
# Load all data
# ============================================================
print("Loading CCI...")
cci_surf, cci_lats, cci_lons = load_cci_surface()
cci_clim, _, _               = load_cci_monthly()

model_surf  = {}
model_clim  = {}
for mn in MODELS:
    result = load_surface_mean(mn)
    if result:
        model_surf[mn] = result
    result2 = load_monthly_clim(mn)
    if result2:
        model_clim[mn] = result2

# Regrid all models to CCI grid for bias computation
model_surf_rg = {}
for mn, (d, lats, lons) in model_surf.items():
    if d.shape != cci_surf.shape:
        d = regrid(d, lats, lons, cci_lats, cci_lons)
    model_surf_rg[mn] = d

model_clim_rg = {}
for mn, (clim, lats, lons) in model_clim.items():
    if clim.shape[1:] != cci_surf.shape:
        clim_r = np.array([regrid(clim[m], lats, lons, cci_lats, cci_lons)
                           for m in range(12)])
    else:
        clim_r = clim
    model_clim_rg[mn] = clim_r


# ============================================================
# Fig 1: Bias maps (2x2 grid of models)
# ============================================================
print("\nFig 1: Bias maps...")

biases  = {mn: model_surf_rg[mn] - cci_surf for mn in model_surf_rg}
all_bv  = np.concatenate([b[np.isfinite(b)] for b in biases.values()])
blim    = float(np.nanpercentile(np.abs(all_bv), 95))
norm_b  = mcolors.Normalize(vmin=-blim, vmax=blim)
proj    = ccrs.PlateCarree()
n_m     = len(biases)
ncols   = 2
nrows   = int(np.ceil(n_m / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(11 * ncols, 6 * nrows),
                          subplot_kw={"projection": proj})
axes = np.atleast_2d(axes)
fig.suptitle(f"Chl-a bias: model − ESA-CCI  |  {YEAR_START}–{YEAR_END}  |  mg m⁻³",
             fontsize=15, fontweight="bold")

im_b = None
for idx, mn in enumerate(biases):
    row, col = divmod(idx, ncols)
    bias  = biases[mn]
    valid = np.isfinite(bias)
    rmse  = float(np.sqrt(np.nanmean(bias**2)))
    r     = float(np.corrcoef(model_surf_rg[mn][valid].ravel(),
                               cci_surf[valid].ravel())[0, 1]) \
            if valid.sum() > 10 else np.nan
    im_b  = map_panel(axes[row, col], cci_lons, cci_lats, bias,
                      norm_b, "RdBu_r", f"Bias: {mn} − CCI",
                      left_labels=(col == 0))
    txt = f"RMSE={rmse:.3f}  r={r:.3f}" if np.isfinite(r) else f"RMSE={rmse:.3f}"
    axes[row, col].text(0.02, 0.05, txt, transform=axes[row, col].transAxes,
                         fontsize=10, bbox=dict(fc="white", alpha=0.8))

for idx in range(n_m, nrows * ncols):
    row, col = divmod(idx, ncols)
    axes[row, col].set_visible(False)

if im_b is not None:
    sm2 = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm_b)
    sm2.set_array([])
    fig.set_size_inches(11 * ncols, 6 * nrows + 2)
    fig.subplots_adjust(top=0.90, bottom=0.12, wspace=0.05, hspace=0.15)
    cax = fig.add_axes([0.15, 0.05, 0.70, 0.025])
    cb  = fig.colorbar(sm2, cax=cax, orientation="horizontal")
    cb.set_label("Bias [mg m⁻³]", fontsize=13)
    cb.ax.tick_params(labelsize=12)
save_fig(fig, "fig1_bias_maps")


# ============================================================
# Fig 2: Seasonal cycles — global + 5 basins
# ============================================================
print("Fig 2: Seasonal cycles...")
months = np.arange(1, 13)

fig, axes = plt.subplots(2, 3, figsize=(20, 12), squeeze=False)
fig.suptitle(f"Chl-a seasonal cycle: models vs ESA-CCI  |  {YEAR_START}–{YEAR_END}  |  mg m⁻³",
             fontsize=15, fontweight="bold")

all_panels = [("Global", None)] + list(BASINS.items())

for ax, (bn, bounds) in zip(axes.flat, all_panels):
    # CCI
    if bounds is None:
        cci_ts = np.array([np.nanmean(cci_clim[m]) for m in range(12)])
    else:
        cci_ts = basin_wmean_monthly(cci_clim, cci_lats, cci_lons, bn)
    ax.plot(months, cci_ts, color=CCI_COLOR, lw=2.5, ls=CCI_LS,
            marker="s", ms=5, label="ESA-CCI")

    # Models
    all_ts = []
    for mn, clim_r in model_clim_rg.items():
        color = MODELS[mn]["color"]
        if bounds is None:
            ts = np.array([np.nanmean(clim_r[m]) for m in range(12)])
        else:
            ts = basin_wmean_monthly(clim_r, cci_lats, cci_lons, bn)
        all_ts.append(ts)
        ax.plot(months, ts, color=color, lw=2.5, marker="o", ms=5, label=mn)

    if len(all_ts) > 1:
        all_arr = np.array(all_ts)
        ax.fill_between(months, np.nanmin(all_arr, axis=0),
                         np.nanmax(all_arr, axis=0),
                         alpha=0.12, color="gray")

    ax.set_xticks(months)
    ax.set_xticklabels(MONTH_LABELS, fontsize=11)
    ax.set_title(bn, fontsize=13, fontweight="bold")
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Chl-a [mg m⁻³]", fontsize=12)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(True, alpha=0.3)

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center",
           ncol=len(MODELS) + 1, fontsize=13,
           frameon=True, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
save_fig(fig, "fig2_seasonal_cycles")


# ============================================================
# Fig 3: Basin bias bar chart
# ============================================================
print("Fig 3: Basin bias bar chart...")

basin_bias = {mn: {} for mn in model_surf_rg}
for mn, d in model_surf_rg.items():
    bias = d - cci_surf
    for bn in BASINS:
        basin_bias[mn][bn] = basin_wmean(bias, cci_lats, cci_lons, bn)

bnames  = list(BASINS.keys())
x       = np.arange(len(bnames))
n_m     = len(basin_bias)
bw      = 0.7 / n_m

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f"Basin-mean Chl-a bias: model − ESA-CCI  |  {YEAR_START}–{YEAR_END}  |  mg m⁻³",
             fontsize=15, fontweight="bold")

for ax, use_abs in zip(axes, [False, True]):
    for i, (mn, bd) in enumerate(basin_bias.items()):
        color  = MODELS[mn]["color"]
        vals   = [abs(bd[bn]) if use_abs else bd[bn] for bn in bnames]
        offset = (i - n_m / 2 + 0.5) * bw
        bars   = ax.bar(x + offset, vals, width=bw, color=color,
                        label=mn, alpha=0.85, edgecolor="white", lw=0.5)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() * 1.02, f"{val:.3f}",
                         ha="center", va="bottom",
                         fontsize=9, color=color, fontweight="bold")
    if not use_abs:
        ax.axhline(0, color="k", lw=1.2, ls="--", alpha=0.6)
    else:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(bnames, fontsize=13, rotation=15, ha="right")
    ax.set_ylabel("|Bias| [mg m⁻³]" if use_abs else "Bias [mg m⁻³]", fontsize=14)
    ax.set_title("Absolute bias (log)" if use_abs else "Signed bias",
                  fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=13)

plt.tight_layout()
save_fig(fig, "fig3_basin_bias")


# ============================================================
# Fig 4: Zonal mean
# ============================================================
print("Fig 4: Zonal mean...")

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle(f"Zonal mean surface Chl-a: models vs ESA-CCI  |  {YEAR_START}–{YEAR_END}  |  mg m⁻³",
             fontsize=15, fontweight="bold")

all_z = []
for mn, d in model_surf_rg.items():
    color = MODELS[mn]["color"]
    z = np.nanmean(d, axis=1)
    all_z.append(z)
    for ax in axes:
        ax.plot(cci_lats, z, color=color, lw=2.5, label=mn)

if len(all_z) > 1:
    for ax in axes:
        ax.fill_between(cci_lats,
                         np.nanmin(all_z, axis=0),
                         np.nanmax(all_z, axis=0),
                         alpha=0.12, color="gray", label="Model spread")

cci_z = np.nanmean(cci_surf, axis=1)
for ax in axes:
    ax.plot(cci_lats, cci_z, color=CCI_COLOR, lw=2.5, ls=CCI_LS, label="ESA-CCI")
    for xv in [0, 40, -40]:
        ax.axvline(xv, color="k", lw=0.5, ls=":", alpha=0.3)
    ax.set_xlabel("Latitude [°]", fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=13)

axes[0].set_ylabel("Chl-a [mg m⁻³]", fontsize=14)
axes[0].set_title("Linear scale", fontsize=13, fontweight="bold")
axes[1].set_yscale("log")
axes[1].set_ylabel("Chl-a [mg m⁻³] (log)", fontsize=14)
axes[1].set_title("Log scale", fontsize=13, fontweight="bold")
plt.tight_layout()
save_fig(fig, "fig4_zonal_mean")

print(f"\nAll figures saved to: {OUT_DIR}")
