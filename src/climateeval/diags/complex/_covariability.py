"""Seasonal covariability of chlorophyll and its physical drivers.

Pure numerical routines, no input or output: given monthly fields on a
regular grid, they return the phase, amplitude and covariability metrics
used to compare a model's seasonal chlorophyll cycle with observations.

The methods here are the ones behind Navarra et al. and the numbers they
produce are the published ones, so this module is deliberately a
verbatim extraction rather than a reimplementation.

Three points are easy to get wrong and are handled explicitly:

* Months wrap. A lag of +6 and one of -6 are the same point, and the
  linear mean of December and January is July. Every phase quantity
  therefore goes through `circ_mean`, `circ_diff` or `circ_sd` rather
  than through `numpy.mean` or a subtraction.
* For a near-sinusoidal pair, lag L and lag L+6 give correlations of
  equal magnitude and opposite sign, so maximising |r| picks between
  them arbitrarily. `seasonal_covariability` resolves it with a
  declared expected sign per driver (`COV_EXPECTED_SIGN`): a deeper
  mixed layer suppresses chlorophyll, warmer water accompanies it.
* A second harmonic on a regional mean does not distinguish a genuine
  autumn bloom from single blooms that peak at different times in
  different places. `region_bimodality` therefore fits each cell and
  then averages circularly, never the reverse.
"""

from __future__ import annotations

import warnings
from collections import defaultdict

import numpy as np

PERIOD = (1997, 2014)

ICE_THRESHOLD = 0.15          # fraction, the conventional ice edge

REGIONS = {
    "Global":           (-90,  90,   0, 360),
    "Arctic":           ( 60,  80,   0, 360),
    "Southern Ocean":   (-80, -30,   0, 360),
    "North Atlantic":   ( 30,  70, 290, 360),
    "North Pacific":    ( 30,  60, 130, 260),
    "Indian":           (-30,  30,  20, 130),
    "Tropical Pacific": (-20,  20, 130, 280),
    "South Pacific":    (-30,  -5, 150, 290),
}

VARS = {
    # chl has no fixed scale: the factor comes from each file's units
    # attribute (see chl_units_scale), because assuming kg m-3 put one
    # model three orders of magnitude out in the first run.
    #
    # "valid" is a physical range applied AFTER scaling. Not every
    # product declares its land fill value, and an undeclared 1e20 passes
    # np.isfinite and is then averaged in as though it were data.
    "chl":    {"table": "Omon",  "scale": None, "depth": 0.0,
               "valid": (0.0, 1.0e4)},
    "mlotst": {"table": "Omon",  "scale": 1.0,  "depth": None,
               "valid": (0.0, 1.0e4)},
    "siconc": {"table": "SImon", "scale": None, "depth": None,
               "valid": (0.0, 1.5)},
    "tos":    {"table": "Omon",  "scale": 1.0,  "depth": None,
               "valid": (-3.0, 40.0)},
}

MIN_YEARS = 5           # a climatological month needs this many years

MIN_MONTHS_FIT = 7      # a harmonic fit needs this many calendar months

MIN_A1_REL = 0.10

PLOT_REGIONS = ["Arctic", "Southern Ocean", "North Atlantic",
                "North Pacific"]

def log(msg: str) -> None:
    print(msg, flush=True)

def latlon_2d(da):
    """(lat2d, lon2d, rectilinear) for the trailing two dims of `da`."""
    def pick(*names):
        for n in names:
            if n in da.coords:
                return da[n]
        return None

    lat = pick("lat", "latitude", "nav_lat")
    lon = pick("lon", "longitude", "nav_lon")
    if lat is None or lon is None:
        raise ValueError("no latitude/longitude coordinates found")
    lat_v = np.asarray(lat.values, dtype=float)
    lon_v = np.asarray(lon.values, dtype=float)
    if lat_v.ndim == 1 and lon_v.ndim == 1:
        lat2d = np.repeat(lat_v[:, None], lon_v.size, axis=1)
        lon2d = np.repeat(lon_v[None, :], lat_v.size, axis=0)
        rect = True
    else:
        lat2d, lon2d, rect = lat_v, lon_v, False
    return lat2d, lon2d % 360.0, rect

def region_masks(lat2d: np.ndarray, lon2d: np.ndarray,
                 regions: dict) -> dict:
    out = {}
    for name, (la0, la1, lo0, lo1) in regions.items():
        m = (lat2d >= la0) & (lat2d <= la1)
        if lo0 <= lo1:
            m &= (lon2d >= lo0) & (lon2d <= lo1)
        else:                                    # wraps the dateline
            m &= (lon2d >= lo0) | (lon2d <= lo1)
        out[name] = m
    return out

def region_weighted_means(field: np.ndarray, weight: np.ndarray,
                          masks: dict) -> dict:
    """Area-weighted mean of a 2-D field inside each mask.

    The valid cells are pulled out as 1-D vectors first. Zeroing the
    weight of an invalid cell does NOT keep it out of a weighted sum,
    because NaN times zero is NaN; one masked cell would make the whole
    regional mean NaN.
    """
    out = {}
    finite = np.isfinite(field)
    for name, m in masks.items():
        sel = finite & m
        if not sel.any():
            continue
        v = field[sel]
        w = weight[sel]
        W = float(np.sum(w))
        if not np.isfinite(W) or W <= 0:
            continue
        out[name] = float(np.sum(v * w) / W)
    return out

class _Reduction:
    """State carried across the files of one variable.

    Holds the region masks, the area weights and the unit scale so that
    they are established once from the first file and reused, rather
    than recomputed per file (and possibly differing between them).
    """

    __slots__ = ("acc", "masks", "weight", "scale", "unit_note", "nfiles")

    def __init__(self, regions, scale):
        self.acc = {r: defaultdict(lambda: [0.0, 0.0]) for r in regions}
        self.masks = None
        self.weight = None
        self.scale = scale
        self.unit_note = None
        self.nfiles = 0

    def series(self):
        out = {}
        for rname, d in self.acc.items():
            got = {k: v[0] / v[1] for k, v in d.items() if v[1] > 0}
            if got:
                out[rname] = got
        return out

def reduce_dataarray(da, regions: dict, period, state: "_Reduction",
                     surface=False, auto_percent=False, grid_clim=None,
                     units_from_file=False, model="", valid_range=None,
                     level_depths=None):
    """Reduce one loaded field into a region-mean accumulator.

    This is the single implementation of the reduction. Both entry
    points call it -- the streaming reader once per file, the in-memory
    one once per field -- so a change to the masking, the weighting or
    the unit handling cannot apply to one path and not the other.

    `da` is an xarray DataArray with a time dimension and latitude and
    longitude coordinates. An iris cube is converted at the boundary
    with xr.DataArray.from_iris.
    """
    if "time" in da.dims and da.dims[0] != "time":
        da = da.transpose("time", ...)

    if surface:
        for zname in ("lev", "olevel", "depth", "deptht", "z_l"):
            if zname in da.dims:
                z = level_depths
                if z is None and zname in da.coords:
                    z = np.asarray(da[zname].values, float)
                k = 0 if z is None else int(np.argmin(np.abs(z)))
                da = da.isel({zname: k})
                break

    if units_from_file and state.unit_note is None:
        state.scale, u, src = chl_units_scale(da.attrs.get("units"), model)
        state.unit_note = f"units '{u}' -> x{state.scale:g} ({src})"

    if state.masks is None:
        lat2d, lon2d, _ = latlon_2d(da)
        state.masks = region_masks(lat2d, lon2d, regions)
        state.weight = np.cos(np.deg2rad(np.clip(lat2d, -89.9, 89.9)))
        if grid_clim is not None:
            grid_clim.bind(lat2d, lon2d, state.weight)

    times = da["time"].values
    yy = np.array([_year_of(t) for t in times])
    mm = np.array([_month_of(t) for t in times])
    keep = (yy >= period[0]) & (yy <= period[1])
    if not keep.any():
        return
    idx = np.flatnonzero(keep)
    da = da.isel(time=idx)
    yy, mm = yy[idx], mm[idx]

    nt = da.sizes["time"]
    per_step = int(np.prod([da.sizes[d] for d in da.dims if d != "time"]))
    nblock = int(max(1, min(240, 256e6 / max(per_step * 8, 1))))
    for t0 in range(0, nt, nblock):
        t1 = min(t0 + nblock, nt)
        vals = np.asarray(da.isel(time=slice(t0, t1)).values, dtype=float)
        if auto_percent and state.scale is None:
            mx = np.nanmax(vals) if np.isfinite(vals).any() else 0.0
            state.scale = 0.01 if mx > 1.5 else 1.0
        s = 1.0 if state.scale is None else state.scale
        vals = vals * s
        if valid_range is not None:
            # Not every product declares its land fill value. ORAS5 over
            # OPeNDAP serves land as a large number, which would pass
            # np.isfinite and be averaged in as if it were a 1e20 m
            # mixed layer.
            lo, hi = valid_range
            vals = np.where((vals >= lo) & (vals <= hi), vals, np.nan)
        if grid_clim is not None:
            grid_clim.add(vals, mm[t0:t1])
        for k in range(vals.shape[0]):
            t = t0 + k
            for rname, mu in region_weighted_means(
                    vals[k], state.weight, state.masks).items():
                rec = state.acc[rname][(int(yy[t]), int(mm[t]))]
                rec[0] += mu
                rec[1] += 1.0
        del vals

def _year_of(t):
    """Calendar year of a cftime object, numpy datetime64 or datetime."""
    y = getattr(t, "year", None)
    if y is not None:
        return int(y)
    return int(str(np.datetime64(t, "Y")))

def _month_of(t):
    m = getattr(t, "month", None)
    if m is not None:
        return int(m)
    return int(str(np.datetime64(t, "M"))[5:7])

def series_from_field(da, regions: dict, period, scale=1.0, surface=False,
                      auto_percent=False, grid_clim=None,
                      units_from_file=False, model="", valid_range=None):
    """Region-mean monthly series from a field already in memory.

    The entry point for a host that has loaded the data itself -- the
    ClimateEval diagnostic, which receives iris cubes. Convert at the
    boundary and call this:

        import xarray as xr
        da = xr.DataArray.from_iris(cube)
        series, _ = series_from_field(da, REGIONS, PERIOD, ...)

    Returns the same {region: {(year, month): value}} as stream_monthly,
    by the same code path, so results are identical whichever way the
    data arrived.
    """
    state = _Reduction(regions, scale)
    reduce_dataarray(da, regions, period, state, surface=surface,
                     auto_percent=auto_percent, grid_clim=grid_clim,
                     units_from_file=units_from_file, model=model,
                     valid_range=valid_range)
    state.nfiles = 1
    if state.unit_note:
        log(f"      {state.unit_note}")
    return state.series(), 1

def to_year_month_matrix(series: dict, period) -> np.ndarray:
    """{(year, month): value} -> array [nyear, 12] with NaN for gaps."""
    years = list(range(period[0], period[1] + 1))
    M = np.full((len(years), 12), np.nan)
    for (y, m), v in series.items():
        if period[0] <= y <= period[1] and 1 <= m <= 12:
            M[y - period[0], m - 1] = v
    return M

def circ_diff(a, b):
    """a - b in months, wrapped to [-6, +6)."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    return float((a - b + 6.0) % 12.0 - 6.0)

def circ_mean(phases):
    """Circular mean of month-valued phases, in [0, 12)."""
    p = np.asarray([x for x in phases if np.isfinite(x)], float)
    if p.size == 0:
        return np.nan
    th = 2.0 * np.pi * p / 12.0
    ang = np.arctan2(np.mean(np.sin(th)), np.mean(np.cos(th)))
    return float((ang * 12.0 / (2.0 * np.pi)) % 12.0)

def circ_sd(phases):
    """Circular standard deviation in months (Mardia), NaN if n < 3.

    sqrt(-2 ln R) is the circular analogue of the standard deviation; it
    grows without bound as the phases spread out, so it is clipped at 6
    months, beyond which 'a typical month' has no meaning anyway.
    """
    p = np.asarray([x for x in phases if np.isfinite(x)], float)
    if p.size < 3:
        return np.nan
    th = 2.0 * np.pi * p / 12.0
    R = np.hypot(np.mean(np.cos(th)), np.mean(np.sin(th)))
    if R <= 1e-12:
        return 6.0
    sd = np.sqrt(-2.0 * np.log(min(R, 1.0))) * 12.0 / (2.0 * np.pi)
    return float(min(sd, 6.0))

def harmonic_fit(vals: np.ndarray, nharm: int = 2):
    """Least-squares annual harmonics of a 12-month series.

    Fitted to whatever months are present rather than requiring a full
    year, because ESA-CCI has no Arctic winter. Returns amplitudes and
    the month at which each harmonic peaks, or None if too few months.
    """
    v = np.asarray(vals, float)
    ok = np.isfinite(v)
    n = int(ok.sum())
    if n < max(MIN_MONTHS_FIT, 2 * nharm + 2):
        return None
    m = np.arange(1, 13)[ok]
    th = 2.0 * np.pi * (m - 1) / 12.0
    cols = [np.ones(n)]
    for k in range(1, nharm + 1):
        cols += [np.cos(k * th), np.sin(k * th)]
    A = np.column_stack(cols)
    try:
        coef, *_ = np.linalg.lstsq(A, v[ok], rcond=None)
    except np.linalg.LinAlgError:
        return None
    out = {"mean": float(coef[0]), "n_months": n}
    for k in range(1, nharm + 1):
        b, c = float(coef[2 * k - 1]), float(coef[2 * k])
        amp = float(np.hypot(b, c))
        psi = np.arctan2(c, b)                 # peak where k*th == psi
        ph = (psi / k) * 12.0 / (2.0 * np.pi) + 1.0
        out[f"A{k}"] = amp
        out[f"phase{k}"] = float(ph % 12.0)
    return out

def seasonal_covariability(a, b, max_lag: int = 6, prefer: int = 0):
    """Circular lagged correlation between two 12-month climatologies.

    This is the quantity Marina's third option asks for: how chlorophyll
    and a physical driver move together THROUGH the seasonal cycle. No
    deseasonalising is involved, so the objection to correlating monthly
    anomalies -- that they mix subseasonal with interannual variance and
    cannot carry a mechanism -- does not apply.

    Convention: r(lag) = corr(a[m], b[m - lag]), so a POSITIVE lag means
    b LEADS a. With a = Chl-a, a positive lag is a bloom that follows the
    driver.

    `prefer` sets which extreme is reported, and it is not cosmetic. For
    a near-sinusoidal pair, the lag of greatest |r| is AMBIGUOUS: lag L
    and lag L+6 give correlations of equal size and opposite sign, so
    maximising |r| alone picks between two contradictory readings by
    accident. Each driver has a known expected sign instead -- a bloom
    goes with a shallow mixed layer and little ice, and with warm water
    -- so the caller asks for the most negative (-1) or most positive
    (+1) correlation. prefer=0 falls back to greatest |r| with ties
    broken toward the shorter lag, which is only safe when the sign is
    genuinely unknown.

    Needs no harmonic fit, so it survives the double-peaked cycles where
    a single phase is ill-defined, and it works on gappy months (Arctic
    winter) from the pairs that exist.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    out = {"lag": np.nan, "r": np.nan, "r0": np.nan, "n": 0,
           "lag_pos": np.nan, "r_pos": np.nan,
           "lag_neg": np.nan, "r_neg": np.nan}
    if a.size != 12 or b.size != 12:
        return out

    best = {}
    for lag in range(-max_lag, max_lag + 1):
        bb = np.roll(b, lag)                 # bb[m] == b[m - lag]
        ok = np.isfinite(a) & np.isfinite(bb)
        if int(ok.sum()) < 6:
            continue
        x, y = a[ok], bb[ok]
        if np.std(x) <= 0 or np.std(y) <= 0:
            continue
        best[lag] = (float(np.corrcoef(x, y)[0, 1]), int(ok.sum()))
    if not best:
        return out
    if 0 in best:
        out["r0"] = best[0][0]

    # Among equal correlations prefer the shorter lag: a relationship
    # explained by no shift should not be reported as a six-month one.
    pos = sorted(best, key=lambda L: (-best[L][0], abs(L)))[0]
    neg = sorted(best, key=lambda L: (best[L][0], abs(L)))[0]
    out["lag_pos"], out["r_pos"] = float(pos), best[pos][0]
    out["lag_neg"], out["r_neg"] = float(neg), best[neg][0]

    if prefer > 0:
        pick = pos
    elif prefer < 0:
        pick = neg
    else:
        pick = sorted(best,
                      key=lambda L: (-abs(best[L][0]), abs(L), -best[L][0]))[0]
    out["lag"], out["r"] = float(pick), best[pick][0]
    out["n"] = best[pick][1]
    return out

COV_EXPECTED_SIGN = {"mld": -1, "ice": -1, "sst": +1}

def peak_month(vals: np.ndarray):
    v = np.asarray(vals, float)
    if not np.isfinite(v).any():
        return np.nan
    return float(np.nanargmax(v) + 1)

def restratification_month(mld_clim: np.ndarray):
    """Month of strongest shoaling, from the cyclic difference of MLD.

    d[m] is the change from month m to month m+1, so the steepest
    shoaling is placed at the midpoint m + 0.5 rather than at either end.
    """
    v = np.asarray(mld_clim, float)
    if np.isfinite(v).sum() < 12:
        return np.nan                       # a cyclic derivative needs all 12
    d = np.roll(v, -1) - v
    k = int(np.nanargmin(d))
    return float(((k + 1) + 0.5 - 1.0) % 12.0 + 1.0)

def ice_retreat_month(sic_clim: np.ndarray, thresh: float = ICE_THRESHOLD):
    """Month the ice climatology falls through `thresh` after its maximum.

    Returns NaN where the region never reaches the threshold (no seasonal
    ice) or never drops below it (perennial ice); in both cases a retreat
    date does not exist and should not be invented.
    """
    v = np.asarray(sic_clim, float)
    if np.isfinite(v).sum() < 12:
        return np.nan
    if np.nanmax(v) < thresh or np.nanmin(v) >= thresh:
        return np.nan
    start = int(np.nanargmax(v))
    for step in range(12):
        i = (start + step) % 12
        j = (i + 1) % 12
        if v[i] >= thresh > v[j]:
            denom = v[i] - v[j]
            frac = 0.0 if denom <= 0 else (v[i] - thresh) / denom
            return float(((i + frac) % 12.0) + 1.0)
    return np.nan

def climatology(M: np.ndarray, min_years: int = MIN_YEARS) -> np.ndarray:
    """Monthly climatology from a [nyear, 12] matrix, NaN where too thin."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clim = np.nanmean(M, axis=0)
    counts = np.sum(np.isfinite(M), axis=0)
    clim[counts < min_years] = np.nan
    return clim

def phase_spread(M: np.ndarray):
    """Circular sd of the per-year first-harmonic phase, in months."""
    phases = []
    for row in M:
        fit = harmonic_fit(row, nharm=1)
        if fit is not None and fit["A1"] > 0:
            phases.append(fit["phase1"])
    return circ_sd(phases), len(phases)

def metrics_for_region(chl_M, mld_M, sic_M, sst_M=None) -> dict:
    """All seasonal-phase metrics for one region of one dataset."""
    out = {k: np.nan for k in
           ("bloom_phase", "bloom_peak", "bloom_phase_sd", "A1", "A2",
            "A1_rel", "A2_over_A1_regmean", "A2_over_A1", "cells_frac",
            "bimodal_frac", "two_peak_frac", "chl_mean", "n_months_chl",
            "n_years_phase",
            "mld_max_month", "restrat_month", "ice_retreat_month",
            "ice_area_frac", "lag_bloom_restrat", "lag_bloom_ice",
            "cov_mld_lag", "cov_mld_r", "cov_mld_r0",
            "cov_sst_lag", "cov_sst_r", "cov_sst_r0",
            "cov_ice_lag", "cov_ice_r", "cov_ice_r0")}

    if chl_M is not None:
        clim = climatology(chl_M)
        fit = harmonic_fit(clim, nharm=2)
        out["bloom_peak"] = peak_month(clim)
        if fit is not None:
            out["bloom_phase"] = fit["phase1"]
            out["A1"] = fit["A1"]
            out["A2"] = fit["A2"]
            out["chl_mean"] = fit["mean"]
            out["n_months_chl"] = fit["n_months"]
            if fit["mean"] > 0:
                out["A1_rel"] = fit["A1"] / fit["mean"]
            # Reported only where there is an annual cycle to divide by.
            if fit["A1"] > 0 and out["A1_rel"] >= MIN_A1_REL:
                out["A2_over_A1_regmean"] = fit["A2"] / fit["A1"]
        sd, nyr = phase_spread(chl_M)
        out["bloom_phase_sd"] = sd
        out["n_years_phase"] = nyr

    if mld_M is not None:
        mclim = climatology(mld_M)
        out["mld_max_month"] = peak_month(mclim)
        out["restrat_month"] = restratification_month(mclim)

    if sic_M is not None:
        sclim = climatology(sic_M)
        out["ice_retreat_month"] = ice_retreat_month(sclim)

    out["lag_bloom_restrat"] = circ_diff(out["bloom_phase"],
                                         out["restrat_month"])
    out["lag_bloom_ice"] = circ_diff(out["bloom_phase"],
                                     out["ice_retreat_month"])

    # Seasonal covariability of Chl-a with each physical driver. A region
    # mean is the right reduction here, including for sea ice: the 15%
    # threshold was only a problem for defining a retreat DATE, while a
    # regional ice concentration varies through the year perfectly well
    # whether or not it ever crosses that line.
    if chl_M is not None:
        chl_clim = climatology(chl_M)
        for key, M in (("mld", mld_M), ("sst", sst_M), ("ice", sic_M)):
            if M is None:
                continue
            cov = seasonal_covariability(
                chl_clim, climatology(M),
                prefer=COV_EXPECTED_SIGN.get(key, 0))
            out[f"cov_{key}_lag"] = cov["lag"]
            out[f"cov_{key}_r"] = cov["r"]
            out[f"cov_{key}_r0"] = cov["r0"]
    return out

def near_wrap(lag, tol: float = 1.0) -> bool:
    """True if a lag sits within `tol` months of the +/-6 wrap point.

    A lag of +5.8 and one of -5.8 are a third of a month apart on the
    circle but land at opposite ends of a linear axis. Such a value is
    reported but should not be read as a large positive or negative lag.
    """
    return bool(np.isfinite(lag) and abs(abs(lag) - 6.0) <= tol)

CHL_UNIT_TO_MG_M3 = {
    "kg m-3": 1.0e6, "kg/m3": 1.0e6, "kg m^-3": 1.0e6, "kgm-3": 1.0e6,
    "kg.m-3": 1.0e6, "kg m**-3": 1.0e6,
    "g m-3": 1.0e3, "g/m3": 1.0e3, "g m**-3": 1.0e3,
    "mg m-3": 1.0, "mg/m3": 1.0, "mg m^-3": 1.0, "mg m**-3": 1.0,
    "ug l-1": 1.0, "ug/l": 1.0, "mg l-1": 1.0e3, "mg/l": 1.0e3,
}

CHL_SCALE_OVERRIDE: dict[str, float] = {
    # IPSL-CM6A-LR declares kg m-3, but its values are a thousand times
    # larger than every other model and than ESA-CCI: 268-374 mg m-3
    # against an observed 0.28-1.42, in all eight regions of the
    # 2026-11-19 run. Reading the units attribute therefore does not
    # help; the attribute is wrong. Phases, amplitude ratios and
    # correlations are scale-invariant and were never affected -- only
    # the concentrations were.
    "IPSL-CM6A-LR": 1.0e3,
}

def chl_units_scale(units, model=""):
    """(factor to mg m-3, normalised unit string, source of the factor)."""
    if model in CHL_SCALE_OVERRIDE:
        return float(CHL_SCALE_OVERRIDE[model]), str(units), "override"
    u = (str(units or "").strip().lower()
         .replace("**", "^").replace("_", " "))
    u = " ".join(u.split()).replace("^-", "-")
    if u in CHL_UNIT_TO_MG_M3:
        return CHL_UNIT_TO_MG_M3[u], u, "units attribute"
    return 1.0e6, (u or "(absent)"), "ASSUMED kg m-3"

class GridClim:
    """Monthly climatology accumulated per grid cell while streaming.

    Needed because two of the metrics cannot be computed from a regional
    mean without changing their meaning. An ice-retreat date averaged over
    a band that is partly perennial ice and partly open water is not a
    retreat date, and the second-harmonic ratio of a regional mean cannot
    tell a genuinely double-peaked cycle from a mixture of single-peaked
    cycles whose phases differ across the region. Both have to be computed
    cell by cell and aggregated afterwards.
    """

    def __init__(self):
        self.sum = None
        self.cnt = None
        self.lat2d = None
        self.lon2d = None
        self.weight = None

    def bind(self, lat2d, lon2d, weight):
        if self.lat2d is None:
            self.lat2d, self.lon2d, self.weight = lat2d, lon2d, weight

    def add(self, vals, months):
        """vals [nt, ny, nx] already scaled; months [nt] of 1..12."""
        if self.sum is None:
            ny, nx = vals.shape[1], vals.shape[2]
            self.sum = np.zeros((12, ny, nx))
            self.cnt = np.zeros((12, ny, nx), dtype=np.int32)
        if vals.shape[1:] != self.sum.shape[1:]:
            return                      # grid changed mid-stream; skip
        ok = np.isfinite(vals)
        safe = np.where(ok, vals, 0.0)  # NaN*1 would poison the sum
        for k in range(vals.shape[0]):
            m = int(months[k]) - 1
            if 0 <= m < 12:
                self.sum[m] += safe[k]
                self.cnt[m] += ok[k]

    def clim(self, min_years=MIN_YEARS):
        """[12, ny, nx] climatology, NaN where a month is too thin."""
        if self.sum is None:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.where(self.cnt >= min_years, self.sum / np.maximum(self.cnt, 1),
                         np.nan)
        return c

def harmonic_fit_grid(clim, nharm: int = 2, min_months: int = MIN_MONTHS_FIT):
    """Per-cell least-squares harmonics of a [12, ny, nx] climatology.

    Cells are grouped by which calendar months they have, so one solve
    covers every cell sharing a coverage pattern. Satellite coverage is
    spatially coherent, so there are only a handful of distinct patterns
    and this stays fast while remaining exact for gappy cells.
    """
    nm, ny, nx = clim.shape
    flat = clim.reshape(nm, -1)
    ok = np.isfinite(flat)
    ncell = flat.shape[1]

    mean = np.full(ncell, np.nan)
    amp = np.full((nharm, ncell), np.nan)
    pha = np.full((nharm, ncell), np.nan)

    bits = (ok.astype(np.int64) * (1 << np.arange(nm, dtype=np.int64))[:, None]
            ).sum(axis=0)
    for key in np.unique(bits):
        if key == 0:
            continue
        cols = np.flatnonzero(bits == key)
        rows = np.flatnonzero(np.asarray(
            [(int(key) >> b) & 1 for b in range(nm)], dtype=bool))
        n = rows.size
        if n < max(min_months, 2 * nharm + 2):
            continue
        th = 2.0 * np.pi * rows / 12.0
        A = [np.ones(n)]
        for k in range(1, nharm + 1):
            A += [np.cos(k * th), np.sin(k * th)]
        A = np.column_stack(A)
        try:
            coef, *_ = np.linalg.lstsq(A, flat[np.ix_(rows, cols)], rcond=None)
        except np.linalg.LinAlgError:
            continue
        mean[cols] = coef[0]
        for k in range(1, nharm + 1):
            b, c = coef[2 * k - 1], coef[2 * k]
            amp[k - 1, cols] = np.hypot(b, c)
            psi = np.arctan2(c, b)
            pha[k - 1, cols] = ((psi / k) * 12.0 / (2.0 * np.pi) + 1.0) % 12.0

    shape = (ny, nx)
    return (mean.reshape(shape),
            amp.reshape((nharm,) + shape),
            pha.reshape((nharm,) + shape))

def per_cell_two_peaks(clim, min_rel: float = 0.5, min_months: int = 8):
    """Fraction-of-area test for a genuinely double-peaked annual cycle.

    A2/A1 above a threshold measures ASYMMETRY, not two blooms: a cycle
    can be strongly skewed with a single maximum. This counts local
    maxima in each cell's climatology directly and asks for a second one
    reaching `min_rel` of the primary, which is what "a spring and an
    autumn bloom" actually means.

    Gaps are filled by circular linear interpolation before the test --
    ESA-CCI has no Arctic winter, and those missing months sit at the
    seasonal minimum, so leaving them out would not change the peak
    structure but would disqualify most Arctic cells. The interpolation
    is linear, so it can only ever REMOVE a peak, never invent one.
    Eight observed months is the floor, set by the ESA-CCI Arctic itself:
    with November-February missing, both a spring and an autumn bloom are
    still observed directly. Cells below that are excluded, not guessed.
    """
    nm, ny, nx = clim.shape
    flat = clim.reshape(nm, -1).copy()
    ok = np.isfinite(flat)
    enough = ok.sum(axis=0) >= min_months
    out = np.full(flat.shape[1], np.nan)

    cols = np.flatnonzero(enough)
    if cols.size == 0:
        return out.reshape(ny, nx)

    # circular linear interpolation of the gaps
    x = np.arange(nm)
    for c in cols:
        v = flat[:, c]
        m = np.isfinite(v)
        if not m.all():
            xp = np.concatenate([x[m] - nm, x[m], x[m] + nm])
            fp = np.tile(v[m], 3)
            v = np.interp(x, xp, fp)
            flat[:, c] = v

    sub = flat[:, cols]
    hi = np.roll(sub, 1, axis=0)
    lo = np.roll(sub, -1, axis=0)
    isloc = (sub > hi) & (sub > lo)
    peak = np.nanmax(sub, axis=0)
    base = np.nanmin(sub, axis=0)
    span = peak - base
    with np.errstate(invalid="ignore"):
        rel = np.where(span > 0, (sub - base) / np.maximum(span, 1e-30), 0.0)
    strong = isloc & (rel >= min_rel)
    out[cols] = (strong.sum(axis=0) >= 2).astype(float)
    return out.reshape(ny, nx)

def per_cell_retreat(sic_clim, thresh: float = ICE_THRESHOLD):
    """Ice-retreat month per cell, and the cells for which it is defined.

    A cell qualifies only if its climatology both reaches and falls below
    the threshold. Perennial ice and permanently open water have no
    retreat date, and inventing one for them is what made the Arctic band
    mean undefined in the first place.
    """
    nm, ny, nx = sic_clim.shape
    out = np.full((ny, nx), np.nan)
    full = np.isfinite(sic_clim).all(axis=0)
    with np.errstate(invalid="ignore"):
        mx = np.where(full, np.nanmax(sic_clim, axis=0), np.nan)
        mn = np.where(full, np.nanmin(sic_clim, axis=0), np.nan)
    cand = full & (mx >= thresh) & (mn < thresh)
    jj, ii = np.nonzero(cand)
    for j, i in zip(jj, ii):
        out[j, i] = ice_retreat_month(sic_clim[:, j, i], thresh)
    return out, cand

def region_circular_mean(field, weight, masks, valid=None):
    """Area-weighted CIRCULAR mean of a month-valued field, per region.

    Dates are circular: a plain mean of December and January retreats
    would return July.
    """
    out, frac = {}, {}
    finite = np.isfinite(field)
    base = finite if valid is None else valid
    for name, m in masks.items():
        sel = finite & m
        denom = float(np.sum(weight[m & base])) if (m & base).any() else 0.0
        if not sel.any():
            continue
        w = weight[sel]
        W = float(np.sum(w))
        if W <= 0:
            continue
        th = 2.0 * np.pi * (field[sel] % 12.0) / 12.0
        ang = np.arctan2(float(np.sum(w * np.sin(th))),
                         float(np.sum(w * np.cos(th))))
        out[name] = float((ang * 12.0 / (2.0 * np.pi)) % 12.0)
        frac[name] = (W / denom) if denom > 0 else np.nan
    return out, frac

def region_bimodality(clim, weight, masks, min_a1_rel: float):
    """Area-weighted per-cell A2/A1, over cells with a real annual cycle.

    A2/A1 is a ratio, so it explodes wherever A1 approaches zero. Cells
    whose annual amplitude is a negligible fraction of their mean carry no
    phase information and are excluded rather than allowed to dominate the
    average. The fraction of area that qualifies is returned alongside, so
    a number computed from a sliver of the region can be recognised.
    """
    mean, amp, _ = harmonic_fit_grid(clim, nharm=2)
    a1, a2 = amp[0], amp[1]
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = np.where((mean > 0) & np.isfinite(mean), a1 / mean, np.nan)
        ratio = np.where(a1 > 0, a2 / a1, np.nan)
    good = np.isfinite(rel) & (rel >= min_a1_rel) & np.isfinite(ratio)
    have = np.isfinite(mean)

    twop = per_cell_two_peaks(clim)

    out, frac, bim, two = {}, {}, {}, {}
    for name, m in masks.items():
        sel = good & m
        denom = float(np.sum(weight[m & have])) if (m & have).any() else 0.0
        if not sel.any() or denom <= 0:
            continue
        w = weight[sel]
        W = float(np.sum(w))
        if W <= 0:
            continue
        out[name] = float(np.sum(w * ratio[sel]) / W)
        frac[name] = W / denom
        bim[name] = float(np.sum(w * (ratio[sel] > 0.5)) / W)
        tsel = m & np.isfinite(twop)
        if tsel.any():
            tw = weight[tsel]
            two[name] = float(np.sum(tw * twop[tsel]) / np.sum(tw))
    return out, frac, bim, two

FIELD_SPEC = {
    "chl":    {"surface": True,  "units_from_file": True,  "auto_percent": False},
    "mlotst": {"surface": False, "units_from_file": False, "auto_percent": False},
    "tos":    {"surface": False, "units_from_file": False, "auto_percent": False},
    "siconc": {"surface": False, "units_from_file": False, "auto_percent": True},
}

def metrics_from_fields(fields: dict, regions: dict, period, name: str = "",
                        scales: dict | None = None):
    """Per-region metrics for one dataset, from fields already loaded.

    `fields` maps variable name to an xarray DataArray: any of chl,
    mlotst, tos, siconc. Returns {region: metrics dict}, the same
    structure the streaming path produces, so the plotting functions do
    not know or care which way the data arrived.

    `scales` overrides the unit handling per variable, for a host whose
    own loaders have already normalised the field. Without it, chl is
    rescaled from its units attribute and a field already in mg m-3
    whose attribute is missing would be multiplied by a million.
    """
    mats, grids = {}, {"chl": GridClim(), "siconc": GridClim()}
    for var, da in fields.items():
        if da is None or var not in FIELD_SPEC:
            continue
        spec = FIELD_SPEC[var]
        if scales and var in scales:
            scale = float(scales[var])
            from_file = False
            pct = False
        else:
            from_file = spec["units_from_file"]
            pct = spec["auto_percent"]
            scale = None if (from_file or pct) else 1.0
        try:
            series, _ = series_from_field(
                da, regions, period, scale=scale,
                surface=spec["surface"],
                auto_percent=pct, units_from_file=from_file,
                model=name, grid_clim=grids.get(var),
                valid_range=VARS.get(var, {}).get("valid"))
        except Exception as exc:
            log(f"   {name} {var}: reduction failed: {exc}")
            continue
        if series:
            mats[var] = {r: to_year_month_matrix(v, period)
                         for r, v in series.items()}

    extra = grid_extras(grids["chl"], grids["siconc"], regions,
                        f"{name} " if name else "")
    out = {}
    for region in regions:
        met = metrics_for_region(
            mats.get("chl", {}).get(region),
            mats.get("mlotst", {}).get(region),
            mats.get("siconc", {}).get(region),
            mats.get("tos", {}).get(region))
        met.update(extra.get(region, {}))
        met["lag_bloom_ice"] = circ_diff(met["bloom_phase"],
                                         met["ice_retreat_month"])
        out[region] = met
    return out

def grid_extras(chl_grid, sic_grid, regions, label=""):
    """Per-region quantities that have to come from the grid, not a mean."""
    extras = {r: {} for r in regions}

    if sic_grid is not None and sic_grid.sum is not None:
        clim = sic_grid.clim()
        if clim is not None:
            masks = region_masks(sic_grid.lat2d, sic_grid.lon2d, regions)
            ret, cand = per_cell_retreat(clim)
            have = np.isfinite(clim).all(axis=0)
            months, frac = region_circular_mean(
                ret, sic_grid.weight, masks, valid=have)
            for r, v in months.items():
                extras[r]["ice_retreat_month"] = v
                extras[r]["ice_area_frac"] = frac.get(r, np.nan)
            n = int(np.sum(cand))
            log(f"   {label}ice: {n} cells have a retreat date; "
                f"regions resolved: {sorted(months)}")

    if chl_grid is not None and chl_grid.sum is not None:
        clim = chl_grid.clim()
        if clim is not None:
            masks = region_masks(chl_grid.lat2d, chl_grid.lon2d, regions)
            ratio, frac, bim, two = region_bimodality(
                clim, chl_grid.weight, masks, MIN_A1_REL)
            for r in ratio:
                extras[r]["A2_over_A1"] = ratio[r]
                extras[r]["cells_frac"] = frac.get(r, np.nan)
                extras[r]["bimodal_frac"] = bim.get(r, np.nan)
                extras[r]["two_peak_frac"] = two.get(r, np.nan)

    return extras
