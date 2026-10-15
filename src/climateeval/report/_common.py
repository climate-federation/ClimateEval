"""Backend-agnostic helpers shared by the Plotly and static figure builders."""

from __future__ import annotations

import numpy as np
import pandas as pd

from climateeval.report._style import _DIVERGING_VARS


def _vrange(series: pd.Series) -> tuple[float, float]:
    s = series.dropna()
    if s.empty:
        return 0.0, 1.0
    return float(s.quantile(0.02)), float(s.quantile(0.98))


def _var_label(var_id: str, variables_df: pd.DataFrame) -> tuple[str, str]:
    """Return (long_name, units) for var_id."""
    row = variables_df[variables_df["id"] == var_id]
    long_name = row["long_name"].values[0] if not row.empty else var_id
    units = row["units"].values[0] if not row.empty else ""
    return str(long_name), str(units)


def _data_range(series: pd.Series, var_id: str) -> tuple[float, float]:
    """Return (vmin, vmax); symmetric around zero for diverging variables."""
    vmin, vmax = _vrange(series)
    if var_id in _DIVERGING_VARS:
        vabs = max(abs(vmin), abs(vmax))
        if vabs == 0:
            vabs = 1.0
        return -vabs, vabs
    return vmin, vmax


def _sort_datasets(data_df: pd.DataFrame, datasets: list[str]) -> list[str]:
    """Sort: reference first, other second, to_benchmark last."""
    priority = {"to_benchmark": 0, "reference": 1, "other": 2}

    def key(ds: str) -> int:
        dtype = data_df[data_df["data_id"] == ds]["data_type"].iloc[0]
        return priority.get(dtype, 3)

    return sorted(datasets, key=key)


def _get_ref_dataset(data_df: pd.DataFrame, metrics_df: pd.DataFrame) -> str | None:
    """Return the data_id of the reference dataset, if exactly one exists."""
    # Fall back to the old method for backwards compat / test fixtures.
    if (
        metrics_df is not None
        and not metrics_df.empty
        and "reference_data_id" in metrics_df.columns
    ):
        refs = metrics_df["reference_data_id"].dropna().unique().tolist()
        return refs[0] if len(refs) == 1 else None
    ref_datasets = (
        data_df[data_df["data_type"] == "reference"]["data_id"].unique().tolist()
    )
    return ref_datasets[0] if len(ref_datasets) == 1 else None


def _get_ref_datasets(
    data_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> list[str]:
    """Return every reference data_id, in stable order (may be empty)."""
    if (
        metrics_df is not None
        and not metrics_df.empty
        and "reference_data_id" in metrics_df.columns
    ):
        refs = list(metrics_df["reference_data_id"].dropna().unique().tolist())
        if refs:
            return refs
    return data_df[data_df["data_type"] == "reference"]["data_id"].unique().tolist()


def _combine_pivots(pivots: list[pd.DataFrame]) -> pd.DataFrame | None:
    """Return the element-wise mean of a list of 2-D pivots (NaN-aware).

    Pivots are aligned on the union of their indices/columns before averaging,
    so references on slightly different grids can still be combined.
    """
    pivots = [p for p in pivots if p is not None and not p.empty]
    if not pivots:
        return None
    if len(pivots) == 1:
        return pivots[0]
    idx = pivots[0].index
    cols = pivots[0].columns
    for p in pivots[1:]:
        idx = idx.union(p.index)
        cols = cols.union(p.columns)
    stacked = np.stack(
        [p.reindex(index=idx, columns=cols).values for p in pivots],
    )
    with np.errstate(invalid="ignore"):
        mean_vals = np.nanmean(stacked, axis=0)
    return pd.DataFrame(mean_vals, index=idx, columns=cols)


def _weighted_mean(pivot: pd.DataFrame) -> float:
    """Area (cos-latitude) weighted mean of a lat×lon pivot."""
    weights = np.cos(np.deg2rad(pivot.index.to_numpy(dtype=float)))
    vals = pivot.to_numpy(dtype=float)
    mask = np.isfinite(vals)
    w2d = np.broadcast_to(weights[:, None], vals.shape)
    wsum = (w2d * mask).sum()
    if wsum == 0:
        return float("nan")
    return float(np.nansum(vals * w2d * mask) / wsum)
