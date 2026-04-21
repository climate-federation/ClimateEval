"""Generate a portable self-contained HTML benchmarking report from DuckDB database(s)."""

from __future__ import annotations

import argparse
import html as html_module
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ibis
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Coastline extraction (cartopy, cached)
# ---------------------------------------------------------------------------

_COASTLINES: tuple[list, list] | None = None


def _extract_coastlines(simplify_deg: float = 0.5) -> tuple[list, list]:
    """Return (lons, lats) coastline coords with None breaks between segments."""
    try:
        import cartopy.feature as cfeature
    except ImportError:
        return [], []
    lons: list = []
    lats: list = []
    for geom in cfeature.COASTLINE.geometries():
        parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        for part in parts:
            part = part.simplify(simplify_deg, preserve_topology=False)
            coords = list(part.coords)
            if len(coords) < 2:
                continue
            lons.extend([c[0] for c in coords] + [None])
            lats.extend([c[1] for c in coords] + [None])
    return lons, lats


def _coastlines() -> tuple[list, list]:
    global _COASTLINES
    if _COASTLINES is None:
        _COASTLINES = _extract_coastlines()
    return _COASTLINES


# ---------------------------------------------------------------------------
# Database reading
# ---------------------------------------------------------------------------

_SKIP_DBS = {"main", "memory", "information_schema", "temp", "system"}


def read_database(db_path: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """Return {diagnostic_name: {table_name: DataFrame}} from a DuckDB file."""
    conn = ibis.duckdb.connect(db_path, read_only=True)
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for db_name in conn.list_databases():
        if db_name.lower() in _SKIP_DBS:
            continue
        tables: dict[str, pd.DataFrame] = {}
        try:
            for table_name in conn.list_tables(database=db_name):
                tables[table_name] = conn.table(
                    table_name,
                    database=db_name,
                ).to_pandas()
        except Exception:
            continue
        if tables:
            result[db_name] = tables
    return result


def get_var_cols(data_df: pd.DataFrame) -> list[str]:
    """Return variable (non-coordinate, non-metadata) column names."""
    skip = {
        "data_id",
        "data_type",
        "latitude",
        "longitude",
        "time",
        "air_pressure",
        "month_number",
    }
    return [c for c in data_df.columns if c not in skip]


def detect_diag_type(columns: list[str]) -> str:
    cols = set(columns)
    if "latitude" in cols and "longitude" in cols:
        return "Map"
    if "air_pressure" in cols and "latitude" in cols:
        return "ZonalProfile"
    if "latitude" in cols:
        return "ZonalLine"
    if "time" in cols:
        return "TimeSeries"
    if "month_number" in cols:
        return "AnnualCycle"
    return "Unknown"


# ---------------------------------------------------------------------------
# Shared plot helpers
# ---------------------------------------------------------------------------

# Per-variable colormaps for 2-D fields (Map / ZonalProfile).
# Mirrors the ICONEval recipe_templates cmap choices so plots match the
# ESMValTool monitor diagnostics.
_VAR_COLORSCALE: dict[str, str] = {
    # Temperature (sequential, "plasma" in ICONEval basics/tuning recipes)
    "tas": "Plasma",
    "ta": "Plasma",
    "ta200": "Plasma",
    "ta800": "Plasma",
    # Precipitation (ICONEval basics_maps: Greens)
    "pr": "Greens",
    # Clouds / water path / humidity (ICONEval clouds_maps / basics_profiles: Blues)
    "clivi": "Blues",
    "clt": "Blues",
    "clwvi": "Blues",
    "lwp": "Blues",
    "prw": "Blues",
    "hus": "Blues",
    "hus400": "Blues",
    # Cloud radiative effects (ICONEval clouds_maps: bwr, centered)
    "lwcre": "RdBu_r",
    "swcre": "RdBu_r",
    # Eastward wind (ICONEval basics_profiles: PuOr, centered)
    "ua": "PuOr",
    "ua200": "PuOr",
    "ua800": "PuOr",
}

# Variables whose fields should be displayed with a symmetric range around zero
# (diverging colormaps rely on this to keep the "zero" colour in the middle).
_DIVERGING_VARS: set[str] = {
    "lwcre",
    "swcre",
    "ua",
    "ua200",
    "ua800",
}

# Per-variable override for the bias (model − reference) colormap. Falls back
# to _DEFAULT_BIAS_COLORSCALE otherwise.
_VAR_BIAS_COLORSCALE: dict[str, str] = {
    "pr": "BrBG",  # ICONEval basics_maps uses BrBG for precipitation bias
}

_DEFAULT_COLORSCALE = "RdBu_r"
_DEFAULT_BIAS_COLORSCALE = "RdBu_r"


def _colorscale_for(var_id: str) -> str:
    return _VAR_COLORSCALE.get(var_id, _DEFAULT_COLORSCALE)


def _bias_colorscale_for(var_id: str) -> str:
    return _VAR_BIAS_COLORSCALE.get(var_id, _DEFAULT_BIAS_COLORSCALE)


def _data_range(series: pd.Series, var_id: str) -> tuple[float, float]:
    """Return (vmin, vmax); symmetric around zero for diverging variables."""
    vmin, vmax = _vrange(series)
    if var_id in _DIVERGING_VARS:
        vabs = max(abs(vmin), abs(vmax))
        if vabs == 0:
            vabs = 1.0
        return -vabs, vabs
    return vmin, vmax


_PLOT_FONT = {
    "family": "'Inter', 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif",
    "size": 12,
}
_AXIS_STYLE = {
    "showgrid": True,
    "gridcolor": "rgba(180,180,180,0.25)",
    "zeroline": False,
    "linecolor": "#ccc",
    "linewidth": 1,
    "mirror": True,
}

# Seaborn color palette for to_benchmark data
_SEABORN_COLORS = [
    "#4C72B0",
    "#C44E52",
    "#55A868",
    "#DD8452",
    "#8172B3",
    "#CF4458",
    "#4B4B83",
    "#D65F5F",
    "#6B6B6B",
]


def _get_line_properties(dtype: str, color_idx: int) -> dict:
    """Return line styling for 1D plots based on data type.

    Args:
        dtype: One of "reference", "to_benchmark", or "other"
        color_idx: Index into seaborn color palette (for to_benchmark only)

    Returns
    -------
        dict with color, dash, show_legend, width
    """
    if dtype == "reference":
        return {
            "color": "black",
            "dash": "solid",
            "show_legend": True,
            "width": 2,
        }
    if dtype == "to_benchmark":
        return {
            "color": _SEABORN_COLORS[color_idx % len(_SEABORN_COLORS)],
            "dash": "solid",
            "show_legend": True,
            "width": 2,
        }
    # "other"
    return {
        "color": "#888",
        "dash": "dash",
        "show_legend": False,
        "width": 1.5,
    }


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
        refs = [r for r in metrics_df["reference_data_id"].dropna().unique().tolist()]
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


def _base_layout(**kwargs: Any) -> dict:
    base = {
        "font": _PLOT_FONT,
        "paper_bgcolor": "white",
        "plot_bgcolor": "#fafbfc",
        "legend": {
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "#ddd",
            "borderwidth": 1,
            "font": {"size": 11},
        },
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------


def build_map_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, go.Figure]:
    """Heatmap maps with coastlines; reference + bias (model − reference) layout when possible."""
    coast_lons, coast_lats = _coastlines()
    figures: dict[str, go.Figure] = {}
    var_cols = get_var_cols(data_df)
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())
    ref_datasets_all = _get_ref_datasets(data_df, metrics_df)

    for var_id in var_cols:
        long_name, units = _var_label(var_id, variables_df)
        vmin, vmax = _data_range(data_df[var_id], var_id)
        cs = _colorscale_for(var_id)
        bias_cs = _bias_colorscale_for(var_id)

        valid_datasets = [
            ds
            for ds in all_datasets
            if not data_df[data_df["data_id"] == ds][var_id].isna().all()
        ]
        if not valid_datasets:
            continue

        def _pivot_for(dataset: str) -> pd.DataFrame | None:
            sub = data_df[data_df["data_id"] == dataset]
            sv = sub[["latitude", "longitude", var_id]].dropna(subset=[var_id]).copy()
            if sv.empty:
                return None
            sv["lon"] = sv["longitude"].apply(lambda x: x - 360 if x > 180 else x)
            p = (
                sv.groupby(["latitude", "lon"], as_index=False)[var_id]
                .mean()
                .pivot(index="latitude", columns="lon", values=var_id)
                .sort_index(axis=1)
            )
            return p if not p.empty else None

        valid_refs = [r for r in ref_datasets_all if r in valid_datasets]
        ref_pivots_list = [(r, _pivot_for(r)) for r in valid_refs]
        ref_pivots_list = [(r, p) for r, p in ref_pivots_list if p is not None]
        if ref_pivots_list:
            ref_pivot = _combine_pivots([p for _, p in ref_pivots_list])
            ref_label = (
                ref_pivots_list[0][0] if len(ref_pivots_list) == 1 else "Multi-obs mean"
            )
        else:
            ref_pivot = None
            ref_label = None
        non_ref = [ds for ds in valid_datasets if ds not in valid_refs]
        has_bias = ref_pivot is not None and bool(non_ref)

        if has_bias:
            ncols = 3
            n_model_rows = len(non_ref)
            show_ref_section = len(valid_refs) > 1
            n_ref_rows = (
                (len(valid_refs) + ncols - 1) // ncols if show_ref_section else 0
            )
            nrows = n_ref_rows + n_model_rows
            titles: list[str] = []
            for r in range(n_ref_rows):
                for c in range(ncols):
                    idx = r * ncols + c
                    if idx < len(valid_refs):
                        titles.append(f"<b>{valid_refs[idx]}</b>")
            for ds in non_ref:
                titles += [
                    f"<b>{ref_label}</b>",
                    f"<b>{ds}</b>",
                    f"<b>{ds}</b> \u2212 {ref_label}",
                ]
            # Compute shared symmetric bias range across all non-ref datasets
            bias_vals: list[float] = []
            for ds in non_ref:
                mp = _pivot_for(ds)
                if mp is not None:
                    bv = mp.subtract(ref_pivot).values
                    bias_vals.extend(bv[np.isfinite(bv)].tolist())
            arr = np.array(bias_vals) if bias_vals else np.array([0.0, 1.0])
            bias_abs = float(
                max(
                    abs(float(np.percentile(arr, 2))),
                    abs(float(np.percentile(arr, 98))),
                ),
            )
            if bias_abs == 0:
                bias_abs = 1.0
        else:
            ncols = min(len(valid_datasets), 3)
            nrows = max(1, (len(valid_datasets) + ncols - 1) // ncols)
            titles = [str(ds) for ds in valid_datasets]

        if has_bias:
            specs: list[list[dict | None]] = []
            for r in range(n_ref_rows):
                row_specs: list[dict | None] = []
                for c in range(ncols):
                    idx = r * ncols + c
                    row_specs.append({} if idx < len(valid_refs) else None)
                specs.append(row_specs)
            for _ in range(n_model_rows):
                specs.append([{}, {}, {}])
        else:
            specs = None
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=titles,
            specs=specs,
            horizontal_spacing=0.05,
            vertical_spacing=(min(0.08, 0.5 / (nrows - 1)) if nrows > 1 else 0),
        )

        lon_ticks = {
            "tickvals": [-120, -60, 0, 60, 120],
            "ticktext": ["120°W", "60°W", "0°", "60°E", "120°E"],
        }
        lat_ticks = {
            "tickvals": [-60, -30, 0, 30, 60],
            "ticktext": ["60°S", "30°S", "0°", "30°N", "60°N"],
        }

        def _add_panel(
            pivot: pd.DataFrame,
            row: int,
            col: int,
            colorscale: str,
            zmin: float,
            zmax: float,
            cb_dict: dict | None,
            name: str,
        ) -> None:
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    colorscale=colorscale,
                    zmin=zmin,
                    zmax=zmax,
                    colorbar=cb_dict,
                    showscale=cb_dict is not None,
                    name=name,
                    hovertemplate=(
                        "lon %{x:.1f}°  lat %{y:.1f}°"
                        f"<br><b>{var_id}</b>: %{{z:.4g}} {units}"
                        "<extra>" + str(name) + "</extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            if coast_lons:
                fig.add_trace(
                    go.Scatter(
                        x=coast_lons,
                        y=coast_lats,
                        mode="lines",
                        line={"color": "rgba(20,20,20,0.80)", "width": 0.6},
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=col,
                )
            fig.update_xaxes(
                range=[-180, 180],
                showgrid=True,
                gridcolor="rgba(180,180,180,0.25)",
                zeroline=True,
                zerolinecolor="rgba(150,150,150,0.4)",
                zerolinewidth=1,
                linecolor="#bbb",
                linewidth=1,
                mirror=True,
                tickfont={"size": 9},
                title_text="Longitude" if row == nrows else "",
                **lon_ticks,
                row=row,
                col=col,
            )
            fig.update_yaxes(
                range=[-90, 90],
                showgrid=True,
                gridcolor="rgba(180,180,180,0.25)",
                zeroline=True,
                zerolinecolor="rgba(150,150,150,0.4)",
                zerolinewidth=1,
                linecolor="#bbb",
                linewidth=1,
                mirror=True,
                tickfont={"size": 9},
                title_text="Latitude" if col == 1 else "",
                **lat_ticks,
                row=row,
                col=col,
            )

        if has_bias:
            ref_cb_shown = False
            bias_cb_shown = False
            if show_ref_section:
                for idx, r_name in enumerate(valid_refs):
                    rrow = idx // ncols + 1
                    rcol = idx % ncols + 1
                    piv = _pivot_for(r_name)
                    if piv is None:
                        continue
                    r_cb = (
                        {
                            "title": {"text": units, "side": "right"},
                            "thickness": 14,
                            "len": 0.85,
                            "x": 1.01,
                            "tickfont": {"size": 10},
                            "outlinewidth": 0,
                        }
                        if not ref_cb_shown
                        else None
                    )
                    _add_panel(piv, rrow, rcol, cs, vmin, vmax, r_cb, r_name)
                    ref_cb_shown = True
            for i, ds in enumerate(non_ref):
                row = n_ref_rows + i + 1
                ref_cb = (
                    {
                        "title": {"text": units, "side": "right"},
                        "thickness": 14,
                        "len": 0.85,
                        "x": 1.01,
                        "tickfont": {"size": 10},
                        "outlinewidth": 0,
                    }
                    if not ref_cb_shown
                    else None
                )
                ref_name = "" if ref_label is None else ref_label
                _add_panel(ref_pivot, row, 1, cs, vmin, vmax, ref_cb, ref_name)
                ref_cb_shown = True

                mp = _pivot_for(ds)
                if mp is None:
                    continue
                _add_panel(mp, row, 2, cs, vmin, vmax, None, str(ds))

                bp = mp.subtract(ref_pivot).dropna(how="all").dropna(axis=1, how="all")
                if bp.empty:
                    continue
                bias_cb = (
                    {
                        "title": {"text": f"\u0394{units}", "side": "right"},
                        "thickness": 14,
                        "len": 0.85,
                        "x": 1.09,
                        "tickfont": {"size": 10},
                        "outlinewidth": 0,
                    }
                    if not bias_cb_shown
                    else None
                )
                _add_panel(
                    bp,
                    row,
                    3,
                    bias_cs,
                    -bias_abs,
                    bias_abs,
                    bias_cb,
                    f"{ds} \u2212 {ref_label}",
                )
                bias_cb_shown = True
        else:
            show_cb = True
            for i, ds in enumerate(valid_datasets):
                row = i // ncols + 1
                col = i % ncols + 1
                piv = _pivot_for(ds)
                if piv is None:
                    continue
                cb = (
                    {
                        "title": {"text": units, "side": "right"},
                        "thickness": 14,
                        "len": 0.85,
                        "x": 1.01,
                        "tickfont": {"size": 10},
                        "outlinewidth": 0,
                    }
                    if show_cb
                    else None
                )
                _add_panel(piv, row, col, cs, vmin, vmax, cb, ds)
                show_cb = False

        title_str = f"<b>{long_name}</b>" + (f"  [{units}]" if units else "")
        fig.update_layout(
            **_base_layout(
                title={
                    "text": title_str,
                    "font": {"size": 14},
                    "x": 0.02,
                    "xanchor": "left",
                },
                plot_bgcolor="#dce9f5",
                height=max(450 * nrows + 140, 560),
                margin={"t": 80, "b": 50, "l": 60, "r": 150 if has_bias else 90},
            ),
        )
        ref_title_texts = {f"<b>{r}</b>" for r in valid_refs} if has_bias else set()
        ref_title_texts.add(f"<b>{ref_label}</b>" if has_bias else "")
        for ann in fig.layout.annotations:
            if has_bias and ann.text in ref_title_texts:
                ann.font = {"size": 11, "color": "#a07000"}
                ann.bgcolor = "rgba(192,134,10,0.08)"
                ann.bordercolor = "#c0860a"
                ann.borderwidth = 1
                ann.borderpad = 3
            else:
                ann.font = {"size": 11, "color": "#333"}

        figures[var_id] = fig

    return figures


def build_timeseries_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, go.Figure]:
    """Annual time series, one figure per variable, all datasets overlaid.

    When a single reference dataset is available, a second panel shows the bias
    (dataset − reference) over the overlapping time range for every non-reference
    dataset.
    """
    figures: dict[str, go.Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())
    ref_dataset = _get_ref_dataset(data_df, metrics_df)

    def _series_for(dataset: str, var_id: str) -> pd.Series | None:
        grp = data_df[data_df["data_id"] == dataset]
        sub = grp[["time", var_id]].dropna(subset=[var_id]).sort_values("time")
        if sub.empty:
            return None
        return sub.set_index("time")[var_id]

    for var_id in get_var_cols(data_df):
        if data_df[var_id].isna().all():
            continue
        long_name, units = _var_label(var_id, variables_df)

        ref_series = _series_for(ref_dataset, var_id) if ref_dataset else None
        non_ref = [d for d in all_datasets if d != ref_dataset]
        has_bias = ref_series is not None and any(
            _series_for(d, var_id) is not None for d in non_ref
        )

        if has_bias:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.65, 0.35],
                subplot_titles=("", f"Bias (dataset − {ref_dataset})"),
            )
        else:
            fig = go.Figure()

        color_idx = 0
        for dataset in all_datasets:
            series = _series_for(dataset, var_id)
            if series is None:
                continue
            grp = data_df[data_df["data_id"] == dataset]
            dtype = grp["data_type"].iloc[0]
            props = _get_line_properties(dtype, color_idx)
            if dtype == "to_benchmark":
                color_idx += 1
            show_legend = props.pop("show_legend")

            trace = go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=str(dataset),
                line=props,
                showlegend=show_legend,
                legendgroup=str(dataset),
            )
            if has_bias:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

        if has_bias:
            color_idx = 0
            for dataset in all_datasets:
                if dataset == ref_dataset:
                    continue
                series = _series_for(dataset, var_id)
                if series is None:
                    continue
                grp = data_df[data_df["data_id"] == dataset]
                dtype = grp["data_type"].iloc[0]
                props = _get_line_properties(dtype, color_idx)
                if dtype == "to_benchmark":
                    color_idx += 1
                props.pop("show_legend")

                common = series.index.intersection(ref_series.index)
                if common.empty:
                    continue
                bias = series.loc[common] - ref_series.loc[common]
                bias = bias.dropna()
                if bias.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=bias.index,
                        y=bias.values,
                        mode="lines",
                        name=str(dataset),
                        line=props,
                        showlegend=False,
                        legendgroup=str(dataset),
                        hovertemplate=(
                            f"<b>{dataset} − {ref_dataset}</b><br>"
                            f"%{{x|%Y-%m}}<br>"
                            f"Δ{var_id}: %{{y:.4g}} {units}"
                            "<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=1,
                )
            fig.add_hline(
                y=0,
                line={"color": "rgba(0,0,0,0.45)", "width": 1, "dash": "dot"},
                row=2,
                col=1,
            )

        title_str = f"<b>{long_name}</b>" + (f"  [{units}]" if units else "")
        layout_kwargs = dict(
            title={
                "text": title_str,
                "font": {"size": 14},
                "x": 0.02,
                "xanchor": "left",
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
                "bgcolor": "rgba(255,255,255,0.85)",
                "bordercolor": "#ddd",
                "borderwidth": 1,
                "font": {"size": 11},
            },
            margin={"t": 80, "b": 60, "l": 70, "r": 30},
        )
        if has_bias:
            fig.update_xaxes(**_AXIS_STYLE, row=1, col=1)
            fig.update_xaxes(title_text="Date", **_AXIS_STYLE, row=2, col=1)
            fig.update_yaxes(title_text=units, **_AXIS_STYLE, row=1, col=1)
            fig.update_yaxes(title_text=f"Δ{units}", **_AXIS_STYLE, row=2, col=1)
            layout_kwargs["height"] = 580
        else:
            layout_kwargs["xaxis"] = dict(title_text="Date", **_AXIS_STYLE)
            layout_kwargs["yaxis"] = dict(title_text=units, **_AXIS_STYLE)
            layout_kwargs["height"] = 440
        fig.update_layout(**_base_layout(**layout_kwargs))
        figures[var_id] = fig

    return figures


def build_annual_cycle_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, go.Figure]:
    """Monthly climatology (annual cycle), one figure per variable, all datasets overlaid.

    When a single reference dataset is available, a second panel shows the bias
    (dataset − reference) for every non-reference dataset.
    """
    figures: dict[str, go.Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())
    ref_dataset = _get_ref_dataset(data_df, metrics_df)

    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    def _series_for(dataset: str, var_id: str) -> pd.Series | None:
        grp = data_df[data_df["data_id"] == dataset]
        sub = grp[["month_number", var_id]].dropna(subset=[var_id])
        if sub.empty:
            return None
        return sub.groupby("month_number")[var_id].mean().sort_index()

    for var_id in get_var_cols(data_df):
        if data_df[var_id].isna().all():
            continue
        long_name, units = _var_label(var_id, variables_df)

        ref_series = _series_for(ref_dataset, var_id) if ref_dataset else None
        non_ref = [d for d in all_datasets if d != ref_dataset]
        has_bias = ref_series is not None and any(
            _series_for(d, var_id) is not None for d in non_ref
        )

        if has_bias:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.65, 0.35],
                subplot_titles=("", f"Bias (dataset − {ref_dataset})"),
            )
        else:
            fig = go.Figure()

        color_idx = 0
        for dataset in all_datasets:
            series = _series_for(dataset, var_id)
            if series is None:
                continue
            grp = data_df[data_df["data_id"] == dataset]
            dtype = grp["data_type"].iloc[0]
            props = _get_line_properties(dtype, color_idx)
            if dtype == "to_benchmark":
                color_idx += 1
            show_legend = props.pop("show_legend")

            trace = go.Scatter(
                x=series.index.tolist(),
                y=series.values.tolist(),
                mode="lines+markers",
                name=str(dataset),
                line=props,
                marker={"size": 6},
                showlegend=show_legend,
                legendgroup=str(dataset),
                hovertemplate=(
                    f"<b>{dataset}</b><br>"
                    f"Month: %{{x:.0f}}<br>"
                    f"{var_id}: %{{y:.4g}} {units}"
                    "<extra></extra>"
                ),
            )
            if has_bias:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

        if has_bias:
            color_idx = 0
            for dataset in all_datasets:
                if dataset == ref_dataset:
                    continue
                series = _series_for(dataset, var_id)
                if series is None:
                    continue
                grp = data_df[data_df["data_id"] == dataset]
                dtype = grp["data_type"].iloc[0]
                props = _get_line_properties(dtype, color_idx)
                if dtype == "to_benchmark":
                    color_idx += 1
                props.pop("show_legend")

                aligned = series.reindex(ref_series.index)
                bias = (aligned - ref_series).dropna()
                if bias.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=bias.index.tolist(),
                        y=bias.values.tolist(),
                        mode="lines+markers",
                        name=str(dataset),
                        line=props,
                        marker={"size": 5},
                        showlegend=False,
                        legendgroup=str(dataset),
                        hovertemplate=(
                            f"<b>{dataset} − {ref_dataset}</b><br>"
                            f"Month: %{{x:.0f}}<br>"
                            f"Δ{var_id}: %{{y:.4g}} {units}"
                            "<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=1,
                )
            fig.add_hline(
                y=0,
                line={"color": "rgba(0,0,0,0.45)", "width": 1, "dash": "dot"},
                row=2,
                col=1,
            )

        x_kwargs = {
            "title_text": "Month",
            "tickvals": list(range(1, 13)),
            "ticktext": month_names,
            **_AXIS_STYLE,
        }
        title_str = f"<b>{long_name}</b>" + (f"  [{units}]" if units else "")
        layout_kwargs = dict(
            title={
                "text": title_str,
                "font": {"size": 14},
                "x": 0.02,
                "xanchor": "left",
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
                "bgcolor": "rgba(255,255,255,0.85)",
                "bordercolor": "#ddd",
                "borderwidth": 1,
                "font": {"size": 11},
            },
            margin={"t": 80, "b": 60, "l": 70, "r": 30},
        )
        if has_bias:
            fig.update_xaxes(
                **{k: v for k, v in x_kwargs.items() if k != "title_text"},
                row=1,
                col=1,
            )
            fig.update_xaxes(**x_kwargs, row=2, col=1)
            fig.update_yaxes(title_text=units, **_AXIS_STYLE, row=1, col=1)
            fig.update_yaxes(title_text=f"Δ{units}", **_AXIS_STYLE, row=2, col=1)
            layout_kwargs["height"] = 560
        else:
            layout_kwargs["xaxis"] = x_kwargs
            layout_kwargs["yaxis"] = dict(title_text=units, **_AXIS_STYLE)
            layout_kwargs["height"] = 420
        fig.update_layout(**_base_layout(**layout_kwargs))
        figures[var_id] = fig

    return figures


def build_zonal_line_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, go.Figure]:
    """Zonal mean line plots, one figure per variable, all datasets overlaid.

    When a single reference dataset is available, a second panel shows the bias
    (dataset − reference) as a function of latitude for every non-reference
    dataset.
    """
    figures: dict[str, go.Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())
    ref_dataset = _get_ref_dataset(data_df, metrics_df)

    def _series_for(dataset: str, var_id: str) -> pd.Series | None:
        grp = data_df[data_df["data_id"] == dataset]
        sub = grp[["latitude", var_id]].dropna(subset=[var_id])
        if sub.empty:
            return None
        return sub.groupby("latitude")[var_id].mean().sort_index()

    lat_tickvals = [-90, -60, -30, 0, 30, 60, 90]
    lat_ticktext = ["90°S", "60°S", "30°S", "0°", "30°N", "60°N", "90°N"]

    for var_id in get_var_cols(data_df):
        if data_df[var_id].isna().all():
            continue
        long_name, units = _var_label(var_id, variables_df)

        ref_series = _series_for(ref_dataset, var_id) if ref_dataset else None
        non_ref = [d for d in all_datasets if d != ref_dataset]
        has_bias = ref_series is not None and any(
            _series_for(d, var_id) is not None for d in non_ref
        )

        if has_bias:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.65, 0.35],
                subplot_titles=("", f"Bias (dataset − {ref_dataset})"),
            )
        else:
            fig = go.Figure()

        color_idx = 0
        for dataset in all_datasets:
            series = _series_for(dataset, var_id)
            if series is None:
                continue
            grp = data_df[data_df["data_id"] == dataset]
            dtype = grp["data_type"].iloc[0]
            props = _get_line_properties(dtype, color_idx)
            if dtype == "to_benchmark":
                color_idx += 1
            show_legend = props.pop("show_legend")

            trace = go.Scatter(
                x=series.index.tolist(),
                y=series.values.tolist(),
                mode="lines",
                name=str(dataset),
                line=props,
                showlegend=show_legend,
                legendgroup=str(dataset),
                hovertemplate=(
                    f"<b>{dataset}</b><br>lat: %{{x:.1f}}°<br>"
                    f"{var_id}: %{{y:.4g}} {units}<extra></extra>"
                ),
            )
            if has_bias:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

        for lat_ref, label in [(0, "Equator"), (23.5, "N Tropic"), (-23.5, "S Tropic")]:
            line_kwargs = {
                "x": lat_ref,
                "line": {"color": "rgba(150,150,150,0.45)", "width": 1, "dash": "dot"},
            }
            if has_bias:
                fig.add_vline(
                    row=1,
                    col=1,
                    annotation={
                        "text": label,
                        "font": {"size": 8, "color": "#aaa"},
                        "y": 0.02,
                    },
                    **line_kwargs,
                )
                fig.add_vline(row=2, col=1, **line_kwargs)
            else:
                fig.add_vline(
                    annotation={
                        "text": label,
                        "font": {"size": 8, "color": "#aaa"},
                        "y": 0.02,
                    },
                    **line_kwargs,
                )

        if has_bias:
            color_idx = 0
            for dataset in all_datasets:
                if dataset == ref_dataset:
                    continue
                series = _series_for(dataset, var_id)
                if series is None:
                    continue
                grp = data_df[data_df["data_id"] == dataset]
                dtype = grp["data_type"].iloc[0]
                props = _get_line_properties(dtype, color_idx)
                if dtype == "to_benchmark":
                    color_idx += 1
                props.pop("show_legend")

                common = series.index.intersection(ref_series.index)
                if common.empty:
                    continue
                bias = (series.loc[common] - ref_series.loc[common]).dropna()
                if bias.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=bias.index.tolist(),
                        y=bias.values.tolist(),
                        mode="lines",
                        name=str(dataset),
                        line=props,
                        showlegend=False,
                        legendgroup=str(dataset),
                        hovertemplate=(
                            f"<b>{dataset} − {ref_dataset}</b><br>"
                            f"lat: %{{x:.1f}}°<br>"
                            f"Δ{var_id}: %{{y:.4g}} {units}"
                            "<extra></extra>"
                        ),
                    ),
                    row=2,
                    col=1,
                )
            fig.add_hline(
                y=0,
                line={"color": "rgba(0,0,0,0.45)", "width": 1, "dash": "dot"},
                row=2,
                col=1,
            )

        title_str = f"<b>{long_name}</b>" + (f"  [{units}]" if units else "")
        layout_kwargs = dict(
            title={
                "text": title_str,
                "font": {"size": 14},
                "x": 0.02,
                "xanchor": "left",
            },
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
                "bgcolor": "rgba(255,255,255,0.85)",
                "bordercolor": "#ddd",
                "borderwidth": 1,
                "font": {"size": 11},
            },
            margin={"t": 80, "b": 60, "l": 70, "r": 30},
        )
        x_kwargs_full = dict(
            title_text="Latitude (°)",
            tickvals=lat_tickvals,
            ticktext=lat_ticktext,
            **_AXIS_STYLE,
        )
        if has_bias:
            fig.update_xaxes(
                tickvals=lat_tickvals,
                ticktext=lat_ticktext,
                **_AXIS_STYLE,
                row=1,
                col=1,
            )
            fig.update_xaxes(**x_kwargs_full, row=2, col=1)
            fig.update_yaxes(title_text=units, **_AXIS_STYLE, row=1, col=1)
            fig.update_yaxes(title_text=f"Δ{units}", **_AXIS_STYLE, row=2, col=1)
            layout_kwargs["height"] = 560
        else:
            layout_kwargs["xaxis"] = x_kwargs_full
            layout_kwargs["yaxis"] = dict(title_text=units, **_AXIS_STYLE)
            layout_kwargs["height"] = 420
        fig.update_layout(**_base_layout(**layout_kwargs))
        figures[var_id] = fig

    return figures


def build_zonal_profile_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, go.Figure]:
    """Zonal mean profile heatmaps (pressure × latitude); reference + bias layout when possible."""
    figures: dict[str, go.Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())
    ref_datasets_all = _get_ref_datasets(data_df, metrics_df)

    for var_id in get_var_cols(data_df):
        long_name, units = _var_label(var_id, variables_df)
        vmin, vmax = _data_range(data_df[var_id], var_id)
        cs = _colorscale_for(var_id)
        bias_cs = _bias_colorscale_for(var_id)

        valid_datasets = [
            ds
            for ds in all_datasets
            if not data_df[data_df["data_id"] == ds][var_id].isna().all()
        ]
        if not valid_datasets:
            continue

        plev_ticks = [1000, 700, 500, 300, 200, 100, 50, 20, 10]
        lat_ticks = {
            "tickvals": [-60, -30, 0, 30, 60],
            "ticktext": ["60°S", "30°S", "0°", "30°N", "60°N"],
        }

        def _pivot_for(dataset: str) -> pd.DataFrame | None:
            sub = data_df[data_df["data_id"] == dataset]
            try:
                p = sub.pivot_table(
                    index="air_pressure",
                    columns="latitude",
                    values=var_id,
                    aggfunc="mean",
                )
                p = p.dropna(how="all").dropna(axis=1, how="all")
            except Exception:
                return None
            if p.empty:
                return None
            p.index = p.index / 100
            return p

        valid_refs = [r for r in ref_datasets_all if r in valid_datasets]
        ref_pivots_list = [(r, _pivot_for(r)) for r in valid_refs]
        ref_pivots_list = [(r, p) for r, p in ref_pivots_list if p is not None]
        if ref_pivots_list:
            ref_pivot = _combine_pivots([p for _, p in ref_pivots_list])
            ref_label = (
                ref_pivots_list[0][0] if len(ref_pivots_list) == 1 else "Multi-obs mean"
            )
        else:
            ref_pivot = None
            ref_label = None
        non_ref = [ds for ds in valid_datasets if ds not in valid_refs]
        has_bias = ref_pivot is not None and bool(non_ref)

        if has_bias:
            nrows, ncols = len(non_ref), 2
            titles: list[str] = []
            for ds in non_ref:
                titles += [f"<b>{ref_label}</b>", f"<b>{ds}</b> \u2212 {ref_label}"]
            bias_vals: list[float] = []
            for ds in non_ref:
                mp = _pivot_for(ds)
                if mp is not None:
                    bv = mp.subtract(ref_pivot).values
                    bias_vals.extend(bv[np.isfinite(bv)].tolist())
            arr = np.array(bias_vals) if bias_vals else np.array([0.0, 1.0])
            bias_abs = float(
                max(
                    abs(float(np.percentile(arr, 2))),
                    abs(float(np.percentile(arr, 98))),
                ),
            )
            if bias_abs == 0:
                bias_abs = 1.0
        else:
            ncols = min(len(valid_datasets), 3)
            nrows = max(1, (len(valid_datasets) + ncols - 1) // ncols)
            titles = [str(ds) for ds in valid_datasets]

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=titles,
            vertical_spacing=0.12,
            horizontal_spacing=0.07,
        )

        def _add_panel(
            pivot: pd.DataFrame,
            row: int,
            col: int,
            colorscale: str,
            zmin: float,
            zmax: float,
            cb_dict: dict | None,
            name: str,
        ) -> None:
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    colorscale=colorscale,
                    zmin=zmin,
                    zmax=zmax,
                    colorbar=cb_dict,
                    showscale=cb_dict is not None,
                    name=name,
                    hovertemplate=(
                        "lat: %{x:.1f}°<br>pressure: %{y:.0f} hPa"
                        f"<br><b>{var_id}</b>: %{{z:.4g}} {units}"
                        "<extra>" + str(name) + "</extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            fig.update_yaxes(
                type="log",
                autorange="reversed",
                tickvals=plev_ticks,
                ticktext=[str(p) for p in plev_ticks],
                title_text="Pressure (hPa)" if col == 1 else "",
                **_AXIS_STYLE,
                row=row,
                col=col,
            )
            fig.update_xaxes(
                title_text="Latitude" if row == nrows else "",
                **lat_ticks,
                **_AXIS_STYLE,
                row=row,
                col=col,
            )

        if has_bias:
            ref_cb_shown = False
            bias_cb_shown = False
            for i, ds in enumerate(non_ref):
                row = i + 1
                ref_cb = (
                    {
                        "title": {"text": units, "side": "right"},
                        "thickness": 14,
                        "len": 0.85,
                        "x": 1.01,
                        "tickfont": {"size": 10},
                        "outlinewidth": 0,
                    }
                    if not ref_cb_shown
                    else None
                )
                ref_name = "" if ref_label is None else ref_label
                _add_panel(ref_pivot, row, 1, cs, vmin, vmax, ref_cb, ref_name)
                ref_cb_shown = True

                mp = _pivot_for(ds)
                if mp is None:
                    continue
                bp = mp.subtract(ref_pivot).dropna(how="all").dropna(axis=1, how="all")
                if bp.empty:
                    continue
                bias_cb = (
                    {
                        "title": {"text": f"\u0394{units}", "side": "right"},
                        "thickness": 14,
                        "len": 0.85,
                        "x": 1.09,
                        "tickfont": {"size": 10},
                        "outlinewidth": 0,
                    }
                    if not bias_cb_shown
                    else None
                )
                _add_panel(
                    bp,
                    row,
                    2,
                    bias_cs,
                    -bias_abs,
                    bias_abs,
                    bias_cb,
                    f"{ds} \u2212 {ref_label}",
                )
                bias_cb_shown = True
        else:
            show_cb = True
            for i, ds in enumerate(valid_datasets):
                row = i // ncols + 1
                col = i % ncols + 1
                piv = _pivot_for(ds)
                if piv is None:
                    continue
                cb = (
                    {
                        "title": {"text": units, "side": "right"},
                        "thickness": 14,
                        "len": 0.85,
                        "x": 1.01,
                        "tickfont": {"size": 10},
                        "outlinewidth": 0,
                    }
                    if show_cb
                    else None
                )
                _add_panel(piv, row, col, cs, vmin, vmax, cb, ds)
                show_cb = False

        title_str = f"<b>{long_name}</b>" + (f"  [{units}]" if units else "")
        fig.update_layout(
            **_base_layout(
                title={
                    "text": title_str,
                    "font": {"size": 14},
                    "x": 0.02,
                    "xanchor": "left",
                },
                height=max(400 * nrows + 120, 520),
                margin={"t": 80, "b": 60, "l": 80, "r": 150 if has_bias else 90},
            ),
        )
        for ann in fig.layout.annotations:
            if (
                has_bias
                and ann.text
                and ref_label in ann.text
                and "\u2212" not in ann.text
            ):
                ann.font = {"size": 11, "color": "#a07000"}
                ann.bgcolor = "rgba(192,134,10,0.08)"
                ann.bordercolor = "#c0860a"
                ann.borderwidth = 1
                ann.borderpad = 3
            else:
                ann.font = {"size": 11, "color": "#333"}
        figures[var_id] = fig

    return figures


_FIGURE_BUILDERS = {
    "Map": build_map_figures,
    "TimeSeries": build_timeseries_figures,
    "AnnualCycle": build_annual_cycle_figures,
    "ZonalLine": build_zonal_line_figures,
    "ZonalProfile": build_zonal_profile_figures,
}

_DIAG_LABELS = {
    "Map": "Map",
    "TimeSeries": "Time Series",
    "AnnualCycle": "Annual Cycle",
    "ZonalLine": "Zonal Mean",
    "ZonalProfile": "Zonal Profile",
}

_DIAG_ICONS = {
    "Map": "🗺",
    "TimeSeries": "📈",
    "AnnualCycle": "📅",
    "ZonalLine": "〰",
    "ZonalProfile": "🌡",
}


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def build_leaderboard_data(
    all_metrics: dict[str, dict[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dfs = []
    for tier_label, tier_metrics in all_metrics.items():
        for diag_name, mdf in tier_metrics.items():
            if mdf.empty:
                continue
            m = mdf.copy()
            m["tier"] = tier_label
            m["diagnostic"] = diag_name
            dfs.append(m)
    if not dfs:
        return pd.DataFrame(), pd.DataFrame()

    detail = pd.concat(dfs, ignore_index=True)

    score_cols: list[str] = []
    for metric, lower_better in [
        ("weighted_rmse", True),
        ("weighted_pearsonr", False),
        ("weighted_emd", True),
    ]:
        if metric not in detail.columns:
            continue
        col = f"_score_{metric}"
        score_cols.append(col)

        def _norm(grp: pd.Series, lb: bool = lower_better) -> pd.Series:
            lo, hi = grp.min(), grp.max()
            rng = hi - lo
            n = (grp - lo) / rng if rng > 1e-12 else pd.Series(0.5, index=grp.index)
            return 1 - n if lb else n

        detail[col] = detail.groupby(["tier", "diagnostic", "var_id"])[
            metric
        ].transform(_norm)

    if score_cols:
        detail["_overall_score"] = detail[score_cols].mean(axis=1)
    else:
        detail["_overall_score"] = np.nan

    summary = (
        detail.groupby("data_id")["_overall_score"]
        .mean()
        .reset_index()
        .rename(columns={"_overall_score": "Overall Score"})
        .sort_values("Overall Score", ascending=False)
    )
    summary["Rank"] = range(1, len(summary) + 1)
    return summary[["Rank", "data_id", "Overall Score"]].reset_index(drop=True), detail


def _score_color(score: float) -> str:
    h = int(score * 120)
    return f"hsl({h},65%,45%)"


def _render_summary_table(summary_df: pd.DataFrame, table_id: str) -> str:
    medal = {1: "medal-1", 2: "medal-2", 3: "medal-3"}
    rows_html = []
    for _, row in summary_df.iterrows():
        rank = int(row["Rank"])
        dataset = html_module.escape(str(row["data_id"]))
        score = row["Overall Score"]
        pct = f"{score * 100:.1f}"
        color = _score_color(score)
        rank_cls = medal.get(rank, "")
        bar_w = f"{score * 100:.1f}%"
        rows_html.append(
            f"<tr>"
            f"<td class='rank {rank_cls}'>{rank}</td>"
            f"<td><strong>{dataset}</strong></td>"
            f"<td class='score-cell bar-cell'>"
            f"  <div class='bar-bg' style='width:{bar_w};background:{color};'></div>"
            f"  <span class='bar-val' style='color:{color}'>{pct}%</span>"
            f"</td>"
            f"</tr>",
        )
    return (
        f"<div class='lb-table-wrap'>"
        f"<table class='lb' id='{table_id}'>"
        f"<thead><tr>"
        f"<th class='sortable' onclick='sortTable(\"{table_id}\",0,true)'>Rank</th>"
        f"<th class='sortable' onclick='sortTable(\"{table_id}\",1,false)'>Data Source</th>"
        f"<th class='sortable' onclick='sortTable(\"{table_id}\",2,true)'>Overall Score</th>"
        f"</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table></div>"
    )


def _render_detail_table(detail_df: pd.DataFrame, table_id: str) -> str:
    if detail_df.empty:
        return ""
    metric_cols = [
        c
        for c in ["weighted_rmse", "weighted_pearsonr", "weighted_emd"]
        if c in detail_df.columns
    ]
    str_cols = {"data_id", "tier", "diagnostic", "var_id"}
    cols_show = ["data_id", "tier", "diagnostic", "var_id", *metric_cols]
    if "tier" in detail_df.columns and detail_df["tier"].nunique() <= 1:
        cols_show = [c for c in cols_show if c != "tier"]
    cols_show = [c for c in cols_show if c in detail_df.columns]
    sub = detail_df[cols_show].copy()

    header_cells = "".join(
        f"<th class='sortable' onclick='sortTable(\"{table_id}\",{i},"
        f"{'false' if c in str_cols else 'true'})'>"
        f"{html_module.escape(c)}</th>"
        for i, c in enumerate(cols_show)
    )
    rows_html = []
    for _, row in sub.iterrows():
        cells = []
        for c in cols_show:
            val = row[c]
            if pd.isna(val):
                cells.append("<td>—</td>")
            elif isinstance(val, float):
                cells.append(f"<td class='score-cell'>{val:.4f}</td>")
            else:
                cells.append(f"<td>{html_module.escape(str(val))}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f"<div class='lb-table-wrap'>"
        f"<table class='lb' id='{table_id}'>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table></div>"
    )


def render_leaderboard_section(
    all_metrics: dict[str, dict[str, pd.DataFrame]],
    table_suffix: str = "",
) -> str:
    summary_df, detail_df = build_leaderboard_data(all_metrics)
    if summary_df.empty:
        return "<p class='empty-note'>No metrics data found.</p>"
    summary_html = _render_summary_table(summary_df, f"lb-summary{table_suffix}")
    detail_html = _render_detail_table(detail_df, f"lb-detail{table_suffix}")
    metric_note = (
        "<div class='metric-note'>"
        "Overall Score = normalized average of weighted RMSE, Pearson r, and Earth Mover's Distance "
        "across all diagnostics and variables. 100% = best possible."
        "</div>"
    )
    return (
        f"<div class='lb-section'>"
        f"{metric_note}"
        f"{summary_html}"
        f"<div class='detail-section'>"
        f"<details><summary>Full metrics table</summary>{detail_html}</details>"
        f"</div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Detail box (per-plot dataset info)
# ---------------------------------------------------------------------------


def _build_detail_html(
    var_id: str,
    variables_df: pd.DataFrame,
    data_df: pd.DataFrame,
    data_sources_df: pd.DataFrame | None = None,
) -> str:
    """Build the expandable dataset-info box shown below each plot."""
    rows: list[str] = []

    # Variable metadata
    var_row = (
        variables_df[variables_df["id"] == var_id]
        if variables_df is not None
        else pd.DataFrame()
    )
    if not var_row.empty:
        r = var_row.iloc[0]
        meta_items = []
        for col, label in [
            ("id", "Variable ID"),
            ("var_name", "Variable Name"),
            ("long_name", "Long Name"),
            ("standard_name", "CF Standard Name"),
            ("units", "Units"),
            ("realm", "Realm"),
            ("frequency", "Frequency"),
            ("timerange", "Time Range"),
            ("vertical_level", "Vertical Level"),
            ("min_lat", "Minimum Latitude"),
            ("max_lat", "Maximum Latitude"),
            ("min_lon", "Minimum Longitude"),
            ("max_lon", "Maximum Longitude"),
            ("landsea_mask", "Land/Sea Mask"),
            ("ar6_regions_mask", "AR6 Regions Mask"),
            ("anomaly", "Anomaly"),
        ]:
            val = r.get(col, "")
            if val is not None and str(val).strip() and str(val) != "nan":
                meta_items.append(
                    f"<div class='detail-meta-row'>"
                    f"<span class='detail-meta-key'>{label}</span>"
                    f"<span class='detail-meta-val'>{html_module.escape(str(val))}</span>"
                    f"</div>",
                )
        if meta_items:
            rows.append(
                f"<div class='detail-block'>"
                f"<div class='detail-block-title'>Variable Metadata</div>"
                f"{''.join(meta_items)}"
                f"</div>",
            )

    # Datasets used for this variable
    if data_df is not None and var_id in data_df.columns:
        sub = data_df[data_df[var_id].notna()][
            ["data_id", "data_type"]
        ].drop_duplicates()
        type_order = {"to_benchmark": 0, "reference": 1, "other": 2}
        sub = sub.sort_values("data_type", key=lambda s: s.map(type_order).fillna(3))

        ds_items: list[str] = []
        for _, ds_row in sub.iterrows():
            ds_name = str(ds_row["data_id"])
            dtype = str(ds_row["data_type"])
            type_badge = {
                "reference": "ref",
                "other": "other",
                "to_benchmark": "model",
            }.get(dtype, dtype)

            # Look up metadata from data_sources table
            full_name = ""
            provider = ""
            reference = ""
            if data_sources_df is not None and not data_sources_df.empty:
                src_row = data_sources_df[data_sources_df["id"] == ds_name]
                if not src_row.empty:
                    r = src_row.iloc[0]
                    full_name = r.get("name", "")
                    provider = r.get("institute", "")
                    refs_str = r.get("references", "")
                    if refs_str and str(refs_str).strip():
                        ref_parts = str(refs_str).split()
                        reference = ref_parts[0]  # treat the first reference as a URL

            detail_lines = []
            if full_name:
                detail_lines.append(f"<em>{html_module.escape(full_name)}</em>")
            if provider:
                detail_lines.append(f"Provider: {html_module.escape(provider)}")
            if reference:
                detail_lines.append(
                    f'Reference: <a href="{html_module.escape(reference)}" target="_blank" '
                    f'rel="noopener">{html_module.escape(reference)}</a>',
                )

            ds_items.append(
                f"<div class='detail-ds-item'>"
                f"<div class='detail-ds-header'>"
                f"<strong>{html_module.escape(ds_name)}</strong>"
                f"<span class='detail-badge detail-badge-{type_badge}'>{type_badge}</span>"
                f"</div>"
                + (
                    f"<div class='detail-ds-body'>{'<br>'.join(detail_lines)}</div>"
                    if detail_lines
                    else ""
                )
                + "</div>",
            )

        if ds_items:
            rows.append(
                f"<div class='detail-block'>"
                f"<div class='detail-block-title'>Datasets</div>"
                f"{''.join(ds_items)}"
                f"</div>",
            )

    if not rows:
        return ""
    return f"<div class='plot-detail-body'>{''.join(rows)}</div>"


# ---------------------------------------------------------------------------
# Diagnostic section rendering
# ---------------------------------------------------------------------------


def render_diagnostic_section(
    scope_id: str,
    figures: dict[str, go.Figure],
    variables_df: pd.DataFrame | None = None,
    data_df: pd.DataFrame | None = None,
    data_sources_df: pd.DataFrame | None = None,
) -> str:
    """Render plots for one diagnostic.

    scope_id namespaces all element IDs.
    Figures are embedded as JSON and rendered lazily by renderPanel().
    """
    if not figures:
        return "<p class='empty-note'>No data available.</p>"

    var_cols = list(figures.keys())
    first_var = var_cols[0]

    def _btn_label(v: str) -> str:
        if variables_df is not None and not variables_df.empty:
            row = variables_df[variables_df["id"] == v]
            if not row.empty:
                ln = str(row["long_name"].values[0])
                u = str(row["units"].values[0])
                return f"{ln} [{u}]" if u else ln
        return v

    def _search_terms(v: str) -> str:
        terms = [v]
        if variables_df is not None and not variables_df.empty:
            row = variables_df[variables_df["id"] == v]
            if not row.empty:
                terms.append(str(row["long_name"].values[0]))
                terms.append(str(row.get("standard_name", pd.Series([""])).values[0]))
        return " ".join(terms).lower()

    var_buttons = "".join(
        f"<button class='var-pill {'active' if v == first_var else ''}' "
        f"data-var='{html_module.escape(v)}' title='{html_module.escape(v)}' "
        f'onclick=\'showVarPanel("{scope_id}", "{v}")\'>'
        f"{html_module.escape(_btn_label(v))}</button>"
        for v in var_cols
    )

    _diag_type = (
        detect_diag_type(data_df.columns.tolist())
        if data_df is not None and not data_df.empty
        else "Unknown"
    )
    if _diag_type in ("Map", "ZonalProfile"):
        legend_note = (
            "<div class='legend-note'>"
            "<span><span class='ln-ref-swatch'></span>Reference (left column)</span>"
            "<span><span class='ln-bias-swatch'></span>Bias = model \u2212 reference (right column, RdBu\u2009r)</span>"
            "</div>"
        )
    else:
        legend_note = (
            "<div class='legend-note'>"
            "<span><span class='ln-solid'></span>Model</span>"
            "<span><span class='ln-dash'></span>Reference</span>"
            "<span><span class='ln-dot'></span>Other</span>"
            "</div>"
        )

    plot_panels = ""
    for var_id, fig in figures.items():
        panel_id = f"plot-{scope_id}-{var_id}"
        graph_id = f"graph-{scope_id}-{var_id}"
        detail_id = f"detail-{scope_id}-{var_id}"
        height = fig.layout.height or 440
        active_cls = "active" if var_id == first_var else ""
        search_key = html_module.escape(_search_terms(var_id))
        btn_label = html_module.escape(_btn_label(var_id))

        detail_html = _build_detail_html(
            var_id,
            variables_df if variables_df is not None else pd.DataFrame(),
            data_df if data_df is not None else pd.DataFrame(),
            data_sources_df if data_sources_df is not None else None,
        )

        detail_block = ""
        if detail_html:
            detail_block = (
                f"<div class='plot-detail-footer'>"
                f"<button class='detail-toggle-btn' onclick='toggleDetail(\"{detail_id}\")'>"
                f"<span class='detail-icon'>i</span> Dataset Info &amp; References"
                f"</button>"
                f"<div class='plot-detail-box' id='{detail_id}'>{detail_html}</div>"
                f"</div>"
            )

        plot_panels += (
            f"<div class='plot-panel {active_cls}' id='{panel_id}' "
            f"data-graph='{graph_id}' data-search='{search_key}'>"
            f"<div class='plot-title-bar'>"
            f"<span class='plot-title-text'>{btn_label}</span>"
            f"</div>"
            f"<div id='{graph_id}' style='height:{height}px;width:100%;'></div>"
            f"<script type='application/json' id='figdata-{graph_id}'>{fig.to_json()}</script>"
            f"{detail_block}"
            f"</div>"
        )

    return (
        f"<div id='diag-{scope_id}' class='diag-content'>"
        f"<div class='var-nav'>{var_buttons}</div>"
        f"{legend_note}"
        f"{plot_panels}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
*, *::before, *::after { box-sizing: border-box; }

:root {
  --navy: #0d1f35;
  --navy-mid: #163657;
  --navy-light: #1a5276;
  --accent: #2e86c1;
  --accent-hover: #1a6fa8;
  --sidebar-w: 230px;
  --header-h: 72px;
  --toolbar-h: 52px;
  --bg: #f0f3f7;
  --surface: #ffffff;
  --border: #dde2eb;
  --text: #1e2535;
  --muted: #6a7890;
  --font: 'Inter', 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
}

html, body {
  height: 100%; margin: 0; padding: 0;
  font-family: var(--font); font-size: 14px; color: var(--text);
  background: var(--bg); overflow: hidden;
}

/* ── App shell ── */
.app-shell {
  display: flex; flex-direction: column; height: 100vh; overflow: hidden;
}

/* ── Header ── */
.app-header {
  flex-shrink: 0; height: var(--header-h);
  background: linear-gradient(110deg, var(--navy) 0%, var(--navy-mid) 50%, var(--navy-light) 100%);
  color: #c8daf0; padding: 0 28px;
  display: flex; align-items: center; gap: 20px;
  border-bottom: 1px solid rgba(255,255,255,0.07);
  box-shadow: 0 2px 12px rgba(0,0,0,.35);
  z-index: 300;
}

.header-logo {
  flex-shrink: 0; width: 42px; height: 42px;
  background: rgba(255,255,255,0.10); border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
}

.header-title { flex: 1; min-width: 0; }
.header-title h1 {
  margin: 0 0 2px; font-size: 1.15rem; font-weight: 700;
  color: #fff; letter-spacing: -.1px; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
}
.header-meta {
  display: flex; gap: 20px; flex-wrap: wrap;
  font-size: 0.76rem; opacity: 0.65; line-height: 1.2;
}
.header-meta span::before { content: ""; }
.header-chip {
  display: flex; align-items: center; gap: 5px;
  background: rgba(255,255,255,0.08); border-radius: 20px;
  padding: 3px 10px; font-size: 0.74rem; color: #b8cde4;
  white-space: nowrap;
}
.header-chip strong { color: #e0edf8; font-weight: 600; }

/* ── Body split ── */
.app-body {
  display: flex; flex: 1; overflow: hidden; position: relative;
}

/* ── Sidebar ── */
.sidebar {
  flex-shrink: 0; width: var(--sidebar-w);
  background: var(--navy); color: #a8bdd4;
  display: flex; flex-direction: column;
  overflow-y: auto; overflow-x: hidden;
  border-right: 1px solid rgba(255,255,255,0.06);
  transition: width .2s ease;
}
.sidebar.collapsed { width: 0; border-right: none; }

.sidebar-toggle {
  position: absolute; top: 12px;
  left: calc(var(--sidebar-w) - 14px);
  width: 28px; height: 28px; border-radius: 50%;
  background: var(--accent); color: #fff;
  border: 2px solid var(--bg);
  cursor: pointer;
  box-shadow: 0 2px 6px rgba(0,0,0,.25);
  z-index: 150; user-select: none;
  transition: left .2s ease, background .15s;
  padding: 0;
}
.sidebar-toggle:hover { background: var(--accent-hover); }
.app-body.sidebar-collapsed .sidebar-toggle { left: 8px; }

.sidebar-toggle .bar {
  position: absolute;
  left: 6px;
  width: 12px; height: 2px;
  background: #fff;
  border-radius: 1px;
  transition: top .22s ease, transform .22s ease, opacity .15s ease;
}
/* Open state: X */
.sidebar-toggle .bar-top { top: 11px; transform: rotate(45deg); }
.sidebar-toggle .bar-mid { top: 11px; opacity: 0; }
.sidebar-toggle .bar-bot { top: 11px; transform: rotate(-45deg); }
/* Collapsed state: hamburger (three horizontal bars) */
.app-body.sidebar-collapsed .sidebar-toggle .bar-top { top: 7px;  transform: none; }
.app-body.sidebar-collapsed .sidebar-toggle .bar-mid { top: 11px; opacity: 1; }
.app-body.sidebar-collapsed .sidebar-toggle .bar-bot { top: 15px; transform: none; }

.sidebar-section-label {
  font-size: 0.68rem; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; color: rgba(255,255,255,0.25);
  padding: 18px 18px 6px;
}

.sidebar-item {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 18px; cursor: pointer;
  font-size: 0.875rem; font-weight: 500; color: #94adc8;
  border-left: 3px solid transparent;
  border-radius: 0 8px 8px 0; margin: 1px 8px 1px 0;
  transition: background .12s, color .12s, border-color .12s;
  user-select: none;
}
.sidebar-item:hover { background: rgba(255,255,255,0.06); color: #cce0f5; }
.sidebar-item.active {
  background: rgba(46,134,193,0.18); color: #fff;
  border-left-color: var(--accent); font-weight: 600;
}
.sidebar-item .sidebar-icon { font-size: 1rem; flex-shrink: 0; opacity: 0.8; }

.sidebar-divider {
  height: 1px; background: rgba(255,255,255,0.07); margin: 8px 16px;
}

/* ── Content area ── */
.content-wrap {
  flex: 1; display: flex; flex-direction: column; overflow: hidden;
}

/* ── Toolbar (diagnostic filter + search) ── */
.content-toolbar {
  flex-shrink: 0; height: var(--toolbar-h);
  background: var(--surface); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 12px;
  padding: 0 24px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
  z-index: 100;
}

.diag-filter-btns {
  display: flex; gap: 6px; align-items: center; flex-shrink: 0;
}
.diag-filter-btn {
  background: none; border: 1.5px solid var(--border);
  border-radius: 20px; padding: 5px 14px;
  font-size: 0.8rem; font-weight: 500; cursor: pointer;
  color: var(--muted); display: flex; align-items: center; gap: 5px;
  transition: background .12s, border-color .12s, color .12s;
  white-space: nowrap;
}
.diag-filter-btn:hover { background: #eef3fa; border-color: var(--accent); color: var(--navy-mid); }
.diag-filter-btn.active {
  background: var(--accent); border-color: var(--accent);
  color: #fff; font-weight: 600;
}

.toolbar-sep {
  width: 1px; height: 24px; background: var(--border); flex-shrink: 0;
}

.search-wrap {
  flex: 1; position: relative; max-width: 320px;
}
.search-wrap svg {
  position: absolute; left: 10px; top: 50%; transform: translateY(-50%);
  width: 15px; height: 15px; color: var(--muted); pointer-events: none;
}
.search-input {
  width: 100%; padding: 7px 12px 7px 32px;
  border: 1.5px solid var(--border); border-radius: 20px;
  font-size: 0.82rem; font-family: var(--font); color: var(--text);
  background: var(--bg); outline: none;
  transition: border-color .15s, background .15s;
}
.search-input:focus { border-color: var(--accent); background: #fff; }
.search-input::placeholder { color: var(--muted); }

/* ── Main scrolling content ── */
.main-content {
  flex: 1; overflow-y: auto; padding: 28px 32px 48px;
}

/* ── Content panels ── */
.content-panel { display: none; }
.content-panel.active { display: block; }

/* ── Diagnostic sections within a tier ── */
.diag-section { margin-bottom: 48px; }
.diag-section.hidden { display: none; }

.diag-section-header {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 18px; padding-bottom: 12px;
  border-bottom: 2px solid var(--border);
}
.diag-section-icon { font-size: 1.2rem; }
.diag-section-title {
  font-size: 1.05rem; font-weight: 700; color: var(--navy-mid);
}

/* ── Section headings ── */
.section-heading {
  font-size: 1.1rem; font-weight: 700; color: var(--navy-mid);
  margin: 0 0 22px; padding-bottom: 12px;
  border-bottom: 2px solid var(--border);
}

/* ── Leaderboard ── */
.lb-section { max-width: 900px; }

.metric-note {
  font-size: 0.82rem; color: #6a7790;
  background: #f0f4fa; border-left: 3px solid var(--accent);
  padding: 9px 14px; border-radius: 0 6px 6px 0; margin-bottom: 20px;
}

.lb-table-wrap {
  overflow-x: auto; border-radius: 10px;
  box-shadow: 0 2px 14px rgba(0,0,0,.09); margin-bottom: 16px;
}
table.lb {
  border-collapse: collapse; width: 100%; min-width: 440px;
  background: #fff; font-size: 0.87rem;
}
table.lb thead th {
  background: var(--navy-mid); color: #d8e4f0; font-weight: 600;
  padding: 11px 18px; text-align: left; white-space: nowrap;
  letter-spacing: .03em; font-size: 0.82rem; text-transform: uppercase;
}
table.lb thead th.sortable { cursor: pointer; user-select: none; }
table.lb thead th.sortable::after { content: " ↕"; opacity: 0.4; font-size: 0.78em; }
table.lb thead th.asc::after   { content: " ↑"; opacity: 1; }
table.lb thead th.desc::after  { content: " ↓"; opacity: 1; }
table.lb tbody tr { border-bottom: 1px solid #edf0f6; }
table.lb tbody tr:last-child { border-bottom: none; }
table.lb tbody tr:nth-child(even) { background: #f7f9fc; }
table.lb tbody tr:hover { background: #ecf2fb; }
table.lb tbody td { padding: 10px 18px; white-space: nowrap; }
td.rank { font-weight: 800; font-size: 1.05rem; text-align: center; width: 50px; }
td.score-cell { text-align: right; font-variant-numeric: tabular-nums; }
.medal-1 { color: #c0860a; }
.medal-2 { color: #7a8390; }
.medal-3 { color: #8b5e3c; }
.bar-cell { position: relative; min-width: 140px; }
.bar-bg {
  position: absolute; top: 20%; left: 8px;
  height: 60%; border-radius: 4px; opacity: 0.18; pointer-events: none;
}
.bar-val { position: relative; z-index: 1; font-weight: 700; font-size: 0.92rem; }
.detail-section { margin-top: 20px; }
details > summary {
  cursor: pointer; font-weight: 600; color: var(--navy-light);
  font-size: 0.87rem; padding: 8px 0; user-select: none;
  list-style: none; display: flex; align-items: center; gap: 6px;
}
details > summary::before { content: "▶"; font-size: 0.7em; transition: transform .15s; }
details[open] > summary::before { transform: rotate(90deg); }
details > summary:hover { opacity: 0.75; }

/* ── Variable pill nav ── */
.var-nav {
  display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;
}
.var-pill {
  background: #fff; border: 1.5px solid var(--border); border-radius: 24px;
  padding: 6px 18px; font-size: 0.80rem; cursor: pointer;
  color: #3a4a66; font-weight: 500; font-family: var(--font);
  box-shadow: 0 1px 3px rgba(0,0,0,.06);
  transition: background .13s, border-color .13s, color .13s, box-shadow .13s;
  max-width: 280px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.var-pill:hover { background: #edf2fb; border-color: var(--accent); color: var(--navy-mid); }
.var-pill.active {
  background: var(--navy-mid); color: #fff; border-color: var(--navy-mid);
  box-shadow: 0 3px 8px rgba(22,54,87,.28);
}

/* ── Plot panel card ── */
.plot-panel { display: none; }
.plot-panel.active { display: block; }

.plot-panel-card {
  background: var(--surface); border-radius: 12px;
  border: 1px solid var(--border);
  box-shadow: 0 2px 12px rgba(0,0,0,.07);
  overflow: hidden; margin-bottom: 8px;
}

.plot-title-bar {
  padding: 10px 18px; background: #f8fafc;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}
.plot-title-text {
  font-size: 0.88rem; font-weight: 600; color: var(--navy-mid);
}

/* ── Dataset detail box ── */
.plot-detail-footer {
  border-top: 1px solid var(--border); padding: 10px 18px 14px;
  background: #f8fafc;
}

.detail-toggle-btn {
  background: none; border: 1.5px solid var(--border);
  border-radius: 6px; padding: 5px 14px;
  font-size: 0.78rem; font-weight: 500; cursor: pointer;
  color: var(--muted); font-family: var(--font);
  display: inline-flex; align-items: center; gap: 6px;
  transition: background .12s, border-color .12s, color .12s;
}
.detail-toggle-btn:hover { background: #eef3fa; border-color: var(--accent); color: var(--navy-mid); }
.detail-icon { font-style: normal; font-size: 1em; }

.plot-detail-box {
  display: none; margin-top: 12px;
}
.plot-detail-box.open { display: block; }

.plot-detail-body {
  display: flex; flex-wrap: wrap; gap: 16px;
}

.detail-block {
  flex: 1; min-width: 260px;
  background: #fff; border: 1px solid var(--border);
  border-radius: 8px; padding: 14px; font-size: 0.82rem;
}

.detail-block-title {
  font-weight: 700; color: var(--navy-mid); font-size: 0.80rem;
  text-transform: uppercase; letter-spacing: .05em;
  margin-bottom: 10px; padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}

.detail-meta-row {
  display: flex; gap: 8px; padding: 3px 0;
  border-bottom: 1px solid #f0f2f5;
}
.detail-meta-row:last-child { border-bottom: none; }
.detail-meta-key {
  flex-shrink: 0; width: 130px; font-weight: 600; color: var(--muted);
  font-size: 0.78rem;
}
.detail-meta-val { color: var(--text); word-break: break-word; }

.detail-ds-item {
  margin-bottom: 10px; padding-bottom: 10px;
  border-bottom: 1px solid #f0f2f5;
}
.detail-ds-item:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.detail-ds-header {
  display: flex; align-items: center; gap: 8px; margin-bottom: 4px;
}
.detail-badge {
  font-size: 0.7rem; font-weight: 700; padding: 2px 8px;
  border-radius: 10px; text-transform: uppercase; letter-spacing: .04em;
}
.detail-badge-ref   { background: #e3f2fd; color: #1565c0; }
.detail-badge-model { background: #e8f5e9; color: #2e7d32; }
.detail-badge-other { background: #fff3e0; color: #e65100; }
.detail-ds-body {
  font-size: 0.79rem; color: var(--muted); line-height: 1.6;
}
.detail-ds-body a { color: var(--accent); text-decoration: none; }
.detail-ds-body a:hover { text-decoration: underline; }

/* ── Legend ── */
.legend-note {
  font-size: 0.79rem; color: #7a8aa0; margin: 0 0 14px;
  display: flex; gap: 18px; flex-wrap: wrap; align-items: center;
}
.legend-note span { display: flex; align-items: center; gap: 7px; }
.ln-solid {
  display: inline-block; width: 26px; height: 2.5px;
  background: #555; border-radius: 2px;
}
.ln-dash {
  display: inline-block; width: 26px; height: 2.5px; border-radius: 2px;
  background: repeating-linear-gradient(to right,#555 0 5px,transparent 5px 9px);
}
.ln-dot {
  display: inline-block; width: 26px; height: 2.5px; border-radius: 2px;
  background: repeating-linear-gradient(to right,#555 0 2px,transparent 2px 6px);
}
.ln-ref-swatch {
  display: inline-block; width: 14px; height: 14px; border-radius: 3px;
  background: rgba(192,134,10,0.15); border: 1.5px solid #c0860a; flex-shrink: 0;
}
.ln-bias-swatch {
  display: inline-block; width: 26px; height: 10px; border-radius: 3px; flex-shrink: 0;
  background: linear-gradient(to right, #2166ac, #f7f7f7, #d6604d);
}

/* ── Search no-results ── */
.search-no-results {
  display: none; padding: 40px 0; text-align: center;
  color: var(--muted); font-style: italic;
}
.search-no-results.visible { display: block; }

/* ── Misc ── */
.db-source { font-size: 0.76rem; color: #aab; margin: 0 0 14px; font-style: italic; }
.empty-note { color: #888; font-style: italic; padding: 20px 0; }
.diag-content { padding: 0; }

/* ── Scrollbar styling ── */
.sidebar::-webkit-scrollbar { width: 5px; }
.sidebar::-webkit-scrollbar-track { background: transparent; }
.sidebar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
.main-content::-webkit-scrollbar { width: 7px; }
.main-content::-webkit-scrollbar-track { background: var(--bg); }
.main-content::-webkit-scrollbar-thumb { background: #c8d0de; border-radius: 4px; }
"""

# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------

_JS = """
var _rendered = {};

function renderPanel(panelId) {
  if (typeof Plotly === 'undefined') return;
  var panel = document.getElementById(panelId);
  if (!panel) return;
  var graphId = panel.getAttribute('data-graph');
  if (!graphId) return;
  requestAnimationFrame(function() {
    requestAnimationFrame(function() {
      var div = document.getElementById(graphId);
      if (!div) return;
      if (_rendered[graphId]) { Plotly.relayout(div, {autosize:true}); return; }
      var scriptEl = document.getElementById('figdata-' + graphId);
      if (!scriptEl) return;
      var figData = JSON.parse(scriptEl.textContent);
      figData.layout.autosize = true;
      Plotly.newPlot(div, figData.data, figData.layout,
        {responsive:true, displayModeBar:true, displaylogo:false});
      _rendered[graphId] = true;
    });
  });
}

// Sidebar collapse/expand toggle
function toggleSidebar() {
  var sidebar = document.getElementById('sidebar');
  var appBody = document.getElementById('appBody');
  if (!sidebar || !appBody) return;
  var collapsed = sidebar.classList.toggle('collapsed');
  appBody.classList.toggle('sidebar-collapsed', collapsed);
  var btn = appBody.querySelector('.sidebar-toggle');
  if (btn) {
    var label = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
    btn.setAttribute('aria-label', label);
    btn.setAttribute('title', label);
  }
  // Re-render the currently visible plot so Plotly resizes to the new width.
  var ap = document.querySelector('.content-panel.active .plot-panel.active');
  if (ap && typeof Plotly !== 'undefined') {
    var graphId = ap.getAttribute('data-graph');
    var div = graphId && document.getElementById(graphId);
    if (div) {
      setTimeout(function(){ Plotly.Plots.resize(div); }, 220);
    }
  }
}

// Sidebar navigation
function sidebarNav(sectionId) {
  document.querySelectorAll('.content-panel').forEach(function(p){ p.classList.remove('active'); });
  document.querySelectorAll('.sidebar-item').forEach(function(b){ b.classList.remove('active'); });
  document.querySelectorAll('.content-toolbar').forEach(function(t){ t.style.display='none'; });

  var panel = document.getElementById('panel-' + sectionId);
  if (panel) panel.classList.add('active');

  var btn = document.querySelector('.sidebar-item[data-section="' + sectionId + '"]');
  if (btn) btn.classList.add('active');

  var toolbar = document.getElementById('toolbar-' + sectionId);
  if (toolbar) toolbar.style.display = 'flex';

  // render first visible plot
  var ap = panel && panel.querySelector('.plot-panel.active');
  if (ap) renderPanel(ap.id);
}

// Diagnostic filter within a tier
function filterDiag(tierId, diagName) {
  var panel = document.getElementById('panel-' + tierId);
  if (!panel) return;

  panel.querySelectorAll('.diag-section').forEach(function(s) {
    if (diagName === 'all' || s.getAttribute('data-diag') === diagName) {
      s.classList.remove('hidden');
    } else {
      s.classList.add('hidden');
    }
  });

  var toolbar = document.getElementById('toolbar-' + tierId);
  if (toolbar) {
    toolbar.querySelectorAll('.diag-filter-btn').forEach(function(b){ b.classList.remove('active'); });
    var activeBtn = toolbar.querySelector('.diag-filter-btn[data-diag="' + diagName + '"]');
    if (activeBtn) activeBtn.classList.add('active');
  }

  // re-render visible active plots
  panel.querySelectorAll('.diag-section:not(.hidden) .plot-panel.active').forEach(function(ap){
    renderPanel(ap.id);
  });
}

// Variable selector within a diagnostic
function showVarPanel(scopeId, varId) {
  var diag = document.getElementById('diag-' + scopeId);
  if (!diag) return;
  diag.querySelectorAll('.plot-panel').forEach(function(p){ p.classList.remove('active'); });
  diag.querySelectorAll('.var-pill').forEach(function(b){ b.classList.remove('active'); });
  var panelId = 'plot-' + scopeId + '-' + varId;
  var panel = document.getElementById(panelId);
  if (panel) panel.classList.add('active');
  var btn = diag.querySelector('.var-pill[data-var="' + varId + '"]');
  if (btn) btn.classList.add('active');
  renderPanel(panelId);
}

// Search bar
function searchPlots(query) {
  query = query.toLowerCase().trim();
  // apply to each tier panel
  document.querySelectorAll('.content-panel[data-type="tier"]').forEach(function(panel) {
    var noResults = panel.querySelector('.search-no-results');
    var anyVisible = false;
    panel.querySelectorAll('.plot-panel').forEach(function(pp) {
      var terms = (pp.getAttribute('data-search') || '').toLowerCase();
      var match = !query || terms.indexOf(query) !== -1;
      if (match) {
        pp.classList.add('active');
        pp.querySelectorAll('.var-pill').forEach(function(b){ b.classList.remove('active'); });
        var pill = pp.closest('.diag-content') &&
          pp.closest('.diag-content').querySelector('.var-pill[data-var="' + pp.id.split('-').pop() + '"]');
        anyVisible = true;
      } else {
        pp.classList.remove('active');
      }
    });
    // show/hide diag sections that have visible plots
    panel.querySelectorAll('.diag-section').forEach(function(sec) {
      if (sec.classList.contains('hidden')) return;
      var hasVisible = Array.from(sec.querySelectorAll('.plot-panel')).some(function(p){
        return p.classList.contains('active');
      });
      sec.style.opacity = hasVisible ? '1' : '0.3';
    });
    if (noResults) noResults.classList.toggle('visible', !anyVisible && query.length > 0);
    if (!query) {
      // restore to first var per diag
      panel.querySelectorAll('.diag-section:not(.hidden) .diag-content').forEach(function(dc) {
        dc.querySelectorAll('.plot-panel').forEach(function(p){ p.classList.remove('active'); });
        dc.querySelectorAll('.var-pill').forEach(function(b){ b.classList.remove('active'); });
        var firstPanel = dc.querySelector('.plot-panel');
        var firstPill = dc.querySelector('.var-pill');
        if (firstPanel) firstPanel.classList.add('active');
        if (firstPill) firstPill.classList.add('active');
      });
      panel.querySelectorAll('.diag-section').forEach(function(s){ s.style.opacity='1'; });
    }
  });
}

// Detail box toggle
function toggleDetail(detailId) {
  var box = document.getElementById(detailId);
  if (!box) return;
  box.classList.toggle('open');
}

// Leaderboard table sort
function sortTable(tableId, colIdx, numeric) {
  var table = document.getElementById(tableId);
  var tbody = table.querySelector('tbody');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var th = table.querySelectorAll('thead th')[colIdx];
  var asc = th.classList.contains('asc');
  table.querySelectorAll('thead th').forEach(function(h){ h.classList.remove('asc','desc'); });
  th.classList.add(asc ? 'desc' : 'asc');
  rows.sort(function(a,b){
    var va = a.cells[colIdx].innerText.trim();
    var vb = b.cells[colIdx].innerText.trim();
    var cmp = numeric ? (parseFloat(va)||0)-(parseFloat(vb)||0) : va.localeCompare(vb);
    return asc ? -cmp : cmp;
  });
  rows.forEach(function(r){ tbody.appendChild(r); });
}
"""


# ---------------------------------------------------------------------------
# HTML assembly helpers
# ---------------------------------------------------------------------------


def _stem_to_label(stem: str) -> str:
    s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", stem)
    return s.replace("_", " ").replace("-", " ").title()


def _plotly_js(offline: bool) -> str:
    if offline:
        import plotly as _plotly

        js_path = Path(_plotly.__file__).parent / "package_data" / "plotly.min.js"
        if js_path.exists():
            return f"<script>{js_path.read_text()}</script>"
    return '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'


def _build_figures_for_tier(
    db_data: dict[str, dict[str, pd.DataFrame]],
    scope_prefix: str,
) -> tuple[dict[str, tuple[str, pd.DataFrame]], dict[str, pd.DataFrame]]:
    """Return (diag_sections {name: (html, variables_df)}, tier_metrics) for one database."""
    diag_sections: dict[str, tuple[str, pd.DataFrame]] = {}
    tier_metrics: dict[str, pd.DataFrame] = {}

    for diag_name, tables in db_data.items():
        has_raw_output = "raw_output" in tables
        data_df = tables.get("raw_output", pd.DataFrame())
        metrics_df = tables.get("metrics", pd.DataFrame())
        variables_df = tables.get("variables", pd.DataFrame())
        data_sources_df = tables.get("data_sources", pd.DataFrame())

        if has_raw_output and not data_df.empty:
            tier_metrics[diag_name] = metrics_df

        # Skip diagnostics without raw_output table (e.g., AnnualCycleSpatialDistribution)
        if not has_raw_output:
            continue

        diag_type = detect_diag_type(data_df.columns.tolist())
        builder = _FIGURE_BUILDERS.get(diag_type)
        if builder is None or data_df.empty:
            diag_sections[diag_name] = (
                "<p class='empty-note'>No data available.</p>",
                variables_df,
            )
            continue

        figures = builder(data_df, variables_df, metrics_df)
        scope_id = f"{scope_prefix}-{diag_name}" if scope_prefix else diag_name
        html = render_diagnostic_section(
            scope_id,
            figures,
            variables_df,
            data_df,
            data_sources_df,
        )
        diag_sections[diag_name] = (html, variables_df)

    return diag_sections, tier_metrics


def _render_tier_panel(
    tier_id: str,
    db_path: Path,
    diag_sections: dict[str, tuple[str, pd.DataFrame]],
) -> tuple[str, str]:
    """Return (toolbar_html, panel_html) for one tier."""
    diag_names = list(diag_sections.keys())

    # Toolbar: All + per-diagnostic filter buttons + search
    filter_btns = (
        f"<button class='diag-filter-btn active' data-diag='all' "
        f'onclick=\'filterDiag("{tier_id}","all")\'>All</button>'
    )
    for dname in diag_names:
        label = _DIAG_LABELS.get(dname, dname)
        icon = _DIAG_ICONS.get(dname, "")
        filter_btns += (
            f"<button class='diag-filter-btn' data-diag='{dname}' "
            f'onclick=\'filterDiag("{tier_id}","{dname}")\'>'
            f"{icon} {label}</button>"
        )

    search_svg = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
        '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'
        "</svg>"
    )
    toolbar_html = (
        f"<div class='content-toolbar' id='toolbar-{tier_id}' style='display:none'>"
        f"<div class='diag-filter-btns'>{filter_btns}</div>"
        f"<div class='toolbar-sep'></div>"
        f"<div class='search-wrap'>"
        f"{search_svg}"
        f"<input class='search-input' type='text' placeholder='Search plots…' "
        f"oninput='searchPlots(this.value)'>"
        f"</div>"
        f"</div>"
    )

    # Panel: stacked diagnostic sections
    sections_html = ""
    for dname, (content, _) in diag_sections.items():
        label = _DIAG_LABELS.get(dname, dname)
        icon = _DIAG_ICONS.get(dname, "")
        sections_html += (
            f"<div class='diag-section' data-diag='{dname}' id='diag-sec-{tier_id}-{dname}'>"
            f"<div class='diag-section-header'>"
            f"<span class='diag-section-icon'>{icon}</span>"
            f"<span class='diag-section-title'>{label}</span>"
            f"</div>"
            f"{content}"
            f"</div>"
        )

    panel_html = (
        f"<div class='content-panel' id='panel-{tier_id}' data-type='tier'>"
        f"<p class='db-source'>{html_module.escape(str(db_path))}</p>"
        f"{sections_html}"
        f"<div class='search-no-results'>No plots match your search.</div>"
        f"</div>"
    )

    return toolbar_html, panel_html


# ---------------------------------------------------------------------------
# Main HTML generator
# ---------------------------------------------------------------------------


def generate_html(
    db_paths: list[Path],
    *,
    offline: bool = True,
    model_name: str = "",
    user_name: str = "",
    developer: str = "",
) -> str:
    """Read one or more databases and produce a self-contained HTML string."""
    _coastlines()

    tiers: list[
        tuple[
            str,
            str,
            Path,
            dict[str, tuple[str, pd.DataFrame]],
            dict[str, pd.DataFrame],
        ]
    ] = []
    all_metrics: dict[str, dict[str, pd.DataFrame]] = {}

    for tier_idx, db_path in enumerate(db_paths):
        label = _stem_to_label(db_path.stem)
        db_data = read_database(db_path)
        if not db_data:
            continue

        tier_id = f"tier{tier_idx}"
        diag_sections, tier_metrics = _build_figures_for_tier(db_data, tier_id)
        all_metrics[label] = tier_metrics
        tiers.append((tier_id, label, db_path, diag_sections, tier_metrics))

    if not tiers:
        msg = "No diagnostic databases found in any of the provided files."
        raise ValueError(msg)

    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")

    # ── Sidebar ──
    sidebar_items = (
        "<div class='sidebar-section-label'>Overview</div>"
        "<div class='sidebar-item active' data-section='leaderboard' "
        "onclick='sidebarNav(\"leaderboard\")'>"
        "<span class='sidebar-icon'>🏆</span>Leaderboard"
        "</div>"
        "<div class='sidebar-divider'></div>"
        "<div class='sidebar-section-label'>Tiers</div>"
    )
    for tier_id, label, *_ in tiers:
        sidebar_items += (
            f"<div class='sidebar-item' data-section='{tier_id}' "
            f"onclick='sidebarNav(\"{tier_id}\")'>"
            f"<span class='sidebar-icon'>📂</span>"
            f"{html_module.escape(label)}"
            f"</div>"
        )

    # ── Leaderboard panel ──
    lb_panel = (
        "<div class='content-panel active' id='panel-leaderboard'>"
        "<h2 class='section-heading'>Performance Leaderboard</h2>"
        + render_leaderboard_section(all_metrics, table_suffix="-main")
        + "</div>"
    )

    # ── Tier panels + toolbars ──
    all_toolbars = ""
    all_panels = lb_panel

    for tier_id, label, db_path, diag_sections, tier_metrics in tiers:
        toolbar_html, panel_html = _render_tier_panel(tier_id, db_path, diag_sections)
        all_toolbars += toolbar_html
        all_panels += panel_html

    # ── Header chips ──
    chips = ""
    if model_name:
        chips += f"<span class='header-chip'><span>Model</span> <strong>{html_module.escape(model_name)}</strong></span>"
    if user_name:
        chips += f"<span class='header-chip'><span>User</span> <strong>{html_module.escape(user_name)}</strong></span>"
    developer_str = developer or "diffESM Benchmarking"
    chips += f"<span class='header-chip'><span>Tool</span> <strong>{html_module.escape(developer_str)}</strong></span>"
    chips += f"<span class='header-chip'><span>Generated</span> <strong>{now}</strong></span>"

    logo_svg = (
        '<svg width="26" height="26" viewBox="0 0 26 26" fill="none" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="13" cy="13" r="11" stroke="#60a5fa" stroke-width="1.5" fill="none"/>'
        '<path d="M2 13 Q7 7 13 13 Q19 19 24 13" stroke="#60a5fa" stroke-width="1.5" fill="none"/>'
        '<path d="M2 13 Q7 19 13 13 Q19 7 24 13" stroke="#93c5fd" stroke-width="1" fill="none" stroke-dasharray="2 2"/>'
        "</svg>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ESM Benchmarking Report</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="app-shell">

  <header class="app-header">
    <div class="header-logo">{logo_svg}</div>
    <div class="header-title">
      <h1>ESM Benchmarking Report</h1>
      <div class="header-meta">{chips}</div>
    </div>
  </header>

  <div class="app-body" id="appBody">

    <aside class="sidebar" id="sidebar">
      {sidebar_items}
    </aside>
    <button class="sidebar-toggle" type="button" onclick="toggleSidebar()"
            aria-label="Collapse sidebar" title="Collapse sidebar">
      <span class="bar bar-top"></span>
      <span class="bar bar-mid"></span>
      <span class="bar bar-bot"></span>
    </button>

    <div class="content-wrap">
      {all_toolbars}
      <main class="main-content">
        {all_panels}
      </main>
    </div>

  </div>
</div>

{_plotly_js(offline)}
<script>{_JS}</script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Generate HTML."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate a portable HTML benchmarking report from one or more DuckDB "
            "databases. Pass multiple files to produce a multi-tier report."
        ),
    )
    parser.add_argument(
        "db",
        type=Path,
        nargs="+",
        help="Path(s) to DuckDB benchmarking database (.ddb).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: <first_db>_report.html).",
    )
    parser.add_argument(
        "--model",
        default="",
        metavar="NAME",
        help="Model name shown in the report header (e.g. MPI-ESM1-2-LR).",
    )
    parser.add_argument(
        "--user",
        default="",
        metavar="NAME",
        help="User / analyst name shown in the report header.",
    )
    parser.add_argument(
        "--developer",
        default="",
        metavar="NAME",
        help="Benchmarking tool developer name (default: 'diffESM Benchmarking').",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Load Plotly.js from CDN instead of embedding (requires internet).",
    )
    args = parser.parse_args()

    db_paths = [p.resolve() for p in args.db]
    for p in db_paths:
        if not p.exists():
            parser.error(f"Not found: {p}")

    output = args.output or db_paths[0].with_name(db_paths[0].stem + "_report.html")

    html_content = generate_html(
        db_paths,
        offline=not args.online,
        model_name=args.model,
        user_name=args.user,
        developer=args.developer,
    )
    output.write_text(html_content, encoding="utf-8")
    print(f"HTML output is saved to {output}")


if __name__ == "__main__":
    main()
