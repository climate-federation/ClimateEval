"""All build_*_figures functions and related helpers for the benchmarking report."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from climateeval.report._db import get_density_cols, get_var_cols
from climateeval.report._style import (
    _AXIS_STYLE,
    _DIVERGING_VARS,
    _PLOT_FONT,
    _SEABORN_COLORS,
    _bias_colorscale_for,
    _colorscale_for,
    _hovmoller_colorscale_for,
)

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
# Shared plot helpers
# ---------------------------------------------------------------------------


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


def _get_line_properties(dtype: str, color_idx: int) -> dict:
    """Return line styling for 1D plots based on data type.

    Args:
        dtype: One of "reference", "to_benchmark", or "other"
        color_idx: Index into seaborn color palette (for to_benchmark only)

    Returns
    -------
        dict with line, showlegend, zorder
    """
    if dtype == "reference":
        return {
            "line": {
                "color": "black",
                "dash": "solid",
                "width": 2,
            },
            "showlegend": True,
            "zorder": 1,
        }
    if dtype == "to_benchmark":
        return {
            "line": {
                "color": _SEABORN_COLORS[color_idx % len(_SEABORN_COLORS)],
                "dash": "solid",
                "width": 2,
            },
            "showlegend": True,
            "zorder": 2,
        }
    # "other"
    return {
        "line": {
            "color": "#888",
            "dash": "dash",
            "width": 1.5,
        },
        "showlegend": False,
        "zorder": 0,
    }


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


def _plot_hlines(figure: go.Figure, variables_df: pd.DataFrame, var_id: str) -> None:
    """Plot horizontal lines."""
    variables_row = variables_df[variables_df["id"] == var_id]
    for col in ("reasonable_min", "reasonable_max"):
        if col not in variables_row:
            continue
        val = variables_row[col].array[0]
        if np.isnan(val):
            continue
        figure.add_hline(y=val, line_dash="dash")


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
            sub = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
            sv = sub[["latitude", "longitude", var_id]].copy()
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
                    f"<b>{ds}</b> − {ref_label}",
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
                        "title": {"text": f"Δ{units}", "side": "right"},
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
                    f"{ds} − {ref_label}",
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
        # Each map panel should be ~2:1 (width:height) to match 360°×180° lon/lat.
        # Set explicit width so the aspect ratio is correct regardless of browser width.
        _panel_w = 440  # px per column
        _panel_h = 260  # px per row  (440/2 + room for axis labels)
        _h_margin = 130  # t=80 + b=50
        _w_margin = (260 if has_bias else 180)  # l + r + colorbars
        fig.update_layout(
            **_base_layout(
                title={
                    "text": title_str,
                    "font": {"size": 14},
                    "x": 0.02,
                    "xanchor": "left",
                },
                plot_bgcolor="#dce9f5",
                height=nrows * _panel_h + _h_margin,
                width=ncols * _panel_w + _w_margin,
                autosize=False,
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
        grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
        sub = grp[["time", var_id]].sort_values("time")
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

        _plot_hlines(fig, variables_df, var_id)

        # Use markers when there are too few points for a visible line.
        n_times = data_df["time"].nunique()
        ts_mode = "lines+markers" if n_times < 3 else "lines"

        color_idx = 0
        for dataset in all_datasets:
            series = _series_for(dataset, var_id)
            if series is None:
                continue
            grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
            dtype = grp["data_type"].iloc[0]

            trace = go.Scatter(
                x=series.index,
                y=series.values,
                mode=ts_mode,
                name=str(dataset),
                legendgroup=str(dataset),
                **_get_line_properties(dtype, color_idx),
            )
            if has_bias:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

            if dtype == "to_benchmark":
                color_idx += 1

        if has_bias:
            color_idx = 0
            for dataset in all_datasets:
                if dataset == ref_dataset:
                    continue
                series = _series_for(dataset, var_id)
                if series is None:
                    continue
                grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
                dtype = grp["data_type"].iloc[0]
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
                        mode=ts_mode,
                        name=str(dataset),
                        legendgroup=str(dataset),
                        hovertemplate=(
                            f"<b>{dataset} − {ref_dataset}</b><br>"
                            f"%{{x|%Y-%m}}<br>"
                            f"Δ{var_id}: %{{y:.4g}} {units}"
                            "<extra></extra>"
                        ),
                        **_get_line_properties(dtype, color_idx),
                    ),
                    row=2,
                    col=1,
                )
                if dtype == "to_benchmark":
                    color_idx += 1
            fig.add_hline(
                y=0,
                line={"color": "rgba(0,0,0,0.45)", "width": 1, "dash": "dot"},
                row=2,
                col=1,
            )

        title_str = f"<b>{long_name}</b>" + (f"  [{units}]" if units else "")
        layout_kwargs = {
            "title": {
                "text": title_str,
                "font": {"size": 14},
                "x": 0.02,
                "xanchor": "left",
            },
            "legend": {
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
            "margin": {"t": 80, "b": 60, "l": 70, "r": 30},
        }
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
        grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
        sub = grp[["month_number", var_id]]
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

        _plot_hlines(fig, variables_df, var_id)

        color_idx = 0
        for dataset in all_datasets:
            series = _series_for(dataset, var_id)
            if series is None:
                continue
            grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
            dtype = grp["data_type"].iloc[0]

            trace = go.Scatter(
                x=series.index.tolist(),
                y=series.values.tolist(),
                mode="lines+markers",
                name=str(dataset),
                marker={"size": 6},
                legendgroup=str(dataset),
                hovertemplate=(
                    f"<b>{dataset}</b><br>"
                    f"Month: %{{x:.0f}}<br>"
                    f"{var_id}: %{{y:.4g}} {units}"
                    "<extra></extra>"
                ),
                **_get_line_properties(dtype, color_idx),
            )
            if has_bias:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

            if dtype == "to_benchmark":
                color_idx += 1

        if has_bias:
            color_idx = 0
            for dataset in all_datasets:
                if dataset == ref_dataset:
                    continue
                series = _series_for(dataset, var_id)
                if series is None:
                    continue
                grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
                dtype = grp["data_type"].iloc[0]

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
                        marker={"size": 5},
                        legendgroup=str(dataset),
                        hovertemplate=(
                            f"<b>{dataset} − {ref_dataset}</b><br>"
                            f"Month: %{{x:.0f}}<br>"
                            f"Δ{var_id}: %{{y:.4g}} {units}"
                            "<extra></extra>"
                        ),
                        **_get_line_properties(dtype, color_idx),
                    ),
                    row=2,
                    col=1,
                )
                if dtype == "to_benchmark":
                    color_idx += 1
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
        layout_kwargs = {
            "title": {
                "text": title_str,
                "font": {"size": 14},
                "x": 0.02,
                "xanchor": "left",
            },
            "legend": {
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
            "margin": {"t": 80, "b": 60, "l": 70, "r": 30},
        }
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


def build_diurnal_cycle_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, go.Figure]:
    """Hourly climatology (dirunaly cycle), one figure per variable, all datasets overlaid.

    When a single reference dataset is available, a second panel shows the bias
    (dataset − reference) for every non-reference dataset.
    """
    figures: dict[str, go.Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())
    ref_dataset = _get_ref_dataset(data_df, metrics_df)

    def _series_for(dataset: str, var_id: str) -> pd.Series | None:
        grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
        sub = grp[["hour", var_id]]
        if sub.empty:
            return None
        return sub.groupby("hour")[var_id].mean().sort_index()

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

        _plot_hlines(fig, variables_df, var_id)

        color_idx = 0
        for dataset in all_datasets:
            series = _series_for(dataset, var_id)
            if series is None:
                continue
            grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
            dtype = grp["data_type"].iloc[0]

            trace = go.Scatter(
                x=series.index.tolist(),
                y=series.values.tolist(),
                mode="lines+markers",
                name=str(dataset),
                marker={"size": 6},
                legendgroup=str(dataset),
                hovertemplate=(
                    f"<b>{dataset}</b><br>"
                    f"Hour: %{{x:.0f}}<br>"
                    f"{var_id}: %{{y:.4g}} {units}"
                    "<extra></extra>"
                ),
                **_get_line_properties(dtype, color_idx),
            )
            if has_bias:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

            if dtype == "to_benchmark":
                color_idx += 1

        if has_bias:
            color_idx = 0
            for dataset in all_datasets:
                if dataset == ref_dataset:
                    continue
                series = _series_for(dataset, var_id)
                if series is None:
                    continue
                grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
                dtype = grp["data_type"].iloc[0]

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
                        marker={"size": 5},
                        legendgroup=str(dataset),
                        hovertemplate=(
                            f"<b>{dataset} − {ref_dataset}</b><br>"
                            f"Hour: %{{x:.0f}}<br>"
                            f"Δ{var_id}: %{{y:.4g}} {units}"
                            "<extra></extra>"
                        ),
                        **_get_line_properties(dtype, color_idx),
                    ),
                    row=2,
                    col=1,
                )
                if dtype == "to_benchmark":
                    color_idx += 1
            fig.add_hline(
                y=0,
                line={"color": "rgba(0,0,0,0.45)", "width": 1, "dash": "dot"},
                row=2,
                col=1,
            )

        x_kwargs = {
            "title_text": "Hour",
            "tickvals": list(range(24)),
            "ticktext": list(range(24)),
            **_AXIS_STYLE,
        }
        title_str = f"<b>{long_name}</b>" + (f"  [{units}]" if units else "")
        layout_kwargs = {
            "title": {
                "text": title_str,
                "font": {"size": 14},
                "x": 0.02,
                "xanchor": "left",
            },
            "legend": {
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
            "margin": {"t": 80, "b": 60, "l": 70, "r": 30},
        }
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
        grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
        sub = grp[["latitude", var_id]]
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

        _plot_hlines(fig, variables_df, var_id)

        color_idx = 0
        for dataset in all_datasets:
            series = _series_for(dataset, var_id)
            if series is None:
                continue
            grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
            dtype = grp["data_type"].iloc[0]

            trace = go.Scatter(
                x=series.index.tolist(),
                y=series.values.tolist(),
                mode="lines",
                name=str(dataset),
                legendgroup=str(dataset),
                hovertemplate=(
                    f"<b>{dataset}</b><br>lat: %{{x:.1f}}°<br>"
                    f"{var_id}: %{{y:.4g}} {units}<extra></extra>"
                ),
                **_get_line_properties(dtype, color_idx),
            )
            if has_bias:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

            if dtype == "to_benchmark":
                color_idx += 1

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
                grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
                dtype = grp["data_type"].iloc[0]

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
                        legendgroup=str(dataset),
                        hovertemplate=(
                            f"<b>{dataset} − {ref_dataset}</b><br>"
                            f"lat: %{{x:.1f}}°<br>"
                            f"Δ{var_id}: %{{y:.4g}} {units}"
                            "<extra></extra>"
                        ),
                        **_get_line_properties(dtype, color_idx),
                    ),
                    row=2,
                    col=1,
                )
                if dtype == "to_benchmark":
                    color_idx += 1
            fig.add_hline(
                y=0,
                line={"color": "rgba(0,0,0,0.45)", "width": 1, "dash": "dot"},
                row=2,
                col=1,
            )

        title_str = f"<b>{long_name}</b>" + (f"  [{units}]" if units else "")
        layout_kwargs = {
            "title": {
                "text": title_str,
                "font": {"size": 14},
                "x": 0.02,
                "xanchor": "left",
            },
            "legend": {
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
            "margin": {"t": 80, "b": 60, "l": 70, "r": 30},
        }
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


def build_depth_profile_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, go.Figure]:
    """Depth profile line plots, one figure per variable, all datasets overlaid.

    When a single reference dataset is available, a second panel shows the bias
    (dataset − reference) as a function of depth for every non-reference
    dataset.
    """
    figures: dict[str, go.Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())
    ref_dataset = _get_ref_dataset(data_df, metrics_df)

    def _series_for(dataset: str, var_id: str) -> pd.Series | None:
        grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
        cols = list(dict.fromkeys(["depth", var_id]))
        sub = grp[cols]
        if sub.empty:
            return None
        return sub.groupby("depth")[var_id].mean().sort_index()

    # DYNAMISCHE TIEFENACHSE: Extrahiert die tatsächlichen Tiefenwerte aus den Daten
    unique_depths = sorted(data_df["depth"].dropna().unique())
    if unique_depths:
        max_d = max(unique_depths)
        if max_d > 1000:
            step = 1000 if max_d > 2000 else 500
            depth_tickvals = list(range(0, int(max_d) + step, step))
        else:
            depth_tickvals = [0, 200, 400, 600, 800, 1000]
        depth_ticktext = [f"{d}" for d in depth_tickvals]
    else:
        depth_tickvals = [0, 1000, 2000, 3000, 4000, 5000]
        depth_ticktext = ["0", "1000", "2000", "3000", "4000", "5000"]

    for var_id in get_var_cols(data_df):
        # FILTER: Überspringt die Generierung, wenn die Variable selbst "depth" ist
        if var_id == "depth" or data_df[var_id].isna().all():
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
                shared_xaxes=False,
                shared_yaxes=True,
                vertical_spacing=0.08,
                row_heights=[0.65, 0.35],
                subplot_titles=("", f"Bias (dataset − {ref_dataset})"),
            )
        else:
            fig = go.Figure()

        try:
            _plot_hlines(fig, variables_df, var_id)
            if hasattr(fig, "layout") and fig.layout.shapes:
                for shape in fig.layout.shapes:
                    if (
                        shape.type == "line"
                        and shape.y0 is not None
                        and shape.y1 is not None
                    ):
                        shape.x0, shape.x1 = shape.y0, shape.y1
                        shape.y0, shape.y1 = 0, 1
                        shape.xref = "x"
                        shape.yref = "y domain"
        except IndexError:
            pass

        color_idx = 0
        for dataset in all_datasets:
            series = _series_for(dataset, var_id)
            if series is None:
                continue
            grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
            dtype = grp["data_type"].iloc[0]

            trace = go.Scatter(
                x=series.values.tolist(),
                y=series.index.tolist(),
                mode="lines",
                name=str(dataset),
                legendgroup=str(dataset),
                hovertemplate=(
                    f"<b>{dataset}</b><br>depth: %{{y:.1f}}m<br>"
                    f"{var_id}: %{{x:.4g}} {units}<extra></extra>"
                ),
                **_get_line_properties(dtype, color_idx),
            )
            if has_bias:
                fig.add_trace(trace, row=1, col=1)
            else:
                fig.add_trace(trace)

            if dtype == "to_benchmark":
                color_idx += 1

        if has_bias:
            color_idx = 0
            for dataset in all_datasets:
                if dataset == ref_dataset:
                    continue
                series = _series_for(dataset, var_id)
                if series is None:
                    continue
                grp = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
                dtype = grp["data_type"].iloc[0]

                common = series.index.intersection(ref_series.index)
                if common.empty:
                    continue
                bias = (series.loc[common] - ref_series.loc[common]).dropna()
                if bias.empty:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=bias.values.tolist(),
                        y=bias.index.tolist(),
                        mode="lines",
                        name=str(dataset),
                        legendgroup=str(dataset),
                        hovertemplate=(
                            f"<b>{dataset} − {ref_dataset}</b><br>"
                            f"depth: %{{y:.1f}}m<br>"
                            f"Δ{var_id}: %{{x:.4g}} {units}"
                            "<extra></extra>"
                        ),
                        **_get_line_properties(dtype, color_idx),
                    ),
                    row=2,
                    col=1,
                )
                if dtype == "to_benchmark":
                    color_idx += 1
            fig.add_vline(
                x=0,
                line={"color": "rgba(0,0,0,0.45)", "width": 1, "dash": "dot"},
                row=2,
                col=1,
            )

        title_str = f"<b>{long_name}</b>" + (f"  [{units}]" if units else "")
        layout_kwargs = {
            "title": {
                "text": title_str,
                "font": {"size": 14},
                "x": 0.02,
                "xanchor": "left",
            },
            "legend": {
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
            "margin": {"t": 80, "b": 60, "l": 70, "r": 30},
        }

        y_kwargs_full = dict(
            title_text="Depth",
            tickvals=depth_tickvals,
            ticktext=depth_ticktext,
            autorange="reversed",
            **_AXIS_STYLE,
        )

        if has_bias:
            fig.update_yaxes(**y_kwargs_full, row=1, col=1)
            fig.update_yaxes(
                autorange="reversed",
                tickvals=depth_tickvals,
                ticktext=depth_ticktext,
                **_AXIS_STYLE,
                row=2,
                col=1,
            )
            fig.update_xaxes(title_text=units, **_AXIS_STYLE, row=1, col=1)
            fig.update_xaxes(title_text=f"Δ{units}", **_AXIS_STYLE, row=2, col=1)
            layout_kwargs["height"] = 560
        else:
            layout_kwargs["yaxis"] = y_kwargs_full
            layout_kwargs["xaxis"] = dict(title_text=units, **_AXIS_STYLE)
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
            sub = data_df[data_df["data_id"] == dataset].dropna(subset=[var_id])
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
                titles += [f"<b>{ref_label}</b>", f"<b>{ds}</b> − {ref_label}"]
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
                        "title": {"text": f"Δ{units}", "side": "right"},
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
                    f"{ds} − {ref_label}",
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
            if has_bias and ann.text and ref_label in ann.text and "−" not in ann.text:
                ann.font = {"size": 11, "color": "#a07000"}
                ann.bgcolor = "rgba(192,134,10,0.08)"
                ann.bordercolor = "#c0860a"
                ann.borderwidth = 1
                ann.borderpad = 3
            else:
                ann.font = {"size": 11, "color": "#333"}
        figures[var_id] = fig

    return figures


def build_bar_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    _metrics_df: pd.DataFrame | None = None,
) -> dict[str, go.Figure]:
    """Horizontal bar chart for scalar (0-D) diagnostics, one figure per variable.

    Each dataset appears as one bar, sorted by value.  The ``to_benchmark``
    dataset is highlighted in blue; reference datasets in gold; all other datasets
    (e.g. the CMIP6 ensemble used for context) in light grey.
    """
    type_color = {
        "to_benchmark": "#4C72B0",
        "reference": "#c0860a",
        "other": "#aaaaaa",
    }
    figures: dict[str, go.Figure] = {}

    for var_id in get_var_cols(data_df):
        sub = data_df[["data_id", "data_type", var_id]].dropna(subset=[var_id]).copy()
        if sub.empty:
            continue
        long_name, units = _var_label(var_id, variables_df)

        # Sort: to_benchmark always on top; within each type sort by value
        type_order = {"to_benchmark": 0, "reference": 1, "other": 2}
        sub["_type_rank"] = sub["data_type"].map(type_order).fillna(3)
        sub = sub.sort_values(["_type_rank", var_id], ascending=[True, True])

        colors = [type_color.get(str(t), "#aaaaaa") for t in sub["data_type"]]
        # Strip long id prefixes for display (keep the model name after last _)
        y_labels = sub["data_id"].tolist()

        # Compute a sensible x-axis padding so text labels are not clipped
        x_vals = sub[var_id].tolist()
        x_range_pad = (
            (max(x_vals) - min(x_vals)) * 0.15
            if len(x_vals) > 1
            else abs(x_vals[0]) * 0.15
        )

        bars = go.Bar(
            x=x_vals,
            y=y_labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.3g}" for v in x_vals],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}: %{x:.4g}<extra></extra>",
        )

        # Legend pseudo-traces (invisible, just for legend)
        legend_traces = []
        for dtype, color in type_color.items():
            if dtype in sub["data_type"].values:
                legend_traces.append(
                    go.Bar(
                        x=[None],
                        y=[None],
                        orientation="h",
                        name={
                            "to_benchmark": "Model",
                            "reference": "Reference",
                            "other": "Other",
                        }.get(dtype, dtype),
                        marker_color=color,
                        showlegend=True,
                    ),
                )

        fig = go.Figure([bars, *legend_traces])
        title_str = f"<b>{long_name}</b>" + (f"  [{units}]" if units else "")
        fig.update_layout(
            **_base_layout(
                title={
                    "text": title_str,
                    "font": {"size": 14},
                    "x": 0.02,
                    "xanchor": "left",
                },
                height=max(300 + 18 * len(sub), 400),
                margin={"t": 70, "b": 50, "l": 240, "r": 80},
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
            ),
            xaxis={
                **_AXIS_STYLE,
                "title": {"text": units or var_id},
                "range": [min(x_vals) - x_range_pad, max(x_vals) + x_range_pad],
            },
            yaxis={**_AXIS_STYLE, "title": ""},
            barmode="overlay",
        )
        figures[var_id] = fig

    return figures


def build_histogram_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    _metrics_df: pd.DataFrame | None = None,
) -> dict[str, go.Figure]:
    """Histogram diagnostics, one figure per density variable, all datasets overlaid.

    Each density variable (e.g., density_pr_global) is plotted as a line chart
    showing density vs bin center. Reference datasets are shown in black; the
    model being benchmarked in blue (with the seaborn palette for multiple
    benchmark models); all other datasets in grey.
    """
    figures: dict[str, go.Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())
    density_cols = get_density_cols(data_df)

    for density_col in density_cols:
        var_id = density_col.replace("density_", "")
        sub = data_df[["data_id", "data_type", var_id, density_col]].dropna(
            subset=[density_col],
        )
        if sub.empty:
            continue
        long_name, units = _var_label(var_id, variables_df)

        fig = go.Figure()
        color_idx = 0

        for dataset in all_datasets:
            ds_sub = sub[sub["data_id"] == dataset].sort_values(var_id)
            if ds_sub.empty:
                continue
            dtype = ds_sub["data_type"].iloc[0]

            if dtype == "reference":
                line_color = "black"
                dash = "solid"
                width = 2
            elif dtype == "to_benchmark":
                line_color = _SEABORN_COLORS[color_idx % len(_SEABORN_COLORS)]
                dash = "solid"
                width = 2
                color_idx += 1
            else:
                line_color = "#888"
                dash = "dash"
                width = 1.5

            fig.add_trace(
                go.Scatter(
                    x=ds_sub[var_id].values,
                    y=ds_sub[density_col].values,
                    mode="lines",
                    name=str(dataset),
                    legendgroup=str(dataset),
                    line={"color": line_color, "dash": dash, "width": width},
                    hovertemplate=(
                        f"<b>{dataset}</b><br>"
                        f"{var_id}: %{{x:.4g}} {units}<br>"
                        f"density: %{{y:.4g}}<extra></extra>"
                    ),
                ),
            )

        title_str = f"<b>{long_name}</b>" + (f"  [{units}]" if units else "")
        fig.update_layout(
            **_base_layout(
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
                height=440,
            ),
            xaxis={
                **_AXIS_STYLE,
                "title": {"text": units or var_id},
            },
            yaxis={
                **_AXIS_STYLE,
                "title": {"text": "Density"},
            },
        )
        figures[var_id] = fig

    return figures


def build_hovmoller_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,
) -> dict[str, go.Figure]:
    """Hovmöller diagrams: longitude × time heatmaps of meridional-mean anomalies.

    One figure per variable; one heatmap per dataset (stacked vertically).
    Default: x-axis = longitude, y-axis = time.
    An axis-swap toggle button is embedded in the figure.
    """
    figures: dict[str, go.Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())

    lon_ticks = {
        "tickvals": [0, 60, 120, 180, 240, 300, 360],
        "ticktext": ["0°", "60°E", "120°E", "180°", "120°W", "60°W", "360°"],
    }

    for var_id in get_var_cols(data_df):
        if data_df[var_id].isna().all():
            continue
        long_name, units = _var_label(var_id, variables_df)
        cs = _hovmoller_colorscale_for(var_id)

        valid_datasets = [
            ds
            for ds in all_datasets
            if not data_df[data_df["data_id"] == ds][var_id].isna().all()
        ]
        if not valid_datasets:
            continue

        def _pivot_for(dataset: str, _vid: str = var_id) -> pd.DataFrame | None:
            sub = data_df[data_df["data_id"] == dataset].dropna(subset=[_vid])
            try:
                p = sub.pivot_table(
                    index="time",       # rows = time
                    columns="longitude",  # columns = longitude
                    values=_vid,
                    aggfunc="mean",
                )
                p = p.dropna(how="all").dropna(axis=1, how="all")
                p.index = p.index.astype(str).str[:10]  # YYYY-MM-DD
            except Exception:
                return None
            return p if not p.empty else None

        # Collect pivot tables in dataset order
        pivots: list[pd.DataFrame] = []
        ds_order: list[str] = []
        for ds in valid_datasets:
            pv = _pivot_for(ds)
            if pv is not None:
                pivots.append(pv)
                ds_order.append(ds)
        if not pivots:
            continue

        all_vals: list[float] = []
        for pv in pivots:
            v = pv.values[np.isfinite(pv.values)]
            all_vals.extend(v.tolist())
        arr = np.array(all_vals)
        abs_max = float(
            max(abs(float(np.percentile(arr, 2))), abs(float(np.percentile(arr, 98)))),
        )
        if abs_max == 0:
            abs_max = 1.0
        zmin, zmax = -abs_max, abs_max

        nrows = len(pivots)
        # Do NOT share x-axes so the transpose button can update each trace's
        # x/y independently.
        fig = make_subplots(
            rows=nrows,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.06,
            subplot_titles=[str(ds) for ds in ds_order],
        )

        for row_i, (dataset, pv) in enumerate(zip(ds_order, pivots), start=1):
            fig.add_trace(
                go.Heatmap(
                    x=pv.columns.tolist(),   # longitude (default)
                    y=pv.index.tolist(),     # time strings (default)
                    z=pv.values.tolist(),
                    colorscale=cs,
                    zmin=zmin,
                    zmax=zmax,
                    colorbar={
                        "title": {"text": units or var_id, "side": "right"},
                        "len": 1 / nrows,
                        "y": 1 - (row_i - 0.5) / nrows,
                        "yanchor": "middle",
                        "thickness": 14,
                        "tickfont": {"size": 10},
                    },
                    showscale=(row_i == 1),
                    hovertemplate=(
                        "lon: %{x:.1f}°<br>time: %{y}<br>"
                        f"{var_id}: %{{z:.4g}} {units}<extra>{dataset}</extra>"
                    ),
                ),
                row=row_i,
                col=1,
            )
            fig.update_yaxes(
                title_text="Time",
                autorange="reversed",
                **_AXIS_STYLE,
                row=row_i,
                col=1,
            )
            fig.update_xaxes(
                title_text="Longitude" if row_i == nrows else "",
                **lon_ticks,
                **_AXIS_STYLE,
                row=row_i,
                col=1,
            )

        # ---- Axis-swap toggle button ------------------------------------
        # Axis layout keys: row 1 → "xaxis"/"yaxis", row 2 → "xaxis2"/"yaxis2", …
        def _ax(prefix: str, i: int) -> str:
            return prefix if i == 0 else f"{prefix}{i + 1}"

        # Trace-data args for each orientation
        lon_x_trace = {
            "x": [pv.columns.tolist() for pv in pivots],
            "y": [pv.index.tolist() for pv in pivots],
            "z": [pv.values.tolist() for pv in pivots],
            "hovertemplate": [
                f"lon: %{{x:.1f}}°<br>time: %{{y}}<br>{var_id}: %{{z:.4g}} {units}<extra>{ds}</extra>"
                for ds in ds_order
            ],
        }
        time_x_trace = {
            "x": [pv.index.tolist() for pv in pivots],
            "y": [pv.columns.tolist() for pv in pivots],
            "z": [pv.values.T.tolist() for pv in pivots],
            "hovertemplate": [
                f"time: %{{x}}<br>lon: %{{y:.1f}}°<br>{var_id}: %{{z:.4g}} {units}<extra>{ds}</extra>"
                for ds in ds_order
            ],
        }

        # Layout args for each (orientation × longitude-direction) combination.
        # Each button is fully self-contained so no client-side state is needed.
        lon_x_layout_we: dict[str, Any] = {}   # Lon × Time, W → E (default)
        lon_x_layout_ew: dict[str, Any] = {}   # Lon × Time, E → W
        time_x_layout_we: dict[str, Any] = {}  # Time × Lon, W → E
        time_x_layout_ew: dict[str, Any] = {}  # Time × Lon, E → W
        for i in range(nrows):
            xa, ya = _ax("xaxis", i), _ax("yaxis", i)
            _lon_we  = {"title": {"text": "Longitude"}, **lon_ticks, **_AXIS_STYLE}
            _lon_ew  = {"title": {"text": "Longitude"}, "autorange": "reversed", **lon_ticks, **_AXIS_STYLE}
            _time_rv = {"title": {"text": "Time"}, "autorange": "reversed", **_AXIS_STYLE}
            _time_fw = {"title": {"text": "Time"}, "autorange": True, **_AXIS_STYLE}
            lon_x_layout_we[xa]  = _lon_we;  lon_x_layout_we[ya]  = _time_rv
            lon_x_layout_ew[xa]  = _lon_ew;  lon_x_layout_ew[ya]  = _time_rv
            time_x_layout_we[xa] = _time_fw; time_x_layout_we[ya] = _lon_we
            time_x_layout_ew[xa] = _time_fw; time_x_layout_ew[ya] = _lon_ew

        _btn_style = {
            "type": "buttons",
            "direction": "left",
            "pad": {"t": 8, "b": 4},
            "bgcolor": "#eef3fa",
            "bordercolor": "#aab8cc",
            "borderwidth": 1,
            "font": {"size": 11, "color": "#1e2535"},
        }
        title_str = (
            f"<b>{long_name}</b> — Hovmöller (meridional-mean anomaly)"
            + (f"  [{units}]" if units else "")
        )
        panel_height = max(400, 320 * nrows)
        fig.update_layout(
            **_base_layout(
                title={"text": title_str, "font": {"size": 14}, "x": 0.02, "xanchor": "left"},
                margin={"t": 110, "b": 60, "l": 90, "r": 90},
                height=panel_height,
            ),
            updatemenus=[
                {
                    **_btn_style,
                    "x": 0.0, "xanchor": "left",
                    "y": 1.0, "yanchor": "bottom",
                    "buttons": [
                        {"label": " Lon × Time ", "method": "update",
                         "args": [lon_x_trace, lon_x_layout_we]},
                        {"label": " Lon × Time  E→W ", "method": "update",
                         "args": [lon_x_trace, lon_x_layout_ew]},
                    ],
                },
                {
                    **_btn_style,
                    "x": 0.55, "xanchor": "left",
                    "y": 1.0, "yanchor": "bottom",
                    "buttons": [
                        {"label": " Time × Lon ", "method": "update",
                         "args": [time_x_trace, time_x_layout_we]},
                        {"label": " Time × Lon  E→W ", "method": "update",
                         "args": [time_x_trace, time_x_layout_ew]},
                    ],
                },
            ],
        )
        figures[var_id] = fig

    return figures


def build_qbo_figures(
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None = None,  # noqa: ARG001
) -> dict[str, go.Figure]:
    """QBO: equatorial stratospheric ua on a time × pressure heatmap.

    Layout (per variable):
      Left 80%:  QBO heatmap — x = time, y = pressure (log, 100 → 1 hPa), one row per dataset
      Right 20%: QBO amplitude profile — std(ua) vs pressure, all datasets overlaid
      Bottom:    QBO index — ua at ~50 hPa vs time, all datasets overlaid
    """
    figures: dict[str, go.Figure] = {}
    all_datasets = _sort_datasets(data_df, data_df["data_id"].unique().tolist())

    # Pressure axis ticks in hPa
    plev_ticks_hpa = [100, 70, 50, 30, 20, 10, 5, 1]

    for var_id in get_var_cols(data_df):
        if data_df[var_id].isna().all():
            continue
        long_name, units = _var_label(var_id, variables_df)
        cs = _colorscale_for(var_id)  # ua → "PuOr" (diverging)

        valid_datasets = [
            ds for ds in all_datasets
            if not data_df[data_df["data_id"] == ds][var_id].isna().all()
        ]
        if not valid_datasets:
            continue

        # Symmetric colour range across all datasets
        all_vals = data_df[var_id].dropna().tolist()
        arr = np.array(all_vals)
        abs_max = float(max(abs(float(np.percentile(arr, 2))), abs(float(np.percentile(arr, 98)))))
        if abs_max == 0:
            abs_max = 1.0
        zmin, zmax = -abs_max, abs_max

        # Build pivot: index = air_pressure (Pa→hPa), columns = time → heatmap z
        def _pivot_for(ds: str, _vid: str = var_id) -> pd.DataFrame | None:
            sub = data_df[data_df["data_id"] == ds].dropna(subset=[_vid])
            if sub.empty:
                return None
            try:
                p = sub.pivot_table(
                    index="air_pressure",
                    columns="time",
                    values=_vid,
                    aggfunc="mean",
                )
                p = p.dropna(how="all").dropna(axis=1, how="all")
                p.index = (p.index / 100).round(1)              # Pa → hPa
                p.columns = p.columns.astype(str).str[:7]       # YYYY-MM
            except Exception:
                return None
            return p if not p.empty else None

        pivots = []
        ds_order = []
        for ds in valid_datasets:
            pv = _pivot_for(ds)
            if pv is not None:
                pivots.append(pv)
                ds_order.append(ds)
        if not pivots:
            continue

        n_ds = len(ds_order)
        nrows = n_ds + 1   # n_ds heatmap rows + 1 QBO-index row
        ncols = 2

        # specs: right col spans all heatmap rows; bottom row spans both cols
        specs: list[list] = []
        for i in range(n_ds):
            right = {"rowspan": n_ds} if i == 0 else None
            specs.append([{"type": "heatmap"}, right])
        specs.append([{"colspan": 2}, None])

        heatmap_frac = 0.65
        row_heights = [heatmap_frac / n_ds] * n_ds + [1.0 - heatmap_frac]

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            specs=specs,
            column_widths=[0.80, 0.20],
            row_heights=row_heights,
            vertical_spacing=0.05,
            horizontal_spacing=0.04,
        )

        # ---- QBO heatmaps (left column) ------------------------------------
        for row_i, (ds, pv) in enumerate(zip(ds_order, pivots), start=1):
            show_cb = row_i == 1
            fig.add_trace(
                go.Heatmap(
                    x=pv.columns.tolist(),   # time (YYYY-MM)
                    y=pv.index.tolist(),     # pressure (hPa)
                    z=pv.values.tolist(),
                    colorscale=cs,
                    zmin=zmin,
                    zmax=zmax,
                    showscale=show_cb,
                    colorbar={
                        "title": {"text": units or var_id, "side": "right"},
                        "len": heatmap_frac,
                        "y": 1.0 - heatmap_frac / 2,
                        "yanchor": "middle",
                        "thickness": 14,
                        "tickfont": {"size": 10},
                        "outlinewidth": 0,
                    },
                    hovertemplate=(
                        f"time: %{{x}}<br>pressure: %{{y:.0f}} hPa<br>"
                        f"{var_id}: %{{z:.3g}} {units}<extra>{ds}</extra>"
                    ),
                ),
                row=row_i,
                col=1,
            )
            fig.update_yaxes(
                type="log",
                autorange="reversed",
                tickvals=plev_ticks_hpa,
                ticktext=[str(p) for p in plev_ticks_hpa],
                title_text="Pressure (hPa)" if row_i == 1 else "",
                **_AXIS_STYLE,
                row=row_i,
                col=1,
            )
            fig.update_xaxes(
                title_text="" if row_i < n_ds else "Date",
                **_AXIS_STYLE,
                row=row_i,
                col=1,
            )
            # Dataset label annotation
            fig.add_annotation(
                text=f"<b>{ds}</b>",
                xref="x domain",
                yref="y domain",
                x=0.5,
                y=1.04,
                xanchor="center",
                yanchor="bottom",
                showarrow=False,
                font={"size": 11, "color": "#333"},
                row=row_i,
                col=1,
            )

        # ---- Amplitude profile (right column, spans all heatmap rows) ------
        color_idx = 0
        for ds in ds_order:
            sub = data_df[data_df["data_id"] == ds].dropna(subset=[var_id])
            dtype = sub["data_type"].iloc[0]
            std_profile = (
                sub.assign(plev_hpa=lambda df: (df["air_pressure"] / 100).round(1))
                .groupby("plev_hpa")[var_id]
                .std()
                .sort_index()
            )
            if std_profile.empty:
                continue
            lp = _get_line_properties(dtype, color_idx)
            fig.add_trace(
                go.Scatter(
                    x=std_profile.values.tolist(),
                    y=std_profile.index.tolist(),
                    mode="lines",
                    name=str(ds),
                    legendgroup=str(ds),
                    showlegend=False,
                    line=lp["line"],
                    hovertemplate=(
                        f"<b>{ds}</b><br>pressure: %{{y:.0f}} hPa<br>"
                        f"σ({var_id}): %{{x:.3g}} {units}<extra></extra>"
                    ),
                ),
                row=1,
                col=2,
            )
            if dtype == "to_benchmark":
                color_idx += 1
        fig.update_yaxes(
            type="log",
            autorange="reversed",
            tickvals=plev_ticks_hpa,
            ticktext=[str(p) for p in plev_ticks_hpa],
            title_text="",
            **_AXIS_STYLE,
            row=1,
            col=2,
        )
        fig.update_xaxes(
            title_text=f"σ [{units}]",
            rangemode="tozero",
            **_AXIS_STYLE,
            row=1,
            col=2,
        )
        fig.add_annotation(
            text="<b>Amplitude</b>",
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=1.04,
            xanchor="center",
            yanchor="bottom",
            showarrow=False,
            font={"size": 10, "color": "#555"},
            row=1,
            col=2,
        )

        # ---- QBO index: ua at ~50 hPa vs time (bottom full-width row) ------
        color_idx = 0
        _50hpa_pa = 5000.0
        for ds in ds_order:
            sub = data_df[data_df["data_id"] == ds].dropna(subset=[var_id])
            dtype = sub["data_type"].iloc[0]
            avail = sub["air_pressure"].unique()
            target = avail[int(np.argmin(np.abs(avail - _50hpa_pa)))]
            idx_sub = (
                sub[sub["air_pressure"] == target][["time", var_id]]
                .sort_values("time")
            )
            if idx_sub.empty:
                continue
            label_hpa = round(target / 100)
            lp = _get_line_properties(dtype, color_idx)
            fig.add_trace(
                go.Scatter(
                    x=idx_sub["time"].astype(str).str[:7].tolist(),
                    y=idx_sub[var_id].tolist(),
                    mode="lines",
                    name=str(ds),
                    legendgroup=str(ds),
                    showlegend=True,
                    line=lp["line"],
                    hovertemplate=(
                        f"<b>{ds}</b><br>%{{x}}<br>"
                        f"{var_id} @ {label_hpa} hPa: %{{y:.3g}} {units}<extra></extra>"
                    ),
                ),
                row=nrows,
                col=1,
            )
            if dtype == "to_benchmark":
                color_idx += 1
        fig.add_hline(
            y=0,
            line={"color": "rgba(0,0,0,0.40)", "width": 1, "dash": "dot"},
            row=nrows,
            col=1,
        )
        fig.update_xaxes(title_text="Date", **_AXIS_STYLE, row=nrows, col=1)
        fig.update_yaxes(
            title_text=f"{var_id} @ 50 hPa [{units}]",
            **_AXIS_STYLE,
            row=nrows,
            col=1,
        )

        title_str = (
            f"<b>{long_name}</b> — QBO (equatorial zonal mean)"
            + (f"  [{units}]" if units else "")
        )
        panel_height = max(600, 280 * n_ds + 220)
        fig.update_layout(
            **_base_layout(
                title={"text": title_str, "font": {"size": 14}, "x": 0.02, "xanchor": "left"},
                margin={"t": 80, "b": 60, "l": 80, "r": 90},
                height=panel_height,
                legend={
                    "orientation": "h",
                    "yanchor": "bottom",
                    "y": -0.15,
                    "xanchor": "center",
                    "x": 0.4,
                    "bgcolor": "rgba(255,255,255,0.85)",
                    "bordercolor": "#ddd",
                    "borderwidth": 1,
                    "font": {"size": 11},
                },
            ),
        )
        figures[var_id] = fig

    return figures


_FIGURE_BUILDERS = {
    "Map": build_map_figures,
    "TimeSeries": build_timeseries_figures,
    "AnnualCycle": build_annual_cycle_figures,
    "DiurnalCycle": build_diurnal_cycle_figures,
    "ZonalLine": build_zonal_line_figures,
    "ZonalProfile": build_zonal_profile_figures,
    "Hovmoller": build_hovmoller_figures,
    "QBO": build_qbo_figures,
    "DepthProfile": build_depth_profile_figures,
    "Histogram": build_histogram_figures,
    "Scalar": build_bar_figures,
}

_DIAG_LABELS = {
    "Map": "Map",
    "TimeSeries": "Time Series",
    "AnnualCycle": "Annual Cycle",
    "DiurnalCycle": "Diurnal Cycle",
    "ZonalLine": "Zonal Mean",
    "ZonalProfile": "Zonal Profile",
    "Hovmoller": "Hovmöller",
    "QBO": "QBO",
    "DepthProfile": "Depth Profile",
    "Histogram": "Histogram",
    "Scalar": "Scalar",
}

_DIAG_ICONS = {
    "Map": "🗺",
    "TimeSeries": "📈",
    "AnnualCycle": "📅",
    "DiurnalCycle": "🌅",
    "ZonalLine": "〰",
    "ZonalProfile": "🌡",
    "Hovmoller": "🌊",
    "QBO": "🌀",
    "DepthProfile": "〰",
    "Histogram": "📊",
    "Scalar": "📊",
}
