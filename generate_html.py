"""Generate a portable self-contained HTML benchmarking report from DuckDB database(s)."""

from __future__ import annotations

import argparse
import html as html_module
import re
from datetime import datetime, timezone
from pathlib import Path

import ibis
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
                tables[table_name] = conn.table(table_name, database=db_name).to_pandas()
        except Exception:
            continue
        if tables:
            result[db_name] = tables
    return result


def get_var_cols(data_df: pd.DataFrame) -> list[str]:
    """Return variable (non-coordinate, non-metadata) column names."""
    skip = {"Dataset", "Data Type", "latitude", "longitude", "time", "air_pressure"}
    return [c for c in data_df.columns if c not in skip]


def detect_diag_type(columns: list[str]) -> str:
    """Detect diagnostic type from coordinate columns of the data table."""
    cols = set(columns)
    if "latitude" in cols and "longitude" in cols:
        return "Map"
    if "air_pressure" in cols and "latitude" in cols:
        return "ZonalProfile"
    if "latitude" in cols:
        return "ZonalLine"
    if "time" in cols:
        return "TimeSeries"
    return "Unknown"


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------

_COLORSCALE = "RdBu_r"
_LINE_STYLES: dict[str, str] = {
    "to_benchmark": "solid",
    "reference": "dash",
    "other": "dot",
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


def build_map_figures(
    data_df: pd.DataFrame, variables_df: pd.DataFrame
) -> dict[str, go.Figure]:
    """Geographic scatter maps with coastlines; one subplot per dataset."""
    figures: dict[str, go.Figure] = {}
    var_cols = get_var_cols(data_df)

    def _sort_key(ds: str) -> int:
        dtype = data_df[data_df["Dataset"] == ds]["Data Type"].iloc[0]
        return {"reference": 0, "other": 1, "to_benchmark": 2}.get(dtype, 3)

    all_datasets = sorted(data_df["Dataset"].unique(), key=_sort_key)

    for var_id in var_cols:
        long_name, units = _var_label(var_id, variables_df)
        vmin, vmax = _vrange(data_df[var_id])

        valid_datasets = [
            ds for ds in all_datasets
            if not data_df[data_df["Dataset"] == ds][var_id].isna().all()
        ]
        if not valid_datasets:
            continue

        ncols = min(len(valid_datasets), 3)
        nrows = max(1, (len(valid_datasets) + ncols - 1) // ncols)

        specs = [[{"type": "geo"} for _ in range(ncols)] for _ in range(nrows)]
        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            specs=specs,
            subplot_titles=[str(ds) for ds in valid_datasets],
            horizontal_spacing=0.02,
            vertical_spacing=0.08,
        )

        show_colorbar = True
        for i, dataset in enumerate(valid_datasets):
            row = i // ncols + 1
            col = i % ncols + 1
            sub = data_df[data_df["Dataset"] == dataset]
            # Aggregate duplicates at the same grid cell
            sub_var = (
                sub[["latitude", "longitude", var_id]]
                .dropna(subset=[var_id])
                .groupby(["latitude", "longitude"], as_index=False)[var_id]
                .mean()
            )
            if sub_var.empty:
                continue

            fig.add_trace(
                go.Scattergeo(
                    lat=sub_var["latitude"],
                    lon=sub_var["longitude"],
                    marker=dict(
                        color=sub_var[var_id],
                        colorscale=_COLORSCALE,
                        cmin=vmin,
                        cmax=vmax,
                        size=6,
                        opacity=1.0,
                        colorbar=dict(title=units, thickness=15, len=0.6)
                        if show_colorbar
                        else None,
                        showscale=show_colorbar,
                    ),
                    mode="markers",
                    name=dataset,
                    hovertemplate=(
                        "lat: %{lat:.1f}<br>lon: %{lon:.1f}"
                        f"<br>{var_id}: %{{marker.color:.3g}} {units}"
                        "<extra>" + dataset + "</extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            show_colorbar = False

        fig.update_geos(
            showcoastlines=True,
            coastlinecolor="#333",
            coastlinewidth=1,
            showland=True,
            landcolor="whitesmoke",
            showocean=True,
            oceancolor="#ddeeff",
            showlakes=False,
            showframe=False,
            projection_type="natural earth",
        )
        fig.update_layout(
            title_text=f"{long_name} [{units}]",
            height=300 * nrows + 80,
            margin=dict(t=80, b=20, l=10, r=10),
        )
        figures[var_id] = fig

    return figures


def build_timeseries_figures(
    data_df: pd.DataFrame, variables_df: pd.DataFrame
) -> dict[str, go.Figure]:
    """Annual time series, one figure per variable, all datasets overlaid."""
    figures: dict[str, go.Figure] = {}
    var_cols = get_var_cols(data_df)

    for var_id in var_cols:
        if data_df[var_id].isna().all():
            continue
        long_name, units = _var_label(var_id, variables_df)
        fig = go.Figure()

        for dataset, grp in data_df.groupby("Dataset"):
            sub = grp[["time", var_id]].dropna(subset=[var_id]).sort_values("time")
            if sub.empty:
                continue
            dtype = grp["Data Type"].iloc[0]
            dash = _LINE_STYLES.get(dtype, "solid")
            width = 2 if dtype == "to_benchmark" else 1.5
            fig.add_trace(
                go.Scatter(
                    x=sub["time"],
                    y=sub[var_id],
                    mode="lines+markers",
                    name=str(dataset),
                    line=dict(dash=dash, width=width),
                    marker=dict(size=4),
                    hovertemplate="%{y:.3g}<extra>" + str(dataset) + "</extra>",
                )
            )

        fig.update_layout(
            title_text=f"{long_name} [{units}]",
            xaxis_title="Year",
            yaxis_title=f"{units}",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=80, b=60, l=60, r=20),
        )
        figures[var_id] = fig

    return figures


def build_zonal_line_figures(
    data_df: pd.DataFrame, variables_df: pd.DataFrame
) -> dict[str, go.Figure]:
    """Zonal mean line plots, one figure per variable, all datasets overlaid."""
    figures: dict[str, go.Figure] = {}
    var_cols = get_var_cols(data_df)

    for var_id in var_cols:
        if data_df[var_id].isna().all():
            continue
        long_name, units = _var_label(var_id, variables_df)
        fig = go.Figure()

        for dataset, grp in data_df.groupby("Dataset"):
            sub = grp[["latitude", var_id]].dropna(subset=[var_id]).sort_values("latitude")
            if sub.empty:
                continue
            dtype = grp["Data Type"].iloc[0]
            dash = _LINE_STYLES.get(dtype, "solid")
            width = 2 if dtype == "to_benchmark" else 1.5
            fig.add_trace(
                go.Scatter(
                    x=sub["latitude"],
                    y=sub[var_id],
                    mode="lines",
                    name=str(dataset),
                    line=dict(dash=dash, width=width),
                    hovertemplate="%{y:.3g}<extra>" + str(dataset) + "</extra>",
                )
            )

        fig.update_layout(
            title_text=f"{long_name} [{units}]",
            xaxis_title="Latitude",
            yaxis_title=f"{units}",
            height=420,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=80, b=60, l=60, r=20),
        )
        figures[var_id] = fig

    return figures


def build_zonal_profile_figures(
    data_df: pd.DataFrame, variables_df: pd.DataFrame
) -> dict[str, go.Figure]:
    """Zonal mean profile heatmaps (pressure × latitude), one figure per variable."""
    figures: dict[str, go.Figure] = {}
    var_cols = get_var_cols(data_df)

    def _sort_key(ds: str) -> int:
        dtype = data_df[data_df["Dataset"] == ds]["Data Type"].iloc[0]
        return {"reference": 0, "other": 1, "to_benchmark": 2}.get(dtype, 3)

    all_datasets = sorted(data_df["Dataset"].unique(), key=_sort_key)

    for var_id in var_cols:
        long_name, units = _var_label(var_id, variables_df)
        vmin, vmax = _vrange(data_df[var_id])

        valid_datasets = [
            ds for ds in all_datasets
            if not data_df[data_df["Dataset"] == ds][var_id].isna().all()
        ]
        if not valid_datasets:
            continue

        ncols = min(len(valid_datasets), 3)
        nrows = max(1, (len(valid_datasets) + ncols - 1) // ncols)

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[str(ds) for ds in valid_datasets],
            vertical_spacing=0.08,
            horizontal_spacing=0.05,
        )
        first_trace = True
        for i, dataset in enumerate(valid_datasets):
            row = i // ncols + 1
            col = i % ncols + 1
            sub = data_df[data_df["Dataset"] == dataset]
            try:
                pivot = sub.pivot_table(
                    index="air_pressure",
                    columns="latitude",
                    values=var_id,
                    aggfunc="mean",
                )
                pivot = pivot.dropna(how="all").dropna(axis=1, how="all")
            except Exception:
                continue
            if pivot.empty:
                continue
            fig.add_trace(
                go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    colorscale=_COLORSCALE,
                    zmin=vmin,
                    zmax=vmax,
                    colorbar=dict(title=units, len=0.8) if first_trace else None,
                    showscale=first_trace,
                    name=dataset,
                    hovertemplate=(
                        "lat: %{x:.1f}<br>plev: %{y:.0f}<br>value: %{z:.3g}"
                        "<extra>" + dataset + "</extra>"
                    ),
                ),
                row=row,
                col=col,
            )
            fig.update_yaxes(autorange="reversed", title_text="Pressure (Pa)", row=row, col=col)
            fig.update_xaxes(title_text="Latitude", row=row, col=col)
            first_trace = False

        fig.update_layout(
            title_text=f"{long_name} [{units}]",
            height=340 * nrows + 80,
            margin=dict(t=80, b=60, l=80, r=60),
        )
        figures[var_id] = fig

    return figures


_FIGURE_BUILDERS = {
    "Map": build_map_figures,
    "TimeSeries": build_timeseries_figures,
    "ZonalLine": build_zonal_line_figures,
    "ZonalProfile": build_zonal_profile_figures,
}

_DIAG_LABELS = {
    "Map": "Map",
    "TimeSeries": "Time Series",
    "ZonalLine": "Zonal Mean",
    "ZonalProfile": "Zonal Profile",
}


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------

def build_leaderboard_data(
    all_metrics: dict[str, dict[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build leaderboard from {tier_label: {diag_name: metrics_df}}.

    Returns (summary_df, detail_df).
    """
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

        detail[col] = detail.groupby(["tier", "diagnostic", "var_id"])[metric].transform(_norm)

    if score_cols:
        detail["_overall_score"] = detail[score_cols].mean(axis=1)
    else:
        detail["_overall_score"] = np.nan

    summary = (
        detail.groupby("Dataset")["_overall_score"]
        .mean()
        .reset_index()
        .rename(columns={"_overall_score": "Overall Score"})
        .sort_values("Overall Score", ascending=False)
    )
    summary["Rank"] = range(1, len(summary) + 1)
    summary = summary[["Rank", "Dataset", "Overall Score"]].reset_index(drop=True)

    return summary, detail


def _score_color(score: float) -> str:
    h = int(score * 120)
    return f"hsl({h},70%,50%)"


def _render_summary_table(summary_df: pd.DataFrame, table_id: str) -> str:
    medal = {1: "medal-1", 2: "medal-2", 3: "medal-3"}
    rows_html = []
    for _, row in summary_df.iterrows():
        rank = int(row["Rank"])
        dataset = html_module.escape(str(row["Dataset"]))
        score = row["Overall Score"]
        pct = f"{score * 100:.1f}"
        color = _score_color(score)
        rank_cls = medal.get(rank, "")
        bar_width = f"{score * 100:.1f}%"
        rows_html.append(
            f"<tr>"
            f"<td class='rank {rank_cls}'>{rank}</td>"
            f"<td>{dataset}</td>"
            f"<td class='score-cell bar-cell'>"
            f"  <div class='bar-bg' style='width:{bar_width};background:{color};'></div>"
            f"  <span class='bar-val' style='color:{color};font-weight:600'>{pct}%</span>"
            f"</td>"
            f"</tr>"
        )
    return (
        f"<div class='lb-table-wrap'>"
        f"<table class='lb' id='{table_id}'>"
        f"<thead><tr>"
        f"<th class='sortable' onclick='sortTable(\"{table_id}\",0,true)'>Rank</th>"
        f"<th class='sortable' onclick='sortTable(\"{table_id}\",1,false)'>Dataset</th>"
        f"<th class='sortable' onclick='sortTable(\"{table_id}\",2,true)'>Overall Score</th>"
        f"</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table></div>"
    )


def _render_detail_table(detail_df: pd.DataFrame, table_id: str) -> str:
    if detail_df.empty:
        return ""
    metric_cols = [
        c for c in ["weighted_rmse", "weighted_pearsonr", "weighted_emd"]
        if c in detail_df.columns
    ]
    str_cols = {"Dataset", "tier", "diagnostic", "var_id"}
    cols_show = ["Dataset", "tier", "diagnostic", "var_id"] + metric_cols
    # Hide "tier" column when there's only one tier
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
    """Build the full leaderboard HTML section."""
    summary_df, detail_df = build_leaderboard_data(all_metrics)

    if summary_df.empty:
        return "<p>No metrics data found.</p>"

    summary_html = _render_summary_table(summary_df, f"lb-summary{table_suffix}")
    detail_html = _render_detail_table(detail_df, f"lb-detail{table_suffix}")

    metric_note = (
        "<p style='font-size:0.83rem;color:#666;margin:0 0 20px'>"
        "Overall Score = normalized average of weighted RMSE, Pearson r, and Earth Mover's Distance "
        "across all variables and diagnostics (100% = best possible)."
        "</p>"
    )

    return (
        f"<div class='lb-section'>"
        f"<h3>Overall Ranking</h3>"
        f"{metric_note}"
        f"{summary_html}"
        f"<div class='detail-section'>"
        f"<details><summary>Full metrics table</summary>{detail_html}</details>"
        f"</div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Diagnostic section rendering
# ---------------------------------------------------------------------------

def render_diagnostic_section(scope_id: str, figures: dict[str, go.Figure]) -> str:
    """Render plots for one diagnostic.

    scope_id uniquely namespaces all element IDs (e.g. "Map" or "t0-Map").
    Figures are stored as embedded JSON and rendered lazily by renderPanel().
    """
    if not figures:
        return "<p>No data available.</p>"

    var_cols = list(figures.keys())
    first_var = var_cols[0]

    var_buttons = "".join(
        f"<button class='{'active' if v == first_var else ''}' "
        f"data-var='{v}' "
        f"onclick='showVarPanel(\"{scope_id}\", \"{v}\")'>{v}</button>"
        for v in var_cols
    )

    legend_note = (
        "<div class='legend-note'>"
        "<span><span class='line-solid'></span> Model (to benchmark)</span>"
        "<span><span class='line-dash'></span> Reference</span>"
        "<span><span class='line-dot'></span> Other</span>"
        "</div>"
    )

    plot_panels = ""
    for var_id, fig in figures.items():
        panel_id = f"plot-{scope_id}-{var_id}"
        graph_id = f"graph-{scope_id}-{var_id}"
        height = fig.layout.height or 420
        active_cls = "active" if var_id == first_var else ""
        fig_json = fig.to_json()
        plot_panels += (
            f"<div class='plot-panel {active_cls}' id='{panel_id}' data-graph='{graph_id}'>"
            f"<div id='{graph_id}' style='height:{height}px;width:100%;'></div>"
            f"<script type='application/json' id='figdata-{graph_id}'>{fig_json}</script>"
            f"</div>"
        )

    return (
        f"<div id='diag-{scope_id}'>"
        f"<div class='var-nav'>{var_buttons}</div>"
        f"{legend_note}"
        f"{plot_panels}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# CSS & JS
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0; padding: 0; background: #f5f7fa; color: #1a1a2e;
}
header {
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
  color: #e0e0e0; padding: 28px 40px;
}
header h1 { margin: 0 0 6px; font-size: 1.8rem; color: #fff; }
header p { margin: 0; font-size: 0.88rem; opacity: 0.7; }

/* Top-level navigation */
.main-nav {
  display: flex; gap: 4px; background: #fff;
  border-bottom: 2px solid #e0e4ea;
  padding: 0 32px; position: sticky; top: 0; z-index: 100;
  box-shadow: 0 2px 8px rgba(0,0,0,.06);
}
.main-nav button {
  background: none; border: none; padding: 14px 20px;
  font-size: 0.92rem; font-weight: 500; cursor: pointer;
  color: #555; border-bottom: 3px solid transparent; margin-bottom: -2px;
  transition: color .15s, border-color .15s;
}
.main-nav button:hover { color: #0f3460; }
.main-nav button.active { color: #0f3460; border-bottom-color: #0f3460; }

.tab-panel { display: none; padding: 32px 40px; }
.tab-panel.active { display: block; }

/* Diagnostic sub-navigation (within a tier) */
.sub-nav {
  display: flex; gap: 4px; flex-wrap: wrap;
  border-bottom: 1px solid #e0e4ea; margin-bottom: 24px;
}
.sub-nav button {
  background: none; border: none; padding: 10px 16px;
  font-size: 0.88rem; font-weight: 500; cursor: pointer;
  color: #555; border-bottom: 2px solid transparent; margin-bottom: -1px;
  transition: color .15s, border-color .15s;
}
.sub-nav button:hover { color: #0f3460; }
.sub-nav button.active { color: #0f3460; border-bottom-color: #0f3460; }

.diag-panel { display: none; }
.diag-panel.active { display: block; }

/* Leaderboard */
.lb-section { margin-bottom: 36px; }
.lb-section h3 { font-size: 1.05rem; color: #16213e; margin: 0 0 12px; }

.lb-table-wrap { overflow-x: auto; border-radius: 8px; box-shadow: 0 2px 12px rgba(0,0,0,.08); }
table.lb {
  border-collapse: collapse; width: 100%; min-width: 480px;
  background: #fff; font-size: 0.88rem;
}
table.lb thead th {
  background: #16213e; color: #e8eaf6; font-weight: 600;
  padding: 11px 16px; text-align: left; white-space: nowrap;
}
table.lb thead th.sortable { cursor: pointer; user-select: none; }
table.lb thead th.sortable::after { content: " ↕"; opacity: 0.5; }
table.lb thead th.asc::after { content: " ↑"; opacity: 1; }
table.lb thead th.desc::after { content: " ↓"; opacity: 1; }
table.lb tbody tr:nth-child(even) { background: #f8f9fd; }
table.lb tbody tr:hover { background: #eef2ff; }
table.lb tbody td { padding: 9px 16px; white-space: nowrap; }
td.rank { font-weight: 700; font-size: 1.0rem; text-align: center; }
td.score-cell { text-align: right; font-variant-numeric: tabular-nums; }
.medal-1 { color: #c9973e; }
.medal-2 { color: #8d9297; }
.medal-3 { color: #8c5e3c; }

.bar-cell { position: relative; }
.bar-bg {
  position: absolute; top: 25%; left: 4px;
  height: 50%; border-radius: 3px; opacity: 0.22; pointer-events: none;
}
.bar-val { position: relative; z-index: 1; }

.detail-section { margin-top: 20px; }
details > summary {
  cursor: pointer; font-weight: 600; color: #0f3460;
  padding: 10px 0; user-select: none;
}
details > summary:hover { opacity: 0.75; }

/* Variable pill navigation */
.var-nav {
  display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 20px;
}
.var-nav button {
  background: #e8eaf6; border: none; border-radius: 20px;
  padding: 6px 16px; font-size: 0.85rem; cursor: pointer;
  color: #16213e; font-weight: 500; transition: background .15s;
}
.var-nav button:hover { background: #c5cae9; }
.var-nav button.active { background: #0f3460; color: #fff; }

.plot-panel { display: none; }
.plot-panel.active { display: block; }

.legend-note {
  font-size: 0.82rem; color: #666; margin: 0 0 14px;
  display: flex; gap: 20px; flex-wrap: wrap;
}
.legend-note span { display: flex; align-items: center; gap: 6px; }
.legend-note .line-solid { display: inline-block; width: 28px; height: 3px; background: #555; }
.legend-note .line-dash {
  display: inline-block; width: 28px; height: 3px;
  background: repeating-linear-gradient(to right,#555 0 6px,transparent 6px 10px);
}
.legend-note .line-dot {
  display: inline-block; width: 28px; height: 3px;
  background: repeating-linear-gradient(to right,#555 0 2px,transparent 2px 6px);
}

.db-source {
  font-size: 0.78rem; color: #999; margin: 0 0 16px;
  font-style: italic;
}
"""

_JS = """
// Lazy Plotly rendering: plots are only initialized when their panel first
// becomes visible. Double requestAnimationFrame ensures the browser has
// computed layout (width/height) before Plotly measures the div.
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
      if (_rendered[graphId]) {
        Plotly.relayout(div, {autosize: true});
        return;
      }
      var scriptEl = document.getElementById('figdata-' + graphId);
      if (!scriptEl) return;
      var figData = JSON.parse(scriptEl.textContent);
      figData.layout.autosize = true;
      Plotly.newPlot(div, figData.data, figData.layout, {responsive: true, displayModeBar: true});
      _rendered[graphId] = true;
    });
  });
}

// Top-level tab switching (works for both flat single-tier and multi-tier layouts)
function showMainTab(tabId) {
  document.querySelectorAll('.tab-panel').forEach(function(p) { p.classList.remove('active'); });
  document.querySelectorAll('.main-nav button').forEach(function(b) { b.classList.remove('active'); });
  document.getElementById('tab-' + tabId).classList.add('active');
  document.querySelector('.main-nav button[data-tab="' + tabId + '"]').classList.add('active');
  // Find and render the first active plot panel anywhere inside this tab
  var activePanel = document.querySelector('#tab-' + tabId + ' .plot-panel.active');
  if (activePanel) renderPanel(activePanel.id);
}

// Diagnostic sub-tab switching within a tier (multi-tier layout)
function showDiagTab(tierId, diagId) {
  var tierEl = document.getElementById('tier-inner-' + tierId);
  if (!tierEl) return;
  tierEl.querySelectorAll('.diag-panel').forEach(function(p) { p.classList.remove('active'); });
  tierEl.querySelectorAll('.sub-nav button').forEach(function(b) { b.classList.remove('active'); });
  var diagEl = document.getElementById('tier-' + tierId + '-' + diagId);
  if (diagEl) diagEl.classList.add('active');
  var btn = tierEl.querySelector('.sub-nav button[data-diag="' + diagId + '"]');
  if (btn) btn.classList.add('active');
  var activePanel = document.querySelector('#tier-' + tierId + '-' + diagId + ' .plot-panel.active');
  if (activePanel) renderPanel(activePanel.id);
}

// Variable panel switching within a diagnostic (both layouts)
function showVarPanel(scopeId, varId) {
  document.querySelectorAll('#diag-' + scopeId + ' .plot-panel').forEach(function(p) { p.classList.remove('active'); });
  document.querySelectorAll('#diag-' + scopeId + ' .var-nav button').forEach(function(b) { b.classList.remove('active'); });
  var panelId = 'plot-' + scopeId + '-' + varId;
  document.getElementById(panelId).classList.add('active');
  document.querySelector('#diag-' + scopeId + ' .var-nav button[data-var="' + varId + '"]').classList.add('active');
  renderPanel(panelId);
}

// Sortable table columns
function sortTable(tableId, colIdx, numeric) {
  var table = document.getElementById(tableId);
  var tbody = table.querySelector('tbody');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var th = table.querySelectorAll('thead th')[colIdx];
  var asc = th.classList.contains('asc');
  table.querySelectorAll('thead th').forEach(function(h) { h.classList.remove('asc','desc'); });
  th.classList.add(asc ? 'desc' : 'asc');
  rows.sort(function(a, b) {
    var va = a.cells[colIdx].innerText.trim();
    var vb = b.cells[colIdx].innerText.trim();
    var cmp = numeric ? (parseFloat(va)||0) - (parseFloat(vb)||0) : va.localeCompare(vb);
    return asc ? -cmp : cmp;
  });
  rows.forEach(function(r) { tbody.appendChild(r); });
}
"""


# ---------------------------------------------------------------------------
# HTML assembly helpers
# ---------------------------------------------------------------------------

def _stem_to_label(stem: str) -> str:
    """Convert a DB filename stem to a human-readable label.

    Examples: "tier1" -> "Tier 1", "tier2_atmosphere" -> "Tier 2 Atmosphere"
    """
    s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", stem)
    return s.replace("_", " ").replace("-", " ").title()


def _plotly_js(offline: bool) -> str:
    if offline:
        from pathlib import Path as _Path
        import plotly as _plotly
        js_path = _Path(_plotly.__file__).parent / "package_data" / "plotly.min.js"
        if js_path.exists():
            return f"<script>{js_path.read_text()}</script>"
    return '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'


def _build_figures_for_tier(
    db_data: dict[str, dict[str, pd.DataFrame]],
    scope_prefix: str,
) -> tuple[dict[str, str], dict[str, pd.DataFrame]]:
    """Return (diag_sections, tier_metrics) for one database."""
    diag_sections: dict[str, str] = {}
    tier_metrics: dict[str, pd.DataFrame] = {}

    for diag_name, tables in db_data.items():
        data_df = tables.get("data", pd.DataFrame())
        metrics_df = tables.get("metrics", pd.DataFrame())
        variables_df = tables.get("variables", pd.DataFrame())

        if not data_df.empty:
            tier_metrics[diag_name] = metrics_df

        diag_type = detect_diag_type(data_df.columns.tolist())
        builder = _FIGURE_BUILDERS.get(diag_type)
        if builder is None or data_df.empty:
            diag_sections[diag_name] = "<p>No data available for this diagnostic.</p>"
            continue

        figures = builder(data_df, variables_df)
        scope_id = f"{scope_prefix}-{diag_name}" if scope_prefix else diag_name
        diag_sections[diag_name] = render_diagnostic_section(scope_id, figures)

    return diag_sections, tier_metrics


def _render_single_tier(
    label: str,
    db_path: Path,
    diag_sections: dict[str, str],
    tier_metrics: dict[str, pd.DataFrame],
) -> str:
    """Assemble flat (single-tier) HTML body content: Leaderboard + one tab per diagnostic."""
    # Leaderboard uses the flat diag_name as scope (no tier prefix)
    all_metrics = {"": tier_metrics}
    lb_html = render_leaderboard_section(all_metrics, table_suffix="-main")

    nav_buttons = (
        "<button class='active' data-tab='leaderboard' "
        "onclick='showMainTab(\"leaderboard\")'>Leaderboard</button>"
    )
    for diag_name in diag_sections:
        diag_label = _DIAG_LABELS.get(diag_name, diag_name)
        nav_buttons += (
            f"<button data-tab='{diag_name}' "
            f"onclick='showMainTab(\"{diag_name}\")'>{diag_label}</button>"
        )

    panels = (
        f"<div class='tab-panel active' id='tab-leaderboard'>"
        f"<h2 style='margin-top:0;color:#16213e'>Performance Leaderboard</h2>"
        f"{lb_html}"
        f"</div>"
    )
    for diag_name, content in diag_sections.items():
        diag_label = _DIAG_LABELS.get(diag_name, diag_name)
        panels += (
            f"<div class='tab-panel' id='tab-{diag_name}'>"
            f"<h2 style='margin-top:0;color:#16213e'>{diag_label} Diagnostic</h2>"
            f"<p class='db-source'>Source: {html_module.escape(str(db_path))}</p>"
            f"{content}"
            f"</div>"
        )

    return nav_buttons, panels


def _render_multi_tier(
    tiers: list[tuple[int, str, Path, dict[str, str], dict[str, pd.DataFrame]]],
    all_metrics: dict[str, dict[str, pd.DataFrame]],
) -> tuple[str, str]:
    """Assemble nested (multi-tier) nav buttons and panels."""
    overall_lb = render_leaderboard_section(all_metrics, table_suffix="-overall")

    nav_buttons = (
        "<button class='active' data-tab='overall-lb' "
        "onclick='showMainTab(\"overall-lb\")'>Overall Leaderboard</button>"
    )
    for tier_idx, label, db_path, diag_sections, tier_metrics in tiers:
        nav_buttons += (
            f"<button data-tab='tier-{tier_idx}' "
            f"onclick='showMainTab(\"tier-{tier_idx}\")'>"
            f"{html_module.escape(label)}</button>"
        )

    # Overall leaderboard panel
    panels = (
        f"<div class='tab-panel active' id='tab-overall-lb'>"
        f"<h2 style='margin-top:0;color:#16213e'>Overall Performance Leaderboard</h2>"
        f"{overall_lb}"
        f"</div>"
    )

    for tier_idx, label, db_path, diag_sections, tier_metrics in tiers:
        # Per-tier sub-nav: Leaderboard + each diagnostic
        tier_lb = render_leaderboard_section(
            {"": tier_metrics}, table_suffix=f"-t{tier_idx}"
        )

        sub_nav = (
            f"<button class='active' data-diag='lb' "
            f"onclick='showDiagTab({tier_idx}, \"lb\")'>Leaderboard</button>"
        )
        for diag_name in diag_sections:
            diag_label = _DIAG_LABELS.get(diag_name, diag_name)
            sub_nav += (
                f"<button data-diag='{diag_name}' "
                f"onclick='showDiagTab({tier_idx}, \"{diag_name}\")'>{diag_label}</button>"
            )

        lb_panel = (
            f"<div class='diag-panel active' id='tier-{tier_idx}-lb'>"
            f"<h3 style='margin-top:0;color:#16213e'>Leaderboard</h3>"
            f"{tier_lb}"
            f"</div>"
        )

        diag_panels = ""
        for diag_name, content in diag_sections.items():
            diag_panels += (
                f"<div class='diag-panel' id='tier-{tier_idx}-{diag_name}'>"
                f"{content}"
                f"</div>"
            )

        panels += (
            f"<div class='tab-panel' id='tab-tier-{tier_idx}'>"
            f"<h2 style='margin-top:0;color:#16213e'>{html_module.escape(label)}</h2>"
            f"<p class='db-source'>Source: {html_module.escape(str(db_path))}</p>"
            f"<div id='tier-inner-{tier_idx}'>"
            f"<nav class='sub-nav'>{sub_nav}</nav>"
            f"{lb_panel}"
            f"{diag_panels}"
            f"</div>"
            f"</div>"
        )

    return nav_buttons, panels


# ---------------------------------------------------------------------------
# Main HTML generator
# ---------------------------------------------------------------------------

def generate_html(db_paths: list[Path], *, offline: bool = True) -> str:
    """Read one or more databases and produce a self-contained HTML string."""
    multi = len(db_paths) > 1

    tiers: list[tuple[int, str, Path, dict[str, str], dict[str, pd.DataFrame]]] = []
    all_metrics: dict[str, dict[str, pd.DataFrame]] = {}

    for tier_idx, db_path in enumerate(db_paths):
        label = _stem_to_label(db_path.stem)
        db_data = read_database(db_path)
        if not db_data:
            print(f"Warning: no diagnostic data found in {db_path}, skipping.")
            continue

        scope_prefix = f"t{tier_idx}" if multi else ""
        diag_sections, tier_metrics = _build_figures_for_tier(db_data, scope_prefix)
        all_metrics[label] = tier_metrics
        tiers.append((tier_idx, label, db_path, diag_sections, tier_metrics))

    if not tiers:
        raise ValueError("No diagnostic databases found in any of the provided files.")

    if multi:
        nav_buttons, panels = _render_multi_tier(tiers, all_metrics)
    else:
        _, label, db_path, diag_sections, tier_metrics = tiers[0]
        nav_buttons, panels = _render_single_tier(label, db_path, diag_sections, tier_metrics)

    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    db_list = ", ".join(str(p) for p in db_paths)
    plotly_js = _plotly_js(offline)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ESM Benchmarking Report</title>
  <style>{_CSS}</style>
</head>
<body>

<header>
  <h1>ESM Benchmarking Report</h1>
  <p>Generated {now} &bull; {html_module.escape(db_list)}</p>
</header>

<nav class="main-nav">
  {nav_buttons}
</nav>

<main>
  {panels}
</main>

{plotly_js}
<script>{_JS}</script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a portable HTML benchmarking report from one or more DuckDB databases. "
            "Pass multiple files to produce a multi-tier report."
        ),
    )
    parser.add_argument(
        "db",
        type=Path,
        nargs="+",
        help="Path(s) to DuckDB benchmarking database (.ddb). Multiple files produce a multi-tier report.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output HTML file path (default: <first_db_stem>_report.html next to the first database).",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Load Plotly.js from CDN instead of embedding it (smaller file, requires internet).",
    )
    args = parser.parse_args()

    db_paths = [p.resolve() for p in args.db]
    for p in db_paths:
        if not p.exists():
            parser.error(f"Database file not found: {p}")

    output: Path = args.output or db_paths[0].with_name(db_paths[0].stem + "_report.html")

    for p in db_paths:
        print(f"Reading database: {p}")

    html_content = generate_html(db_paths, offline=not args.online)

    output.write_text(html_content, encoding="utf-8")
    size_kb = output.stat().st_size / 1024
    print(f"Report written to: {output} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
