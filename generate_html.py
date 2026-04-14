"""Generate a portable self-contained HTML benchmarking report from a DuckDB database."""

from __future__ import annotations

import argparse
import html as html_module
import json
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
    """Detect diagnostic type from the coordinate columns of the data table."""
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
# Figure builders — return {var_id: go.Figure}
# ---------------------------------------------------------------------------

_COLORSCALE = "RdBu_r"
_LINE_STYLES: dict[str, str] = {
    "to_benchmark": "solid",
    "reference": "dash",
    "other": "dot",
}


def _style_for_dataset(data_df: pd.DataFrame, dataset: str) -> str:
    rows = data_df[data_df["Dataset"] == dataset]["Data Type"]
    dtype = rows.iloc[0] if not rows.empty else "to_benchmark"
    return _LINE_STYLES.get(dtype, "solid")


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
    """Lat/lon heatmaps, one figure per variable, one subplot per dataset."""
    figures: dict[str, go.Figure] = {}
    var_cols = get_var_cols(data_df)

    # Sort: reference/other first so model appears last (easier comparison)
    def _sort_key(ds: str) -> int:
        dtype = data_df[data_df["Dataset"] == ds]["Data Type"].iloc[0]
        return {"reference": 0, "other": 1, "to_benchmark": 2}.get(dtype, 3)

    all_datasets = sorted(data_df["Dataset"].unique(), key=_sort_key)

    for var_id in var_cols:
        long_name, units = _var_label(var_id, variables_df)
        vmin, vmax = _vrange(data_df[var_id])

        # Only include datasets that actually have data for this variable
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
            row, col = divmod(i, ncols)
            row += 1
            col += 1
            sub = data_df[data_df["Dataset"] == dataset]
            try:
                pivot = sub.pivot_table(
                    index="latitude", columns="longitude", values=var_id, aggfunc="mean"
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
                    hovertemplate="lon: %{x:.1f}<br>lat: %{y:.1f}<br>value: %{z:.3g}<extra>" + dataset + "</extra>",
                ),
                row=row,
                col=col,
            )
            first_trace = False

        fig.update_layout(
            title_text=f"{long_name} [{units}]",
            height=320 * nrows + 80,
            margin=dict(t=80, b=40, l=40, r=60),
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
    """Zonal mean profile heatmaps (lat × pressure), one figure per variable."""
    figures: dict[str, go.Figure] = {}
    var_cols = get_var_cols(data_df)

    def _sort_key(ds: str) -> int:
        dtype = data_df[data_df["Dataset"] == ds]["Data Type"].iloc[0]
        return {"reference": 0, "other": 1, "to_benchmark": 2}.get(dtype, 3)

    all_datasets = sorted(data_df["Dataset"].unique(), key=_sort_key)

    for var_id in var_cols:
        long_name, units = _var_label(var_id, variables_df)
        vmin, vmax = _vrange(data_df[var_id])

        # Only include datasets that actually have data for this variable
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
            row, col = divmod(i, ncols)
            row += 1
            col += 1
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
                    hovertemplate="lat: %{x:.1f}<br>plev: %{y:.0f}<br>value: %{z:.3g}<extra>" + dataset + "</extra>",
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
    all_metrics: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary_df, detail_df) for the leaderboard.

    summary_df has one row per Dataset with an overall score.
    detail_df has one row per (Dataset, diagnostic, var_id) with raw metrics.
    """
    dfs = []
    for diag_name, mdf in all_metrics.items():
        if mdf.empty:
            continue
        m = mdf.copy()
        m["diagnostic"] = diag_name
        dfs.append(m)

    if not dfs:
        return pd.DataFrame(), pd.DataFrame()

    detail = pd.concat(dfs, ignore_index=True)

    # Normalize each metric within (diagnostic, var_id) group so models are
    # compared fairly within the same variable.
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

        detail[col] = detail.groupby(["diagnostic", "var_id"])[metric].transform(_norm)

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


# ---------------------------------------------------------------------------
# HTML rendering helpers
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

/* Score bar inside cell */
.bar-cell { position: relative; }
.bar-bg {
  position: absolute; top: 25%; left: 4px;
  height: 50%; border-radius: 3px; opacity: 0.22; pointer-events: none;
}
.bar-val { position: relative; z-index: 1; }

/* Detail table */
.detail-section { margin-top: 20px; }
details > summary {
  cursor: pointer; font-weight: 600; color: #0f3460;
  padding: 10px 0; user-select: none;
}
details > summary:hover { opacity: 0.75; }

/* Plots */
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
"""

_JS_TABS = """
// Lazy Plotly rendering: plots are only initialized when their panel first becomes visible.
// This avoids the zero-width problem that occurs when Plotly renders inside display:none containers.
// requestAnimationFrame ensures the browser has applied display:block before Plotly measures the div.
var _rendered = {};

function renderPanel(panelId) {
  if (typeof Plotly === 'undefined') return;
  var panel = document.getElementById(panelId);
  if (!panel) return;
  var graphId = panel.getAttribute('data-graph');
  if (!graphId) return;
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
}

function showMainTab(tabId) {
  document.querySelectorAll('.tab-panel').forEach(function(p) { p.classList.remove('active'); });
  document.querySelectorAll('.main-nav button').forEach(function(b) { b.classList.remove('active'); });
  document.getElementById('tab-' + tabId).classList.add('active');
  document.querySelector('.main-nav button[data-tab="' + tabId + '"]').classList.add('active');
  var activePanel = document.querySelector('#tab-' + tabId + ' .plot-panel.active');
  if (activePanel) renderPanel(activePanel.id);
}

function showVarPanel(diagId, varId) {
  document.querySelectorAll('#diag-' + diagId + ' .plot-panel').forEach(function(p) { p.classList.remove('active'); });
  document.querySelectorAll('#diag-' + diagId + ' .var-nav button').forEach(function(b) { b.classList.remove('active'); });
  var panelId = 'plot-' + diagId + '-' + varId;
  document.getElementById(panelId).classList.add('active');
  document.querySelector('#diag-' + diagId + ' .var-nav button[data-var="' + varId + '"]').classList.add('active');
  renderPanel(panelId);
}

// Sortable table
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


def _score_color(score: float) -> str:
    """Map 0–1 score to an HSL color (red→yellow→green)."""
    h = int(score * 120)  # 0=red, 120=green
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
    """Render per-variable, per-diagnostic metrics table."""
    if detail_df.empty:
        return ""
    metric_cols = [c for c in ["weighted_rmse", "weighted_pearsonr", "weighted_emd"] if c in detail_df.columns]
    score_cols = [f"_score_{m}" for m in metric_cols if f"_score_{m}" in detail_df.columns]

    cols_show = ["Dataset", "diagnostic", "var_id"] + metric_cols
    sub = detail_df[cols_show].copy()

    # Build header
    header_cells = "".join(
        f"<th class='sortable' onclick='sortTable(\"{table_id}\",{i},{ 'false' if c in ['Dataset','diagnostic','var_id'] else 'true' })'>"
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
    all_metrics: dict[str, pd.DataFrame],
) -> str:
    """Build the full leaderboard HTML section."""
    summary_df, detail_df = build_leaderboard_data(all_metrics)

    if summary_df.empty:
        return "<p>No metrics data found in this database.</p>"

    summary_html = _render_summary_table(summary_df, "lb-summary")
    detail_html = _render_detail_table(detail_df, "lb-detail")

    metric_note = (
        "<p style='font-size:0.83rem;color:#666;margin:0 0 20px'>"
        "Overall Score = normalized average of weighted RMSE, Pearson r, and Earth Mover's Distance "
        "across all variables and diagnostics (100% = best possible)."
        "</p>"
    )

    detail_section = (
        f"<div class='detail-section'>"
        f"<details><summary>Full metrics table</summary>"
        f"{detail_html}"
        f"</details></div>"
    )

    return (
        f"<div class='lb-section'>"
        f"<h3>Overall Ranking</h3>"
        f"{metric_note}"
        f"{summary_html}"
        f"{detail_section}"
        f"</div>"
    )


def render_diagnostic_section(
    diag_name: str,
    data_df: pd.DataFrame,
    variables_df: pd.DataFrame,
    figures: dict[str, go.Figure],
) -> str:
    """Render the plots tab content for one diagnostic.

    Figures are stored as JSON and rendered lazily by renderPanel() when their
    panel first becomes visible. This avoids the Plotly zero-width problem that
    occurs when plots are initialized inside display:none containers.
    """
    if not figures:
        return "<p>No data available.</p>"

    var_cols = list(figures.keys())
    first_var = var_cols[0]

    # Variable nav buttons
    var_buttons = "".join(
        f"<button class='{'active' if v == first_var else ''}' "
        f"data-var='{v}' "
        f"onclick='showVarPanel(\"{diag_name}\", \"{v}\")'>{v}</button>"
        for v in var_cols
    )

    # Legend note
    legend_note = (
        "<div class='legend-note'>"
        "<span><span class='line-solid'></span> Model (to benchmark)</span>"
        "<span><span class='line-dash'></span> Reference</span>"
        "<span><span class='line-dot'></span> Other</span>"
        "</div>"
    )

    # Plot panels — embed figure JSON, render on demand via renderPanel()
    plot_panels = ""
    for var_id, fig in figures.items():
        panel_id = f"plot-{diag_name}-{var_id}"
        graph_id = f"graph-{diag_name}-{var_id}"
        height = fig.layout.height or 420
        active_cls = "active" if var_id == first_var else ""
        fig_json = fig.to_json()  # compact JSON; layout.autosize will be set by renderPanel()
        plot_panels += (
            f"<div class='plot-panel {active_cls}' id='{panel_id}' data-graph='{graph_id}'>"
            f"<div id='{graph_id}' style='height:{height}px;width:100%;'></div>"
            f"<script type='application/json' id='figdata-{graph_id}'>{fig_json}</script>"
            f"</div>"
        )

    return (
        f"<div id='diag-{diag_name}'>"
        f"<div class='var-nav'>{var_buttons}</div>"
        f"{legend_note}"
        f"{plot_panels}"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Main HTML assembly
# ---------------------------------------------------------------------------

def generate_html(db_path: Path, *, offline: bool = False) -> str:
    """Read the database and produce a self-contained HTML string."""
    db_data = read_database(db_path)

    if not db_data:
        raise ValueError(f"No diagnostic databases found in {db_path}")

    # Collect metrics and build figures per diagnostic
    all_metrics: dict[str, pd.DataFrame] = {}
    diag_sections: dict[str, str] = {}  # diag_name -> HTML

    for diag_name, tables in db_data.items():
        data_df = tables.get("data", pd.DataFrame())
        metrics_df = tables.get("metrics", pd.DataFrame())
        variables_df = tables.get("variables", pd.DataFrame())

        if not data_df.empty:
            all_metrics[diag_name] = metrics_df

        diag_type = detect_diag_type(data_df.columns.tolist())
        builder = _FIGURE_BUILDERS.get(diag_type)
        if builder is None or data_df.empty:
            diag_sections[diag_name] = "<p>No data available for this diagnostic.</p>"
            continue

        figures = builder(data_df, variables_df)
        diag_sections[diag_name] = render_diagnostic_section(
            diag_name, data_df, variables_df, figures
        )

    leaderboard_html = render_leaderboard_section(all_metrics)

    # Build navigation
    all_diags = list(diag_sections.keys())
    nav_buttons = "<button class='active' data-tab='leaderboard' onclick='showMainTab(\"leaderboard\")'>Leaderboard</button>"
    for d in all_diags:
        label = _DIAG_LABELS.get(d, d)
        nav_buttons += f"<button data-tab='{d}' onclick='showMainTab(\"{d}\")'>{label}</button>"

    # Build tab panels
    panels_html = (
        f"<div class='tab-panel active' id='tab-leaderboard'>"
        f"<h2 style='margin-top:0;color:#16213e'>Performance Leaderboard</h2>"
        f"{leaderboard_html}"
        f"</div>"
    )
    for d, content in diag_sections.items():
        label = _DIAG_LABELS.get(d, d)
        panels_html += (
            f"<div class='tab-panel' id='tab-{d}'>"
            f"<h2 style='margin-top:0;color:#16213e'>{label} Diagnostic</h2>"
            f"{content}"
            f"</div>"
        )

    # Plotly JS
    if offline:
        import plotly
        js_path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        if js_path.exists():
            plotly_js = f"<script>{js_path.read_text()}</script>"
        else:
            plotly_js = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    else:
        plotly_js = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'

    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

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
  <p>Generated {now} &bull; Database: {html_module.escape(str(db_path))}</p>
</header>

<nav class="main-nav">
  {nav_buttons}
</nav>

<main>
  {panels_html}
</main>

{plotly_js}
<script>{_JS_TABS}</script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a portable HTML benchmarking report from a DuckDB database.",
    )
    parser.add_argument("db", type=Path, help="Path to DuckDB benchmarking database (.ddb)")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output HTML file path (default: <db_stem>_report.html next to the database)",
    )
    parser.add_argument(
        "--offline", action="store_true",
        help="Embed Plotly.js in the HTML for fully offline use (increases file size significantly)",
    )
    args = parser.parse_args()

    db_path: Path = args.db.resolve()
    if not db_path.exists():
        parser.error(f"Database file not found: {db_path}")

    output: Path = args.output or db_path.with_name(db_path.stem + "_report.html")

    print(f"Reading database: {db_path}")
    html_content = generate_html(db_path, offline=args.offline)

    output.write_text(html_content, encoding="utf-8")
    size_kb = output.stat().st_size / 1024
    print(f"Report written to: {output} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
