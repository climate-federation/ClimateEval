"""Leaderboard data building and scoreboard figure for the benchmarking report."""

from __future__ import annotations

import html as html_module
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from climateeval.report._figures import _base_layout


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


def build_scoreboard_figure(
    all_metrics: dict[str, dict[str, pd.DataFrame]],
    metric: Literal["weighted_emd", "weighted_rmse"] = "weighted_emd",
) -> go.Figure | None:
    """ClimateBench-style performance scoreboard heatmap.

    Rows = models (`data_id`), columns = variables (`var_id`).

    Each cell shows a relative score calculated as

        relative score = 1 - model_metric / median_metric

    Positive values (blue) = better than median; negative (red) = worse. A
    "Mean relative score" column shows the mean relative score across all variables.
    Models are sorted by mean relative score (best first).

    Returns None if `metric` is not available.
    """
    dfs = []
    for tier_metrics in all_metrics.values():
        for diag_name, mdf in tier_metrics.items():
            if mdf is None or mdf.empty:
                continue
            m = mdf.copy()
            m["_diag"] = diag_name
            dfs.append(m)
    if not dfs:
        return None

    detail = pd.concat(dfs, ignore_index=True)

    if metric not in detail.columns:
        return None

    # Compute median metric per (diag, var_id) across all data_ids, then
    # relative score = 1 - model_metric / median_metric
    metric_df = detail[["_diag", "var_id", "data_id", metric]].dropna(
        subset=[metric],
    )
    median_metric = (
        metric_df.groupby(["_diag", "var_id"])[metric].median().rename("median_metric")
    )
    metric_df = metric_df.join(median_metric, on=["_diag", "var_id"])
    metric_df["relative score"] = (
        (1.0 - metric_df[metric] / metric_df["median_metric"]).abs().clip(lower=1e-12)
    )

    # One relative score per (data_id, var_id) averaged over diagnostics
    pivot = (
        metric_df.groupby(["data_id", "var_id"])["relative score"]
        .mean()
        .unstack("var_id")
    )
    if pivot.empty:
        return None

    # Add "Mean relative score" column (mean relative score across all variables)
    pivot["Mean relative score"] = pivot.mean(axis=1)

    # Sort models by mean absolute relative score (best = smallest MAD first)
    pivot = pivot.loc[pivot["Mean relative score"].sort_values(ascending=True).index]

    single_model = len(pivot.index) < 2

    if single_model:
        # Only 1 model: relative score is always 0 → show raw metric instead.
        raw_pivot = (
            metric_df.groupby(["data_id", "var_id"])[metric].mean().unstack("var_id")
        )
        raw_pivot["Mean relative score"] = raw_pivot.mean(axis=1)
        z = raw_pivot.values.tolist()
        x_labels = [str(c) for c in raw_pivot.columns]
        y_labels = [str(i) for i in raw_pivot.index]
        finite_vals = raw_pivot.values[np.isfinite(raw_pivot.values)]
        zmax = float(np.nanpercentile(finite_vals, 98)) if len(finite_vals) > 0 else 1.0
        zmax = max(zmax, 1e-6)
        text_annots = [
            [f"{v:.3g}" if np.isfinite(v) else "" for v in row]
            for row in raw_pivot.values
        ]
        heatmap_kwargs = dict(zmin=0, zmax=zmax)
        colorbar_title = f"{metric.upper()}<br>(lower = better)"
        title_note = f"(weighted {metric.upper()} per variable, blue = lower = better)"
        hover_suffix = "<extra></extra>"
        hover_tmpl = "%{y}<br><b>%{x}</b>: %{z:.4g}" + hover_suffix
        colorbar_ticks = {}
    else:
        # Multiple models: show relative score.
        z = pivot.values.tolist()
        x_labels = [str(c) for c in pivot.columns]
        y_labels = [str(i) for i in pivot.index]
        finite_vals = pivot.values[np.isfinite(pivot.values)]
        text_annots = [
            [f"{v:.1%}" if np.isfinite(v) else "" for v in row] for row in pivot.values
        ]
        mintext = "▼-1" if len(finite_vals) > 0 and np.min(finite_vals) < 1.0 else "-1"
        colorbar_ticks = {
            "tickvals": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "ticktext": [mintext, "-0.5", "0", "0.5", "1.0"],
        }
        heatmap_kwargs = dict(zmin=-1.0, zmax=1.0)
        colorbar_title = "Relative score (blue = better)"
        title_note = f"Relative score = 1 - weighted {metric.upper()} / median weighted {metric.upper()}"
        hover_tmpl = "%{y}<br><b>%{x}</b>: %{z:.1f}%<extra></extra>"

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale="RdBu",
            **heatmap_kwargs,
            text=text_annots,
            texttemplate="%{text}",
            textfont={"size": 9},
            hovertemplate=hover_tmpl,
            colorbar={
                "title": {"text": colorbar_title, "side": "right"},
                "thickness": 14,
                "len": 0.8,
                **colorbar_ticks,
            },
            xgap=2,
            ygap=2,
        ),
    )

    overall_col_idx = len(x_labels) - 1
    fig.update_layout(
        **_base_layout(
            title={
                "text": f"<b>Performance Scoreboard</b>"
                f" <span style='font-size:0.8em;font-weight:400'>{title_note}</span>",
                "font": {"size": 15},
                "x": 0.5,
                "xanchor": "center",
            },
            height=max(280 + 22 * len(y_labels), 380),
            margin={"t": 70, "b": 120, "l": 260, "r": 80},
        ),
        xaxis={
            "tickangle": -40,
            "side": "bottom",
            "tickfont": {"size": 10},
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "tickfont": {"size": 10},
            "showgrid": False,
            "zeroline": False,
        },
        shapes=[
            {
                "type": "rect",
                "xref": "x",
                "yref": "paper",
                "x0": overall_col_idx - 0.5,
                "x1": overall_col_idx + 0.5,
                "y0": 0,
                "y1": 1,
                "line": {"color": "#333", "width": 2},
                "fillcolor": "rgba(0,0,0,0)",
            },
        ],
    )
    return fig
