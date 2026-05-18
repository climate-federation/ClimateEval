"""Leaderboard data building and scoreboard figure for the benchmarking report."""

from __future__ import annotations

import html as html_module

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ._figures import _base_layout


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
) -> go.Figure | None:
    """ClimateBench-style performance scoreboard heatmap.

    Rows = models (`data_id`), columns = variables (`var_id`).
    Each cell shows the % deviation from the median RMSE:

        deviation_pct = (model_rmse - median_rmse) / median_rmse * 100

    Negative values (blue) = better than median; positive (red) = worse.
    An "Overall" column shows the mean deviation across all variables.
    Models are sorted by mean absolute deviation (best first).

    Only uses `weighted_rmse`. Returns None if `weighted_rmse` is not available.
    """
    dfs = []
    for tier_label, tier_metrics in all_metrics.items():
        for diag_name, mdf in tier_metrics.items():
            if mdf is None or mdf.empty:
                continue
            m = mdf.copy()
            m["_diag"] = diag_name
            dfs.append(m)
    if not dfs:
        return None

    detail = pd.concat(dfs, ignore_index=True)

    if "weighted_rmse" not in detail.columns:
        return None

    # Compute median RMSE per (diag, var_id) across all data_ids, then
    # deviation_pct = (model_rmse - median_rmse) / median_rmse * 100
    rmse_df = detail[["_diag", "var_id", "data_id", "weighted_rmse"]].dropna(
        subset=["weighted_rmse"],
    )

    median_rmse = (
        rmse_df.groupby(["_diag", "var_id"])["weighted_rmse"]
        .median()
        .rename("median_rmse")
    )
    rmse_df = rmse_df.join(median_rmse, on=["_diag", "var_id"])
    rmse_df["deviation_pct"] = (
        (rmse_df["weighted_rmse"] - rmse_df["median_rmse"])
        / rmse_df["median_rmse"].abs().clip(lower=1e-12)
        * 100
    )

    # One deviation per (data_id, var_id) averaged over diagnostics
    pivot = (
        rmse_df.groupby(["data_id", "var_id"])["deviation_pct"].mean().unstack("var_id")
    )
    if pivot.empty:
        return None

    # Add "Overall" column (mean deviation across all variables)
    pivot["Overall"] = pivot.mean(axis=1)

    # Sort models by mean absolute deviation (best = smallest MAD first)
    pivot = pivot.loc[
        pivot.drop(columns="Overall").abs().mean(axis=1).sort_values().index
    ]

    z = pivot.values.tolist()
    x_labels = [str(c) for c in pivot.columns]
    y_labels = [str(i) for i in pivot.index]

    # Symmetric color range clamped at ±50% or the actual data range
    finite_vals = pivot.values[np.isfinite(pivot.values)]
    if len(finite_vals) > 0:
        max_abs = float(np.nanpercentile(np.abs(finite_vals), 98))
        max_abs = max(max_abs, 1.0)
        clamp = min(max_abs, 50.0)
    else:
        clamp = 50.0

    # Cell text: rounded deviation with % sign, blank for NaN
    text_annots = [
        [
            (f"+{v:.0f}%" if v > 0 else f"{v:.0f}%") if np.isfinite(v) else ""
            for v in row
        ]
        for row in pivot.values
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=[[0, "#2166ac"], [0.5, "#ffffff"], [1, "#d6604d"]],
            zmid=0,
            zmin=-clamp,
            zmax=clamp,
            text=text_annots,
            texttemplate="%{text}",
            textfont={"size": 9},
            hovertemplate="%{y}<br><b>%{x}</b>: %{z:.1f}%<extra></extra>",
            colorbar={
                "title": {"text": "% deviation<br>from median", "side": "right"},
                "thickness": 14,
                "len": 0.8,
                "ticksuffix": "%",
            },
            xgap=2,
            ygap=2,
        ),
    )

    # Highlight the "Overall" column with a distinct border via a shape
    overall_col_idx = len(x_labels) - 1
    fig.update_layout(
        **_base_layout(
            title={
                "text": "<b>Performance Scoreboard</b>"
                " <span style='font-size:0.8em;font-weight:400'>"
                "(% deviation from median RMSE, blue = better)</span>",
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
