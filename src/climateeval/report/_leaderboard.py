"""Leaderboard data building and scoreboard figure for the benchmarking report."""

from __future__ import annotations

import html as html_module
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from climateeval.report._figures import _base_layout

if TYPE_CHECKING:
    from collections.abc import Iterable


# Metric descriptions shown as hover tooltips next to the column headers.
METRIC_INFO: dict[str, tuple[str, str]] = {
    "weighted_rmse": (
        "Weighted RMSE",
        "Area-weighted root-mean-square error of the dataset against the "
        "reference, in the variable's units. Lower is better.",
    ),
    "weighted_pearsonr": (
        "Pearson r",
        "Area-weighted Pearson correlation between the dataset and the "
        "reference. 1 means identical patterns; higher is better.",
    ),
    "weighted_emd": (
        "EMD",
        "Area-weighted Earth Mover's Distance between the value distributions "
        "of the dataset and the reference, in the variable's units. Lower is "
        "better.",
    ),
}
SCORE_INFO: dict[str, tuple[str, str]] = {
    "overall": (
        "Overall score",
        "Mean of the min-max normalised weighted RMSE, Pearson r and EMD over "
        "every diagnostic and variable in which the dataset was scored. "
        "100 % is the best of all datasets on every metric.",
    ),
    "relative": (
        "Relative score",
        "1 - metric / median metric across all datasets, per variable and "
        "averaged over diagnostics. Blue is better than the median, red worse; "
        "the boxed column is the mean over variables.",
    ),
}
ROLE_LABELS: dict[str, str] = {
    "to_benchmark": "Model",
    "reference": "Reference",
    "other": "Comparison",
}


def info_icon(text: str) -> str:
    """Render a small hoverable/focusable "i" icon carrying an explanation."""
    safe = html_module.escape(text)
    return (
        f"<span class='info-tip' tabindex='0' role='img' aria-label='{safe}' "
        f"onclick='event.stopPropagation()'>i"
        f"<span class='tip' aria-hidden='true'>{safe}</span></span>"
    )


def dataset_roles(
    db_data_by_tier: Iterable[dict[str, dict[str, pd.DataFrame]]],
) -> dict[str, dict[str, str]]:
    """Map every dataset id to its role and display name across all tiers.

    Returns ``{data_id: {"role": to_benchmark|reference|other, "name": ...}}``.
    A dataset benchmarked in any tier counts as the benchmarked model.
    """
    roles: dict[str, dict[str, str]] = {}
    for db_data in db_data_by_tier:
        for tables in db_data.values():
            names: dict[str, str] = {}
            sources = tables.get("data_sources")
            if sources is not None and not sources.empty and "name" in sources:
                names = dict(
                    zip(sources["id"].astype(str), sources["name"].astype(str))
                )
            raw = tables.get("raw_output")
            if raw is None or raw.empty or "data_type" not in raw:
                continue
            pairs = raw[["data_id", "data_type"]].drop_duplicates()
            for data_id, data_type in pairs.itertuples(index=False):
                did, role = str(data_id), str(data_type)
                entry = roles.setdefault(
                    did, {"role": role, "name": names.get(did, did)}
                )
                if role == "to_benchmark":
                    entry["role"] = role
                if entry["name"] == did and did in names:
                    entry["name"] = names[did]
    return roles


def role_badge(data_id: str, roles: dict[str, dict[str, str]] | None) -> str:
    """Small badge naming the dataset's role, empty when the role is unknown."""
    role = (roles or {}).get(data_id, {}).get("role", "")
    label = ROLE_LABELS.get(role)
    if not label:
        return ""
    cls = " lb-role-model" if role == "to_benchmark" else ""
    return f"<span class='lb-role{cls}'>{label}</span>"


def render_compare_line(
    roles: dict[str, dict[str, str]] | None,
    model_name: str = "",
) -> str:
    """One line stating which model is benchmarked, against what."""
    roles = roles or {}
    by_role: dict[str, list[str]] = {"to_benchmark": [], "other": [], "reference": []}
    for data_id, entry in roles.items():
        name = entry.get("name") or data_id
        names = by_role.setdefault(entry["role"], [])
        if name not in names:  # one model may carry several ids across tiers
            names.append(name)
    for names in by_role.values():
        names.sort()
    model = ", ".join(by_role["to_benchmark"]) or model_name or "not identified"
    blocks = [("Benchmarked model", f"<strong>{html_module.escape(model)}</strong>")]
    if by_role["other"]:
        blocks.append(
            ("Compared with", html_module.escape(", ".join(by_role["other"])))
        )
    if by_role["reference"]:
        blocks.append(
            ("References", html_module.escape(", ".join(by_role["reference"])))
        )
    items = "".join(f"<div><dt>{k}</dt><dd>{v}</dd></div>" for k, v in blocks)
    return f"<dl class='lb-compare'>{items}</dl>"


def build_leaderboard_data(
    all_metrics: dict[str, dict[str, pd.DataFrame]],
    roles: dict[str, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (summary, detail) tables of the leaderboard.

    The summary lists every dataset by overall score, best first. Only the
    models are ranked; reference datasets (per ``roles``) keep their row as a
    yardstick but get no rank number.
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
    is_ref = summary["data_id"].map(
        lambda d: (roles or {}).get(str(d), {}).get("role") == "reference",
    )
    summary["Rank"] = np.nan
    summary.loc[~is_ref, "Rank"] = np.arange(1, int((~is_ref).sum()) + 1, dtype=float)
    return summary[["Rank", "data_id", "Overall Score"]].reset_index(drop=True), detail


def _render_summary_table(
    summary_df: pd.DataFrame,
    table_id: str,
    roles: dict[str, dict[str, str]] | None = None,
) -> str:
    medal = {1: "medal-1", 2: "medal-2", 3: "medal-3"}
    rows_html = []
    unranked = False
    for _, row in summary_df.iterrows():
        data_id = str(row["data_id"])
        score = row["Overall Score"]
        pct = f"{score * 100:.1f}"
        is_model = (roles or {}).get(data_id, {}).get("role") == "to_benchmark"
        if pd.isna(row["Rank"]):
            rank_html, rank_cls, row_cls = "–", "", "is-ref"
            unranked = True
        else:
            rank = int(row["Rank"])
            rank_html, rank_cls, row_cls = str(rank), medal.get(rank, ""), ""
        if is_model:
            row_cls = "is-model"
        rows_html.append(
            f"<tr class='{row_cls}'>"
            f"<td class='rank {rank_cls}'>{rank_html}</td>"
            f"<td>{html_module.escape(data_id)}{role_badge(data_id, roles)}</td>"
            f"<td class='score-cell bar-cell'>"
            f"<div class='bar-bg' style='width:{pct}%;'></div>"
            f"<span class='bar-val'>{pct}%</span>"
            f"</td>"
            f"</tr>",
        )
    overall_label, overall_text = SCORE_INFO["overall"]
    return (
        f"<div class='lb-table-wrap'>"
        f"<table class='lb' id='{table_id}'>"
        f"<thead><tr>"
        f"<th class='sortable' onclick='sortTable(\"{table_id}\",0,true)'>Rank</th>"
        f"<th class='sortable' onclick='sortTable(\"{table_id}\",1,false)'>Dataset</th>"
        f"<th class='sortable' onclick='sortTable(\"{table_id}\",2,true)'>"
        f"{overall_label}{info_icon(overall_text)}</th>"
        f"</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        f"</table></div>"
        + (f"<p class='lb-note'>{UNRANKED_NOTE}</p>" if unranked else "")
    )


UNRANKED_NOTE = "Reference datasets are scored for comparison but not ranked."

_DETAIL_HEADERS = {
    "data_id": "Dataset",
    "tier": "Tier",
    "diagnostic": "Diagnostic",
    "var_id": "Variable",
}


def detail_header(col: str) -> str:
    """Header cell text for a full-metrics-table column, with an info icon."""
    if col in METRIC_INFO:
        label, text = METRIC_INFO[col]
        return f"{label}{info_icon(text)}"
    if col == "overall":
        label, text = SCORE_INFO["overall"]
        return f"{label}{info_icon(text)}"
    return html_module.escape(_DETAIL_HEADERS.get(col, col))


def _render_detail_table(
    detail_df: pd.DataFrame,
    table_id: str,
    roles: dict[str, dict[str, str]] | None = None,
) -> str:
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
        f"{detail_header(c)}</th>"
        for i, c in enumerate(cols_show)
    )
    rows_html = []
    for _, row in sub.iterrows():
        is_model = (roles or {}).get(str(row["data_id"]), {}).get(
            "role"
        ) == "to_benchmark"
        cells = []
        for c in cols_show:
            val = row[c]
            if pd.isna(val):
                cells.append("<td>—</td>")
            elif isinstance(val, float):
                cells.append(f"<td class='score-cell'>{val:.4f}</td>")
            else:
                cells.append(f"<td>{html_module.escape(str(val))}</td>")
        rows_html.append(
            f"<tr class='{'is-model' if is_model else ''}'>{''.join(cells)}</tr>"
        )
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
    roles: dict[str, dict[str, str]] | None = None,
) -> go.Figure | None:
    """ClimateBench-style performance scoreboard heatmap.

    Rows = models (`data_id`), columns = variables (`var_id`).

    Each cell shows a relative score calculated as

        relative score = 1 - model_metric / median_metric

    Positive values (blue) = better than median; negative (red) = worse. A
    "Mean relative score" column shows the mean relative score across all variables.
    Models are sorted by mean relative score (best first).

    The benchmarked model's row label (from ``roles``) is set in bold.

    Returns None if `metric` is not available.
    """

    def _row_label(data_id: str) -> str:
        if (roles or {}).get(data_id, {}).get("role") == "to_benchmark":
            return f"<b>{data_id}</b>"
        return data_id

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
    metric_df["relative score"] = 1.0 - metric_df[metric] / metric_df["median_metric"]

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

    # Best first: the highest mean relative score is the furthest below the
    # median metric
    pivot = pivot.loc[pivot["Mean relative score"].sort_values(ascending=False).index]

    single_model = len(pivot.index) < 2

    if single_model:
        # Only 1 model: relative score is always 0 → show raw metric instead.
        raw_pivot = (
            metric_df.groupby(["data_id", "var_id"])[metric].mean().unstack("var_id")
        )
        raw_pivot["Mean relative score"] = raw_pivot.mean(axis=1)
        z = raw_pivot.values.tolist()
        x_labels = [str(c) for c in raw_pivot.columns]
        y_labels = [_row_label(str(i)) for i in raw_pivot.index]
        finite_vals = raw_pivot.values[np.isfinite(raw_pivot.values)]
        zmax = float(np.nanpercentile(finite_vals, 98)) if len(finite_vals) > 0 else 1.0
        zmax = max(zmax, 1e-6)
        text_annots = [
            [f"{v:.3g}" if np.isfinite(v) else "" for v in row]
            for row in raw_pivot.values
        ]
        heatmap_kwargs = dict(zmin=0, zmax=zmax)
        metric_label = METRIC_INFO.get(metric, (metric, ""))[0]
        colorbar_title = f"{metric_label}<br>(lower = better)"
        title_note = f"{metric_label} per variable, blue = lower = better"
        hover_suffix = "<extra></extra>"
        hover_tmpl = "%{y}<br><b>%{x}</b>: %{z:.4g}" + hover_suffix
        colorbar_ticks = {}
    else:
        # Multiple models: show relative score.
        z = pivot.values.tolist()
        x_labels = [str(c) for c in pivot.columns]
        y_labels = [_row_label(str(i)) for i in pivot.index]
        finite_vals = pivot.values[np.isfinite(pivot.values)]
        text_annots = [
            [f"{v:.2f}" if np.isfinite(v) else "" for v in row] for row in pivot.values
        ]
        mintext = "▼-1" if len(finite_vals) > 0 and np.min(finite_vals) < -1.0 else "-1"
        colorbar_ticks = {
            "tickvals": [-1.0, -0.5, 0.0, 0.5, 1.0],
            "ticktext": [mintext, "-0.5", "0", "0.5", "1.0"],
        }
        heatmap_kwargs = dict(zmin=-1.0, zmax=1.0)
        metric_label = METRIC_INFO.get(metric, (metric, ""))[0]
        colorbar_title = "Relative score (blue = better)"
        title_note = f"Relative score = 1 − {metric_label} / median {metric_label}"
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
                "thickness": 10,
                "len": 0.8,
                "outlinewidth": 0,
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
                "text": f"<span style='font-size:0.85em;color:#7a7a74'>{title_note}</span>",
                "font": {"size": 13},
                "x": 0,
                "xanchor": "left",
            },
            height=max(220 + 24 * len(y_labels), 340),
            margin={"t": 40, "b": 110, "l": 240, "r": 40},
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
            # heatmap rows are drawn bottom-up; best first means top first
            "autorange": "reversed",
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
                "line": {"color": "#1b1b19", "width": 1.5},
                "fillcolor": "rgba(0,0,0,0)",
            },
        ],
    )
    return fig
