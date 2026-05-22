"""HTML template rendering for the benchmarking report."""

from __future__ import annotations

import html as html_module
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go

from climateeval.report._db import detect_diag_type, read_database
from climateeval.report._figures import _DIAG_ICONS, _DIAG_LABELS, _FIGURE_BUILDERS
from climateeval.report._leaderboard import (
    _render_detail_table,
    _render_summary_table,
    build_leaderboard_data,
    build_scoreboard_figure,
)
from climateeval.report._style import _CSS, _JS

# ---------------------------------------------------------------------------
# HTML assembly helpers
# ---------------------------------------------------------------------------


def _stem_to_label(stem: str) -> str:
    s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", stem)
    return s.replace("_", " ").replace("-", " ").title()


def _plotly_js_tag(offline: bool) -> str:
    if offline:
        import plotly as _plotly

        js_path = Path(_plotly.__file__).parent / "package_data" / "plotly.min.js"
        if js_path.exists():
            return f"<script>{js_path.read_text()}</script>"
    return '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'


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
            "<span><span class='ln-bias-swatch'></span>Bias = model − reference (right column, RdBu r)</span>"
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


def render_leaderboard_section(
    all_metrics: dict[str, dict[str, pd.DataFrame]],
    table_suffix: str = "",
) -> str:
    """Render the leaderboard HTML section including scoreboard figure."""
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

    # Build scoreboard heatmap (uses same figdata- convention as diagnostic plots)
    scoreboard_html = ""
    scoreboard_fig = build_scoreboard_figure(all_metrics)
    if scoreboard_fig is not None:
        sb_graph_id = f"scoreboard-graph{table_suffix}"
        sb_fig_json = scoreboard_fig.to_json()
        scoreboard_html = (
            f"<div class='scoreboard-wrap'>"
            f"<div id='{sb_graph_id}' style='width:100%;min-height:320px;'></div>"
            f"<script type='application/json' id='figdata-{sb_graph_id}'>{sb_fig_json}</script>"
            f"<script>"
            f"(function(){{"
            f"var el=document.getElementById('{sb_graph_id}');"
            f"var fd=JSON.parse(document.getElementById('figdata-{sb_graph_id}').textContent);"
            f"fd.layout.autosize=true;"
            f"Plotly.newPlot(el,fd.data,fd.layout,{{responsive:true,displayModeBar:false}});"
            f"}})();"
            f"</script>"
            f"</div>"
        )

    return (
        f"<div class='lb-section'>"
        f"{metric_note}"
        f"{scoreboard_html}"
        f"{summary_html}"
        f"<div class='detail-section'>"
        f"<details><summary>Full metrics table</summary>{detail_html}</details>"
        f"</div>"
        f"</div>"
    )


def _build_figures_for_tier(
    db_data: dict[str, dict[str, pd.DataFrame]],
    scope_prefix: str,
) -> tuple[
    dict[str, tuple[str, pd.DataFrame]],
    dict[str, pd.DataFrame],
    dict[str, str],
]:
    """Return (diag_sections, tier_metrics, diag_types) for one database.

    diag_sections: {diag_name: (html, variables_df)}
    tier_metrics: {diag_name: metrics_df}
    diag_types: {diag_name: detected_type}  – used for label/icon lookup
    """
    diag_sections: dict[str, tuple[str, pd.DataFrame]] = {}
    tier_metrics: dict[str, pd.DataFrame] = {}
    diag_types: dict[str, str] = {}

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
        diag_types[diag_name] = diag_type
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

    return diag_sections, tier_metrics, diag_types


def _render_tier_panel(
    tier_id: str,
    db_path: Path,
    diag_sections: dict[str, tuple[str, pd.DataFrame]],
    diag_types: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Return (toolbar_html, panel_html) for one tier."""
    diag_names = list(diag_sections.keys())
    diag_types = diag_types or {}

    def _label(dname: str) -> str:
        # First try direct name, then fall back to detected type
        return _DIAG_LABELS.get(dname) or _DIAG_LABELS.get(
            diag_types.get(dname, ""),
            dname,
        )

    def _icon(dname: str) -> str:
        return _DIAG_ICONS.get(dname) or _DIAG_ICONS.get(diag_types.get(dname, ""), "")

    # Toolbar: All + per-diagnostic filter buttons + search
    filter_btns = (
        f"<button class='diag-filter-btn active' data-diag='all' "
        f'onclick=\'filterDiag("{tier_id}","all")\'>All</button>'
    )
    for dname in diag_names:
        label = _label(dname)
        icon = _icon(dname)
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
        label = _label(dname)
        icon = _icon(dname)
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


def generate_portable_html(
    db_paths: list[Path],
    *,
    offline: bool = True,
    model_name: str = "",
    user_name: str = "",
    developer: str = "",
) -> str:
    """Read one or more databases and produce a self-contained HTML string."""
    from climateeval.report._figures import _coastlines

    _coastlines()

    tiers: list[
        tuple[
            str,
            str,
            Path,
            dict[str, tuple[str, pd.DataFrame]],
            dict[str, pd.DataFrame],
            dict[str, str],
        ]
    ] = []
    all_metrics: dict[str, dict[str, pd.DataFrame]] = {}

    for tier_idx, db_path in enumerate(db_paths):
        label = _stem_to_label(db_path.stem)
        db_data = read_database(db_path)
        if not db_data:
            continue

        tier_id = f"tier{tier_idx}"
        diag_sections, tier_metrics, diag_types = _build_figures_for_tier(
            db_data,
            tier_id,
        )
        all_metrics[label] = tier_metrics
        tiers.append((tier_id, label, db_path, diag_sections, tier_metrics, diag_types))

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

    for tier_id, label, db_path, diag_sections, tier_metrics, diag_types in tiers:
        toolbar_html, panel_html = _render_tier_panel(
            tier_id,
            db_path,
            diag_sections,
            diag_types,
        )
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

{_plotly_js_tag(offline)}
<script>{_JS}</script>

</body>
</html>"""


def build_html_shell(
    tiers: list[dict],
    all_metrics: dict[str, dict[str, pd.DataFrame]],
    model_name: str = "",
    user_name: str = "",
    developer: str = "",
) -> str:
    """Render the full HTML shell for the web server (sidebar + empty content panels).

    Args:
        tiers: List of tier dicts (from _load_state), each with keys:
               id, label, db_path, db_data, diag_types
        all_metrics: {tier_label: {diag_name: metrics_df}}
        model_name: Model label shown in the report header.
        user_name: Analyst / user name shown in the report header.
        developer: Benchmarking tool developer name.

    Returns
    -------
        Full HTML string for the shell page served at GET /.
    """
    from climateeval.report._db import get_var_cols
    from climateeval.report._figures import _var_label

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
    for tier in tiers:
        tid = tier["id"]
        tlabel = html_module.escape(tier["label"])
        sidebar_items += (
            f"<div class='sidebar-item' data-section='{tid}' "
            f"onclick='sidebarNav(\"{tid}\")'>"
            f"<span class='sidebar-icon'>📂</span>{tlabel}"
            f"</div>"
        )

    # ── Leaderboard panel (figures fetched via JS on load) ──
    lb_panel = """
<div class='content-panel active' id='panel-leaderboard'>
  <h2 class='section-heading'>Performance Leaderboard</h2>
  <div class='lb-section'>
    <div class='metric-note'>
      Overall Score = normalized average of weighted RMSE, Pearson&nbsp;r, and
      Earth Mover&rsquo;s Distance across all diagnostics and variables.
      100&nbsp;% = best possible.
    </div>
    <div class='scoreboard-wrap'>
      <div id='scoreboard-graph' style='width:100%;min-height:320px;'></div>
    </div>
    <div id='lb-summary-container'></div>
    <div class='detail-section'>
      <details><summary>Full metrics table</summary>
        <div id='lb-detail-container'></div>
      </details>
    </div>
  </div>
</div>"""

    # ── Tier panels ──
    all_toolbars = ""
    all_panels = lb_panel

    def _diag_label(diag_name: str, dtype: str) -> str:
        return _DIAG_LABELS.get(diag_name) or _DIAG_LABELS.get(dtype, diag_name)

    def _diag_icon(diag_name: str, dtype: str) -> str:
        return _DIAG_ICONS.get(diag_name) or _DIAG_ICONS.get(dtype, "")

    def _render_shell_diagnostic(
        tier_id: str,
        diag_name: str,
        dtype: str,
        variables: list[dict[str, str]],
    ) -> str:
        """HTML for one diagnostic section: variable tabs + empty plot divs with API URLs."""
        scope_id = f"{tier_id}-{diag_name}"
        first_var = variables[0]["id"] if variables else ""
        is_2d = dtype in ("Map", "ZonalProfile")
        height = 700 if is_2d else 450

        def _btn_label(v: dict) -> str:
            ln = v.get("long_name") or v["id"]
            u = v.get("units", "")
            return f"{ln} [{u}]" if u else ln

        var_buttons = "".join(
            "<button class='var-pill {active}' data-var='{vid}' title='{vid}' "
            'onclick=\'showVarPanel("{scope}", "{vid}")\'>{label}</button>'.format(
                active="active" if v["id"] == first_var else "",
                vid=html_module.escape(v["id"]),
                scope=scope_id,
                label=html_module.escape(_btn_label(v)),
            )
            for v in variables
        )

        panels = ""
        for v in variables:
            vid = v["id"]
            panel_id = f"plot-{scope_id}-{vid}"
            graph_id = f"graph-{scope_id}-{vid}"
            api_url = f"/api/figure/{tier_id}/{diag_name}/{vid}"
            active = "active" if vid == first_var else ""
            panels += (
                f"<div class='plot-panel {active}' id='{panel_id}' "
                f"data-graph='{graph_id}' data-search='{html_module.escape(vid.lower())}'>"
                f"<div class='plot-title-bar'>"
                f"<span class='plot-title-text'>{html_module.escape(_btn_label(v))}</span>"
                f"</div>"
                f"<div id='{graph_id}' style='height:{height}px;width:100%;' "
                f"data-api-url='{html_module.escape(api_url)}'></div>"
                f"</div>"
            )

        return (
            f"<div id='diag-{scope_id}' class='diag-content'>"
            f"<div class='var-nav'>{var_buttons}</div>"
            f"{panels}"
            f"</div>"
        )

    for tier in tiers:
        tier_id = tier["id"]
        db_data: dict[str, Any] = tier["db_data"]
        diag_types: dict[str, str] = tier["diag_types"]

        # Collect diagnostics with variables
        diags_for_toolbar: list[tuple[str, str]] = []  # (diag_name, dtype)
        sections_html = ""

        for diag_name, tables in db_data.items():
            if "raw_output" not in tables:
                continue
            data_df: pd.DataFrame = tables["raw_output"]
            variables_df: pd.DataFrame = tables.get("variables", pd.DataFrame())
            dtype = diag_types.get(diag_name, "Unknown")

            variables = []
            for var_id in get_var_cols(data_df):
                if data_df[var_id].isna().all():
                    continue
                long_name, units = _var_label(var_id, variables_df)
                variables.append({"id": var_id, "long_name": long_name, "units": units})

            if not variables:
                continue

            diags_for_toolbar.append((diag_name, dtype))
            label = _diag_label(diag_name, dtype)
            icon = _diag_icon(diag_name, dtype)
            content = _render_shell_diagnostic(tier_id, diag_name, dtype, variables)
            sections_html += (
                f"<div class='diag-section' data-diag='{diag_name}' "
                f"id='diag-sec-{tier_id}-{diag_name}'>"
                f"<div class='diag-section-header'>"
                f"<span class='diag-section-icon'>{icon}</span>"
                f"<span class='diag-section-title'>{html_module.escape(label)}</span>"
                f"</div>"
                f"{content}"
                f"</div>"
            )

        # Toolbar
        filter_btns = (
            f"<button class='diag-filter-btn active' data-diag='all' "
            f'onclick=\'filterDiag("{tier_id}","all")\'>All</button>'
        )
        for dname, dtype in diags_for_toolbar:
            label = _diag_label(dname, dtype)
            icon = _diag_icon(dname, dtype)
            filter_btns += (
                f"<button class='diag-filter-btn' data-diag='{dname}' "
                f'onclick=\'filterDiag("{tier_id}","{dname}")\'>'
                f"{icon} {html_module.escape(label)}</button>"
            )

        search_svg = (
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
            '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'
            "</svg>"
        )
        all_toolbars += (
            f"<div class='content-toolbar' id='toolbar-{tier_id}' style='display:none'>"
            f"<div class='diag-filter-btns'>{filter_btns}</div>"
            f"<div class='toolbar-sep'></div>"
            f"<div class='search-wrap'>{search_svg}"
            f"<input class='search-input' type='text' placeholder='Search plots…' "
            f"oninput='searchPlots(this.value)'></div>"
            f"</div>"
        )
        all_panels += (
            f"<div class='content-panel' id='panel-{tier_id}' data-type='tier'>"
            f"<p class='db-source'>{html_module.escape(str(tier['db_path']))}</p>"
            f"{sections_html}"
            f"<div class='search-no-results'>No plots match your search.</div>"
            f"</div>"
        )

    # ── Header chips ──
    chips = ""
    if model_name:
        chips += (
            f"<span class='header-chip'><span>Model</span> "
            f"<strong>{html_module.escape(model_name)}</strong></span>"
        )
    if user_name:
        chips += (
            f"<span class='header-chip'><span>User</span> "
            f"<strong>{html_module.escape(user_name)}</strong></span>"
        )
    dev_str = developer or "diffESM Benchmarking"
    chips += (
        f"<span class='header-chip'><span>Tool</span> "
        f"<strong>{html_module.escape(dev_str)}</strong></span>"
    )
    chips += f"<span class='header-chip'><span>Generated</span> <strong>{now}</strong></span>"

    logo_svg = (
        '<svg width="26" height="26" viewBox="0 0 26 26" fill="none" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="13" cy="13" r="11" stroke="#60a5fa" stroke-width="1.5" fill="none"/>'
        '<path d="M2 13 Q7 7 13 13 Q19 19 24 13" stroke="#60a5fa" stroke-width="1.5" fill="none"/>'
        '<path d="M2 13 Q7 19 13 13 Q19 7 24 13" stroke="#93c5fd" stroke-width="1" fill="none" stroke-dasharray="2 2"/>'
        "</svg>"
    )

    # Plotly.js tag (try bundled first, fall back to CDN)
    try:
        import plotly as _plotly

        js_path = Path(_plotly.__file__).parent / "package_data" / "plotly.min.js"
        if js_path.exists():
            plotly_tag = f"<script>{js_path.read_text()}</script>"
        else:
            plotly_tag = (
                '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
            )
    except Exception:
        plotly_tag = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'

    # Leaderboard JS: fetch data from /api/leaderboard and render
    lb_js = """
<script>
(function() {
  fetch('/api/leaderboard')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.scoreboard) {
        var el = document.getElementById('scoreboard-graph');
        if (el) {
          data.scoreboard.layout.autosize = true;
          Plotly.newPlot(el, data.scoreboard.data, data.scoreboard.layout,
            {responsive:true, displayModeBar:false});
        }
      }
      if (data.summary) _renderSummaryTable(data.summary);
      if (data.detail)  _renderDetailTable(data.detail);
    })
    .catch(function(e) { console.warn('Leaderboard load failed:', e); });
})();

function _renderSummaryTable(rows) {
  var medals = {1:'medal-1', 2:'medal-2', 3:'medal-3'};
  var html = "<div class='lb-table-wrap'><table class='lb' id='lb-summary'>"
    + "<thead><tr>"
    + "<th class='sortable' onclick='sortTable(\"lb-summary\",0,true)'>Rank</th>"
    + "<th class='sortable' onclick='sortTable(\"lb-summary\",1,false)'>Data Source</th>"
    + "<th class='sortable' onclick='sortTable(\"lb-summary\",2,true)'>Overall Score</th>"
    + "</tr></thead><tbody>";
  rows.forEach(function(row) {
    var rank = row['Rank'] || 0;
    var ds   = row['data_id'] || '';
    var sc   = typeof row['Overall Score'] === 'number' ? row['Overall Score'] : 0;
    var pct  = (sc * 100).toFixed(1);
    var h    = Math.round(sc * 120);
    var color = 'hsl(' + h + ',65%,45%)';
    var cls  = medals[rank] || '';
    html += "<tr>"
      + "<td class='rank " + cls + "'>" + rank + "</td>"
      + "<td><strong>" + _esc(ds) + "</strong></td>"
      + "<td class='score-cell bar-cell'>"
      + "<div class='bar-bg' style='width:" + pct + "%;background:" + color + ";'></div>"
      + "<span class='bar-val' style='color:" + color + "'>" + pct + "%</span>"
      + "</td></tr>";
  });
  html += "</tbody></table></div>";
  var el = document.getElementById('lb-summary-container');
  if (el) el.innerHTML = html;
}

function _renderDetailTable(rows) {
  if (!rows || !rows.length) return;
  var strCols = {data_id:1, tier:1, diagnostic:1, var_id:1};
  var keys = Object.keys(rows[0]);
  var html = "<div class='lb-table-wrap'><table class='lb' id='lb-detail'><thead><tr>";
  keys.forEach(function(k, i) {
    var num = !strCols[k];
    html += "<th class='sortable' onclick='sortTable(\"lb-detail\"," + i + "," + num + ")'>"
      + _esc(k) + "</th>";
  });
  html += "</tr></thead><tbody>";
  rows.forEach(function(row) {
    html += "<tr>";
    keys.forEach(function(k) {
      var v = row[k];
      if (v === null || v === undefined) {
        html += "<td>—</td>";
      } else if (typeof v === 'number') {
        html += "<td class='score-cell'>" + v.toFixed(4) + "</td>";
      } else {
        html += "<td>" + _esc(String(v)) + "</td>";
      }
    });
    html += "</tr>";
  });
  html += "</tbody></table></div>";
  var el = document.getElementById('lb-detail-container');
  if (el) el.innerHTML = html;
}

function _esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
</script>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>ClimateEval Benchmarking Report</title>
  <style>{_CSS}</style>
</head>
<body>
<div id='appBody' class='app-body'>
  <aside id='sidebar' class='sidebar'>
    <div class='sidebar-header'>
      {logo_svg}
      <span class='sidebar-title'>ClimateEval</span>
      <button class='sidebar-toggle' onclick='toggleSidebar()' title='Collapse sidebar'
        aria-label='Collapse sidebar'>
        <svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2'>
          <polyline points='15 18 9 12 15 6'/>
        </svg>
      </button>
    </div>
    <nav class='sidebar-nav'>
      {sidebar_items}
    </nav>
  </aside>

  <div class='content-area' id='contentArea'>
    <header class='top-header'>
      <div class='header-chips'>{chips}</div>
      <div id='toolbarMount'>{all_toolbars}</div>
    </header>

    <main class='main-content' id='mainContent'>
      {all_panels}
    </main>
  </div>
</div>

{plotly_tag}
<script>{_JS}</script>
{lb_js}
</body>
</html>"""
