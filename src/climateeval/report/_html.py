"""HTML template rendering for the benchmarking report."""

from __future__ import annotations

import html as html_module
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from plotly.colors import sample_colorscale

from climateeval.report._db import detect_diag_type, read_database
from climateeval.report._figures import (
    _CATEGORY_ORDER,
    _DIAG_LABELS,
    _FIGURE_BUILDERS,
    _diag_category,
    _diag_display_label,
)
from climateeval.report._common import _var_label
from climateeval.report._leaderboard import (
    UNRANKED_NOTE,
    METRIC_INFO,
    SCORE_INFO,
    _render_detail_table,
    _render_summary_table,
    build_leaderboard_data,
    build_scoreboard_figure,
    dataset_roles,
    detail_header,
    info_icon,
    render_compare_line,
    role_badge,
)
from climateeval.report._style import (
    _CSS,
    _JS,
    _bias_colorscale_for,
    _colorscale_for,
    _tier_accent,
)

if TYPE_CHECKING:
    import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# HTML assembly helpers
# ---------------------------------------------------------------------------


def _render_app_header(
    model_name: str,
    user_name: str,
    developer: str,
    run_info: dict[str, str] | None,
    now: str,
) -> str:
    """Render the page header: title plus one chip per piece of run metadata."""

    def _chip(label: str, value: str) -> str:
        return (
            f"<span class='header-chip'><span>{html_module.escape(label)}</span> "
            f"<strong>{html_module.escape(value)}</strong></span>"
        )

    chips = ""
    if model_name:
        chips += _chip("Model", model_name)
    if user_name:
        chips += _chip("User", user_name)
    for key, value in (run_info or {}).items():
        if value:
            chips += _chip(key, str(value))
    if developer:
        chips += _chip("Tool", developer)
    chips += _chip("Generated", now)
    return (
        '<header class="app-header">'
        '<div class="header-title">'
        "<h1>ClimateEval<span>Benchmark report</span></h1>"
        f'<div class="header-meta">{chips}</div>'
        "</div>"
        f"{_THEME_SWITCH}"
        "</header>"
    )


_THEME_ICONS = {
    "auto": (
        '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" '
        'aria-hidden="true"><circle cx="8" cy="8" r="6"/>'
        '<path d="M8 2a6 6 0 0 1 0 12z" fill="currentColor" stroke="none"/></svg>'
    ),
    "light": (
        '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" '
        'stroke-linecap="round" aria-hidden="true"><circle cx="8" cy="8" r="3"/>'
        '<path d="M8 1.5v1.8M8 12.7v1.8M1.5 8h1.8M12.7 8h1.8M3.4 3.4l1.3 1.3'
        'M11.3 11.3l1.3 1.3M3.4 12.6l1.3-1.3M11.3 4.7l1.3-1.3"/></svg>'
    ),
    "dark": (
        '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" '
        'stroke-linejoin="round" aria-hidden="true">'
        '<path d="M13.5 10.2A6 6 0 0 1 5.8 2.5a6 6 0 1 0 7.7 7.7z"/></svg>'
    ),
}
_THEME_SWITCH = (
    '<div class="theme-switch" role="group" aria-label="Colour theme">'
    + "".join(
        f'<button type="button" data-theme-pref="{pref}" aria-pressed="false" '
        f'onclick="setTheme(\'{pref}\')" title="{title}">'
        f"{_THEME_ICONS[pref]}<span>{label}</span></button>"
        for pref, label, title in (
            ("auto", "Auto", "Follow the system setting"),
            ("light", "Light", "Light theme"),
            ("dark", "Dark", "Dark theme"),
        )
    )
    + "</div>"
)

# Applies the stored theme before the first paint so a dark page never
# flashes light; the full logic lives in _JS (setTheme).
_THEME_BOOT = (
    "<script>(function(){try{var t=localStorage.getItem('climateeval-theme');"
    "if(t==='light'||t==='dark'){document.documentElement.setAttribute('data-theme',t);}}"
    "catch(e){}})();</script>"
)


_LEADERBOARD_ICON = (
    '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    '<path d="M2 13.5h12"/><rect x="2.75" y="7.5" width="2.5" height="6" rx=".5"/>'
    '<rect x="6.75" y="2.5" width="2.5" height="11" rx=".5"/>'
    '<rect x="10.75" y="5.5" width="2.5" height="8" rx=".5"/></svg>'
)


def _render_sidebar(tiers: list[tuple[str, str]]) -> str:
    """Render the sidebar navigation: leaderboard, then one entry per tier."""
    items = (
        "<div class='sidebar-section-label'>Overview</div>"
        "<button type='button' class='sidebar-item active' data-section='leaderboard' "
        "aria-current='page' onclick='sidebarNav(\"leaderboard\")'>"
        f"<span class='sidebar-icon'>{_LEADERBOARD_ICON}</span>"
        "<span class='sidebar-item-text'>Leaderboard</span>"
        "</button>"
        "<div class='sidebar-divider'></div>"
        "<div class='sidebar-section-label'>Tiers</div>"
    )
    for tier_id, label in tiers:
        accent, badge = _tier_accent(label)
        items += (
            f"<button type='button' class='sidebar-item' data-section='{tier_id}' "
            f"onclick='sidebarNav(\"{tier_id}\")'>"
            f"<span class='tier-badge' style='{accent}'>{badge}</span>"
            f"<span class='sidebar-item-text'>{html_module.escape(label)}</span>"
            f"</button>"
        )
    return items


def _render_leaderboard_cards(
    scoreboard_html: str,
    summary_html: str,
    detail_html: str,
    roles: dict[str, dict[str, str]] | None = None,
    model_name: str = "",
) -> str:
    """Lay out the leaderboard panel: comparison line, scoreboard, ranking, table."""
    legend = (
        "<div class='lb-legend'><span>worse than median</span>"
        "<span class='lb-legend-scale' aria-hidden='true'></span>"
        "<span>better than median</span></div>"
    )
    relative_label, relative_text = SCORE_INFO["relative"]
    overall_label, overall_text = SCORE_INFO["overall"]
    metric_line = " &middot; ".join(
        f"{label}{info_icon(text)}" for label, text in METRIC_INFO.values()
    )
    return (
        "<h2 class='section-heading'>Leaderboard</h2>"
        "<p class='section-subheading'>Every dataset is scored against the same "
        "reference per variable; the benchmarked model is highlighted. Metrics: "
        f"{metric_line}.</p>"
        f"{render_compare_line(roles, model_name)}"
        "<div class='lb-section'>"
        "<div class='lb-card'>"
        f"<div class='lb-card-title'>Performance scoreboard{info_icon(relative_text)}</div>"
        f"<p class='lb-card-sub'>{relative_label} per variable, weighted EMD. "
        "Rows are sorted best first; the benchmarked model is in bold.</p>"
        f"{scoreboard_html}{legend}"
        "</div>"
        "<div class='lb-card'>"
        f"<div class='lb-card-title'>Overall ranking{info_icon(overall_text)}</div>"
        f"<p class='lb-card-sub'>{overall_label} over all diagnostics and variables, "
        "100 % is the best possible.</p>"
        f"{summary_html}"
        "</div>"
        "<div class='detail-section'>"
        f"<details><summary>Full metrics table</summary>{detail_html}</details>"
        "</div>"
        "</div>"
    )


def _render_toolbar(
    tier_id: str,
    db_path: Path,
    diag_buttons: list[tuple[str, str]],
    diag_types: dict[str, str] | None = None,
) -> str:
    """Render a tier's toolbar: diagnostic filter tabs, search box, database name.

    The tabs follow the plot-type category order of the panel below, with a
    thin separator between categories, and scroll sideways instead of
    wrapping when a tier has many diagnostics.
    """
    labels = dict(diag_buttons)
    filter_btns = (
        f"<button type='button' class='diag-filter-btn active' data-diag='all' "
        f"aria-pressed='true' onclick='filterDiag(\"{tier_id}\",\"all\")'>All</button>"
    )
    for i, (category, names) in enumerate(
        _grouped_diag_sections(list(labels), diag_types or {})
    ):
        if i:
            filter_btns += (
                "<span class='diag-filter-sep' aria-hidden='true' "
                f"title='{html_module.escape(category or '')}'></span>"
            )
        for dname in names:
            filter_btns += (
                f"<button type='button' class='diag-filter-btn' data-diag='{dname}' "
                f"aria-pressed='false' onclick='filterDiag(\"{tier_id}\",\"{dname}\")'>"
                f"{html_module.escape(labels[dname])}</button>"
            )
    search_svg = (
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        'aria-hidden="true">'
        '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>'
        "</svg>"
    )
    return (
        f"<div class='content-toolbar' id='toolbar-{tier_id}' style='display:none' "
        f"role='toolbar' aria-label='Diagnostic filter'>"
        f"<div class='diag-filter-btns'>{filter_btns}</div>"
        f"<div class='search-wrap'>{search_svg}"
        f"<input class='search-input' type='search' placeholder='Search variables' "
        f"aria-label='Search variables' oninput='searchPlots(this.value)'></div>"
        f"<div class='toolbar-sep'></div>"
        f"<span class='toolbar-db' title='{html_module.escape(str(db_path))}'>"
        f"{html_module.escape(db_path.name)}</span>"
        f"</div>"
    )


def _render_var_nav(scope_id: str, buttons_html: str, n_vars: int) -> str:
    """Wrap the variable buttons.

    The nav sticks to the top while a tall figure scrolls. With many
    variables it starts collapsed to two rows behind a "Show all" button.
    """
    many = n_vars > _VAR_NAV_COLLAPSE_ABOVE
    more = (
        f"<button type='button' class='var-nav-more' aria-expanded='false' "
        f"onclick='toggleVarNav(this)'>Show all {n_vars}</button>"
        if many
        else ""
    )
    return (
        f"<div class='var-nav-wrap'>"
        f"<div class='var-nav{' many' if many else ''}' role='group' aria-label='Variables'>"
        f"{buttons_html}</div>{more}</div>"
    )


# Variable navs with more buttons than this start collapsed to two rows.
_VAR_NAV_COLLAPSE_ABOVE = 14


def _section_header(label: str, dtype: str) -> str:
    """Diagnostic card header: title plus a small plot-type chip."""
    chip = _DIAG_LABELS.get(dtype, dtype)
    chip_html = (
        f"<span class='diag-type-chip'>{html_module.escape(chip)}</span>"
        if chip and chip != label
        else ""
    )
    return (
        "<div class='diag-section-header'>"
        f"<span class='diag-section-title'>{html_module.escape(label)}</span>"
        f"{chip_html}</div>"
    )


def _css_gradient(colorscale: str) -> str:
    """Convert a Plotly colorscale name into a CSS linear-gradient."""
    colors = sample_colorscale(colorscale, [0.0, 0.25, 0.5, 0.75, 1.0])
    return f"linear-gradient(to right, {', '.join(colors)})"


def _stem_to_label(stem: str) -> str:
    s = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", stem)
    return s.replace("_", " ").replace("-", " ").title()


def _var_title(vid: str, long_name: str, units: str) -> tuple[str, str]:
    """Return (tooltip text, title-bar inner HTML) for a variable.

    Pills keep the compact short name; the long name goes into the pill
    tooltip and the plot title bar (with the short name as a small chip).
    """
    has_long_name = bool(long_name) and long_name != vid
    tooltip = long_name if has_long_name else vid
    if units:
        tooltip += f" [{units}]"
    title_html = html_module.escape(long_name if has_long_name else vid)
    if has_long_name:
        title_html += f" <span class='var-id-chip'>{html_module.escape(vid)}</span>"
    if units:
        title_html += (
            f" <span class='plot-title-units'>[{html_module.escape(units)}]</span>"
        )
    return tooltip, title_html


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
        return v

    def _search_terms(v: str) -> str:
        terms = [v]
        if variables_df is not None and not variables_df.empty:
            row = variables_df[variables_df["id"] == v]
            if not row.empty:
                terms.append(str(row["long_name"].values[0]))
                terms.append(str(row.get("standard_name", pd.Series([""])).values[0]))
        return " ".join(terms).lower()

    def _title_parts(v: str) -> tuple[str, str]:
        long_name, units = (
            _var_label(v, variables_df)
            if variables_df is not None and not variables_df.empty
            else (v, "")
        )
        return _var_title(v, long_name, units)

    title_parts = {v: _title_parts(v) for v in var_cols}

    var_buttons = "".join(
        f"<button type='button' class='var-pill {'active' if v == first_var else ''}' "
        f"data-var='{html_module.escape(v)}' "
        f"aria-pressed='{'true' if v == first_var else 'false'}' "
        f"title='{html_module.escape(title_parts[v][0])}' "
        f'onclick=\'showVarPanel("{scope_id}", "{v}")\'>'
        f"{html_module.escape(_btn_label(v))}</button>"
        for v in var_cols
    )

    _diag_type = (
        detect_diag_type(data_df.columns.tolist())
        if data_df is not None and not data_df.empty
        else "Unknown"
    )
    is_2d = _diag_type in ("Map", "ZonalProfile")
    if is_2d:
        # Rendered per variable panel: swatches mirror each plot's colormaps.
        legend_note = ""
    else:
        # Mirror _get_line_properties(): model = coloured solid,
        # reference = black solid, other = gray dashed.
        legend_note = (
            "<div class='legend-note'>"
            "<span><span class='ln-model'></span>Benchmarked model: solid, thick</span>"
            "<span><span class='ln-refline'></span>Reference: solid, dark</span>"
            "<span><span class='ln-other'></span>Compared models: thin, own colour and dash</span>"
            "</div>"
        )

    def _map_legend_note(var_id: str) -> str:
        data_grad = _css_gradient(_colorscale_for(var_id))
        bias_grad = _css_gradient(_bias_colorscale_for(var_id))
        # Maps put the model first; zonal profiles still draw reference then
        # bias, so they get their own wording rather than the map's.
        columns: tuple[tuple[str, str, str], ...]
        if _diag_type == "Map":
            columns = (
                (data_grad, "ln-data-swatch", "Model (left column)"),
                (
                    data_grad,
                    "ln-data-swatch",
                    "Observations, named per panel (middle column)",
                ),
                (
                    bias_grad,
                    "ln-bias-swatch",
                    "Bias = model − observations (right column)",
                ),
            )
        else:
            columns = (
                (data_grad, "ln-data-swatch", "Observations (left column)"),
                (
                    bias_grad,
                    "ln-bias-swatch",
                    "Bias = model − observations (right column)",
                ),
            )
        spans = "".join(
            f"<span><span class='{cls}' style='background:{grad}'></span>{text}</span>"
            for grad, cls, text in columns
        )
        return f"<div class='legend-note'>{spans}</div>"

    plot_panels = ""
    for var_id, fig in figures.items():
        panel_id = f"plot-{scope_id}-{var_id}"
        graph_id = f"graph-{scope_id}-{var_id}"
        detail_id = f"detail-{scope_id}-{var_id}"
        height = fig.layout.height or 440
        active_cls = "active" if var_id == first_var else ""
        search_key = html_module.escape(_search_terms(var_id))
        title_html = title_parts[var_id][1]

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
                f"<button type='button' class='detail-toggle-btn' "
                f"aria-expanded='false' aria-controls='{detail_id}' "
                f"onclick='toggleDetail(\"{detail_id}\", this)'>"
                f"<span class='detail-icon' aria-hidden='true'>i</span>"
                f"Dataset info &amp; references"
                f"</button>"
                f"<div class='plot-detail-box' id='{detail_id}'>{detail_html}</div>"
                f"</div>"
            )

        panel_legend = _map_legend_note(var_id) if is_2d else ""
        plot_panels += (
            f"<div class='plot-panel {active_cls}' id='{panel_id}' "
            f"data-graph='{graph_id}' data-search='{search_key}'>"
            f"<div class='plot-title-bar'>"
            f"<span class='plot-title-text'>{title_html}</span>"
            f"</div>"
            f"{panel_legend}"
            f"<div id='{graph_id}' style='height:{height}px;width:100%;'></div>"
            f"<script type='application/json' id='figdata-{graph_id}'>{fig.to_json()}</script>"
            f"{detail_block}"
            f"</div>"
        )

    return (
        f"<div id='diag-{scope_id}' class='diag-content'>"
        f"{_render_var_nav(scope_id, var_buttons, len(var_cols))}"
        f"{legend_note}"
        f"{plot_panels}"
        f"</div>"
    )


def render_leaderboard_section(
    all_metrics: dict[str, dict[str, pd.DataFrame]],
    table_suffix: str = "",
    roles: dict[str, dict[str, str]] | None = None,
    model_name: str = "",
) -> str:
    """Render the leaderboard HTML section including scoreboard figure."""
    summary_df, detail_df = build_leaderboard_data(all_metrics, roles)
    if summary_df.empty:
        return (
            "<h2 class='section-heading'>Leaderboard</h2>"
            "<p class='empty-note'>No metrics data found.</p>"
        )
    summary_html = _render_summary_table(summary_df, f"lb-summary{table_suffix}", roles)
    detail_html = _render_detail_table(detail_df, f"lb-detail{table_suffix}", roles)

    # Build scoreboard heatmap (uses same figdata- convention as diagnostic plots)
    scoreboard_html = ""
    scoreboard_fig = build_scoreboard_figure(all_metrics, roles=roles)
    if scoreboard_fig is not None:
        sb_graph_id = f"scoreboard-graph{table_suffix}"
        sb_fig_json = scoreboard_fig.to_json()
        scoreboard_html = (
            f"<div class='scoreboard-wrap'>"
            f"<div id='{sb_graph_id}' style='width:100%;min-height:320px;'></div>"
            f"<script type='application/json' id='figdata-{sb_graph_id}'>{sb_fig_json}</script>"
            f"<script>"
            f"(function(){{"
            f"function draw(){{"
            f"var el=document.getElementById('{sb_graph_id}');"
            f"var fd=JSON.parse(document.getElementById('figdata-{sb_graph_id}').textContent);"
            f"fd.layout.autosize=true;"
            f"_plotThemed(el,fd.data,fd.layout,{{responsive:true,displayModeBar:false}});"
            f"}}"
            # Plotly.js and the page script are loaded at the end of the body;
            # draw once they are there.
            f"if(typeof Plotly==='undefined'||typeof _plotThemed==='undefined')"
            f"{{window.addEventListener('load',draw);}}"
            f"else{{draw();}}"
            f"}})();"
            f"</script>"
            f"</div>"
        )

    return _render_leaderboard_cards(
        scoreboard_html, summary_html, detail_html, roles, model_name
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


def _grouped_diag_sections(
    diag_names: list[str],
    diag_types: dict[str, str],
) -> list[tuple[str | None, list[str]]]:
    """Group diagnostics by plot-type category in a fixed order.

    Returns ``[(category_label_or_None, [diag_name, ...]), ...]``. The label is
    ``None`` when only one category is present, so no header gets rendered.
    """
    buckets: dict[str, list[str]] = {cat: [] for cat in _CATEGORY_ORDER}
    for dname in diag_names:
        buckets[_diag_category(diag_types.get(dname, ""))].append(dname)
    groups = [(cat, buckets[cat]) for cat in _CATEGORY_ORDER if buckets[cat]]
    show = len(groups) >= 2
    return [(cat if show else None, names) for cat, names in groups]


def _wrap_sections_by_category(
    section_html_by_diag: dict[str, str],
    diag_types: dict[str, str],
) -> str:
    """Concatenate per-diagnostic section HTML grouped into category wrappers.

    Diagnostics are reordered into plot-type category groups; each group is a
    ``.diag-category`` wrapper with an optional ``.diag-category-header``.
    """
    out = ""
    for category, names in _grouped_diag_sections(
        list(section_html_by_diag), diag_types
    ):
        header = (
            f"<div class='diag-category-header'>{html_module.escape(category)}</div>"
            if category
            else ""
        )
        inner = "".join(section_html_by_diag[name] for name in names)
        out += f"<div class='diag-category'>{header}{inner}</div>"
    return out


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
        return _diag_display_label(dname, diag_types.get(dname, ""))

    toolbar_html = _render_toolbar(
        tier_id,
        db_path,
        [(dname, _label(dname)) for dname in diag_names],
        diag_types,
    )

    # Panel: diagnostic sections grouped by plot-type category
    section_html_by_diag: dict[str, str] = {}
    for dname, (content, _) in diag_sections.items():
        section_html_by_diag[dname] = (
            f"<div class='diag-section' data-diag='{dname}' id='diag-sec-{tier_id}-{dname}'>"
            f"{_section_header(_label(dname), diag_types.get(dname, ''))}"
            f"{content}"
            f"</div>"
        )
    sections_html = _wrap_sections_by_category(section_html_by_diag, diag_types)

    panel_html = (
        f"<div class='content-panel' id='panel-{tier_id}' data-type='tier'>"
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
    run_info: dict[str, str] | None = None,
) -> str:
    """Read one or more databases and produce a self-contained HTML string.

    ``run_info`` is extra run metadata (e.g. model data path, time range)
    rendered as chips in the page header.
    """
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
    all_db_data: list[dict[str, dict[str, pd.DataFrame]]] = []

    for tier_idx, db_path in enumerate(db_paths):
        label = _stem_to_label(db_path.stem)
        db_data = read_database(db_path)
        if not db_data:
            continue
        all_db_data.append(db_data)

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
    sidebar_items = _render_sidebar([(tier_id, label) for tier_id, label, *_ in tiers])

    # ── Leaderboard panel ──
    roles = dataset_roles(all_db_data)
    lb_panel = (
        "<div class='content-panel active' id='panel-leaderboard'>"
        + render_leaderboard_section(
            all_metrics, table_suffix="-main", roles=roles, model_name=model_name
        )
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

    header_html = _render_app_header(model_name, user_name, developer, run_info, now)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ClimateEval Report</title>
  {_THEME_BOOT}
  <style>{_CSS}</style>
</head>
<body>
<div class="app-shell">

  {header_html}

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
    run_info: dict[str, str] | None = None,
) -> str:
    """Render the full HTML shell for the web server (sidebar + empty content panels).

    Args:
        tiers: List of tier dicts (from _load_state), each with keys:
               id, label, db_path, db_data, diag_types
        all_metrics: {tier_label: {diag_name: metrics_df}}
        model_name: Model label shown in the report header.
        user_name: Analyst / user name shown in the report header.
        developer: Benchmarking tool developer name.
        run_info: Extra run metadata rendered as chips in the header.

    Returns
    -------
        Full HTML string for the shell page served at GET /.
    """
    from climateeval.report._db import get_var_cols

    now = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")

    # ── Sidebar ──
    sidebar_items = _render_sidebar([(tier["id"], tier["label"]) for tier in tiers])

    # ── Leaderboard panel (figures fetched via JS on load) ──
    roles = dataset_roles(tier["db_data"] for tier in tiers)
    lb_panel = (
        "<div class='content-panel active' id='panel-leaderboard'>"
        + _render_leaderboard_cards(
            "<div class='scoreboard-wrap'>"
            "<div id='scoreboard-graph' style='width:100%;min-height:320px;'></div>"
            "</div>",
            "<div id='lb-summary-container'></div>",
            "<div id='lb-detail-container'></div>",
            roles,
            model_name,
        )
        + "</div>"
    )
    # The JS table renderers need the roles, header labels and badges.
    lb_meta_json = json.dumps(
        {
            "roles": roles,
            "badges": {did: role_badge(did, roles) for did in roles},
            "headers": {
                col: detail_header(col)
                for col in ("data_id", "tier", "diagnostic", "var_id", *METRIC_INFO)
            },
            "overallHeader": detail_header("overall"),
            "unrankedNote": UNRANKED_NOTE,
        }
    )

    # ── Tier panels ──
    all_toolbars = ""
    all_panels = lb_panel

    def _diag_label(diag_name: str, dtype: str) -> str:
        return _diag_display_label(diag_name, dtype)

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
            return v["id"]

        title_parts = {
            v["id"]: _var_title(v["id"], v["long_name"], v["units"]) for v in variables
        }

        var_buttons = "".join(
            "<button type='button' class='var-pill {active}' data-var='{vid}' "
            "aria-pressed='{pressed}' title='{tooltip}' "
            'onclick=\'showVarPanel("{scope}", "{vid}")\'>{label}</button>'.format(
                active="active" if v["id"] == first_var else "",
                pressed="true" if v["id"] == first_var else "false",
                vid=html_module.escape(v["id"]),
                tooltip=html_module.escape(title_parts[v["id"]][0]),
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
            search_key = html_module.escape(f"{vid} {v['long_name']}".lower())
            panels += (
                f"<div class='plot-panel {active}' id='{panel_id}' "
                f"data-graph='{graph_id}' data-search='{search_key}'>"
                f"<div class='plot-title-bar'>"
                f"<span class='plot-title-text'>{title_parts[vid][1]}</span>"
                f"</div>"
                f"<div id='{graph_id}' style='height:{height}px;width:100%;' "
                f"data-api-url='{html_module.escape(api_url)}'></div>"
                f"</div>"
            )

        return (
            f"<div id='diag-{scope_id}' class='diag-content'>"
            f"{_render_var_nav(scope_id, var_buttons, len(variables))}"
            f"{panels}"
            f"</div>"
        )

    for tier in tiers:
        tier_id = tier["id"]
        db_data: dict[str, Any] = tier["db_data"]
        diag_types: dict[str, str] = tier["diag_types"]

        # Collect diagnostics with variables
        diags_for_toolbar: list[tuple[str, str]] = []  # (diag_name, dtype)
        section_html_by_diag: dict[str, str] = {}

        for diag_name, tables in db_data.items():
            if "raw_output" not in tables:
                continue
            data_df: pd.DataFrame = tables["raw_output"]
            variables_df: pd.DataFrame = tables.get("variables", pd.DataFrame())
            dtype = diag_types.get(diag_name, "Unknown")

            variables = []
            for var_id in get_var_cols(data_df):
                # Histogram figures are keyed by the binned variable; the
                # density columns are their y-values, not variables of their own.
                if dtype == "Histogram" and var_id.startswith("density_"):
                    continue
                if data_df[var_id].isna().all():
                    continue
                long_name, units = _var_label(var_id, variables_df)
                variables.append({"id": var_id, "long_name": long_name, "units": units})

            if not variables:
                continue

            diags_for_toolbar.append((diag_name, dtype))
            label = _diag_label(diag_name, dtype)
            content = _render_shell_diagnostic(tier_id, diag_name, dtype, variables)
            section_html_by_diag[diag_name] = (
                f"<div class='diag-section' data-diag='{diag_name}' "
                f"id='diag-sec-{tier_id}-{diag_name}'>"
                f"{_section_header(label, dtype)}"
                f"{content}"
                f"</div>"
            )
        sections_html = _wrap_sections_by_category(section_html_by_diag, diag_types)

        all_toolbars += _render_toolbar(
            tier_id,
            Path(tier["db_path"]),
            [(dname, _diag_label(dname, dtype)) for dname, dtype in diags_for_toolbar],
            diag_types,
        )
        all_panels += (
            f"<div class='content-panel' id='panel-{tier_id}' data-type='tier'>"
            f"{sections_html}"
            f"<div class='search-no-results'>No plots match your search.</div>"
            f"</div>"
        )

    header_html = _render_app_header(model_name, user_name, developer, run_info, now)

    # In server mode, serve Plotly.js via the /plotly.js endpoint so the HTML
    # shell stays small and loads instantly.
    plotly_tag = '<script src="/plotly.js"></script>'

    # Leaderboard JS: fetch data from /api/leaderboard and render
    # Raw string: the \" inside the JS must reach the browser as an escape.
    lb_js = r"""
<script>
var _lbMeta = __LB_META__;
(function() {
  fetch('/api/leaderboard')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (data.scoreboard) {
        var el = document.getElementById('scoreboard-graph');
        if (el) {
          data.scoreboard.layout.autosize = true;
          _plotThemed(el, data.scoreboard.data, data.scoreboard.layout,
            {responsive:true, displayModeBar:false});
        }
      }
      if (data.summary) _renderSummaryTable(data.summary);
      if (data.detail)  _renderDetailTable(data.detail);
    })
    .catch(function(e) { console.warn('Leaderboard load failed:', e); });
})();

function _isModel(ds) {
  return !!(_lbMeta.roles[ds] && _lbMeta.roles[ds].role === 'to_benchmark');
}

function _renderSummaryTable(rows) {
  var medals = {1:'medal-1', 2:'medal-2', 3:'medal-3'};
  var html = "<div class='lb-table-wrap'><table class='lb' id='lb-summary'>"
    + "<thead><tr>"
    + "<th class='sortable' onclick='sortTable(\"lb-summary\",0,true)'>Rank</th>"
    + "<th class='sortable' onclick='sortTable(\"lb-summary\",1,false)'>Dataset</th>"
    + "<th class='sortable' onclick='sortTable(\"lb-summary\",2,true)'>" + _lbMeta.overallHeader + "</th>"
    + "</tr></thead><tbody>";
  var unranked = false;
  rows.forEach(function(row) {
    var rank = typeof row['Rank'] === 'number' ? row['Rank'] : null;
    var ds   = row['data_id'] || '';
    var sc   = typeof row['Overall Score'] === 'number' ? row['Overall Score'] : 0;
    var pct  = (sc * 100).toFixed(1);
    var cls  = rank ? (medals[rank] || '') : '';
    var rowCls = _isModel(ds) ? 'is-model' : (rank ? '' : 'is-ref');
    if (!rank) unranked = true;
    html += "<tr class='" + rowCls + "'>"
      + "<td class='rank " + cls + "'>" + (rank ? rank : '–') + "</td>"
      + "<td>" + _esc(ds) + (_lbMeta.badges[ds] || '') + "</td>"
      + "<td class='score-cell bar-cell'>"
      + "<div class='bar-bg' style='width:" + pct + "%;'></div>"
      + "<span class='bar-val'>" + pct + "%</span>"
      + "</td></tr>";
  });
  html += "</tbody></table></div>";
  if (unranked) html += "<p class='lb-note'>" + _lbMeta.unrankedNote + "</p>";
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
      + (_lbMeta.headers[k] || _esc(k)) + "</th>";
  });
  html += "</tr></thead><tbody>";
  rows.forEach(function(row) {
    html += "<tr class='" + (_isModel(row['data_id']) ? 'is-model' : '') + "'>";
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
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ClimateEval Report</title>
  {_THEME_BOOT}
  <style>{_CSS}</style>
</head>
<body>
<div class="app-shell">

  {header_html}

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

{plotly_tag}
<script>{_JS}</script>
{lb_js.replace("__LB_META__", lb_meta_json)}
</body>
</html>"""
