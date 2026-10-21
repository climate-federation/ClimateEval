"""CSS, JS, colour constants and colorscale helpers for the benchmarking report."""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

_PLOT_FONT = {
    "family": (
        "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
        "'Helvetica Neue', Arial, sans-serif"
    ),
    "size": 12,
    "color": "#1b1b19",
}
_AXIS_STYLE = {
    "showgrid": True,
    "gridcolor": "#ececea",
    "gridwidth": 1,
    "zeroline": False,
    "linecolor": "#cfcfca",
    "linewidth": 1,
    "mirror": False,
    "ticks": "outside",
    "tickcolor": "#cfcfca",
    "ticklen": 4,
}

# Line colours for the benchmarked model(s): the Okabe-Ito palette, the
# colour-blind-safe set standard in scientific figures. References are black
# and the comparison ensemble grey (see ``_get_line_properties``).
_SEABORN_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermilion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]
_OTHER_LINE_COLOR = "#9a9a94"

# ---------------------------------------------------------------------------
# Colorscales
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
    # Sea-ice concentration (ICONEval sea_ice_maps: Blues_r)
    "siconc": "Blues_r",
}

# Variables whose fields should be displayed with a symmetric range around zero
# (diverging colormaps rely on this to keep the "zero" colour in the middle).
_DIVERGING_VARS: set[str] = {
    "lwcre",
    "swcre",
    "ua",
    "ua200",
    "ua800",
    # TEM diagnostics — all symmetric around zero
    "psi_star",
    "v_star",
    "w_star",
    "w_dia",
}

# Per-variable override for the bias (model − reference) colormap. Falls back
# to _DEFAULT_BIAS_COLORSCALE otherwise.
_VAR_BIAS_COLORSCALE: dict[str, str] = {
    "pr": "BrBG",  # ICONEval basics_maps uses BrBG for precipitation bias
}

_DEFAULT_COLORSCALE = "RdBu_r"
_DEFAULT_BIAS_COLORSCALE = "RdBu_r"


_HOVMOLLER_COLORSCALE: dict[str, str] = {
    "pr": "BrBG",  # wet (green) / dry (brown) anomaly
    "rlut": "RdBu",  # warm (red) / cold (blue) OLR anomaly
}
_DEFAULT_HOVMOLLER_COLORSCALE = "RdBu_r"


def _base_var(var_id: str) -> str:
    """Return the physical variable behind a composite id (siconc_nh_map → siconc)."""
    return var_id.split("_", maxsplit=1)[0]


def _colorscale_for(var_id: str) -> str:
    if var_id in _VAR_COLORSCALE:
        return _VAR_COLORSCALE[var_id]
    return _VAR_COLORSCALE.get(_base_var(var_id), _DEFAULT_COLORSCALE)


def _hovmoller_colorscale_for(var_id: str) -> str:
    if var_id in _HOVMOLLER_COLORSCALE:
        return _HOVMOLLER_COLORSCALE[var_id]
    return _HOVMOLLER_COLORSCALE.get(_base_var(var_id), _DEFAULT_HOVMOLLER_COLORSCALE)


def _bias_colorscale_for(var_id: str) -> str:
    if var_id in _VAR_BIAS_COLORSCALE:
        return _VAR_BIAS_COLORSCALE[var_id]
    return _VAR_BIAS_COLORSCALE.get(_base_var(var_id), _DEFAULT_BIAS_COLORSCALE)


# ---------------------------------------------------------------------------
# Per-tier accent (visual separation of Tier 1/2/3 in the report)
# ---------------------------------------------------------------------------

# Muted (ink, tint, line) triples keyed by tier number (parsed from the
# database/panel label), rendered as a small badge rather than a colour flood.
# Panels of the same tier share a badge; the badge text carries the identity,
# the tint only supports it.
_TIER_ACCENTS: dict[int, tuple[str, str, str]] = {
    1: ("#2a6b63", "#e3efec", "#bcd9d3"),  # Tier 1 (sanity & consistency)
    2: ("#3d4f86", "#e6e9f4", "#c3c9e4"),  # Tier 2 (mean state vs obs)
    3: ("#8a5a1c", "#f6ecdc", "#e4cfae"),  # Tier 3 (variability / processes)
    4: ("#6b3f7a", "#f0e8f3", "#d9c6df"),
}
_TIER_ACCENT_FALLBACK = ("#4b4b47", "#f0f0ee", "#d9d9d5")  # no tier number
_TIER_RE = re.compile(r"tier\s*([0-9]+)", re.IGNORECASE)


def _tier_accent(label: str) -> tuple[str, str]:
    """Return (inline CSS custom properties, short badge text) for a tier.

    The first element is meant for a ``style`` attribute and sets the
    ``--tier-ink`` / ``--tier-tint`` / ``--tier-line`` variables that
    ``.tier-badge`` reads.
    """
    match = _TIER_RE.search(label)
    if match:
        num = int(match.group(1))
        ink, tint, line = _TIER_ACCENTS.get(num, _TIER_ACCENT_FALLBACK)
        badge = f"T{num}"
    else:
        ink, tint, line = _TIER_ACCENT_FALLBACK
        badge = "•"
    return f"--tier-ink:{ink};--tier-tint:{tint};--tier-line:{line};", badge


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
*, *::before, *::after { box-sizing: border-box; }

:root {
  /* neutral surfaces */
  --bg: #f6f6f4;
  --surface: #ffffff;
  --surface-2: #fafaf8;
  --border: #e3e3df;
  --border-strong: #cfcfca;
  /* text */
  --text: #1b1b19;
  --text-2: #4b4b47;
  --muted: #7a7a74;
  /* accent: deep teal, used sparingly for active states and links */
  --accent: #1d5c63;
  --accent-ink: #164a50;
  --accent-tint: #e8f0f0;
  --accent-tint-2: #d3e3e3;
  /* status */
  --danger: #a33a2d;
  --danger-tint: #fbf0ee;
  /* sizing */
  --sidebar-w: 248px;
  --toolbar-h: 48px;
  --radius: 6px;
  --space: 8px;
  --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace;
}

html, body {
  height: 100%; margin: 0; padding: 0;
  font-family: var(--font); font-size: 14px; line-height: 1.45; color: var(--text);
  background: var(--bg); overflow: hidden;
  -webkit-font-smoothing: antialiased;
}

button { font-family: inherit; }

/* Keyboard focus: one consistent ring everywhere, hidden for mouse users */
:focus { outline: none; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 3px; }

/* ── App shell ── */
.app-shell { display: flex; flex-direction: column; height: 100vh; overflow: hidden; }

/* ── Header ── */
.app-header {
  flex-shrink: 0; height: 52px;
  background: var(--surface); color: var(--text);
  padding: 0 24px; display: flex; align-items: center; gap: 24px;
  border-bottom: 1px solid var(--border);
  z-index: 300;
}
.header-title { flex: 1; min-width: 0; display: flex; align-items: baseline; gap: 20px; }
.header-title h1 {
  margin: 0; font-size: 15px; font-weight: 600; letter-spacing: .01em;
  color: var(--text); white-space: nowrap;
}
.header-title h1 span { color: var(--muted); font-weight: 400; margin-left: 8px; }
.header-meta {
  display: flex; gap: 0; flex-wrap: nowrap; min-width: 0; overflow: hidden;
  font-size: 12.5px; color: var(--muted);
}
.header-chip {
  display: inline-flex; align-items: baseline; gap: 5px; white-space: nowrap;
  padding: 0 14px; border-left: 1px solid var(--border);
}
.header-chip:first-child { border-left: none; padding-left: 0; }
.header-chip strong { color: var(--text-2); font-weight: 500; }

/* ── Body split ── */
.app-body { display: flex; flex: 1; overflow: hidden; position: relative; }

/* ── Sidebar ── */
.sidebar {
  flex-shrink: 0; width: var(--sidebar-w);
  background: var(--surface); color: var(--text-2);
  display: flex; flex-direction: column;
  overflow-y: auto; overflow-x: hidden;
  border-right: 1px solid var(--border);
  padding: 12px 0 24px;
  transition: width .18s ease, padding .18s ease;
}
.sidebar.collapsed { width: 0; border-right: none; padding: 0; }

.sidebar-toggle {
  position: absolute; top: 10px;
  left: calc(var(--sidebar-w) - 16px);
  width: 28px; height: 28px; border-radius: 50%;
  background: var(--surface); color: var(--text-2);
  border: 1px solid var(--border-strong);
  cursor: pointer; z-index: 150; user-select: none; padding: 0;
  transition: left .18s ease, background .12s;
}
.sidebar-toggle:hover { background: var(--surface-2); }
.app-body.sidebar-collapsed .sidebar-toggle { left: 12px; }
.content-toolbar { transition: padding-left .18s ease; }
.app-body.sidebar-collapsed .content-toolbar { padding-left: 52px; }

.sidebar-toggle .bar {
  position: absolute; left: 7px; width: 12px; height: 1.5px;
  background: var(--text-2); border-radius: 1px;
  transition: top .2s ease, transform .2s ease, opacity .15s ease;
}
.sidebar-toggle .bar-top { top: 12.5px; transform: rotate(45deg); }
.sidebar-toggle .bar-mid { top: 12.5px; opacity: 0; }
.sidebar-toggle .bar-bot { top: 12.5px; transform: rotate(-45deg); }
.app-body.sidebar-collapsed .sidebar-toggle .bar-top { top: 8.5px;  transform: none; }
.app-body.sidebar-collapsed .sidebar-toggle .bar-mid { top: 12.5px; opacity: 1; }
.app-body.sidebar-collapsed .sidebar-toggle .bar-bot { top: 16.5px; transform: none; }

.sidebar-section-label {
  font-size: 11px; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: var(--muted);
  padding: 14px 20px 6px;
}
.sidebar-item {
  display: flex; align-items: center; gap: 10px; width: 100%;
  padding: 7px 20px 7px 18px; cursor: pointer; text-align: left;
  font-size: 13.5px; font-weight: 450; color: var(--text-2);
  background: none; border: none; border-left: 2px solid transparent;
  transition: background .1s, color .1s;
  user-select: none;
}
.sidebar-item:hover { background: var(--surface-2); color: var(--text); }
.sidebar-item.active {
  background: var(--accent-tint); color: var(--accent-ink);
  border-left-color: var(--accent); font-weight: 600;
}
.sidebar-item .sidebar-icon { flex-shrink: 0; width: 16px; height: 16px; color: inherit; opacity: .8; }
.sidebar-item .sidebar-icon svg { width: 16px; height: 16px; display: block; }
.sidebar-item-text { min-width: 0; white-space: normal; line-height: 1.3; }

/* tier badges: small, mono, tinted - identity without colour floods */
.tier-badge {
  flex-shrink: 0; display: inline-flex; align-items: center; justify-content: center;
  min-width: 26px; height: 18px; padding: 0 5px;
  border-radius: 4px; font-size: 10.5px; font-weight: 600; font-family: var(--mono);
  letter-spacing: .02em; color: var(--tier-ink, var(--text-2));
  background: var(--tier-tint, var(--surface-2)); border: 1px solid var(--tier-line, var(--border));
}
.sidebar-divider { height: 1px; background: var(--border); margin: 10px 20px; }

/* ── Content area ── */
.content-wrap { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }

/* ── Toolbar (diagnostic filter + search) ── */
.content-toolbar {
  flex-shrink: 0; min-height: var(--toolbar-h);
  background: var(--surface); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 16px;
  padding: 0 24px; z-index: 100;
}
.diag-filter-btns {
  display: flex; gap: 2px; align-items: stretch; flex-shrink: 1; min-width: 0;
  overflow-x: auto; scrollbar-width: none; height: var(--toolbar-h);
}
.diag-filter-btns::-webkit-scrollbar { display: none; }
.diag-filter-btn {
  background: none; border: none; border-bottom: 2px solid transparent;
  padding: 0 10px; font-size: 13px; font-weight: 500; cursor: pointer;
  color: var(--muted); white-space: nowrap;
  transition: color .1s, border-color .1s;
}
.diag-filter-btn:hover { color: var(--text); }
.diag-filter-btn.active { color: var(--accent-ink); border-bottom-color: var(--accent); }
.diag-filter-btn:focus-visible { outline-offset: -3px; }

.toolbar-sep { width: 1px; height: 22px; background: var(--border); flex-shrink: 0; }

.search-wrap { position: relative; width: 240px; flex-shrink: 0; margin-left: auto; }
.search-wrap svg {
  position: absolute; left: 9px; top: 50%; transform: translateY(-50%);
  width: 14px; height: 14px; color: var(--muted); pointer-events: none;
}
.search-input {
  width: 100%; padding: 6px 10px 6px 30px;
  border: 1px solid var(--border); border-radius: var(--radius);
  font-size: 13px; font-family: var(--font); color: var(--text);
  background: var(--surface); outline: none;
  transition: border-color .12s;
}
.search-input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-tint); }
.search-input::placeholder { color: var(--muted); }

.toolbar-db {
  flex-shrink: 1; min-width: 0; max-width: 260px;
  font-size: 12px; color: var(--muted); font-family: var(--mono);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ── Main scrolling content ── */
.main-content { flex: 1; overflow-y: auto; padding: 24px 32px 64px; }
.content-panel { display: none; max-width: 1400px; }
.content-panel.active { display: block; }

/* ── Plot-type category grouping ── */
.diag-category + .diag-category { margin-top: 36px; }
.diag-category:not(:has(.diag-section:not(.hidden))) { display: none; }
.diag-category-header {
  margin: 0 0 14px; padding-bottom: 6px;
  font-size: 11.5px; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; color: var(--muted);
  border-bottom: 1px solid var(--border);
}
.content-panel.filtered .diag-category-header { display: none; }

/* ── Diagnostic sections (cards) ── */
.diag-section {
  margin-bottom: 24px; background: var(--surface);
  border: 1px solid var(--border); border-radius: var(--radius);
}
.diag-section.hidden { display: none; }
.diag-section-header {
  display: flex; align-items: center; gap: 12px;
  padding: 14px 20px; border-bottom: 1px solid var(--border);
}
.diag-section-title { font-size: 15px; font-weight: 600; color: var(--text); }
.diag-type-chip {
  font-size: 11px; font-weight: 500; color: var(--muted);
  padding: 1px 7px; border: 1px solid var(--border); border-radius: 999px;
}
.diag-content { padding: 16px 20px 20px; }

/* ── Section headings ── */
.section-heading {
  font-size: 20px; font-weight: 600; color: var(--text); letter-spacing: -.01em;
  margin: 0 0 6px;
}
.section-subheading { margin: 0 0 24px; color: var(--muted); font-size: 13.5px; max-width: 70ch; }

/* ── Variable nav ── */
.var-nav {
  display: flex; flex-wrap: wrap; gap: 4px; margin: 0 0 16px;
  padding-bottom: 12px; border-bottom: 1px solid var(--border);
}
.var-nav.many { max-height: 92px; overflow-y: auto; }
.var-pill {
  background: none; border: 1px solid transparent; border-radius: 4px;
  padding: 3px 9px; font-size: 12.5px; font-family: var(--mono); cursor: pointer;
  color: var(--text-2); line-height: 1.5;
  max-width: 240px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  transition: background .1s, color .1s, border-color .1s;
}
.var-pill:hover { background: var(--surface-2); border-color: var(--border); color: var(--text); }
.var-pill.active, .var-pill[aria-pressed="true"] {
  background: var(--accent-tint); color: var(--accent-ink); border-color: var(--accent-tint-2);
  font-weight: 600;
}

/* ── Plot panel ── */
.plot-panel { display: none; }
.plot-panel.active { display: block; }
/* figures with an explicit width (maps, profiles) scroll sideways inside the card */
.plot-panel > div[id^="graph-"] { overflow-x: auto; overflow-y: hidden; }
.plot-title-bar {
  display: flex; align-items: baseline; justify-content: space-between; gap: 12px;
  margin: 0 0 8px;
}
.plot-title-text { font-size: 14px; font-weight: 500; color: var(--text); }
.var-id-chip {
  display: inline-block; margin-left: 8px; font-family: var(--mono);
  font-size: 12px; font-weight: 400; color: var(--muted);
}
.plot-title-units { color: var(--muted); font-weight: 400; margin-left: 4px; }

/* ── Dataset detail box ── */
.plot-detail-footer { margin-top: 12px; padding-top: 10px; border-top: 1px solid var(--border); }
.detail-toggle-btn {
  background: none; border: none; padding: 2px 0;
  font-size: 12.5px; font-weight: 500; cursor: pointer; color: var(--accent-ink);
  display: inline-flex; align-items: center; gap: 6px;
}
.detail-toggle-btn:hover { text-decoration: underline; }
.detail-icon {
  font-style: normal; font-size: 10px; font-weight: 700; font-family: var(--mono);
  width: 15px; height: 15px; border-radius: 50%; border: 1px solid currentColor;
  display: inline-flex; align-items: center; justify-content: center;
}
.plot-detail-box { display: none; margin-top: 12px; }
.plot-detail-box.open { display: block; }
.plot-detail-body { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
.detail-block {
  background: var(--surface-2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 14px 16px; font-size: 12.5px;
}
.detail-block-title {
  font-weight: 600; color: var(--muted); font-size: 11px;
  text-transform: uppercase; letter-spacing: .08em; margin-bottom: 10px;
}
.detail-meta-row { display: flex; gap: 12px; padding: 3px 0; border-bottom: 1px solid var(--border); }
.detail-meta-row:last-child { border-bottom: none; }
.detail-meta-key { flex-shrink: 0; width: 140px; color: var(--muted); }
.detail-meta-val { color: var(--text); word-break: break-word; font-family: var(--mono); font-size: 12px; }
.detail-ds-item { margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid var(--border); }
.detail-ds-item:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.detail-ds-header { display: flex; align-items: center; gap: 8px; margin-bottom: 3px; font-family: var(--mono); font-size: 12px; }
.detail-badge {
  font-size: 10.5px; font-weight: 600; padding: 1px 7px; border-radius: 999px;
  text-transform: uppercase; letter-spacing: .04em; font-family: var(--font);
  border: 1px solid var(--border); color: var(--text-2); background: var(--surface);
}
.detail-badge-model { color: var(--accent-ink); border-color: var(--accent-tint-2); background: var(--accent-tint); }
.detail-ds-body { color: var(--muted); line-height: 1.6; }
.detail-ds-body a { color: var(--accent-ink); text-decoration: underline; text-decoration-color: var(--border-strong); }
.detail-ds-body a:hover { text-decoration-color: currentColor; }

/* ── Legend ── */
.legend-note {
  font-size: 12.5px; color: var(--muted); margin: 0 0 10px;
  display: flex; gap: 20px; flex-wrap: wrap; align-items: center;
}
.legend-note span { display: inline-flex; align-items: center; gap: 8px; }
.ln-model { display: inline-block; width: 24px; height: 2px; background: #0072b2; }
.ln-refline { display: inline-block; width: 24px; height: 2px; background: #1b1b19; }
.ln-other {
  display: inline-block; width: 24px; height: 2px;
  background: repeating-linear-gradient(to right, #9a9a94 0 5px, transparent 5px 9px);
}
.ln-data-swatch, .ln-bias-swatch {
  display: inline-block; width: 28px; height: 9px; border-radius: 2px; flex-shrink: 0;
  border: 1px solid rgba(0,0,0,.08);
}

/* ── Leaderboard ── */
.lb-section { max-width: 1100px; }
.metric-note { display: none; }
.lb-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 16px 20px; margin-bottom: 20px;
}
.lb-card-title { font-size: 14px; font-weight: 600; margin: 0 0 2px; }
.lb-card-sub { font-size: 12.5px; color: var(--muted); margin: 0 0 12px; }
.scoreboard-wrap { margin: 0; }
.scoreboard-plot { width: 100%; min-height: 320px; }
.lb-legend {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  font-size: 12px; color: var(--muted); margin-top: 8px;
}
.lb-legend-scale {
  width: 140px; height: 8px; border-radius: 2px; border: 1px solid rgba(0,0,0,.08);
  background: linear-gradient(to right, #b2182b, #f7f7f7, #2166ac);
}

.lb-table-wrap { overflow-x: auto; }
table.lb { border-collapse: collapse; width: 100%; background: transparent; font-size: 13.5px; }
table.lb thead th {
  background: transparent; color: var(--muted); font-weight: 600;
  padding: 8px 12px; text-align: left; white-space: nowrap;
  letter-spacing: .06em; font-size: 11px; text-transform: uppercase;
  border-bottom: 1px solid var(--border-strong);
}
table.lb thead th.sortable { cursor: pointer; user-select: none; }
table.lb thead th.sortable::after { content: " ↕"; opacity: .35; }
table.lb thead th.asc::after  { content: " ↑"; opacity: 1; }
table.lb thead th.desc::after { content: " ↓"; opacity: 1; }
table.lb tbody tr { border-bottom: 1px solid var(--border); }
table.lb tbody tr:last-child { border-bottom: none; }
table.lb tbody tr:hover { background: var(--surface-2); }
table.lb tbody td { padding: 9px 12px; white-space: nowrap; }
td.rank { font-variant-numeric: tabular-nums; color: var(--muted); text-align: right; width: 48px; }
td.score-cell { text-align: right; font-variant-numeric: tabular-nums; }
.medal-1, .medal-2, .medal-3 { color: var(--text); font-weight: 600; }
.bar-cell { position: relative; min-width: 180px; }
.bar-bg {
  position: absolute; top: 50%; left: 12px; height: 6px; margin-top: -3px;
  border-radius: 2px; background: var(--accent); opacity: .25; pointer-events: none;
  max-width: calc(100% - 80px);
}
.bar-val { position: relative; z-index: 1; font-weight: 600; font-variant-numeric: tabular-nums; }
.detail-section { margin-top: 8px; }
details > summary {
  cursor: pointer; font-weight: 500; color: var(--accent-ink);
  font-size: 13px; padding: 8px 0; user-select: none;
  list-style: none; display: flex; align-items: center; gap: 6px;
}
details > summary::-webkit-details-marker { display: none; }
details > summary::before { content: "▸"; font-size: 11px; transition: transform .12s; color: var(--muted); }
details[open] > summary::before { transform: rotate(90deg); }
details > summary:hover { text-decoration: underline; }

/* ── Search no-results ── */
.search-no-results { display: none; padding: 48px 0; text-align: center; color: var(--muted); }
.search-no-results.visible { display: block; }

/* ── Misc ── */
.empty-note { color: var(--muted); padding: 16px 20px; margin: 0; }

/* ── Server-mode loading / error states ── */
.plot-loading {
  display: flex; align-items: center; justify-content: center; gap: 12px;
  height: 200px; color: var(--muted); font-size: 13px;
}
.plot-loading-spinner {
  display: inline-block; width: 18px; height: 18px; border-radius: 50%;
  border: 2px solid var(--border); border-top-color: var(--accent);
  animation: spin .8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.plot-error {
  display: flex; align-items: center; justify-content: center;
  height: 120px; color: var(--danger); font-size: 13px;
  background: var(--danger-tint); border: 1px solid #ecc9c3; border-radius: var(--radius);
}

/* ── Scrollbars ── */
.sidebar::-webkit-scrollbar, .main-content::-webkit-scrollbar, .var-nav::-webkit-scrollbar { width: 8px; }
.sidebar::-webkit-scrollbar-thumb, .main-content::-webkit-scrollbar-thumb, .var-nav::-webkit-scrollbar-thumb {
  background: var(--border-strong); border-radius: 4px; border: 2px solid transparent; background-clip: content-box;
}
.sidebar::-webkit-scrollbar-track, .main-content::-webkit-scrollbar-track { background: transparent; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}
"""

# ---------------------------------------------------------------------------
# JavaScript
# ---------------------------------------------------------------------------

_JS = """
var _rendered = {};

function _applyFigData(div, figData) {
  // Only set autosize if the figure has no explicit width (maps have explicit width).
  if (!figData.layout.width) figData.layout.autosize = true;
  Plotly.newPlot(div, figData.data, figData.layout,
    {responsive:true, displayModeBar:true, displaylogo:false});
}

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
      // Already rendered: just trigger a lightweight resize check, not a full re-render.
      if (_rendered[graphId]) { Plotly.Plots.resize(div); return; }

      // Offline mode: read figure JSON embedded in the page
      var scriptEl = document.getElementById('figdata-' + graphId);
      if (scriptEl) {
        try {
          var figData = JSON.parse(scriptEl.textContent);
          _applyFigData(div, figData);
          _rendered[graphId] = true;
        } catch(e) { console.warn('renderPanel error:', e); }
        return;
      }

      // Server mode: fetch figure JSON from the API endpoint stored in data-api-url
      var apiUrl = div.getAttribute('data-api-url');
      if (!apiUrl) return;
      div.innerHTML = '<div class="plot-loading"><span class="plot-loading-spinner"></span>Loading…</div>';
      fetch(apiUrl)
        .then(function(r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function(figData) {
          div.innerHTML = '';
          _applyFigData(div, figData);
          _rendered[graphId] = true;
        })
        .catch(function(e) {
          div.innerHTML = '<div class="plot-error">Failed to load figure: ' + e.message + '</div>';
        });
    });
  });
}

// Progressively pre-render all offline figures during browser idle time so that
// switching variables feels instant after the page has settled.
function _preRenderQueue() {
  if (typeof Plotly === 'undefined' || typeof requestIdleCallback === 'undefined') return;
  var scripts = Array.from(document.querySelectorAll('script[type="application/json"][id^="figdata-"]'));
  var idx = 0;
  function next(deadline) {
    while (idx < scripts.length && deadline.timeRemaining() > 10) {
      var scriptEl = scripts[idx++];
      var graphId = scriptEl.id.slice(8); // strip 'figdata-' prefix
      if (_rendered[graphId]) continue;
      var div = document.getElementById(graphId);
      if (!div) continue;
      try {
        var figData = JSON.parse(scriptEl.textContent);
        _applyFigData(div, figData);
        _rendered[graphId] = true;
      } catch(e) {}
    }
    if (idx < scripts.length) requestIdleCallback(next, {timeout:8000});
  }
  // Delay start so the visible first panel renders before we begin background work.
  setTimeout(function() { requestIdleCallback(next, {timeout:8000}); }, 800);
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
  document.querySelectorAll('.sidebar-item').forEach(function(b){
    b.classList.remove('active'); b.removeAttribute('aria-current');
  });
  document.querySelectorAll('.content-toolbar').forEach(function(t){ t.style.display='none'; });

  var panel = document.getElementById('panel-' + sectionId);
  if (panel) panel.classList.add('active');

  var btn = document.querySelector('.sidebar-item[data-section="' + sectionId + '"]');
  if (btn) { btn.classList.add('active'); btn.setAttribute('aria-current', 'page'); }

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

  // Category headers only group things in the "All" view; with a single
  // diagnostic selected they just repeat the active filter pill.
  panel.classList.toggle('filtered', diagName !== 'all');

  var toolbar = document.getElementById('toolbar-' + tierId);
  if (toolbar) {
    toolbar.querySelectorAll('.diag-filter-btn').forEach(function(b){
      b.classList.remove('active'); b.setAttribute('aria-pressed', 'false');
    });
    var activeBtn = toolbar.querySelector('.diag-filter-btn[data-diag="' + diagName + '"]');
    if (activeBtn) { activeBtn.classList.add('active'); activeBtn.setAttribute('aria-pressed', 'true'); }
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
  diag.querySelectorAll('.var-pill').forEach(function(b){
    b.classList.remove('active'); b.setAttribute('aria-pressed', 'false');
  });
  var panelId = 'plot-' + scopeId + '-' + varId;
  var panel = document.getElementById(panelId);
  if (panel) panel.classList.add('active');
  var btn = diag.querySelector('.var-pill[data-var="' + varId + '"]');
  if (btn) { btn.classList.add('active'); btn.setAttribute('aria-pressed', 'true'); }
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
function toggleDetail(detailId, btn) {
  var box = document.getElementById(detailId);
  if (!box) return;
  var open = box.classList.toggle('open');
  if (btn) btn.setAttribute('aria-expanded', open ? 'true' : 'false');
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

// Kick off background pre-rendering for the portable HTML mode.
_preRenderQueue();
"""
