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
# References: solid dark lines, black first.
_REFERENCE_COLORS = ["#000000", "#6b6b66", "#8c564b", "#7b3294"]
# Compared ("other") models: thin lines, each a distinct colour + dash pair.
# The palette starts away from the benchmarked model's blue, and the colour
# and dash cycles have different lengths so the pairs stay distinct for 30
# datasets.
_OTHER_COLORS = ["#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#0072B2"]
_OTHER_DASHES = ["dash", "dot", "dashdot", "longdash", "longdashdot"]

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
# The same badges on dark surfaces: lighter ink on a deep tint.
_TIER_ACCENTS_DARK: dict[int, tuple[str, str, str]] = {
    1: ("#8fd0c4", "#1b3330", "#2f5a53"),
    2: ("#aab6e6", "#222a45", "#3a4672"),
    3: ("#e0b36b", "#3a2a14", "#64491f"),
    4: ("#cfa6dc", "#32223a", "#5a3f66"),
}
_TIER_ACCENT_FALLBACK_DARK = ("#b4b4ae", "#2a2a28", "#45453f")
_TIER_RE = re.compile(r"tier\s*([0-9]+)", re.IGNORECASE)


def _tier_accent(label: str) -> tuple[str, str]:
    """Return (inline CSS custom properties, short badge text) for a tier.

    The first element is meant for a ``style`` attribute and sets the
    ``--tier-ink`` / ``--tier-tint`` / ``--tier-line`` variables that
    ``.tier-badge`` reads, plus their ``-d`` twins for the dark theme.
    """
    match = _TIER_RE.search(label)
    if match:
        num = int(match.group(1))
        light = _TIER_ACCENTS.get(num, _TIER_ACCENT_FALLBACK)
        dark = _TIER_ACCENTS_DARK.get(num, _TIER_ACCENT_FALLBACK_DARK)
        badge = f"T{num}"
    else:
        light, dark = _TIER_ACCENT_FALLBACK, _TIER_ACCENT_FALLBACK_DARK
        badge = "•"
    style = "".join(
        f"--tier-ink{suffix}:{ink};--tier-tint{suffix}:{tint};--tier-line{suffix}:{line};"
        for suffix, (ink, tint, line) in (("", light), ("-d", dark))
    )
    return style, badge


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
  /* text placed on a filled accent / tooltip background */
  --on-accent: #ffffff;
  --danger-line: #ecc9c3;
  color-scheme: light;
}

/* Dark theme: chosen explicitly, or inherited from the system in "auto" */
:root[data-theme="dark"] { --bg: #151514;
  --surface: #1e1e1c;
  --surface-2: #262624;
  --border: #2f2f2c;
  --border-strong: #45453f;
  --text: #e6e6e2;
  --text-2: #c0c0ba;
  --muted: #8f8f89;
  --accent: #5fa8ae;
  --accent-ink: #8fc7cb;
  --accent-tint: #1b3134;
  --accent-tint-2: #244247;
  --danger: #e08a7e;
  --danger-tint: #33201d;
  --danger-line: #5a2f29;
  --on-accent: #0f1f21;
  color-scheme: dark; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg: #151514;
  --surface: #1e1e1c;
  --surface-2: #262624;
  --border: #2f2f2c;
  --border-strong: #45453f;
  --text: #e6e6e2;
  --text-2: #c0c0ba;
  --muted: #8f8f89;
  --accent: #5fa8ae;
  --accent-ink: #8fc7cb;
  --accent-tint: #1b3134;
  --accent-tint-2: #244247;
  --danger: #e08a7e;
  --danger-tint: #33201d;
  --danger-line: #5a2f29;
  --on-accent: #0f1f21;
  color-scheme: dark; }
}
:root[data-theme="dark"] { .tier-badge {
    color: var(--tier-ink-d, var(--text-2)); background: var(--tier-tint-d, var(--surface-2));
    border-color: var(--tier-line-d, var(--border));
  }
  .ln-data-swatch, .ln-bias-swatch, .lb-legend-scale { border-color: rgba(255,255,255,.12); }
  .info-tip .tip { box-shadow: 0 2px 8px rgba(0,0,0,.5); }
  /* Plotly's modebar icons are tuned for white paper */
  .modebar-btn .icon path { fill: var(--muted) !important; }
  .modebar-btn:hover .icon path, .modebar-btn.active .icon path { fill: var(--text) !important; } }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { .tier-badge {
    color: var(--tier-ink-d, var(--text-2)); background: var(--tier-tint-d, var(--surface-2));
    border-color: var(--tier-line-d, var(--border));
  }
  .ln-data-swatch, .ln-bias-swatch, .lb-legend-scale { border-color: rgba(255,255,255,.12); }
  .info-tip .tip { box-shadow: 0 2px 8px rgba(0,0,0,.5); }
  /* Plotly's modebar icons are tuned for white paper */
  .modebar-btn .icon path { fill: var(--muted) !important; }
  .modebar-btn:hover .icon path, .modebar-btn.active .icon path { fill: var(--text) !important; } }
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

/* three-way theme switch: auto (system) / light / dark */
.theme-switch {
  flex-shrink: 0; margin-left: auto; display: inline-flex; align-items: center;
  border: 1px solid var(--border-strong); border-radius: var(--radius); overflow: hidden;
}
.theme-switch button {
  display: inline-flex; align-items: center; gap: 6px;
  background: var(--surface); border: none; border-left: 1px solid var(--border-strong);
  padding: 4px 10px; font-size: 12px; color: var(--text-2); cursor: pointer; line-height: 1.4;
}
.theme-switch button:first-child { border-left: none; }
.theme-switch button:hover { background: var(--surface-2); color: var(--text); }
.theme-switch button[aria-pressed="true"] { background: var(--accent); color: var(--on-accent); font-weight: 600; }
.theme-switch button:focus-visible { outline-offset: -2px; }
.theme-switch svg { width: 13px; height: 13px; display: block; }
@media (max-width: 760px) { .theme-switch button span { display: none; } .theme-switch button { padding: 5px 8px; } }

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
/* a single selected diagnostic is named by its toolbar tab: no card title,
   and no group spacing left over from the hidden categories */
.content-panel.filtered .diag-section-header { display: none; }
.content-panel.filtered .diag-category + .diag-category { margin-top: 0; }

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
.diag-content { padding: 0 20px 20px; }

/* ── Section headings ── */
.section-heading {
  font-size: 20px; font-weight: 600; color: var(--text); letter-spacing: -.01em;
  margin: 0 0 6px;
}
.section-subheading { margin: 0 0 24px; color: var(--muted); font-size: 13.5px; max-width: 70ch; }

/* ── Variable nav ── */
/* sticks to the top while a tall figure scrolls; the negative side margins
   let its background span the card so nothing shows beside it */
.var-nav-wrap {
  position: sticky; top: 0; z-index: 5; background: var(--surface);
  margin: 0 -20px 16px; padding: 16px 20px 0; border-bottom: 1px solid var(--border);
}
.var-nav { display: flex; flex-wrap: wrap; gap: 4px; padding-bottom: 12px; }
/* many variables: two rows until expanded; the height is set from the real
   pill height by _syncVarNavs so no partial row is ever clipped */
.var-nav.many:not(.expanded) { overflow: hidden; }
.var-nav-more {
  display: block; margin: -4px 0 8px; padding: 2px 0; background: none; border: none;
  font-size: 12px; color: var(--accent-ink); cursor: pointer;
}
.var-nav-more:hover { text-decoration: underline; }
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
.ln-refline { display: inline-block; width: 24px; height: 2px; background: var(--text); }
.ln-other {
  display: inline-block; width: 24px; height: 2px;
  background: repeating-linear-gradient(to right, var(--muted) 0 5px, transparent 5px 9px);
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

/* ── Info tooltips ── */
.info-tip {
  position: relative; display: inline-flex; align-items: center; justify-content: center;
  width: 15px; height: 15px; margin-left: 6px; border-radius: 50%;
  border: 1px solid var(--border-strong); color: var(--muted);
  font: 600 10px/1 var(--mono); cursor: help; vertical-align: middle;
  text-transform: none; letter-spacing: 0;
}
.info-tip:hover, .info-tip:focus-visible { color: var(--accent-ink); border-color: var(--accent); }
.info-tip .tip {
  display: none; position: absolute; left: 50%; top: calc(100% + 8px); transform: translateX(-50%);
  width: 280px; padding: 8px 10px; background: var(--text); color: var(--surface);
  font: 400 12px/1.45 var(--font); text-transform: none; letter-spacing: 0;
  border-radius: var(--radius); white-space: normal; text-align: left;
  z-index: 400; box-shadow: 0 2px 8px rgba(0,0,0,.18); pointer-events: none;
}
.info-tip:hover .tip, .info-tip:focus-visible .tip { display: block; }
th .info-tip .tip { left: auto; right: 0; transform: none; }

/* ── Leaderboard: who is compared ── */
.lb-compare {
  display: flex; flex-wrap: wrap; gap: 10px 32px; margin: 0 0 20px; padding: 12px 16px;
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  font-size: 13px; color: var(--text-2); max-width: 1100px;
}
.lb-compare dt {
  color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  margin: 0 0 2px;
}
.lb-compare dd { margin: 0; }
.lb-compare strong { color: var(--accent-ink); font-weight: 600; }
.lb-role {
  display: inline-block; margin-left: 8px; padding: 1px 7px; border-radius: 999px;
  font-size: 10.5px; font-weight: 600; letter-spacing: .04em; text-transform: uppercase;
  border: 1px solid var(--border); color: var(--muted); background: var(--surface);
  vertical-align: middle; font-family: var(--font);
}
.lb-role-model { color: var(--accent-ink); border-color: var(--accent-tint-2); background: var(--accent-tint); }
table.lb tr.is-model td { background: var(--accent-tint); font-weight: 600; }
table.lb tr.is-model td:first-child { box-shadow: inset 2px 0 0 var(--accent); }
table.lb tr.is-model:hover td { background: var(--accent-tint-2); }

/* ── Figure toggles (segmented controls above a plot) ── */
.fig-toggles { display: flex; flex-wrap: wrap; gap: 8px 24px; margin: 0 0 10px; align-items: center; }
.seg-wrap { display: inline-flex; align-items: center; gap: 8px; }
.seg-label { font-size: 12px; color: var(--muted); }
.seg { display: inline-flex; border: 1px solid var(--border-strong); border-radius: var(--radius); overflow: hidden; }
.seg button {
  background: var(--surface); border: none; border-left: 1px solid var(--border-strong);
  padding: 4px 12px; font-size: 12.5px; color: var(--text-2); cursor: pointer; line-height: 1.4;
}
.seg button:first-child { border-left: none; }
.seg button:hover { background: var(--surface-2); color: var(--text); }
.seg button[aria-pressed="true"] { background: var(--accent); color: var(--on-accent); font-weight: 600; }
.seg button:focus-visible { outline-offset: -2px; }

/* ── Figure range controls (Hovmöller time / longitude windows) ── */
.fig-ranges {
  display: flex; flex-wrap: wrap; gap: 10px 28px; align-items: center; margin: 0 0 12px;
}
.range-ctl { display: inline-flex; align-items: center; gap: 8px; font-size: 12px; color: var(--muted); }
.range-ctl input[type="text"] {
  width: 96px; padding: 3px 6px; font: 12px var(--mono); color: var(--text);
  background: var(--surface); border: 1px solid var(--border-strong); border-radius: 4px;
}
.range-ctl input[type="text"]:focus { border-color: var(--accent); outline: none; }
.range-ctl input[type="text"].invalid { border-color: var(--danger); }
.range-reset {
  background: none; border: 1px solid var(--border-strong); border-radius: var(--radius);
  padding: 3px 10px; font-size: 12px; color: var(--text-2); cursor: pointer;
}
.range-reset:hover { background: var(--surface-2); color: var(--text); }
.dual-range { position: relative; width: 220px; height: 20px; flex-shrink: 0; }
.dual-range .track {
  position: absolute; left: 0; right: 0; top: 8.5px; height: 3px;
  background: var(--border-strong); border-radius: 2px;
}
.dual-range .fill { position: absolute; top: 8.5px; height: 3px; background: var(--accent); border-radius: 2px; }
.dual-range input[type="range"] {
  position: absolute; left: 0; top: 0; width: 100%; height: 20px; margin: 0;
  background: none; pointer-events: none; -webkit-appearance: none; appearance: none;
}
.dual-range input[type="range"]::-webkit-slider-runnable-track { height: 20px; background: transparent; }
.dual-range input[type="range"]::-moz-range-track { background: transparent; }
.dual-range input[type="range"]::-webkit-slider-thumb {
  -webkit-appearance: none; pointer-events: auto; cursor: pointer;
  width: 14px; height: 14px; margin-top: 3px; border-radius: 50%;
  background: var(--accent); border: 2px solid var(--surface); box-shadow: 0 0 0 1px var(--accent);
}
.dual-range input[type="range"]::-moz-range-thumb {
  pointer-events: auto; cursor: pointer; width: 10px; height: 10px; border-radius: 50%;
  background: var(--accent); border: 2px solid var(--surface); box-shadow: 0 0 0 1px var(--accent);
}
.dual-range input[type="range"]:focus-visible { outline: none; }
.dual-range input[type="range"]:focus-visible::-webkit-slider-thumb { box-shadow: 0 0 0 3px var(--accent-tint-2); }

/* ── Search no-results ── */
.search-no-results { display: none; padding: 48px 0; text-align: center; color: var(--muted); }
.search-no-results.visible { display: block; }

/* ── Misc ── */
.empty-note { color: var(--muted); padding: 16px 20px; margin: 0; }
.lb-note { margin: 8px 0 0; font-size: 12.5px; color: var(--muted); }
table.lb tr.is-ref td { color: var(--muted); }

/* diagnostic tabs: one row, grouped by plot-type category */
.diag-filter-btns { flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none; }
.diag-filter-sep {
  flex-shrink: 0; align-self: center; width: 1px; height: 16px;
  margin: 0 8px; background: var(--border-strong);
}

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
  background: var(--danger-tint); border: 1px solid var(--danger-line); border-radius: var(--radius);
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

_JS = r"""
var _rendered = {};
var _inflight = {};  // server-mode figure fetches not yet drawn
var _PLOT_CONFIG = {responsive:true, displayModeBar:true, displaylogo:false};

// ---------------------------------------------------------------------------
// Theme: "auto" follows the system, "light"/"dark" are explicit. The choice is
// kept in localStorage and applied to <html data-theme> (a tiny script in
// <head> does the same before first paint). Figures are built with the light
// palette; _themeFig swaps every known paper/ink/grid colour at plot time, and
// the swap is reversed when the page goes back to light.
// ---------------------------------------------------------------------------
var _THEME_KEY = 'climateeval-theme';
var _DARK_COLORS = {
  'white': '#1e1e1c', '#fff': '#1e1e1c', '#ffffff': '#1e1e1c',
  'rgba(255,255,255,0.9)': 'rgba(30,30,28,0.9)',
  'rgba(255,255,255,0.85)': 'rgba(30,30,28,0.85)',
  '#dce9f5': '#1c242e',
  'black': '#f2f2ee', '#000': '#f2f2ee', '#000000': '#f2f2ee',
  '#1b1b19': '#e6e6e2', '#333': '#d0d0cb', '#333333': '#d0d0cb', '#555': '#b0b0aa',
  '#1e2535': '#d8dde8', '#4b4b47': '#c0c0ba', '#7a7a74': '#9c9c96', '#6b6b66': '#c8c8c2',
  '#9a9a94': '#77776f', '#aaa': '#6e6e68', '#aaaaaa': '#6e6e68',
  '#ececea': '#2e2e2b', '#e3e3df': '#3a3a36', '#ddd': '#3b3b37', '#dddddd': '#3b3b37',
  '#bbb': '#4a4a45', '#bbbbbb': '#4a4a45', '#cfcfca': '#45453f',
  'rgba(0,0,0,0.45)': 'rgba(255,255,255,0.45)', 'rgba(0,0,0,0.40)': 'rgba(255,255,255,0.40)',
  'rgba(0,0,0,0.4)': 'rgba(255,255,255,0.4)', 'rgba(20,20,20,0.80)': 'rgba(235,235,230,0.80)',
  '#a07000': '#e0b24a'
};
var _LIGHT_COLORS = {};
Object.keys(_DARK_COLORS).forEach(function(k) { _LIGHT_COLORS[_DARK_COLORS[k]] = k; });
var _COLOR_KEY = /color(way)?$/i;

function _systemDark() {
  return !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);
}
function _themePref() {
  var t = document.documentElement.getAttribute('data-theme');
  return (t === 'light' || t === 'dark') ? t : 'auto';
}
function _isDark() {
  var t = _themePref();
  return t === 'dark' || (t === 'auto' && _systemDark());
}
// Deep-copy v, swapping colour-valued strings; numeric arrays are shared.
function _themeVal(v, key, map) {
  if (typeof v === 'string') {
    if (!key || !_COLOR_KEY.test(key)) return v;
    var m = map[v.replace(/\s+/g, '').toLowerCase()];
    return m === undefined ? v : m;
  }
  if (Array.isArray(v)) {
    if (!v.length || typeof v[0] === 'number') return v;
    return v.map(function(x) { return _themeVal(x, key, map); });
  }
  if (v && typeof v === 'object') {
    var o = {};
    for (var k in v) o[k] = _themeVal(v[k], k, map);
    return o;
  }
  return v;
}
function _themeFig(obj) { return _themeVal(obj, null, _isDark() ? _DARK_COLORS : _LIGHT_COLORS); }
// Every figure goes through here so a theme change can re-draw it in place.
function _plotThemed(div, data, layout, config) {
  div._plotCfg = config || _PLOT_CONFIG;
  return Plotly.newPlot(div, _themeFig(data), _themeFig(layout), div._plotCfg);
}
function _rethemeAll() {
  if (typeof Plotly === 'undefined') return;
  document.querySelectorAll('.js-plotly-plot').forEach(function(div) {
    if (!div.data || !div.layout) return;
    Plotly.react(div, _themeFig(div.data), _themeFig(div.layout), div._plotCfg || _PLOT_CONFIG);
  });
}
function _syncThemeSwitch() {
  var pref = _themePref();
  document.querySelectorAll('.theme-switch button').forEach(function(b) {
    b.setAttribute('aria-pressed', b.getAttribute('data-theme-pref') === pref ? 'true' : 'false');
  });
}
function setTheme(pref) {
  var root = document.documentElement;
  if (pref === 'light' || pref === 'dark') root.setAttribute('data-theme', pref);
  else root.removeAttribute('data-theme');
  try { localStorage.setItem(_THEME_KEY, pref); } catch (e) {}
  _syncThemeSwitch();
  _rethemeAll();
}
(function _initTheme() {
  var pref = 'auto';
  try { pref = localStorage.getItem(_THEME_KEY) || 'auto'; } catch (e) {}
  var root = document.documentElement;
  if (pref === 'light' || pref === 'dark') root.setAttribute('data-theme', pref);
  else root.removeAttribute('data-theme');
  _syncThemeSwitch();
  if (window.matchMedia) {
    var mq = window.matchMedia('(prefers-color-scheme: dark)');
    var onChange = function() { if (_themePref() === 'auto') _rethemeAll(); };
    if (mq.addEventListener) mq.addEventListener('change', onChange);
    else if (mq.addListener) mq.addListener(onChange);
  }
})();

// ---------------------------------------------------------------------------
// Responsive subplot grids (maps): side by side when the card is wide enough,
// stacked in one column with the colorbars along the bottom when it is not.
// The figure's layout.meta.responsive (written by _tag_responsive_grid in
// _figures.py) lists the panels, their axes and their title annotations.
// ---------------------------------------------------------------------------
function _gridLayout(figData, width) {
  var meta = figData.layout.meta.responsive;
  var layout = JSON.parse(JSON.stringify(figData.layout));
  var wide = width >= meta.wideMinWidth;
  var cols = wide ? meta.cols : 1;
  var margin = wide ? meta.marginWide : meta.marginNarrow;
  var nPanels = meta.panels.length;
  var rows = wide ? meta.rows : nPanels;
  var cbN = meta.colorbars.length;
  // px reserved under the panels: room for the last panel's tick labels and
  // axis title, then one row per horizontal colorbar
  var cbRow = 48, cbPad = 64;
  var cbBand = wide ? 0 : (cbN ? cbPad + cbRow * cbN : 0);
  // stacked panels each carry their own tick labels, so they need more air
  // two-line bias titles plus the tick labels of the panel above need ~56px
  var hgapPx = 24, vgapPx = wide ? 56 : 64;
  var plotW = Math.max(width - margin.l - margin.r, 200);
  // stacked panels are capped in width so a wide-but-not-wide-enough card
  // does not produce a very tall column of huge maps
  var usableW = wide ? plotW : Math.min(plotW, 640);
  var panelW = (usableW - hgapPx * (cols - 1)) / cols;
  var panelH = panelW * meta.aspect;
  var plotH = rows * panelH + vgapPx * (rows - 1);
  var height = Math.round(margin.t + margin.b + cbBand + plotH);
  var hgap = hgapPx / plotW, vgap = vgapPx / plotH;
  var cellW = ((usableW / plotW) - hgap * (cols - 1)) / cols;
  var xOffset = (1 - usableW / plotW) / 2;  // centre a capped stack
  var cellH = (1 - vgap * (rows - 1)) / rows;
  // paper fraction of the band below the panels (narrow mode only)
  var band = cbBand / (plotH + cbBand);
  var panelArea = 1 - band;

  meta.panels.forEach(function(p, i) {
    var r = wide ? p.row - 1 : i;
    var c = wide ? p.col - 1 : 0;
    var x0 = xOffset + c * (cellW + hgap), x1 = x0 + cellW;
    var yTop = 1 - r * (cellH + vgap) * panelArea;
    var yBot = yTop - cellH * panelArea;
    layout[p.x] = layout[p.x] || {}; layout[p.y] = layout[p.y] || {};
    layout[p.x].domain = [x0, x1];
    layout[p.y].domain = [Math.max(yBot, 0), yTop];
    if (p.ann !== null && layout.annotations && layout.annotations[p.ann]) {
      layout.annotations[p.ann].x = (x0 + x1) / 2;
      layout.annotations[p.ann].y = yTop;
    }
    if (!wide) {
      // stacked: only the last panel carries the x-axis title; every panel
      // keeps its own tick labels
      var src = figData.layout[p.x] || {};
      layout[p.x].title = (i === nPanels - 1) ? (src.title || {text: 'Longitude'}) : {text: ''};
    }
  });
  layout.margin = margin;
  layout.height = height;
  delete layout.width;
  layout.autosize = true;
  return {layout: layout, wide: wide, band: band, paperH: plotH + cbBand, cbRow: cbRow};
}

function _gridColorbars(figData, res) {
  var meta = figData.layout.meta.responsive;
  var cbN = meta.colorbars.length;
  var wide = res.wide;
  return meta.colorbars.map(function(ti, k) {
    var orig = figData.data[ti].colorbar || {};
    if (wide) return JSON.parse(JSON.stringify(orig));
    var cb = JSON.parse(JSON.stringify(orig));
    cb.orientation = 'h';
    cb.x = 0.5; cb.xanchor = 'center';
    cb.len = 0.9; cb.thickness = 12;
    // stack the bars in the lower rows of the band reserved below the
    // panels, first bar nearest the panels
    cb.y = (cbN - k - 0.5) * res.cbRow / res.paperH;
    cb.yanchor = 'middle';
    if (cb.title) cb.title.side = 'right';
    return cb;
  });
}

function _applyGrid(div, figData, first) {
  var width = div.clientWidth || div.parentNode.clientWidth || 900;
  var res = _gridLayout(figData, width);
  var data = figData.data.map(function(t) { return t; });
  var cbs = _gridColorbars(figData, res);
  var meta = figData.layout.meta.responsive;
  meta.colorbars.forEach(function(ti, k) {
    data[ti] = Object.assign({}, figData.data[ti], {colorbar: cbs[k]});
  });
  div.style.height = res.layout.height + 'px';
  // A full re-plot rather than Plotly.react: react keeps the previous
  // colorbars' margin pushers alive, and a vertical bar's pusher applied to
  // the much taller stacked layout swallows half the figure.
  if (!first) Plotly.purge(div);
  _plotThemed(div, data, res.layout, _PLOT_CONFIG);
  div._gridWidth = width;
}

function _watchGrid(div, figData) {
  if (typeof ResizeObserver === 'undefined') return;
  var pending = null;
  var ro = new ResizeObserver(function() {
    var w = div.clientWidth;
    if (!w || Math.abs(w - (div._gridWidth || 0)) < 4) return;
    clearTimeout(pending);
    pending = setTimeout(function() { _applyGrid(div, figData, false); }, 120);
  });
  ro.observe(div);
}

// ---------------------------------------------------------------------------
// Segmented toggles (Hovmöller axes / direction) in place of Plotly buttons.
// layout.meta.toggles.groups lists the controls; .buttons[i][j] names the
// updatemenu button whose arguments realise option i of group 0 with option j
// of group 1.
// ---------------------------------------------------------------------------
function _renderToggles(div, figData) {
  var meta = figData.layout.meta.toggles;
  var menus = figData.layout.updatemenus || [];
  if (!menus.length) return;
  figData.layout.updatemenus = [];
  var state = meta.groups.map(function() { return 0; });
  div._toggleState = state;
  var old = div.previousElementSibling;
  if (old && old.classList.contains('fig-toggles')) old.remove();
  var bar = document.createElement('div');
  bar.className = 'fig-toggles';
  function apply() {
    var ref = meta.buttons[state[0]][state[1]];
    var btn = menus[ref[0]] && menus[ref[0]].buttons[ref[1]];
    if (!btn) return;
    div._rangesBusy = true;
    Plotly.update(div, _themeFig(btn.args[0] || {}), _themeFig(btn.args[1] || {}));
    div._rangesBusy = false;
    // the update resets the axes to autorange; put the chosen windows back
    if (div._applyRanges) div._applyRanges();
    bar.querySelectorAll('.seg').forEach(function(seg, g) {
      seg.querySelectorAll('button').forEach(function(b, o) {
        b.setAttribute('aria-pressed', state[g] === o ? 'true' : 'false');
      });
    });
  }
  meta.groups.forEach(function(group, g) {
    var wrap = document.createElement('div');
    wrap.className = 'seg-wrap';
    var label = document.createElement('span');
    label.className = 'seg-label'; label.textContent = group.label;
    var seg = document.createElement('div');
    seg.className = 'seg'; seg.setAttribute('role', 'group'); seg.setAttribute('aria-label', group.label);
    group.options.forEach(function(opt, o) {
      var b = document.createElement('button');
      b.type = 'button'; b.textContent = opt;
      b.setAttribute('aria-pressed', o === 0 ? 'true' : 'false');
      b.addEventListener('click', function() { if (state[g] !== o) { state[g] = o; apply(); } });
      seg.appendChild(b);
    });
    wrap.appendChild(label); wrap.appendChild(seg); bar.appendChild(wrap);
  });
  div.parentNode.insertBefore(bar, div);
}

// ---------------------------------------------------------------------------
// Range controls (Hovmöller): a dual-handle slider and from/to text boxes per
// coordinate. layout.meta.ranges gives the extents and the axes of each
// panel; which coordinate sits on which axis follows the toggle state
// (Lon × Time: lon on x, time on y). Slider, boxes, plot zoom and toggles
// all keep each other in step.
// ---------------------------------------------------------------------------
var _DAY = 86400000;
function _isoDay(ms) { return new Date(ms).toISOString().slice(0, 10); }
function _parseDay(s) {
  s = String(s).trim();
  var m = /^(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?/.exec(s);
  if (!m) return NaN;
  var ms = Date.UTC(+m[1], m[2] ? +m[2] - 1 : 0, m[3] ? +m[3] : 1);
  return isNaN(ms) ? NaN : ms;
}

function _renderRanges(div, figData) {
  var meta = figData.layout.meta.ranges;
  var old = div.previousElementSibling;
  if (old && old.classList.contains('fig-ranges')) old.remove();
  var bar = document.createElement('div');
  bar.className = 'fig-ranges';

  // each coordinate: numeric domain [lo, hi] (dates as ms), current window
  var coords = {};
  ['lon', 'time'].forEach(function(key) {
    var spec = meta[key];
    if (!spec) return;
    var isDate = spec.type === 'date';
    var lo = isDate ? _parseDay(spec.min) : +spec.min;
    var hi = isDate ? _parseDay(spec.max) : +spec.max;
    coords[key] = {
      spec: spec, isDate: isDate, lo: lo, hi: hi, cur: [lo, hi],
      step: isDate ? _DAY : (hi - lo > 20 ? 1 : (hi - lo) / 200),
      fmt: function(v) { return isDate ? _isoDay(v) : (Math.round(v * 10) / 10).toString(); },
      parse: function(s) { return isDate ? _parseDay(s) : parseFloat(String(s).replace(/[^0-9.+-]/g, '')); },
      plotVal: function(v) { return isDate ? _isoDay(v) : v; }
    };
  });

  function axisFor(key) {
    // toggle group 0: 0 = Lon × Time (lon on x), 1 = Time × Lon (lon on y)
    var swapped = div._toggleState && div._toggleState[0] === 1;
    return (key === 'lon') === !swapped ? 'x' : 'y';
  }
  function orderedRange(key, lo, hi) {
    var c = coords[key];
    var reversed;
    if (key === 'lon') reversed = !!(div._toggleState && div._toggleState[1] === 1);  // E → W
    else reversed = axisFor('time') === 'y';  // time runs downwards on the y axis
    var a = c.plotVal(lo), b = c.plotVal(hi);
    return reversed ? [b, a] : [a, b];
  }
  function isFull(key) {
    var c = coords[key];
    return c.cur[0] <= c.lo && c.cur[1] >= c.hi;
  }

  // push the current windows into every panel's axes
  function applyRanges() {
    if (typeof Plotly === 'undefined' || !div.data) return;
    var patch = {};
    Object.keys(coords).forEach(function(key) {
      var ax = axisFor(key);
      meta.panels.forEach(function(p) {
        var name = p[ax];
        if (isFull(key)) {
          // back to the figure's own autorange, direction included
          var src = (div.layout && div.layout[name]) || {};
          var rev = (key === 'lon' && div._toggleState && div._toggleState[1] === 1) ||
                    (key === 'time' && ax === 'y');
          patch[name + '.autorange'] = rev ? 'reversed' : true;
          void src;
        } else {
          patch[name + '.range'] = orderedRange(key, coords[key].cur[0], coords[key].cur[1]);
        }
      });
    });
    div._rangesBusy = true;
    Plotly.relayout(div, patch).then(function() { div._rangesBusy = false; },
                                    function() { div._rangesBusy = false; });
  }
  div._applyRanges = applyRanges;

  var widgets = {};
  Object.keys(coords).forEach(function(key) {
    var c = coords[key];
    var ctl = document.createElement('div');
    ctl.className = 'range-ctl';
    var label = document.createElement('span');
    label.textContent = c.spec.label;
    var from = document.createElement('input');
    from.type = 'text'; from.setAttribute('aria-label', c.spec.label + ' from');
    var to = document.createElement('input');
    to.type = 'text'; to.setAttribute('aria-label', c.spec.label + ' to');
    var dual = document.createElement('div');
    dual.className = 'dual-range';
    var track = document.createElement('div'); track.className = 'track';
    var fill = document.createElement('div'); fill.className = 'fill';
    var sLo = document.createElement('input'); var sHi = document.createElement('input');
    [sLo, sHi].forEach(function(s, i) {
      s.type = 'range'; s.min = c.lo; s.max = c.hi; s.step = c.step;
      s.setAttribute('aria-label', c.spec.label + (i ? ' to' : ' from'));
    });
    dual.appendChild(track); dual.appendChild(fill); dual.appendChild(sLo); dual.appendChild(sHi);
    var dash = document.createElement('span'); dash.textContent = '–';
    ctl.appendChild(label); ctl.appendChild(from); ctl.appendChild(dual); ctl.appendChild(to);
    if (c.spec.unit) { var u = document.createElement('span'); u.textContent = c.spec.unit; ctl.appendChild(u); }
    bar.appendChild(ctl);
    widgets[key] = {from: from, to: to, sLo: sLo, sHi: sHi, fill: fill};

    function show() {
      from.value = c.fmt(c.cur[0]); to.value = c.fmt(c.cur[1]);
      sLo.value = c.cur[0]; sHi.value = c.cur[1];
      var span = c.hi - c.lo || 1;
      fill.style.left = ((c.cur[0] - c.lo) / span * 100) + '%';
      fill.style.width = ((c.cur[1] - c.cur[0]) / span * 100) + '%';
      from.classList.remove('invalid'); to.classList.remove('invalid');
      // keep the upper thumb reachable when both sit at the same end
      sHi.style.zIndex = c.cur[0] >= c.hi - c.step ? 3 : 2;
    }
    widgets[key].show = show;

    function setWindow(lo, hi, apply) {
      lo = Math.max(c.lo, Math.min(c.hi, lo));
      hi = Math.max(c.lo, Math.min(c.hi, hi));
      if (lo > hi) { var t = lo; lo = hi; hi = t; }
      c.cur = [lo, hi];
      show();
      if (apply !== false) applyRanges();
    }
    widgets[key].setWindow = setWindow;

    sLo.addEventListener('input', function() { setWindow(Math.min(+sLo.value, c.cur[1]), c.cur[1]); });
    sHi.addEventListener('input', function() { setWindow(c.cur[0], Math.max(+sHi.value, c.cur[0])); });
    function fromText(input, which) {
      var v = c.parse(input.value);
      if (isNaN(v)) { input.classList.add('invalid'); return; }
      if (which === 0) setWindow(v, c.cur[1]); else setWindow(c.cur[0], v);
    }
    from.addEventListener('change', function() { fromText(from, 0); });
    to.addEventListener('change', function() { fromText(to, 1); });
    [from, to].forEach(function(inp) {
      inp.addEventListener('keydown', function(e) { if (e.key === 'Enter') { e.preventDefault(); inp.blur(); } });
    });
    show();
  });

  var reset = document.createElement('button');
  reset.type = 'button'; reset.className = 'range-reset'; reset.textContent = 'Full range';
  reset.addEventListener('click', function() {
    Object.keys(coords).forEach(function(key) {
      widgets[key].setWindow(coords[key].lo, coords[key].hi, false);
    });
    applyRanges();
  });
  bar.appendChild(reset);
  div.parentNode.insertBefore(bar, div);

  // the plot's own zoom / pan / reset moves the controls
  function onRelayout(ev) {
    if (div._rangesBusy || !ev) return;
    var p0 = meta.panels[0];
    Object.keys(coords).forEach(function(key) {
      var c = coords[key];
      var name = p0[axisFor(key)];
      var lo, hi;
      if (ev[name + '.autorange'] || (ev[name] && ev[name].autorange)) {
        lo = c.lo; hi = c.hi;
      } else if (ev[name + '.range[0]'] !== undefined) {
        lo = ev[name + '.range[0]']; hi = ev[name + '.range[1]'];
      } else if (ev[name + '.range']) {
        lo = ev[name + '.range'][0]; hi = ev[name + '.range'][1];
      } else if (ev[name] && ev[name].range) {
        lo = ev[name].range[0]; hi = ev[name].range[1];
      } else return;
      lo = c.isDate ? _parseDay(lo) : +lo; hi = c.isDate ? _parseDay(hi) : +hi;
      if (isNaN(lo) || isNaN(hi)) return;
      widgets[key].setWindow(Math.min(lo, hi), Math.max(lo, hi), false);
    });
  }
  // Plotly's event emitter exists once the plot is drawn
  var tries = 0;
  (function hook() {
    if (div.on) div.on('plotly_relayout', onRelayout);
    else if (tries++ < 50) setTimeout(hook, 100);
  })();
}

function _applyFigData(div, figData) {
  var meta = figData.layout.meta || {};
  if (meta.toggles) _renderToggles(div, figData);
  if (meta.ranges) _renderRanges(div, figData);
  if (meta.responsive) {
    _applyGrid(div, figData, true);
    _watchGrid(div, figData);
    return;
  }
  // Only set autosize if the figure has no explicit width.
  if (!figData.layout.width) figData.layout.autosize = true;
  // the placeholder height came from the page; the figure knows its own
  if (figData.layout.height) div.style.height = figData.layout.height + 'px';
  _plotThemed(div, figData.data, figData.layout, _PLOT_CONFIG);
}

// Variable nav: "Show all N" / "Show fewer" past two rows
function toggleVarNav(btn) {
  var nav = btn.previousElementSibling;
  var open = nav.classList.toggle('expanded');
  btn.setAttribute('aria-expanded', open ? 'true' : 'false');
  var n = nav.querySelectorAll('.var-pill').length;
  btn.textContent = open ? 'Show fewer' : 'Show all ' + n;
  _syncVarNavs();
}
// A collapsed nav shows exactly two rows of pills (measured, so fonts and zoom
// do not matter); the button is only shown while rows are actually hidden.
var _VAR_NAV_ROWS = 2, _VAR_NAV_GAP = 4;
function _syncVarNavs() {
  document.querySelectorAll('.var-nav.many').forEach(function(nav) {
    var btn = nav.nextElementSibling;
    if (!btn || !btn.classList.contains('var-nav-more')) return;
    var pills = nav.querySelectorAll('.var-pill');
    if (!pills.length) return;
    var first = pills[0], last = pills[pills.length - 1];
    if (nav.classList.contains('expanded')) {
      nav.style.maxHeight = '';
      btn.style.display = '';
      return;
    }
    var padBottom = parseFloat(getComputedStyle(nav).paddingBottom) || 0;
    var rowH = first.offsetHeight;
    nav.style.maxHeight = (_VAR_NAV_ROWS * rowH + (_VAR_NAV_ROWS - 1) * _VAR_NAV_GAP + padBottom) + 'px';
    var hidden = last.getBoundingClientRect().bottom > nav.getBoundingClientRect().bottom - padBottom + 1;
    btn.style.display = hidden ? '' : 'none';
  });
}
window.addEventListener('resize', function() { clearTimeout(window._vnT); window._vnT = setTimeout(_syncVarNavs, 150); });

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
      // Navigation and filter clicks can ask for the same panel twice before
      // its figure has arrived; one fetch is enough.
      if (_inflight[graphId]) return;
      _inflight[graphId] = true;
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
        })
        .then(function() { delete _inflight[graphId]; });
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

  // render the first visible plot now, the sections below it as they scroll
  // into view
  var ap = panel && panel.querySelector('.plot-panel.active');
  if (ap) renderPanel(ap.id);
  if (panel) _watchPanels(panel);
  _syncVarNavs();
}

var _panelObserver = null;
function _watchPanels(panel) {
  var panels = panel.querySelectorAll('.plot-panel.active');
  if (typeof IntersectionObserver === 'undefined') {
    panels.forEach(function(p) { renderPanel(p.id); });
    return;
  }
  if (!_panelObserver) {
    _panelObserver = new IntersectionObserver(function(entries) {
      entries.forEach(function(e) {
        if (!e.isIntersecting) return;
        _panelObserver.unobserve(e.target);
        renderPanel(e.target.id);
      });
    }, {root: document.querySelector('.main-content'), rootMargin: '240px 0px'});
  }
  panels.forEach(function(p) { _panelObserver.observe(p); });
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
  _syncVarNavs();
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
  if (btn) {
    btn.classList.add('active'); btn.setAttribute('aria-pressed', 'true');
    // a collapsed nav must not hide the active button
    var nav = btn.closest('.var-nav');
    if (nav && nav.classList.contains('many') && !nav.classList.contains('expanded') &&
        btn.offsetTop + btn.offsetHeight > nav.clientHeight) {
      toggleVarNav(nav.nextElementSibling);
    }
  }
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
