"""CSS, JS, colour constants and colorscale helpers for the benchmarking report."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------------

_PLOT_FONT = {
    "family": "'Inter', 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif",
    "size": 12,
}
_AXIS_STYLE = {
    "showgrid": True,
    "gridcolor": "rgba(180,180,180,0.25)",
    "zeroline": False,
    "linecolor": "#ccc",
    "linewidth": 1,
    "mirror": True,
}

# Seaborn color palette for to_benchmark data
_SEABORN_COLORS = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#CF4458",
    "#4B4B83",
    "#D65F5F",
    "#6B6B6B",
]

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


def _colorscale_for(var_id: str) -> str:
    return _VAR_COLORSCALE.get(var_id, _DEFAULT_COLORSCALE)


def _bias_colorscale_for(var_id: str) -> str:
    return _VAR_BIAS_COLORSCALE.get(var_id, _DEFAULT_BIAS_COLORSCALE)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
*, *::before, *::after { box-sizing: border-box; }

:root {
  --navy: #0d1f35;
  --navy-mid: #163657;
  --navy-light: #1a5276;
  --accent: #2e86c1;
  --accent-hover: #1a6fa8;
  --sidebar-w: 230px;
  --header-h: 72px;
  --toolbar-h: 52px;
  --bg: #f0f3f7;
  --surface: #ffffff;
  --border: #dde2eb;
  --text: #1e2535;
  --muted: #6a7890;
  --font: 'Inter', 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
}

html, body {
  height: 100%; margin: 0; padding: 0;
  font-family: var(--font); font-size: 14px; color: var(--text);
  background: var(--bg); overflow: hidden;
}

/* ── App shell ── */
.app-shell {
  display: flex; flex-direction: column; height: 100vh; overflow: hidden;
}

/* ── Header ── */
.app-header {
  flex-shrink: 0; height: var(--header-h);
  background: linear-gradient(110deg, var(--navy) 0%, var(--navy-mid) 50%, var(--navy-light) 100%);
  color: #c8daf0; padding: 0 28px;
  display: flex; align-items: center; gap: 20px;
  border-bottom: 1px solid rgba(255,255,255,0.07);
  box-shadow: 0 2px 12px rgba(0,0,0,.35);
  z-index: 300;
}

.header-logo {
  flex-shrink: 0; width: 42px; height: 42px;
  background: rgba(255,255,255,0.10); border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
}

.header-title { flex: 1; min-width: 0; }
.header-title h1 {
  margin: 0 0 2px; font-size: 1.15rem; font-weight: 700;
  color: #fff; letter-spacing: -.1px; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
}
.header-meta {
  display: flex; gap: 20px; flex-wrap: wrap;
  font-size: 0.76rem; opacity: 0.65; line-height: 1.2;
}
.header-meta span::before { content: ""; }
.header-chip {
  display: flex; align-items: center; gap: 5px;
  background: rgba(255,255,255,0.08); border-radius: 20px;
  padding: 3px 10px; font-size: 0.74rem; color: #b8cde4;
  white-space: nowrap;
}
.header-chip strong { color: #e0edf8; font-weight: 600; }

/* ── Body split ── */
.app-body {
  display: flex; flex: 1; overflow: hidden; position: relative;
}

/* ── Sidebar ── */
.sidebar {
  flex-shrink: 0; width: var(--sidebar-w);
  background: var(--navy); color: #a8bdd4;
  display: flex; flex-direction: column;
  overflow-y: auto; overflow-x: hidden;
  border-right: 1px solid rgba(255,255,255,0.06);
  transition: width .2s ease;
}
.sidebar.collapsed { width: 0; border-right: none; }

.sidebar-toggle {
  position: absolute; top: 12px;
  left: calc(var(--sidebar-w) - 14px);
  width: 28px; height: 28px; border-radius: 50%;
  background: var(--accent); color: #fff;
  border: 2px solid var(--bg);
  cursor: pointer;
  box-shadow: 0 2px 6px rgba(0,0,0,.25);
  z-index: 150; user-select: none;
  transition: left .2s ease, background .15s;
  padding: 0;
}
.sidebar-toggle:hover { background: var(--accent-hover); }
.app-body.sidebar-collapsed .sidebar-toggle { left: 8px; }

.sidebar-toggle .bar {
  position: absolute;
  left: 6px;
  width: 12px; height: 2px;
  background: #fff;
  border-radius: 1px;
  transition: top .22s ease, transform .22s ease, opacity .15s ease;
}
/* Open state: X */
.sidebar-toggle .bar-top { top: 11px; transform: rotate(45deg); }
.sidebar-toggle .bar-mid { top: 11px; opacity: 0; }
.sidebar-toggle .bar-bot { top: 11px; transform: rotate(-45deg); }
/* Collapsed state: hamburger (three horizontal bars) */
.app-body.sidebar-collapsed .sidebar-toggle .bar-top { top: 7px;  transform: none; }
.app-body.sidebar-collapsed .sidebar-toggle .bar-mid { top: 11px; opacity: 1; }
.app-body.sidebar-collapsed .sidebar-toggle .bar-bot { top: 15px; transform: none; }

.sidebar-section-label {
  font-size: 0.68rem; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; color: rgba(255,255,255,0.25);
  padding: 18px 18px 6px;
}

.sidebar-item {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 18px; cursor: pointer;
  font-size: 0.875rem; font-weight: 500; color: #94adc8;
  border-left: 3px solid transparent;
  border-radius: 0 8px 8px 0; margin: 1px 8px 1px 0;
  transition: background .12s, color .12s, border-color .12s;
  user-select: none;
}
.sidebar-item:hover { background: rgba(255,255,255,0.06); color: #cce0f5; }
.sidebar-item.active {
  background: rgba(46,134,193,0.18); color: #fff;
  border-left-color: var(--accent); font-weight: 600;
}
.sidebar-item .sidebar-icon { font-size: 1rem; flex-shrink: 0; opacity: 0.8; }

.sidebar-divider {
  height: 1px; background: rgba(255,255,255,0.07); margin: 8px 16px;
}

/* ── Content area ── */
.content-wrap {
  flex: 1; display: flex; flex-direction: column; overflow: hidden;
}

/* ── Toolbar (diagnostic filter + search) ── */
.content-toolbar {
  flex-shrink: 0; height: var(--toolbar-h);
  background: var(--surface); border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 12px;
  padding: 0 24px;
  box-shadow: 0 1px 4px rgba(0,0,0,.06);
  z-index: 100;
}

.diag-filter-btns {
  display: flex; gap: 6px; align-items: center; flex-shrink: 0;
}
.diag-filter-btn {
  background: none; border: 1.5px solid var(--border);
  border-radius: 20px; padding: 5px 14px;
  font-size: 0.8rem; font-weight: 500; cursor: pointer;
  color: var(--muted); display: flex; align-items: center; gap: 5px;
  transition: background .12s, border-color .12s, color .12s;
  white-space: nowrap;
}
.diag-filter-btn:hover { background: #eef3fa; border-color: var(--accent); color: var(--navy-mid); }
.diag-filter-btn.active {
  background: var(--accent); border-color: var(--accent);
  color: #fff; font-weight: 600;
}

.toolbar-sep {
  width: 1px; height: 24px; background: var(--border); flex-shrink: 0;
}

.search-wrap {
  flex: 1; position: relative; max-width: 320px;
}
.search-wrap svg {
  position: absolute; left: 10px; top: 50%; transform: translateY(-50%);
  width: 15px; height: 15px; color: var(--muted); pointer-events: none;
}
.search-input {
  width: 100%; padding: 7px 12px 7px 32px;
  border: 1.5px solid var(--border); border-radius: 20px;
  font-size: 0.82rem; font-family: var(--font); color: var(--text);
  background: var(--bg); outline: none;
  transition: border-color .15s, background .15s;
}
.search-input:focus { border-color: var(--accent); background: #fff; }
.search-input::placeholder { color: var(--muted); }

/* ── Main scrolling content ── */
.main-content {
  flex: 1; overflow-y: auto; padding: 28px 32px 48px;
}

/* ── Content panels ── */
.content-panel { display: none; }
.content-panel.active { display: block; }

/* ── Diagnostic sections within a tier ── */
.diag-section { margin-bottom: 48px; }
.diag-section.hidden { display: none; }

.diag-section-header {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 18px; padding-bottom: 12px;
  border-bottom: 2px solid var(--border);
}
.diag-section-icon { font-size: 1.2rem; }
.diag-section-title {
  font-size: 1.05rem; font-weight: 700; color: var(--navy-mid);
}

/* ── Section headings ── */
.section-heading {
  font-size: 1.1rem; font-weight: 700; color: var(--navy-mid);
  margin: 0 0 22px; padding-bottom: 12px;
  border-bottom: 2px solid var(--border);
}

/* ── Leaderboard ── */
.lb-section { max-width: 1100px; }
.scoreboard-wrap { margin-bottom: 24px; background: #fff; border-radius: 10px;
  box-shadow: 0 2px 14px rgba(0,0,0,.09); overflow: hidden; }
.scoreboard-plot { width: 100%; min-height: 320px; }

.metric-note {
  font-size: 0.82rem; color: #6a7790;
  background: #f0f4fa; border-left: 3px solid var(--accent);
  padding: 9px 14px; border-radius: 0 6px 6px 0; margin-bottom: 20px;
}

.lb-table-wrap {
  overflow-x: auto; border-radius: 10px;
  box-shadow: 0 2px 14px rgba(0,0,0,.09); margin-bottom: 16px;
}
table.lb {
  border-collapse: collapse; width: 100%; min-width: 440px;
  background: #fff; font-size: 0.87rem;
}
table.lb thead th {
  background: var(--navy-mid); color: #d8e4f0; font-weight: 600;
  padding: 11px 18px; text-align: left; white-space: nowrap;
  letter-spacing: .03em; font-size: 0.82rem; text-transform: uppercase;
}
table.lb thead th.sortable { cursor: pointer; user-select: none; }
table.lb thead th.sortable::after { content: " ↕"; opacity: 0.4; font-size: 0.78em; }
table.lb thead th.asc::after   { content: " ↑"; opacity: 1; }
table.lb thead th.desc::after  { content: " ↓"; opacity: 1; }
table.lb tbody tr { border-bottom: 1px solid #edf0f6; }
table.lb tbody tr:last-child { border-bottom: none; }
table.lb tbody tr:nth-child(even) { background: #f7f9fc; }
table.lb tbody tr:hover { background: #ecf2fb; }
table.lb tbody td { padding: 10px 18px; white-space: nowrap; }
td.rank { font-weight: 800; font-size: 1.05rem; text-align: center; width: 50px; }
td.score-cell { text-align: right; font-variant-numeric: tabular-nums; }
.medal-1 { color: #c0860a; }
.medal-2 { color: #7a8390; }
.medal-3 { color: #8b5e3c; }
.bar-cell { position: relative; min-width: 140px; }
.bar-bg {
  position: absolute; top: 20%; left: 8px;
  height: 60%; border-radius: 4px; opacity: 0.18; pointer-events: none;
}
.bar-val { position: relative; z-index: 1; font-weight: 700; font-size: 0.92rem; }
.detail-section { margin-top: 20px; }
details > summary {
  cursor: pointer; font-weight: 600; color: var(--navy-light);
  font-size: 0.87rem; padding: 8px 0; user-select: none;
  list-style: none; display: flex; align-items: center; gap: 6px;
}
details > summary::before { content: "▶"; font-size: 0.7em; transition: transform .15s; }
details[open] > summary::before { transform: rotate(90deg); }
details > summary:hover { opacity: 0.75; }

/* ── Variable pill nav ── */
.var-nav {
  display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px;
}
.var-pill {
  background: #fff; border: 1.5px solid var(--border); border-radius: 24px;
  padding: 6px 18px; font-size: 0.80rem; cursor: pointer;
  color: #3a4a66; font-weight: 500; font-family: var(--font);
  box-shadow: 0 1px 3px rgba(0,0,0,.06);
  transition: background .13s, border-color .13s, color .13s, box-shadow .13s;
  max-width: 280px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.var-pill:hover { background: #edf2fb; border-color: var(--accent); color: var(--navy-mid); }
.var-pill.active {
  background: var(--navy-mid); color: #fff; border-color: var(--navy-mid);
  box-shadow: 0 3px 8px rgba(22,54,87,.28);
}

/* ── Plot panel card ── */
.plot-panel { display: none; }
.plot-panel.active { display: block; }

.plot-panel-card {
  background: var(--surface); border-radius: 12px;
  border: 1px solid var(--border);
  box-shadow: 0 2px 12px rgba(0,0,0,.07);
  overflow: hidden; margin-bottom: 8px;
}

.plot-title-bar {
  padding: 10px 18px; background: #f8fafc;
  border-bottom: 1px solid var(--border);
  display: flex; align-items: center; justify-content: space-between;
}
.plot-title-text {
  font-size: 0.88rem; font-weight: 600; color: var(--navy-mid);
}

/* ── Dataset detail box ── */
.plot-detail-footer {
  border-top: 1px solid var(--border); padding: 10px 18px 14px;
  background: #f8fafc;
}

.detail-toggle-btn {
  background: none; border: 1.5px solid var(--border);
  border-radius: 6px; padding: 5px 14px;
  font-size: 0.78rem; font-weight: 500; cursor: pointer;
  color: var(--muted); font-family: var(--font);
  display: inline-flex; align-items: center; gap: 6px;
  transition: background .12s, border-color .12s, color .12s;
}
.detail-toggle-btn:hover { background: #eef3fa; border-color: var(--accent); color: var(--navy-mid); }
.detail-icon { font-style: normal; font-size: 1em; }

.plot-detail-box {
  display: none; margin-top: 12px;
}
.plot-detail-box.open { display: block; }

.plot-detail-body {
  display: flex; flex-wrap: wrap; gap: 16px;
}

.detail-block {
  flex: 1; min-width: 260px;
  background: #fff; border: 1px solid var(--border);
  border-radius: 8px; padding: 14px; font-size: 0.82rem;
}

.detail-block-title {
  font-weight: 700; color: var(--navy-mid); font-size: 0.80rem;
  text-transform: uppercase; letter-spacing: .05em;
  margin-bottom: 10px; padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}

.detail-meta-row {
  display: flex; gap: 8px; padding: 3px 0;
  border-bottom: 1px solid #f0f2f5;
}
.detail-meta-row:last-child { border-bottom: none; }
.detail-meta-key {
  flex-shrink: 0; width: 130px; font-weight: 600; color: var(--muted);
  font-size: 0.78rem;
}
.detail-meta-val { color: var(--text); word-break: break-word; }

.detail-ds-item {
  margin-bottom: 10px; padding-bottom: 10px;
  border-bottom: 1px solid #f0f2f5;
}
.detail-ds-item:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.detail-ds-header {
  display: flex; align-items: center; gap: 8px; margin-bottom: 4px;
}
.detail-badge {
  font-size: 0.7rem; font-weight: 700; padding: 2px 8px;
  border-radius: 10px; text-transform: uppercase; letter-spacing: .04em;
}
.detail-badge-ref   { background: #e3f2fd; color: #1565c0; }
.detail-badge-model { background: #e8f5e9; color: #2e7d32; }
.detail-badge-other { background: #fff3e0; color: #e65100; }
.detail-ds-body {
  font-size: 0.79rem; color: var(--muted); line-height: 1.6;
}
.detail-ds-body a { color: var(--accent); text-decoration: none; }
.detail-ds-body a:hover { text-decoration: underline; }

/* ── Legend ── */
.legend-note {
  font-size: 0.79rem; color: #7a8aa0; margin: 0 0 14px;
  display: flex; gap: 18px; flex-wrap: wrap; align-items: center;
}
.legend-note span { display: flex; align-items: center; gap: 7px; }
.ln-solid {
  display: inline-block; width: 26px; height: 2.5px;
  background: #555; border-radius: 2px;
}
.ln-dash {
  display: inline-block; width: 26px; height: 2.5px; border-radius: 2px;
  background: repeating-linear-gradient(to right,#555 0 5px,transparent 5px 9px);
}
.ln-dot {
  display: inline-block; width: 26px; height: 2.5px; border-radius: 2px;
  background: repeating-linear-gradient(to right,#555 0 2px,transparent 2px 6px);
}
.ln-ref-swatch {
  display: inline-block; width: 14px; height: 14px; border-radius: 3px;
  background: rgba(192,134,10,0.15); border: 1.5px solid #c0860a; flex-shrink: 0;
}
.ln-bias-swatch {
  display: inline-block; width: 26px; height: 10px; border-radius: 3px; flex-shrink: 0;
  background: linear-gradient(to right, #2166ac, #f7f7f7, #d6604d);
}

/* ── Search no-results ── */
.search-no-results {
  display: none; padding: 40px 0; text-align: center;
  color: var(--muted); font-style: italic;
}
.search-no-results.visible { display: block; }

/* ── Misc ── */
.db-source { font-size: 0.76rem; color: #aab; margin: 0 0 14px; font-style: italic; }
.empty-note { color: #888; font-style: italic; padding: 20px 0; }
.diag-content { padding: 0; }

/* ── Server-mode loading / error states ── */
.plot-loading {
  display: flex; align-items: center; justify-content: center; gap: 12px;
  height: 200px; color: var(--muted); font-size: 0.95rem;
}
.plot-loading-spinner {
  display: inline-block; width: 22px; height: 22px; border-radius: 50%;
  border: 3px solid #e0e6f0; border-top-color: var(--accent);
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.plot-error {
  display: flex; align-items: center; justify-content: center;
  height: 120px; color: #c44e52; font-size: 0.88rem;
  background: #fff5f5; border: 1px solid #f5c6c6; border-radius: 8px; margin: 16px;
}

/* ── Scrollbar styling ── */
.sidebar::-webkit-scrollbar { width: 5px; }
.sidebar::-webkit-scrollbar-track { background: transparent; }
.sidebar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }
.main-content::-webkit-scrollbar { width: 7px; }
.main-content::-webkit-scrollbar-track { background: var(--bg); }
.main-content::-webkit-scrollbar-thumb { background: #c8d0de; border-radius: 4px; }
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
  document.querySelectorAll('.sidebar-item').forEach(function(b){ b.classList.remove('active'); });
  document.querySelectorAll('.content-toolbar').forEach(function(t){ t.style.display='none'; });

  var panel = document.getElementById('panel-' + sectionId);
  if (panel) panel.classList.add('active');

  var btn = document.querySelector('.sidebar-item[data-section="' + sectionId + '"]');
  if (btn) btn.classList.add('active');

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

  var toolbar = document.getElementById('toolbar-' + tierId);
  if (toolbar) {
    toolbar.querySelectorAll('.diag-filter-btn').forEach(function(b){ b.classList.remove('active'); });
    var activeBtn = toolbar.querySelector('.diag-filter-btn[data-diag="' + diagName + '"]');
    if (activeBtn) activeBtn.classList.add('active');
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
  diag.querySelectorAll('.var-pill').forEach(function(b){ b.classList.remove('active'); });
  var panelId = 'plot-' + scopeId + '-' + varId;
  var panel = document.getElementById(panelId);
  if (panel) panel.classList.add('active');
  var btn = diag.querySelector('.var-pill[data-var="' + varId + '"]');
  if (btn) btn.classList.add('active');
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
function toggleDetail(detailId) {
  var box = document.getElementById(detailId);
  if (!box) return;
  box.classList.toggle('open');
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
