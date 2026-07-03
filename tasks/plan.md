# Implementation plan — report visualization

Two related features for the `climateeval.report` module:

- **Feature A — Report visual hierarchy** ([#17](https://github.com/climate-federation/ClimateEval/issues/17)):
  visual polish of the interactive HTML report — separate tiers and plot types,
  reduce crowding. Follows dashboard/data-viz best practices + WCAG AA. Keeps
  the current IA, components, and Plotly figures.
- **Feature B — Static plot export** (issue drafted, not yet published):
  render ESMValTool/ICONEval-style static figures (PNG/PDF) from the DuckDB
  output, mirroring the interactive Plotly path.

> **Spec** = the two GitHub issues + `docs/reporting.md` + `AGENTS.md`
> (`docs/output-schema.md` for the table layout). No `SPEC.md` exists.

## Current architecture (grounded)

**Report module** `src/climateeval/report/`: `_db.py` (read), `_figures.py`
(Plotly builders + registry), `_html.py` (render), `_style.py` (`_CSS` + `_JS`
strings + colormap helpers), `_leaderboard.py`, `_server.py`.

- **Data flow**: `read_database(db_path) -> {diag_name: {table: DataFrame}}`;
  `detect_diag_type(columns) -> type`; `get_var_cols(data_df) -> [var_id]`.
  Iteration lives in `_server.py` (`_load_state`, `_build_diag_figures`,
  `_build_structure`) and `_html.py` (`_build_figures_for_tier`).
- **Figure registry** `_figures.py::_FIGURE_BUILDERS` — 10 types, each
  `build_X_figures(data_df, variables_df, metrics_df=None) -> dict[var_id, go.Figure]`.
  Labels/icons: `_DIAG_LABELS`, `_DIAG_ICONS`.
- **Two render paths** (⚠️ key gotcha): `build_html_shell()` (server, lazy —
  renders diagnostics inline) and `generate_portable_html()` (embeds all figures,
  via `_render_tier_panel()` + `render_diagnostic_section()`). Both share `_CSS`
  and `_JS` from `_style.py`. **Tier/diagnostic/category DOM changes must land in
  BOTH paths** unless we unify them first.
- **Tiers → DOM**: one `.content-panel#panel-tierN` + one `.sidebar-item` per
  tier; label from `_stem_to_label(db_path.stem)`; a single `--accent` today.
- **Diagnostics → DOM**: `.diag-section` per diagnostic with
  `.diag-section-header` (icon+title). Ordered by DB row order.
- **Variables → DOM**: `.var-pill` nav + one `.plot-panel` each; JS
  `showVarPanel()` toggles. Interactivity JS: `filterDiag`, `showVarPanel`,
  `sidebarNav`, `searchPlots`, `sortTable`, `renderPanel` (all in `_JS`).
- **Colormaps** (`_style.py`): `_colorscale_for`, `_bias_colorscale_for`,
  `_hovmoller_colorscale_for`, `_DIVERGING_VARS` — names (`RdBu_r`, `Plasma`,
  `Greens`, `BrBG`, `PuOr`) are **valid matplotlib cmaps** → reusable for static.
- **Deps for static**: `matplotlib-base`, `cartopy`, `scitools-iris`,
  `mapgenerator`, `iris-esmf-regrid` all present (esmvaltool feature).

## Dependency graph

```
Feature A (HTML/CSS)                    Feature B (matplotlib module)
  A0 unify/parity of render paths         B0 scaffold + Map (cartopy) end-to-end
      │                                        │
  A1 tier identity (color+badge)          B1 2-D heatmaps (ZonalProfile/Hovmoller/QBO)
      │                                        │
  A2 plot-type category grouping          B2 1-D lines (TS/AnnualCycle/Diurnal/Zonal/Histogram)
      │                                        │
  A3 spacing/dividers/card polish         B3 Scalar/bar + ref/bias/metrics
      │                                        │
  A4 variable-nav polish                  B4 ESMValTool/ICONEval style match + PDF/DPI
      │                                        │
  A5 leaderboard hierarchy                B5 CLI + API integration + docs + tests
      │
  A6 accessibility pass (AA/focus/keyboard)
```

A and B are **independent** (A = HTML/CSS; B = new module) and may proceed in
parallel or sequence. Recommended order: **A first** (published, lower risk,
smaller), then **B**. `_style.py` colormap helpers are the only shared surface;
B reads the same registry/DB as A but does not modify A's code.

## Slicing principle

Each task is a **vertical slice** — a complete, verifiable increment (a
regenerated HTML showing the new visual, or a generated PNG), not a horizontal
layer. Every task's verification **regenerates a report from a real `.ddb`**
(root-level `*.ddb`, or `showcase_output/*.ddb` once the showcase job lands) and
checks the result in **both** server and portable modes for Feature A.

---

## Phase A — Report visual hierarchy (#17)

### A0 — Unify the two render paths (enabling)
Refactor so tier-panel + diagnostic + (future) category rendering is produced by
**one** shared helper used by both `build_html_shell` and
`generate_portable_html`. Server keeps its lazy empty-plot stubs; only the
structural HTML is shared.
- **Acceptance**: server and portable emit identical tier/diagnostic/section DOM
  (diff the structural markup); all existing behavior unchanged.
- **Verify**: generate portable HTML + run `serve` on a sample `.ddb`; diff the
  DOM skeletons; existing JS (filter/search/var-switch) still works.

### A1 — Tier identity: per-tier accent color + badge
Categorical, colorblind-safe, AA-contrast palette (one accent per tier);
`--tierN-accent` tokens set in Python. Apply to `.sidebar-item` (badge "T1/T2/T3"
+ left accent), `.content-panel` (accent header banner/left border), toolbar
strip. **Color never the sole cue** — always paired with the badge/label.
- **Acceptance**: each tier is visually distinct at a glance; badge+label present;
  AA contrast on badge/text.
- **Verify**: regenerate report; confirm distinct tier color+badge in sidebar and
  panel, both modes; contrast check the badge/label.

### A2 — Plot-type category grouping
Group `.diag-section`s within a tier into plot-type **categories** (Spatial =
Map/ZonalProfile; Temporal = TimeSeries/AnnualCycle/DiurnalCycle/Hovmöller/QBO;
Distributions = Histogram; Scalars) with a `.diag-category-header` divider when
the category changes (Gestalt proximity). Define category → types map next to
`_DIAG_LABELS`.
- **Acceptance**: related plots cluster under labeled category headers; filter/
  search still work across categories.
- **Verify**: regenerate; confirm category headers + grouping order; `filterDiag`
  and `searchPlots` still hide/show correctly.

**⏸ Checkpoint 1** — the headline "separation" is in. Review screenshots (both
modes) before continuing.

### A3 — Spacing, dividers, card polish
Consistent 8 px spacing scale; larger inter-section gaps; clearer dividers;
`.plot-panel-card`/`.plot-title-bar` carries plot-type icon + subtle category
color chip.
- **Acceptance**: noticeably less crowded; consistent rhythm; cards read as units.
- **Verify**: regenerate; visual density check; no overflow/regression.

### A4 — Variable-nav polish
Make `.var-nav` scannable with many variables (~19): sort/group, sticky within
section, wrap-to-scroll or compact selector past N, clearer active state.
- **Acceptance**: variable nav stays tidy and usable at 19+ variables.
- **Verify**: regenerate `Tier2_atmosphere_monthly`; confirm nav is scannable;
  `showVarPanel` still switches plots.

### A5 — Leaderboard hierarchy
Emphasize rank/score, de-emphasize secondary columns, add scoreboard-heatmap
legend/scale, optionally group the detail table by tier.
- **Acceptance**: leaderboard reads clearly; scoreboard has a legend.
- **Verify**: regenerate a multi-model report; confirm hierarchy + `sortTable`.

### A6 — Accessibility pass
AA contrast (text, badges, pills), visible focus states, keyboard operability for
pills/tabs/sidebar, color never the sole signal, `aria` labels on icon-only
controls.
- **Acceptance**: WCAG 2.1 AA contrast verified; full keyboard nav; no color-only
  encoding.
- **Verify**: contrast-check key pairs; tab through the UI; test with a
  colorblind simulation.

**⏸ Checkpoint 2** — Feature A done. Attach before/after screenshots to #17.

---

## Phase B — Static plot export

### B0 — Scaffold + Map renderer end-to-end (enabling vertical slice)
New `src/climateeval/report/_static_figures.py`: a `_STATIC_BUILDERS` registry
mirroring `_FIGURE_BUILDERS`, and `generate_static_plots(db_paths, out_dir, *,
formats=("png",), dpi=150) -> list[Path]` reusing `read_database`,
`detect_diag_type`, `get_var_cols`, and `_style.py` colormaps. Implement **Map**
first (matplotlib + cartopy coastlines, reference + bias small-multiples).
- **Acceptance**: `generate_static_plots(["X.ddb"], "out/")` writes a Map PNG per
  atmosphere variable to `out/`.
- **Verify**: run on a real `.ddb`; assert PNG files exist, are non-empty, and
  eyeball one Map (coastlines, colorbar, ref+bias).

### B1 — 2-D heatmap renderers
ZonalProfile (log-pressure y, lat x), Hovmöller (time × lon), QBO (pressure ×
time) — reuse the heatmap/colorbar/log-axis + pivot patterns.
- **Acceptance**: PNGs for these three types render correctly.
- **Verify**: run on dynamics/atmosphere `.ddb`; eyeball each.

### B2 — 1-D line renderers
TimeSeries, AnnualCycle, DiurnalCycle, ZonalLine, Histogram — reference/model/
other overlays + bias where applicable, matching the Plotly builders' semantics.
- **Acceptance**: PNGs for the five 1-D types render with correct axes/overlays.
- **Verify**: run; eyeball; confirm reference/model/other lines + bias.

### B3 — Scalar/bar renderer + annotations
Bar chart for 0-D scalars; titles/annotations carrying model·reference·metrics.
- **Acceptance**: Scalar PNGs render; all 10 types now covered.
- **Verify**: run on `Tier3_ecs.ddb`; eyeball.

### B4 — ESMValTool/ICONEval style match
Fonts, figure sizes, colorbars, coastlines, small-multiples, symmetric ranges
(`_DIVERGING_VARS`), ICONEval colormaps; add **PDF** output + configurable DPI.
**Self-contained matplotlib** (decided — more control, less coupling); reuse only
ClimateEval's own `read_database` / `detect_diag_type` / `_style` colormaps.
- **Acceptance**: static figures visually match the ESMValTool/ICONEval look;
  PDF + PNG.
- **Verify**: side-by-side against an ESMValTool/ICONEval reference figure.

### B5 — CLI + API integration, docs, tests
Wire `report.py` (`--static-out DIR --format png,pdf`); export
`generate_static_plots` from `climateeval.report`; document in
`docs/reporting.md`; add a smoke test (`tests/`) that generates from a sample
`.ddb` and asserts files exist + non-empty.
- **Acceptance**: `python report.py X.ddb --static-out out/` works offline; API
  documented; test passes; `pre-commit` green.
- **Verify**: `pixi run pytest` (new test) + `pixi run pre-commit run --all-files`.

**⏸ Checkpoint 3** — Feature B done. Publish/close the static-plots issue.

---

## Cross-cutting verification
- Every Feature A task regenerates a report in **both** server (`serve`) and
  portable (`generate_portable_html`) modes and checks parity.
- Report module is excluded from most ruff rules but still linted +
  type-checked; run `pixi run pre-commit run --all-files` before each commit.
- Reuse existing sample `.ddb` files (repo root) or `showcase_output/*.ddb`.

## Risks
- **Duplicated render paths** (server vs portable) — mitigated by **A0** first.
- **Report code is loosely typed/linted** (broad `report/*.py` per-file ruff
  ignores) — keep new static code cleaner; it isn't under that ignore.
- **Static-plot styling parity** with ESMValTool/ICONEval is the fuzzy part —
  self-contained matplotlib (decided); verified via a side-by-side check at B4.
- **Portable HTML size** — visual changes are CSS/DOM only; keep JS/figure
  payload unchanged.
</content>
