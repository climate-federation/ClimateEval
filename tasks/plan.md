# Implementation Plan: Open GitHub Issues (#18, #17)

Status of the issue tracker (2026-07-04): two open issues, no open PRs, clean `main`.

- **#18 — Static plot export**: new matplotlib/cartopy backend that renders
  ESMValTool/ICONEval-style PNG/PDF figures from the same `.ddb` diagnostic
  output. Untouched so far. **Primary workstream.**
- **#17 — Report HTML visual hierarchy**: partially done. PR #19 (merged)
  delivered tier accent colors + T1/T2/T3 badges, plot-type category grouping,
  variable long names, and linked zoom. **Remaining**: spacing scale, var-nav
  taming, toolbar grouping, leaderboard hierarchy, card polish, accessibility.
  **Secondary workstream** (independent files from #18 except `docs/reporting.md`).

Both workstreams branch off `main`, one PR per phase (matching the repo's
slice-into-main workflow). They touch disjoint code (#18: new
`_static_figures.py` + CLI; #17: `_style.py`/`_html.py`/`_leaderboard.py`), so
they can be developed in parallel; only `docs/reporting.md` overlaps — do docs
last in each stream.

---

## Workstream A — Issue #18: Static figure export

### Architecture decisions

1. **New module `src/climateeval/report/_static_figures.py`** mirroring the
   existing `_FIGURE_BUILDERS` pattern (`_figures.py:2290`): a
   `_STATIC_FIGURE_BUILDERS: dict[str, Callable]` keyed by the same 10
   diagnostic-type strings from `detect_diag_type` (`_db.py:62`). Builders keep
   the exact interactive-builder signature —
   `build(data_df, variables_df, metrics_df=None) -> dict[var_id, matplotlib.figure.Figure]`
   — so DB reading, type detection, and iteration logic are shared unchanged.
2. **Reuse `_db.py` + `_style.py` wholesale.** `read_database`, `get_var_cols`,
   `get_density_cols`, `detect_diag_type` need no changes. The colorscale names
   in `_style.py` (`Plasma`, `Greens`, `Blues`, `RdBu_r`, `PuOr`, `BrBG`) are
   valid matplotlib cmap names, so `_colorscale_for` / `_bias_colorscale_for` /
   `_hovmoller_colorscale_for` and `_DIVERGING_VARS` are reused directly.
   Shared numeric helpers (`_data_range`, `_get_ref_datasets`, `_pivot_for`,
   `_var_label`) are currently private to `_figures.py` — extract the ones both
   backends need into a small `report/_common.py` (or import from `_figures`;
   decide in Task A1, prefer extraction to avoid importing plotly-heavy code
   paths for a headless export).
3. **Self-contained matplotlib implementation** (per the issue): no
   iris.quickplot / ESMValTool mapgenerator. Cartopy only for Map coastlines
   (`ccrs.Robinson()` or `PlateCarree` — match the interactive report's look;
   decide in Task A3 by eyeballing ICONEval references).
4. **Entry points**:
   - API: `climateeval.report.generate_static_plots(db_paths, out_dir, *, formats=("png",), dpi=150, model_name="", ...)`,
     modeled on `generate_portable_html` (`_html.py:539`) — reads DBs, loops
     (tier → diagnostic → variable), writes
     `out_dir/<tier>/<diagnostic>/<var_id>.<fmt>`.
   - CLI: `report.py --static-out DIR [--static-format png,pdf] [--dpi N]`;
     like `--save-portable-html-output`, it renders and exits without starting
     the server (mirror `_server.py:370-381` early-return path).
5. **Dependencies**: `matplotlib-base` and `cartopy` exist only in the
   `esmvaltool-dev` pixi feature (pyproject.toml:84,106), not in core deps.
   Add `matplotlib-base` + `cartopy` to `[tool.pixi.dependencies]` and
   `matplotlib` + `cartopy` to `[project].dependencies`. Rationale: the issue
   says "prefer already-available deps", and the package already carries the
   ESMValTool stack, so this adds no real weight. Import cartopy lazily inside
   the Map builder so non-map exports work even if cartopy is missing.
6. **Figure aesthetics**: reference + bias small-multiples (same layout logic
   as the interactive builders), colorbars per panel group, title carrying
   variable long name + units (from `variables_df`), subtitle/annotation with
   model vs reference names and metrics (RMSE/corr from `metrics_df`).
   Match ESMValTool/ICONEval look: serif-free font, coastlines, tight layout.

### Testing strategy

Follow `tests/unit/report/test__figures.py`: synthetic in-memory long-format
DataFrames (no real DuckDB), call builders directly, assert on the returned
`matplotlib.figure.Figure` (number of axes, colormap names, titles, presence of
colorbars) — plus one smoke test per entry point that writes an actual PNG to
`tmp_path` and asserts the expected files exist. Use `matplotlib.use("Agg")`.

### Tasks

#### Task A1: Skeleton, shared helpers, entry points, first figure type (TimeSeries)
**Description:** Create `_static_figures.py` with the registry, a
`_save_figure()` helper (formats/dpi/filenames), extract shared helpers from
`_figures.py` into `_common.py` (`_data_range`, `_get_ref_datasets`,
`_pivot_for`, `_var_label`) with `_figures.py` re-importing them (no behavior
change), add deps to `pyproject.toml`, implement
`generate_static_plots()` + the `report.py` CLI flags, and implement
**TimeSeries** (line plot, model vs reference, seaborn palette from
`_SEABORN_COLORS`) as the first end-to-end slice.
**Acceptance criteria:**
- [ ] `pixi run python report.py out.ddb --static-out /tmp/figs` writes one PNG per (tier, TimeSeries diagnostic, variable); other types log "not yet supported" and are skipped
- [ ] `--static-format png,pdf --dpi 300` produces both formats at the requested DPI
- [ ] Existing interactive report still works (unit suite green)
**Verification:** `pixi run pytest tests/unit/report/` (new `test__static_figures.py` with TimeSeries builder test + CLI/API smoke test); `pixi run pre-commit run --all-files`
**Dependencies:** None · **Scope:** M (5 files: `_static_figures.py`, `_common.py`, `_figures.py`, `report.py`, `pyproject.toml` + test)

#### Task A2: Map builder (cartopy)
**Description:** `build_static_map_figures`: reference panel + per-model bias
small-multiples with cartopy coastlines, per-variable cmaps
(`_colorscale_for`), symmetric ranges for `_DIVERGING_VARS`, shared colorbars,
metrics annotation. Lazy cartopy import.
**Acceptance criteria:**
- [ ] Map figures visually match the interactive report's data (same ranges/cmaps) with coastlines and colorbars
- [ ] No-reference case (no metrics/ref dataset) renders a plain grid of maps
- [ ] Diverging variables get symmetric color limits; `pr` bias uses `BrBG`
**Verification:** unit tests on axes count / cmap names for bias + no-reference layouts; manual render against a showcase `.ddb`
**Dependencies:** A1 · **Scope:** S-M (2 files)

#### Task A3: ZonalProfile + ZonalLine
**Description:** ZonalProfile: lat×pressure filled contour/pcolormesh with
inverted log pressure axis, ref + bias panels. ZonalLine: lat line plot,
multi-dataset.
**Acceptance criteria:**
- [ ] ZonalProfile pressure axis decreases upward (log scale), bias panels symmetric
- [ ] ZonalLine draws one line per dataset with legend and units label
**Verification:** unit tests (axis inversion/scale, line count); manual render
**Dependencies:** A1 · **Scope:** S (2 files)

#### Task A4: AnnualCycle + DiurnalCycle + Histogram
**Description:** Cycle plots: month/hour on x, one line per dataset. Histogram:
`density_*` columns (`get_density_cols`) as step/bar plot per dataset.
**Acceptance criteria:**
- [ ] AnnualCycle x-axis labeled with month names; DiurnalCycle 0–23 h
- [ ] Histogram renders all density columns with dataset legend
**Verification:** unit tests per type; manual render
**Dependencies:** A1 · **Scope:** S (2 files)

#### Task A5: Hovmöller + QBO
**Description:** Hovmöller: time×lon pcolormesh per dataset (stacked panels),
`_hovmoller_colorscale_for`. QBO: stacked per-dataset time×pressure heatmaps +
amplitude/index line panels, mirroring the interactive layout.
**Acceptance criteria:**
- [ ] Hovmöller one panel per dataset with shared color range and one colorbar
- [ ] QBO heatmap rows share color range; amplitude/index panels present
**Verification:** unit tests (panel counts, shared clim); manual render
**Dependencies:** A1 · **Scope:** M (2 files)

#### Task A6: Scalar (bar) + annotation polish
**Description:** Scalar diagnostics as bar charts (one bar per dataset,
reference highlighted). Sweep all builders for consistent titles (long name +
units + short id), metrics annotations, figure size, font.
**Acceptance criteria:**
- [ ] All 10 types registered in `_STATIC_FIGURE_BUILDERS`; unsupported-type warning path removed
- [ ] Consistent title/annotation scheme across types
**Verification:** registry-completeness unit test (10 keys, matching `_FIGURE_BUILDERS`); full-suite manual render of the MPI-ESM1-2-HR showcase DBs, visual review
**Dependencies:** A2–A5 · **Scope:** S (2 files)

#### Task A6b: Portrait plot (overall model evaluation summary)
**Description:** Static matplotlib portrait plot in the style of ESMValTool's
`portrait_plot.py` (ICONEval `recipe_portrait_plot.yml`): models on x,
variables on y, cell colour = RMSE normalized by the centered median across
models (diverging cmap, ±0.5 "Relative RMSE" colorbar, small tick labels).
Reuses the normalization logic of `build_leaderboard_data` /
`build_scoreboard_figure` (`_leaderboard.py`). Written once per
`generate_static_plots` run (`out_dir/portrait_plot.<fmt>`) when metrics
exist; falls back to raw RMSE colouring for a single model (like the
scoreboard). Optionally restyle the interactive Plotly scoreboard to the same
cmap/scale so both backends match. Split cells for alternative references
deferred.
**Acceptance criteria:**
- [ ] Multi-model DBs produce a centered-median portrait matching the scoreboard's numbers
- [ ] Single-model DBs fall back gracefully (raw RMSE)
**Verification:** unit test against synthetic metrics; manual render with cmip6_historical_dbs
**Dependencies:** A1 · **Scope:** S-M (2-3 files)

#### Task A7: Docs + full validation
**Description:** New "Static figure export (PNG/PDF)" section in
`docs/reporting.md` (CLI + API examples); update CLAUDE.md report-module
listing; changelog-style PR description with sample figures attached.
**Acceptance criteria:**
- [ ] Docs cover CLI flag, API signature, output layout, formats/dpi
- [ ] Full run over showcase `.ddb`s produces one file per (tier, diagnostic, variable) with no errors
**Verification:** `pixi run pytest`; `pixi run pre-commit run --all-files`; manual doc read-through
**Dependencies:** A6 · **Scope:** S (2-3 files)

### Checkpoints (Workstream A)
- **After A1:** end-to-end path works for one type; PR-able slice. Human review of figure aesthetics direction (one sample PNG) before mass-producing the other 9 types.
- **After A2:** Map is the flagship figure — attach sample PNG/PDF to the PR/issue for style sign-off vs ICONEval look.
- **After A7:** definition of done from #18 met; close the issue.

PR strategy: either one PR per checkpoint (A1 → A2 → A3-A6+A7) or a single PR
built up in reviewable commits — decide at the A1 checkpoint based on diff size.

---

## Workstream B — Issue #17: remaining report-HTML polish

Done via PR #19: tier identity ✓, plot-type grouping ✓ (+ long names, linked
zoom). Remaining checklist below. All items are CSS/HTML-only
(`_style.py`, `_html.py`, `_leaderboard.py`), no figure-computation changes,
and must work in **both** render paths (server shell + portable HTML) — they
share the same renderers, so test both outputs each time.

#### Task B1: Spacing scale + card separation ("Whitespace & rhythm" + "Diagnostic cards")
**Description:** Introduce `--space-1..--space-6` (8 px base) and `--radius-*`
tokens in `:root` (`_style.py:145`), replace hardcoded paddings/margins/gaps,
increase inter-section spacing, strengthen `diag-section` card separation
(surface + border + title bar already carries icon; add subtle category chip).
**Acceptance criteria:**
- [ ] All spacing in `_style.py` uses the token scale (grep: no stray `margin:/padding:` px literals off-scale)
- [ ] Sections/groups visibly separated; before/after screenshots attached
**Verification:** `pixi run pytest`; render portable HTML from a showcase `.ddb`, visual check in browser (server + portable)
**Dependencies:** None · **Scope:** S (1-2 files)

#### Task B2: Tame var-nav + toolbar
**Description:** `var-nav` (`_html.py:236-243,308`): sort pills, sticky within
the section, compact overflow behavior past ~10 pills (wrap-to-scroll row or
"+N more" expander), clearer active state. Toolbar `diag-filter-btns`
(`_html.py:482-503`): group filter pills by plot-type category (reuse
`_diag_category`) so the row stays tidy.
**Acceptance criteria:**
- [ ] `Tier2_atmosphere_monthly` (19 vars) var-nav fits one tidy row/control
- [ ] Toolbar groups by category and no longer overflows with many diagnostics
- [ ] Behavior identical in server + portable modes
**Verification:** manual check against Tier-2 showcase DB in both modes; unit test if any Python-side grouping logic is added
**Dependencies:** B1 (uses the new tokens) · **Scope:** M (2 files)

#### Task B3: Leaderboard hierarchy + legend
**Description:** `_leaderboard.py`: emphasize Rank + Overall Score,
de-emphasize secondary metric columns (muted color/size), add an HTML
legend/scale for the score heatmap colors (the Plotly colorbar exists but the
summary-table HSL bars have no legend, `_leaderboard.py:68-91`).
**Acceptance criteria:**
- [ ] Rank/score visually dominant; secondary columns muted
- [ ] Legend explains the red→green score bar and the blue/white/red scoreboard scale
**Verification:** render leaderboard with multi-model DBs (cmip6_historical_dbs), visual check; existing tests green
**Dependencies:** B1 · **Scope:** S (2 files)

#### Task B4: Accessibility pass
**Description:** AA contrast audit of text/badges/pills (tier accents were
AA-checked in #19; verify the rest), add `:focus-visible` styles to all
interactive elements (`var-pill`, `diag-filter-btn`, `sidebar-item`, sort
headers), remove the bare `outline: none` (`_style.py:337`), keyboard
operability for pills/tabs (they are `<button>`s — verify tab order and add
`aria-pressed`/`aria-label` where missing), color never sole signal.
**Acceptance criteria:**
- [ ] Every interactive element has a visible focus state and is keyboard-operable
- [ ] Contrast spot-checks (badges, pills, muted text, score bars) meet AA (4.5:1 text, 3:1 UI)
**Verification:** keyboard walkthrough in browser; contrast checks with a tool (e.g. axe or manual ratio calc); both render modes
**Dependencies:** B1-B3 (audits their output) · **Scope:** S-M (2 files)

#### Task B5: Docs + close-out
**Description:** Note the visual/a11y conventions in `docs/reporting.md`;
attach before/after screenshots to issue #17; tick the issue checklist; close.
**Dependencies:** B4 · **Scope:** XS

### Checkpoints (Workstream B)
- **After B1:** screenshots to the issue; confirm direction before B2-B4.
- **After B4:** full definition of done from #17 (incl. items shipped in #19); close the issue.

Suggested PR slicing: B1+B2 as one PR ("layout & density"), B3+B4+B5 as a
second ("leaderboard & a11y") — or one PR if diffs stay small.

---

## Recommended order

1. **A1 → A2** first (the flagship new capability; A1 unblocks all of A3-A6).
2. **B1** can run in parallel or in between (disjoint files, quick win).
3. Then A3-A7 (mostly independent of each other, parallelizable), interleaved with B2-B5.

## Risks and mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Adding cartopy to core deps bloats install / solver conflicts | Med | Lazy import in Map builder; if the pixi solve fights, keep cartopy in a `report-static` optional feature and document it |
| Extracting helpers from `_figures.py` breaks interactive report | Med | Pure move + re-import, covered by existing `test__figures.py`; do it in its own commit |
| Static aesthetics don't match ICONEval expectations | Med | A1/A2 checkpoints: sample figures posted for review before building the remaining 8 types |
| Sticky var-nav / overflow behavior differs between server and portable HTML | Low | Both modes share renderers; test both at every B task |
| QBO/Hovmöller static layouts are fiddly (multi-panel, mixed axes) | Low | Scheduled last among builders (A5); interactive layout is the reference |

## Open questions (non-blocking; defaults chosen)

- Map projection for static maps: default **Robinson** with coastlines
  (typical ESMValTool look); flip to PlateCarree at the A2 checkpoint if
  preferred.
- Dependency placement: default **core deps**; fall back to an optional pixi
  feature if the solve or reviewer objects.
- PR granularity: decided at checkpoints (see per-workstream notes).
