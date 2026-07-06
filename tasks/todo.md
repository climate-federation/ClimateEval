# Report visualization — task list

See [`tasks/plan.md`](plan.md) for full context, dependency graph, and
verification steps. Tasks are vertical slices; **⏸** marks review checkpoints.

## Phase A — Report visual hierarchy (#17)
Order: A0 → A1 → A2 → (checkpoint) → A3 → A4 → A5 → A6. Each lands in **both**
server + portable modes and is verified by regenerating a report from a `.ddb`.

- [x] **A0 — Unify render paths** — *folded*: A1 is CSS-only (no DOM dup) and A2
  uses a shared `_wrap_sections_by_category` helper for both paths, so a separate
  refactor wasn't needed.
- [x] **A1 — Tier identity** — teal/indigo/amber accent + "T1/T2/T3" badge parsed
  from the panel label, both render paths (commit `4fa7d82`). Verified in demo.
- [x] **A2 — Plot-type category grouping** — shared `_wrap_sections_by_category`
  groups diag-sections into Spatial / Temporal / Distributions / Scalars; `:has()`
  auto-hides empty category headers on filter (commit `a8f4b44`). Verified.
- [x] **⏸ Checkpoint 1** — `~/report_a2_demo.html` (A1 badges + A2 grouping). ← review
- [ ] **A3 — Spacing / dividers / card polish** — 8 px scale, bigger gaps, clearer
  dividers, plot-type icon + category chip on cards.
  *Done when*: noticeably less crowded, consistent rhythm, no regressions.
- [ ] **A4 — Variable-nav polish** — scannable `var-nav` at 19+ variables
  (sort/group, sticky, overflow, active clarity).
  *Done when*: nav tidy+usable on `Tier2_atmosphere_monthly`; `showVarPanel` works.
- [ ] **A5 — Leaderboard hierarchy** — emphasize rank/score, scoreboard legend,
  optional tier grouping in detail table.
  *Done when*: leaderboard reads clearly; `sortTable` works.
- [ ] **A6 — Accessibility pass** — AA contrast, focus states, keyboard nav, aria
  labels, no color-only encoding.
  *Done when*: WCAG 2.1 AA verified; full keyboard nav; colorblind-checked.
- [ ] **⏸ Checkpoint 2** — Feature A done; attach before/after screenshots to #17.

## Phase B — Static plot export
Order: B0 → B1 → B2 → B3 → B4 → B5. B0 proves the architecture; B1–B3 add
renderers; B4 styles; B5 integrates. New module `report/_static_figures.py`.

- [ ] **B0 — Scaffold + Map** — `_STATIC_BUILDERS` registry +
  `generate_static_plots(db_paths, out_dir, *, formats, dpi)`; Map via
  matplotlib+cartopy (ref + bias); reuse `read_database`/`detect_diag_type`/
  `get_var_cols`/`_style` colormaps.
  *Done when*: writes a Map PNG per atmosphere variable from a `.ddb`.
- [ ] **B1 — 2-D heatmaps** — ZonalProfile, Hovmöller, QBO.
  *Done when*: correct PNGs (log-pressure/time axes, colorbars).
- [ ] **B2 — 1-D lines** — TimeSeries, AnnualCycle, DiurnalCycle, ZonalLine,
  Histogram (ref/model/other + bias).
  *Done when*: correct axes + overlays for all five.
- [ ] **B3 — Scalar/bar + annotations** — bars + model·reference·metrics titles.
  *Done when*: all 10 figure types covered.
- [ ] **⏸ Checkpoint 3a** — review a full set of static PNGs (all types).
- [ ] **B4 — ESMValTool/ICONEval style match** — fonts/sizes/colorbars/coastlines/
  small-multiples/symmetric ranges; PDF + DPI. **Self-contained matplotlib.**
  *Done when*: static figures match the familiar look; PDF+PNG.
- [ ] **B5 — CLI + API + docs + tests** — `report.py --static-out DIR --format`,
  export API, `docs/reporting.md`, smoke test.
  *Done when*: works offline; documented; test + `pre-commit` green.
- [ ] **⏸ Checkpoint 4** — Feature B done; publish/close the static-plots issue.

## Decisions
- **B4**: **self-contained matplotlib** (decided).
- **Static-plot issue**: published as #18.
- **Interactive-report issue**: #17.
- **Feature branch**: `report-viz`.
- **Per-tier palette (A1)**: user chooses — implement a **modern** default, send
  the rendered report, refine from what looks best in context.
</content>
