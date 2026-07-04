# Todo — open GitHub issues (see tasks/plan.md for full task specs)

## Workstream A — #18 Static plot export (PNG/PDF)

- [ ] **A1** (in progress, branch `ap/static-figures`) Skeleton + shared helpers + deps + `generate_static_plots()` API + `report.py --static-out/--static-format/--dpi` CLI + TimeSeries builder (end-to-end slice). Style source: ICONEval `recipe_templates` → ESMValTool `monitor/multi_datasets.py` (seaborn ticks, 7×5 in, gray gridlines, legend below axes, black refs w/ linestyle cycle, lightgray "other" bundle, dpi 300 tight bbox; maps later: Robinson central_lon=10, discrete levels, common_cbar, fontsize 8)
- [ ] **Checkpoint:** sample TimeSeries PNG reviewed; PR strategy decided
- [ ] **A2** Map builder — cartopy coastlines, ref + bias small-multiples, per-var cmaps, symmetric diverging ranges
- [ ] **Checkpoint:** Map sample PNG/PDF posted to #18 for style sign-off
- [ ] **A3** ZonalProfile (log-pressure, inverted) + ZonalLine
- [ ] **A4** AnnualCycle + DiurnalCycle + Histogram
- [ ] **A5** Hovmöller + QBO (stacked panels, shared clim)
- [ ] **A6** Scalar bars + consistent titles/metrics annotations; registry covers all 10 types
- [ ] **A7** docs/reporting.md section + full showcase-DB render validation
- [ ] **Checkpoint:** #18 definition of done met → close issue

## Workstream B — #17 Report HTML polish (remaining after PR #19)

- [ ] **B1** Spacing/radius token scale (8 px base) + stronger diag-section cards
- [ ] **Checkpoint:** before/after screenshots to #17; confirm direction
- [ ] **B2** Var-nav taming (sort, sticky, overflow past ~10 pills) + toolbar grouped by category
- [ ] **B3** Leaderboard hierarchy (rank/score dominant) + heatmap/score-bar legend
- [ ] **B4** Accessibility: `:focus-visible` everywhere, drop bare `outline:none`, keyboard walkthrough, AA contrast spot-checks
- [ ] **B5** Docs note + screenshots + tick #17 checklist → close issue

## Standing verification (every task)

- [ ] `pixi run pytest` green
- [ ] `pixi run pre-commit run --all-files` green
- [ ] Report changes verified in BOTH server and portable-HTML modes
