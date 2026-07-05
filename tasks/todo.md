# Todo — open GitHub issues (see tasks/plan.md for full task specs)

## Workstream A — #18 Static plot export (PNG/PDF)

- [x] **A1** (branch `ap/static-figures`) Skeleton + shared helpers + deps + `generate_static_plots()` API + `report.py --static-out/--static-format/--dpi` CLI + TimeSeries builder. Style source: ICONEval `recipe_templates` → ESMValTool `monitor/multi_datasets.py` (seaborn ticks, 7×5 in, gray gridlines, legend below axes, black refs w/ linestyle cycle, lightgray "other" bundle, dpi 300 tight bbox)
- [x] **Checkpoint:** TimeSeries sample reviewed by user ("sample figures are working")
- [x] **A2** Map builder — multi_datasets.py style: Robinson central_lon=10, coastlines, gridlines, 12 discrete contourf levels, 3-panel (model|ref|bias) for single model, ref+bias grid otherwise, per-panel mean/RMSE/R² stats, polar stereo for sea ice, cyclic seam fix, rasterized
- [x] **Checkpoint:** Map samples reviewed by user ("maps are working"); missing pieces from monitor scripts added (gridlines, stats, rasterize)
- [x] **A3** ZonalProfile (log-pressure inverted, vertical cbars, stats at 0.01/0.7) + ZonalLine (variable_vs_lat)
- [x] **A4** AnnualCycle + DiurnalCycle (multi_datasets 1D cycles) + Histogram (plot_histogram.py bars, log-y for pr)
- [x] **A5** Hovmöller (stacked lon×time pcolormesh, shared clim) + QBO (heatmaps + amplitude + ~50 hPa index; Pa→hPa)
- [x] **A6** Scalar bars (blue benchmark on top, gold refs, gray others); registry now equals _FIGURE_BUILDERS (all 10 types, test-enforced)
- [x] **A6b** Portrait plot (ESMValTool `portrait_plot.py` style): centered_median relative RMSE, RdYlBu_r ±0.5, NaN white, single-model fallback annotates raw RMSE; written to `out_dir/portrait_plot.<fmt>` per run. (Plotly-scoreboard restyle deferred to workstream B3.)
- [x] **A7** docs/reporting.md "Static figure export" section + CLI table + AGENTS.md module reference; full 8-tier showcase render validated
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
