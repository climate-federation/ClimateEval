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

## Workstream C — Benchmarking throughput (parallel multi-model tier runs)

- [x] **C1** `scripts/merge_tier_dbs.py` — merge per-model tier dbs into one combined db (dedup on data_sources.id / variables.id / raw_output.data_id / metrics.data_id; column-drift tolerant; promotes `data_type` to `to_benchmark` for models the source benchmarked)
- [x] **Checkpoint:** merged db from 3 real `cmip6_historical_dbs/*.ddb` → 1 db (912 804 rows, dups skipped), 3× to_benchmark labels correct, portable HTML builds
- [x] **C2** `scripts/run_tier_benchmark_model.py` (one model per process) + `scripts/submit_tier_benchmark.sh` (SLURM array + afterok merge job; account bb1596/compute) — CLI smoke-tested; full SLURM validation pending first real submission
- [ ] **C3** (core, own PR off main) refs-only run mode: compute ERA5/GPCP/CERES + CMIP6 ensemble once, per-model runs compute only the model + metrics; merge via C1
- [ ] **C4** docs/benchmarking.md workflow section; promote merge into the package with unit tests

## Workstream D — In-process parallelization (single suite run; core, own PR off main)

- [ ] **D1** `scripts/profile_tier2.py` — per-phase wall-time breakdown (diagnostic × {refs,to_benchmark,other} × {load,preprocess,metrics}) for one full Tier2 model run
  - [x] script implemented (monkey-patched timers, `--max-other`/`--diags` quick modes, CSV + summary tables) and smoke-tested on Tier1_consistency_checks
  - [x] first signal already: refs phase 8.3 s vs 1.75 s load — gap is the eager `reference_cube.data` realization (PR #3038 workaround)
  - [x] full Tier2 profile run (2026-07-05, MPI-ESM1-2-LR, 61.4 min total; CSV in session scratchpad `profile_full.csv`)
- [ ] **Checkpoint:** review breakdown; set worker counts; D4 go/no-go. **D1 numbers:**
  - "other" (ensemble) phase = 2 332 s (63 %) → D3 target confirmed. Step mix there: metrics 1 122 s, to_table 453 s, load 389 s, preprocess 365 s.
  - references phase = 1 043 s (28 %) but only 29 s of it is load → ~990 s is the eager `reference_cube.data` realization (PR #3038 workaround), repeated **per diagnostic** (6× the same ERA5 data). Quick win + cross-diagnostic ref sharing.
  - metrics overall ≈ 1 200 s (33 %) → single-realization 3-metric quick win matters.
  - diagnostics are evenly sized (595–911 s; performance_metrics biggest, zonal_profile 202 s) → D4 could give ~4× more if memory allows, decide after D3.
- [ ] **D2** dask/OMP thread sizing knob (`CLIMATEEVAL_NUM_THREADS`, default = cores) in `_config.py` + docs + submit-script alignment
- [ ] **D3** `max_workers` diagnostic kwarg — parallel (variable × other_source) processing in `_get_output_of_other_data()`/`_get_reference_cubes()`; hoist global ESMValTool config context once; equivalence tests; ≥2× on ensemble-heavy run
- [ ] **D4** (decide after D3) concurrent diagnostics in `Suite.get_database` with serialized DuckDB writes
- [ ] Quick wins in D3 PR: drop obsolete `reference_cube.data` workaround (ESMValCore #3038); single-realization 3-metric computation

## Workstream E — `climateeval` console CLI

- [ ] **E1** `climateeval/_loader.py` — `load_cmor_dir()`: flat dirs, single files, CMOR DRS trees (grid preference + latest version), year-split concatenation, timerange file filter; unit tests with synthetic NetCDF trees
- [ ] **E2** `climateeval/_cli.py` + `[project.scripts]` entry point — `climateeval benchmark MODEL_DIR` (multi-suite, one .ddb per suite, portable HTML + optional static figures + --serve) and `climateeval report DB...`; docs/usage.md section
- [ ] **E3** (follow-up PR) `climateeval merge` (promote scripts/merge_tier_dbs.py into package + tests, closes C4 half) and `climateeval suites list/show`
