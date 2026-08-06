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

## Workstream C — Benchmarking throughput (multi-model tier runs)

**Motivation (2026-07-05):** the sequential MPI-vs-4-CMIP6 Tier2 run
(`scripts/run_mpi_vs_cmip6_tiers.py`) takes ~1 h *per model* because every
model run appends into one DuckDB (single-writer) and recomputes the ERA5 /
GPCP / CERES references **and** the ~50-model CMIP6 comparison ensemble from
scratch. Goal: parallel per-model runs + a merge step, then kill the redundant
recomputation.

**Architecture decisions**
- Per-model scratch `.ddb` files (proven by the 52-model array job) + a
  **merge tool** that unions them into one combined db per tier — DuckDB has no
  concurrent writers, so parallelism must happen at the file level.
- Merge dedups on natural keys (`data_sources.id`, `variables.id`,
  `raw_output.data_id`, `metrics.data_id`) — reference/ensemble rows appear in
  every per-model db; first occurrence wins. Tolerates column drift
  (`INSERT … BY NAME` + `ALTER TABLE ADD COLUMN`) since a model missing a
  variable may lack that raw_output column.
- Merge lives in `scripts/` first (machine-specific driver territory);
  promote into the package + tests once the workflow stabilizes.
- Reference-reuse (C3) is a **core** change → own branch/PR, not on
  `ap/static-figures`.

#### Task C1: `scripts/merge_tier_dbs.py`
**Description:** CLI `python merge_tier_dbs.py OUT.ddb SRC1.ddb SRC2.ddb …`.
For each attached database (diagnostic) in each source: create missing
schemas/tables in OUT, add missing columns, insert rows whose dedup key is not
yet present. Source order defines precedence (pass the benchmarked model first).
**Acceptance criteria:**
- [ ] Merging N per-model dbs yields one db; each dataset's rows appear once
      (no duplicated ERA5/reference rows).
- [ ] Works with column drift (model missing a variable).
- [ ] `report.py OUT.ddb` loads it and the leaderboard shows all models.
**Verification:** merge 3 real dbs from `cmip6_historical_dbs/`, compare
per-table row counts vs dedup expectation; open with report.py.
**Dependencies:** none · **Scope:** S (1 file)

#### Task C2: parallel SLURM driver + submit script
**Description:** `scripts/run_tier_benchmark_model.py` (one model per process:
`--suite`, `--model INSTITUTE/DATASET` or `--model-index` + `--models-file`,
`--timerange`, `--out-dir`; loads Amon from the DKRZ CMIP6 archive, any grid,
latest version) + `scripts/submit_tier_benchmark.sh` (SLURM array over models,
then a `--dependency=afterok` merge job calling C1).
**Acceptance criteria:**
- [ ] One invocation = one model = one scratch db named `<suite>__<dataset>.ddb`.
- [ ] `sbatch --test-only` validates; merge job chained via afterok.
- [ ] Wall time for a 5-model Tier2 run ≈ 1 model run + merge (< ~1.5 h).
**Verification:** argparse smoke locally; full validation on next real run.
**Dependencies:** C1 · **Scope:** S-M (2 files)

#### Task C3: stop recomputing references + comparison ensemble (core)
**Description:** ~90 % of each Tier2 model run is spent re-processing datasets
that are identical across runs (ERA5/GPCP/CERES refs + CMIP6 ensemble).
Design options: (a) diagnostic kwarg `skip_data_ids` — suite checks the target
db and skips datasets already present; (b) "references-only" run mode that
builds a shared refs db once, per-model runs compute *only* the benchmarked
model + its metrics, and C1 merges refs in. Option (b) composes with C2 and
needs no db introspection inside diagnostics — preferred.
**Acceptance criteria:**
- [ ] Per-model run after the refs db exists takes minutes, not ~1 h.
- [ ] Merged output byte-equivalent (per table, key-sorted) to today's
      sequential append output.
- [ ] Unit test covering the skip/refs-only path.
**Dependencies:** C1, C2 · **Scope:** M (core: suites/diags kwargs + tests) —
**own PR off `main`**

#### Task C4: docs + promote merge into the package
**Description:** document the parallel workflow in `docs/benchmarking.md`;
move the merge logic into `climateeval` proper (e.g. `climateeval.db.merge`)
with unit tests once C1-C3 have proven the shape.
**Dependencies:** C1-C3 · **Scope:** S-M

**Further improvement candidates (unscoped, noted for later):**
- Resume/skip-existing at the per-model-db level (trivially free with C2:
  re-submit only failed array indices).
- Dask multi-worker within a model run (`cpus-per-task` currently mostly idle
  during regridding).
- Cache prepared (regridded/masked) reference cubes on disk keyed by
  (variable, grid, timerange) — helps even single-model interactive runs.
- Tier1 suites are cheap (~10 s) — no parallelization needed; keep them in the
  benchmarked model's array task only.

### Checkpoints (Workstream C)
- **After C1:** merged db from 3 real per-model dbs opens in report.py with a
  correct leaderboard → then wire C2.
- **After C3 design, before implementation:** confirm option (a) vs (b) with
  the team (touches suite semantics).

---

## Workstream D — In-process parallelization (speed up ONE suite run)

**Motivation (2026-07-05):** Workstream C parallelizes *across models* at the
job level; a single Tier2 run still takes ~1 h. Code reading shows why:

- `SimpleDiagnostic._get_output_of_other_data()`
  (`diags/simple/_base.py:314`) loops **sequentially** over
  (variable × other_source) — for Tier2 that is ~10 variables × ~50 CMIP6
  ensemble members ≈ 500 independent load→preprocess→metric chains.
- `Suite.get_database()` (`suites/_base.py:75`) runs the 6 diagnostics
  sequentially; each re-derives everything it needs.
- There is **no dask/threading configuration anywhere** in the package —
  iris/dask defaults decide chunking and worker counts; SLURM
  `cpus-per-task` is mostly idle.
- Reference cubes are eagerly realized (`reference_cube.data`,
  `diags/simple/_base.py:474` — workaround for ESMValCore PR #3038, possibly
  obsolete) and `_get_metrics()` calls `distance_metric` 3× per pair.

**Constraint discovered:** `setup_esmvaltool_config_and_logging()`
(`_config.py`) mutates process-global state (esmvalcore `CFG.context`,
logger levels; iris `COMBINE_POLICY` likewise). Naïve threading around it
is unsafe → either hoist the context once around a parallel section, or use
process workers.

**Architecture decisions**
- **Profile before parallelizing** — the ~1 h/model estimate is anecdotal;
  D1 produces the breakdown that decides how far to go (D3 vs also D4).
- Dataset-level parallelism (D3) is the highest-leverage target (the ~500
  independent chains) and is invisible to callers — a `max_workers`
  diagnostic kwarg, default 1 (today's behavior).
- All D tasks are **core changes → own branch/PR off `main`**, not
  `ap/static-figures`. C3 (refs-only mode) removes the ensemble loop from
  per-model runs entirely; D3 still pays off for the *first*/refs run and
  for interactive `benchmark.py` users.

#### Task D1: Profile a real Tier2 run
**Description:** `scripts/profile_tier2.py` — run one model through
Tier2_atmosphere_monthly with per-phase wall-time logging (per diagnostic ×
{references, to_benchmark, other} × {load, preprocess, metrics}), on a
reduced variable set first, then one full run; optionally a py-spy flame
graph. Pure scripts/ addition, no core changes.
**Acceptance criteria:**
- [ ] Table of top cost centers (phase × step) for a full single-model run.
- [ ] Answers: is time CPU-bound (regrid/metrics) or I/O-bound (load)?
      Are dask graphs realized single-threaded?
**Verification:** breakdown numbers sum to ≈ total wall time.
**Dependencies:** none · **Scope:** S (1 file)

#### Task D2: Dask/threading configuration knob
**Description:** size dask's threaded scheduler + OMP/BLAS thread pools to
the available cores (env-driven, e.g. `CLIMATEEVAL_NUM_THREADS`, defaulting
to `len(os.sched_getaffinity(0))`); set it in `_config.py` alongside the
existing setup; document in docs/benchmarking.md; align submit scripts'
`cpus-per-task`/OMP exports.
**Acceptance criteria:**
- [ ] Measurable speedup (or documented null result) on the D1 scenario.
- [ ] No change in output values (bitwise per table on the D1 scenario).
**Verification:** re-run D1 profile with/without; compare tables.
**Dependencies:** D1 · **Scope:** S (1-2 files + docs)

#### Task D3: Parallel dataset processing inside a diagnostic (core)
**Description:** `max_workers` kwarg on `SimpleDiagnostic` (suite-settable
via `diagnostic_kwargs`); fan the (variable × other_source) chains of
`_get_output_of_other_data()` (and `_get_reference_cubes()`) over a worker
pool. Design point: hoist `setup_esmvaltool_config_and_logging()` ONCE
around the parallel section (it is currently entered per `distance_metric`
call) so workers don't race on global config; workers return plain
DataFrames, table assembly stays on the main thread. Threads first (dask
releases the GIL for numeric work); switch to processes only if D1 shows
GIL saturation.
**Acceptance criteria:**
- [ ] Output identical to sequential (key-sorted table comparison) for a
      2-variable × 3-other-source test fixture.
- [ ] ≥2× wall-time reduction on the ensemble-heavy D1 scenario with
      `max_workers=8`.
- [ ] `max_workers=1` (default) takes the exact sequential code path.
**Verification:** new unit tests + D1 re-profile; full
`pixi run pytest` green.
**Dependencies:** D1 (D2 numbers inform worker count) · **Scope:** M
(diags/simple/_base.py, _config.py, tests) — **own PR off `main`**

#### Task D4: Concurrent diagnostics in `Suite.get_database` (decide after D3)
**Description:** run the 6 diagnostics' `get_output()` concurrently and
serialize only the DuckDB writes (single writer). Only worth it if D1 shows
no single diagnostic dominating; skip if D3 already saturates cores or
memory (concurrent realization of several ensembles on a 64 G node).
**Acceptance criteria:**
- [ ] Go/no-go note in this plan backed by D1/D3 numbers; if go: identical
      db content vs sequential + measured speedup.
**Dependencies:** D1, D3 · **Scope:** M (suites/_base.py + tests)

**Quick-win candidates to fold into D3's PR (verify in D1 first):**
- Drop the eager `reference_cube.data` realization if ESMValCore ≥ current
  pin no longer needs the PR #3038 workaround.
- `_get_metrics()`: compute the 3 distance metrics without re-realizing the
  cube pair 3× (realize once, or `dask.compute` the three together).

### Checkpoints (Workstream D)
- **After D1:** review the breakdown together; pick worker counts and decide
  D4 go/no-go before writing any core code.
- **After D3:** equivalence tests green + speedup measured → then D4
  decision.

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
