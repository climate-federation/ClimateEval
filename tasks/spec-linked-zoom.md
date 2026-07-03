# Spec: Synchronized zoom across linked map panels in the interactive report

## Objective

When a user box-zooms (or pans) into a region on one panel of a multi-panel map
figure — reference, model, or bias — the other panels of that figure must zoom
to the same region, and a view reset (double-click or the modebar "Reset axes"
button) must restore all panels together. Comparing the same region across
model/reference/bias is the whole point of the side-by-side layout; today each
panel zooms independently, which makes regional comparison tedious.

**User story:** As a climate modeler reviewing a `Map` diagnostic, I draw a
zoom box over, say, the North Atlantic on the bias panel and immediately see
the model and reference panels zoom to the same lon/lat window, so I can
compare the fields over that region without manually re-zooming each panel.

**Scope decision (confirmed with user):** apply to *all* multi-panel 2D
heatmap figure types, not just maps — same mechanism, consistent UX.
**Adjusted during implementation** (constraints found in the builders):

| Figure type   | Builder                       | Sync scope                          |
|---------------|-------------------------------|-------------------------------------|
| Map           | `build_map_figures`           | all panels (lon × lat)              |
| ZonalProfile  | `build_zonal_profile_figures` | all panels (lat × pressure)         |
| QBO           | `build_qbo_figures`           | stacked heatmap rows only — the amplitude-profile and QBO-index panels have different axis semantics (σ × pressure, time × wind) and stay independent |
| Hovmoller     | `build_hovmoller_figures`     | **excluded** — its embedded Lon×Time / Time×Lon transpose buttons swap every panel's axes between numeric and categorical types, which a matched-axis group would break (the builder already forbids `shared_xaxes` for the same reason) |

**Zoom aspect decision (confirmed with user):** free-form zoom — any rectangle
the user draws becomes the view (no `scaleanchor` geographic locking).

## Technical approach

Each of these figures is a **single Plotly figure** built with
`plotly.subplots.make_subplots`, whose panels are `go.Heatmap` traces on
ordinary cartesian axes (`xaxis`, `xaxis2`, …). Plotly has a native axis
constraint for exactly this: `matches`. Setting every x-axis to
`matches="x"` and every y-axis to `matches="y"` makes zoom, pan, autorange,
and reset propagate across all subplots **client-side, with no custom
JavaScript**, so it works identically in server mode (`report.py` /
`build_html_shell`) and the portable offline HTML (`generate_portable_html`)
— the sync ships inside the figure JSON itself.

Sketch (one shared helper in `_figures.py`, called by the four builders after
all panels are added):

```python
def _sync_subplot_axes(fig: go.Figure) -> None:
    """Link all subplot axes so zoom/pan/reset applies to every panel."""
    fig.update_xaxes(matches="x")
    fig.update_yaxes(matches="y")
```

No changes to diagnostics, the DuckDB schema, `_server.py` endpoints, or
`_style.py` JS.

## Tech Stack

- Python ≥ 3.11, Plotly (`go.Figure` / `make_subplots`) — already a dependency.
- No new dependencies.

## Commands

```bash
pixi run pytest                                   # full test suite
pixi run pytest tests/unit/report -x              # fast loop on the new tests
pixi run pre-commit run --files src/climateeval/report/_figures.py tests/unit/report/test_figures.py
pixi run python report.py <output.ddb> --model <name>              # manual check, server mode
pixi run python report.py <output.ddb> --model <name> \
    --save-portable-html-output /tmp/r.html --no-browser           # manual check, portable
```

## Project Structure

```
src/climateeval/report/_figures.py   → _sync_subplot_axes() helper + calls in the
                                        4 heatmap builders (only file with logic changes)
tests/unit/report/__init__.py        → new package (repo has no report unit tests yet)
tests/unit/report/test_figures.py    → new tests for axis-matching wiring
docs/reporting.md                    → one-line mention of linked zoom on multi-panel figures
```

## Code Style

Match `_figures.py` conventions: module-private helpers prefixed `_`, typed
signatures, one-line imperative docstrings, ruff + mypy clean (enforced by
pre-commit / CI). Example of the calling convention inside a builder:

```python
        _sync_subplot_axes(fig)
        fig.update_layout(**_base_layout(...))
        figures[var_id] = fig
```

## Testing Strategy

- **Framework:** pytest (existing `pixi run pytest` setup); new tests live in
  `tests/unit/report/test_figures.py`.
- **Unit tests:** build figures from a small synthetic `data_df` (model +
  reference datasets on a coarse lat/lon grid, plus minimal frames for the
  profile/Hovmöller/QBO builders) and assert that *every* `xaxis*` in
  `fig.layout` has `matches == "x"` and every `yaxis*` has `matches == "y"`,
  for all four builders, in both the bias layout and the no-reference
  fallback grid.
- **Manual verification (required before done):** render a report from an
  existing `.ddb`, box-zoom one map panel and confirm the sibling panels
  follow; double-click to confirm the reset re-syncs; repeat once in portable
  HTML. Existing behavior of single-panel figures (TimeSeries, ZonalLine,
  Histogram, …) must be untouched.
- Coverage expectation: the new helper and its four call sites are fully
  exercised by the unit tests.

## Boundaries

- **Always:** run `pre-commit` on touched files before handing back; keep both
  render paths (server + portable) working; keep the feature self-contained in
  the figure JSON (no server-side state, no external JS).
- **Ask first:** adding any dependency; introducing custom JavaScript relayout
  handlers (only needed if `matches` proves insufficient — see risks); changing
  initial axis ranges or figure layout beyond the sync; committing (user
  decides when; branch `report-viz`).
- **Never:** touch diagnostic computation or the DuckDB output schema; break
  the offline portable report; remove or weaken existing figure behavior
  (hover, colorbars, coastlines).

## Success Criteria

1. Box-zoom on any panel of a Map figure re-ranges **all** panels of that
   figure (including other model rows) to the same window; pan behaves the
   same way.
2. Double-click / "Reset axes" on any panel restores every panel of the figure
   to the initial full view.
3. Behavior 1–2 holds for ZonalProfile, Hovmöller, and QBO figures, and in
   both server and portable reports.
4. Unit tests assert the `matches` wiring for all four builders; full test
   suite and pre-commit pass.
5. Single-panel figure types are unaffected.

## Risks / notes

- **Reversed & log axes:** ZonalProfile/QBO pressure axes may use
  `autorange="reversed"` or log scale; `matches` requires consistent axis
  types across matched axes (they are, per panel column/row layout) — verify
  during manual check.
- **Union of ranges:** for Hovmöller/QBO, datasets with different time spans
  currently autorange per panel; `matches` will unify them to a common range.
  This is a (desirable) visible behavior change — note it in the PR.
- **Bias panels:** share lon/lat with data panels, so a single matched group
  per figure is correct; no per-column groups needed.

## Open Questions

None — sync scope (all multi-panel 2D types) and zoom aspect (free-form)
were confirmed. Spec file location: project root per convention; can move to
`tasks/` alongside `plan.md`/`todo.md` if preferred.
