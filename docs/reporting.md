# Reporting

Once a suite has written one or more `.ddb` databases, ClimateEval turns them
into an interactive benchmarking report: a sidebar of tiers/diagnostics, a plot
panel per variable, and a leaderboard heatmap summarising the metrics. The
same databases can also be rendered as **publication-ready static figures**
(PNG/PDF) styled like the ESMValTool/ICONEval monitoring plots — see
[Static figure export](#static-figure-export-pngpdf).

There are two ways to view a report, both driven by
[`report.py`](../report.py) and the `climateeval.report` module.

## `report.py` (command line)

```bash
python report.py output.ddb [output2.ddb ...] [options]
```

By default it starts a **local web server** and opens your browser. The browser
gets a thin HTML shell and fetches each plot only when you select it, so it
stays fast no matter how many models or variables are in the databases.

| Option | Default | Meaning |
|---|---|---|
| `db` (positional) | — | One or more DuckDB `.ddb` files. |
| `--model NAME` | `""` | Model name shown in the header. |
| `--user NAME` | `""` | Analyst name shown in the header. |
| `--developer NAME` | `""` | Developer / tool name shown in the header. |
| `--host ADDR` | `127.0.0.1` | Server bind address. |
| `--port N` | `8765` | Server port. |
| `--no-browser` | off | Don't open the browser automatically. |
| `--save-portable-html-output [PATH]` | — | Also write a self-contained HTML file. Omit `PATH` to place it next to the first database as `<stem>_report.html`. |
| `--static-out DIR` | — | Render static figures (see below) into `DIR` and exit; no server is started. |
| `--static-format FMT[,FMT...]` | `png` | Static image format(s), e.g. `png,pdf`. |
| `--dpi N` | `300` | Resolution of raster static figures. |

Examples:

```bash
# Interactive server (recommended for large databases)
python report.py Tier2_atmosphere_monthly.ddb Tier1_sanity_checks.ddb --model MPI-ESM1-2-LR --user Alice

# Export a portable, offline single-file HTML and don't open a browser
python report.py output.ddb --model MPI-ESM1-2-LR --save-portable-html-output report.html --no-browser
```

### Server endpoints

The server exposes:

| Endpoint | Description |
|---|---|
| `GET /` | HTML shell (sidebar + empty panels). |
| `GET /api/structure` | Tier / diagnostic / variable metadata. |
| `GET /api/figure/<tier>/<diag>/<var>` | Plotly figure JSON (built lazily and cached). |
| `GET /api/leaderboard` | Scoreboard heatmap + summary table. |
| `GET /plotly.js` | Bundled Plotly.js (so the page works offline). |

## Python API

```python
from climateeval.report import serve, generate_portable_html

# Start the interactive server (blocks until interrupted):
serve(
    ["output.ddb"],
    model_name="MPI-ESM1-2-LR",
    user_name="Alice",
    host="127.0.0.1",
    port=8765,
    open_browser=True,
)

# Or build one self-contained HTML string with every figure embedded:
html = generate_portable_html(["output.ddb"], offline=True, model_name="MPI-ESM1-2-LR")
open("report.html", "w").write(html)
```

- **`serve(...)`** — interactive server; best for day-to-day exploration of
  large databases. Also accepts `save_html=<path>` to write a portable file too.
- **`generate_portable_html(...)`** — a single offline HTML file, ideal for
  sharing. With many models it can be large (hundreds of MB); pass
  `offline=False` to load Plotly.js from a CDN instead of embedding it.

## Static figure export (PNG/PDF)

For papers, reports, and quick review, the same databases can be rendered as
static matplotlib/cartopy figures that match the look of the ESMValTool
`monitor/multi_datasets.py` plots used by ICONEval's `recipe_templates`
(seaborn "ticks" theme, Robinson maps with coastlines and discrete contour
levels, black reference lines, legends below the axes, per-panel mean/RMSE/R²
annotations).

```bash
# One PNG per (database, diagnostic, variable) into figs/
python report.py output.ddb --static-out figs/

# PNG + PDF at 300 dpi from several tiers
python report.py Tier2_atmosphere_monthly.ddb Tier3_dynamics.ddb \
    --static-out figs/ --static-format png,pdf --dpi 300
```

Files land in `figs/<database>/<diagnostic>/<variable>.<format>`. When any
diagnostic recorded `weighted_rmse` metrics, an **overall portrait plot**
(models × variables, RMSE normalized by the centred median across models —
ESMValTool `portrait_plot.py` style) is written once to
`figs/portrait_plot.<format>`.

The equivalent Python API:

```python
from pathlib import Path
from climateeval.report import generate_static_plots

written = generate_static_plots(
    [Path("output.ddb")],
    Path("figs"),
    formats=("png", "pdf"),
    dpi=300,
)
```

All ten figure types below have a static counterpart (registered in
[`src/climateeval/report/_static_figures.py`](../src/climateeval/report/_static_figures.py)).
Layout rules for the 2-D types follow `multi_datasets.py`: with a single
benchmarked model and a reference, maps and zonal profiles use the three-panel
layout (model | reference on a common colorbar, bias below); with several
models the reference is shown once followed by one bias panel per model;
without a reference a plain grid of absolute panels is drawn. Sea-ice maps
switch automatically to polar stereographic views.

## Sharing a report (DKRZ Swift)

A portable HTML report is a single self-contained file, so it can be published
to [DKRZ's Swift object storage](https://swiftbrowser.dkrz.de) and opened by
anyone with the link — no DKRZ account required. This mirrors how ICONEval
publishes its output, so ClimateEval reports get the same style of URL. Add
`--share` when writing the HTML:

```bash
# Write the self-contained HTML and publish it to Swift as a public link
climateeval report output.ddb --model MPI-ESM1-2-LR \
    --html report.html --share
# → prints e.g.
#   https://swift.dkrz.de/v1/dkrz_<uuid>/climateeval/MPI-ESM1-2-LR/index.html
```

The report is stored as `<container>/<name>/index.html`, where `<name>` is the
`--model` (or the HTML file stem), so each report gets a clean, collision-free
URL under its own prefix. Defaults: container `climateeval`
(`--share-container NAME`), prefix from `--model` (`--share-name NAME`).
For `report`, `--share` requires `--html`.

`climateeval benchmark` takes the same `--share`/`--share-container`/
`--share-name` flags and publishes the report it writes in one step:

```bash
climateeval benchmark /path/to/model/cmor/Amon --name MPI-ESM1-2-LR --share
```

### Credentials

`--share` uses the first of these it finds (matching DKRZ's `swift-token` /
ICONEval workflow):

1. **A cached token in `~/.swiftenv`** — the usual path on Levante. Create or
   refresh it with `swift-token` (from `py-python-swiftclient`), entering your
   **project account**, username and password:
   ```bash
   swift-token          # writes ~/.swiftenv (OS_STORAGE_URL + OS_AUTH_TOKEN)
   ```
   Using a project account publishes into the shared 10 TiB project space
   (alongside ICONEval) rather than your 50 GiB user account. The token
   expires; re-run `swift-token` when sharing starts failing with `401`.
2. **A pre-fetched token in the environment:**
   ```bash
   export OS_STORAGE_URL=https://swift.dkrz.de/v1/dkrz_<uuid>
   export OS_AUTH_TOKEN=<token>
   ```
3. **Legacy tempauth** (a one-shot login, no caching):
   ```bash
   export ST_USER=<account>:<user>   # a project account for the shared space
   export ST_KEY=<password>
   ```

The container's read ACL is set to `.r:*`, so the returned URL is
world-readable — treat the link as public; Swift keeps no backup. Note that a
benchmark report embeds run metadata in its header, including the **absolute
model path** and the **hostname** it ran on, which become public along with the
figures. The uploader is a thin wrapper over Swift's RESTful API in
[`src/climateeval/_share.py`](../src/climateeval/_share.py) (`share_file()`),
usable directly from Python too.

## Reading the report

- **Theme.** The switch at the top right offers *Auto* (follow the operating
  system), *Light* and *Dark*. The choice is remembered in the browser. The
  figures follow the theme too, in the server and in the portable HTML.
- **Lines.** In every 1-D plot the benchmarked model is a thick solid line in
  its own colour, references are solid lines in dark colours (black first), and
  each compared dataset is a thin line with its own colour + dash pair. Every
  drawn line is named in the legend below the axes; with more than eight
  compared datasets a single "n other datasets" entry stands for the bundle
  and toggles it as a whole. The static PNG/PDF export uses the same scheme.
- **Maps.** Map grids fit the width of the card: side by side with vertical
  colorbars on a wide screen, stacked in one column with the colorbars at the
  bottom on a narrow one.
- **Hovmöller.** Two toggles above the plot choose the axes (longitude × time
  or time × longitude) and the direction (W → E or E → W).
- **Tiers.** The toolbar tabs select one diagnostic or *All*; in the *All*
  view the diagnostics are grouped by plot type (spatial, temporal, …).
  Figures render as they scroll into view. A long variable row starts
  collapsed to two rows behind *Show all*.
- **Leaderboard.** The header says which model is benchmarked, which datasets
  it is compared with and which references score it. Every dataset is scored
  against the same reference per variable; reference datasets appear in the
  table as a yardstick but are not ranked. Hover the ⓘ icons for the metric
  definitions. In the scoreboard a relative score above zero (blue) means a
  smaller error than the median dataset, below zero (red) a larger one.

## Figure types

The report auto-detects how to plot each diagnostic from its output coordinates.
The supported types (registered in
[`src/climateeval/report/_figures.py`](../src/climateeval/report/_figures.py)):

| Type | Label | Plot style |
|---|---|---|
| `Map` | 🗺 Map | 2-D heatmap with coastlines; reference and model−reference bias. |
| `ZonalProfile` | 🌡 Zonal Profile | 2-D pressure × latitude heatmap; reference and bias. |
| `ZonalLine` | 〰 Zonal Mean | 1-D line, zonal mean vs. latitude. |
| `TimeSeries` | 📈 Time Series | 1-D line over time. |
| `AnnualCycle` | 📅 Annual Cycle | 1-D line, monthly climatology. |
| `DiurnalCycle` | 🌅 Diurnal Cycle | 1-D line, hourly climatology (local solar time). |
| `Hovmoller` | 🌊 Hovmöller | 2-D time × space heatmap. |
| `QBO` | 🌀 QBO | 2-D time × pressure (equatorial zonal wind). |
| `Histogram` | 📊 Histogram | Distribution histogram. |
| `Scalar` | 📊 Scalar | Horizontal bar chart (0-D values). |

Multi-panel figures have linked zoom: box-zooming or panning one panel of a
`Map`, `ZonalProfile`, or `QBO` figure applies the same view to its sibling
panels (e.g. reference, model, and bias maps stay aligned), and double-click
or "Reset axes" restores all panels together. Hovmöller panels stay
independent because of their axis and direction toggles.

The coordinate columns each diagnostic writes are listed in
[Output schema](output-schema.md).
