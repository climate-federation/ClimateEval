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
independent because of their axis-swap buttons.

The coordinate columns each diagnostic writes are listed in
[Output schema](output-schema.md).
