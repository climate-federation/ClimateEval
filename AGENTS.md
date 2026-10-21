# AGENTS.md file

> **User-facing documentation lives in [`docs/`](docs/README.md)** (installation,
> usage, suites, data sources, reporting, output schema, benchmarking). This file
> is the internal architecture reference for contributors and coding agents
> (it is also surfaced to Claude Code as `CLAUDE.md`, a symlink).

## Project Overview

ClimateEval is a framework for evaluating and benchmarking Earth System Model
output. It allows climate modelers to systematically assess their simulations
across multiple variables and diagnostics, and to benchmark them against
established observational references within a unified, extensible framework.

ClimateEval employs a suite-based architecture where users define diagnostics
(e.g., spatial maps, time series, vertical profiles), variables, and reference
data sets (e.g., ERA5 reanalysis, GPCP precipitation, CERES-EBAF radiative
fluxes) through YAML configuration files, allowing flexible and reproducible
evaluation workflows. The modular approach also facilitates the usage of custom
diagnostics, variables, and reference data sets. When executed, the suites
process input data through standardized preprocessing steps such as unit
conversion, spatial regridding, regional masking, calculation of statistics.
This preprocessing is based on ESMValTool, which provides a set of robust and
well-established preprocessing functions. Any missing reference data is
automatically downloaded if required. Diagnostics results (raw output and
metrics calculated relative to the reference data) are stored in a database
(either in-memory or on disk for post-hoc analyses), which can be then explored
through an interactive HTML report.

## Common Commands

```bash
# Install (Pixi + conda-forge). See docs/installation.md.
pixi install

# Run tests / coverage (inside the pixi env)
pixi run pytest
pixi run pytest --cov=climateeval --cov-report=html

# Lint (ruff + mypy + more, via pre-commit; enforced in CI)
pixi run pre-commit run --all-files

# Benchmark a model vs ERA5 + CMIP6 and write an HTML report (quickest start)
pixi run climateeval benchmark /path/to/model/cmor/Amon --name MyModel

# Run the quickstart example and inspect its output database
pixi run python examples/quickstart/example.py
pixi run python examples/quickstart/db_example.py

# Serve an interactive benchmarking report (recommended for large databases)
pixi run python report.py output.ddb [output2.ddb ...] --model MPI-ESM1-2-LR
# → opens http://127.0.0.1:8765 in the browser

# Export a portable single-file HTML report (offline, no server needed)
pixi run python report.py output.ddb --model MPI-ESM1-2-LR --save-portable-html-output report.html --no-browser
```

When inspecting databases, use `ibis`. See [docs/output-schema.md](docs/output-schema.md)
for the table layout and [docs/reporting.md](docs/reporting.md) for the report.

## Repository layout

```
report.py                     # report CLI (thin wrapper around climateeval.report.serve)
src/climateeval/              # the package
examples/                     # curated, runnable examples (quickstart, legoesm, ocean)
scripts/                      # experimental, machine-specific drivers (NOT linted; see scripts/README.md)
docs/                         # user documentation (plain Markdown)
tests/                        # unit/ + integration/ + sample_data/
```

## Architecture

### Core Modules (`src/climateeval/`)

- `_variable.py` - `Variable`, `Coordinate`, `VerticalLevel`, `NO_LEVEL` dataclasses. Loads metadata from `variables.yml` and `coordinates.yml` (override paths with `CLIMATEEVAL_VARIABLES_FILE` / `CLIMATEEVAL_COORDINATES_FILE`). Frozen dataclasses with auto-filled fields (units, standard_name, coordinates, derived flag, etc.)
- `_cli.py` - the `climateeval` console script (`benchmark` + `report` subcommands; also `python -m climateeval`); `_loader.py` - `load_cmor_dir()` loads model output from flat dirs, single files, or CMOR DRS trees (grid/version selection, year-split concatenation)
- `_config.py` - ESMValCore config setup (`load_from_dirs([])`), logging control, and the Iris cube combine policy
- `_utils.py` - `get_prepared_cube()` performs preprocessing (unit conversion, longitude normalization, time range extraction, level extraction, spatial masking via landsea/AR6 region masks). Also provides `str_to_class()` for dynamic class loading and `check_coordinates()` to validate cube dimensions

### Diagnostics (`src/climateeval/diags/`)

- `_base.py` - Abstract `Diagnostic` class with the `DiagnosticOutput` dataclass (fields `raw_output`, `metrics`, `variables`, `data_sources` as Ibis tables; `union()` merges outputs). `get_output(data, data_information)` is the entry point.
- `simple/` - `SimpleDiagnostic` base + the simple diagnostics: `Map`, `ZonalLine`, `ZonalProfileAirPressure`, `AnnualMeanTimeSeries`, `MeanTimeSeries`, `TimeSeriesNoPreprocessing`, `AnnualCycle`, `AnnualCycleSpatialDistribution`, `DiurnalCycle`, `Histogram`, `Hovmoller`, `QBO`, `Nino34`, `SeaIceAreaAnnualCycle`, `SeaIceAreaTimeSeries`, `SeaIceExtentAnnualCycle`, `SeaIceExtentTimeSeries`, `AMOCTimeSeries`, `OceanHeatContentTimeSeries`.
- `complex/` - `ComplexDiagnostic` base + `ECS`, `TEMDiagnostic`, `LandOceanWarmingRatio`, `ArcticAmplification`, `MeridionalHeatTransport`, `EnergyBalance`, `BudgetClosure`, `ClausiusClapeyronScaling`, `ClearSkyLongwaveFeedback`, `PrecipitationBuoyancy`, `GeostrophicBalance`, `MJOEastWestPowerRatio`, `ITCZEnergyFluxEquator`, `BjerknesCompensation`, `ENSOTeleconnectionPatterns`.
- `complex/_scalar.py` - `ScalarExperimentDiagnostic`, the base of the complex diagnostics that reduce an experiment to scalars (regridding, global means, scalar output tables, CMIP6 comparison ensemble)
- `_physics.py` - pure array formulae shared by the diagnostics (budget residuals, running-mean drift, gridpoint/pooled regressions, column integrals); no Iris or ESMValCore, unit-tested on their own.

### Suites (`src/climateeval/suites/`)

- `_base.py` - `Suite` class: reads a YAML suite definition, builds diagnostics, orchestrates execution, and writes results to an Ibis/DuckDB backend. Each diagnostic becomes an **attached database named after its `name:` field** in the YAML.
- `*.yml` - the shipped suites, organised into tiers of increasing complexity: **Tier 1** sanity/consistency (`Tier1_sanity_checks`, `Tier1_consistency_checks`, `Tier1_control_consistency`), **Tier 2** mean state vs observations (`Tier2_atmosphere_monthly`, `Tier2_ocean_monthly`, `Tier2_sea_ice_monthly`), **Tier 3** variability/processes/feedbacks (`Tier3_dynamics`, `Tier3_atmosphere_subdaily`, `Tier3_ecs`, `Tier3_covariances`, `Tier3_daily_dynamics`, `Tier3_coupled_variability`, `Tier3_itcz_efe`). Format: [docs/suites.md](docs/suites.md).

### Data Sources (`src/climateeval/data/`)

- `_base.py` - `DataSource` / `ComplexDataSource` abstract classes and `DataSourceGenerator`; `_utils.py` holds `DataSourceInformation` (provenance: name, category, institute, exp, variant, references, auto-generated `id`).
- Concrete implementations: `ERA5Monthly`/`ERA5Daily`/`ERA5Hourly`, `MERRA2`, `CERESEBAF`, `ESACCICloud`, `GPCP`, `HadCRUT5`, `HadISST`, `ESACCISST`, `NOAAERSSTv5`, `GISTEMP`, `BerkeleyEarth`, `NOAAGlobalTemp`, `CRU`, `ORAS5`, `EN4`, `WOA`, `IAP`, `ESACCIOC`, `RAPID`, `HadEX3`, `OSI450NH`/`OSI450SH`, `NSIDCG02202SH`, `ESACCISeaIceNH`/`ESACCISeaIceSH`, `CMIP6HistoricalR1I1P1F1`/`CMIP6Abrupt4xCO2R1I1P1F1`/`CMIP6PiControlR1I1P1F1`, and the generic `ESMValToolIntakeDataSource` / `ESMValToolCMORizerDataSource` / `LocalCMORDataSource` (a user-supplied local CMOR directory, loaded via `load_cmor_dir`; e.g. a held-out model as the reference of a suite). See [docs/data-sources.md](docs/data-sources.md).

### Report (`src/climateeval/report/`)

Turns result databases into an interactive report or static figures.
`report.py` (repo root) is a thin CLI over `climateeval.report.serve` and
`generate_static_plots`.

- `_server.py` - `serve()`: threaded HTTP server, lazy figure building + caching, API endpoints (`/api/structure`, `/api/figure/...`, `/api/leaderboard`, `/plotly.js`).
- `_html.py` - `build_html_shell()` (server mode) and `generate_portable_html()` (self-contained offline HTML).
- `_figures.py` - Plotly `build_*_figures()` builders and the `_FIGURE_BUILDERS` registry (10 figure types: Map, TimeSeries, AnnualCycle, DiurnalCycle, ZonalLine, ZonalProfile, Hovmoller, QBO, Histogram, Scalar).
- `_static_figures.py` - matplotlib/cartopy mirror of the registry (`_STATIC_FIGURE_BUILDERS`, same 10 types, ESMValTool `monitor/multi_datasets.py` styling) + `generate_static_plots()` (PNG/PDF export, `report.py --static-out`) + the `portrait_plot.py`-style overall portrait figure.
- `_common.py` - backend-agnostic helpers shared by both figure backends (`_var_label`, `_data_range`, `_get_ref_datasets`, `_combine_pivots`, ...).
- `_leaderboard.py`, `_db.py`, `_style.py` - scoreboard, DB reading, styling/colormaps (shared by both backends).

Details: [docs/reporting.md](docs/reporting.md).

## Data Flow

### Option 1: Use a Suite (recommended)

1. Create a `Suite` with a YAML config, e.g. `Suite("Tier2_atmosphere_monthly", diagnostic_kwargs=..., variable_kwargs=...)`
2. Call `suite.get_database(cubes, data_information, database_resource="duckdb://output.ddb")`
3. Output: an Ibis connection with one attached database per diagnostic, each containing `raw_output`, `metrics`, `variables`, and `data_sources` tables (empty tables are omitted).

### Option 2: Direct Diagnostic usage (advanced)

Diagnostic classes can be constructed and run directly via
`diagnostic.get_output(data, data_information) -> DiagnosticOutput`. The
constructors are lower-level than the suite path — see the classes under
`src/climateeval/diags/` and `tests/integration/` for concrete usage.

### Example Usage

See [examples/quickstart/example.py](examples/quickstart/example.py) for a
complete workflow, [examples/quickstart/db_example.py](examples/quickstart/db_example.py)
for inspecting the output, and [docs/usage.md](docs/usage.md) for the walkthrough.

## Key Patterns

- Variables with `derived: true` in YAML are derived via ESMValTool's derive module
- `_preprocess()` must be implemented by each diagnostic subclass
- Final coordinates are specified via the `_FINAL_COORDINATES` class variable (ensures data has expected dimensions)
- `_OUTPUT_DATA_TABLE` and `_OUTPUT_METRICS_TABLE` class variables control which tables a diagnostic outputs
- Metrics use ESMValTool's `distance_metric` function (`weighted_rmse`, `weighted_pearsonr`, `weighted_emd`)
- `fail_on_missing_data` diagnostic kwarg controls whether missing data raises errors or logs warnings
- Suites support custom `diagnostic_kwargs` and `variable_kwargs` for customization
- `DataSourceInformation` auto-generates an `id` field from category, name, exp, and variant

## DuckDB Output Schema

Suites output to DuckDB using **attached databases** (one per diagnostic). The
full table/column reference has moved to
**[docs/output-schema.md](docs/output-schema.md)**.

## Future Extensibility

The hierarchy supports expansion with:

1. **Additional diagnostics**: new classes under `diags/simple/` or `diags/complex/`, referenced by class path in a suite
2. **Additional variables**: new entries in `variables.yml` (regenerate `docs/variables.md` with `scripts/gen_variables_doc.py`)
3. **Additional data sources**: new `DataSource` subclasses under `data/`
4. **Additional suites**: new YAML configurations under `suites/`
