# AGENTS.md file

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
through an interactive HTML file.

## Common Commands

```bash
# Install with development dependencies
pip install -e ".[develop]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/climateeval --cov-report=html

# Lint
pre-commit run --all

# Run the example suite and inspect the output database
python example.py
python db.py

# Serve an interactive benchmarking report (recommended for large databases)
python report.py output.ddb [output2.ddb ...] --model MPI-ESM1-2-LR
# → opens http://127.0.0.1:8765 in the browser

# Export a portable single-file HTML report (offline, no server needed)
python report.py output.ddb [output2.ddb ...] --model MPI-ESM1-2-LR \
    --save-portable-html-output report.html --no-browser
```

## Visualization: `report.py`

By default, starts a local HTTP server that reads one or more DuckDB files and
serves Plotly figure JSON on demand. The browser receives a thin HTML shell
and fetches each plot only when the user selects a tab — fast regardless of
how many models or variables are in the database. Pass
`--save-portable-html-output` to also (or instead, with `--no-browser`) write
a single self-contained HTML file with all figures embedded, useful for
sharing a report without a running server (can be several hundred MB for
large databases with many models).

API endpoints exposed by the server:

| Endpoint | Description |
|---|---|
| `GET /` | HTML shell (sidebar + empty panels) |
| `GET /api/structure` | Tier / diagnostic / variable metadata |
| `GET /api/figure/<tier>/<diag>/<var>` | Plotly figure JSON (lazily built + cached) |
| `GET /api/leaderboard` | Scoreboard heatmap + summary table |
| `GET /plotly.js` | Bundled Plotly.js for offline use |

Figure-building logic lives in `climateeval/report/_figures.py`
(`build_*_figures` functions) and supports the following diagnostic types:

| Detected type | Plot style | Coordinate columns |
|---|---|---|
| `Map` | 2-D heatmap + coastlines | `latitude`, `longitude` |
| `ZonalProfile` | 2-D pressure × latitude heatmap | `air_pressure`, `latitude` |
| `ZonalLine` | 1-D line (zonal mean) | `latitude` |
| `AnnualMeanTimeSeries` | 1-D line (annual mean) | `time` |
| `AnnualCycle` | 1-D line (monthly climatology) | `month_number` |
| `Scalar` | Horizontal bar chart (0-D) | *(none)* |

When inspecting databases, use `ibis`.

## Architecture

### Core Modules

- `_variable.py` - `Variable`, `Coordinate`, `VerticalLevel`, `NO_LEVEL` dataclasses. Loads metadata from `variables.yml` and `coordinates.yml`. Frozen dataclasses with auto-filled fields (units, standard_name, coordinates, derived flag, etc.)
- `_config.py` - ESMValTool config reset and intake-ESGF data source setup
- `_utils.py` - `get_prepared_cube()` performs preprocessing (unit conversion, longitude normalization, time range extraction, level extraction, spatial masking via landsea/AR6 region masks). Also provides `str_to_class()` for dynamic class loading and `check_coordinates()` to validate cube dimensions

### Diagnostics (`src/climateeval/diags/`)

- `_base.py` - Abstract `Diagnostic` class with `DiagnosticOutput` dataclass: manages reference/other data, computes metrics (weighted_rmse, weighted_pearsonr, weighted_emd), generates Ibis tables. Also provides `DiagnosticOutput.union()` to merge outputs
- `_annual_cycle.py` - `AnnualCycle` diagnostic: evaluates seasonal cycle by comparing monthly means
- `_annual_cycle_spatial_distribution.py` - `AnnualCycleSpatialDistribution` diagnostic: evaluates monthly spatial fields; outputs only metrics (no data table)
- `_diurnal_cycle.py` - `DiurnalCycle` diagnostic: evaluates diurnal cycle via hourly climatology in local solar time
- `_map.py` - `Map` diagnostic: regrid to 2x2, compute spatial mean; output coordinates: latitude, longitude
- `_timeseries.py` - `AnnualMeanTimeSeries` diagnostic: regrid to 2x2, area mean, annual mean; output coordinates: time
- `_zonal_line.py` - `ZonalLine` diagnostic: zonal mean fields; output coordinates: latitude
- `_zonal_profile.py` - `ZonalProfileAirPressure` diagnostic: vertical profiles of humidity/temperature/wind; output coordinates: air_pressure, latitude

### Suites (`src/climateeval/suites/`)

- `_base.py` - `Suite` class: reads a YAML suite definition, builds diagnostics, orchestrates execution, and writes results to an Ibis/DuckDB backend
- `Tier2_atmosphere_monthly.yml` - YAML-defined suite: 19 2-D variables (clivi, clt, clwvi, hfls, hfss, hus400, lwcre, lwp, pr, prw, rlut, rsut, rtnt, swcre, ta200, ta800, tas, ua200, ua800) across five 2-D diagnostics (AnnualCycle, AnnualCycleSpatialDistribution, Map, AnnualMeanTimeSeries, ZonalLine) plus 3 3-D variables (hus, ta, ua) for ZonalProfileAirPressure

### Data Sources (`src/climateeval/data/`)

- `_base.py` - `DataSource` abstract class for data loading/downloading; `DiagnosticData` dataclass pairs a variable with its reference and optional "other" data sources; `DataSourceInformation` dataclass holds provenance metadata (name, category, institute, exp, variant, references, auto-generated id)
- Concrete implementations: `CERESEBAF`, `ERA5`, `ESACCICloud`, `GPCP`, `HadCRUT5`, `MERRA2`

## Data Flow

### Option 1: Use a Suite (recommended)

1. Create a `Suite` instance with a YAML config (e.g., `Suite("Tier2_atmosphere", ...)`)
2. Call `suite.get_database(cubes, data_information, database_resource="duckdb://output.ddb")`
3. Output: Ibis connection with per-diagnostic databases, each containing `raw_output`, `metrics`, `variables`, and `data_sources` tables

### Option 2: Direct Diagnostic usage

1. Define variables with `Variable("id", "var_name", "frequency", **settings)`
2. Create `DiagnosticData` with reference/other data sources
3. Pass dict of `{Variable: DiagnosticData}` to diagnostic constructor
4. Call `diagnostic.get_output(cubes, data_information)` with Iris cubes or Xarray datasets
5. Output: `DiagnosticOutput(raw_output, metrics, variables, data_sources)` as Ibis tables

### Example Usage

See [example.py](example.py) for a complete workflow loading CMIP6 data and computing diagnostics.
See [db.py](db.py) for an example of how to inspect the output database.

## Key Patterns

- Variables with `derived: true` in YAML are derived via ESMValTool's derive module
- `_preprocess()` must be implemented by each diagnostic subclass
- Final coordinates are specified via `_FINAL_COORDINATES` class variable (ensures data has expected dimensions)
- `_OUTPUT_DATA_TABLE` and `_OUTPUT_METRICS_TABLE` class variables control which tables a diagnostic outputs
- Metrics use ESMValTool's `distance_metric` function
- `fail_on_missing_data` diagnostic kwarg controls whether missing data raises errors or logs warnings
- Suites support custom `diagnostic_kwargs` and `variable_kwargs` for customization
- `DataSourceInformation` auto-generates an `id` field from category, name, exp, and variant

## DuckDB Output Schema

Suites output to DuckDB using **attached databases** (one per diagnostic) to organize output.

### Accessing via Ibis

```python
import ibis

conn = ibis.connect("duckdb://climateeval.ddb")

# List diagnostic databases
dbs = conn.list_databases()  # e.g., ['AnnualCycle', 'AnnualCycleSpatialDistribution', 'Map', 'AnnualMeanTimeSeries', 'ZonalLine', 'ZonalProfileAirPressure']

# Access tables within a diagnostic
raw_output = conn.table("raw_output", database="Map")
metrics = conn.table("metrics", database="Map")
variables = conn.table("variables", database="Map")
data_sources = conn.table("data_sources", database="Map")
```

### Database Hierarchy

Each diagnostic is stored as an attached database containing four tables:
`raw_output`, `metrics`, `variables`, and `data_sources`.

```bash
climateeval.ddb
├── AnnualCycle/
│   ├── raw_output      # Monthly climatology (month_number coordinate)
│   ├── metrics         # Comparison metrics
│   ├── variables       # Variable metadata
│   └── data_sources    # Provenance of all datasets used
├── AnnualCycleSpatialDistribution/
│   ├── metrics         # Spatial monthly metrics (no raw_output table)
│   ├── variables
│   └── data_sources
├── Map/
│   ├── raw_output      # Gridded data (latitude, longitude coordinates)
│   ├── metrics
│   ├── variables
│   └── data_sources
├── AnnualMeanTimeSeries/
│   ├── raw_output      # Temporal data (time coordinate, annual means)
│   ├── metrics
│   ├── variables
│   └── data_sources
├── ZonalLine/
│   ├── raw_output      # Zonally-averaged data (latitude coordinate)
│   ├── metrics
│   ├── variables
│   └── data_sources
└── ZonalProfileAirPressure/
    ├── raw_output      # Vertical profile data (air_pressure, latitude)
    ├── metrics
    ├── variables
    └── data_sources
```

The set of diagnostics and variables depends entirely on the suite configuration.
Different runs will produce different databases with different tables and columns.

### `raw_output` Table (per diagnostic)

The data table name is `raw_output` (not `data`). Each diagnostic has its own
coordinate columns. Variable columns are dynamically added based on the suite
configuration.

| Diagnostic                     | Coordinate columns           |
| ------------------------------ | ---------------------------- |
| AnnualCycle                    | `month_number` (float, 1–12) |
| AnnualCycleSpatialDistribution | `month_number`, `lat`, `lon` |
| Map                            | `latitude`, `longitude`      |
| AnnualMeanTimeSeries           | `time` (timestamp)           |
| ZonalLine                      | `latitude`                   |
| ZonalProfileAirPressure        | `air_pressure`, `latitude`   |

Common columns in all `raw_output` tables:

| Column    | Type   | Description                                                              |
| --------- | ------ | ------------------------------------------------------------------------ |
| data_id   | string | Unique identifier of the dataset (e.g., from `DataSourceInformation.id`) |
| data_type | string | One of: `to_benchmark` (model being evaluated), `reference`, `other`     |
| [coords]  | varies | Diagnostic-specific coordinate columns (see above)                       |
| var...    | float  | Variable columns (one per variable ID in the suite, float32)             |

Rows where a variable has no data contain NaN. Superfluous rows (all-NaN across
all variable columns) are dropped.

### `metrics` Table (all diagnostics)

| Column            | Type   | Description                           |
| ----------------- | ------ | ------------------------------------- |
| var_id            | string | Variable identifier                   |
| weighted_rmse     | float  | Weighted root mean square error       |
| weighted_pearsonr | float  | Weighted Pearson correlation          |
| weighted_emd      | float  | Weighted Earth mover's distance       |
| data_id           | string | Dataset this metric applies to        |
| reference_data_id | string | Reference dataset used for comparison |

Metrics are computed for each `(data_id, var_id, reference_data_id)` combination.
Different variables use different reference datasets (e.g., ERA5 for temperature,
CERES-EBAF for radiative fluxes, GPCP for precipitation). Metrics rows also
exist for `other` data sources (e.g., ERA5 as an alternative to GPCP for
precipitation).

### `variables` Table (all diagnostics)

| Column           | Type    | Description                                                                    |
| ---------------- | ------- | ------------------------------------------------------------------------------ |
| id               | string  | Variable identifier (unique within diagnostic)                                 |
| var_name         | string  | CMIP variable name                                                             |
| frequency        | string  | Temporal frequency (e.g., mon)                                                 |
| vertical_level   | string  | Vertical level if applicable (e.g., "air_pressure = 200.0 hPa"); empty if none |
| min_lat          | float   | Minimum latitude of the data                                                   |
| max_lat          | float   | Maximum latitude of the data                                                   |
| min_lon          | float   | Minimum longitude of the data                                                  |
| max_lon          | float   | Maximum longitude of the data                                                  |
| landsea_mask     | string  | Land-sea mask used (space-joined list)                                         |
| ar6_regions_mask | string  | AR6 regions mask used (space-joined list)                                      |
| timerange        | string  | Time range of the data                                                         |
| units            | string  | Units of the variable                                                          |
| anomaly          | string  | Anomaly type (e.g., "full"); empty if not computed                             |
| coordinates      | string  | Required coordinates (space-joined CMIP var names)                             |
| derived          | boolean | Whether the variable is derived via ESMValTool                                 |
| long_name        | string  | Long descriptive name                                                          |
| realm            | string  | CMIP realm (e.g., atmos)                                                       |
| standard_name    | string  | CF standard name                                                               |
| cmip6_table_id   | string  | CMIP6 table ID (e.g., Amon)                                                    |

### `data_sources` Table (all diagnostics)

| Column     | Type   | Description                                           |
| ---------- | ------ | ----------------------------------------------------- |
| id         | string | Unique data source identifier (auto-generated)        |
| name       | string | Dataset name (e.g., ERA5, GPCP, MPI-ESM1-2-LR)        |
| category   | string | Category (e.g., reanalysis, observation, CMIP6 model) |
| institute  | string | Institute (for model runs)                            |
| exp        | string | Experiment (e.g., historical)                         |
| variant    | string | Variant label (e.g., r1i1p1f1)                        |
| references | string | Space-separated reference URLs                        |

This table stores provenance for every dataset used in the diagnostic: the
data being benchmarked, all reference datasets, and all `other` datasets.

## Future Extensibility

The hierarchy supports expansion with:

1. **Additional diagnostics**: New attached database entries (e.g., `TaylorDiagram`)
2. **Additional variables**: New columns added to data tables dynamically
3. **Additional suites**: Different YAML configurations produce their own diagnostic hierarchy
