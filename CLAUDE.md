# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmarking diagnostics for Earth System Models (ESMs). The package provides
diagnostics (e.g., Map, TimeSeries) to compare climate model output against
reference observational datasets (e.g., ERA5, GPCP, HadCRUT5, CERES-EBAF). Uses
ESMValTool for data loading/preprocessing and Ibis/DuckDB for output.

## Common Commands

```bash
# Install with development dependencies
pip install -e ".[develop]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/benchmarking --cov-report=html

# Lint
pre-commit run --all

# Run test script and visualize example database
python example.py
python db.py
```

Whenever inspecting databases, use `ibis`.

## Architecture

### Core Modules

- `_variable.py` - `Variable`, `Coordinate`, `VerticalLevel`, `NO_LEVEL` dataclasses. Loads metadata from `variables.yml` and `coordinates.yml`. Frozen dataclasses with auto-filled fields (units, standard_name, coordinates, derived flag, etc.)
- `_config.py` - ESMValTool config reset
- `_utils.py` - `get_prepared_cube()` performs preprocessing (unit conversion, longitude normalization, time range extraction, level extraction). Also provides `check_coordinates()` to validate cube dimensions.

### Diagnostics (`diags/`)

- `_base.py` - Abstract `Diagnostic` class with `DiagnosticOutput` dataclass: manages reference/other data, computes metrics (weighted_rmse, weighted_pearsonr, weighted_emd), generates Ibis tables
- Examples:
  - `_map.py` - `Map` diagnostic: regrid to 10x10, compute spatial mean, output coordinates: latitude, longitude
  - `_timeseries.py` - `TimeSeries` diagnostic: regrid to 2x2, area mean, annual mean, output coordinates: time

### Suites (`suites/`)

- `_base.py` - `Suite` abstract base class: orchestrates diagnostics execution, manages database output (DuckDB via Ibis), handles data loading
- `_atmosphere.py` - `AtmosphereMonthly` suite: pre-configured set of variables (lwcre, pr, rsut, tas, ta200, ta800) with reference/other data sources

### Data Sources (`data/`)

- `_base.py` - `DataSource` abstract class handles data loading/downloading via ESMValTool or intake-esgf. `DiagnosticData` dataclass pairs a variable with its reference and optional "other" data sources
- Example implementations: `HadCRUT5`, `ERA5`, `GPCP`, `CERESEBAF`

### Data Flow

#### Option 1: Use a Suite (recommended)

1. Create a Suite subclass instance (e.g., `AtmosphereMonthly`)
2. Call `suite.run(cubes, "dataset_name", database_path=...)`
3. Output: Ibis connection with `data`, `metrics`, and `variables` tables per diagnostic

#### Option 2: Direct Diagnostic usage

1. Define variables with `Variable("id", "var_name", "frequency", **settings)`
2. Create `DiagnosticData` with reference/other data sources
3. Pass dict of `{Variable: DiagnosticData}` to diagnostic constructor
4. Call `diagnostic.get_output(cubes, "dataset_name")` with Iris cubes
5. Output: `DiagnosticOutput(data_table, metrics_table, variables_table)` as Ibis tables

### Example Usage

See [example.py](example.py) for a complete workflow loading CMIP6 data and computing diagnostics.
See [db.py](db.py) for an example how to work with the output database.

## Key Patterns

- Variables with `derived: true` in YAML are derived via ESMValTool's derive module
- `_preprocess()` must be implemented by each diagnostic subclass
- Final coordinates are specified via `FINAL_COORDINATES` class variable (ensures data has expected dimensions)
- Metrics use ESMValTool's `distance_metric` function
- Output tables use Ibis with DuckDB backend for efficient in-memory analytics
- `fail_on_missing_data` diagnostic kwarg controls whether missing data raises errors or logs warnings
- Suites support custom `diagnostic_kwargs` and `variable_kwargs` for customization

## DuckDB Output Schema

The suites output to DuckDB using a **hierarchy of databases** (attached databases) to organize diagnostics.

### Accessing via Ibis

```python
import ibis

conn = ibis.duckdb.connect('benchmarking.ddb')

# List diagnostic databases
dbs = conn.list_databases()  # e.g., ['Map', 'TimeSeries', ...]

# Access tables within a diagnostic
data = conn.table('data', database='Map')
metrics = conn.table('metrics', database='Map')
variables = conn.table('variables', database='Map')
```

### Database Hierarchy

For example,

```bash
benchmarking.ddb
├── Map/           # Map diagnostic database
│   ├── data       # Spatial data (lat/lon grid)
│   ├── metrics    # Comparison metrics
│   └── variables  # Variable metadata
└── TimeSeries/    # TimeSeries diagnostic database
    ├── data       # Temporal data (time series)
    ├── metrics    # Comparison metrics
    └── variables  # Variable metadata
```

### Table Schemas

#### data table (both diagnostics)

| Column    | Type     | Description                                                 |
| --------- | -------- | ----------------------------------------------------------- |
| Dataset   | string   | Model/dataset name (e.g., MPI-ESM1-2-LR, ERA5)              |
| Data Type | string   | One of: `to_benchmark`, `reference`, `other`                |
| [coords]  | float/ts | Diagnostic-specific coordinates (lat/lon or time)           |
| var...    | float    | Variable columns (e.g., lwcre, pr, rsut, ta200, ta800, tas) |

#### metrics table (both diagnostics)

| Column            | Type   | Description                     |
| ----------------- | ------ | ------------------------------- |
| var_id            | string | Variable identifier             |
| weighted_rmse     | float  | Weighted root mean square error |
| weighted_pearsonr | float  | Weighted Pearson correlation    |
| weighted_emd      | float  | Weighted Earth mover's distance |
| Dataset           | string | Model being evaluated           |
| reference         | string | Reference dataset used          |

#### variables table (both diagnostics)

| Column         | Type    | Description                                    |
| -------------- | ------- | ---------------------------------------------- |
| id             | string  | Variable identifier                            |
| var_name       | string  | CMIP variable name                             |
| frequency      | string  | Temporal frequency (e.g., mon)                 |
| vertical_level | string  | Vertical level if applicable (e.g., 200 hPa)   |
| min_lat        | float   | Minimum latitude of the data                   |
| max_lat        | float   | Maximum latitude of the data                   |
| min_lon        | float   | Minimum longitude of the data                  |
| max_lon        | float   | Maximum longitude of the data                  |
| timerange      | string  | Time range of the data                         |
| units          | string  | Units of the variable                          |
| coordinates    | string  | Required coordinates                           |
| derived        | boolean | Whether the variable is derived via ESMValTool |
| long_name      | string  | Long descriptive name                          |
| realm          | string  | CMIP realm (e.g., atmosphere)                  |
| standard_name  | string  | CF standard name                               |

### Future Extensibility

The hierarchy supports expansion with:

1. **Additional diagnostics**: New database entries (e.g., `ZonalMean`, `TaylorDiagram`)
2. **Additional variables**: New columns added to data tables
3. **Additional suites**: Different suite types create their own diagnostic hierarchy
