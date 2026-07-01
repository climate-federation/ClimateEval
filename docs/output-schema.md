# Output schema

A suite writes its results to DuckDB using **attached databases**: one database
per diagnostic, each with up to four tables. The exact set of databases,
columns and rows depends entirely on the suite — different suites and different
model data produce different output.

## Layout

Each diagnostic becomes an attached database **named after its `name:` field in
the suite YAML** (not the diagnostic class). For example,
[`Tier2_atmosphere_monthly`](../src/climateeval/suites/Tier2_atmosphere_monthly.yml)
produces:

```
Tier2_atmosphere_monthly.ddb
├── annual_cycle/            (AnnualCycle)
│   ├── raw_output
│   ├── metrics
│   ├── variables
│   └── data_sources
├── performance_metrics/     (AnnualCycleSpatialDistribution — metrics only, no raw_output)
│   ├── metrics
│   ├── variables
│   └── data_sources
├── annual_mean_timeseries/  (AnnualMeanTimeSeries)
├── map/                     (Map)
├── zonal_line/              (ZonalLine)
└── zonal_profile/           (ZonalProfileAirPressure)
```

A table is omitted when it would be empty (e.g. diagnostics that emit only
metrics have no `raw_output`).

## Accessing via ibis

```python
import ibis

con = ibis.duckdb.connect("Tier2_atmosphere_monthly.ddb")

con.list_databases()                              # ['annual_cycle', 'map', ...]
raw = con.table("raw_output", database="map")     # a diagnostic's data table
metrics = con.table("metrics", database="map")
print(metrics.to_pandas())
```

## `raw_output` (per diagnostic)

The data table. Coordinate columns depend on the diagnostic; variable columns
are added dynamically, one per variable `id` in the suite (`float32`).

Common columns:

| Column | Type | Description |
|---|---|---|
| `data_id` | string | Dataset identifier (from `DataSourceInformation.id`). |
| `data_type` | string | `to_benchmark` (the model), `reference`, or `other`. |
| *(coords)* | varies | Diagnostic-specific coordinate columns (below). |
| *(vars)* | float | One column per variable `id`; `NaN` where a variable has no data. |

Coordinate columns by diagnostic:

| Diagnostic | Coordinate columns |
|---|---|
| `AnnualCycle` | `month_number` (1–12) |
| `AnnualCycleSpatialDistribution` | `month_number`, `lat`, `lon` |
| `Map` | `latitude`, `longitude` |
| `AnnualMeanTimeSeries` | `time` |
| `ZonalLine` | `latitude` |
| `ZonalProfileAirPressure` | `air_pressure`, `latitude` |

Rows that are all-`NaN` across every variable column are dropped.

## `metrics` (all diagnostics)

| Column | Type | Description |
|---|---|---|
| `var_id` | string | Variable identifier. |
| `weighted_rmse` | float | Weighted root-mean-square error. |
| `weighted_pearsonr` | float | Weighted Pearson correlation. |
| `weighted_emd` | float | Weighted Earth mover's distance. |
| `data_id` | string | Dataset this metric applies to. |
| `reference_data_id` | string | Reference dataset used for the comparison. |

One row per `(data_id, var_id, reference_data_id)`. Different variables use
different references (e.g. ERA5 for temperature, CERES-EBAF for radiation, GPCP
for precipitation), and `other` datasets get metric rows too.

## `variables` (all diagnostics)

Metadata for each variable in the diagnostic: `id`, `var_name`, `frequency`,
`vertical_level`, spatial bounds (`min_lat`/`max_lat`/`min_lon`/`max_lon`),
`landsea_mask`, `ar6_regions_mask`, `timerange`, `units`, `anomaly`,
`coordinates`, `derived`, `long_name`, `realm`, `standard_name`,
`cmip6_table_id`. See [Variables & coordinates](variables.md) for the source
metadata.

## `data_sources` (all diagnostics)

Provenance for every dataset used — the benchmarked model plus all reference and
`other` datasets:

| Column | Type | Description |
|---|---|---|
| `id` | string | Auto-generated identifier. |
| `name` | string | Dataset name (e.g. `ERA5`, `GPCP`, `MPI-ESM1-2-LR`). |
| `category` | string | e.g. `reanalysis`, `observation`, `CMIP6`. |
| `institute` | string | Institute (for model runs). |
| `exp` | string | Experiment (e.g. `historical`). |
| `variant` | string | Variant label (e.g. `r1i1p1f1`). |
| `references` | string | Space-separated reference URLs. |
