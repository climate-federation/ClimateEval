# Data sources

A **data source** is a class that knows how to load (and optionally download) a
reference or model dataset in a CMOR-compliant form. In a [suite](suites.md),
`reference_data` and `other_data` name these classes; ClimateEval loads them,
preprocesses them the same way as the model, and compares.

## Shipped data sources

All live in `climateeval.data`:

| Category | Classes |
|---|---|
| Reanalysis | `ERA5Monthly`, `ERA5Hourly`, `MERRA2` |
| Radiation / clouds | `CERESEBAF`, `ESACCICloud` |
| Precipitation | `GPCP` |
| Surface temperature | `HadCRUT5`, `GISTEMP`, `BerkeleyEarth`, `NOAAGlobalTemp`, `HadISST` |
| Land climate | `CRU` |
| Sea-surface temperature | `ESACCISST`, `NOAAERSSTv5` |
| Ocean | `ORAS5`, `EN4`, `WOA`, `IAP`, `ESACCIOC` |
| AMOC | `RAPID` |
| Sea ice | `OSI450NH`, `OSI450SH`, `NSIDCG02202SH` |
| CMIP6 model data | `CMIP6HistoricalR1I1P1F1`, `CMIP6Abrupt4xCO2R1I1P1F1`, `CMIP6PiControlR1I1P1F1` |
| Generic loaders | `ESMValToolIntakeDataSource`, `ESMValToolCMORizerDataSource` |

`CMIP6*` and `ESMValToolIntake*` also have *generator* variants
(`CMIP6DataSourceGenerator`, `ESMValToolIntakeDataSourceGenerator`) that build a
data source on demand.

> Data source *generators* do not yet work with `derived: true` variables
> ([ESMValCore#2769](https://github.com/ESMValGroup/ESMValCore/issues/2769)).

The gridded surface temperature analyses differ in what they provide, which
matters when a suite needs an absolute field rather than an anomaly:

| Data source | Variables | Coverage |
|---|---|---|
| `HadCRUT5` | `tas`, `tasa` | global (land + ocean) |
| `GISTEMP` | `tasa` | global (land + ocean) |
| `BerkeleyEarth` | `tas`, `tasa` | global (land + ocean) |
| `NOAAGlobalTemp` | `tasa` | global (land + ocean) |
| `CRU` | `tas`, `tasmax`, `tasmin`, `pr`, `clt` | land only |

## Where references come from

- **Staged CMORized references.** Most references are read from a directory you
  point ClimateEval at via the `data_root_dir` diagnostic kwarg. On DKRZ Levante
  they are CMORized in the shared ESMValTool OBS pool; the ocean example's
  [`stage_obs_refs.sh`](../examples/ocean/stage_obs_refs.sh) symlinks them into
  a directory of your choice.
- **On-the-fly download.** With `download_missing_data=True`, missing references
  are fetched (ERA5 and other CDS data via `cdsapi`, ESGF data via
  `intake-esgf`). `cdsapi` needs a `~/.cdsapirc` with your CDS API key.
- **Direct paths.** A few sources read raw files directly (see the environment
  variables below).

## Provenance: `DataSourceInformation`

Every dataset — the model being benchmarked and each reference — is recorded in
the output with a `DataSourceInformation`:

```python
from climateeval.data import DataSourceInformation

info = DataSourceInformation(
    name="MPI-ESM1-2-LR",
    category="CMIP6",
    institute="MPI-M",     # optional
    exp="historical",      # optional
    variant="r1i1p1f1",    # optional
    references=("https://…",),  # optional
)
```

An `id` is generated automatically from the non-empty `category`, `name`, `exp`
and `variant` (joined with underscores), e.g. `CMIP6_MPI-ESM1-2-LR_historical_r1i1p1f1`.
These fields populate the `data_sources` table (see [Output schema](output-schema.md)).

## Environment variables

Read by the library:

| Variable | Used by | Purpose |
|---|---|---|
| `CLIMATEEVAL_VARIABLES_FILE` | `_variable.py` | Override the packaged `variables.yml`. |
| `CLIMATEEVAL_COORDINATES_FILE` | `_variable.py` | Override the packaged `coordinates.yml`. |
| `CLIMATEEVAL_ORAS5_RAW_DIR` | `climateeval.data.ORAS5` | Directory of raw ORAS5 NEMO files. |
| `CLIMATEEVAL_ORAS5_GRID_DIR` | `climateeval.data.ORAS5` | Directory of ORAS5 grid files. |
| `CLIMATEEVAL_RAPID_FILE` | `climateeval.data.RAPID` | Path to the RAPID AMOC observations file. |

Conventions used by the example scripts (not the library itself):
`CLIMATEEVAL_DATA_ROOT` (feeds `data_root_dir`) and `CLIMATEEVAL_RESULTS_DIR`
(output directory). See the [ocean example](../examples/ocean/README.md) for the
Levante defaults.

## Adding a custom data source

Subclass `climateeval.data.DataSource` (or `ComplexDataSource`) and implement
the loading/downloading hooks, following an existing implementation in
[`src/climateeval/data/`](../src/climateeval/data/) (e.g. `_gpcp.py`,
`_era5.py`) as a template. Then reference it by its fully-qualified class path in
a suite's `reference_data` / `other_data`.
