# Usage

The core ClimateEval workflow is:

1. **Describe** the dataset you are evaluating with a `DataSourceInformation`.
2. **Load** its output as Iris cubes (or an Xarray dataset).
3. **Run** a suite, which preprocesses the data, compares it against reference
   datasets, and writes a DuckDB database.
4. **Inspect** the database (or build a [report](reporting.md)).

A complete, runnable version of this is
[`examples/quickstart/example.py`](../examples/quickstart/example.py). Run
everything from inside the Pixi environment (`pixi shell`, or prefix with
`pixi run`).

## Quickest path: the `climateeval` CLI

To benchmark a model against ERA5/observations and CMIP6 without writing any
code, use the `climateeval` command (installed with the package; also
available as `python -m climateeval`):

```bash
climateeval benchmark /path/to/model/cmor/Amon
# writes <model>_benchmark/: one .ddb per suite + <model>_report.html
```

The model path may be a single NetCDF file, a flat directory of NetCDF files,
or a CMOR DRS tree (e.g. an ESGF-style `Amon` table directory — the native
grid and latest version are selected per variable, and year-split files are
concatenated automatically). The default suite `Tier2_atmosphere_monthly`
compares against ERA5/GPCP/CERES-EBAF references and CMIP6 historical models.

| Option | Meaning |
|---|---|
| `--name NAME` | Model name shown in the report (default: derived from the CMOR file names). |
| `--suite NAME` | Suite name or YAML path; repeat for several suites. |
| `--timerange A/B` | ISO time range applied to every dataset. |
| `--data-root DIR` | Directory with staged reference datasets. |
| `--download` | Download missing references (ERA5 needs `~/.cdsapirc`). |
| `-o, --out DIR` | Output directory (default `<name>_benchmark/`). |
| `--static` | Also render static PNG figures into `<out>/static_figures/`. |
| `--serve [--port N]` | Open an interactive server instead of writing files. |

`climateeval suites list` shows the shipped suites (name, number of
diagnostics and variables); any of those names — or a path to your own suite
YAML — works with `--suite` (see [Suites](suites.md)).

Reports for existing databases can be served or exported with
`climateeval report DB... [--html FILE | --static-out DIR | --port N]`
(see [reporting](reporting.md)). The root-level
[`benchmark.py`](../benchmark.py) and [`report.py`](../report.py) scripts
remain as thin equivalents for running from a repository checkout.

The rest of this page shows the underlying Python API that the CLI uses.

## Run a suite

```python
import iris
from climateeval.data import DataSourceInformation
from climateeval.suites import Suite

# 1. Provenance recorded in the output database.
data_info = DataSourceInformation(
    name="MPI-ESM1-2-LR",
    category="CMIP6",
    institute="MPI-M",
    exp="historical",
    variant="r1i1p1f1",
    references=(
        "https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.MPI-M.MPI-ESM1-2-LR",
    ),
)

# 2. Load the model output. Any files Iris can read work; one variable per file
#    or many is fine.
cubes = iris.load([
    "/path/to/tas_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc",
    "/path/to/pr_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc",
    # ...
])

# 3. Build and run a suite.
suite = Suite(
    "Tier2_atmosphere_monthly",           # a shipped suite name, or a path to a YAML file
    diagnostic_kwargs={
        "fail_on_missing_data": False,    # skip (vs. error on) missing references
        "download_missing_data": False,   # fetch missing references automatically
        "data_root_dir": "/path/to/climateeval_input",  # where staged references live
    },
    variable_kwargs={
        "timerange": "20000101/20011231", # ISO range applied to every dataset
    },
)

con = suite.get_database(
    cubes,
    data_info,
    database_resource="duckdb://Tier2_atmosphere_monthly.ddb",
)
```

`get_database` returns an [ibis](https://ibis-project.org/) backend connection
and writes the results to `database_resource`.

### `Suite(...)` arguments

| Argument | Meaning |
|---|---|
| `suite_definition` | A shipped suite name (e.g. `"Tier2_atmosphere_monthly"`) or a path to your own suite YAML. |
| `diagnostic_kwargs` | Passed to every diagnostic. Common keys: `data_root_dir`, `fail_on_missing_data`, `download_missing_data`. |
| `variable_kwargs` | Applied to every variable. Most commonly `timerange`. |

See [Suites](suites.md) for the full set of keys and how a suite is defined.

### `get_database(...)` arguments

| Argument | Meaning |
|---|---|
| `data` | The cubes/dataset to evaluate — a `CubeList`, an Xarray `Dataset`, or a dict mapping variable ids to those. |
| `data_information` | The `DataSourceInformation` for the data being benchmarked. |
| `database_resource` | Ibis/DuckDB resource string. `"duckdb://out.ddb"` writes to a file; the default `"duckdb://"` is **in-memory**. |
| `append` | If `True`, append to an existing database instead of replacing it. |

> **Time ranges.** One `timerange` is applied to *every* dataset, model and
> reference alike. Each reference covers a different period, so pick a window
> they all overlap — a reference that doesn't cover the requested start year is
> dropped.

## Inspect the results

The output is a DuckDB file with one attached database per diagnostic, each
holding `raw_output`, `metrics`, `variables` and `data_sources` tables. Open it
with ibis:

```python
import ibis

con = ibis.duckdb.connect("Tier2_atmosphere_monthly.ddb")

con.list_databases()                                  # the diagnostics present
metrics = con.table("metrics", database="map")        # a diagnostic's metrics
print(metrics.to_pandas())
```

[`examples/quickstart/db_example.py`](../examples/quickstart/db_example.py)
walks every table in a database. The full column layout is documented in
[Output schema](output-schema.md).

## Running a diagnostic directly

The suite is the supported entry point. Individual diagnostics
(`climateeval.diags.simple.*`, `climateeval.diags.complex.*`) can also be
constructed and run via their `get_output(data, data_information)` method, but
their constructors are lower-level — see the diagnostic classes in
`src/climateeval/diags/` and the tests in `tests/integration/` for examples.
