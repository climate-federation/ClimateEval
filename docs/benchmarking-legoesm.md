# Benchmarking your own model (legoESM)

ClimateEval isn't limited to CMIP archive models — you can benchmark any model
whose output you can CMORize and load with Iris. This page uses the legoESM AMIP
experiment as the running example; the steps are the same for any external
model.

A ready-to-edit version of everything below is
[`examples/legoesm/run_legoesm.py`](../examples/legoesm/run_legoesm.py).

## The workflow

### 1. CMORize / load your model output

ClimateEval expects CMOR-style NetCDF — the exact requirements (variable
names, coordinates, units) and how to convert non-CMOR output are documented
in the [input data spec](input-data-spec.md). Validate your files with

```bash
climateeval check /path/to/legoesm/cmor/Amon --timerange 19790101/19790930
```

and fix reported FAILs until it exits 0. Then point Iris at the files:

```python
import iris
from pathlib import Path

model_paths = sorted(Path("/path/to/legoesm/cmor/Amon").glob("*.nc"))
cubes = iris.load([str(p) for p in model_paths])
```

If your files come in per-decade chunks, concatenate them first:

```python
from iris.util import equalise_attributes, unify_time_units

equalise_attributes(cubes)
unify_time_units(cubes)
cubes = cubes.concatenate()
```

### 2. Describe the model

```python
from climateeval.data import DataSourceInformation

model_info = DataSourceInformation(
    name="legoESM-1-0",
    category="legoESM",   # free-form; groups the dataset in the report
    institute="MPI-M",
    exp="amip",
    variant="r1i1p1f1",
)
```

### 3. Run a suite

```python
from climateeval.suites import Suite

suite = Suite(
    "Tier2_atmosphere_monthly",
    diagnostic_kwargs={
        "data_root_dir": "/path/to/climateeval_input",  # staged references
        "fail_on_missing_data": False,
        "download_missing_data": False,
    },
    variable_kwargs={"timerange": "19790101/19790930"},  # your run's period
)
suite.get_database(cubes, model_info, database_resource="duckdb://legoesm_amip.ddb")
```

Pick a `timerange` your model **and** the references both cover — it is applied
to every dataset, and a reference that doesn't reach the start year is dropped.

### 4. Build the report

```python
from climateeval.report import generate_portable_html

html = generate_portable_html(["legoesm_amip.ddb"], offline=True, model_name="legoESM-1-0")
open("report.html", "w").write(html)
```

or serve it interactively with `python report.py legoesm_amip.ddb --model legoESM-1-0`.
See [Reporting](reporting.md).

## Observations-only

By default suites like `Tier2_atmosphere_monthly` include CMIP6 models as
`other_data` for context. To benchmark **only against observations** — useful
when comparing model configurations to each other — strip the `other_data`
entries and force an observational reference. Load the suite YAML, edit it in
memory, and write a temporary file:

```python
import tempfile, yaml
from pathlib import Path
from importlib import resources

def observations_only(suite_name: str) -> Path:
    suite_dir = Path(str(resources.files("climateeval.suites")))
    suite_def = yaml.safe_load((suite_dir / f"{suite_name}.yml").read_text())
    for diag in suite_def:
        for var in diag.get("variables", []):
            if isinstance(var, dict):
                var.pop("other_data", None)
    tmp = Path(tempfile.mkstemp(suffix=".yml")[1])
    tmp.write_text(yaml.dump(suite_def))
    return tmp

suite = Suite(observations_only("Tier2_atmosphere_monthly"), variable_kwargs={...})
```

## Comparing runs

To compare two runs (e.g. an initial-condition sensitivity test), run each into
its **own** database, then pass all of them to one report:

```python
db_paths = [run_suite(tag) for tag in ("std", "era5")]   # one .ddb each
serve(db_paths, model_name="legoESM-1-0")
```

The report shows every dataset side by side and ranks them on the leaderboard.

## Running on HPC

For many models or long runs, drive ClimateEval from a SLURM job. A single-run
template is [`examples/legoesm/run.slurm`](../examples/legoesm/run.slurm).

For a **job array** — one model per array task — see
[`scripts/submit_cmip6_historical.sh`](../scripts/submit_cmip6_historical.sh)
with [`scripts/run_cmip6_historical_all_models.py`](../scripts/run_cmip6_historical_all_models.py),
which benchmarks 50+ CMIP6 historical models by index:

```bash
sbatch scripts/submit_cmip6_historical.sh   # #SBATCH --array=0-51
```

Submit it from the repository root (so `.pixi` resolves). The pattern
generalises to any list of runs — one `.ddb` per task, then a single report over
all of them.

> The `scripts/` drivers are experimental and contain machine-specific paths;
> treat them as references, not turnkey tools. See
> [`scripts/README.md`](../scripts/README.md).
