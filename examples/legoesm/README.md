# legoESM benchmarking example

Benchmark output from an external model — one that is **not** in a CMIP archive,
here the legoESM AMIP experiment — against observational references, and produce
a self-contained interactive HTML report.

This is a template for evaluating *any* model whose CMOR-compliant NetCDF files
you can point [Iris](https://scitools-iris.readthedocs.io/) at. For the full
walkthrough (loading cubes, choosing a suite, HPC array jobs, comparing runs)
see [`docs/benchmarking-legoesm.md`](../../docs/benchmarking-legoesm.md).

## What it does

[`run_legoesm.py`](run_legoesm.py):

1. loads the model's CMORized `*.nc` files from `LEGOESM_CMOR_DIR`,
2. runs the `Tier2_atmosphere_monthly` suite against the references in
   `CLIMATEEVAL_DATA_ROOT`, writing one `.ddb` database, and
3. builds `report.html` with `climateeval.report.generate_portable_html()`.

## Run it

From inside the pixi environment (`pixi shell`, or prefix with `pixi run`):

```bash
export LEGOESM_CMOR_DIR=/path/to/legoesm/cmor/Amon
export CLIMATEEVAL_DATA_ROOT=~/climateeval_input
export CLIMATEEVAL_RESULTS_DIR=~/legoesm_results
python run_legoesm.py
```

Outputs land in `CLIMATEEVAL_RESULTS_DIR`:

```
Tier2_atmosphere_monthly.ddb
report.html      # self-contained — open in any browser
```

On DKRZ Levante, submit to the **compute** partition (an exclusive node) rather
than a login node — see [`run.slurm`](run.slurm) for a template (edit the
account and paths).

## Adapt it to your model

- **Paths** — set `LEGOESM_CMOR_DIR` and `CLIMATEEVAL_DATA_ROOT`, or edit the
  constants near the top of `run_legoesm.py`.
- **Provenance** — edit the `DataSourceInformation(...)` block (name, category,
  experiment, variant) recorded in the report.
- **Time range** — `TIMERANGE` is applied to *every* dataset; choose a window
  your model and the references both cover (a reference that doesn't cover the
  start year is dropped).
- **Suite** — swap `SUITE_NAME` for another suite (see
  [`docs/suites.md`](../../docs/suites.md)).

Staging the observational references on Levante works exactly as in the
[ocean example](../ocean/README.md#2-stage-the-observational-references).
