# Quickstart example

Two tiny scripts that show the full ClimateEval loop: run a suite on a model,
then inspect the results database.

Both assume you are on **DKRZ Levante** (the paths inside point at the shared
CMIP6 archive and a staged reference pool) and that you run them from inside the
pixi environment — `pixi shell`, or prefix each command with `pixi run`.

## [`example.py`](example.py) — run some suites

Loads MPI-ESM1-2-LR CMIP6 output with Iris and runs it through several suites
(`Tier2_atmosphere_monthly`, `Tier1_consistency_checks`, `Tier1_sanity_checks`,
`Tier2_ocean_monthly`, `Tier2_atmosphere_subdaily`), writing one `.ddb` file per
suite:

```bash
python example.py
```

Edit the paths and the `DataSourceInformation(...)` block near the top to point
at your own model. See [`docs/usage.md`](../../docs/usage.md) for the workflow
explained step by step.

## [`db_example.py`](db_example.py) — inspect a database

Opens a `.ddb` file with [ibis](https://ibis-project.org/) and prints every
table in every diagnostic database:

```bash
python db_example.py
```

Change the `db_path` near the top to the database you want to inspect. The
output schema (which tables and columns to expect) is documented in
[`docs/output-schema.md`](../../docs/output-schema.md).
