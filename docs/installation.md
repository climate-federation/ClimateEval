# Installation

## Requirements

- **Python 3.12 or 3.13**.
- **[Pixi](https://pixi.prefix.dev/)** — ClimateEval is distributed as a Pixi
  project and its dependencies come from `conda-forge`. Supported platforms:
  `linux-64` and `osx-arm64`.
- A fair amount of memory and disk: ClimateEval builds on **ESMValTool /
  ESMValCore ≥ 2.14**, which pulls in the full geoscience stack (Iris, ESMPy,
  Cartopy, …). Preprocessing regrids and masks data on the fly.

Key runtime dependencies (installed automatically by Pixi):
`esmvalcore` / `esmvaltool`, `scitools-iris`, `ibis-framework` (with the DuckDB
backend), `intake-esgf`, `xarray`, `pandas ≥ 3`, `plotly`, `cdsapi`,
`cf-units`, `ncdata`, `loguru`, `pyyaml`.

ClimateEval is developed and run on **DKRZ Levante** (HPC), where the
observational references are already staged, but it installs anywhere Pixi and
conda-forge work.

## Install from source

```bash
git clone https://github.com/climate-federation/ClimateEval.git
cd ClimateEval
pixi install            # resolve and install everything
# or: pixi install --frozen   # faster; uses the committed pixi.lock as-is
```

Then work inside the Pixi environment — either open a shell in it:

```bash
pixi shell              # or: pixi shell --frozen
python examples/quickstart/example.py
```

or prefix individual commands with `pixi run`:

```bash
pixi run python report.py output.ddb
```

### Environments

The Pixi project defines several environments (feature combinations):

| Environment | Purpose |
|---|---|
| `default` | ESMValTool release + dev + test tools. What you normally use. |
| `esmvaltool-dev` | ESMValTool / ESMValCore installed from their `main` git branches. |
| `test-py312` | ESMValTool release pinned to Python 3.12. |
| `test-py313` | ESMValTool release pinned to Python 3.13. |

Select a non-default environment with `-e`:

```bash
pixi run -e esmvaltool-dev python examples/quickstart/example.py
pixi shell -e test-py313
```

## Verify the install

```bash
pixi run python -c "from climateeval.suites import Suite; print('ClimateEval OK')"
```

To run the test suite, see [Contributing](contributing.md#tests).

## HPC notes (DKRZ Levante)

- Submit real runs to the **`compute`** partition (an exclusive node), not a
  login node. The `shared` partition's thread/PID cgroup limit can make
  preprocessing fail with `can't start new thread`.
- Cap the BLAS/threading pools so libraries don't size themselves to the whole
  node:
  ```bash
  export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
  ```
- SLURM templates: [`examples/ocean/run.slurm`](../examples/ocean/run.slurm),
  [`examples/legoesm/run.slurm`](../examples/legoesm/run.slurm), and
  [`scripts/submit_cmip6_historical.sh`](../scripts/submit_cmip6_historical.sh)
  (a job array).

## Troubleshooting

- **Missing reference data.** Diagnostics compare against observational
  references. Point ClimateEval at your staged references with the
  `data_root_dir` diagnostic kwarg (see [Data sources](data-sources.md)), or set
  `download_missing_data=True` to fetch what's missing. Use
  `fail_on_missing_data=False` to skip a missing reference instead of erroring.
- **Downloads need credentials.** ERA5 and other CDS-hosted data are fetched via
  `cdsapi`, which needs a `~/.cdsapirc` with your CDS API key. ESGF data is
  discovered/fetched via `intake-esgf`.
- **`can't start new thread`.** You are on a thread-limited partition — use the
  `compute` partition and cap the thread env vars (see HPC notes above).
- **Derived variables + data source generators.** Data source *generators* do
  not yet work with `derived: true` variables
  ([ESMValCore#2769](https://github.com/ESMValGroup/ESMValCore/issues/2769)).
- **No ESMValTool config needed.** ClimateEval sets up its own ESMValCore
  configuration at runtime; you do not need an `~/.config/esmvaltool` file.
