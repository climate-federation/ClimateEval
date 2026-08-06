# scripts/

Experimental, author-specific driver scripts used to produce benchmarking
databases during development. They are kept here for reference and
reproducibility, **not** as polished, portable examples.

> **These scripts are not linted or type-checked.** `scripts/` is excluded from
> pre-commit (see `.pre-commit-config.yaml`). They contain hard-coded
> `/scratch` and `/work` paths, SLURM job IDs, and other machine-specific
> details, and are not guaranteed to run outside the author's DKRZ Levante
> setup. For clean, runnable starting points use [`examples/`](../examples)
> instead.

## Contents

| Script | What it drives |
|---|---|
| `showcase_all_tiers.py` | **Showcase** — run *every* tier on MPI-ESM1-2-HR historical vs ERA5 + CMIP6 and build one combined report. Exercises all diagnostics/figure types; fail-soft per tier. For developing/verifying the visualizations. Driven by `submit_showcase.sh`. |
| `submit_showcase.sh` | SLURM job (compute partition) wrapping the showcase. Submit from the repo root: `sbatch scripts/submit_showcase.sh`. |
| `run_cmip6_historical_all_models.py` | Benchmark 50+ CMIP6 historical models through `Tier2_atmosphere_monthly` (one `.ddb` per model). Driven by `submit_cmip6_historical.sh`. |
| `submit_cmip6_historical.sh` | SLURM array job wrapping the script above. Submit from the repository root: `sbatch scripts/submit_cmip6_historical.sh`. |
| `run_cmip6_historical_benchmarks.py` | Alternative CMIP6 historical benchmarking driver. |
| `run_benchmarking_25124195.py` | Benchmark a specific legoESM AMIP run (job 25124195). |
| `run_benchmarking_amip_1yr_iccompare.py` | Compare two 1-year AMIP runs differing only in initial conditions (ERA5-only references). |
| `run_deck_hines_90d_all_tiers.py` | Run a DECK-ERA5-Hines 90-day run across Tier1–Tier3 suites. |
| `run_dynamics_suite.py` | Drive the `Tier3_dynamics` suite. |
| `run_sea_ice_suite.py` | Drive the `Tier2_sea_ice_monthly` suite. |
| `run_mpi_esm1_2_lr.py` | Benchmark MPI-ESM1-2-LR historical output. |
| `run_hovmoller_diag.py` | Exercise the Hovmöller diagnostic. |
| `run_qbo_diag.py` | Exercise the QBO diagnostic. |
| `run_tem_diag.py` | Exercise the TEM (Transformed Eulerian Mean) diagnostic. |
| `ecs_example.py` | Compute effective climate sensitivity (`Tier3_ecs`) from abrupt-4xCO2 / piControl runs. |

For the supported, generalized workflow these are derived from, see
[`docs/benchmarking-legoesm.md`](../docs/benchmarking-legoesm.md).
