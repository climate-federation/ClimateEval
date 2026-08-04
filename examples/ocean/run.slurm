#!/bin/bash
#SBATCH --job-name=ocean_diag
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --account=bd1179
#SBATCH --output=ocean_diag_%j.out
#
# Template for running the ocean/sea-ice diagnostics on DKRZ Levante.
# Use the `compute` partition (exclusive node): the `shared` partition's
# thread/PID cgroup limit can make the preprocessing fail with
# "can't start new thread". Edit the account and the paths below.
set -eu

# Cap the BLAS/threading so libraries don't try to size pools to the whole node.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export DASK_NUM_WORKERS=8

export CLIMATEEVAL_DATA_ROOT=/home/b/b384036/ClimateEval/examples/ocean/climateeval_input
export CLIMATEEVAL_RESULTS_DIR=$HOME/ocean_results

# Run inside the pixi environment (adjust if you use a different setup).
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
python run_ocean_diagnostics.py
