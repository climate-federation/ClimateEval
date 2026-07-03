#!/bin/bash
#SBATCH --job-name=legoesm_benchmark
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --output=legoesm_benchmark_%j.out
#
# Template for benchmarking a model with ClimateEval on DKRZ Levante.
# Use the `compute` partition (exclusive node): the `shared` partition's
# thread/PID cgroup limit can make the preprocessing fail with
# "can't start new thread". Edit the account and the paths below.
set -eu

# Cap the BLAS/threading so libraries don't size pools to the whole node.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

export LEGOESM_CMOR_DIR=/path/to/legoesm/cmor/Amon
export CLIMATEEVAL_DATA_ROOT=$HOME/climateeval_input
export CLIMATEEVAL_RESULTS_DIR=$HOME/legoesm_results

# Run inside the pixi environment (adjust if you use a different setup).
cd "$(dirname "$0")"
pixi run python run_legoesm.py
