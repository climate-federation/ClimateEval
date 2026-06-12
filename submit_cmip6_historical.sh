#!/bin/bash
#SBATCH --job-name=climateeval_cmip6
#SBATCH --account=bd1083
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-51
#SBATCH --output=/work/bd1083/b309178/diffESM/ClimateEval/logs/cmip6_historical_%A_%a.out
#SBATCH --error=/work/bd1083/b309178/diffESM/ClimateEval/logs/cmip6_historical_%A_%a.err

set -euo pipefail

WORKDIR=/work/bd1083/b309178/diffESM/ClimateEval
mkdir -p "${WORKDIR}/logs"

cd "${WORKDIR}"

eval "$("${WORKDIR}/.pixi/bin/pixi" shell-hook)"

echo "Job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"

pixi run python run_cmip6_historical_all_models.py --model-index "${SLURM_ARRAY_TASK_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
