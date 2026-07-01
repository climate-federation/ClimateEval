#!/bin/bash
#SBATCH --job-name=climateeval_cmip6
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-51
#SBATCH --output=logs/cmip6_historical_%A_%a.out
#SBATCH --error=logs/cmip6_historical_%A_%a.err

set -euo pipefail

# Submit from the repository root so WORKDIR (and thus .pixi) resolves there:
#   sbatch scripts/submit_cmip6_historical.sh
WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p "${WORKDIR}/logs"

cd "${WORKDIR}"

eval "$("${WORKDIR}/.pixi/bin/pixi" shell-hook)"

echo "Job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID} started on $(hostname) at $(date)"

pixi run python scripts/run_cmip6_historical_all_models.py --model-index "${SLURM_ARRAY_TASK_ID}"

echo "Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
