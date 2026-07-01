#!/bin/bash
#SBATCH --job-name=climateeval_showcase
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=logs/showcase_%j.out
#SBATCH --error=logs/showcase_%j.err
#
# Run the all-tiers showcase for MPI-ESM1-2-HR on DKRZ Levante.
# Submit from the repository root:  sbatch scripts/submit_showcase.sh
set -euo pipefail

# Cap threading so libraries don't size pools to the whole node.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"

eval "$("${WORKDIR}/.pixi/bin/pixi" shell-hook)"

echo "Showcase started on $(hostname) at $(date)"
pixi run python scripts/showcase_all_tiers.py
echo "Showcase finished at $(date) — see showcase_output/report.html"
