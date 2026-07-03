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
# Submit from the repository root:  sbatch --account=<acct> scripts/submit_showcase.sh
set -euo pipefail

# Cap threading so libraries don't size pools to the whole node.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

# Batch shells don't source ~/.bashrc, so put the user-level pixi on PATH
# explicitly (pixi is installed under $HOME/.pixi/bin; the repo's .pixi/ only
# holds the solved environments).
export PATH="$HOME/.pixi/bin:$PATH"

WORKDIR="${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p "${WORKDIR}/logs"
cd "${WORKDIR}"

echo "Showcase started on $(hostname) at $(date)"
# --frozen: use the committed lock/envs as-is (no re-solve; compute nodes may
# have no internet).
pixi run --frozen python scripts/showcase_all_tiers.py
echo "Showcase finished at $(date) — see showcase_output/report.html"
