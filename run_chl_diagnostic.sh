#!/bin/bash
#SBATCH --job-name=chl_diagnostic
#SBATCH --account=bd1179
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/home/b/b384036/ClimateEval/logs/chl_diagnostic_%j.out
#SBATCH --error=/home/b/b384036/ClimateEval/logs/chl_diagnostic_%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=navarra@uni-bremen.de

# ============================================================
# run_chl_diagnostic.sh
#
# Runs example_chl_nodatabse5.py on Levante (DKRZ).
# Produces paper_figures/fig5c_siconc_bias_combined.png
# (and all other paper figures).
#
# Submit with:
#   mkdir -p logs
#   sbatch run_chl_diagnostic.sh
#
# To run interactively for testing (Phase E only):
#   salloc --account=bd1179 --partition=compute \
#          --nodes=1 --cpus-per-task=8 --mem=64G --time=01:00:00
#   bash run_chl_diagnostic.sh
# ============================================================

set -euo pipefail

# ---- Directories ---------------------------------------------------
SCRIPT_DIR="/home/b/b384036/ClimateEval"  # hardcoded: BASH_SOURCE[0] unreliable in SLURM batch jobs
SCRIPT="${SCRIPT_DIR}/example_chl_nodatabse5.py"
OUT_DIR="${SCRIPT_DIR}/paper_figures"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

echo "============================================================"
echo "Job ID       : ${SLURM_JOB_ID:-interactive}"
echo "Node         : $(hostname)"
echo "Start        : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Script       : ${SCRIPT}"
echo "Output dir   : ${OUT_DIR}"
echo "CPUs         : ${SLURM_CPUS_PER_TASK:-8}"
echo "Memory       : ${SLURM_MEM_PER_NODE:-128G}"
echo "============================================================"

echo "Python       : $(which python)"
echo "Python ver   : $(python --version)"

# ---- Verify key input paths exist ----------------------------------
echo ""
echo "--- Checking input paths ---"

check_path() {
    if [ -e "$1" ]; then
        echo "  OK  $1"
    else
        echo "  MISSING  $1"
        # Do not exit — the script handles missing data gracefully
        # with per-model warnings. Only abort for the obs reference.
        if [ "$2" = "required" ]; then
            echo "ERROR: required path missing, aborting."
            exit 1
        fi
    fi
}

# Observational references (required)
check_path "/work/bd0854/DATA/ESMValTool2/OBS/Tier2/ESACCI-OC" required
check_path "/work/bd0854/DATA/ESMValTool2/OBS/Tier2/HadISST/OBS6_HadISST_reanaly_1_SImon_siconc_187001-202112.nc" required
check_path "/work/bd0854/DATA/ESMValTool2/OBS/Tier2/NOAA-ERSSTv3b/OBS6_NOAA-ERSSTv3b_reanaly_v3b_Omon_tos_185401-202001.nc" required

# CMIP6 pool (optional per model — script skips missing ones)
check_path "/work/ik1017/CMIP6/data/CMIP6/CMIP" optional
check_path "/work/bd1179/b384036/climateeval_input" optional

echo ""

# ---- OMP / threading limits ----------------------------------------
# xarray/numpy can spawn many threads; cap them to the allocated CPUs
# to avoid oversubscription on a shared node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ---- Run -----------------------------------------------------------
echo "--- Starting diagnostic ---"
cd "${SCRIPT_DIR}"

python "${SCRIPT}" 2>&1 | tee "${LOG_DIR}/chl_diagnostic_${SLURM_JOB_ID:-interactive}.log"

EXIT_CODE=${PIPESTATUS[0]}

# ---- Summary -------------------------------------------------------
echo ""
echo "============================================================"
echo "End          : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Exit code    : ${EXIT_CODE}"
echo ""
echo "--- Figures written ---"
if [ -d "${OUT_DIR}" ]; then
    ls -lh "${OUT_DIR}"/fig*.png 2>/dev/null || echo "  (none found)"
fi
echo ""

# Highlight the new combined sea-ice figure specifically
COMBINED="${OUT_DIR}/fig5c_siconc_bias_combined.png"
if [ -f "${COMBINED}" ]; then
    echo "  NEW FIGURE OK: ${COMBINED}"
else
    echo "  WARNING: fig5c_siconc_bias_combined.png not produced"
    echo "  Check logs for 'Sea-ice bias maps' section"
fi

echo "============================================================"
exit ${EXIT_CODE}
