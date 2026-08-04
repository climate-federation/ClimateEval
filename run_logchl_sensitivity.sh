#!/bin/bash
#SBATCH --job-name=chl_logsens
#SBATCH --account=bd1179
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --mem=0
#SBATCH --output=logsens_%j.out
#SBATCH --error=logsens_%j.err
# ---------------------------------------------------------------
# Chl-a log-transform sensitivity test for the pointwise sea-ice
# diagnostics (Figs 10 and 11).
#
# Runs the seaice stage TWICE from an identical code path:
#   pass 1  CHL_LOG_ANOM=0  -> paper_figures/         (linear baseline)
#   pass 2  CHL_LOG_ANOM=1  -> paper_figures_logchl/  (log10 Chl-a)
#
# Figs 1-9 are unaffected by CHL_LOG_ANOM and are not regenerated.
# ---------------------------------------------------------------
set -euo pipefail

SCRIPT=example_chl_nodatabse8.py
STAGES=seaice

echo "=== pass 1/2: linear Chl-a (baseline) ==="
CHL_STAGES=$STAGES CHL_LOG_ANOM=0 python "$SCRIPT" 2>&1 | tee run_linear.log

echo "=== pass 2/2: log10 Chl-a (sensitivity) ==="
CHL_STAGES=$STAGES CHL_LOG_ANOM=1 python "$SCRIPT" 2>&1 | tee run_log.log

echo
echo "=== REDUNDANCY CHECK: linear vs log ==="
printf '%s\n' "--- linear ---"
grep "REDUNDANCY CHECK" run_linear.log || echo "(none found)"
printf '%s\n' "--- log10 ---"
grep "REDUNDANCY CHECK" run_log.log || echo "(none found)"

echo
echo "Figures:"
echo "  baseline : $(pwd)/paper_figures/"
echo "  log10    : $(pwd)/paper_figures_logchl/"
