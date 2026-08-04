#!/bin/bash
#SBATCH --job-name=chl_diag
#SBATCH --account=bd1179
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --output=chl_paper_%j.out
#SBATCH --error=chl_paper_%j.err
#SBATCH --mail-type=END,FAIL
#
# Runs the manuscript's seven CMIP6 models through ClimateEval.
#
# PARTITION: compute --exclusive, NOT shared. The ocean example's own
# run.slurm carries this warning and it is the most likely explanation
# for the missing ocean_sst.ddb in examples/ocean/results/: the shared
# partition's thread and PID cgroup limit makes the preprocessing fail
# with "can't start new thread", and the sea-ice and OHC runs happen to
# be light enough to survive it while the chlorophyll run is not.
#
# USAGE
#   sbatch run_chl_paper.sh
#   CHL_MODELS=CESM2 sbatch --export=ALL,CHL_MODELS run_chl_paper.sh
#   CHL_PHASES=all   sbatch --export=ALL,CHL_PHASES run_chl_paper.sh

# NOT set -e. The previous version had it, and everything before the
# first echo - the cd, the micromamba lookup, the shell hook, the
# activation - was therefore able to kill the job with no output at all:
# an empty .err and nothing but the DKRZ epilogue in .out. Each step is
# now checked explicitly and says what it is doing.
set -u
set -o pipefail

echo "============================================================"
echo "job     : ${SLURM_JOB_ID:-interactive}"
echo "node    : $(hostname)"
echo "started : $(date)"
echo "script  : ${BASH_SOURCE[0]}"
echo "workdir : $(pwd)"
echo "partition (requested compute; Slurm reports): ${SLURM_JOB_PARTITION:-?}"
echo "============================================================"

# Cap BLAS threading so the libraries do not size their pools to the whole
# exclusive node.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# Which figure phases ChlBGCEvaluation runs. "AB" (the default set in
# the diagnostic) is the manuscript's first block: mean state and bias,
# the interannual anomalies, the seasonal cycle and the driver bias
# maps. "all" adds the lag-correlation and sea-ice memory phases, which
# re-load and regrid every model's full monthly record.
export CHL_PHASES=${CHL_PHASES:-AB}

# Observational sampling mask.
#
# Without it the bias maps compare a model annual mean over all twelve
# months against an ESA-CCI annual mean over only the months the
# satellite could see - and that count falls in discrete steps with
# latitude, which puts sharp horizontal edges into the high-latitude
# bias where no physical step exists. The same mismatch biases the
# seasonal cycle and every correlation.
#
# With it, each model is blanked wherever the retrieval is missing in
# that month, BEFORE the climatology is removed, so climatology, trend
# and anomalies all rest on the same calendar sampling. Output goes to
# paper_figures_maskedobs/ so a masked run cannot be mistaken for an
# unmasked one. Set CHL_MASK_TO_OBS=0 for the unmasked comparison.
export CHL_MASK_TO_OBS=${CHL_MASK_TO_OBS:-1}

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

export CLIMATEEVAL_DATA_ROOT=${CLIMATEEVAL_DATA_ROOT:-\
/home/b/b384036/ClimateEval/examples/ocean/climateeval_input}
export CLIMATEEVAL_RESULTS_DIR=${CLIMATEEVAL_RESULTS_DIR:-\
/home/b/b384036/ClimateEval/examples/ocean/results_chl_paper}

# SLURM_SUBMIT_DIR, not dirname. Slurm copies the batch script to the
# node's spool directory before running it, so ${BASH_SOURCE[0]} resolves
# to /var/spool/slurmd/job<N> and the driver next to the original script
# is nowhere near it. The dirname form only works when the script is run
# directly, which is why it is kept as the fallback.
HERE=${CHL_DIR:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}} || {
    echo "FATAL: cannot resolve script directory"; exit 1; }
cd "$HERE" || { echo "FATAL: cannot cd to $HERE"; exit 1; }
echo "--- running in $HERE ---"
[ -r run_chl_paper.py ] || {
    echo "FATAL: run_chl_paper.py not found in $HERE"
    echo "       (SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset})"
    echo "       submit from the directory holding run_chl_paper.py,"
    echo "       or set CHL_DIR to point at it"
    exit 1; }

# The ocean example uses `pixi run`, but pixi has no workspace manifest on
# this account and its solve fails on the esmvaltool-dev environment.
# climateeval is installed in the micromamba env instead, so activate that
# directly; set CHL_ENV to override.
ENVNAME=${CHL_ENV:-$HOME/micromamba/envs/climateval}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}

echo "--- environment ---"
echo "wanted env : $ENVNAME"
[ -d "$ENVNAME" ] || { echo "FATAL: env directory does not exist"; exit 1; }

# Run through `micromamba run -p`, not through shell activation and not
# by putting bin/ on PATH.
#
#   * The shell hook emits code that trips `set -u` and kills the job
#     silently - that was the earlier failure.
#   * A bare PATH prepend runs the right python but never executes the
#     environment's activate.d scripts (proj4, libarrow, libglib), so
#     PROJ_DATA and GDAL_DATA stay unset and cartopy/pyproj fail to
#     initialise. That was the "climateeval will not import" failure,
#     which is why the same import succeeds in an interactive shell
#     where activation has already run.
#
# `micromamba run` sets the whole environment, activation scripts
# included, for one command, with no shell integration at all.
MAMBA_EXE=${MAMBA_EXE:-$HOME/.local/bin/micromamba}
[ -x "$MAMBA_EXE" ] || MAMBA_EXE=$(command -v micromamba || true)
[ -x "$MAMBA_EXE" ] || { echo "FATAL: micromamba not found"; exit 1; }
echo "micromamba : $MAMBA_EXE"

RUN="$MAMBA_EXE run -p $ENVNAME"

echo "python     : $($RUN which python 2>/dev/null || echo NONE)"

echo "============================================================"
echo "job     : ${SLURM_JOB_ID:-interactive}"
echo "node    : $(hostname)"
echo "started : $(date)"
echo "env     : ${CONDA_PREFIX:-unset}"
echo "data    : $CLIMATEEVAL_DATA_ROOT"
echo "results : $CLIMATEEVAL_RESULTS_DIR"
echo "phases  : $CHL_PHASES  (all = include lag/sea-ice phases)"
echo "obs mask: $CHL_MASK_TO_OBS  (1 = models masked to the ESA-CCI sampling)"
echo "============================================================"

# Report WHY the import fails, not just that it did. The previous form
# printed a one-line FATAL and discarded the traceback, which is the
# only part that says anything useful.
$RUN python - <<'PYEOF'
import sys, pathlib, traceback
print("python     :", sys.executable)
print("version    :", sys.version.split()[0])
try:
    import climateeval
    print("climateeval:", pathlib.Path(climateeval.__file__).parent)
except Exception:
    print("--- climateeval import FAILED ---")
    traceback.print_exc(file=sys.stdout)
    print("--- sys.path ---")
    for entry in sys.path:
        print("   ", entry)
    sys.exit(1)
PYEOF
IMPORT_RC=$?
if [ $IMPORT_RC -ne 0 ]; then
    echo "FATAL: climateeval will not import (see traceback above)"
    echo "       env      : $ENVNAME"
    echo "       repo     : /home/b/b384036/ClimateEval"
    echo "       If the package is an editable install pointing at the"
    echo "       repo, the repo must be importable from here too."
    exit 1
fi

START=$(date +%s)
set +e
$RUN python run_chl_paper.py
RC=$?
END=$(date +%s)
# errexit deliberately stays off: the summary below must run even when
# the driver failed, which is the case where it is most informative.

echo "============================================================"
echo "exit code : $RC"
echo "elapsed   : $(( (END-START)/60 )) min $(( (END-START)%60 )) s"

# The driver deletes each database before writing it, so a file present
# but empty means that run failed after the delete. Check the row counts
# rather than trusting the file listing.
echo "--- databases ---"
$RUN python - <<'PY'
import os, pathlib
try:
    import ibis
except ImportError:
    print("  ibis not available; skipping the check"); raise SystemExit
out = pathlib.Path(os.environ["CLIMATEEVAL_RESULTS_DIR"])
files = sorted(out.glob("*.ddb"))
if not files:
    print("  none written"); raise SystemExit
for path in files:
    try:
        con = ibis.duckdb.connect(str(path))
        dbs = con.list_databases()
        rows = 0
        for db in dbs:
            try:
                rows += con.table("metrics", database=db).count().execute()
            except Exception:
                pass
        flag = "  <-- NO METRICS" if rows == 0 else ""
        print(f"  {path.name:34s} {len(dbs):2d} diagnostics  "
              f"{rows:6d} metric rows{flag}")
    except Exception as exc:
        print(f"  {path.name:34s} unreadable: {exc}")
PY
echo "--- figures ---"
# The diagnostic writes to OUT_DIR inside the repo, not to the results
# directory, so a run can produce figures and no metrics or the reverse.
# The script builds OUT_DIR as Path.cwd()/"paper_figures[_maskedobs]",
# so it lands beside the driver, not at the repo root - which is why
# earlier runs reported "0 figures" while figures existed.
case "${CHL_MASK_TO_OBS:-0}" in
    1|true|yes|on) _figsub=paper_figures_maskedobs ;;
    *)             _figsub=paper_figures ;;
esac
FIGDIR=${CHL_FIGDIR:-$HERE/$_figsub}
if [ -d "$FIGDIR" ]; then
    NFIG=$(find "$FIGDIR" -name '*.png' -newermt "@$START" 2>/dev/null | wc -l)
    find "$FIGDIR" -name '*.png' -newermt "@$START" \
        -printf '  %TH:%TM  %8s  %f\n' 2>/dev/null | sort -k3
    echo "  $NFIG written this run  ($FIGDIR)"
    [ "$NFIG" -eq 0 ] && echo "  WARNING: no figures -- check the log above"
else
    echo "  $FIGDIR does not exist"
fi
echo "============================================================"

exit $RC
