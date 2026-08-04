#!/bin/bash
#SBATCH --job-name=chl_ceval
#SBATCH --account=bd1179
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/home/b/b384036/ClimateEval/logs/ceval_%j.out
#SBATCH --error=/home/b/b384036/ClimateEval/logs/ceval_%j.err
#SBATCH --mail-type=END,FAIL

# ===================================================================
# ClimateEval: manuscript figures only.
#
# Runs example_chl_climateeval_paper.py, which reads its configuration
# from recipe_chl_paper.yml. Produces a DuckDB database and the nine
# pipeline figures; Fig. 10 (dissolved iron against GEOTRACES) is not
# produced here and needs so_fe_bias.py run separately.
#
# USAGE
#   sbatch run_ceval.sh                          # masked, as the recipe says
#   CHL_RECIPE=other.yml sbatch run_ceval.sh     # a different recipe
#
#   The masked/unmasked choice lives in the recipe's `masking:` block,
#   not here, so that the setting travels with the configuration that
#   produced the output rather than with the submit command. Output
#   names carry the masking suffix automatically, so a masked and an
#   unmasked run do not overwrite each other.
# ===================================================================

set -uo pipefail          # deliberately NOT -e: the summary block
                          # below must run even when python fails.

WORKDIR=/home/b/b384036/ClimateEval
SCRIPT=${CEVAL_SCRIPT:-example_chl_climateeval_paper.py}
RECIPE=${CHL_RECIPE:-recipe_chl_paper.yml}
ENVNAME=${ENVNAME:-/work/bd1179/b384036/envs/climateval_gpu}

mkdir -p "$WORKDIR/logs"
cd "$WORKDIR" || { echo "FATAL: cannot cd to $WORKDIR"; exit 1; }
[ -r "$SCRIPT" ] || { echo "FATAL: driver not readable: $WORKDIR/$SCRIPT"; exit 1; }
[ -r "$RECIPE" ] || { echo "FATAL: recipe not readable: $WORKDIR/$RECIPE"; exit 1; }

echo "============================================================"
echo "job     : ${SLURM_JOB_ID:-interactive}"
echo "node    : $(hostname)"
echo "started : $(date)"
echo "driver  : $SCRIPT   $(stat -c '%y' "$SCRIPT" 2>/dev/null)"
echo "recipe  : $RECIPE   $(stat -c '%y' "$RECIPE" 2>/dev/null)"
echo "env     : $ENVNAME"
echo "============================================================"

# micromamba is not on PATH in a non-interactive shell, and .bashrc
# usually returns early when non-interactive, so activate explicitly.
export MAMBA_EXE=${MAMBA_EXE:-$HOME/.local/bin/micromamba}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
[ -x "$MAMBA_EXE" ] || MAMBA_EXE=$(command -v micromamba)
[ -x "$MAMBA_EXE" ] || { echo "FATAL: micromamba not found"; exit 1; }
eval "$("$MAMBA_EXE" shell hook --shell bash)"

# ENVNAME may be a bare name or a full prefix path; try both before
# giving up. A silent fallback to base is the failure worth guarding
# against, since it surfaces much later as a confusing ImportError.
if [[ "$ENVNAME" == */* ]]; then
    [ -d "$ENVNAME" ] || { echo "FATAL: env prefix not found: $ENVNAME"; exit 1; }
fi
micromamba activate "$ENVNAME" 2>/dev/null \
    || micromamba activate -p "$ENVNAME" \
    || { echo "FATAL: cannot activate env '$ENVNAME'"; exit 1; }
echo "prefix  : ${CONDA_PREFIX:-unset}"
echo "python  : $(command -v python)"

export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1        # readable log while the job runs
export CHL_RECIPE="$RECIPE"
export CUDA_VISIBLE_DEVICES=""   # GPU-built env on a CPU partition

echo "--- environment ---"
python - <<'PY'
import sys
print("python     ", sys.version.split()[0])
for m in ("numpy", "iris", "yaml", "duckdb", "matplotlib", "cartopy",
          "climateeval"):
    try:
        print(f"{m:11s}", getattr(__import__(m), "__version__", "present"))
    except Exception as e:
        print(f"{m:11s} MISSING ({e})")
PY

# Read the recipe here as well as in the driver, so the log records what
# was asked for even if the driver dies during the Suite call.
echo "--- recipe ---"
python - <<PY
import sys, yaml
try:
    cfg = yaml.safe_load(open("$RECIPE"))
except Exception as e:
    print(f"FATAL: cannot parse recipe: {e}"); sys.exit(1)
mask = cfg.get("masking", {}) or {}
on = bool(mask.get("mask_models_to_obs"))
suffix = mask.get("output_suffix", "") if on else ""
db = cfg["suite"]["database"]
db = db[:-4] + suffix + ".ddb" if db.endswith(".ddb") else db + suffix
figs = str(cfg["suite"]["figure_dir"]) + suffix
print(f"  suite     : {cfg['suite']['name']}")
print(f"  models    : {len(cfg['models'])}")
print(f"  figures   : {len(cfg['figures'])}  {list(cfg['figures'])}")
print(f"  variables : "
      f"{sorted({v for f in cfg['figures'].values() for v in f['vars']})}")
print(f"  masking   : {on}" + (f"  (ref {mask.get('reference')})" if on else ""))
print(f"  database  : {db}")
print(f"  figures to: {figs}/")
# Written out so the summary block below can check the right paths
# without re-deriving the suffix logic and getting it subtly different.
open(".ceval_paths", "w").write(f"{db}\n{figs}\n{len(cfg['figures'])}\n")
PY
RC=$?
[ $RC -ne 0 ] && { echo "FATAL: recipe invalid; not submitting work"; exit $RC; }

echo "--- run ---"
START=$(date +%s)
python "$SCRIPT"
RC=$?
END=$(date +%s)

echo "============================================================"
echo "exit code : $RC"
echo "elapsed   : $(( (END-START)/60 )) min $(( (END-START)%60 )) s"
echo "finished  : $(date)"

# ---- verification -------------------------------------------------
# The driver creates the database before the suite writes to it, so the
# file existing proves nothing: a run that failed on every model still
# leaves one behind. Check the row counts.
if [ -r .ceval_paths ]; then
    DB=$(sed -n 1p .ceval_paths)
    FIGS=$(sed -n 2p .ceval_paths)
    WANT=$(sed -n 3p .ceval_paths)

    echo "--- database ---"
    if [ -f "$DB" ]; then
        python - <<PY
import duckdb
try:
    con = duckdb.connect("$DB", read_only=True)
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if not tables:
        print("  WARNING: database has no tables -- every model failed")
    for t in tables:
        n = con.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0]
        flag = "  <-- EMPTY" if n == 0 else ""
        print(f"  {t:40s} {n:8d} rows{flag}")
except Exception as e:
    print(f"  could not read database: {e}")
PY
    else
        echo "  WARNING: $DB was not created"
    fi

    echo "--- figures ---"
    if [ -d "$FIGS" ]; then
        NFIG=$(find "$FIGS" -name '*.png' -newermt "@$START" | wc -l)
        find "$FIGS" -name '*.png' -newermt "@$START" \
            -printf '  %TH:%TM  %8s  %f\n' 2>/dev/null | sort -k3
        echo "  $NFIG written this run (recipe asks for $WANT)"
        if [ "$NFIG" -gt "$WANT" ] 2>/dev/null; then
            echo "  WARNING: more figures than the recipe requested."
            echo "           The 'figures' kwarg was probably ignored and the"
            echo "           suite emitted its full set; the figure selection"
            echo "           is not taking effect."
        elif [ "$NFIG" -eq 0 ]; then
            echo "  WARNING: nothing written -- check the log above"
        fi
    else
        echo "  WARNING: $FIGS/ does not exist"
    fi
    rm -f .ceval_paths
fi

echo "--- key diagnostics ---"
LOG="$WORKDIR/logs/ceval_${SLURM_JOB_ID:-manual}.out"
if [ -r "$LOG" ]; then
    grep -E "Recipe:|Figures requested|Variables required|masked to the|\
NOT masked|Figure coverage|models  \[|missing:|skipping|failed|Traceback|\
retention|TypeError" "$LOG" | tail -60
else
    echo "  (log not readable yet: $LOG)"
fi
echo "============================================================"

exit $RC
