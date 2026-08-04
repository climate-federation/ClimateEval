#!/bin/bash
#SBATCH --job-name=chl_diag
#SBATCH --account=bd1179
#SBATCH --partition=shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/b/b384036/ClimateEval/logs/chl_%j.out
#SBATCH --error=/home/b/b384036/ClimateEval/logs/chl_%j.err
#SBATCH --mail-type=END,FAIL

# ===================================================================
# CMIP6 Chl-a process-oriented diagnostics.
#
# USAGE
#   CHL_STAGES=meanstate sbatch run_chl.sh    # Figs 1-2, 3, 4    (~30 min)
#   CHL_STAGES=seaice    sbatch run_chl.sh    # sea-ice x Chl-a    (~2.5 h)
#   CHL_STAGES=ptcorr    sbatch run_chl.sh    # filtered corr maps (~2 h)
#   CHL_STAGES=all       sbatch run_chl.sh    # everything         (~7 h)
#   sbatch run_chl.sh                         # same as "all"
#
#   ENVNAME=/work/bd1179/b384036/envs/climateval CHL_STAGES=seaice \
#       sbatch run_chl.sh
#       run in a different conda/micromamba environment. ENVNAME may be
#       a bare env name or a full prefix path. Use this if the cartopy
#       "getX called on empty Point" warnings appear, or if the polar
#       panels render with artefacts: cartopy 0.23 / shapely 2.0.4 draws
#       them cleanly, 0.25 / 2.1.2 does not. Both sea-ice seasonal
#       figures and the Southern/Arctic columns of the basin correlation
#       maps are polar stereographic, so this matters for those runs.
#
# STAGE TAGS (comma-separated, or use a preset above)
#   A B C E F   phase letters
#   light       rsds bias map + polar light coupling
#   drivers     Fig 5/6/7 bias maps
#   coupling    Fig 8/8b/9/9b/10 MLD- and SST-Chl-a lag correlation
#   memory      Fig 11/11b/11c/12/S1 sea-ice memory
#   peaklag     figS_spatial_peak_lag_*       <-- the ~2.5 h block
#   seaice      pointwise corr + R2/beta/gain maps + seasonal figures
#   ptcorr      6-deg filtered basin correlation maps, per-model maps,
#               seasonal Hovmoller, area-mean lag curves (SST and MLD,
#               raw and with AMO/ENSO/PDO each removed separately)
#   polar       figS_seaice_chl_polar_*
#   bloom       figS_bloom_phenology_*
#   eof         figS_seaice_eof + figS_chl_eof
#   regional    figS_regional_corr_*
#
# Data loads are gated too: "meanstate" reads only Chl-a and rsds,
# skipping the MLD, SST and cloud fields entirely.
#
# RESOURCES
#   Memory: the ESA-CCI sea-ice loader streams 18 yearly files per
#   hemisphere at 864x864 via dask (chunked, not eager), and the
#   regrid cache holds ~1.8 GB of z-scored fields. The filtered-field
#   cache used by the ptcorr stage adds up to ~1 GB more (float32).
#   96G is comfortable; 64G is usually enough if the queue is busy.
#   Time: 8 h covers a full run with margin. Figures are written
#   incrementally, so a timeout still leaves completed output on disk.
# ===================================================================

set -uo pipefail          # deliberately NOT -e: the summary block
                          # below must run even when python fails.

WORKDIR=/home/b/b384036/ClimateEval
FIGDIR=$WORKDIR/paper_figures
SCRIPT=${CHL_SCRIPT:-example_chl_nodatabse10.py}
ENVNAME=${ENVNAME:-/work/bd1179/b384036/envs/climateval_gpu}
STAGES=${CHL_STAGES:-all}

mkdir -p "$WORKDIR/logs" "$FIGDIR"
cd "$WORKDIR" || { echo "FATAL: cannot cd to $WORKDIR"; exit 1; }
[ -r "$SCRIPT" ] || { echo "FATAL: script not readable: $WORKDIR/$SCRIPT"; exit 1; }

echo "============================================================"
echo "job        : ${SLURM_JOB_ID:-interactive}"
echo "node       : $(hostname)"
echo "started    : $(date)"
echo "script     : $SCRIPT"
echo "           : $(stat -c '%y  %s bytes' "$SCRIPT" 2>/dev/null)"
echo "env        : $ENVNAME"
echo "CHL_STAGES : $STAGES        <<< stages actually enabled"
echo "============================================================"

# micromamba is not on PATH in a non-interactive shell, and .bashrc
# usually returns early when non-interactive, so activate explicitly.
export MAMBA_EXE=${MAMBA_EXE:-$HOME/.local/bin/micromamba}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-$HOME/micromamba}
[ -x "$MAMBA_EXE" ] || MAMBA_EXE=$(command -v micromamba)
[ -x "$MAMBA_EXE" ] || { echo "FATAL: micromamba not found"; exit 1; }
eval "$("$MAMBA_EXE" shell hook --shell bash)"

# ENVNAME may be a bare name (resolved against the envs dirs) or a full
# prefix path. micromamba accepts a path directly in recent versions but
# older ones require -p, so try both before giving up. Activation that
# silently falls back to the base environment is the failure mode worth
# guarding against: it surfaces much later as a confusing ImportError,
# so the resolved prefix is echoed and checked below.
if [[ "$ENVNAME" == */* ]]; then
    [ -d "$ENVNAME" ] || { echo "FATAL: env prefix not found: $ENVNAME"; exit 1; }
fi
micromamba activate "$ENVNAME" 2>/dev/null \
    || micromamba activate -p "$ENVNAME" \
    || { echo "FATAL: cannot activate env '$ENVNAME'"; exit 1; }

echo "env prefix : ${CONDA_PREFIX:-unset}"
echo "python     : $(command -v python)"
case "${CONDA_PREFIX:-}" in
    *"$(basename "$ENVNAME")"*) ;;
    *) echo "WARNING: activated prefix does not match requested env;"
       echo "         micromamba may have fallen back to base." ;;
esac

# Headless plotting; keep BLAS from spawning threads for all 128
# physical cores when SLURM granted 8 (oversubscription is slower).
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export NUMEXPR_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1        # readable log while the job runs
export CHL_STAGES="$STAGES"

# This is a GPU-built environment running on a CPU partition. Nothing in
# the diagnostic needs a GPU, but a CUDA-linked package that finds no
# device can emit pages of warnings or stall on device discovery, so
# hide any GPUs explicitly and keep the log readable.
export CUDA_VISIBLE_DEVICES=""

echo "--- environment ---"
python - <<'PY'
import sys
print("python     ", sys.version.split()[0])
for m in ("numpy", "xarray", "dask", "scipy",
          "matplotlib", "cartopy", "shapely"):
    try:
        print(f"{m:11s}", getattr(__import__(m), "__version__", "?"))
    except Exception as e:
        print(f"{m:11s} MISSING ({e})")
# The polar panels of the sea-ice and basin-correlation figures render
# incorrectly under cartopy 0.25 with shapely 2.1. Warn rather than
# fail: the rest of the run is unaffected and still worth having.
try:
    import cartopy
    if cartopy.__version__.startswith("0.25"):
        print("NOTE: cartopy 0.25 detected -- polar stereographic panels "
              "may show artefacts; rerun with "
              "ENVNAME=/work/bd1179/b384036/envs/climateval if so")
except Exception:
    pass
PY

echo "--- pre-flight check on $SCRIPT ---"
python - <<PY
import ast, sys
src = open("$SCRIPT").read()
try:
    ast.parse(src)
except SyntaxError as e:
    print(f"FATAL: syntax error at line {e.lineno}: {e.msg}"); sys.exit(1)

# Long-standing checks: these date from earlier revisions and catch a
# copy that predates the sea-ice regression work.
checks = {
    "stage gating":          "def _parse_stages" in src,
    "ESA-CCI loader":        "def load_obs_siconc_esacci" in src,
    "re-shadow fixed":       "re_m = regrid_to_obs" in src,
    "EOF sign template":     "_EOF1_TEMPLATES" in src,
    "PC normalisation":      "per 1 s.d. of PC1" in src,
    "complete-case regr.":   "complete-case regression" in src,
    "adjusted R2":           "r2_g = 1.0 - (1.0 - r2_g)" in src,
    "simple-PC1 beta":       "BETA_MIN_OBS_SIMPLE" in src,
    "R2 gain diagnostic":    "REDUNDANCY CHECK" in src,
    "shared regions":        "SEAICE_REGIONS" in src,
    "cartopy guards":        "transform_first=True" in src,
    "per-region isolation":  "regression FAILED" in src,
    "regrid cache":          "_REGRID_Z_CACHE" in src,
    "Fig 3 amplitude":       "_seasonal_amplitude" in src,
}
# Current work. Every string above predates the correlation-map rewrite,
# so without these a stale copy passes all fourteen checks and runs to
# completion producing the wrong figures -- which is exactly what a
# filename mismatch between script versions would cause.
checks_new = {
    "6deg spatial filter":   "def _spatial_filter" in src,
    "filter footprint fix":  "restrict_to_input" in src,
    "cyclic seam fix":       "wrap gap" in src,
    "basin corr maps":       "make_figS_basin_corr_obs_ens" in src,
    "per-model corr maps":   "make_figS_basin_corr_per_model" in src,
    "seasonal Hovmoller":    "make_figS_seasonal_hovmoller" in src,
    "area-mean lag curves":  "make_fig_lag_areamean" in src,
    "mode tags":             "def _mode_tag" in src,
    "per-source indices":    "def _indices_for_source" in src,
    "seasonal seaice figs":  "make_figS_seaice_chl_seasonal_corr" in src,
    "growing-season split":  "_SEASON_MONTHS" in src,
    "lag convention test":   "_assert_lag_convention" in src,
    "cache fingerprints":    "def _da_fingerprint" in src,
}
for k, v in checks.items():
    print(f"  {'OK  ' if v else 'MISS'}  {k}")
print("  --- current revision ---")
for k, v in checks_new.items():
    print(f"  {'OK  ' if v else 'MISS'}  {k}")

if not all(checks.values()):
    print("WARNING: this copy is older than expected -- check you "
          "pulled the current file before submitting.")
if not all(checks_new.values()):
    print("WARNING: this copy PREDATES the correlation-map rewrite. The "
          "run will produce the superseded figures. Check that "
          "CHL_SCRIPT / SCRIPT names the file you edited.")
PY

echo "--- run ---"
START=$(date +%s)
python "$SCRIPT"
RC=$?
END=$(date +%s)

echo "============================================================"
echo "exit code  : $RC"
echo "elapsed    : $(( (END-START)/60 )) min $(( (END-START)%60 )) s"
echo "finished   : $(date)"

echo "--- files written in this run ---"
find "$FIGDIR" \( -name '*.png' -o -name '*.csv' \) -newermt "@$START" \
    -printf '%TH:%TM  %8s  %f\n' 2>/dev/null | sort -k3 || true
NEW=$(find "$FIGDIR" \( -name '*.png' -o -name '*.csv' \) \
      -newermt "@$START" 2>/dev/null | wc -l)
echo "  total: $NEW files"
[ "$NEW" -eq 0 ] && echo "  WARNING: nothing written -- check the log above"

# Figures whose filenames changed in the current revision are not
# overwritten by this run, so an old copy can sit in the output
# directory looking current. Flag any that are still present.
echo "--- superseded figures still on disk (delete these) ---"
find "$FIGDIR" \( -name 'figS_pointwise_corr_*' \
                  -o -name '*_modesremoved.png' \) \
    -printf '  %f\n' 2>/dev/null || true

echo "--- key diagnostics ---"
LOG="$WORKDIR/logs/chl_${SLURM_JOB_ID:-manual}.out"
if [ -r "$LOG" ]; then
    grep -E "ESACCI-SEAICE|regrid cache MISS|filtered-field cache MISS|\
PC1 s.d\.|cancellation check|ice variance at PC|REDUNDANCY CHECK|\
pattern r=|beta mask|colour scale|ocean-colour coverage|\
area-mean R2|no mode removed|of ocean cells robust|\
fixed footprint|models contributed|\
SKIPPED|FAILED|crashed|failed on|Traceback" "$LOG" | tail -80
else
    echo "  (log not readable yet: $LOG)"
fi
echo "============================================================"

exit $RC
