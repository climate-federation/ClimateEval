#!/bin/bash
# Parallel multi-model tier benchmark on SLURM (DKRZ Levante).
#
# Submits one array task per model (each writing its own scratch DuckDB via
# run_tier_benchmark_model.py) plus a dependent merge job that unions them
# into one combined database per suite with merge_tier_dbs.py.
#
# Usage (from the repository root):
#   scripts/submit_tier_benchmark.sh MODELS_FILE [OUT_DIR] [SUITE] [TIMERANGE]
#
#   MODELS_FILE  one "INSTITUTE/DATASET" per line, benchmarked model FIRST
#                (its rows win the merge and it lands first in the report)
#   OUT_DIR      combined output dir            (default: output)
#   SUITE        suite name                     (default: Tier2_atmosphere_monthly)
#   TIMERANGE    ISO timerange                  (default: 19900101/20091231)
#
# Env overrides: ACCOUNT (default bb1596), PARTITION (default compute),
# TIME (default 02:00:00), MEM (default 64G).
set -euo pipefail

MODELS_FILE="${1:?usage: submit_tier_benchmark.sh MODELS_FILE [OUT_DIR] [SUITE] [TIMERANGE]}"
OUT_DIR="${2:-output}"
SUITE="${3:-Tier2_atmosphere_monthly}"
TIMERANGE="${4:-19900101/20091231}"

ACCOUNT="${ACCOUNT:-bb1596}"
PARTITION="${PARTITION:-compute}"
TIME="${TIME:-02:00:00}"
MEM="${MEM:-64G}"

WORKDIR="$(pwd)"
SCRATCH_DIR="${OUT_DIR}/parallel"
mkdir -p "${WORKDIR}/logs" "${SCRATCH_DIR}"

mapfile -t MODELS < <(grep -vE '^\s*(#|$)' "${MODELS_FILE}")
N=${#MODELS[@]}
[ "${N}" -ge 1 ] || { echo "No models in ${MODELS_FILE}" >&2; exit 1; }
echo "Submitting ${N} model task(s) for suite ${SUITE}"

ARRAY_JID=$(sbatch --parsable \
  --job-name="ce_${SUITE}" \
  --account="${ACCOUNT}" --partition="${PARTITION}" \
  --nodes=1 --ntasks=1 --cpus-per-task=8 --mem="${MEM}" --time="${TIME}" \
  --array="0-$((N - 1))" \
  --output="logs/tier_bench_%A_%a.out" --error="logs/tier_bench_%A_%a.err" \
  --wrap "cd '${WORKDIR}' && eval \"\$('${WORKDIR}/.pixi/bin/pixi' shell-hook)\" && \
python scripts/run_tier_benchmark_model.py \
  --models-file '${MODELS_FILE}' --model-index \${SLURM_ARRAY_TASK_ID} \
  --suites '${SUITE}' --timerange '${TIMERANGE}' --out-dir '${SCRATCH_DIR}'")
echo "Array job: ${ARRAY_JID}"

# Merge sources in models-file order so the benchmarked model comes first.
MERGE_SOURCES=""
for m in "${MODELS[@]}"; do
  MERGE_SOURCES+=" '${SCRATCH_DIR}/${SUITE}__${m##*/}.ddb'"
done

MERGE_JID=$(sbatch --parsable \
  --job-name="ce_merge_${SUITE}" \
  --account="${ACCOUNT}" --partition="${PARTITION}" \
  --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=00:20:00 \
  --dependency="afterok:${ARRAY_JID}" \
  --output="logs/tier_merge_%j.out" --error="logs/tier_merge_%j.err" \
  --wrap "cd '${WORKDIR}' && eval \"\$('${WORKDIR}/.pixi/bin/pixi' shell-hook)\" && \
rm -f '${OUT_DIR}/${SUITE}.ddb' && \
python scripts/merge_tier_dbs.py '${OUT_DIR}/${SUITE}.ddb'${MERGE_SOURCES}")
echo "Merge job:  ${MERGE_JID} (afterok:${ARRAY_JID})"
echo "Combined database will be: ${OUT_DIR}/${SUITE}.ddb"
