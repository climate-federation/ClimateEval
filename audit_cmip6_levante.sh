#!/usr/bin/env bash
# ============================================================================
# audit_cmip6_levante.sh
#
# Scan the DKRZ Levante CMIP6 archive and list every model that has ALL six
# variables required by the variability-based diagnostic in the manuscript:
#
#     chl       (ocean surface chlorophyll-a, Omon)
#     mlotst    (ocean mixed-layer thickness,  Omon)
#     tos       (sea surface temperature,      Omon)
#     clt       (total cloud cover,            Amon)
#     siconc    (sea-ice concentration,        SImon)
#
# It also reports rsds (surface downwelling shortwave, Amon) as a sixth
# variable for completeness, matching the docstring of example_chl_nodatabse.py,
# but a model is considered "complete" if it has the five core variables.
#
# Scope:
#   experiment: historical
#   frequency : monthly (Omon / SImon / Amon)
#   members   : any r*i*p*f* (script prints the first matching member per model)
#
# Run on Levante (the script uses purely local filesystem traversal — no
# ESGF query, no network). Should take a few seconds.
#
# Usage:
#   bash audit_cmip6_levante.sh             # print summary table to stdout
#   bash audit_cmip6_levante.sh > audit.txt # save to file
# ============================================================================

set -u  # treat unset variables as error; do NOT set -e because find may exit
        # with non-zero on missing branches and that is fine for an audit.

ROOT="/work/ik1017/CMIP6/data/CMIP6/CMIP"
EXPERIMENT="historical"

# Required ensemble member. Strict equality: a model is COMPLETE only
# if all required variables are present under r1i1p1f1 specifically.
# Models for which r1i1p1f1 does not exist (e.g. UKESM1-0-LL, which
# only provides r*i*p*f2) will NOT be listed as COMPLETE, even if they
# have all variables under a different member.
REQUIRED_MEMBER="r1i1p1f1"

# Core variables (must ALL be present for the model to be "complete")
CORE_VARS=("chl" "mlotst" "tos" "clt" "siconc")

# Optional variable, reported but does not gate completeness
OPTIONAL_VARS=("rsds")

# Table mapping for each variable. CMIP6 stores variables in different
# tables depending on the realm: Omon (ocean), SImon (sea ice), Amon
# (atmosphere). The script tries the table listed here first; if the
# variable directory does not exist, the model is recorded as missing.
declare -A TABLE_OF
TABLE_OF[chl]="Omon"
TABLE_OF[mlotst]="Omon"
TABLE_OF[tos]="Omon"
TABLE_OF[clt]="Amon"
TABLE_OF[siconc]="SImon"
TABLE_OF[rsds]="Amon"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

# Return 0 (true) if at least one .nc file exists for
# (model, REQUIRED_MEMBER, table, var). The check is strict on the
# member: any other ensemble member is ignored. Sets FOUND_MEMBER to
# REQUIRED_MEMBER on success so the report column reads "r1i1p1f1"
# rather than "-".
FOUND_MEMBER=""
has_variable() {
    local model="$1"
    local var="$2"
    local table="${TABLE_OF[$var]}"
    FOUND_MEMBER=""

    # Path glob:
    # <root>/<inst>/<model>/<exp>/<REQUIRED_MEMBER>/<table>/<var>/<grid>/<ver>/*.nc
    # The member is hard-coded into the path (not a wildcard), so
    # variables that only exist under a different member are not
    # counted toward completeness.
    local match
    match=$(find "$ROOT"/*/"$model"/"$EXPERIMENT"/"$REQUIRED_MEMBER"/"$table"/"$var" \
              -maxdepth 3 -name "*.nc" -print -quit 2>/dev/null)

    if [[ -n "$match" ]]; then
        FOUND_MEMBER="$REQUIRED_MEMBER"
        return 0
    fi
    return 1
}

# ----------------------------------------------------------------------------
# Discover all models in the archive
# ----------------------------------------------------------------------------

echo "============================================================"
echo "CMIP6 audit on Levante: $ROOT"
echo "Experiment: $EXPERIMENT"
echo "Required member: $REQUIRED_MEMBER (STRICT — no fallback)"
echo "Required vars (all 5 must be present): ${CORE_VARS[*]}"
echo "Also reported            : ${OPTIONAL_VARS[*]}"
echo "============================================================"

# All model directories: <inst>/<model>/
# Use find to two depths under ROOT to enumerate every model.
mapfile -t MODELS < <(
    find "$ROOT" -mindepth 2 -maxdepth 2 -type d \
        | awk -F/ '{print $NF}' \
        | sort -u
)

echo "Found ${#MODELS[@]} model directories under $ROOT"
echo ""

# ----------------------------------------------------------------------------
# Per-model audit
# ----------------------------------------------------------------------------

# Build a fixed-width header
printf "%-25s | %-12s | %-12s | %-12s | %-12s | %-12s | %-12s | %s\n" \
    "MODEL" "chl(Omon)" "mlotst(Omon)" "tos(Omon)" "clt(Amon)" \
    "siconc(SImon)" "rsds(Amon)" "STATUS"
printf -- "%.s-" {1..130}; echo ""

COMPLETE_MODELS=()
PARTIAL_MODELS=()

for model in "${MODELS[@]}"; do
    declare -A PRESENT
    declare -A MEMBERS
    for v in "${CORE_VARS[@]}" "${OPTIONAL_VARS[@]}"; do
        if has_variable "$model" "$v"; then
            PRESENT[$v]=1
            MEMBERS[$v]="$FOUND_MEMBER"
        else
            PRESENT[$v]=0
            MEMBERS[$v]="-"
        fi
    done

    # Decide completeness on the 5 core variables only.
    n_present=0
    for v in "${CORE_VARS[@]}"; do
        n_present=$((n_present + PRESENT[$v]))
    done

    if [[ "$n_present" -eq "${#CORE_VARS[@]}" ]]; then
        status="COMPLETE"
        COMPLETE_MODELS+=("$model")
    elif [[ "$n_present" -gt 0 ]]; then
        status="partial ($n_present/${#CORE_VARS[@]})"
        PARTIAL_MODELS+=("$model")
    else
        # Skip models with zero overlap with our variable set — usually
        # atmosphere-only or chemistry-only configurations.
        unset PRESENT MEMBERS
        continue
    fi

    # Format the row: show member string when present, "-" when not.
    fmt() {
        if [[ "${PRESENT[$1]}" -eq 1 ]]; then
            printf "%-12s" "${MEMBERS[$1]}"
        else
            printf "%-12s" "-"
        fi
    }

    printf "%-25s | %s | %s | %s | %s | %s | %s | %s\n" \
        "$model" \
        "$(fmt chl)" "$(fmt mlotst)" "$(fmt tos)" \
        "$(fmt clt)" "$(fmt siconc)" "$(fmt rsds)" \
        "$status"

    unset PRESENT MEMBERS
done

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "SUMMARY"
echo "============================================================"
echo "Models with ALL ${#CORE_VARS[@]} core variables present:"
for m in "${COMPLETE_MODELS[@]}"; do
    echo "  - $m"
done
echo ""
echo "Total complete:   ${#COMPLETE_MODELS[@]}"
echo "Total partial:    ${#PARTIAL_MODELS[@]}"
echo ""
echo "Models in your study (check that these appear COMPLETE above):"
echo "  - GFDL-ESM4"
echo "  - UKESM1-0-LL    (uses r1i1p1f2, NOT r1i1p1f1, so under the strict"
echo "                    member filter this audit will report it as"
echo "                    INCOMPLETE / not listed. Re-run with"
echo "                    REQUIRED_MEMBER=\"r1i1p1f2\" near the top of"
echo "                    the script to check it explicitly.)"
echo "  - MPI-ESM-1-2-HAM"
echo "  - CESM2"
echo ""
echo "Any other model that appears COMPLETE is a candidate you could"
echo "have included. Any of your four that does NOT appear COMPLETE means"
echo "either r1i1p1f1 isn't on Levante or one of the variables is missing"
echo "for that exact member."
