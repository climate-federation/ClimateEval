#!/bin/bash
# Stage ocean + sea-ice observational references for ClimateEval by symlinking
# the already-CMORized ESMValTool OBS files (on DKRZ Levante) into the directory
# layout ClimateEval expects:  <data_root_dir>/<source_id>/<frequency>/<var>/
#
# No downloads: these datasets are already CMORized in the shared ESMValTool OBS
# pool. Pass the directory you want to use as `data_root_dir` for the runs.
#
# Usage:  bash stage_obs_refs.sh <data_root_dir>
# Example: bash stage_obs_refs.sh ~/climateeval_input
set -eu

if [ "$#" -lt 1 ]; then
  echo "Usage: bash stage_obs_refs.sh <data_root_dir>" >&2
  exit 1
fi

ST=$1                                    # where to create the staged tree
P=/work/bd0854/DATA/ESMValTool2/OBS      # shared ESMValTool OBS pool on Levante

link() { mkdir -p "$1"; ln -sf $2 "$1"/; echo "  $1 <- $(ls "$1" | wc -l) files"; }

echo "Staging ocean/sea-ice obs references into $ST"
# Sea ice (OSI-450 files use the legacy cube name 'sic'; renamed on load).
link "$ST/reanalysis_OSI-450-nh/mon/siconc"      "$P/Tier2/OSI-450-nh/OBS_OSI-450-nh_reanaly_v3_OImon_sic_*.nc"
link "$ST/reanalysis_OSI-450-sh/mon/siconc"      "$P/Tier2/OSI-450-sh/OBS_OSI-450-sh_reanaly_v3_OImon_sic_*.nc"
link "$ST/reanalysis_HadISST/mon/siconc"         "$P/Tier2/HadISST/OBS6_HadISST_reanaly_1_SImon_siconc_*.nc"
link "$ST/reanalysis_NSIDC-G02202-sh/mon/siconc" "$P/Tier3/NSIDC-G02202-sh/OBS6_NSIDC-G02202-sh_reanaly_4_SImon_siconc_*.nc"
# SST
link "$ST/observation_ESACCI-SST/mon/tos"        "$P/Tier2/ESACCI-SST/OBS6_ESACCI-SST_sat_3.0-L4-analysis_Omon_tos_*.nc"
link "$ST/reanalysis_NOAA-ERSSTv5/mon/tos"       "$P/Tier2/NOAA-ERSSTv5/OBS6_NOAA-ERSSTv5_reanaly_v5_Omon_tos_*.nc"
link "$ST/reanalysis_HadISST/mon/tos"            "$P/Tier2/HadISST/OBS_HadISST_reanaly_1_Omon_tos_*.nc"
# Salinity (WOA climatology)
link "$ST/observation_WOA/mon/so"                "$P/Tier2/WOA/OBS6_WOA_clim_2018_Omon_so_*.nc"
# Chlorophyll
link "$ST/observation_ESACCI-OC/mon/chl"         "$P/Tier2/ESACCI-OC/OBS6_ESACCI-OC_sat_fv5.0_Omon_chl_*.nc"
# Ocean heat content references (thetao -> phcint is derived by ClimateEval).
link "$ST/reanalysis_EN4/mon/thetao"             "$P/Tier2/EN4/OBS6_EN4_reanaly_4.2.2_Omon_thetao_*.nc"
link "$ST/reanalysis_IAP/mon/thetao"             "$P/Tier2/IAP/OBS6_IAP_reanaly_v4.2_Omon_thetao_*.nc"
echo "Done."
echo "ORAS5 (mlotst) and RAPID (amoc) are read from shared paths via env vars;"
echo "see README.md if you need to override them."
