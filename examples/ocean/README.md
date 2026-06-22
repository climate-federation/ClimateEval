# Ocean & sea-ice diagnostics — setup and run guide

This example runs the ClimateEval ocean and sea-ice diagnostics on a CMIP6 model
and produces a self-contained interactive HTML report. The diagnostics covered:

- **SST, salinity, chlorophyll** — maps, zonal-mean lines, annual-mean time series
- **Mixed-layer depth** — annual cycle in the Labrador Sea, Weddell Sea, Southern Ocean
- **Niño 3.4** — the Oceanic Niño Index
- **Sea-ice area** — annual cycle and time series (NH September / SH February)
- **AMOC** — Atlantic overturning, benchmarked against RAPID
- **Ocean heat content** — full / 0–300 m / 0–2000 m, benchmarked against EN4 and IAP

Everything below assumes you are on **DKRZ Levante**, where the observational
references are already available (no downloads needed).

---

## 1. Install ClimateEval

Follow the main [README](../../README.md). In short:

```bash
git clone https://github.com/climate-federation/ClimateEval.git
cd ClimateEval
pixi install
```

Then either open a shell in the environment (`pixi shell`) or prefix the
commands below with `pixi run`.

> **Note on Ocean Heat Content.** The `phcint` derivation used by the OHC
> diagnostic currently lives only on the ESMValCore `expand_icon_xpp_ocean`
> branch (not yet upstream). On a stock install the OHC run will fail until that
> branch is installed. The SST/sea-ice/AMOC runs work without it.

## 2. Stage the observational references

The references are already CMORized in the shared ESMValTool OBS pool on Levante.
Symlink them into a directory of your choice (your `data_root_dir`):

```bash
cd examples/ocean
bash stage_obs_refs.sh ~/climateeval_input
```

This requires read access to `/work/bd0854` (group `bd0854`). It stages OSI-450,
HadISST, NSIDC-G02202, ESACCI-SST, NOAA-ERSSTv5, WOA, ESACCI-OC, EN4 and IAP.

ORAS5 (for mixed-layer depth) and RAPID (for AMOC) are read directly from shared
paths and do **not** need staging. They are pointed at by environment variables
with sensible Levante defaults — override only if those files move:

```bash
export CLIMATEEVAL_ORAS5_RAW_DIR=/work/bd1083/b382555/extraobsraw/Tier2/ORAS5/data
export CLIMATEEVAL_ORAS5_GRID_DIR=/work/bd1083/b382555/extraobsraw/Tier2/ORAS5/grids
export CLIMATEEVAL_RAPID_FILE=/work/bd0854/DATA/ESMValTool2/OBS/Tier3/RAPID/OBS6_RAPID_exp_2024.1a_Omon_amoc_2004_2024.nc
```

## 3. Run the diagnostics

Point the runner at your staged references and an output directory, then run it:

```bash
export CLIMATEEVAL_DATA_ROOT=~/climateeval_input
export CLIMATEEVAL_RESULTS_DIR=~/ocean_results
python run_ocean_diagnostics.py
```

This writes three databases and the report into `CLIMATEEVAL_RESULTS_DIR`:

```
seaice.ddb
ocean_ohc.ddb
ocean_sst.ddb
report.html      # self-contained — open in any browser
```

The diagnostics regrid and preprocess on the fly, so a full run uses a fair
amount of memory. On Levante, submit it to the **compute** partition (an
exclusive node) rather than a login node — see `run.slurm` for a template.

## 4. Evaluate a different model

The example is configured for MPI-ESM1-2-LR. To evaluate another model, edit the
`MODEL` block near the top of `run_ocean_diagnostics.py`:

- **`MODEL_DIR`** — the directory holding the model's CMIP6 output. The runner
  finds files with the pattern
  `MODEL_DIR/<MIP>/<var>/<grid>/<version>/<var>_<MIP>_*.nc`.
- **`MODEL_GRID`** / **`MODEL_VERSION`** — the grid label (`gn`/`gr`) and data
  version in the path (`MODEL_VERSION` may be `"*"` to match any).
- **`MODEL_INFO`** — the `DataSourceInformation` (name, institute, experiment,
  variant) recorded in the report.

Pick a variant that actually contains **all** the required variables
(`siconc`, `tos`, `so`, `thetao`, `chl`, `msftmz`, `mlotst`). Then re-run
step 3 — staging and references stay the same.

### Notes

- The three runs use different time ranges on purpose: ClimateEval applies one
  time range to every dataset, and each reference covers a different period, so
  the ranges are chosen so each reference overlaps the model.
- AMOC derivation accepts either `msftmz` or `msftyz` (both optional inputs).
- Missing references are skipped, not fatal (`fail_on_missing_data=False`), so a
  partial setup still produces a report for whatever is available.
