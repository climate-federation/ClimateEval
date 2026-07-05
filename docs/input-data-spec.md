# Input data specification

ClimateEval reads **CMOR-style NetCDF**. If your model output is not CMORized,
it only needs a light conversion — the actual requirements are much narrower
than full CMOR compliance, and everything on this page can be done with
xarray, CDO or NCO.

> **This page is written to be handed to an AI agent** (or a human) doing that
> conversion. The feedback loop is `climateeval check`: it validates converted
> files against a suite's requirements in seconds, without reference data, and
> exits with status 0 once everything `climateeval benchmark` needs is in
> place.

```bash
climateeval check /path/to/converted [--suite NAME] [--timerange A/B] [--json]
```

## TL;DR checklist

1. One NetCDF variable per required variable, **named exactly** by its CMOR
   short name (`tas`, `pr`, `ta`, ...) — this name is the only thing used for
   matching.
2. Every coordinate carries its CF `standard_name` (`time`, `latitude`,
   `longitude`, `air_pressure`, ...), and there are **no extra dimensions**
   (squeeze out ensemble/member/singleton dims).
3. The data variable has a valid `units` attribute convertible to the target
   units (any convertible unit is fine; also set the variable's CF
   `standard_name` — required for cross-quantity conversions like
   `kg m-2 s-1` ↔ `mm day-1`).
4. Monthly means (for the `mon` suites) on a time axis that fully covers the
   `--timerange` you will benchmark with.
5. For 3-D atmosphere variables, a pressure axis spanning the levels the suite
   extracts (e.g. 200–800 hPa).
6. Run `climateeval check` and fix reported FAILs until it exits 0; then run
   `climateeval benchmark` with the same `--suite`/`--timerange` flags.

## How ClimateEval matches your data

A variable is selected **only by the NetCDF variable name** (`var_name`),
matched against the CMOR short name — e.g. the 2-m temperature cube must be
the NetCDF variable `tas`. The data variable's `standard_name`/`long_name`
play no role in matching (but see [Units](#units) for why `standard_name`
should still be set). After loading, exactly one cube per variable must
remain: files split in time are concatenated automatically, but only if their
metadata is consistent (same calendar, units and attributes across files).

## Which variables you need

Depends on the suite(s) you run:

- `climateeval suites list` shows the shipped suites;
- the full per-variable reference (short name, long name, standard name,
  target units, coordinates, derived flag) is in
  [Variables & coordinates](variables.md);
- `climateeval check --suite NAME` on your data directory lists every
  required variable and what is missing.

Variables you don't provide are skipped with a warning by default
(`climateeval benchmark` runs with `fail_on_missing_data=False`) — a partial
conversion still produces a report for the variables present.

## Coordinates

Coordinates are recognized by CF `standard_name`, `long_name` **or** variable
name — setting the `standard_name` is the robust choice. Internally they are
renamed to canonical names, so your coordinate variable names don't matter.

| Coordinate | Recognized by (standard_name) | Typical units |
|---|---|---|
| time | `time` | `days since ...` (any CF calendar) |
| latitude | `latitude` | `degrees_north` |
| longitude | `longitude` | `degrees_east` (any convention, see below) |
| pressure level | `air_pressure` | `Pa` or `hPa` |
| altitude | `altitude` | `m` |
| ocean depth | `depth` | `m` |
| ocean basin | `region` | – |

Atmosphere 3-D variables (`ta`, `ua`, `hus`, ...) accept **either** an
`air_pressure` or an `altitude` vertical coordinate; ocean 3-D variables
(`so`, ...) need `depth`.

**Every array dimension must be covered by expected coordinates.** A stray
ensemble/member dimension or a leftover singleton dimension fails the check —
squeeze such dimensions out (`ds.squeeze()` / select one member).

## Units

The `units` attribute must be a valid [udunits2](https://docs.unidata.ucar.edu/udunits/current/)
string convertible to the target units listed in [variables.md](variables.md)
— the conversion itself is automatic, so `degC` for `tas` or `hPa` for `ps`
are fine. A missing or invalid `units` attribute fails.

Some suites convert to a *different physical quantity* (e.g. precipitation
`kg m-2 s-1` → `mm day-1`). Those conversions are resolved via the variable's
CF `standard_name` (e.g. `precipitation_flux`), so set the `standard_name`
from [variables.md](variables.md) on the data variable as well — it is cheap
and makes the conversion unambiguous.

## Time axis

- CF time units (`days since 1850-01-01` etc.); any calendar iris/cftime
  understands (`standard`, `proleptic_gregorian`, `365_day`, `360_day`, ...).
- Monthly means for the `*_monthly` suites (`Tier3_atmosphere_subdaily` needs
  sub-daily data).
- The data must **fully cover** the requested `--timerange` (default
  `19790101/20141231`). Do not pad the data — pass a `--timerange` matching
  your run instead, e.g. `--timerange 19900101/20091231` for a 1990–2009 run.
  The same range is applied to the references, so pick one they cover too.

## Vertical levels

The shipped atmosphere suites extract fixed pressure levels (200, 400 and
800 hPa) from `ta`, `ua` and `hus`. Provide a pressure axis spanning those
values (in any pressure unit); levels outside the data's range produce masked
output and a WARN from `climateeval check`.

## What you do NOT need to fix

ClimateEval normalizes these automatically — don't spend effort on them:

- **Cell bounds**: missing time/lat/lon bounds are guessed.
- **Longitude convention**: `[-180, 180]` and `[0, 360)` both work.
- **Coordinate variable names**: anything recognizable is renamed internally.
- **Unit conversion**: any convertible unit is accepted (see above).
- **Files split in time**: per-year/per-decade files are concatenated.
- **CMOR DRS trees**: grid (`gn` preferred) and latest version are selected
  per variable.
- **Regional/level extraction, masking, regridding**: done by the
  diagnostics.

## Derived variables

Some suite variables are computed from other variables. Either provide the
**source variables** below (by their CMOR names), or provide the derived
variable itself pre-computed under its own name:

| Derived variable | Computed from |
|---|---|
| `amoc` | `msftmz` *or* `msftyz` |
| `asr` | `rsdt`, `rsut` |
| `lwcre` | `rlut`, `rlutcs` |
| `lwp` | `clwvi`, `clivi` |
| `netcre` | `rlut`, `rlutcs`, `rsut`, `rsutcs` |
| `qep` | `evspsbl`, `pr` |
| `rtnt` | `rsdt`, `rsut`, `rlut` |
| `swcre` | `rsut`, `rsutcs` |
| `phcint` | no derivation available — must be provided directly |

(Generated with ESMValCore's `get_required(var, "CMIP6")`; `climateeval
check` reports exactly which sources are missing.)

## File layout and naming

Accepted layouts: a **single NetCDF file**, a **flat directory** of NetCDF
files, or a **CMOR DRS tree** (per-variable subdirectories with
`<grid>/<version>` levels, as on ESGF).

Recommended file name (not required, but useful):

```
<var>_<table>_<ModelName>_<experiment>_<variant>_<grid>_<YYYYMM>-<YYYYMM>.nc
e.g. tas_Amon_MyModel-1-0_historical_r1i1p1f1_gn_199001-200912.nc
```

- the trailing `<YYYYMM>-<YYYYMM>` token lets the loader skip files outside
  the `--timerange` without opening them;
- the third token (`MyModel-1-0`) supplies the report's model name when
  `--name` is not given.

Files with other names still load — matching happens on the NetCDF variable
name, not the file name.

## Worked example (xarray)

Convert a raw model file whose 2-m temperature is called `T2M` with
dimensions `(time, member, lat, lon)` in Kelvin:

```python
import xarray as xr

ds = xr.open_dataset("raw_output.nc")

da = ds["T2M"].isel(member=0, drop=True)   # drop extra dimensions
da = da.rename("tas")                       # CMOR short name = var_name
da.attrs["units"] = "K"                     # valid, convertible units
da.attrs["standard_name"] = "air_temperature"

da["lat"].attrs["standard_name"] = "latitude"
da["lat"].attrs["units"] = "degrees_north"
da["lon"].attrs["standard_name"] = "longitude"
da["lon"].attrs["units"] = "degrees_east"
da["time"].attrs["standard_name"] = "time"

da.to_netcdf(
    "converted/tas_Amon_MyModel-1-0_historical_r1i1p1f1_gn_199001-200912.nc",
    encoding={"time": {"units": "days since 1850-01-01", "dtype": "double"}},
)
```

With CDO/NCO, a rename plus attribute edit is often enough:

```bash
cdo chname,T2M,tas raw_output.nc tas_tmp.nc
ncatted -a standard_name,tas,o,c,air_temperature \
        -a units,tas,o,c,K tas_tmp.nc converted/tas_....nc
```

## Validation loop

```bash
climateeval check converted/ --suite Tier2_atmosphere_monthly --timerange 19900101/20091231
# fix the reported FAILs, re-run until the exit code is 0
# (use --json for machine-readable findings)
climateeval benchmark converted/ --suite Tier2_atmosphere_monthly --timerange 19900101/20091231
```

Notes:

- WARNs (e.g. an extraction level outside the data's range) do not block
  benchmarking; FAILs mean the variable would be skipped or crash the run.
- `Tier3_ecs` needs multiple experiments (abrupt-4xCO2 / piControl) and is not
  covered by `climateeval check`.
