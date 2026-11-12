# Suites

A **suite** is a YAML file that lists the diagnostics to run, the variables each
diagnostic evaluates, and the reference dataset each variable is compared
against. Suites are the reproducible, declarative core of ClimateEval.

Shipped suites live in
[`src/climateeval/suites/`](../src/climateeval/suites/). Pass a suite by name
(the file stem) or give `Suite(...)` a path to your own YAML.

## The YAML format

A suite file is a **list of diagnostics**. Each entry has:

| Key | Meaning |
|---|---|
| `name` | A label for this diagnostic instance, unique within the suite. It becomes the attached-database name in the output (see [Output schema](output-schema.md)). |
| `diagnostic` | The fully-qualified diagnostic class, e.g. `climateeval.diags.simple.Map`. |
| `variables` | A list of variable definitions (below). |

Each variable definition has:

| Key | Required | Meaning |
|---|---|---|
| `id` | yes | Identifier for this variable within the diagnostic (also the output column name). |
| `var_name` | yes | CMIP variable name; must exist in [`variables.yml`](variables.md). |
| `frequency` | yes | Temporal frequency, e.g. `mon`. |
| `reference_data` | usually | The reference `DataSource` class, e.g. `climateeval.data.ERA5Monthly`. Use [`LocalCMORDataSource`](data-sources.md#locally-staged-cmor-data) to point at your own CMOR-compliant files (e.g. a held-out model, for perfect-model tests). |
| `reference_timerange` | no | `"YYYYMMDD/YYYYMMDD"`: a window for the reference only; the model and `other_data` keep `timerange`. See [Reference windows](#reference-windows). |
| `other_data` | no | List of additional `DataSource` classes to show alongside the reference. |
| `vertical_level` | no | Select a level for 3-D variables: `{coordinate_name, units, value}`. |
| `units` | no | Override/convert the variable's units, e.g. `mm day-1`. |

Additional [`Variable`](../src/climateeval/_variable.py) fields (spatial bounds
`min_lat`/`max_lat`/`min_lon`/`max_lon`, `landsea_mask`, `ar6_regions_mask`,
`anomaly`, …) can also be set per variable.

### Example

From [`Tier2_atmosphere_monthly.yml`](../src/climateeval/suites/Tier2_atmosphere_monthly.yml)
(abridged). Note the YAML **anchor** `&2d_variables` defined on the first
diagnostic and reused with `*2d_variables` on the others, so the same variable
list is evaluated by several diagnostics without repetition:

```yaml
- name: annual_cycle
  diagnostic: climateeval.diags.simple.AnnualCycle
  variables: &2d_variables
    - id: pr
      var_name: pr
      frequency: mon
      units: mm day-1                        # unit conversion
      reference_data: climateeval.data.GPCP
      other_data: [climateeval.data.CMIP6HistoricalR1I1P1F1]
    - id: ta200
      var_name: ta
      frequency: mon
      vertical_level:                        # pick 200 hPa from a 3-D field
        coordinate_name: air_pressure
        units: hPa
        value: 200.0
      reference_data: climateeval.data.ERA5Monthly
    # ... more variables ...

- name: map
  diagnostic: climateeval.diags.simple.Map
  variables: *2d_variables                   # reuse the same list

- name: zonal_profile
  diagnostic: climateeval.diags.simple.ZonalProfileAirPressure
  variables:
    - id: ta
      var_name: ta
      frequency: mon
      reference_data: climateeval.data.ERA5Monthly
```

## Reference windows

By default one window (`timerange`, set per variable or with
`variable_kwargs={"timerange": ...}` / `--timerange`) applies to the model and
to every reference. A reference whose record does not reach the first or last
**year** of that window raises `MissingDataError`; with
`fail_on_missing_data=False` it is logged and the reference is dropped (no
metrics, no reference row for that variable). Two opt-in options relax this for
**simple** diagnostics; neither changes anything unless set.

- `reference_timerange: "YYYYMMDD/YYYYMMDD"` (per variable, next to
  `reference_data`; `YYYY` and `YYYYMM` are accepted too) gives the reference,
  and only the reference, its own window. It is a property of the reference
  binding and not a `Variable` field, because a `Variable` is also what the
  model and the other data are loaded with. It can also be passed in
  `variable_kwargs`, where it overrides the YAML.
- `clip_reference_timerange: true` (diagnostic kwarg, in the entry's
  `diagnostic_kwargs` or `Suite(diagnostic_kwargs=...)`): a reference that only
  partly covers its window is clipped to the overlap instead of being dropped,
  and the window it was given is logged (`INFO`). A reference with no overlap
  at all is still missing data. Clipping reads the reference's whole record
  before clipping, so it costs more than the plain path and is only attempted
  after the strict check fails.

When the windows differ the metrics are computed as follows. Time-mean
diagnostics (maps, zonal lines and profiles, annual and diurnal cycles,
histograms) have no time dimension, so each side is simply averaged over its own
window. For time-series diagnostics, whose model and reference series then have
different lengths (this used to be a metric error), both series are restricted
to their common period, which is logged; with no common period the metric fails
like any other metric error (`fail_on_metric_error`). Series of equal length are
never touched. The raw output keeps each series over its own period.
The `variables` table gets a `reference_timerange` column, but only if some
variable sets the option, so existing outputs keep their columns.

```yaml
- name: toa_map
  diagnostic: climateeval.diags.simple.Map
  diagnostic_kwargs:
    clip_reference_timerange: true       # CERES-EBAF starts in 2000-03
  variables:
    - id: rsut
      var_name: rsut
      frequency: mon
      timerange: 19850101/20241231       # model window
      reference_data: climateeval.data.CERESEBAF
      reference_timerange: 20010101/20241231   # reference-only window
```

## Regridding grid

Diagnostics that regrid do so to a regular grid, `2x2` by default (`1x1` for the
sea-ice diagnostics). The `grid` diagnostic kwarg changes it; the value is any
ESMValCore `target_grid` string such as `1x1` or `2.5x2.5`. It works for simple
and complex diagnostics and is ignored by diagnostics that do not regrid, so it
can be given to any of them:

```yaml
- name: map_fine
  diagnostic: climateeval.diags.simple.Map
  diagnostic_kwargs:            # complex diagnostics: additional_diagnostic_kwargs
    grid: 1x1
  variables: [...]
```

`Suite(..., diagnostic_kwargs={"grid": "1x1"})` sets it for every diagnostic of
the suite, overriding entries (as for all diagnostic kwargs). The grid is not
written to the output tables; for reproducibility keep it in the suite or the
call that produced the output. Land diagnostics regrid conservatively to the
same grid.

## Overriding at run time

`Suite(...)` takes `diagnostic_kwargs` and `variable_kwargs` that are applied on
top of the YAML — handy for changing the time range or reference behaviour
without editing the file:

```python
Suite(
    "Tier2_atmosphere_monthly",
    diagnostic_kwargs={"data_root_dir": "...", "fail_on_missing_data": False},
    variable_kwargs={"timerange": "19790101/19791231"},
)
```

A common trick — benchmark against observations only by stripping the CMIP6
`other_data` entries and forcing an observational reference — is shown in
[Benchmarking your own model](benchmarking-legoesm.md#observations-only).

## Shipped suites

The suites are organised into **tiers of increasing complexity** — run them in
order to diagnose a model from the basics upward:

- **Tier 1 — does it run sanely and conserve?** Fast, cheap checks with little or
  no reference data. Run these first on any new model.
- **Tier 2 — does it get the mean climate right?** Climatological mean state
  benchmarked against observations (and CMIP6), realm by realm. The workhorse.
- **Tier 3 — does it capture variability, processes and feedbacks?** Higher-order
  behaviour that needs more (or special) data and experiments.

### Tier 1 — sanity & consistency

| Suite | Focus |
|---|---|
| `Tier1_sanity_checks` | Global-mean fluxes and state within physically reasonable ranges. |
| `Tier1_consistency_checks` | Conservation of dry-air mass, water vapour and moisture flux. |
| `Tier1_control_consistency` | Energy balance and drift, water- and atmospheric-energy-budget closure and Clausius-Clapeyron scaling of a piControl run. |

### Tier 2 — mean state vs observations

| Suite | Focus |
|---|---|
| `Tier2_atmosphere_monthly` | Atmospheric climatology (annual cycle, maps, zonal means/profiles, time series) vs ERA5/GPCP/CERES-EBAF + CMIP6. |
| `Tier2_ocean_monthly` | Ocean climatology: SST, salinity, chlorophyll, mixed-layer depth, Niño 3.4, AMOC, ocean heat content. |
| `Tier2_sea_ice_monthly` | Sea-ice area: annual cycle, seasonal minimum (NH September / SH February), maps. |
| `Tier2_land_monthly` | Land carbon fluxes (GPP) vs FLUXCOM + CMIP6: global land totals (annual cycle, time series) and maps. Needs `sftlf` and `areacella`. |

### Tier 3 — variability, processes & feedbacks

| Suite | Focus |
|---|---|
| `Tier3_dynamics` | Tropical convection (Hovmöller / MJO) and the stratospheric QBO. |
| `Tier3_atmosphere_subdaily` | Sub-daily precipitation: diurnal cycle and intensity histograms. |
| `Tier3_ecs` | Effective climate sensitivity from abrupt-4xCO2 / piControl experiments. |
| `Tier3_forced_response` | Equilibrium forced responses (land-ocean warming ratio, Arctic amplification) from abrupt-4xCO2 / piControl. |
| `Tier3_covariances` | Thermodynamic covariances of a historical run: clear-sky longwave feedback and the tropical precipitation-buoyancy relationship. |
| `Tier3_transport` | Implied meridional heat transport partitioning (atmosphere vs ocean) from a piControl run. |
| `Tier3_daily_dynamics` | Midlatitude geostrophic balance and the MJO eastward power ratio, from daily output. |
| `Tier3_coupled_variability` | Bjerknes compensation of the atmospheric and oceanic heat transports and ENSO teleconnection patterns, from a piControl run. |
| `Tier3_ocean_bgc_coupling` | Physical-biogeochemical coupling of surface chlorophyll: lag correlations with mixed-layer depth, SST and sea ice per basin and their seasonality, compared with ESA CCI OC, HadISST and ORAS5. |
| `Tier3_itcz_efe` | Seasonal migration of the ITCZ with the cross-equatorial atmospheric energy transport, from a historical run. |
| `Tier3_amip_erf` | Fixed-SST effective radiative forcing of quadrupled CO2, from amip-4xCO2 and amip. |
| `Tier3_patch_feedback` | Dependence of the radiative feedback on where the ocean warms, from the GFMIP SST patch experiments. |
| `Tier3_aerosol_erf` | Aerosol effective radiative forcing of the DAMIP aerosol-only historical run. |

Open the corresponding file in
[`src/climateeval/suites/`](../src/climateeval/suites/) to see the exact
variables and references each one uses.

## Available diagnostics

Use these class paths in the `diagnostic:` field. **Simple** diagnostics
(`climateeval.diags.simple.*`): `Map`, `ZonalLine`, `ZonalProfileAirPressure`,
`AnnualMeanTimeSeries`, `AnnualCycle`, `AnnualCycleSpatialDistribution`,
`DiurnalCycle`, `Histogram`, `Hovmoller`, `QBO`, `Nino34`,
`SeaIceAreaAnnualCycle`, `SeaIceAreaTimeSeries`, `SeaIceExtentAnnualCycle`,
`SeaIceExtentTimeSeries`, `LandTotalAnnualCycle`, `LandTotalTimeSeries`,
`LandMap`, `LandAnnualCycleSpatialDistribution`, `AMOCTimeSeries`,
`OceanHeatContentTimeSeries`. **Complex** diagnostics
(`climateeval.diags.complex.*`): `ECS`, `TEMDiagnostic`,
`LandOceanWarmingRatio`, `ArcticAmplification`, `MeridionalHeatTransport`,
`EnergyBalance`, `BudgetClosure`, `ClausiusClapeyronScaling`,
`ClearSkyLongwaveFeedback`, `PrecipitationBuoyancy`, `GeostrophicBalance`,
`MJOEastWestPowerRatio`, `ITCZEnergyFluxEquator`, `BjerknesCompensation`,
`ENSOTeleconnectionPatterns`, `ChlorophyllDriverCoupling`, `Amip4xCO2ERF`, `GFMIPPatchFeedback`,
`AerosolERF`.

How each renders in a report (and its coordinate columns) is summarised in
[Reporting](reporting.md#figure-types).

Sea ice comes in two flavours: the `SeaIceArea*` diagnostics integrate the
sea-ice concentration (the area actually covered by ice), the `SeaIceExtent*`
ones sum the area of every cell whose concentration reaches 15 % — the NSIDC /
OSI SAF **extent**, which is how most sea-ice observations are reported. Both
take the same variables and render as the same figure types.

Land fluxes such as `gpp` are given per unit land area, so the `Land*`
diagnostics weight them by the land fraction `f` (`sftlf`) and the cell area
`A` (`areacella`). The data to benchmark must provide both (they are not
listed by `climateeval check`); reference and other data bring their own, and
datasets without a land fraction are skipped with a warning.

- `LandTotalAnnualCycle` and `LandTotalTimeSeries` integrate `sum(x * f * A)`
  on each dataset's native grid, without regridding, in `Pg yr-1`.
- `LandMap` and `LandAnnualCycleSpatialDistribution` regrid the climatology
  conservatively as `R(x * f) / R(f)` (unstructured grids via an ESMF mesh),
  which conserves the land total, and weight their metrics by the land area
  `A * R(f)` on the cells where both datasets have data.

## Writing a custom suite

Copy a shipped suite, edit the diagnostics/variables/references, and point
`Suite(...)` at your file:

```python
Suite("/path/to/my_suite.yml", variable_kwargs={"timerange": "19790101/19791231"})
```

Every `var_name` must be in [`variables.yml`](variables.md); to add a new
variable or reference dataset, see [Data sources](data-sources.md) and
[Contributing](contributing.md).
