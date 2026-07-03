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
| `reference_data` | usually | The reference `DataSource` class, e.g. `climateeval.data.ERA5Monthly`. |
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

| Suite | Focus |
|---|---|
| `Tier1_sanity_checks` | Fast sanity checks on the model output. |
| `Tier1_consistency_checks` | Internal consistency checks. |
| `Tier1_ecs` | Effective climate sensitivity (needs abrupt-4xCO2 / piControl runs). |
| `Tier2_atmosphere_monthly` | 2-D atmospheric fields + 3-D humidity/temperature/wind profiles across annual-cycle, spatial, map, time-series and zonal diagnostics. |
| `Tier2_atmosphere_subdaily` | Sub-daily atmospheric diagnostics (e.g. diurnal cycle). |
| `Tier2_dynamics` | Atmospheric dynamics diagnostics. |
| `Tier2_ocean_monthly` | Ocean fields (SST, salinity, chlorophyll, MLD, AMOC, OHC). |
| `Tier2_sea_ice_monthly` | Sea-ice area (annual cycle and time series). |

Open the corresponding file in
[`src/climateeval/suites/`](../src/climateeval/suites/) to see the exact
variables and references each one uses.

## Available diagnostics

Use these class paths in the `diagnostic:` field. **Simple** diagnostics
(`climateeval.diags.simple.*`): `Map`, `ZonalLine`, `ZonalProfileAirPressure`,
`AnnualMeanTimeSeries`, `AnnualCycle`, `AnnualCycleSpatialDistribution`,
`DiurnalCycle`, `Histogram`, `Hovmoller`, `QBO`, `Nino34`,
`SeaIceAreaAnnualCycle`, `SeaIceAreaTimeSeries`, `AMOCTimeSeries`,
`OceanHeatContentTimeSeries`. **Complex** diagnostics
(`climateeval.diags.complex.*`): `ECS`, `TEMDiagnostic`.

How each renders in a report (and its coordinate columns) is summarised in
[Reporting](reporting.md#figure-types).

## Writing a custom suite

Copy a shipped suite, edit the diagnostics/variables/references, and point
`Suite(...)` at your file:

```python
Suite("/path/to/my_suite.yml", variable_kwargs={"timerange": "19790101/19791231"})
```

Every `var_name` must be in [`variables.yml`](variables.md); to add a new
variable or reference dataset, see [Data sources](data-sources.md) and
[Contributing](contributing.md).
