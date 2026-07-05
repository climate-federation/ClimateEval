# ClimateEval

ClimateEval is a framework for evaluating and benchmarking Earth System Model
output. It allows climate modelers to systematically assess their simulations
across multiple variables and diagnostics, and to benchmark them against
established observational references within a unified, extensible framework.

ClimateEval employs a suite-based architecture where users define diagnostics
(e.g., spatial maps, time series, vertical profiles), variables, and reference
data sets (e.g., ERA5 reanalysis, GPCP precipitation, CERES-EBAF radiative
fluxes) through YAML configuration files, allowing flexible and reproducible
evaluation workflows. The modular approach also facilitates the usage of custom
diagnostics, variables, and reference data sets. When executed, the suites
process input data through standardized preprocessing steps such as unit
conversion, spatial regridding, regional masking, calculation of statistics.
This preprocessing is based on [ESMValTool](https://esmvaltool.org/), which
provides a set of robust and well-established preprocessing functions. Any
missing reference data is automatically downloaded if required. Diagnostics
results (raw output and metrics calculated relative to the reference data) are
stored in a database (either in-memory or on disk for post-hoc analyses), which
can be then explored through an interactive HTML file.

## Documentation

Full documentation is in [`docs/`](docs/README.md):

- [Installation](docs/installation.md) — requirements, Pixi, HPC notes, troubleshooting
- [Usage](docs/usage.md) — run a suite and inspect the results
- [Suites](docs/suites.md) and [Variables & coordinates](docs/variables.md)
- [Data sources](docs/data-sources.md) and [Output schema](docs/output-schema.md)
- [Reporting](docs/reporting.md) — the interactive / portable HTML report
- [Benchmarking your own model (legoESM)](docs/benchmarking-legoesm.md)
- [Contributing](docs/contributing.md)

## Installation from source

See [docs/installation.md](docs/installation.md) for requirements, Pixi
environments and troubleshooting. In short:

1. Clone the ClimateEval repository:

   ```bash
   git clone https://github.com/climate-federation/ClimateEval.git
   ```

   or

   ```bash
   git clone git@github.com:climate-federation/ClimateEval.git
   ```

   if you prefer to connect to the repository over SSH

1. Install [Pixi](https://pixi.prefix.dev/) as described in their [installation
   instructions](https://pixi.prefix.dev/latest/installation/).

1. Install ClimateEval:

   ```bash
   cd ClimateEval
   pixi install
   ```

   or

   ```bash
   cd ClimateEval
   pixi install --frozen
   ```

   for a faster installation with slightly outdated dependencies.

Now you can use `pixi shell`/`pixi shell --frozen` to launch a shell in your
pixi environment or `pixi run ...`/`pixi run --frozen ...` to run commands in
your pixi environment.

## Quick start

Benchmark your model against ERA5, GPCP, CERES-EBAF and CMIP6 historical models
and build a self-contained interactive HTML report in one command:

```bash
pixi run climateeval benchmark /path/to/model/cmor/Amon
# → writes <model>_benchmark/: one results database per suite + an HTML report
```

Point it at your model's CMOR-compliant NetCDF output — a single file, a flat
directory, or a DRS tree; the model name is read from the file names. If your
output is not CMORized, `pixi run climateeval check /path/to/model` reports
what to fix (see [docs/input-data-spec.md](docs/input-data-spec.md)). Use
`--data-root DIR` for staged references, `--download` to fetch missing ones,
`--static` to also render static PNG figures, or `--serve` to open an
interactive server instead of writing files. Run `climateeval --help` or see
[docs/usage.md](docs/usage.md) for all options.

## Examples

- [`examples/quickstart`](examples/quickstart/README.md) — the minimal loop: run
  a suite on a model and inspect the results database.
- [`examples/legoesm`](examples/legoesm/README.md) — benchmark an external
  (non-CMIP archive) model against observations and build a report.
- [`examples/ocean`](examples/ocean/README.md) — run the ocean and sea-ice
  diagnostics on a CMIP6 model and produce an interactive HTML report (staging
  the observational references, running the suites, and switching to a different
  model).

## Known limitations

- Data source generators do not work yet with derived variables (see <https://github.com/ESMValGroup/ESMValCore/issues/2769>).
