# ClimateEval

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![arXiv](https://img.shields.io/badge/arXiv-2610.04558-b31b1b.svg)](https://arxiv.org/abs/2610.04558)

**A community diagnostic repository for the evaluation of Earth System Models
during model development.**

ClimateEval is a community diagnostic repository for the evaluation of Earth
System Models during model development which encourages broad participation.
Developed based on ICONEval ([Lauer et al., 2026](https://doi.org/10.5194/egusphere-2026-2288)),
it can be run on every new simulation, giving modelers immediate feedback and
guiding iterative model development. It complements existing tools such as
[ESMValTool](https://esmvaltool.org/) ([Eyring et al., 2020](https://doi.org/10.5194/gmd-13-3383-2020))
and the CMIP Rapid Evaluation Framework ([Hoffman et al., 2025](https://doi.org/10.5194/egusphere-2025-2685)),
which focus on the evaluation of CMIP multi-model ensembles, by providing a
flexible environment for individual modeling groups to systematically test and
diagnose their simulations during model development or before participation in
a Model Intercomparison Project.

ClimateEval employs a suite-based architecture where diagnostics (e.g., spatial
maps, time series, vertical profiles), variables, and reference datasets (e.g.,
ERA5 reanalysis, GPCP precipitation, CERES-EBAF radiative fluxes) are specified
through YAML files, allowing flexible and reproducible evaluation workflows.
Diagnostics include "simple" diagnostics, which create plots for a single
dataset and calculate metrics relative to a reference dataset, and "complex"
diagnostics, which are more flexible and can process arbitrary combinations of
multiple datasets. The modular approach also facilitates the use of custom
diagnostics, variables, and reference datasets. When executed, the suites
process input data through standardized preprocessing steps such as unit
conversion, spatial regridding, regional masking, and calculation of
statistics. This preprocessing is based on
[ESMValCore](https://github.com/ESMValGroup/ESMValCore) ([Righi et al., 2020](https://doi.org/10.5194/gmd-13-1179-2020)),
which provides a set of robust and well-established preprocessing functions.
Any missing reference data is automatically downloaded if required. Diagnostic
results (raw output and metrics calculated relative to the reference data) are
stored in a database (either in-memory or on disk for post-hoc analyses), which
can then be explored through an interactive HTML report.

ClimateEval provides the diagnostics behind the
[ClimateBench v2.0](https://arxiv.org/abs/2610.04558) benchmark.

## Why ClimateEval?

- **Fast feedback during development:** run it on every new simulation, not
  only on finished CMIP submissions.
- **One command to a report:** benchmark a model against observations and
  CMIP6 models and get a self-contained interactive HTML report.
- **Built on established tools:** preprocessing uses the well-tested
  ESMValCore functions used across the ESMValTool community.
- **Extensible by design:** diagnostics, variables, reference datasets and
  suites are plain Python classes and YAML files, so new contributions are
  easy to add and share.

## Lead Authors

- **Dr. Aytaç Paçal**, University of Tübingen, connAIx Research School for Applied AI, Heilbronn & Tübingen AI Center, Germany
- **Prof. Dr. Veronika Eyring**, University of Tübingen, connAIx Research School for Applied AI, Heilbronn & Tübingen AI Center, Germany
- **Prof. Dr. Duncan Watson-Paris**, Scripps Institution of Oceanography, University of California & Halicioglu Data Science Institute, University of California, San Diego, USA

See the [contributors page](https://github.com/climate-federation/ClimateEval/graphs/contributors)
for everyone who has contributed to ClimateEval.

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

   or, if you prefer to connect to the repository over SSH:

   ```bash
   git clone git@github.com:climate-federation/ClimateEval.git
   ```

2. Install [Pixi](https://pixi.prefix.dev/) as described in their
   [installation instructions](https://pixi.prefix.dev/latest/installation/).

3. Install ClimateEval:

   ```bash
   cd ClimateEval
   pixi install
   ```

   or, for a faster installation with slightly outdated dependencies:

   ```bash
   cd ClimateEval
   pixi install --frozen
   ```

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
  the observational references, running the suites, and switching to a
  different model).

## Contributing

ClimateEval is a community project and contributions are very welcome, whether
you add a new diagnostic, variable, reference dataset or suite, fix a bug, or
improve the documentation.

1. Open an [issue](https://github.com/climate-federation/ClimateEval/issues)
   to report a bug or propose a new diagnostic, so we can discuss it before
   you start.
2. Fork the repository and create a branch for your change.
3. Install the development environment and pre-commit hooks
   (`pixi install && pixi run pre-commit install`), and make sure
   `pixi run pre-commit run --all-files` and `pixi run pytest` pass.
4. Open a pull request describing what you changed and why.

See [docs/contributing.md](docs/contributing.md) for the development setup,
coding standards, tests and how to extend ClimateEval. Please note that this
project follows a [Code of Conduct](CODE_OF_CONDUCT.md); by participating you
agree to abide by its terms.

## Known limitations

- Data source generators do not work yet with derived variables (see <https://github.com/ESMValGroup/ESMValCore/issues/2769>).

See the [issue tracker](https://github.com/climate-federation/ClimateEval/issues)
for other known issues and planned features.

## Support

Questions, bug reports and feature requests are handled through
[GitHub Issues](https://github.com/climate-federation/ClimateEval/issues).

## Citing ClimateEval

If you use ClimateEval in your work, please cite the ClimateBench v2.0 paper
(GitHub's "Cite this repository" button gives the details from
[`CITATION.cff`](CITATION.cff)):

```bibtex
@article{watsonparris2026climatebench,
  title   = {{ClimateBench} v2.0: Probabilistic Climate Model Benchmarking},
  author  = {Watson-Parris, Duncan and Tobin, Willa and Pa{\c{c}}al, Ayta{\c{c}} and
             Schlund, Manuel and Balaji, V. and Bowman, Kevin and
             Bretherton, Chris and Caldwell, Peter M. and Chapman, Will and
             Collins, William D. and Elsaesser, Gregory S. and Gentine, Pierre and
             Hewitt, Helene and Hoyer, Stephan and Keeling, Ralph and
             Koldunov, Nikolay and Lawrence, David M. and Lessig, Christian and
             Lunt, Daniel J. and Neelin, J. David and Pritchard, Mike and
             Purkey, Sarah and Schmidt, Gavin and Schneider, Tapio and
             Schulz, Michael and Shaw, Tiffany and Simpson, Isla R. and
             Stephens, Graeme and Subramanian, Aneesh C. and Teixeira, Joao and
             Tierney, Jessica and Williams, Andrew I. L. and Zanna, Laure and
             Eyring, Veronika and Yu, Rose},
  journal = {arXiv preprint arXiv:2610.04558},
  year    = {2026},
  url     = {https://arxiv.org/abs/2610.04558}
}
```

As the preprocessing is based on ESMValCore, please also cite
[Righi et al. (2020)](https://doi.org/10.5194/gmd-13-1179-2020).


## License

ClimateEval is released under the [Apache License 2.0](LICENSE).
