# ClimateEval documentation

ClimateEval is a framework for evaluating and benchmarking Earth System Model
output. You define diagnostics, variables and reference datasets in YAML
*suites*; ClimateEval preprocesses the data (with
[ESMValTool](https://esmvaltool.org/)), computes metrics against observational
references, stores everything in a DuckDB database, and lets you explore the
results in an interactive HTML report.

## Contents

| Page | What it covers |
|---|---|
| [Installation](installation.md) | Requirements and installing with Pixi; HPC notes and troubleshooting. |
| [Usage](usage.md) | The core workflow: load model output, run a suite, inspect the results. |
| [Suites](suites.md) | The suite YAML format, the shipped suites, and writing your own. |
| [Variables & coordinates](variables.md) | Reference of the known variables and coordinates (auto-generated). |
| [Data sources](data-sources.md) | Reference datasets, provenance, environment variables, adding custom sources. |
| [Reporting](reporting.md) | `report.py`, the interactive server vs. portable HTML, and the figure types. |
| [Output schema](output-schema.md) | The DuckDB tables and columns a suite produces. |
| [Benchmarking your own model (legoESM)](benchmarking-legoesm.md) | Evaluate an external model; HPC array jobs. |
| [Contributing](contributing.md) | Dev setup, tests, linting, and where the architecture is documented. |

## Examples

Runnable examples live in [`../examples`](../examples):

- [`examples/quickstart`](../examples/quickstart) — run a suite, inspect the database.
- [`examples/legoesm`](../examples/legoesm) — benchmark an external model and build a report.
- [`examples/ocean`](../examples/ocean) — ocean and sea-ice diagnostics on a CMIP6 model.

For the internal architecture (module layout, extension points), see
[`../AGENTS.md`](../AGENTS.md).
