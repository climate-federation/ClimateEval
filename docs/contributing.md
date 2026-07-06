# Contributing

## Development setup

Install the project with Pixi (see [Installation](installation.md)) and enable
the pre-commit hooks:

```bash
pixi install
pixi run pre-commit install
```

The `default` Pixi environment already includes the dev and test tools
(`ruff`, `pre-commit`, `pytest`, …).

## Linting and formatting

Linting is done with [ruff](https://docs.astral.sh/ruff/) (`select = ["ALL"]`),
plus `mypy`, `codespell`, `yamllint` and a few file hygiene hooks, all wired
through pre-commit and enforced in CI:

```bash
pixi run pre-commit run --all-files
```

The experimental drivers in [`scripts/`](../scripts/) are **excluded** from
pre-commit (they are unpolished, machine-specific benchmarking scripts). Code in
`src/`, `examples/` and `tests/` is expected to pass.

## Tests

Tests use `pytest` (unit tests in `tests/unit/`, integration tests in
`tests/integration/`, synthetic data helpers in `tests/sample_data/`):

```bash
pixi run pytest                 # everything (also runs doctests via --doctest-modules)
pixi run pytest -n 2            # parallel, as CI runs it
pixi run pytest tests/unit      # a subset
pixi run pytest --cov=climateeval --cov-report=html
```

## Documentation

User docs are plain Markdown in [`docs/`](.) and render on GitHub — no build
step. When you change behaviour, update the relevant page.

[`docs/variables.md`](variables.md) is **generated**. After editing
`src/climateeval/variables.yml` or `src/climateeval/coordinates.yml`, regenerate
it:

```bash
python scripts/gen_variables_doc.py
```

## Architecture

The internal architecture — module layout, the `Diagnostic` / `Suite` /
`DataSource` base classes, key patterns, and extension points (adding
diagnostics, variables, data sources, suites) — is documented in
[`../AGENTS.md`](../AGENTS.md) (also surfaced to coding agents as `CLAUDE.md`).

Extending ClimateEval, in brief:

- **New diagnostic** — subclass `Diagnostic` under `src/climateeval/diags/`,
  implement `_preprocess()` and set the `_FINAL_COORDINATES` /
  `_OUTPUT_*_TABLE` class variables; reference it by class path in a suite.
- **New variable** — add it to `variables.yml` (and any coordinate to
  `coordinates.yml`) and regenerate the docs.
- **New data source** — subclass `DataSource` under `src/climateeval/data/`.
- **New suite** — add a YAML file under `src/climateeval/suites/` (see
  [Suites](suites.md)).
