# benchmarking

Benchmarking diagnostics for Earth System Models.

This repository provides diagnostics for comparing model output against reference
observations, and a small set of scripts for running a benchmark and exporting
the results as a portable HTML report.

## Setup

The recommended setup is a dedicated `mamba` environment named `benchmarking`.
This is safer than using a local `.venv` because the project depends on a larger
scientific Python stack, including Iris, ESMValTool, Ibis, and DuckDB.

Create the environment from the repository root:

```bash
mamba env create -f environment.yml
```

If the environment already exists and you want to refresh it:

```bash
mamba env update -f environment.yml --prune
```

Activate it and install the package in editable mode:

```bash
mamba activate benchmarking
python -m pip install -e ".[develop]"
```

The editable install makes the local `benchmarking` package importable while you
edit the source code.

## Running The Benchmark

The main manual benchmark entry point is [`example.py`](example.py). It loads a
set of example CMIP6 cubes, runs the `Tier2Atmosphere` suite, and writes a DuckDB
database to `~/benchmarking/benchmarking.ddb`.

Run it from the repository root after activating the environment:

```bash
python example.py
```

The script uses hard-coded input paths in the current example. If those files are
not available on your system, update the paths in `example.py` before running it.

To inspect the generated DuckDB output in the terminal, use [`db.py`](db.py):

```bash
python db.py
```

That script opens `~/benchmarking/benchmarking.ddb` with Ibis and prints the
available catalogs, databases, and tables.

## Generate HTML Output

Use [`generate_html.py`](generate_html.py) to turn one or more DuckDB benchmark
databases into a self-contained HTML report.

For a single benchmark database:

```bash
python generate_html.py ~/benchmarking/benchmarking.ddb
```

This creates `~/benchmarking/benchmarking_report.html` by default.

If you pass multiple `.ddb` files, the script builds a multi-tier report:

```bash
python generate_html.py tier1.ddb tier2.ddb tier3.ddb
```

Useful optional flags:

```bash
python generate_html.py ~/benchmarking/benchmarking.ddb \
	--model MPI-ESM1-2-LR \
	--user "Your Name" \
	--developer "diffESM Benchmarking"
```

Use `--online` if you want Plotly.js loaded from the CDN instead of embedded in
the HTML file.

## Reference

More implementation details and the current code layout are documented in
[`CLAUDE.md`](CLAUDE.md).
