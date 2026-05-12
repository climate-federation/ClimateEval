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

## Installation from source

1. Clone the ClimateEval repository:

   ```bash
   git clone https://github.com/EyringMLClimateGroup/ClimateEval.git
   ```

   or

   ```bash
   git clone git@github.com:EyringMLClimateGroup/ClimateEval.git
   ```

   if you prefer to connect to the repository over SSH

1. Install Pixi as described in their [installation
   instructions](https://pixi.prefix.dev/latest/installation/).

1. Install ClimateEval:

   ```bash
   cd ClimateEval
   pixi install
   ```

Now you can use `pixi shell` to launch a shell in your pixi environment or
`pixi run ...` to run commands in your pixi environment.

## Known limitations

- Data source generators do not work yet with derived variables (see <https://github.com/ESMValGroup/ESMValCore/issues/2769>).
