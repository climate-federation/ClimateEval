"""Small script to test everything manually."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import iris
from loguru import logger

from benchmarking.diags.complex import ECS

logger = logger.opt(colors=True)
logger.remove()
format_str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> |  <level>{message}</level>"
)
logger.add(sys.stdout, level="DEBUG", format=format_str, colorize=True)

from benchmarking.data import DataSourceInformation  # noqa: E402
from benchmarking.suites import Suite  # noqa: E402

variables = (
    "rsdt",
    "rsut",
    "rsutcs",
    "ta",
)
paths = [
    f"/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/{var}/gn/v20190710/{var}_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc"
    for var in variables
]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cubes = iris.load(paths)

data = {
    "4xco2": cubes[0:2],
    "picontrol": cubes[2:4],
}
data_info = DataSourceInformation(
    name="EXAMPLE(MPI-ESM1-2-LR)",
    category="CMIP6",
    institute="MPI-M",
    exp="historical",
    variant="r1i1p1f1",
    references=(
        "https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.MPI-M.MPI-ESM1-2-LR",
    ),
)
diagnostic = ECS()

output = diagnostic.get_output(data, data_info)

print(output)
print()
print(output.raw_output.to_pandas())
print()
print(output.variables.to_pandas())
print()
print(output.data_sources.to_pandas())
