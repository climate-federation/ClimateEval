"""Small script to test everything manually."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import iris
from loguru import logger

from benchmarking.suites import Suite

logger = logger.opt(colors=True)
logger.remove()
format_str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> |  <level>{message}</level>"
)
logger.add(sys.stdout, level="DEBUG", format=format_str, colorize=True)

variables = (
    "hfls",
    "hfss",
    "hus",
    "pr",
    "rlut",
    "rlutcs",
    "rsdt",
    "rsut",
    "rsutcs",
    "ta",
    "tas",
    "ua",
)
paths = [
    f"/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/{var}/gn/v20190710/{var}_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc"
    for var in variables
]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cubes = iris.load(paths)


data_root_dir = Path("/work/bd0854/b309141/benchmarking_input/data")
db_path = Path("benchmarking.ddb")
if db_path.is_file():
    db_path.unlink()

suite = Suite(
    "Tier2",
    diagnostic_kwargs={
        "data_root_dir": data_root_dir,
        "fail_on_missing_data": True,
    },
    variable_kwargs={
        "timerange": "20000101/20011231",
    },
)
con = suite.get_database(cubes, "MPI-ESM1-2-LR", database_resource=f"duckdb://{db_path}")
print(con)
