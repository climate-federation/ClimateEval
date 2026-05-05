"""Small script to test everything manually."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import iris
from loguru import logger

logger = logger.opt(colors=True)
logger.remove()
format_str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> |  <level>{message}</level>"
)
logger.add(sys.stdout, level="DEBUG", format=format_str, colorize=True)

from climateeval.data import DataSourceInformation  # noqa: E402
from climateeval.suites import Suite  # noqa: E402

variables = (
    "clivi",
    "clt",
    "clwvi",
    "hfls",
    "hfss",
    "hus",
    "pr",
    "prw",
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


suite_name = "Tier2_atmosphere"
data_root_dir = Path("/work/bd1179/b309141/climateeval_input")
db_path = Path(f"{suite_name}.ddb")
if db_path.is_file():
    db_path.unlink()

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

suite = Suite(
    suite_name,
    diagnostic_kwargs={
        "data_root_dir": data_root_dir,
        "fail_on_missing_data": False,
        "download_missing_data": False,
    },
    variable_kwargs={
        "timerange": "20000101/20011231",
    },
)
con = suite.get_database(cubes, data_info, database_resource=f"duckdb://{db_path}")
print(con)
