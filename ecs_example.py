"""Small script to test everything manually."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import iris
from loguru import logger

from climateeval._config import setup_iris_combine_policy
from climateeval.suites import Suite

logger = logger.opt(colors=True)
logger.remove()
format_str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> |  <level>{message}</level>"
)
logger.add(sys.stdout, level="DEBUG", format=format_str, colorize=True)

from climateeval.data import DataSourceInformation  # noqa: E402

variables = (
    "rsdt",
    "rsut",
    "rlut",
    "tas",
)
years = [
    "185001-186912",
    "187001-188912",
    "189001-190912",
    "191001-192912",
    "193001-194912",
    "195001-196912",
    "197001-198912",
    "199001-200912",
]
picontrol_paths = [
    f"/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/piControl/r1i1p1f1/Amon/{var}/gn/v20190710/{var}_Amon_MPI-ESM1-2-LR_piControl_r1i1p1f1_gn_{year}.nc"
    for var in variables
    for year in years
]
abrupt4xco2_paths = [
    f"/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/abrupt-4xCO2/r1i1p1f1/Amon/{var}/gn/v20190710/{var}_Amon_MPI-ESM1-2-LR_abrupt-4xCO2_r1i1p1f1_gn_{year}.nc"
    for var in variables
    for year in years
]

with warnings.catch_warnings() and setup_iris_combine_policy():
    warnings.simplefilter("ignore")
    picontrol_cubes = iris.load(picontrol_paths)
    abrupt4xco2_cubes = iris.load(abrupt4xco2_paths)

suite_name = "Tier1"
db_path = Path(f"{suite_name}.ddb")
if db_path.is_file():
    db_path.unlink()
data_root_dir = Path("/work/bd1179/b309141/climateeval_input")
data = {
    "4xco2": abrupt4xco2_cubes,
    "picontrol": picontrol_cubes,
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

suite = Suite(
    suite_name,
    diagnostic_kwargs={
        "data_root_dir": data_root_dir,
        "fail_on_missing_data": False,
        "download_missing_data": False,
    },
)
con = suite.get_database(data, data_info, database_resource=f"duckdb://{db_path}")
print(con)
