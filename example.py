"""Small script to test everything manually."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import iris
from loguru import logger

from benchmarking.suites import AtmosphereMonthly

logger = logger.opt(colors=True)
logger.remove()
format_str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> |  <level>{message}</level>"
)
logger.add(sys.stdout, level="DEBUG", format=format_str, colorize=True)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    cubes = iris.load(
        [
            "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/tas/gn/v20190710/tas_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc",
            "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/tas/gn/v20190710/tas_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_201001-201412.nc",
            "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/pr/gn/v20190710/pr_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc",
            "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/pr/gn/v20190710/pr_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_201001-201412.nc",
            "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/rlut/gn/v20190710/rlut_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc",
            "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/rlut/gn/v20190710/rlut_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_201001-201412.nc",
            "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/rlutcs/gn/v20190710/rlutcs_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc",
            "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/rlutcs/gn/v20190710/rlutcs_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_201001-201412.nc",
            "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/ta/gn/v20190710/ta_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc",
            "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/ta/gn/v20190710/ta_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_201001-201412.nc",
        ],
    )


data_dir = Path("/work/bd0854/b309141/benchmarking_input/data")
db_path = Path.home() / "benchmarking" / "benchmarking.ddb"

suite = AtmosphereMonthly(diagnostic_kwargs={"data_dir": data_dir})
con = suite.run(cubes, "MPI-ESM1-2-LR", database_path=db_path, overwrite=True)
