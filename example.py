"""Small script to test everything manually."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import TYPE_CHECKING

import ibis
import iris

from benchmarking import Variable
from benchmarking.data import CERESEBAF, ERA5, GPCP, HadCRUT5, DiagnosticData
from benchmarking.diags import TimeSeries

from loguru import logger


if TYPE_CHECKING:
    from benchmarking.data._base import DataSource

logger = logger.opt(colors=True)
logger.remove()
format_str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> |  <level>{message}</level>"
)
logger.add(sys.stdout, level="DEBUG", format=format_str, colorize=True)

data = {
    Variable("my_pr", "pr", "mon", timerange="20000101/2005231"): DiagnosticData(
        reference=GPCP,
    ),
    Variable("my_tas", "tas", "mon", timerange="20030101/20061231"): DiagnosticData(
        reference=ERA5,
        other=(HadCRUT5,),
    ),
    Variable("my_lwcre", "lwcre", "mon", timerange="20000101/20101231"): DiagnosticData(
        reference=CERESEBAF,
    ),
}

diag = TimeSeries(
    data,
    data_dir=Path.home() / "work" / "benchmarking_input" / "data",
)

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
    ],
)

table, metrics = diag.get_tables(cubes, "MPI-ESM1-2-LR")

print(table.to_pandas())
print(table.schema())
print(metrics.to_pandas())
print(metrics.schema())

table_path = Path.home() / "benchmarking" / "table.html"
table.to_pandas().to_html(table_path)
print("Wrote", table_path)

db_path = Path.home() / "benchmarking.ddb"
con = ibis.connect(f"duckdb://{db_path}")
con.create_table("timeseries", table, overwrite=True)
print("Wrote", db_path)
