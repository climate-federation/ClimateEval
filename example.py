"""Small script to test everything manually."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import ibis
import iris

from benchmarking import Config, Variable
from benchmarking.data import ERA5, GPCP, HadCRUT5
from benchmarking.diags import TimeSeries

if TYPE_CHECKING:
    from benchmarking.data._base import DataSource

reference_data = {
    Variable("pr", "mon"): ERA5,
    Variable("tas", "mon"): ERA5,
    # Variable("pr", "mon"): GPCP,
    # Variable("tas", "mon"): HadCRUT5,
}
other_data: dict[Variable, list[type[DataSource]]] = {
    Variable("pr", "mon"): [GPCP],
    Variable("tas", "mon"): [HadCRUT5],
}
config = Config(
    data_dir=Path.home() / "work" / "benchmarking_input" / "data",
    timerange="19950101/20141231",
)

diag = TimeSeries(
    reference_data=reference_data,
    other_data=other_data,
    config=config,
)

cubes = iris.load(
    [
        "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/tas/gn/v20190710/tas_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc",
        "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/tas/gn/v20190710/tas_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_201001-201412.nc",
        "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/pr/gn/v20190710/pr_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc",
        "/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/pr/gn/v20190710/pr_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_201001-201412.nc",
    ],
)

table, metrics = diag.get_tables(cubes, "MPI-ESM1-2-LR")

print(table.to_pandas())
print(table.schema())
print(metrics.to_pandas())
print(metrics.schema())

db_path = Path.home() / "benchmarking.ddb"
con = ibis.connect(f"duckdb://{db_path}")
con.create_table("timeseries", table, overwrite=True)
print("Wrote", db_path)
