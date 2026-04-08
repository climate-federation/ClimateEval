"""Small script to test everything manually."""

from pathlib import Path

import iris
import plotly.express as px

from benchmarking import Config
from benchmarking.data import GPCPSG, HadCRUT5
from benchmarking.diags import TimeSeries

reference_data = {
    "pr": GPCPSG,
    "tas": HadCRUT5,
}
config = Config(
    output_dir=Path.home() / "benchmarking_output",
    timerange="19950101/20141231",
)

diag = TimeSeries(
    reference_data=reference_data,
    other_data={},
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

df = diag.get_table(cubes, "MPI-ESM1-2-LR")

path = Path.home() / "benchmarking" / "index.html"
fig = px.line(
    data_frame=df,
    x="year",
    y="tas",
    color="Dataset",
    template="simple_white",
)
fig.write_html(Path.home() / "benchmarking" / "index.html")
print(path)
