"""Small script to test everything manually."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any

import ibis
import iris
from iris.cube import CubeList
from loguru import logger

from climateeval.data import DataSourceInformation
from climateeval.suites import Suite

logger = logger.opt(colors=True)


def run_suite(
    suite_name: str,
    cubes: CubeList,
    data_info: DataSourceInformation,
    diagnostic_kwargs: dict[str, Any],
    variable_kwargs: dict[str, Any],
) -> ibis.BaseBackend:
    """Run suite."""
    data_root_dir = Path("/work/bd1179/b309141/climateeval_input")
    db_path = Path(f"{suite_name}.ddb")
    if db_path.is_file():
        db_path.unlink()
    diagnostic_kwargs["data_root_dir"] = data_root_dir
    suite = Suite(
        suite_name,
        diagnostic_kwargs=diagnostic_kwargs,
        variable_kwargs=variable_kwargs,
    )
    return suite.get_database(cubes, data_info, database_resource=f"duckdb://{db_path}")


def main() -> None:
    """Run example script."""
    logger.remove()
    format_str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> |  <level>{message}</level>"
    )
    logger.add(sys.stdout, level="DEBUG", format=format_str, colorize=True)


    atmosphere_variables = (
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
    ocean_variables = (
        "chl",
    )
    paths = [
        f"/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r1i1p1f1/Amon/{var}/gn/v20190710/{var}_Amon_MPI-ESM1-2-LR_historical_r1i1p1f1_gn_199001-200912.nc"
        for var in atmosphere_variables
    ] + [
        f"/work/ik1017/CMIP6/data/CMIP6/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r22i1p1f1/Omon/{var}/gn/v20210901/{var}_Omon_MPI-ESM1-2-LR_historical_r22i1p1f1_gn_199001-200912.nc"
        for var in ocean_variables
    ]


    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cubes = iris.load(paths)

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

    # con = run_suite(
    #     "Tier2_atmosphere",
    #     cubes,
    #     data_info,
    #     diagnostic_kwargs={
    #         "fail_on_missing_data": False,
    #         "download_missing_data": False,
    #     },
    #     variable_kwargs={
    #         "timerange": "20000101/20011231",
    #     },
    # )
    # print(con)

    con = run_suite(
        "Tier2_ocean",
        cubes,
        data_info,
        diagnostic_kwargs={
            "fail_on_missing_data": False,
            "download_missing_data": True,
        },
        variable_kwargs={
            "timerange": "20000101/20011231",
        },
    )
    print(con)


if __name__ == "__main__":
    main()
