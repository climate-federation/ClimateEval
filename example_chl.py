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
    data_root_dir = Path("/work/bd1179/b384036/climateeval_input")
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
    logger.info(f"Loaded {len(cubes)} cubes")
    for c in cubes:
        logger.info(f"  {c.name()}  shape={c.shape}")

    logger.info(f"Loaded {len(cubes)} cubes from {len(paths)} paths")
    for c in cubes:
        logger.info(f"  {c.name():15s}  shape={c.shape}  "
        f"dims={[d.name() for d in c.dim_coords]}")

    from iris.cube import CubeList
    surface_cubes = CubeList()
    for cube in cubes:
        if cube.ndim == 4:
           # Find the depth-like dimension by coord name
           depth_names = {"lev", "depth", "deptht", "olevel", "z"}
           depth_coord = None
           for c in cube.dim_coords:
               if c.name().lower() in depth_names:
                  depth_coord = c
                  break
           if depth_coord is None:
               logger.warning(
                  f"{cube.name()}: 4D but no recognised depth coord; skipping")
               continue
           # Index 0 = surface (verify: depth_coord.points[0] should be the smallest)
           surface_idx = int(depth_coord.points.argmin())
           cube = cube.subset(depth_coord[surface_idx])
           # Or equivalently for simple cases:
           # cube = cube[:, surface_idx, :, :]
           logger.info(
           f"{cube.name()}: reduced to surface "
           f"(depth = {depth_coord.points[surface_idx]:.1f} "
           f"{depth_coord.units})")
        surface_cubes.append(cube)

    cubes = surface_cubes
    
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
    #         "fail_on_missing_data": True,
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
            "fail_on_missing_data": True,
            "download_missing_data": True,
        },
        variable_kwargs={
            "timerange": "20000101/20011231",
        },
    )
    print(con)


if __name__ == "__main__":
    main()

