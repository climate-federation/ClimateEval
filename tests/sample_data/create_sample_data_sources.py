"""Create sample data sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import iris
import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube

ROOT_DIR: Path = Path(__file__).parent / "data_sources"

iris.FUTURE.save_split_attrs = True


def get_altitude_coord() -> DimCoord:
    return DimCoord(
        [100.0, 200.0],
        var_name="alt",
        standard_name="altitude",
        long_name="Altitude",
        units="m",
        attributes={"positive": "up"},
    )


def get_air_pressure_coord() -> DimCoord:
    return DimCoord(
        [100000.0, 80000.0],
        var_name="plev",
        standard_name="air_pressure",
        long_name="Pressure Levels",
        units="Pa",
        attributes={"positive": "down"},
    )


def get_lat_coord() -> DimCoord:
    """Get latitude coordinate."""
    return DimCoord(
        [-30.0, 0.0, 30.0],
        bounds=[[-45.0, -15.0], [-15.0, 15.0], [15.0, 45.0]],
        var_name="lat",
        standard_name="latitude",
        long_name="Latitude",
        units="degrees_north",
    )


def get_lon_coord() -> DimCoord:
    """Get longitude coordinate."""
    return DimCoord(
        [50.0, 100.0],
        bounds=[[25.0, 75.0], [75.0, 125.0]],
        var_name="lon",
        standard_name="longitude",
        long_name="Longitude",
        units="degrees_east",
    )


def get_time_coord() -> DimCoord:
    """Get time coordinate."""
    return DimCoord(
        [15.0, 380.0, 779.0],
        bounds=[[0.0, 31.0], [365.0, 396.0], [761.0, 790.0]],
        var_name="time",
        standard_name="time",
        long_name="Time Intervals",
        units="days since 1970-01-01 00:00:00",
    )


def get_simple_2d_cube(**kwargs: Any) -> Cube:
    """Get simple 2D cube."""
    return Cube(
        np.arange(3 * 3 * 2, dtype=np.float32).reshape(3, 3, 2),
        dim_coords_and_dims=[
            (get_time_coord(), 0),
            (get_lat_coord(), 1),
            (get_lon_coord(), 2),
        ],
        **kwargs,
    )


def get_simple_3d_cube(z_coord: DimCoord, **kwargs: Any) -> Cube:
    """Get simple 3D cube."""
    return Cube(
        np.arange(3 * 2 * 3 * 2, dtype=np.float32).reshape(3, 2, 3, 2),
        dim_coords_and_dims=[
            (get_time_coord(), 0),
            (z_coord, 1),
            (get_lat_coord(), 2),
            (get_lon_coord(), 3),
        ],
        **kwargs,
    )


def save(cube: Cube, rel_path: str) -> None:
    """Save cube."""
    path = ROOT_DIR / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    iris.save(cube, path)
    print(f"Saved {path}")


def main() -> None:
    """Create sample data sources."""
    era5_tas_cube = get_simple_2d_cube(
        var_name="tas",
        standard_name="air_temperature",
        long_name="Near-Surface Air Temperature",
        units="K",
    )
    era5_tas_cube.data = era5_tas_cube.core_data() * 2.0
    era5_tas_path = "reanalysis_ERA5/tas/data.nc"
    save(era5_tas_cube, era5_tas_path)

    era5_pr_cube = get_simple_2d_cube(
        var_name="pr",
        standard_name="precipitation_flux",
        long_name="Precipitation",
        units="kg m-2 s-1",
    )
    era5_pr_cube.data = era5_pr_cube.core_data()
    era5_pr_path = "reanalysis_ERA5/pr/data.nc"
    save(era5_pr_cube, era5_pr_path)

    gpcp_pr_cube = get_simple_2d_cube(
        var_name="pr",
        standard_name="precipitation_flux",
        long_name="Precipitation",
        units="kg m-2 s-1",
    )
    gpcp_pr_cube.data = gpcp_pr_cube.core_data() * 2.0
    gpcp_pr_path = "observation_GPCP/pr/data.nc"
    save(gpcp_pr_cube, gpcp_pr_path)

    era5_ta_cube = get_simple_3d_cube(
        get_air_pressure_coord(),
        var_name="ta",
        standard_name="air_temperature",
        long_name="Air Temperature",
        units="K",
    )
    era5_ta_cube.data = era5_ta_cube.core_data() * 2.0
    era5_ta_path = "reanalysis_ERA5/ta/data.nc"
    save(era5_ta_cube, era5_ta_path)


if __name__ == "__main__":
    main()
