from __future__ import annotations

import pytest
from iris.cube import CubeList

from climateeval.data import (
    DataSourceInformation,
)
from tests.sample_data.create_sample_data_sources import (
    get_air_pressure_coord,
    get_simple_2d_cube,
    get_simple_3d_cube,
)


@pytest.fixture(scope="module")
def data() -> CubeList:
    tas_cube = get_simple_2d_cube(
        var_name="tas",
        standard_name="air_temperature",
        long_name="Near-Surface Air Temperature",
        units="K",
    )
    pr_cube = get_simple_2d_cube(
        var_name="pr",
        standard_name="precipitation_flux",
        long_name="Precipitation",
        units="kg m-2 s-1",
    )
    rsdt_cube = get_simple_2d_cube(
        var_name="rsdt",
        standard_name="toa_incoming_shortwave_flux",
        long_name="TOA Incident Shortwave Radiation",
        units="W m-2",
    )
    rsut_cube = get_simple_2d_cube(
        var_name="rsut",
        standard_name="toa_outgoing_shortwave_flux",
        long_name="TOA Outgoing Shortwave Radiation",
        units="W m-2",
    )
    z_coord = get_air_pressure_coord()
    ta_cube = get_simple_3d_cube(
        z_coord,
        var_name="ta",
        standard_name="air_temperature",
        long_name="Air Temperature",
        units="K",
    )
    return CubeList([tas_cube, pr_cube, rsdt_cube, rsut_cube, ta_cube])


@pytest.fixture(scope="module")
def data_information() -> DataSourceInformation:
    return DataSourceInformation(name="N", category="C", exp="E", variant="V")
