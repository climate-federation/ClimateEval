"""Tests for the diagnostics evaluated at the RAPID array (26.5N)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

from climateeval import Variable
from climateeval.data import RAPID
from climateeval.diags._utils import DEPTH_LEVELS, extract_basin_at_latitude
from climateeval.diags.simple import (
    AMOCDepthProfile,
    AtlanticHeatTransportTimeseries,
    SimpleDiagnosticInputData,
)
from tests.sample_data.create_sample_data_sources import (
    get_lat_coord,
    get_time_coord,
)

if TYPE_CHECKING:
    from climateeval.diags.simple import SimpleDiagnostic


def _basin_coord() -> AuxCoord:
    return AuxCoord(
        ["atlantic_arctic_ocean", "global_ocean"],
        var_name="basin",
        standard_name="region",
        long_name="Ocean Basin",
    )


@pytest.fixture
def cube_msftmz() -> Cube:
    depth_coord = DimCoord(
        [5.0, 100.0],
        var_name="lev",
        standard_name="depth",
        long_name="ocean depth coordinate",
        units="m",
        attributes={"positive": "down"},
    )
    data = np.arange(3 * 2 * 2 * 3, dtype=np.float32).reshape(3, 2, 2, 3) * 1e9
    return Cube(
        data,
        var_name="msftmz",
        standard_name="ocean_meridional_overturning_mass_streamfunction",
        units="kg s-1",
        dim_coords_and_dims=[
            (get_time_coord(), 0),
            (depth_coord, 2),
            (get_lat_coord(), 3),
        ],
        aux_coords_and_dims=[(_basin_coord(), 1)],
    )


@pytest.fixture
def cube_hfbasin() -> Cube:
    data = np.arange(3 * 2 * 3, dtype=np.float32).reshape(3, 2, 3) * 1e14
    return Cube(
        data,
        var_name="hfbasin",
        standard_name="northward_ocean_heat_transport",
        units="W",
        dim_coords_and_dims=[(get_time_coord(), 0), (get_lat_coord(), 2)],
        aux_coords_and_dims=[(_basin_coord(), 1)],
    )


def test_extract_basin_at_latitude(cube_hfbasin: Cube) -> None:
    cube = extract_basin_at_latitude(cube_hfbasin, "global_ocean", 26.5)
    assert [c.name() for c in cube.dim_coords] == ["time"]
    assert not cube.coords("region")
    assert cube.coord("latitude").points == [30.0]
    np.testing.assert_allclose(cube.data, cube_hfbasin.data[:, 1, 2])


def test_extract_basin_at_latitude_unknown_basin(cube_hfbasin: Cube) -> None:
    with pytest.raises(ValueError, match="Basin 'pacific_ocean' not found"):
        extract_basin_at_latitude(cube_hfbasin, "pacific_ocean", 26.5)


def _diagnostic(cls: type[SimpleDiagnostic], var_name: str) -> SimpleDiagnostic:
    variable = Variable(var_name, var_name, "mon")
    return cls("diagnostic", {variable: SimpleDiagnosticInputData(reference=RAPID)})


def test_amoc_depth_profile(cube_msftmz: Cube) -> None:
    diagnostic = _diagnostic(AMOCDepthProfile, "msftmz")
    cube = diagnostic._safely_preprocess(
        cube_msftmz,
        Variable("msftmz", "msftmz", "mon"),
    )
    assert cube.units == "Tg s-1"
    assert [c.name() for c in cube.dim_coords] == ["depth"]
    np.testing.assert_allclose(cube.coord("depth").points, DEPTH_LEVELS)
    assert cube.coord("latitude").points == [30.0]
    # 100 m level: time-weighted mean of the Atlantic basin at 30N in Tg s-1
    weights = np.diff(cube_msftmz.coord("time").bounds, axis=1).squeeze()
    expected = np.average(cube_msftmz.data[:, 0, 1, 2], weights=weights) / 1e9
    np.testing.assert_allclose(cube.data[5], expected, rtol=1e-5)


def test_atlantic_heat_transport_timeseries(cube_hfbasin: Cube) -> None:
    diagnostic = _diagnostic(AtlanticHeatTransportTimeseries, "hfbasin")
    cube = diagnostic._safely_preprocess(
        cube_hfbasin,
        Variable("hfbasin", "hfbasin", "mon"),
    )
    assert cube.units == "PW"
    assert [c.name() for c in cube.dim_coords] == ["time"]
    assert cube.shape == (3,)
    np.testing.assert_allclose(cube.data, cube_hfbasin.data[:, 0, 2] / 1e15)
