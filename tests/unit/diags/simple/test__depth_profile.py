from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from iris.coords import DimCoord

from climateeval import Variable
from climateeval.data import IAP
from climateeval.diags._utils import DEPTH_LEVELS
from climateeval.diags.simple import DepthProfile, SimpleDiagnosticInputData
from tests.sample_data.create_sample_data_sources import get_simple_3d_cube

if TYPE_CHECKING:
    from iris.cube import Cube


@pytest.fixture
def variable_thetao() -> Variable:
    return Variable("thetao", "thetao", "mon")


@pytest.fixture
def cube_depth(variable_thetao: Variable) -> Cube:
    depth_coord = DimCoord(
        [5.0, 100.0],
        var_name="lev",
        standard_name="depth",
        long_name="ocean depth coordinate",
        units="m",
        attributes={"positive": "down"},
    )
    return get_simple_3d_cube(
        depth_coord,
        var_name="thetao",
        standard_name="sea_water_potential_temperature",
        long_name="Sea Water Potential Temperature",
        units="degC",
    )


@pytest.fixture
def diagnostic(variable_thetao: Variable) -> DepthProfile:
    return DepthProfile(
        "diagnostic",
        {variable_thetao: SimpleDiagnosticInputData(reference=IAP)},
        lats=[0.0, 0.0],
        lons=[50.0, 100.0],
        n_points=3,
    )


def test_name(diagnostic: DepthProfile) -> None:
    assert diagnostic.name == "diagnostic"


def test_longitudes_are_normalized(variable_thetao: Variable) -> None:
    diagnostic = DepthProfile(
        "diagnostic",
        {variable_thetao: SimpleDiagnosticInputData(reference=IAP)},
        lons=[-80.0, -15.0],
    )
    assert diagnostic._lons == [280.0, 345.0]
    assert diagnostic._lats == [26.5, 26.5]


def test_requires_two_way_points(variable_thetao: Variable) -> None:
    with pytest.raises(ValueError, match="exactly two way points"):
        DepthProfile(
            "diagnostic",
            {variable_thetao: SimpleDiagnosticInputData(reference=IAP)},
            lats=[26.5],
            lons=[-30.0],
        )


def test__safely_preprocess(
    diagnostic: DepthProfile,
    cube_depth: Cube,
    variable_thetao: Variable,
) -> None:
    cube = diagnostic._safely_preprocess(cube_depth, variable_thetao)
    assert [c.name() for c in cube.dim_coords] == ["depth"]
    np.testing.assert_allclose(cube.coord("depth").points, DEPTH_LEVELS)
    # Levels within the input depth range are filled, the others masked
    assert not np.ma.is_masked(cube.data[1:6])
    assert np.ma.is_masked(cube.data[-1])
