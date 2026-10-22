"""Tests for the ACC transport diagnostic through the Drake Passage."""

from __future__ import annotations

import cf_units
import numpy as np
import pytest
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

from climateeval import Variable
from climateeval.diags.simple import (
    ACCTransportZonalTransect,
    SimpleDiagnosticInputData,
)

EARTH_RADIUS = 6371000.0
METRES_PER_DEGREE = EARTH_RADIUS * np.pi / 180.0
DEPTH_BOUNDS = ((0.0, 100.0), (100.0, 300.0), (300.0, 600.0))
SVERDRUP = 1e6

# The transect spans 70S-54S: 17 grid points of a 1 degree grid, each of which
# is 1 degree wide.
N_TRANSECT_POINTS = 17


@pytest.fixture
def variable_uo() -> Variable:
    return Variable("uo", "uo", "mon")


@pytest.fixture
def diagnostic(variable_uo: Variable) -> ACCTransportZonalTransect:
    return ACCTransportZonalTransect(
        "acc_transport",
        {variable_uo: SimpleDiagnosticInputData(reference=None)},
    )


def _time_coord(n_time: int = 2) -> DimCoord:
    return DimCoord(
        30.0 * np.arange(n_time) + 15.0,
        standard_name="time",
        var_name="time",
        units=cf_units.Unit("days since 1850-01-01", calendar="360_day"),
    )


def _depth_coord(
    points: tuple[float, ...] = (50.0, 200.0, 450.0),
    bounds: tuple[tuple[float, float], ...] | None = DEPTH_BOUNDS,
) -> DimCoord:
    return DimCoord(
        points,
        bounds=bounds,
        standard_name="depth",
        var_name="lev",
        units="m",
        attributes={"positive": "down"},
    )


def _regular_uo(
    data: np.ndarray,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    depth: DimCoord | None = None,
) -> Cube:
    """Get a (time, depth, lat, lon) uo cube on a regular grid."""
    depth = _depth_coord() if depth is None else depth
    return Cube(
        data,
        var_name="uo",
        standard_name="sea_water_x_velocity",
        units="m s-1",
        dim_coords_and_dims=[
            (_time_coord(data.shape[0]), 0),
            (depth, 1),
            (
                DimCoord(
                    lat,
                    standard_name="latitude",
                    var_name="lat",
                    units="degrees_north",
                ),
                2,
            ),
            (
                DimCoord(
                    lon,
                    standard_name="longitude",
                    var_name="lon",
                    units="degrees_east",
                ),
                3,
            ),
        ],
    )


def _curvilinear_uo(data: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> Cube:
    """Get a (time, depth, j, i) uo cube with 2D latitude/longitude."""
    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
    return Cube(
        data,
        var_name="uo",
        standard_name="sea_water_x_velocity",
        units="m s-1",
        dim_coords_and_dims=[(_time_coord(data.shape[0]), 0), (_depth_coord(), 1)],
        aux_coords_and_dims=[
            (
                AuxCoord(
                    lat_2d,
                    standard_name="latitude",
                    var_name="lat",
                    units="degrees_north",
                ),
                (2, 3),
            ),
            (
                AuxCoord(
                    lon_2d,
                    standard_name="longitude",
                    var_name="lon",
                    units="degrees_east",
                ),
                (2, 3),
            ),
        ],
    )


LAT_1DEG = np.arange(-80.0, -39.0, 1.0)
LON_0_360 = np.arange(0.0, 360.0, 1.0)


def test_uniform_flow_through_drake_passage(
    diagnostic: ACCTransportZonalTransect,
    variable_uo: Variable,
) -> None:
    """U = 0.1 m/s through 17 deg x 600 m gives U * 17 * R * pi/180 * 600 m2."""
    uo = _regular_uo(
        np.full((2, 3, LAT_1DEG.size, LON_0_360.size), 0.1),
        lat=LAT_1DEG,
        lon=LON_0_360,
    )

    transport = diagnostic._safely_preprocess(uo, variable_uo)

    expected = 0.1 * N_TRANSECT_POINTS * METRES_PER_DEGREE * 600.0 / SVERDRUP
    assert expected == pytest.approx(113.4188, abs=1e-4)
    assert transport.shape == (2,)
    assert transport.units == "Sv"
    np.testing.assert_allclose(transport.data, expected, rtol=1e-12)
    # The section is at 67W, i.e. 293E on a 0-360 grid
    assert transport.long_name == "ACC_Transport_at_293.00_degrees_east"
    assert not transport.coords("longitude")
    assert transport.coord("time").var_name == "time"


def test_only_the_transect_contributes(
    diagnostic: ACCTransportZonalTransect,
    variable_uo: Variable,
) -> None:
    """Flow outside 67W and 70S-54S must not leak into the transport."""
    data = np.full((2, 3, LAT_1DEG.size, LON_0_360.size), 1000.0)
    in_band = (LAT_1DEG >= -70.0) & (LAT_1DEG <= -54.0)
    # Depth-dependent velocity (0.3, 0.2, 0.1 m/s) and time-dependent sign
    profile = np.array([0.3, 0.2, 0.1])
    data[:, :, in_band, 293] = (
        np.array([1.0, -2.0])[:, None, None] * profile[None, :, None]
    )
    uo = _regular_uo(data, lat=LAT_1DEG, lon=LON_0_360)

    transport = diagnostic._safely_preprocess(uo, variable_uo)

    dz = np.array([100.0, 200.0, 300.0])
    per_unit = N_TRANSECT_POINTS * METRES_PER_DEGREE * (profile @ dz) / SVERDRUP
    np.testing.assert_allclose(transport.data, [per_unit, -2.0 * per_unit])


def test_longitudes_from_minus_180_to_180(
    diagnostic: ACCTransportZonalTransect,
    variable_uo: Variable,
) -> None:
    lon = np.arange(-180.0, 180.0, 1.0)
    data = np.zeros((1, 3, LAT_1DEG.size, lon.size))
    data[..., lon == -67.0] = 0.05
    uo = _regular_uo(data, lat=LAT_1DEG, lon=lon)

    transport = diagnostic._safely_preprocess(uo, variable_uo)

    expected = 0.05 * N_TRANSECT_POINTS * METRES_PER_DEGREE * 600.0 / SVERDRUP
    np.testing.assert_allclose(transport.data, [expected])
    assert transport.long_name == "ACC_Transport_at_-67.00_degrees_east"


def test_nearest_longitude_is_used(
    diagnostic: ACCTransportZonalTransect,
    variable_uo: Variable,
) -> None:
    """On a 5 deg grid, 67W (293E) is closest to 295E."""
    lon = np.arange(0.0, 360.0, 5.0)
    data = np.zeros((1, 3, LAT_1DEG.size, lon.size))
    data[..., lon == 295.0] = 0.1
    uo = _regular_uo(data, lat=LAT_1DEG, lon=lon)

    transport = diagnostic._safely_preprocess(uo, variable_uo)

    expected = 0.1 * N_TRANSECT_POINTS * METRES_PER_DEGREE * 600.0 / SVERDRUP
    np.testing.assert_allclose(transport.data, [expected])
    assert transport.long_name == "ACC_Transport_at_295.00_degrees_east"


def test_layer_thickness_from_depth_points_without_bounds(
    diagnostic: ACCTransportZonalTransect,
    variable_uo: Variable,
) -> None:
    """Without bounds, dz is np.gradient of the depths: (20, 25, 30) m."""
    depth = _depth_coord(points=(10.0, 30.0, 60.0), bounds=None)
    uo = _regular_uo(
        np.full((1, 3, LAT_1DEG.size, LON_0_360.size), 0.1),
        lat=LAT_1DEG,
        lon=LON_0_360,
        depth=depth,
    )

    transport = diagnostic._safely_preprocess(uo, variable_uo)

    expected = 0.1 * N_TRANSECT_POINTS * METRES_PER_DEGREE * 75.0 / SVERDRUP
    np.testing.assert_allclose(transport.data, [expected])


def test_curvilinear_grid_with_irregular_latitudes(
    diagnostic: ACCTransportZonalTransect,
    variable_uo: Variable,
) -> None:
    """On a native grid the cell widths are np.gradient of the latitudes.

    Of the latitudes below, the transect keeps 70S, 67S, 63S, 58S and 54S,
    whose widths are 3, 3.5, 4.5, 4.5 and 4 degrees, i.e. 19.5 degrees.
    """
    lat = np.array([-76.0, -72.0, -70.0, -67.0, -63.0, -58.0, -54.0, -50.0])
    lon = np.arange(0.0, 360.0, 1.0)
    data = np.zeros((2, 3, lat.size, lon.size))
    # The whole column flows; only the latitudes in 70S-54S must count
    data[..., 293] = 0.1
    uo = _curvilinear_uo(data, lat, lon)

    transport = diagnostic._safely_preprocess(uo, variable_uo)

    expected = 0.1 * 19.5 * METRES_PER_DEGREE * 600.0 / SVERDRUP
    np.testing.assert_allclose(transport.data, expected, rtol=1e-12)
    assert transport.units == "Sv"
    assert transport.long_name == "ACC_Transport_at_293.00_degrees_east"
    assert not transport.coords("longitude")


def test_curvilinear_grid_without_transect_latitudes(
    diagnostic: ACCTransportZonalTransect,
    variable_uo: Variable,
) -> None:
    lat = np.array([-40.0, -30.0, -20.0])
    lon = np.arange(0.0, 360.0, 10.0)
    uo = _curvilinear_uo(np.zeros((1, 3, lat.size, lon.size)), lat, lon)

    with pytest.raises(ValueError, match=r"No latitude between -70\.0 and -54\.0"):
        diagnostic._safely_preprocess(uo, variable_uo)


def test_single_depth_level(
    diagnostic: ACCTransportZonalTransect,
    variable_uo: Variable,
) -> None:
    depth = _depth_coord(points=(50.0,), bounds=((0.0, 100.0),))
    uo = _regular_uo(
        np.full((1, 1, LAT_1DEG.size, LON_0_360.size), 0.1),
        lat=LAT_1DEG,
        lon=LON_0_360,
        depth=depth,
    )

    transport = diagnostic._safely_preprocess(uo, variable_uo)

    expected = 0.1 * N_TRANSECT_POINTS * METRES_PER_DEGREE * 100.0 / SVERDRUP
    np.testing.assert_allclose(transport.data, [expected])
