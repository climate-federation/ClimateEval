"""Tests for the Gulf Stream SSH gradient diagnostic."""

from __future__ import annotations

import cf_units
import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube

from climateeval import Variable
from climateeval.diags.simple import GulfStreamSshGradient, SimpleDiagnosticInputData

EARTH_RADIUS = 6371000.0
METRES_PER_DEGREE = EARTH_RADIUS * np.pi / 180.0

# Points of the 2x2 degree target grid of the diagnostic
LAT_2DEG = np.arange(-89.0, 90.0, 2.0)
LON_2DEG = np.arange(1.0, 360.0, 2.0)

DAYS_PER_MONTH = np.array([31.0, 28.0, 31.0])


@pytest.fixture
def variable_zos() -> Variable:
    return Variable("zos", "zos", "mon")


@pytest.fixture
def diagnostic(variable_zos: Variable) -> GulfStreamSshGradient:
    return GulfStreamSshGradient(
        "gulf_stream_ssh",
        {variable_zos: SimpleDiagnosticInputData(reference=None)},
    )


def _zos_cube(
    data: np.ndarray,
    lat: np.ndarray = LAT_2DEG,
    lon: np.ndarray = LON_2DEG,
) -> Cube:
    """Get a monthly (time, lat, lon) SSH cube starting in January 1850."""
    n_time = data.shape[0]
    month_starts = np.concatenate([[0.0], np.cumsum(DAYS_PER_MONTH[:n_time])])
    month_bounds = np.stack([month_starts[:-1], month_starts[1:]], axis=-1)
    time = DimCoord(
        month_bounds.mean(axis=1),
        bounds=month_bounds,
        standard_name="time",
        var_name="time",
        units=cf_units.Unit("days since 1850-01-01", calendar="standard"),
    )
    lat_coord = DimCoord(
        lat,
        standard_name="latitude",
        var_name="lat",
        units="degrees_north",
    )
    lon_coord = DimCoord(
        lon,
        standard_name="longitude",
        var_name="lon",
        units="degrees_east",
        circular=True,
    )
    lat_coord.guess_bounds()
    lon_coord.guess_bounds()
    return Cube(
        data,
        var_name="zos",
        standard_name="sea_surface_height_above_geoid",
        units="m",
        dim_coords_and_dims=[(time, 0), (lat_coord, 1), (lon_coord, 2)],
    )


def _on_grid(field: np.ndarray, n_time: int = 1) -> np.ndarray:
    return np.broadcast_to(field, (n_time, *field.shape)).astype(np.float64)


def test_meridional_slope_gives_uniform_gradient(
    diagnostic: GulfStreamSshGradient,
    variable_zos: Variable,
) -> None:
    """SSH = a * lat rises by a per degree, i.e. a / (R pi/180) per metre."""
    slope = 0.01  # m per degree latitude
    lat_2d = np.broadcast_to(LAT_2DEG[:, None], (LAT_2DEG.size, LON_2DEG.size))
    zos = _zos_cube(_on_grid(slope * lat_2d))

    grad = diagnostic._safely_preprocess(zos, variable_zos)

    # Centred differences keep every grid point (one-sided at the poles)
    assert grad.shape == (LAT_2DEG.size, LON_2DEG.size)
    np.testing.assert_allclose(grad.coord("latitude").points, LAT_2DEG)
    np.testing.assert_allclose(grad.coord("longitude").points, LON_2DEG)
    assert grad.standard_name is None
    assert grad.units == "m m-1"
    assert grad.var_name == "ssh_grad_mag"
    assert grad.long_name == "SSH Gradient Magnitude"
    np.testing.assert_allclose(grad.data, slope / METRES_PER_DEGREE, rtol=1e-6)


def test_zonal_slope_scales_with_inverse_cosine_latitude(
    diagnostic: GulfStreamSshGradient,
    variable_zos: Variable,
) -> None:
    """SSH = b * lon: a degree of longitude is R cos(lat) pi/180 long."""
    slope = 0.002  # m per degree longitude
    lon_2d = np.broadcast_to(LON_2DEG, (LAT_2DEG.size, LON_2DEG.size))
    zos = _zos_cube(_on_grid(slope * lon_2d))

    grad = diagnostic._safely_preprocess(zos, variable_zos)

    # SSH = b * lon jumps at the dateline, so skip the two columns next to it
    cos_lat = np.cos(np.deg2rad(LAT_2DEG))[:, None]
    expected = np.broadcast_to(
        slope / (METRES_PER_DEGREE * cos_lat),
        grad[:, 1:-1].shape,
    )
    np.testing.assert_allclose(grad.data[:, 1:-1], expected, rtol=1e-5)
    # Gulf Stream latitudes: twice the equatorial gradient at 60N
    i_60n = int(np.argmin(np.abs(LAT_2DEG - 59.0)))
    assert grad.data[i_60n, 1] == pytest.approx(
        slope / (METRES_PER_DEGREE * np.cos(np.deg2rad(59.0))),
        rel=1e-5,
    )


def test_gradient_magnitude_combines_both_directions(
    diagnostic: GulfStreamSshGradient,
    variable_zos: Variable,
) -> None:
    """At the equator a 3-4-5 triangle of slopes gives magnitude 5."""
    lat_2d, lon_2d = np.meshgrid(LAT_2DEG, LON_2DEG, indexing="ij")
    zos = _zos_cube(_on_grid(3e-3 * lon_2d + 4e-3 * lat_2d))

    grad = diagnostic._safely_preprocess(zos, variable_zos)

    # The zonal slope jumps at the dateline, so skip the two columns next to it
    cos_lat = np.cos(np.deg2rad(LAT_2DEG))[:, None]
    expected = (
        np.sqrt((3e-3 / cos_lat) ** 2 + (4e-3) ** 2) / METRES_PER_DEGREE
    ) * np.ones(grad[:, 1:-1].shape)
    np.testing.assert_allclose(grad.data[:, 1:-1], expected, rtol=1e-5)
    i_1n = int(np.argmin(np.abs(LAT_2DEG - 1.0)))
    assert grad.data[i_1n, 1] == pytest.approx(
        5e-3 / METRES_PER_DEGREE,
        rel=2e-4,
    )


def test_time_mean_of_gradient_magnitudes(
    diagnostic: GulfStreamSshGradient,
    variable_zos: Variable,
) -> None:
    """The magnitude is averaged, so a sign-flipping slope does not cancel.

    The climatology weights the months by their length (31, 28, 31 days).
    """
    lat_2d = np.broadcast_to(LAT_2DEG[:, None], (LAT_2DEG.size, LON_2DEG.size))
    amplitudes = np.array([0.01, -0.01, 0.04])
    zos = _zos_cube(amplitudes[:, None, None] * lat_2d)

    grad = diagnostic._safely_preprocess(zos, variable_zos)

    assert grad.shape == (LAT_2DEG.size, LON_2DEG.size)
    assert not grad.coords("time", dim_coords=True)
    mean_slope = np.sum(DAYS_PER_MONTH * np.abs(amplitudes)) / DAYS_PER_MONTH.sum()
    assert mean_slope == pytest.approx(1.83 / 90.0)
    np.testing.assert_allclose(grad.data, mean_slope / METRES_PER_DEGREE, rtol=1e-6)


def test_gulf_stream_front_position(
    diagnostic: GulfStreamSshGradient,
    variable_zos: Variable,
) -> None:
    """A 1 m tanh SSH front at 40N peaks at the 39N and 41N grid points.

    Centred differences place the gradient on the grid points themselves, so
    the two points either side of the front carry the same, largest value.
    """
    width = 2.0
    ssh_1d = 0.5 * np.tanh((LAT_2DEG - 40.0) / width)
    zos = _zos_cube(
        _on_grid(np.broadcast_to(ssh_1d[:, None], (LAT_2DEG.size, LON_2DEG.size))),
    )

    grad = diagnostic._safely_preprocess(zos, variable_zos)

    lats = grad.coord("latitude").points
    profile = grad.data[:, 0]
    i_39n, i_41n = np.flatnonzero(np.isin(lats, (39.0, 41.0)))
    assert np.argmax(profile) == i_39n
    assert profile[i_41n] == pytest.approx(profile[i_39n], rel=1e-12)
    expected = np.abs(np.gradient(ssh_1d, LAT_2DEG)) / METRES_PER_DEGREE
    np.testing.assert_allclose(profile, expected, rtol=1e-6, atol=1e-20)
    # (tanh(3/w) - tanh(-1/w)) / 2 over 4 degrees of latitude
    assert profile.max() == pytest.approx(
        0.5 * (np.tanh(3.0 / width) + np.tanh(1.0 / width)) / (4.0 * METRES_PER_DEGREE),
        rel=1e-6,
    )


def test_input_is_regridded_to_2x2_degrees(
    diagnostic: GulfStreamSshGradient,
    variable_zos: Variable,
) -> None:
    """Linear interpolation of a linear field keeps its gradient exactly."""
    lat = np.arange(-89.5, 90.0, 1.0)
    lon = np.arange(0.5, 360.0, 1.0)
    slope = 0.01
    zos = _zos_cube(
        _on_grid(np.broadcast_to(slope * lat[:, None], (180, 360))),
        lat,
        lon,
    )

    grad = diagnostic._safely_preprocess(zos, variable_zos)

    assert grad.shape == (LAT_2DEG.size, LON_2DEG.size)
    # 89S and 89N are outside the range of the input points, and their
    # neighbours difference across them: skip two rows at each end
    np.testing.assert_allclose(grad.data[2:-2], slope / METRES_PER_DEGREE, rtol=1e-6)


def test_gradient_does_not_keep_ssh_standard_name(
    diagnostic: GulfStreamSshGradient,
    variable_zos: Variable,
) -> None:
    lat_2d = np.broadcast_to(LAT_2DEG[:, None], (LAT_2DEG.size, LON_2DEG.size))
    zos = _zos_cube(_on_grid(0.01 * lat_2d))

    grad = diagnostic._safely_preprocess(zos, variable_zos)

    assert grad.standard_name != "sea_surface_height_above_geoid"


def test_land_mask_spreads_to_direct_neighbours_only(
    diagnostic: GulfStreamSshGradient,
    variable_zos: Variable,
) -> None:
    """A masked (land) cell stays masked and masks its four neighbours only."""
    slope = 0.01
    lat_2d = np.broadcast_to(LAT_2DEG[:, None], (LAT_2DEG.size, LON_2DEG.size))
    data = np.ma.masked_array(_on_grid(slope * lat_2d))
    i_land, j_land = 60, 50
    data[0, i_land, j_land] = np.ma.masked
    zos = _zos_cube(data)

    grad = diagnostic._safely_preprocess(zos, variable_zos)

    mask = np.ma.getmaskarray(grad.data)
    neighbours = {
        (i_land, j_land),
        (i_land - 1, j_land),
        (i_land + 1, j_land),
        (i_land, j_land - 1),
        (i_land, j_land + 1),
    }
    assert set(zip(*np.nonzero(mask), strict=True)) == neighbours
    np.testing.assert_allclose(
        grad.data.compressed(),
        slope / METRES_PER_DEGREE,
        rtol=1e-6,
    )
