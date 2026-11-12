"""Test the circulation-index formulae of ``climateeval.diags._physics``."""

from __future__ import annotations

import numpy as np
import pytest

from climateeval.diags import _physics

LAT = np.arange(-89.0, 90.0, 2.0)
PLEV = np.arange(2500.0, 100001.0, 2500.0)  # Pa, 40 levels, top first


def _sin_profile_v(amplitude: np.ndarray) -> np.ndarray:
    """v(p, phi) = A(phi) sin(pi p / 1e5), whose integral is analytic."""
    return np.sin(np.pi * PLEV / 1.0e5)[:, None] * amplitude[None, :]


def test_streamfunction_analytic() -> None:
    amplitude = 2.0 * np.cos(np.deg2rad(LAT))
    v = _sin_profile_v(amplitude)

    psi = _physics.meridional_streamfunction(v, PLEV, LAT)

    # int_0^p A sin(pi p'/P) dp' = A P / pi (1 - cos(pi p / P)); at P/2: A P / pi
    k = int(np.flatnonzero(PLEV == 50000.0)[0])
    expected = (
        2.0
        * np.pi
        * _physics.EARTH_RADIUS
        * np.cos(np.deg2rad(LAT))
        / _physics.GRAVITY
        * amplitude
        * 1.0e5
        / np.pi
    )
    # The top 2500 Pa are taken as constant v: a 0.3 % effect on the integral
    np.testing.assert_allclose(psi[k], expected, rtol=5e-3)
    # The streamfunction vanishes at the surface (no net mass flux): int_0^P = 2AP/pi
    # for this profile, so it is *not* zero here; check the factor-2 relation instead
    np.testing.assert_allclose(psi[-1], 2.0 * psi[k], rtol=5e-3)


def test_streamfunction_level_order_is_irrelevant() -> None:
    v = _sin_profile_v(np.cos(np.deg2rad(LAT)))
    psi = _physics.meridional_streamfunction(v, PLEV, LAT)

    psi_reversed = _physics.meridional_streamfunction(v[::-1], PLEV[::-1], LAT)

    np.testing.assert_allclose(psi_reversed[::-1], psi)


def test_streamfunction_sign_northern_hadley_cell_positive() -> None:
    # Poleward flow aloft (v>0 in the upper half), equatorward below: clockwise
    v = np.where(PLEV[:, None] < 50000.0, 1.0, -1.0) * np.ones((PLEV.size, LAT.size))

    psi = _physics.meridional_streamfunction(v, PLEV, LAT)

    assert psi[PLEV == 50000.0][0, LAT == 1.0][0] > 0.0


def test_streamfunction_missing_levels_do_not_contaminate_levels_above() -> None:
    v = _sin_profile_v(np.cos(np.deg2rad(LAT)))
    reference = _physics.meridional_streamfunction(v, PLEV, LAT)
    below_ground = PLEV >= 90000.0
    v_masked = np.where(below_ground[:, None], np.nan, v)

    psi = _physics.meridional_streamfunction(v_masked, PLEV, LAT)

    np.testing.assert_allclose(psi[~below_ground], reference[~below_ground])
    assert np.isnan(psi[below_ground]).all()


def test_streamfunction_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="expected"):
        _physics.meridional_streamfunction(np.zeros((3, 4)), PLEV, LAT)


def test_interpolate_to_pressure() -> None:
    field = np.stack([np.full(3, p) for p in PLEV])  # field equals its pressure

    np.testing.assert_allclose(
        _physics.interpolate_to_pressure(field, PLEV, 51000.0),
        51000.0,
    )
    # An exact level is returned even if a neighbour is missing
    field[PLEV == 52500.0] = np.nan
    np.testing.assert_allclose(
        _physics.interpolate_to_pressure(field, PLEV, 50000.0),
        50000.0,
    )
    assert np.isnan(_physics.interpolate_to_pressure(field, PLEV, 51000.0)).all()
    assert np.isnan(_physics.interpolate_to_pressure(field, PLEV, 1.0e6)).all()


@pytest.mark.parametrize(
    ("hemisphere", "edge", "sign"),
    [("north", 35.0, 1.0), ("south", -37.0, -1.0)],
)
def test_hadley_cell_edge_known_zero(
    hemisphere: str,
    edge: float,
    sign: float,
) -> None:
    # psi = sign * sin(pi |phi| / |edge|) for |phi| < 2 |edge|: a cell reaching |edge|
    psi = sign * np.sin(np.pi * np.abs(LAT) / abs(edge)) * (np.sign(LAT) == sign)
    psi = np.where(np.sign(LAT) == sign, psi, -0.3 * sign * np.abs(psi))

    result = _physics.hadley_cell_edge(psi, LAT, hemisphere)

    assert result == pytest.approx(edge, abs=0.05)


def test_hadley_cell_edge_interpolates_linearly_between_grid_points() -> None:
    lat = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    psi = np.array([0.0, 2.0, 1.0, -3.0, -2.0])

    # Linear interpolation between (20, 1) and (30, -3): zero at 22.5
    assert _physics.hadley_cell_edge(psi, lat, "north") == pytest.approx(22.5)
    # Descending latitude gives the same answer
    assert _physics.hadley_cell_edge(psi[::-1], lat[::-1], "north") == pytest.approx(
        22.5,
    )


def test_hadley_cell_edge_undefined_cases() -> None:
    lat = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    # No cell: the streamfunction is negative everywhere in the northern hemisphere
    assert np.isnan(
        _physics.hadley_cell_edge(-np.ones_like(lat), lat, "north"),
    )
    # No crossing within the domain
    assert np.isnan(_physics.hadley_cell_edge(np.ones_like(lat), lat, "north"))
    # The crossing lies in missing data
    psi = np.array([0.0, 2.0, np.nan, np.nan, -2.0])
    assert np.isnan(_physics.hadley_cell_edge(psi, lat, "north"))
    # Everything missing
    assert np.isnan(_physics.hadley_cell_edge(np.full(5, np.nan), lat, "south"))


def test_hadley_cell_edge_ignores_the_other_hemisphere() -> None:
    lat = np.array([-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0])
    psi = np.array([2.0, 3.0, -1.0, -4.0, 2.0, -1.0, -2.0])  # NH cell, SH Ferrel-like

    assert _physics.hadley_cell_edge(psi, lat, "north") == pytest.approx(
        10.0 + 10.0 * 2.0 / 3.0,
    )


def test_hadley_cell_edge_invalid_hemisphere() -> None:
    with pytest.raises(ValueError, match="hemisphere"):
        _physics.hadley_cell_edge(np.zeros(3), np.arange(3.0), "east")


@pytest.mark.parametrize(
    ("hemisphere", "centre"),
    [("north", 47.3), ("south", -52.6), ("north", 31.2)],
)
def test_jet_latitude_recovers_gaussian(hemisphere: str, centre: float) -> None:
    u = 12.0 * np.exp(-(((LAT - centre) / 8.0) ** 2) / 2.0) + 1.0

    result = _physics.jet_latitude(u, LAT, hemisphere)

    # Finer than the 2 degree grid: the parabola fit locates the true maximum
    assert abs(result - centre) < 0.1


def test_jet_latitude_outside_range_is_not_selected() -> None:
    # A stronger jet at 80N (outside 20-70) must not be picked
    u = 5.0 * np.exp(-(((LAT - 45.0) / 8.0) ** 2) / 2.0) + 20.0 * np.exp(
        -(((LAT - 80.0) / 3.0) ** 2) / 2.0,
    )

    assert _physics.jet_latitude(u, LAT, "north") == pytest.approx(45.0, abs=0.2)


def test_jet_latitude_missing_data() -> None:
    u = 10.0 * np.exp(-(((LAT - 47.0) / 8.0) ** 2) / 2.0)
    u_missing = u.copy()
    u_missing[np.abs(LAT - 47.0) < 30.0] = np.nan  # the jet core is masked

    assert np.isnan(_physics.jet_latitude(np.full_like(u, np.nan), LAT))
    result = _physics.jet_latitude(u_missing, LAT)
    assert np.isnan(result) or abs(result - 47.0) > 1.0  # never invents a core


def test_jet_latitude_flat_profile_falls_back_to_grid_maximum() -> None:
    u = np.zeros_like(LAT)
    u[LAT == 41.0] = 1.0  # a spike: the parabola through it is concave
    result = _physics.jet_latitude(u, LAT)
    assert result == pytest.approx(41.0, abs=1.0)


def test_box_mean_matches_hand_calculation() -> None:
    lat = np.array([-4.0, 0.0, 4.0, 20.0])
    lon = np.array([100.0, 170.0, 210.0, 270.0, 300.0])
    field = np.arange(20.0).reshape(4, 5)

    result = _physics.box_mean(field, lat, lon, (-5.0, 5.0), (200.0, 280.0))

    weights = np.cos(np.deg2rad([-4.0, 0.0, 4.0]))[:, None] * np.ones((3, 2))
    expected = np.sum(weights * field[:3, 2:4]) / weights.sum()
    assert result == pytest.approx(expected)


def test_box_mean_date_line_and_nan() -> None:
    lat = np.array([0.0])
    lon = np.array([170.0, 190.0, 350.0])
    field = np.array([[1.0, 3.0, np.nan]])

    # (-20, 20) spans the 0 meridian: only 350 is in, and it is missing
    assert np.isnan(_physics.box_mean(field, lat, lon, (-1.0, 1.0), (-20.0, 20.0)))
    # 170-190 straddles the date line, written either way
    assert _physics.box_mean(field, lat, lon, (-1.0, 1.0), (170.0, 190.0)) == 2.0
    assert _physics.box_mean(field, lat, lon, (-1.0, 1.0), (170.0, -170.0)) == 2.0


def test_box_mean_empty_box() -> None:
    with pytest.raises(ValueError, match="no grid cell"):
        _physics.box_mean(
            np.zeros((1, 1)),
            np.array([60.0]),
            np.array([0.0]),
            (-5, 5),
            (0, 10),
        )


def test_walker_circulation_strength_known_box_difference() -> None:
    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(1.0, 360.0, 2.0)
    time_scale = np.array([1.0, 2.0, 3.0])
    psl = np.full((3, lat.size, lon.size), 101000.0)
    east = (np.abs(lat) <= 5.0)[:, None] & ((lon >= 200.0) & (lon <= 280.0))[None, :]
    west = (np.abs(lat) <= 5.0)[:, None] & ((lon >= 80.0) & (lon <= 160.0))[None, :]
    psl += time_scale[:, None, None] * (300.0 * east - 200.0 * west)[None]
    # Pressure outside the boxes must not matter
    psl[:, np.abs(lat) > 10.0, :] += 5000.0

    result = _physics.walker_circulation_strength(psl, lat, lon)

    np.testing.assert_allclose(result, 500.0 * time_scale)


def test_walker_circulation_strength_nan_cells_are_ignored() -> None:
    lat = np.arange(-89.0, 90.0, 2.0)
    lon = np.arange(1.0, 360.0, 2.0)
    psl = np.full((lat.size, lon.size), 100000.0)
    psl[lat.size // 2, 105] = np.nan  # one cell in the eastern box

    assert _physics.walker_circulation_strength(psl, lat, lon) == pytest.approx(0.0)
