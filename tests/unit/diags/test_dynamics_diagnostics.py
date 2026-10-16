"""Tests for the daily large-scale dynamics diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from climateeval.data import DataSourceInformation
from climateeval.diags.complex import GeostrophicBalance, MJOEastWestPowerRatio
from climateeval.diags.complex._dynamics import (
    _GRAVITY,
    EARTH_ANGULAR_VELOCITY,
    banded_pattern_correlation,
    east_west_power_ratio,
    geostrophic_wind_u,
)
from climateeval.diags.complex._transport import EARTH_RADIUS_M

if TYPE_CHECKING:
    from collections.abc import Callable

N_LAT = 36
N_LON = 36
PRESSURE_LEVELS = (100000.0, 50000.0)
HEIGHT_AMPLITUDE = 500.0


def _info() -> DataSourceInformation:
    return DataSourceInformation(
        name="SynthModel",
        category="model",
        institute="Synth",
        exp="historical",
        variant="r1i1p1f1",
    )


def _coords(n_time: int) -> tuple[DimCoord, DimCoord, DimCoord]:
    time = DimCoord(
        np.arange(n_time, dtype=float) + 0.5,
        standard_name="time",
        units=Unit("days since 1850-01-01", calendar="360_day"),
    )
    lat = DimCoord(
        np.linspace(-87.5, 87.5, N_LAT),
        standard_name="latitude",
        units="degrees",
    )
    lon = DimCoord(
        np.linspace(5.0, 355.0, N_LON),
        standard_name="longitude",
        units="degrees",
        circular=True,
    )
    for coord in (time, lat, lon):
        coord.guess_bounds()
    return time, lat, lon


def _daily_cube(
    var_name: str,
    units: str,
    data_fn: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
    *,
    n_time: int,
) -> Cube:
    """Global daily cube of a surface field."""
    time, lat, lon = _coords(n_time)
    data = data_fn(n_time, lat.points, lon.points).astype(np.float32)
    return Cube(
        data,
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
    )


def _daily_profile_cube(
    var_name: str,
    units: str,
    data_fn: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
    *,
    n_time: int,
) -> Cube:
    """Global daily cube of a field repeated on every pressure level."""
    time, lat, lon = _coords(n_time)
    plev = DimCoord(
        np.array(PRESSURE_LEVELS),
        standard_name="air_pressure",
        units="Pa",
    )
    level = data_fn(n_time, lat.points, lon.points)
    data = np.broadcast_to(
        level[:, None, :, :],
        (n_time, plev.points.size, lat.points.size, lon.points.size),
    ).astype(np.float32)
    return Cube(
        data,
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(time, 0), (plev, 1), (lat, 2), (lon, 3)],
    )


def _balanced_height(
    n_time: int,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Z = A cos^2(phi) / 2, whose geostrophic wind is analytic."""
    profile = HEIGHT_AMPLITUDE * 0.5 * np.cos(np.deg2rad(lats)) ** 2
    return np.broadcast_to(
        profile[None, :, None],
        (n_time, lats.size, lons.size),
    ).copy()


def _analytic_geostrophic_wind(lats: np.ndarray) -> np.ndarray:
    """u_g = g A cos(phi) / (2 Omega a) for the balanced height field."""
    return (
        _GRAVITY
        * HEIGHT_AMPLITUDE
        * np.cos(np.deg2rad(lats))
        / (2 * EARTH_ANGULAR_VELOCITY * EARTH_RADIUS_M)
    )


def test_geostrophic_wind_recovers_a_balanced_flow() -> None:
    lats = np.linspace(-80.0, 80.0, 81)
    profile = HEIGHT_AMPLITUDE * 0.5 * np.cos(np.deg2rad(lats)) ** 2
    height = np.broadcast_to(profile[:, None], (lats.size, 72))
    wind = geostrophic_wind_u(height, lats)
    expected = _analytic_geostrophic_wind(lats)
    band = (np.abs(lats) >= 30.0) & (np.abs(lats) <= 60.0)
    np.testing.assert_allclose(wind[band, 0], expected[band], rtol=0.01)
    # The tropics, where the geostrophic approximation fails, are masked
    assert np.isnan(wind[np.abs(lats) < 10.0]).all()


def test_banded_pattern_correlation_over_a_band() -> None:
    rng = np.random.default_rng(12)
    lats = np.linspace(-88.0, 88.0, 45)
    pattern = rng.normal(0.0, 1.0, (45, 36))
    band = (-30.0, 30.0)
    assert banded_pattern_correlation(
        pattern,
        pattern,
        lats,
        band=band,
    ) == pytest.approx(1.0)
    assert banded_pattern_correlation(
        pattern,
        -pattern,
        lats,
        band=band,
    ) == pytest.approx(-1.0)
    # Only the band matters: scrambling the extratropics changes nothing
    scrambled = pattern.copy()
    outside = np.abs(lats) > 30.0
    scrambled[outside] = rng.normal(0.0, 5.0, scrambled[outside].shape)
    assert banded_pattern_correlation(
        pattern,
        scrambled,
        lats,
        band=band,
    ) == pytest.approx(1.0)


def test_banded_pattern_correlation_is_centred_and_nan_aware() -> None:
    lats = np.linspace(-30.0, 30.0, 31)
    rng = np.random.default_rng(13)
    field = rng.normal(0.0, 1.0, (31, 20))
    # A constant offset is removed by the centring
    assert banded_pattern_correlation(field, field + 7.0, lats) == pytest.approx(1.0)
    # Points missing in one field (an SST product over land) drop from both
    masked = field.copy()
    masked[:, :5] = np.nan
    over_land = field.copy()
    over_land[:, :5] = 99.0  # would ruin the correlation if it were not masked
    assert banded_pattern_correlation(over_land, masked, lats) == pytest.approx(1.0)


def test_banded_pattern_correlation_spans_both_hemispheres() -> None:
    lats = np.linspace(-80.0, 80.0, 41)
    rng = np.random.default_rng(14)
    field = rng.normal(0.0, 1.0, (41, 36))
    other = field.copy()
    tropics = np.abs(lats) < 30.0
    other[tropics] = rng.normal(0.0, 9.0, other[tropics].shape)
    assert banded_pattern_correlation(
        field,
        other,
        lats,
        band=(30.0, 60.0),
        absolute_latitude=True,
    ) == pytest.approx(1.0)


def test_east_west_power_ratio_of_a_propagating_wave() -> None:
    """A pure eastward wavenumber-2, 45-day wave is strongly eastward."""
    n_time, n_lon = 720, 144
    time = np.arange(n_time)[:, None]
    longitude = np.arange(n_lon)[None, :] / n_lon
    eastward = np.cos(2 * np.pi * (2 * longitude - time / 45.0))
    assert east_west_power_ratio(eastward) > 10.0

    westward = np.cos(2 * np.pi * (2 * longitude + time / 45.0))
    assert east_west_power_ratio(westward) < 0.1

    rng = np.random.default_rng(2)
    noise = east_west_power_ratio(rng.normal(0.0, 1.0, (n_time, n_lon)))
    assert noise == pytest.approx(1.0, abs=0.35)


def _geostrophic_cubes(
    *,
    n_time: int = 20,
    corrupt_longitudes: tuple[float, float] | None = None,
    with_surface_pressure: bool = False,
) -> CubeList:
    """Daily winds in geostrophic balance with their geopotential height.

    With ``corrupt_longitudes`` the zonal wind is replaced by nonsense over a
    longitude sector; ``with_surface_pressure`` adds a ``ps`` field that
    flags a wider sector as lying below the 850 hPa surface.
    """

    def wind(n_time: int, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        field = np.broadcast_to(
            _analytic_geostrophic_wind(lats)[None, :, None],
            (n_time, lats.size, lons.size),
        ).copy()
        if corrupt_longitudes is not None:
            corrupt = (lons >= corrupt_longitudes[0]) & (lons <= corrupt_longitudes[1])
            field[:, :, corrupt] = -50.0
        return field

    def surface_pressure(
        n_time: int,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> np.ndarray:
        field = np.full((n_time, lats.size, lons.size), 100000.0)
        if corrupt_longitudes is not None:
            elevated = (lons >= corrupt_longitudes[0] - 40.0) & (
                lons <= corrupt_longitudes[1] + 40.0
            )
            field[:, :, elevated] = 80000.0
        return field

    cubes = CubeList(
        [
            _daily_profile_cube("ua", "m s-1", wind, n_time=n_time),
            _daily_profile_cube("zg", "m", _balanced_height, n_time=n_time),
        ],
    )
    if with_surface_pressure:
        cubes.append(_daily_cube("ps", "Pa", surface_pressure, n_time=n_time))
    return cubes


def _correlation(diag: GeostrophicBalance, cubes: CubeList) -> float:
    output = diag.get_output({"day": cubes}, _info())
    assert output.raw_output is not None
    row = output.raw_output.to_pandas().iloc[0]
    assert list(row.index) == ["data_id", "data_type", "geostrophic_correlation"]
    return float(row["geostrophic_correlation"])


def test_geostrophic_balance_of_a_balanced_atmosphere() -> None:
    diag = GeostrophicBalance(
        "geostrophic_balance",
        fail_on_missing_data=True,
        download_missing_data=False,
    )
    assert _correlation(diag, _geostrophic_cubes()) == pytest.approx(1.0, abs=0.01)


def test_geostrophic_balance_masks_sub_surface_points() -> None:
    """Nonsense winds below ground are excluded when ``ps`` is available."""
    diag = GeostrophicBalance(
        "geostrophic_balance",
        fail_on_missing_data=True,
        download_missing_data=False,
    )
    corrupt = (100.0, 200.0)
    unmasked = _correlation(
        diag,
        _geostrophic_cubes(corrupt_longitudes=corrupt),
    )
    masked = _correlation(
        diag,
        _geostrophic_cubes(corrupt_longitudes=corrupt, with_surface_pressure=True),
    )
    assert unmasked < 0.6
    assert masked == pytest.approx(1.0, abs=0.01)


def test_mjo_east_west_power_ratio_of_an_eastward_wave() -> None:
    n_time = 360

    def precipitation(
        n_time: int,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> np.ndarray:
        time = np.arange(n_time)[:, None, None]
        longitude = (lons / 360.0)[None, None, :]
        wave = np.cos(2 * np.pi * (2 * longitude - time / 45.0))
        return (3.0 + np.broadcast_to(wave, (n_time, lats.size, lons.size))) / 86400.0

    diag = MJOEastWestPowerRatio(
        "mjo_east_west_power_ratio",
        fail_on_missing_data=True,
        download_missing_data=False,
    )
    cubes = CubeList([_daily_cube("pr", "kg m-2 s-1", precipitation, n_time=n_time)])
    output = diag.get_output({"day": cubes}, _info())
    assert output.raw_output is not None
    row = output.raw_output.to_pandas().iloc[0]
    assert list(row.index) == ["data_id", "data_type", "mjo_east_west_ratio"]
    assert row["mjo_east_west_ratio"] > 10.0
