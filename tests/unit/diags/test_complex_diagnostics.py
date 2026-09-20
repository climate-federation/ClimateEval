"""Tests for the forced-response and transport complex diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from climateeval.data import DataSourceInformation
from climateeval.diags.complex import (
    ArcticAmplification,
    LandOceanWarmingRatio,
    MeridionalHeatTransport,
)
from climateeval.diags.complex._transport import implied_meridional_transport

N_YEARS = 20


def _monthly_cube(
    var_name: str,
    units: str,
    data_fn,  # noqa: ANN001 - (n_time, lats, lons) -> array
) -> Cube:
    """Global monthly cube with data from a builder function."""
    n_time = N_YEARS * 12
    time = DimCoord(
        np.arange(n_time, dtype=float) * 30.0 + 15.0,
        standard_name="time",
        units=Unit("days since 1850-01-01", calendar="360_day"),
    )
    lat = DimCoord(
        np.linspace(-85.0, 85.0, 18),
        standard_name="latitude",
        units="degrees",
    )
    lon = DimCoord(
        np.linspace(5.0, 355.0, 36),
        standard_name="longitude",
        units="degrees",
        circular=True,
    )
    lat.guess_bounds()
    lon.guess_bounds()
    time.guess_bounds()
    data = data_fn(n_time, lat.points, lon.points).astype(np.float32)
    return Cube(
        data,
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
    )


def _constant(value: float):  # noqa: ANN202
    def build(n_time: int, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        return np.full((n_time, lats.size, lons.size), value)

    return build


def _info() -> DataSourceInformation:
    return DataSourceInformation(
        name="SynthModel",
        category="model",
        institute="Synth",
        exp="piControl abrupt-4xCO2",
        variant="r1i1p1f1",
    )


def test_implied_meridional_transport_closes_globally() -> None:
    lats = np.linspace(-89.5, 89.5, 180)
    # F = A(cos(phi) - pi/4) integrates to zero over the sphere
    flux = 150.0 * (np.cos(np.deg2rad(lats)) - np.pi / 4)
    transport = implied_meridional_transport(flux, lats)
    peak = transport.max()
    assert peak > 1e15  # PW-scale poleward transport
    assert abs(transport[-1]) < 0.05 * peak  # closure at the N pole
    assert transport.min() == pytest.approx(-peak, rel=0.1)  # antisymmetry


def test_arctic_amplification_polar_warming() -> None:
    """4xCO2 warming of 2 K globally, 6 K poleward of 60N -> ratio > 1.5."""

    def warmed(n_time: int, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        base = np.full((n_time, lats.size, lons.size), 288.0)
        warming = np.where(lats >= 60.0, 6.0, 2.0)[None, :, None]
        return base + warming

    data = {
        "picontrol": CubeList([_monthly_cube("tas", "K", _constant(288.0))]),
        "4xco2": CubeList([_monthly_cube("tas", "K", warmed)]),
    }
    diag = ArcticAmplification(
        "arctic_amplification",
        fail_on_missing_data=True,
        equilibrium_years=10,
    )
    output = diag.get_output(data, _info())
    assert output.raw_output is not None
    raw = output.raw_output.to_pandas().iloc[0]
    assert raw["delta_t_global"] == pytest.approx(2.5, abs=0.4)
    assert raw["arctic_amplification"] > 1.5


def test_land_ocean_warming_ratio_is_mask_sensitive() -> None:
    """NH-weighted warming warms the (land-heavy) NH more, so land > ocean.

    A warming anomaly that increases linearly northward puts more warming
    over the land-heavy Northern Hemisphere, giving a ratio distinctly above
    1 — and below 1 if the land and sea masks were swapped. A uniform field
    (ratio == 1) would not constrain the mask orientation.
    """

    def warmed(n_time: int, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        base = np.full((n_time, lats.size, lons.size), 288.0)
        anomaly = ((lats + 90.0) / 180.0 * 6.0)[None, :, None]  # 0 K @ SP, 6 K @ NP
        return base + anomaly

    data = {
        "picontrol": CubeList([_monthly_cube("tas", "K", _constant(288.0))]),
        "4xco2": CubeList([_monthly_cube("tas", "K", warmed)]),
    }
    diag = LandOceanWarmingRatio(
        "lo_ratio",
        fail_on_missing_data=True,
        equilibrium_years=10,
    )
    output = diag.get_output(data, _info())
    assert output.raw_output is not None
    raw = output.raw_output.to_pandas()
    assert list(raw.columns) == [
        "data_id",
        "data_type",
        "land_ocean_warming_ratio",
        "delta_t_land",
        "delta_t_ocean",
    ]
    row = raw.iloc[0]
    assert row["delta_t_land"] > row["delta_t_ocean"]
    assert row["land_ocean_warming_ratio"] > 1.1


def test_forced_response_rejects_run_shorter_than_equilibrium() -> None:
    """A run shorter than equilibrium_years must fail, not silently truncate."""
    data = {
        "picontrol": CubeList([_monthly_cube("tas", "K", _constant(288.0))]),
        "4xco2": CubeList([_monthly_cube("tas", "K", _constant(291.0))]),
    }
    # The synthetic run spans N_YEARS, fewer than the default 50 equilibrium
    # years, so the diagnostic must refuse rather than average the short run.
    diag = LandOceanWarmingRatio("too_short", fail_on_missing_data=True)
    with pytest.raises(ValueError, match="equilibrium"):
        diag.get_output(data, _info())


def test_meridional_heat_transport_balanced_control() -> None:
    """Attribute all transport to the atmosphere when surface flux is zero.

    A tropically-heated, polar-cooled TOA with zero net surface flux implies
    purely atmospheric energy transport.
    """

    def toa_pattern(sign: float):  # noqa: ANN202
        def build(n_time: int, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
            profile = sign * 150.0 * (np.cos(np.deg2rad(lats)) - np.pi / 4)
            return np.broadcast_to(
                profile[None, :, None],
                (n_time, lats.size, lons.size),
            ).copy()

        return build

    zero = _constant(0.0)
    cubes = CubeList(
        [
            _monthly_cube("rsdt", "W m-2", toa_pattern(+1.0)),
            _monthly_cube("rsut", "W m-2", zero),
            _monthly_cube("rlut", "W m-2", zero),
            _monthly_cube("rsds", "W m-2", zero),
            _monthly_cube("rsus", "W m-2", zero),
            _monthly_cube("rlds", "W m-2", zero),
            _monthly_cube("rlus", "W m-2", zero),
            _monthly_cube("hfss", "W m-2", zero),
            _monthly_cube("hfls", "W m-2", zero),
        ],
    )
    diag = MeridionalHeatTransport("mht", fail_on_missing_data=True)
    output = diag.get_output({"picontrol": cubes}, _info())
    assert output.raw_output is not None
    raw = output.raw_output.to_pandas().iloc[0]
    assert raw["amet_peak"] > 1.0  # PW
    # F = A(cos(phi) - pi/4) implies a transport peaking analytically near
    # 37 degN; asserting a band tighter than the (25, 55) search window is a
    # genuine constraint, not the tautology of landing inside the search band.
    assert 30.0 <= raw["amet_peak_lat"] <= 44.0
    assert abs(raw["omet_peak"]) < 0.2  # no surface flux -> no ocean transport


def test_meridional_heat_transport_search_bands_are_configurable() -> None:
    """The AMET/OMET peak-search bands are exposed as kwargs."""
    diag = MeridionalHeatTransport(
        "mht_bands",
        amet_search_band=(30.0, 40.0),
        omet_search_band=(0.0, 10.0),
    )
    assert diag._amet_search_band == (30.0, 40.0)
    assert diag._omet_search_band == (0.0, 10.0)


def test_meridional_heat_transport_peak_is_refined_below_the_grid_spacing():
    """A peak between two grid lines is reported at its parabolic vertex."""
    import numpy as np

    lats = np.arange(-89.0, 90.0, 2.0)
    true_lat, true_val = 39.9, 4.97
    profile = true_val - 0.01 * (lats - true_lat) ** 2  # exact parabola
    peak, lat = MeridionalHeatTransport._nh_peak(profile, lats, 25.0, 55.0)
    assert abs(lat - true_lat) < 1e-6
    assert abs(peak - true_val) < 1e-6
    # An argmax on the band edge falls back to the grid cell, never NaN.
    edge_profile = lats.astype(float)  # monotonic: maximum at the band's upper edge
    peak_e, lat_e = MeridionalHeatTransport._nh_peak(edge_profile, lats, 25.0, 55.0)
    assert lat_e == 55.0 and peak_e == 55.0
    # A flat band has no vertex: the grid value is returned.
    flat = np.ones_like(lats)
    peak_f, lat_f = MeridionalHeatTransport._nh_peak(flat, lats, 25.0, 55.0)
    assert np.isfinite(lat_f) and peak_f == 1.0
