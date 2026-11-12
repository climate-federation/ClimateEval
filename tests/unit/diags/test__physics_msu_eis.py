"""Test the MSU weighting and inversion-strength formulae of ``_physics``."""

from __future__ import annotations

import numpy as np
import pytest

from climateeval.diags import _physics

PLEV19 = np.array(
    [
        100000.0,
        92500.0,
        85000.0,
        70000.0,
        60000.0,
        50000.0,
        40000.0,
        30000.0,
        25000.0,
        20000.0,
        15000.0,
        10000.0,
        7000.0,
        5000.0,
        3000.0,
        2000.0,
        1000.0,
        500.0,
        100.0,
    ],
)
PS_STANDARD = np.array(101325.0)


@pytest.mark.parametrize("channel", _physics.MSU_CHANNELS)
@pytest.mark.parametrize("surface", ["land", "ocean"])
def test_vendored_tables_are_complete(channel: str, surface: str) -> None:
    table = _physics.load_msu_weighting_function(channel, surface)

    n = len(table["height"])
    assert n == 300
    assert table["height"][0] == 0.0
    assert table["pressure"][0] == pytest.approx(101325.0)
    assert np.all(np.diff(table["pressure"]) < 0.0)
    # Atmosphere plus surface weight integrate to one (to the tabulation accuracy)
    height_km = table["height"] / 1000.0
    atmosphere = np.sum(
        0.5 * (table["weight"][1:] + table["weight"][:-1]) * np.diff(height_km),
    )
    assert atmosphere + table["surface_weight"] == pytest.approx(1.0, abs=1.5e-2)


def test_published_surface_weights() -> None:
    # Values quoted in the headers of the RSS files
    expected = {
        ("tlt", "ocean"): 0.11863,
        ("tlt", "land"): 0.15104,
        ("tmt", "ocean"): 0.04976,
        ("tmt", "land"): 0.06527,
        ("tls", "ocean"): 0.0,
    }
    for (channel, surface), weight in expected.items():
        table = _physics.load_msu_weighting_function(channel, surface)
        assert table["surface_weight"] == pytest.approx(weight)


def test_unknown_channel_or_surface() -> None:
    with pytest.raises(ValueError, match="channel"):
        _physics.load_msu_weighting_function("tmx")
    with pytest.raises(ValueError, match="surface"):
        _physics.load_msu_weighting_function("tmt", "ice")


@pytest.mark.parametrize("channel", _physics.MSU_CHANNELS)
@pytest.mark.parametrize("ps", [101325.0, 101800.0, 85000.0, 60000.0])
def test_isothermal_atmosphere_returns_its_temperature(
    channel: str,
    ps: float,
) -> None:
    ta = np.full((PLEV19.size, 3), 250.0)

    result = _physics.msu_weighted_temperature(
        ta,
        PLEV19,
        np.full(3, 250.0),
        np.full(3, ps),
        channel,
        land_fraction=np.array([0.0, 0.4, 1.0]),
    )

    np.testing.assert_allclose(result, 250.0)


def test_weights_sum_to_one_for_a_standard_surface_and_truncate_below() -> None:
    for channel in ("tlt", "tmt"):
        weights, surface_weight = _physics.msu_layer_weights(
            PLEV19,
            PS_STANDARD,
            channel,
        )
        assert weights.sum() + surface_weight == pytest.approx(1.0, abs=1.5e-2)

        # Truncation at 600 hPa removes the weight of everything below
        weights_high, _ = _physics.msu_layer_weights(
            PLEV19,
            np.array(60000.0),
            channel,
        )
        assert np.all(weights_high[PLEV19 > 60000.0] == 0.0)
        assert weights_high.sum() < weights.sum()
        # ... but the brightness temperature stays a weighted mean (see isothermal)


def test_levels_order_is_irrelevant() -> None:
    rng = np.random.default_rng(0)
    ta = 220.0 + 60.0 * rng.random((PLEV19.size, 2))
    ts = np.array([288.0, 300.0])
    ps = np.array([101000.0, 95000.0])

    result = _physics.msu_weighted_temperature(ta, PLEV19, ts, ps, "tmt")
    reordered = _physics.msu_weighted_temperature(
        ta[::-1],
        PLEV19[::-1],
        ts,
        ps,
        "tmt",
    )

    np.testing.assert_allclose(result, reordered)


def _layer_shares(channel: str) -> dict[str, float]:
    weights, surface_weight = _physics.msu_layer_weights(PLEV19, PS_STANDARD, channel)
    total = weights.sum() + surface_weight
    return {
        "stratosphere": weights[PLEV19 <= 10000.0].sum() / total,
        "peak_pressure": float(PLEV19[np.argmax(weights)]),
        "mean_pressure": float((weights * PLEV19).sum() / weights.sum()),
        "surface": surface_weight / total,
    }


def test_channel_vertical_structure() -> None:
    tlt, tmt, tts, tls = (_layer_shares(c) for c in ("tlt", "tmt", "tts", "tls"))

    # TMT peaks in the mid troposphere with ~10 % of its weight above 100 hPa
    assert 30000.0 <= tmt["peak_pressure"] <= 60000.0
    assert 0.05 <= tmt["stratosphere"] <= 0.12
    # TLT peaks lower and has no net stratospheric contribution
    assert 70000.0 <= tlt["peak_pressure"] <= 85000.0
    assert abs(tlt["stratosphere"]) < 0.03
    # The channels order in height: TLT < TMT < TTS < TLS (mean pressure)
    pressures = [c["mean_pressure"] for c in (tlt, tmt, tts, tls)]
    assert pressures == sorted(pressures, reverse=True)
    # TLS is stratospheric
    assert tls["stratosphere"] > 0.8
    # The surface matters more for the lower channel
    assert tlt["surface"] > tmt["surface"] > tls["surface"] == 0.0


def test_surface_only_warming_changes_channel_by_its_surface_weight() -> None:
    ta = np.full((PLEV19.size, 1), 250.0)
    base = _physics.msu_weighted_temperature(
        ta,
        PLEV19,
        np.array([250.0]),
        np.array([101325.0]),
        "tmt",
    )
    warmer = _physics.msu_weighted_temperature(
        ta,
        PLEV19,
        np.array([251.0]),
        np.array([101325.0]),
        "tmt",
    )

    shares = _layer_shares("tmt")
    assert warmer - base == pytest.approx(shares["surface"])
    assert shares["surface"] == pytest.approx(0.05, abs=0.01)


def test_masked_levels_below_the_surface_are_missing_not_data() -> None:
    ps = np.full(2, 85000.0)
    ts = np.array([280.0, 280.0])
    rng = np.random.default_rng(1)
    ta = 230.0 + 50.0 * rng.random((PLEV19.size, 2))
    below = PLEV19 > 85000.0
    ta_nan = ta.copy()
    ta_nan[below] = np.nan
    ta_garbage = ta.copy()
    ta_garbage[below] = 1.0e9  # a fill value that must never be used

    expected = _physics.msu_weighted_temperature(ta_garbage, PLEV19, ts, ps, "tlt")
    result = _physics.msu_weighted_temperature(ta_nan, PLEV19, ts, ps, "tlt")

    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, expected)


def test_a_missing_level_above_the_surface_is_skipped() -> None:
    ta = np.full((PLEV19.size, 1), 250.0)
    ta[PLEV19 == 40000.0] = np.nan

    result = _physics.msu_weighted_temperature(
        ta,
        PLEV19,
        np.array([250.0]),
        np.array([101325.0]),
        "tmt",
    )

    np.testing.assert_allclose(result, 250.0)


def test_missing_surface_fields_give_nan() -> None:
    ta = np.full((PLEV19.size, 2), 250.0)

    result = _physics.msu_weighted_temperature(
        ta,
        PLEV19,
        np.array([np.nan, 250.0]),
        np.array([101325.0, np.nan]),
        "tmt",
    )

    assert np.isnan(result).all()


def test_land_fraction_blends_land_and_ocean() -> None:
    rng = np.random.default_rng(2)
    ta = 220.0 + 70.0 * rng.random((PLEV19.size, 1))
    args = (PLEV19, np.array([300.0]), np.array([100000.0]), "tlt")

    ocean = _physics.msu_weighted_temperature(ta, *args)
    land = _physics.msu_weighted_temperature(ta, *args, land_fraction=np.array([1.0]))
    half = _physics.msu_weighted_temperature(ta, *args, land_fraction=np.array([0.5]))

    assert land != pytest.approx(ocean, abs=1e-3)  # surface weight differs
    assert half == pytest.approx(0.5 * (land + ocean))


def test_msu_shape_errors() -> None:
    with pytest.raises(ValueError, match="expected ta"):
        _physics.msu_weighted_temperature(
            np.zeros((3, 2)),
            PLEV19,
            np.zeros(2),
            np.zeros(2),
            "tmt",
        )


def test_potential_temperature_and_lts_by_hand() -> None:
    # theta = T (1000 hPa / p)^(R/c_p), R/c_p = 287.04 / 1004.6 = 0.285726
    theta_700 = 283.0 * (1000.0 / 700.0) ** 0.285726
    theta_sfc = 290.0 * (1000.0 / 1013.0) ** 0.285726

    lts = _physics.lower_tropospheric_stability(
        np.array(283.0),
        np.array(290.0),
        np.array(101300.0),
    )

    assert theta_700 == pytest.approx(313.37, abs=0.01)
    assert theta_sfc == pytest.approx(288.93, abs=0.01)
    assert lts == pytest.approx(theta_700 - theta_sfc, abs=1e-3)
    # A standard-pressure surface at 1000 hPa has theta_sfc = T_sfc
    assert _physics.potential_temperature(290.0, 100000.0) == pytest.approx(290.0)


def test_moist_adiabat_matches_textbook_lapse_rates() -> None:
    # Moist-adiabatic lapse rate g/c_p - Gamma_theta: ~4.5-4.7 K/km at 13 degC,
    # 850 hPa, close to the dry value 9.76 K/km when cold and dry
    warm = _physics.moist_adiabatic_theta_gradient(286.5, 85000.0)
    lapse = _physics.GRAVITY / _physics.SPECIFIC_HEAT_DRY_AIR - warm
    assert 4.4e-3 < lapse < 4.8e-3
    assert warm == pytest.approx(5.19e-3, abs=5e-5)
    cold = _physics.moist_adiabatic_theta_gradient(200.0, 85000.0)
    assert abs(cold) < 1e-5
    # Saturation vapour pressure at 0 and 20 degC
    assert _physics.saturation_vapor_pressure(273.15) == pytest.approx(611.2)
    assert _physics.saturation_vapor_pressure(293.15) == pytest.approx(2339.0, rel=5e-3)


def test_eis_by_hand_calculation() -> None:
    t_sfc, t_700, ps = 288.15, 275.15, 101325.0
    # LTS from the definition
    theta_700 = t_700 * (1.0e5 / 70000.0) ** (287.04 / 1004.6)
    theta_sfc = t_sfc * (1.0e5 / ps) ** (287.04 / 1004.6)
    lts = theta_700 - theta_sfc
    # z700 = R T850 / g ln(ps / 700 hPa) with T850 = 281.65 K
    z700 = 287.04 * 281.65 / 9.80665 * np.log(ps / 70000.0)
    # LCL of 80 % relative humidity air: 125 m per kelvin of dew-point depression,
    # T - T_d = (100 - 80) / 5 = 4 K
    z_lcl = 125.0 * 4.0
    # Gamma_m at 281.65 K, 850 hPa: e_s = 11.1 hPa -> q_s = 8.2e-3, so
    # X = (1 + 0.254) / (1 + 1.394) = 0.524 and Gamma = (g / c_p)(1 - X) = 4.65 K/km
    gamma = 4.65e-3

    eis = _physics.estimated_inversion_strength(
        np.array(t_700),
        np.array(t_sfc),
        np.array(ps),
    )

    assert z700 == pytest.approx(3000.0, rel=0.03)
    assert eis == pytest.approx(lts - gamma * (z700 - z_lcl), abs=0.1)
    assert 5.0 < float(eis) < 9.0


def test_eis_dependencies() -> None:
    args = (np.array(280.0), np.array(290.0), np.array(101300.0))

    default = _physics.estimated_inversion_strength(*args)
    explicit = _physics.estimated_inversion_strength(*args, rh_sfc=80.0)
    humid = _physics.estimated_inversion_strength(*args, rh_sfc=100.0)
    dry = _physics.estimated_inversion_strength(*args, rh_sfc=50.0)
    warmer_free_troposphere = _physics.estimated_inversion_strength(
        np.array(282.0),
        args[1],
        args[2],
    )

    assert default == explicit
    # A higher LCL (drier surface air) shortens the moist-adiabatic correction
    assert dry > default > humid
    assert warmer_free_troposphere > default


def test_eis_is_smaller_than_lts() -> None:
    t700 = np.linspace(270.0, 285.0, 7)
    tsfc = np.full(7, 292.0)
    ps = np.full(7, 101300.0)

    eis = _physics.estimated_inversion_strength(t700, tsfc, ps)
    lts = _physics.lower_tropospheric_stability(t700, tsfc, ps)

    assert np.all(eis < lts)
    assert np.all(np.diff(eis) > 0.0)
