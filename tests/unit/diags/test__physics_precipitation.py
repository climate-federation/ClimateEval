"""Test the precipitation frequency/intensity formulae of ``_physics``."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from climateeval.diags import _physics

N_DAYS = 400_000
WET_PROBABILITY = 0.3
SHAPE, SCALE = 1.5, 4.0  # wet-day excess over 1 mm day-1 ~ gamma(1.5, scale 4)


@pytest.fixture(scope="module")
def gamma_rain() -> np.ndarray:
    """Get daily rain: dry (some with drizzle) or 1 mm day-1 + gamma, so wet >= 1."""
    rng = np.random.default_rng(42)
    wet = rng.random(N_DAYS) < WET_PROBABILITY
    amount = 1.0 + rng.gamma(SHAPE, SCALE, N_DAYS)
    drizzle = np.where(rng.random(N_DAYS) < 0.5, 0.01, 0.0)  # below the 0.03 bin edge
    return np.where(wet, amount, drizzle)


def test_wet_day_fraction_and_sdii_of_gamma_rain(gamma_rain: np.ndarray) -> None:
    sigma = np.sqrt(WET_PROBABILITY * (1 - WET_PROBABILITY) / N_DAYS)
    assert _physics.wet_day_fraction(gamma_rain) == pytest.approx(
        WET_PROBABILITY,
        abs=4 * sigma,
    )
    # Mean of 1 + gamma(1.5, 4) = 7 mm day-1 on wet days
    assert _physics.sdii(gamma_rain) == pytest.approx(7.0, rel=0.01)
    # A higher threshold removes the light end only
    assert _physics.wet_day_fraction(
        gamma_rain, threshold=5.0,
    ) < _physics.wet_day_fraction(
        gamma_rain,
    )


def test_wet_day_fraction_hand_case_axis_nan_and_threshold() -> None:
    pr = np.array([[0.0, 1.0], [0.99, 5.0], [np.nan, 2.0], [3.0, 0.0]])

    # Per column over time: NaN days are not counted
    np.testing.assert_allclose(_physics.wet_day_fraction(pr), [1 / 3, 3 / 4])
    # The threshold is inclusive
    assert _physics.wet_day_fraction(pr, threshold=1.0, axis=None) == pytest.approx(
        4 / 7,
    )
    np.testing.assert_allclose(_physics.sdii(pr), [3.0, 8 / 3])
    assert np.isnan(_physics.sdii(np.zeros((4, 2)))).all()
    assert np.isnan(_physics.wet_day_fraction(np.full((3,), np.nan)))


def test_weights_act_like_repeated_samples() -> None:
    pr = np.array([0.0, 2.0, 10.0, 0.5, 4.0])
    repeats = np.array([3, 1, 2, 1, 4])
    expanded = np.repeat(pr, repeats)

    for function in (_physics.wet_day_fraction, _physics.sdii):
        assert function(pr, axis=None, weights=repeats) == pytest.approx(
            function(expanded, axis=None),
        )


def test_weighted_percentiles() -> None:
    pr = np.array([2.0, 3.0, 5.0, 8.0, 13.0])

    # Equal weights change nothing
    np.testing.assert_allclose(
        _physics.wet_day_percentiles(pr, (10, 50, 90), weights=np.full(5, 7.0)),
        _physics.wet_day_percentiles(pr, (10, 50, 90)),
    )
    # The median of an odd sample is its middle value; weight moves it
    assert _physics.wet_day_percentiles(pr, (50,))[0] == pytest.approx(5.0)
    heavy = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
    assert _physics.wet_day_percentiles(pr, (50,), weights=heavy)[0] > 8.0


def test_wet_day_percentiles_of_gamma_rain(gamma_rain: np.ndarray) -> None:
    result = _physics.wet_day_percentiles(gamma_rain, (10, 50, 90))

    expected = 1.0 + stats.gamma.ppf([0.1, 0.5, 0.9], SHAPE, scale=SCALE)
    np.testing.assert_allclose(result, expected, rtol=0.02)
    assert np.all(result >= 1.0)


def test_wet_day_percentiles_without_wet_days_is_nan() -> None:
    assert np.isnan(_physics.wet_day_percentiles(np.array([0.0, 0.5]))).all()


def test_log_rain_bins() -> None:
    edges = _physics.log_rain_bins()

    assert edges.size == 151
    assert edges[0] == 0.03
    np.testing.assert_allclose(edges[1:] / edges[:-1], 1.07)
    assert 700.0 < edges[-1] < 900.0
    assert _physics.log_rain_bins(0.1, 2.0, 4).tolist() == pytest.approx(
        [0.1, 0.2, 0.4, 0.8, 1.6],
    )
    for arguments in ((0.0, 1.07, 10), (0.03, 1.0, 10), (0.03, 1.07, 0)):
        with pytest.raises(ValueError, match="need r_min"):
            _physics.log_rain_bins(*arguments)


def test_distributions_by_hand() -> None:
    edges = np.array([0.03, 1.0, 3.0, 30.0])
    pr = np.array([0.0, 0.02, 0.5, 2.0, 2.0, 10.0])

    frequency, dry = _physics.rain_frequency_distribution(pr, edges)
    amount = _physics.rain_amount_distribution(pr, edges)

    np.testing.assert_allclose(frequency, [1 / 6, 2 / 6, 1 / 6])
    assert dry == pytest.approx(2 / 6)
    np.testing.assert_allclose(
        amount,
        [
            0.5 / 6 / np.log(1.0 / 0.03),
            4.0 / 6 / np.log(3.0),
            10.0 / 6 / np.log(10.0),
        ],
    )
    # Rain above the top edge is counted in the (open-ended) last bin
    frequency_high, dry_high = _physics.rain_frequency_distribution(
        np.array([100.0, 100.0]),
        edges,
    )
    np.testing.assert_allclose(frequency_high, [0.0, 0.0, 1.0])
    assert dry_high == 0.0


def test_frequency_plus_dry_is_one_and_amount_integrates_to_mean(
    gamma_rain: np.ndarray,
) -> None:
    edges = _physics.log_rain_bins()

    frequency, dry = _physics.rain_frequency_distribution(gamma_rain, edges)
    amount = _physics.rain_amount_distribution(gamma_rain, edges)
    normalised = _physics.rain_amount_distribution(gamma_rain, edges, normalize=True)

    assert frequency.sum() + dry == pytest.approx(1.0, abs=1e-12)
    # Dry = days with no rain or drizzle: 70 % of the days, half of them drizzle
    assert dry == pytest.approx(1 - WET_PROBABILITY, abs=0.005)
    # All rain is at least 1 mm day-1 > r_min, so nothing is lost below the lowest bin
    integral = np.sum(amount * np.diff(np.log(edges)))
    assert integral == pytest.approx(
        gamma_rain[gamma_rain >= 1.0].mean() * WET_PROBABILITY, rel=0.01,
    )
    assert integral == pytest.approx(
        np.sum(gamma_rain[gamma_rain >= edges[0]]) / gamma_rain.size,
        rel=1e-10,
    )
    assert np.sum(normalised * np.diff(np.log(edges))) == pytest.approx(1.0)


def test_lognormal_rain_peaks_at_the_analytic_rates() -> None:
    # ln r ~ N(mu, sigma): the frequency per ln r peaks at exp(mu), the rain
    # amount per ln r, proportional to r f(ln r), at exp(mu + sigma^2)
    mu, sigma = np.log(2.0), 1.0
    rng = np.random.default_rng(7)
    pr = np.exp(rng.normal(mu, sigma, 1_000_000))
    edges = _physics.log_rain_bins()

    frequency, _dry = _physics.rain_frequency_distribution(pr, edges)
    amount = _physics.rain_amount_distribution(pr, edges)
    f_rate, f_height = _physics.distribution_peak(edges, frequency)
    a_rate, a_height = _physics.distribution_peak(edges, amount)

    assert f_rate == pytest.approx(np.exp(mu), rel=0.05)
    assert a_rate == pytest.approx(np.exp(mu + sigma**2), rel=0.05)
    # Height of the frequency per bin: pdf in ln r times the bin width
    pdf_peak = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    assert f_height == pytest.approx(pdf_peak * np.log(1.07), rel=0.05)
    # Amount peak: E[r] f_amount-normalised, A = r pdf(ln r)
    assert a_height == pytest.approx(
        np.exp(mu + sigma**2) * pdf_peak * np.exp(-(sigma**2) / 2.0),
        rel=0.05,
    )


def test_distribution_peak_edge_cases() -> None:
    edges = _physics.log_rain_bins(0.1, 2.0, 5)
    centres = _physics.bin_centers(edges)

    # A parabola in ln r is recovered exactly, between the grid points
    ln_peak = np.log(centres[2]) + 0.1
    values = 3.0 - (np.log(centres) - ln_peak) ** 2
    rate, height = _physics.distribution_peak(edges, values)
    assert rate == pytest.approx(np.exp(ln_peak))
    assert height == pytest.approx(3.0)
    # A maximum at the end is returned as is
    rate, height = _physics.distribution_peak(edges, np.arange(5.0))
    assert (rate, height) == (pytest.approx(centres[-1]), 4.0)
    assert all(np.isnan(_physics.distribution_peak(edges, np.full(5, np.nan))))
    with pytest.raises(ValueError, match="expected"):
        _physics.distribution_peak(edges, np.zeros(3))


def test_invalid_bin_edges() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _physics.rain_frequency_distribution(np.ones(3), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="positive"):
        _physics.rain_amount_distribution(np.ones(3), np.array([0.0, 1.0]))


def test_empty_sample_distributions_are_nan() -> None:
    edges = _physics.log_rain_bins(n=5)

    frequency, dry = _physics.rain_frequency_distribution(np.full(4, np.nan), edges)

    assert np.isnan(frequency).all()
    assert np.isnan(dry)
    assert np.isnan(_physics.rain_amount_distribution(np.full(4, np.nan), edges)).all()


def test_precipitation_statistics_summary(gamma_rain: np.ndarray) -> None:
    edges = _physics.log_rain_bins()

    summary = _physics.precipitation_statistics(gamma_rain, edges, percentiles=(10, 90))

    assert set(summary) == {
        "mean",
        "wet_fraction",
        "sdii",
        "p10",
        "p90",
        "dry_fraction",
        "amount_peak_rate",
        "amount_peak_height",
        "frequency_peak_rate",
        "frequency_peak_height",
    }
    assert summary["mean"] == pytest.approx(gamma_rain.mean())
    assert summary["wet_fraction"] == pytest.approx(WET_PROBABILITY, abs=0.005)
    assert summary["sdii"] == pytest.approx(7.0, rel=0.01)
    assert summary["p10"] < summary["sdii"] < summary["p90"]
    assert summary["dry_fraction"] == pytest.approx(1 - WET_PROBABILITY, abs=0.005)


def test_regress_on_index_recovers_a_known_line() -> None:
    rng = np.random.default_rng(3)
    index = rng.normal(size=40)
    stat = 2.0 + 0.5 * index + 0.1 * rng.normal(size=40)

    slope, intercept, stderr = _physics.regress_on_index(stat, index)

    reference = stats.linregress(index, stat)
    assert slope == pytest.approx(reference.slope)
    assert intercept == pytest.approx(reference.intercept)
    assert stderr == pytest.approx(reference.stderr)
    assert slope == pytest.approx(0.5, abs=3 * stderr)


def test_regress_on_index_nan_handling_and_many_statistics() -> None:
    rng = np.random.default_rng(4)
    index = rng.normal(size=30)
    stat = np.stack([1.0 + 2.0 * index, -3.0 * index], axis=1) + 0.01 * rng.normal(
        size=(30, 2),
    )
    stat[3, 0] = np.nan  # a missing year of one statistic only
    index_gaps = index.copy()
    index_gaps[7] = np.nan

    slope, intercept, stderr = _physics.regress_on_index(stat, index_gaps)

    assert slope.shape == intercept.shape == stderr.shape == (2,)
    np.testing.assert_allclose(slope, [2.0, -3.0], atol=0.02)
    keep = np.isfinite(stat[:, 0]) & np.isfinite(index_gaps)
    assert slope[0] == pytest.approx(
        stats.linregress(index_gaps[keep], stat[keep, 0]).slope,
    )


def test_regress_on_index_too_few_points_and_errors() -> None:
    slope, intercept, stderr = _physics.regress_on_index(
        np.array([1.0, 2.0, np.nan, np.nan]),
        np.array([0.0, 1.0, 2.0, 3.0]),
    )
    assert np.isnan([slope, intercept, stderr]).all()
    slope, _, _ = _physics.regress_on_index(np.ones(5), np.arange(5.0))
    assert slope == pytest.approx(0.0)  # constant statistic: zero slope
    with pytest.raises(ValueError, match="differ in length"):
        _physics.regress_on_index(np.zeros(4), np.zeros(5))
