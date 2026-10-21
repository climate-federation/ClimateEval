"""Tests for the chlorophyll-driver coupling diagnostic."""

from __future__ import annotations

import numpy as np
import pytest
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from climateeval.data import DataSourceInformation
from climateeval.diags.complex import ChlorophyllDriverCoupling
from climateeval.diags.complex._biogeochemistry import (
    BASINS,
    area_weighted_basin_mean,
    basin_mask,
    deseasonalised_standardised,
    lag_correlation,
    monthly_latitude_correlation,
    peak_lag,
    pearson_correlation,
)

N_LAT = 36
N_LON = 36
N_YEARS = 17
LEAD = 2  # months the mixed layer leads the chlorophyll in the synthetic data


def _info() -> DataSourceInformation:
    return DataSourceInformation(
        name="SynthESM",
        category="model",
        institute="Synth",
        exp="historical",
        variant="r1i1p1f1",
    )


def _grid() -> tuple[DimCoord, DimCoord, DimCoord]:
    n_time = N_YEARS * 12
    time = DimCoord(
        np.arange(n_time, dtype=float) * 30.0 + 15.0,
        standard_name="time",
        units=Unit("days since 1998-01-01", calendar="360_day"),
    )
    half_cell = 90.0 / N_LAT
    lat = DimCoord(
        np.linspace(-90.0 + half_cell, 90.0 - half_cell, N_LAT),
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


def _cube(
    var_name: str,
    units: str,
    data: np.ndarray,
    *,
    surface: bool = False,
) -> Cube:
    """Monthly cube; ``surface`` adds a one-level depth axis (as for chl)."""
    time, lat, lon = _grid()
    if surface:
        depth = DimCoord(
            [0.0],
            standard_name="depth",
            var_name="lev",
            units="m",
            attributes={"positive": "down"},
        )
        return Cube(
            data[:, None].astype(np.float32),
            var_name=var_name,
            units=units,
            dim_coords_and_dims=[(time, 0), (depth, 1), (lat, 2), (lon, 3)],
        )
    return Cube(
        data.astype(np.float32),
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
    )


def _coupled_cubes(*, seed: int, sign: float = 1.0) -> CubeList:
    """Chlorophyll driven by the mixed layer and, at the poles, by sea ice.

    The log chlorophyll anomaly is ``sign * (c(month, lat) * mld(t)
    - 0.9 * mld(t - LEAD))``: a contemporaneous coupling whose strength
    varies with the season and the latitude, and a stronger one with the
    mixed layer leading by ``LEAD`` months. Within the ice-covered boxes an
    anti-correlated sea-ice term is added. SST is independent noise.
    """
    rng = np.random.default_rng(seed)
    _time, lat, _lon = _grid()
    n_time = N_YEARS * 12
    months = np.arange(n_time) % 12 + 1
    lats = lat.points

    mld_anom = rng.normal(0.0, 1.0, (n_time, N_LAT, N_LON))
    mld_clim = 60.0 + 30.0 * (
        np.cos(2 * np.pi * (months[:, None, None] - 1) / 12)
        * np.sign(lats)[None, :, None]
    )
    mld = mld_clim + 8.0 * mld_anom + 0.01 * np.arange(n_time)[:, None, None]

    strength = -0.5 * (
        1.0
        + np.sin(2 * np.pi * months / 12)[:, None, None]
        * np.tanh(lats / 30.0)[None, :, None]
    )
    lagged = np.zeros_like(mld_anom)
    lagged[LEAD:] = mld_anom[:-LEAD]
    log_chl_anom = strength * mld_anom - 0.9 * lagged

    polar = (np.abs(lats) >= 60.0)[None, :, None]
    sic_anom = rng.normal(0.0, 1.0, (n_time, N_LAT, N_LON))
    siconc = np.where(polar, np.clip(80.0 + 10.0 * sic_anom, 0.0, 100.0), 0.0)
    log_chl_anom = log_chl_anom + np.where(polar, -0.7 * sic_anom, 0.0)

    log_chl_anom = sign * log_chl_anom + rng.normal(0.0, 0.05, log_chl_anom.shape)
    chl_mg_m3 = 10.0 ** (-0.6 + 0.3 * log_chl_anom)
    chl_kg_m3 = chl_mg_m3 * 1.0e-6

    tos = 15.0 + rng.normal(0.0, 1.0, (n_time, N_LAT, N_LON))

    return CubeList(
        [
            _cube("chl", "kg m-3", chl_kg_m3, surface=True),
            _cube("mlotst", "m", mld),
            _cube("tos", "degC", tos),
            _cube("siconc", "%", siconc),
        ],
    )


def _with_observations(
    monkeypatch: pytest.MonkeyPatch,
    diag: ChlorophyllDriverCoupling,
    observations: CubeList,
) -> ChlorophyllDriverCoupling:
    monkeypatch.setattr(
        diag,
        "_observation_cube",
        lambda _source, variable: diag._prepared_cube(observations, variable),
    )
    return diag


def _row(output) -> dict:  # noqa: ANN001
    assert output.raw_output is not None
    return output.raw_output.to_pandas().iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Pure formulae
# ---------------------------------------------------------------------------


def test_basin_mask_wraps_across_the_date_line_and_includes_edges() -> None:
    lats = np.array([-70.0, 0.0, 70.0])
    lons = np.array([0.0, 100.0, 300.0, 359.0])
    mask = basin_mask(lats, lons, BASINS["Atlantic"])
    assert mask.tolist() == [
        [False, False, False, False],
        [False, False, True, True],
        [False, False, True, True],
    ]
    wrapped = basin_mask(lats, lons, (-90.0, 90.0, 350.0, 10.0))
    assert wrapped[1].tolist() == [True, False, False, True]


def test_area_weighted_basin_mean_skips_missing_cells() -> None:
    lats = np.array([-60.0, 0.0, 60.0])
    field = np.ones((2, 3, 2))
    field[0, 1, 0] = np.nan
    field[1] = np.nan
    mask = np.ones((3, 2), dtype=bool)
    series = area_weighted_basin_mean(field, lats, mask)
    assert series[0] == pytest.approx(1.0)
    assert np.isnan(series[1])


def test_deseasonalised_standardised_removes_cycle_and_trend() -> None:
    n_time = 10 * 12
    months = np.arange(n_time) % 12 + 1
    series = 5.0 * np.cos(2 * np.pi * (months - 1) / 12) + 0.1 * np.arange(n_time)
    series[7] = np.nan
    out = deseasonalised_standardised(series, months, detrend=True)
    assert np.isnan(out[7])
    assert np.nanstd(out, ddof=1) == pytest.approx(1.0, abs=0.05)
    assert np.abs(np.nanmean(out)) < 0.05
    with pytest.raises(ValueError, match="months"):
        deseasonalised_standardised(series, months[:-1])


def test_lag_correlation_positive_lag_means_the_driver_leads() -> None:
    rng = np.random.default_rng(1)
    driver = rng.normal(0.0, 1.0, 240)
    chl = np.full_like(driver, np.nan)
    chl[3:] = -driver[:-3]  # chlorophyll follows the driver three months later
    lags, r = lag_correlation(chl, driver, lag_max=6, min_pairs=24)
    assert lags.tolist() == list(range(-6, 7))
    lag, corr = peak_lag(lags, r)
    assert lag == 3.0
    assert corr == pytest.approx(-1.0)
    assert np.abs(r[lags == 0]) < 0.2
    assert np.isnan(lag_correlation(chl[:20], driver[:20], lag_max=2)[1]).all()


def test_pearson_correlation_needs_three_finite_pairs() -> None:
    assert np.isnan(pearson_correlation(np.array([1.0, 2.0]), np.array([1.0, 2.0])))
    assert pearson_correlation(
        np.array([1.0, 2.0, 3.0, np.nan]),
        np.array([2.0, 4.0, 6.0, 1.0]),
    ) == pytest.approx(1.0)
    assert np.isnan(
        pearson_correlation(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])),
    )


def test_monthly_latitude_correlation_recovers_a_seasonal_structure() -> None:
    rng = np.random.default_rng(3)
    n_years, n_lat, n_lon = 20, 5, 30
    months = np.arange(n_years * 12) % 12 + 1
    driver = rng.normal(0.0, 1.0, (n_years * 12, n_lat, n_lon))
    strength = np.where(months[:, None, None] <= 6, 1.0, -1.0)
    chl = strength * driver + rng.normal(0.0, 0.1, driver.shape)
    r = monthly_latitude_correlation(chl, driver, months, min_years=10, min_cells=10)
    assert r.shape == (12, n_lat)
    assert np.all(r[:6] > 0.9)
    assert np.all(r[6:] < -0.9)
    sparse = monthly_latitude_correlation(chl, driver, months, min_years=30)
    assert np.isnan(sparse).all()


# ---------------------------------------------------------------------------
# The diagnostic
# ---------------------------------------------------------------------------


def _diag(**kwargs) -> ChlorophyllDriverCoupling:  # noqa: ANN003
    return ChlorophyllDriverCoupling(
        "chl_driver_coupling",
        fail_on_missing_data=True,
        download_missing_data=False,
        **kwargs,
    )


def test_coupling_of_a_model_built_like_the_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diag = _with_observations(monkeypatch, _diag(), _coupled_cubes(seed=11))
    row = _row(diag.get_output({"historical": _coupled_cubes(seed=12)}, _info()))
    assert list(row) == [
        "data_id",
        "data_type",
        "chl_mld_coupling_corr",
        "chl_sst_coupling_corr",
        "chl_siconc_coupling_corr",
        "chl_mld_seasonal_coupling_corr",
        "chl_sst_seasonal_coupling_corr",
        "chl_mld_corr_lag0",
        "chl_sst_corr_lag0",
        "chl_mld_peak_lag",
        "chl_sst_peak_lag",
    ]
    # The curves hold 23 lags of sampling noise per basin next to the two
    # coupled lags, so 17 years of a perfectly coupled model give ~0.85.
    assert row["chl_mld_coupling_corr"] > 0.75
    assert row["chl_mld_seasonal_coupling_corr"] == pytest.approx(1.0, abs=0.1)
    # two polar basins, one coupled lag each: a lower floor still
    assert row["chl_siconc_coupling_corr"] > 0.5
    # the mixed layer leads by LEAD months; the lag-zero coupling is negative
    assert row["chl_mld_peak_lag"] == LEAD
    assert row["chl_mld_corr_lag0"] < -0.2
    # SST is noise in both: the comparison exists but carries no signal
    assert np.isfinite(row["chl_sst_coupling_corr"])
    assert np.abs(row["chl_sst_corr_lag0"]) < 0.3


def test_coupling_of_a_model_with_the_wrong_sign(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diag = _with_observations(monkeypatch, _diag(), _coupled_cubes(seed=11))
    row = _row(
        diag.get_output({"historical": _coupled_cubes(seed=12, sign=-1.0)}, _info()),
    )
    assert row["chl_mld_coupling_corr"] < -0.75
    assert row["chl_mld_seasonal_coupling_corr"] == pytest.approx(-1.0, abs=0.1)
    assert row["chl_siconc_coupling_corr"] < -0.5
    assert row["chl_mld_corr_lag0"] > 0.2


def test_coupling_without_observations() -> None:
    diag = _diag(use_observations=False)
    row = _row(diag.get_output({"historical": _coupled_cubes(seed=12)}, _info()))
    assert np.isnan(row["chl_mld_coupling_corr"])
    assert np.isnan(row["chl_mld_seasonal_coupling_corr"])
    assert row["chl_mld_peak_lag"] == LEAD
    assert row["chl_mld_corr_lag0"] < -0.2


def test_missing_driver_leaves_its_scalars_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cubes = _coupled_cubes(seed=12)
    cubes.remove(cubes.extract_cube("siconc"))
    diag = ChlorophyllDriverCoupling(
        "chl_driver_coupling",
        fail_on_missing_data=False,
        download_missing_data=False,
    )
    diag = _with_observations(monkeypatch, diag, _coupled_cubes(seed=11))
    row = _row(diag.get_output({"historical": cubes}, _info()))
    assert np.isnan(row["chl_siconc_coupling_corr"])
    assert row["chl_mld_coupling_corr"] > 0.75
