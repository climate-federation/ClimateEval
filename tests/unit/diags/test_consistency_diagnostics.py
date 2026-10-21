"""Tests for the physical-consistency complex diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pytest
from cf_units import Unit
from iris import NameConstraint
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from climateeval.data import DataSourceInformation
from climateeval.data._esmvaltool_intake import (
    ESMValToolIntakeDataSource,
    ESMValToolIntakeDataSourceGenerator,
)
from climateeval.diags import _physics
from climateeval.diags.complex import (
    BudgetClosure,
    ClausiusClapeyronScaling,
    ClearSkyLongwaveFeedback,
    EnergyBalance,
    PrecipitationBuoyancy,
)
from climateeval.exceptions import DownloadError, MissingDataError

if TYPE_CHECKING:
    from collections.abc import Callable, Generator
    from pathlib import Path

    from climateeval import Variable

N_LAT = 18
N_LON = 36
PRESSURE_LEVELS = (100000.0, 50000.0)


def _time_coord(n_time: int) -> DimCoord:
    time = DimCoord(
        np.arange(n_time, dtype=float) * 30.0 + 15.0,
        standard_name="time",
        units=Unit("days since 1850-01-01", calendar="360_day"),
    )
    time.guess_bounds()
    return time


def _horizontal_coords() -> tuple[DimCoord, DimCoord]:
    lat = DimCoord(
        np.linspace(-85.0, 85.0, N_LAT),
        standard_name="latitude",
        units="degrees",
    )
    lon = DimCoord(
        np.linspace(5.0, 355.0, N_LON),
        standard_name="longitude",
        units="degrees",
        circular=True,
    )
    lat.guess_bounds()
    lon.guess_bounds()
    return lat, lon


def _monthly_cube(
    var_name: str,
    units: str,
    data_fn: Callable[[int, np.ndarray, np.ndarray], np.ndarray],
    *,
    n_years: int = 10,
) -> Cube:
    """Global monthly cube with data from a builder function."""
    n_time = n_years * 12
    time = _time_coord(n_time)
    lat, lon = _horizontal_coords()
    data = data_fn(n_time, lat.points, lon.points).astype(np.float32)
    return Cube(
        data,
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
    )


def _monthly_profile_cube(
    var_name: str,
    units: str,
    data_fn: Callable[[int, np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    *,
    n_years: int = 10,
) -> Cube:
    """Global monthly cube on pressure levels."""
    n_time = n_years * 12
    time = _time_coord(n_time)
    lat, lon = _horizontal_coords()
    plev = DimCoord(
        np.array(PRESSURE_LEVELS),
        standard_name="air_pressure",
        units="Pa",
    )
    data = data_fn(n_time, plev.points, lat.points, lon.points).astype(np.float32)
    return Cube(
        data,
        var_name=var_name,
        units=units,
        dim_coords_and_dims=[(time, 0), (plev, 1), (lat, 2), (lon, 3)],
    )


def _constant(value: float) -> Callable[[int, np.ndarray, np.ndarray], np.ndarray]:
    def build(n_time: int, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        return np.full((n_time, lats.size, lons.size), value)

    return build


def _uniform_series(
    values: np.ndarray,
) -> Callable[[int, np.ndarray, np.ndarray], np.ndarray]:
    """Spatially uniform field following a time series."""

    def build(n_time: int, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        return np.broadcast_to(
            np.asarray(values, dtype=float)[:n_time, None, None],
            (n_time, lats.size, lons.size),
        ).copy()

    return build


def _annual_to_monthly(annual: np.ndarray) -> np.ndarray:
    """Repeat every annual value over the twelve months of its year."""
    return np.repeat(np.asarray(annual, dtype=float), 12)


def _info(exp: str = "piControl") -> DataSourceInformation:
    return DataSourceInformation(
        name="SynthModel",
        category="model",
        institute="Synth",
        exp=exp,
        variant="r1i1p1f1",
    )


def _row(output) -> dict:  # noqa: ANN001
    assert output.raw_output is not None
    return output.raw_output.to_pandas().iloc[0].to_dict()


def _toa_cubes(
    toa_net_annual: np.ndarray,
    n_years: int,
) -> CubeList:
    """rsdt/rsut/rlut cubes realising a given annual TOA net flux series."""
    rsdt = 340.0 + _annual_to_monthly(toa_net_annual)
    return CubeList(
        [
            _monthly_cube("rsdt", "W m-2", _uniform_series(rsdt), n_years=n_years),
            _monthly_cube("rsut", "W m-2", _constant(240.0), n_years=n_years),
            _monthly_cube("rlut", "W m-2", _constant(100.0), n_years=n_years),
        ],
    )


def _fake_ensemble(
    cubes: CubeList | None,
) -> type[ESMValToolIntakeDataSourceGenerator]:
    """Intake generator serving one synthetic CMIP6 piControl model from memory.

    With ``cubes`` None, the model is catalogued but cannot be loaded, the way
    a real one that is neither staged locally nor downloadable behaves.
    """
    information = DataSourceInformation(
        name="FakeESM",
        category="CMIP6",
        institute="Fake",
        exp="piControl",
        variant="r1i1p1f1",
    )

    def get_cube(
        self: ESMValToolIntakeDataSource,
        _data_root_dir: Path | str,
        variable: Variable,
        *,
        download_missing_data: bool = True,
    ) -> Cube:
        if cubes is None:
            msg = f"{variable} of {self.id} would need to be downloaded"
            raise DownloadError(msg)
        return cubes.extract_cube(NameConstraint(var_name=variable.var_name))

    class FakeEnsemble(ESMValToolIntakeDataSourceGenerator):
        esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
            "project": "CMIP6",
            "exp": "piControl",
            "ensemble": "r1i1p1f1",
        }

        def generate(
            self,
            variable: Variable,
        ) -> Generator[ESMValToolIntakeDataSource]:
            facets = {
                **self.esmvaltool_dataset_facets,
                "dataset": "FakeESM",
                "short_name": variable.var_name,
            }
            data_source_cls = type(
                "FakeEnsembleDataSource",
                (ESMValToolIntakeDataSource,),
                {
                    "_information": information,
                    "esmvaltool_dataset_facets": facets,
                    "get_cube": get_cube,
                },
            )
            yield data_source_cls()

        def _get_references(self, **_facets: Any) -> tuple[str, ...]:
            return ()

    return FakeEnsemble


def test_energy_balance_mean_and_drift() -> None:
    """A control drifting by 0.002 W m-2 yr-1 has a 0.02 W m-2 decade-1 drift."""
    n_years = 20
    toa_net = 0.002 * np.arange(n_years, dtype=float)
    diag = EnergyBalance(
        "energy_balance",
        fail_on_missing_data=True,
        download_missing_data=False,
        evaluation_years=15,
        running_mean_years=5,
    )
    output = diag.get_output({"picontrol": _toa_cubes(toa_net, n_years)}, _info())
    row = _row(output)
    assert list(row) == [
        "data_id",
        "data_type",
        "toa_net_mean",
        "toa_net_drift",
        "toa_net_n_years",
    ]
    assert row["toa_net_n_years"] == 15
    # The last 15 of 20 years, i.e. years 5-19 of the linear drift
    assert row["toa_net_mean"] == pytest.approx(0.002 * 12.0, abs=1e-4)
    assert row["toa_net_drift"] == pytest.approx(0.02, abs=1e-4)


def test_energy_balance_uses_a_short_control_whole() -> None:
    n_years = 20
    diag = EnergyBalance(
        "energy_balance",
        fail_on_missing_data=True,
        download_missing_data=False,
        running_mean_years=5,
    )
    output = diag.get_output(
        {"picontrol": _toa_cubes(np.zeros(n_years), n_years)},
        _info(),
    )
    row = _row(output)
    assert row["toa_net_n_years"] == n_years
    assert row["toa_net_mean"] == pytest.approx(0.0, abs=1e-5)
    assert row["toa_net_drift"] == pytest.approx(0.0, abs=1e-5)


def test_energy_balance_includes_the_comparison_ensemble(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ensemble member becomes an ``other`` row of the same diagnostic."""
    n_years = 20
    control = _toa_cubes(np.zeros(n_years), n_years)
    drifting = _toa_cubes(0.002 * np.arange(n_years, dtype=float), n_years)
    monkeypatch.setattr(
        EnergyBalance,
        "_other_data_generator",
        _fake_ensemble(drifting),
    )
    diag = EnergyBalance(
        "energy_balance",
        fail_on_missing_data=True,
        download_missing_data=False,
        running_mean_years=5,
    )
    output = diag.get_output({"picontrol": control}, _info())
    assert output.raw_output is not None
    rows = output.raw_output.to_pandas().set_index("data_type")
    assert set(rows.index) == {"to_benchmark", "other"}
    assert rows.loc["other", "data_id"] == "CMIP6_FakeESM_piControl_r1i1p1f1"
    assert rows.loc["other", "toa_net_drift"] == pytest.approx(0.02, abs=1e-4)
    assert rows.loc["to_benchmark", "toa_net_drift"] == pytest.approx(0.0, abs=1e-5)
    assert output.data_sources is not None
    assert len(output.data_sources.to_pandas()) == 2


def test_unobtainable_ensemble_member_warns_unless_failing_is_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """As in ECS: skip with a warning, or raise with ``fail_on_missing_data``."""
    n_years = 20
    control = _toa_cubes(np.zeros(n_years), n_years)
    monkeypatch.setattr(EnergyBalance, "_other_data_generator", _fake_ensemble(None))
    lenient = EnergyBalance(
        "energy_balance",
        fail_on_missing_data=False,
        download_missing_data=False,
        running_mean_years=5,
    )
    output = lenient.get_output({"picontrol": control}, _info())
    assert output.raw_output is not None
    assert list(output.raw_output.to_pandas()["data_type"]) == ["to_benchmark"]
    strict = EnergyBalance(
        "energy_balance",
        fail_on_missing_data=True,
        download_missing_data=False,
        running_mean_years=5,
    )
    with pytest.raises(MissingDataError, match="downloaded"):
        strict.get_output({"picontrol": control}, _info())


def _closure_cubes(
    *,
    latent_heat_flux: float = 80.0,
    surface_downwelling_longwave: float = 300.0,
    n_years: int = 5,
) -> CubeList:
    """Build a control closing both budgets exactly, unless inputs shift.

    Precipitation releases 80 W m-2 of latent heat, the sensible heat flux is
    20 W m-2 and the atmospheric column is cooled by 100 W m-2 (surface net
    radiation 100 W m-2 with a vanishing TOA net flux).
    """
    precipitation = 80.0 / _physics.LATENT_HEAT_VAPORIZATION
    fluxes = {
        "pr": (precipitation, "kg m-2 s-1"),
        "hfls": (latent_heat_flux, "W m-2"),
        "hfss": (20.0, "W m-2"),
        "rsdt": (340.0, "W m-2"),
        "rsut": (240.0, "W m-2"),
        "rlut": (100.0, "W m-2"),
        "rsds": (200.0, "W m-2"),
        "rsus": (50.0, "W m-2"),
        "rlds": (surface_downwelling_longwave, "W m-2"),
        "rlus": (350.0, "W m-2"),
    }
    return CubeList(
        [
            _monthly_cube(name, units, _constant(value), n_years=n_years)
            for name, (value, units) in fluxes.items()
        ],
    )


def test_budget_closure_of_a_closed_control() -> None:
    diag = BudgetClosure(
        "budget_closure",
        fail_on_missing_data=True,
        download_missing_data=False,
    )
    output = diag.get_output({"picontrol": _closure_cubes()}, _info())
    row = _row(output)
    assert list(row) == [
        "data_id",
        "data_type",
        "water_budget_residual",
        "atmospheric_energy_residual",
    ]
    assert row["water_budget_residual"] == pytest.approx(0.0, abs=1e-4)
    assert row["atmospheric_energy_residual"] == pytest.approx(0.0, abs=1e-3)


def test_budget_closure_reports_real_imbalances() -> None:
    """An evaporation deficit and 5 W m-2 less downwelling longwave."""
    # 0.1 mm day-1 less evaporation than precipitation
    deficit = 0.1 / _physics.SECONDS_PER_DAY * _physics.LATENT_HEAT_VAPORIZATION
    diag = BudgetClosure(
        "budget_closure",
        fail_on_missing_data=True,
        download_missing_data=False,
    )
    output = diag.get_output(
        {
            "picontrol": _closure_cubes(
                latent_heat_flux=80.0 - deficit,
                surface_downwelling_longwave=295.0,
            ),
        },
        _info(),
    )
    row = _row(output)
    assert row["water_budget_residual"] == pytest.approx(0.1, abs=1e-3)
    assert row["atmospheric_energy_residual"] == pytest.approx(5.0, abs=1e-2)


def test_clausius_clapeyron_scaling_of_a_control() -> None:
    n_years = 10
    rng = np.random.default_rng(3)
    temperature = 288.0 + rng.normal(0.0, 0.5, n_years)
    water_vapour_path = 25.0 * (1.0 + 0.07 * (temperature - 288.0))
    cubes = CubeList(
        [
            _monthly_cube(
                "tas",
                "K",
                _uniform_series(_annual_to_monthly(temperature)),
                n_years=n_years,
            ),
            _monthly_cube(
                "prw",
                "kg m-2",
                _uniform_series(_annual_to_monthly(water_vapour_path)),
                n_years=n_years,
            ),
        ],
    )
    diag = ClausiusClapeyronScaling(
        "clausius_clapeyron_scaling",
        fail_on_missing_data=True,
        download_missing_data=False,
    )
    row = _row(diag.get_output({"picontrol": cubes}, _info()))
    assert row["clausius_clapeyron_scaling"] == pytest.approx(7.0, abs=0.2)


def test_clear_sky_longwave_feedback_recovers_a_prescribed_slope() -> None:
    """Clear-sky OLR built as 2.2 W m-2 K-1 times ts gives that feedback."""
    n_years = 10
    rng = np.random.default_rng(4)
    anomaly = rng.normal(0.0, 1.0, n_years * 12)
    cubes = CubeList(
        [
            _monthly_cube(
                "ts",
                "K",
                _uniform_series(288.0 + anomaly),
                n_years=n_years,
            ),
            _monthly_cube(
                "rlutcs",
                "W m-2",
                _uniform_series(240.0 + 2.2 * anomaly),
                n_years=n_years,
            ),
        ],
    )
    diag = ClearSkyLongwaveFeedback(
        "clear_sky_lw_feedback",
        fail_on_missing_data=True,
        download_missing_data=False,
    )
    row = _row(diag.get_output({"historical": cubes}, _info("historical")))
    assert row["clear_sky_lw_feedback"] == pytest.approx(2.2, abs=0.01)


def _precip_buoyancy_cubes(slope: float, *, n_years: int = 6) -> CubeList:
    """Cubes whose tropical P'-on-column-MSE' slope is ``slope``.

    The column is isothermal and dry, so its moist static energy is
    ``c_p ta`` and the column integral is ``c_p ta dp / g``. Precipitation
    follows the column anomaly with the requested slope inside 35S-35N and
    with the opposite sign poleward of 45 degrees, so a diagnostic that
    pooled the whole globe would not recover it.
    """
    rng = np.random.default_rng(5)
    temperature_anomaly = rng.normal(0.0, 1.0, n_years * 12)
    # d<h>/dT for a dry isothermal column, in MJ m-2 K-1
    column_per_kelvin = (
        _physics.SPECIFIC_HEAT_DRY_AIR
        * (PRESSURE_LEVELS[0] - PRESSURE_LEVELS[1])
        / _physics.GRAVITY
        / 1.0e6
    )

    def temperature(
        n_time: int,
        plevs: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> np.ndarray:
        field = 280.0 + temperature_anomaly[:n_time, None, None, None]
        return np.broadcast_to(field, (n_time, plevs.size, lats.size, lons.size)).copy()

    def zeros(
        n_time: int,
        plevs: np.ndarray,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> np.ndarray:
        return np.zeros((n_time, plevs.size, lats.size, lons.size))

    def precipitation(
        n_time: int,
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> np.ndarray:
        local_slope = np.where(np.abs(lats) <= 40.0, slope, -2.0)[None, :, None]
        anomaly = (
            local_slope
            * column_per_kelvin
            * temperature_anomaly[:n_time, None, None]
            / _physics.SECONDS_PER_DAY  # mm day-1 -> kg m-2 s-1
        )
        return np.broadcast_to(
            3.0 / _physics.SECONDS_PER_DAY + anomaly,
            (n_time, lats.size, lons.size),
        ).copy()

    return CubeList(
        [
            _monthly_profile_cube("ta", "K", temperature, n_years=n_years),
            _monthly_profile_cube("zg", "m", zeros, n_years=n_years),
            _monthly_profile_cube("hus", "1", zeros, n_years=n_years),
            _monthly_cube("pr", "kg m-2 s-1", precipitation, n_years=n_years),
        ],
    )


def test_precipitation_buoyancy_recovers_the_tropical_slope() -> None:
    cubes = _precip_buoyancy_cubes(0.4)
    diag = PrecipitationBuoyancy(
        "precipitation_buoyancy",
        fail_on_missing_data=True,
        download_missing_data=False,
        use_observations=False,
    )
    output = diag.get_output({"historical": cubes}, _info("historical"))
    row = _row(output)
    assert list(row) == [
        "data_id",
        "data_type",
        "precip_buoyancy_slope",
        "precip_buoyancy_slope_obs",
    ]
    assert row["precip_buoyancy_slope"] == pytest.approx(0.4, rel=1e-3)
    # No observational reference was requested
    assert np.isnan(row["precip_buoyancy_slope_obs"])


def test_precipitation_buoyancy_emits_the_observed_slope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reference slope is the same statistic applied to observations."""
    diag = PrecipitationBuoyancy(
        "precipitation_buoyancy",
        fail_on_missing_data=True,
        download_missing_data=False,
    )
    observations = _precip_buoyancy_cubes(0.6)
    monkeypatch.setattr(
        diag,
        "_observation_cubes",
        lambda: tuple(
            diag._prepared_cube(observations, variable)
            for variable in diag._input_variables
        ),
    )
    row = _row(
        diag.get_output(
            {"historical": _precip_buoyancy_cubes(0.4)},
            _info("historical"),
        ),
    )
    assert row["precip_buoyancy_slope"] == pytest.approx(0.4, rel=1e-3)
    assert row["precip_buoyancy_slope_obs"] == pytest.approx(0.6, rel=1e-3)


def test_precipitation_buoyancy_without_observations_keeps_going(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unobtainable observations leave the model slope alone, not an error."""
    diag = PrecipitationBuoyancy(
        "precipitation_buoyancy",
        fail_on_missing_data=False,
        download_missing_data=False,
    )

    def _fail() -> tuple[Cube, ...]:
        msg = "no network"
        raise RuntimeError(msg)

    monkeypatch.setattr(diag, "_observation_cubes", _fail)
    row = _row(
        diag.get_output(
            {"historical": _precip_buoyancy_cubes(0.4)},
            _info("historical"),
        ),
    )
    assert row["precip_buoyancy_slope"] == pytest.approx(0.4, rel=1e-3)
    assert np.isnan(row["precip_buoyancy_slope_obs"])


def test_precipitation_buoyancy_needs_two_common_levels() -> None:
    cubes = _precip_buoyancy_cubes(0.4)
    single_level = CubeList(
        [cube[:, :1] if cube.ndim == 4 else cube for cube in cubes],
    )
    diag = PrecipitationBuoyancy(
        "precipitation_buoyancy",
        fail_on_missing_data=True,
        download_missing_data=False,
        use_observations=False,
    )
    with pytest.raises(ValueError, match="common pressure levels"):
        diag.get_output({"historical": single_level}, _info("historical"))


def test_diagnostics_reject_a_wrong_experiment_key() -> None:
    diag = EnergyBalance("energy_balance", fail_on_missing_data=True)
    with pytest.raises(ValueError, match="Expected keys"):
        diag.get_output({"historical": CubeList([])}, _info())
