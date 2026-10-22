"""Tests for the equilibrium climate sensitivity (Gregory regression) diagnostic."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from cf_units import Unit
from iris import NameConstraint
from iris.coords import DimCoord
from iris.cube import Cube, CubeList

from climateeval.data import ComplexDataSource, DataSourceInformation
from climateeval.diags._base import DiagnosticOutput
from climateeval.diags.complex import ECS, _ecs
from climateeval.exceptions import MissingDataError

if TYPE_CHECKING:
    from pathlib import Path

    from climateeval import Variable

N_LAT = 18
N_LON = 36
DAYS_PER_MONTH = 30.0

CONTROL_TAS = 287.0
F4X = 7.4  # W m-2
LAMBDA = -0.9  # W m-2 K-1
EXPECTED_ECS = -F4X / 2.0 / LAMBDA  # 4.11 K

OUTPUT_COLUMNS = [
    "data_id",
    "data_type",
    "ecs",
    "f2x",
    "f4x",
    "lambda",
    "lambda_stderr",
    "p_value",
    "r2",
]


def _info(name: str = "SynthModel") -> DataSourceInformation:
    return DataSourceInformation(
        name=name,
        category="model",
        institute="Synth",
        exp="piControl abrupt-4xCO2",
        variant="r1i1p1f1",
    )


STANDARD_NAMES = {
    "tas": "air_temperature",
    "rsdt": "toa_incoming_shortwave_flux",
    "rsut": "toa_outgoing_shortwave_flux",
    "rlut": "toa_outgoing_longwave_flux",
}


def _monthly_cube(var_name: str, units: str, values: np.ndarray) -> Cube:
    """Spatially uniform monthly cube following an annual series."""
    monthly = np.repeat(np.asarray(values, dtype=float), 12)
    n_time = monthly.size
    time = DimCoord(
        np.arange(n_time, dtype=float) * DAYS_PER_MONTH + 15.0,
        standard_name="time",
        units=Unit("days since 1850-01-01", calendar="360_day"),
    )
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
    for coord in (time, lat, lon):
        coord.guess_bounds()
    data = np.broadcast_to(monthly[:, None, None], (n_time, N_LAT, N_LON))
    return Cube(
        data.astype(np.float32),
        var_name=var_name,
        standard_name=STANDARD_NAMES[var_name],
        units=units,
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
    )


def _experiment(tas: np.ndarray, toa_net: np.ndarray) -> CubeList:
    """Cubes of tas and the three TOA fluxes realising a net flux ``toa_net``."""
    toa_net = np.asarray(toa_net, dtype=float)
    return CubeList(
        [
            _monthly_cube("tas", "K", tas),
            _monthly_cube("rsdt", "W m-2", 340.0 + toa_net),
            _monthly_cube("rsut", "W m-2", np.full(toa_net.size, 240.0)),
            _monthly_cube("rlut", "W m-2", np.full(toa_net.size, 100.0)),
        ],
    )


def _gregory_data(n_years: int = 20) -> dict[str, CubeList]:
    """Build a balanced control and an abrupt-4xCO2 run on an exact Gregory line."""
    warming = 6.0 * (1.0 - np.exp(-np.arange(n_years) / 5.0)) + 0.3
    return {
        "picontrol": _experiment(
            np.full(n_years, CONTROL_TAS),
            np.zeros(n_years),
        ),
        "4xco2": _experiment(CONTROL_TAS + warming, F4X + LAMBDA * warming),
    }


def _rows(output: DiagnosticOutput):  # noqa: ANN202
    assert output.raw_output is not None
    return output.raw_output.to_pandas().set_index("data_type")


def test_gregory_regression_recovers_known_ecs() -> None:
    """An exact line N = F4x + lambda dT gives ECS = -F4x / (2 lambda)."""
    diag = ECS("ecs", fail_on_missing_data=True, download_missing_data=False)
    output = diag.get_output(_gregory_data(), _info())
    assert output.raw_output is not None
    table = output.raw_output.to_pandas()
    assert list(table.columns) == OUTPUT_COLUMNS
    assert len(table) == 1
    row = table.iloc[0]
    assert row["data_type"] == "to_benchmark"
    assert row["ecs"] == pytest.approx(EXPECTED_ECS, rel=1e-4)
    assert row["f4x"] == pytest.approx(F4X, rel=1e-4)
    assert row["f2x"] == pytest.approx(F4X / 2.0, rel=1e-4)
    assert row["lambda"] == pytest.approx(LAMBDA, rel=1e-4)
    assert row["lambda_stderr"] == pytest.approx(0.0, abs=1e-4)
    assert row["p_value"] == pytest.approx(0.0, abs=1e-10)
    assert row["r2"] == pytest.approx(1.0, abs=1e-8)
    assert output.metrics is None
    variables = set(output.variables.to_pandas()["id"])
    assert set(OUTPUT_COLUMNS[2:]) <= variables


def test_control_climatology_is_subtracted() -> None:
    """Shifting both runs by the same offsets leaves every result unchanged."""
    diag = ECS("ecs", fail_on_missing_data=True, download_missing_data=False)
    data = _gregory_data()
    reference = diag.get_output(data, _info()).raw_output
    assert reference is not None
    n_years = 20
    warming = 6.0 * (1.0 - np.exp(-np.arange(n_years) / 5.0)) + 0.3
    shifted = {
        # A warmer control with a 2 W m-2 TOA imbalance
        "picontrol": _experiment(
            np.full(n_years, CONTROL_TAS + 5.0),
            np.full(n_years, 2.0),
        ),
        "4xco2": _experiment(
            CONTROL_TAS + 5.0 + warming,
            2.0 + F4X + LAMBDA * warming,
        ),
    }
    row = diag.get_output(shifted, _info()).raw_output
    assert row is not None
    expected = reference.to_pandas().iloc[0]
    actual = row.to_pandas().iloc[0]
    for column in ("ecs", "f4x", "lambda"):
        assert actual[column] == pytest.approx(expected[column], rel=1e-4)


def test_year_slice_restricts_the_regression() -> None:
    """Only the selected years enter the fit; a list is turned into a slice."""
    n_years = 20
    warming = np.linspace(0.5, 6.0, n_years)
    toa_net = F4X + LAMBDA * warming
    toa_net[10:] += 1.5  # years outside the slice leave the Gregory line
    data = {
        "picontrol": _experiment(np.full(n_years, CONTROL_TAS), np.zeros(n_years)),
        "4xco2": _experiment(CONTROL_TAS + warming, toa_net),
    }
    sliced = ECS(
        "ecs",
        fail_on_missing_data=True,
        download_missing_data=False,
        year_slice=[None, 10],
    )
    assert sliced._years_slice == slice(None, 10)
    row = _rows(sliced.get_output(data, _info())).loc["to_benchmark"]
    assert row["lambda"] == pytest.approx(LAMBDA, rel=1e-4)
    assert row["ecs"] == pytest.approx(EXPECTED_ECS, rel=1e-4)
    assert row["r2"] == pytest.approx(1.0, abs=1e-8)

    # The default (first 150 years) uses all 20 years: the jump spoils the fit
    full = ECS("ecs", fail_on_missing_data=True, download_missing_data=False)
    assert full._years_slice == slice(None, 150)
    row = _rows(full.get_output(data, _info())).loc["to_benchmark"]
    assert row["r2"] < 0.99
    assert row["lambda"] != pytest.approx(LAMBDA, rel=1e-2)


def test_year_slice_given_as_slice_is_kept() -> None:
    diag = ECS("ecs", year_slice=slice(5, 15))
    assert diag._years_slice == slice(5, 15)


def test_missing_required_key_raises() -> None:
    diag = ECS("ecs")
    with pytest.raises(ValueError, match="Expected keys"):
        diag.get_output({"4xco2": CubeList()}, _info())


def test_f4x_cube_has_its_own_var_name() -> None:
    diag = ECS("ecs", fail_on_missing_data=True, download_missing_data=False)
    raw = diag._calculate_raw_output(
        ComplexDataSource(data=_gregory_data(), information=_info()),
    )
    assert {variable.var_name: cube.var_name for variable, cube in raw.items()} == {
        "ecs": "ecs",
        "f2x": "f2x",
        "f4x": "f4x",
        "lambda": "lambda",
        "lambda_stderr": "lambda_stderr",
        "p_value": "p_value",
        "r2": "r2",
    }


def test_postprocess_passes_through_empty_raw_output() -> None:
    diag = ECS("ecs")
    output = DiagnosticOutput(
        raw_output=None,
        metrics=None,
        variables=None,  # type: ignore[arg-type]
        data_sources=None,  # type: ignore[arg-type]
    )
    assert diag._postprocess_output(output) is output


# ---------------------------------------------------------------------------
# CMIP6 comparison ensemble
# ---------------------------------------------------------------------------


class _FakeDataSource:
    """In-memory stand-in for one CMIP6 intake data source."""

    def __init__(self, exp: str, cubes: CubeList | None) -> None:
        self._cubes = cubes
        self.information = DataSourceInformation(
            name="FakeESM",
            category="CMIP6",
            institute="Fake",
            exp=exp,
            variant="r1i1p1f1",
        )
        self.id = self.information.id

    def get_cube(
        self,
        _data_root_dir: Path | str,
        variable: Variable,
        *,
        download_missing_data: bool = True,  # noqa: ARG002
    ) -> Cube:
        if self._cubes is None:
            msg = f"{variable} of {self.id} would need to be downloaded"
            raise FileNotFoundError(msg)
        return self._cubes.extract_cube(NameConstraint(var_name=variable.var_name))


class CMIP6Abrupt4xCO2R1I1P1F1DataSource(_FakeDataSource):
    """Named like the real class: ECS picks the data sources by class name."""


class CMIP6PiControlR1I1P1F1DataSource(_FakeDataSource):
    """Named like the real class: ECS picks the data sources by class name."""


def _patch_ensemble(
    monkeypatch: pytest.MonkeyPatch,
    data: dict[str, CubeList] | None,
) -> list[dict[str, object]]:
    """Serve one fake CMIP6 model; record the filter arguments."""
    calls: list[dict[str, object]] = []

    def fake_filter(*generators: object, **kwargs: object) -> dict[str, list]:
        calls.append({"n_generators": len(generators), **kwargs})
        return {
            "FakeESM": [
                CMIP6PiControlR1I1P1F1DataSource(
                    "piControl",
                    None if data is None else data["picontrol"],
                ),
                CMIP6Abrupt4xCO2R1I1P1F1DataSource(
                    "abrupt-4xCO2",
                    None if data is None else data["4xco2"],
                ),
            ],
        }

    monkeypatch.setattr(_ecs, "filter_intake_data_sources", fake_filter)
    return calls


def test_cmip6_ensemble_member_is_evaluated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second model with a different feedback appears as 'other' data."""
    n_years = 20
    warming = np.linspace(0.5, 6.0, n_years)
    other_lambda = -1.5
    other = {
        "picontrol": _experiment(np.full(n_years, CONTROL_TAS), np.zeros(n_years)),
        "4xco2": _experiment(CONTROL_TAS + warming, F4X + other_lambda * warming),
    }
    calls = _patch_ensemble(monkeypatch, other)
    diag = ECS("ecs", fail_on_missing_data=True, download_missing_data=False)
    output = diag.get_output(_gregory_data(), _info())
    rows = _rows(output)
    assert set(rows.index) == {"to_benchmark", "other"}
    assert (
        rows.loc["other", "data_id"] == "CMIP6_FakeESM_piControl-abrupt-4xCO2_r1i1p1f1"
    )
    assert rows.loc["other", "lambda"] == pytest.approx(other_lambda, rel=1e-4)
    assert rows.loc["other", "ecs"] == pytest.approx(
        -F4X / 2.0 / other_lambda,
        rel=1e-4,
    )
    assert rows.loc["to_benchmark", "ecs"] == pytest.approx(EXPECTED_ECS, rel=1e-4)
    assert len(output.data_sources.to_pandas()) == 2

    # Four variables in each of the two experiments are requested
    assert len(calls) == 1
    assert calls[0]["n_generators"] == 8
    assert calls[0]["group_by"] == "dataset"
    required = calls[0]["required_facets"]
    assert isinstance(required, tuple)
    assert {(f["exp"], f["short_name"]) for f in required} == {
        (exp, var)
        for exp in ("piControl", "abrupt-4xCO2")
        for var in ("rlut", "rsdt", "rsut", "tas")
    }


def test_unobtainable_ensemble_member_warns_unless_failing_is_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_ensemble(monkeypatch, None)
    lenient = ECS("ecs", fail_on_missing_data=False, download_missing_data=False)
    rows = _rows(lenient.get_output(_gregory_data(), _info()))
    assert list(rows.index) == ["to_benchmark"]

    strict = ECS("ecs", fail_on_missing_data=True, download_missing_data=False)
    with pytest.raises(MissingDataError, match="downloaded"):
        strict.get_output(_gregory_data(), _info())
