"""Tests for :mod:`climateeval._check`."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import iris
import numpy as np
import yaml
from iris.coords import DimCoord
from iris.cube import Cube

from climateeval._check import check_model_data, suite_variable_requirements

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any

TIMERANGE = "20000101/20011231"


def _make_cube(
    var_name: str,
    units: str | None = "K",
    year: int = 2000,
    n_years: int = 2,
    plev: list[float] | None = None,
) -> Cube:
    """Create a small monthly cube covering ``n_years`` years."""
    n_time = 12 * n_years
    time = DimCoord(
        np.arange(n_time) * 30.0 + 15.0,
        standard_name="time",
        units=f"days since {year}-01-01",
    )
    lat = DimCoord([0.0, 10.0], standard_name="latitude", units="degrees")
    lon = DimCoord([0.0, 10.0], standard_name="longitude", units="degrees")
    coords: list[tuple[DimCoord, int]] = [(time, 0)]
    shape: tuple[int, ...] = (n_time,)
    if plev is not None:
        plev_coord = DimCoord(plev, standard_name="air_pressure", units="Pa")
        coords.append((plev_coord, 1))
        shape = (n_time, len(plev))
    coords.extend([(lat, len(shape)), (lon, len(shape) + 1)])
    shape = (*shape, 2, 2)
    return Cube(
        np.ones(shape, dtype=np.float32),
        var_name=var_name,
        units=units,
        dim_coords_and_dims=coords,
    )


def _save(cube: Cube, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fname = f"{cube.var_name}_Amon_M_hist_r1i1p1f1_gn_200001-200112.nc"
    iris.save(cube, str(directory / fname))


def _suite(tmp_path: Path, variables: Sequence[dict[str, Any]]) -> Path:
    """Write a minimal suite YAML with the given variable entries."""
    suite_path = tmp_path / "suite.yml"
    suite_path.write_text(
        yaml.dump([{"name": "map", "diagnostic": "x", "variables": variables}]),
        encoding="utf-8",
    )
    return suite_path


_TAS = {"id": "tas", "var_name": "tas", "frequency": "mon"}
_TA_LEVEL_200 = {
    "id": "ta200",
    "var_name": "ta",
    "frequency": "mon",
    "vertical_level": {
        "coordinate_name": "air_pressure",
        "units": "hPa",
        "value": 200.0,
    },
}
_TA_LEVEL_800 = {
    "id": "ta800",
    "var_name": "ta",
    "frequency": "mon",
    "vertical_level": {
        "coordinate_name": "air_pressure",
        "units": "hPa",
        "value": 800.0,
    },
}
_LWCRE = {"id": "lwcre", "var_name": "lwcre", "frequency": "mon"}


def _checks_by_name(report, var_name):  # noqa: ANN001, ANN202
    (variable,) = [v for v in report.variables if v.var_name == var_name]
    return {c.name: c for c in variable.checks}


def test_all_good(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _save(_make_cube("tas"), data)
    _save(_make_cube("ta", plev=[100000.0, 50000.0, 10000.0]), data)
    suite = _suite(tmp_path, [_TAS, _TA_LEVEL_200, _TA_LEVEL_800])
    report = check_model_data(data, suites=[suite], timerange=TIMERANGE)
    assert report.passed
    assert all(c.status == "pass" for v in report.variables for c in v.checks)
    assert report.extra_var_names == ()


def test_wrong_var_name(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _save(_make_cube("t2m"), data)
    suite = _suite(tmp_path, [_TAS])
    report = check_model_data(data, suites=[suite], timerange=TIMERANGE)
    assert not report.passed
    found = _checks_by_name(report, "tas")["found"]
    assert found.status == "fail"
    assert "t2m" in found.message
    assert report.extra_var_names == ("t2m",)


def test_unknown_units(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _save(_make_cube("tas", units=None), data)
    suite = _suite(tmp_path, [_TAS])
    report = check_model_data(data, suites=[suite], timerange=TIMERANGE)
    units = _checks_by_name(report, "tas")["units"]
    assert units.status == "fail"
    assert "units missing or invalid" in units.message


def test_non_convertible_units(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _save(_make_cube("tas", units="Pa"), data)
    suite = _suite(tmp_path, [_TAS])
    report = check_model_data(data, suites=[suite], timerange=TIMERANGE)
    units = _checks_by_name(report, "tas")["units"]
    assert units.status == "fail"
    assert "not convertible to 'K'" in units.message


def test_convertible_units_pass_with_note(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _save(_make_cube("tas", units="degC"), data)
    suite = _suite(tmp_path, [_TAS])
    report = check_model_data(data, suites=[suite], timerange=TIMERANGE)
    units = _checks_by_name(report, "tas")["units"]
    assert units.status == "pass"
    assert "converted" in units.message


def test_unrecognizable_vertical_coordinate(tmp_path: Path) -> None:
    data = tmp_path / "data"
    cube = _make_cube("ta", plev=[100000.0, 50000.0])
    vertical = cube.coord("air_pressure")
    vertical.standard_name = None
    vertical.var_name = "mylev"
    _save(cube, data)
    suite = _suite(tmp_path, [_TA_LEVEL_200])
    report = check_model_data(data, suites=[suite], timerange=TIMERANGE)
    coordinates = _checks_by_name(report, "ta")["coordinates"]
    assert coordinates.status == "fail"
    assert "standard_name" in coordinates.message


def test_extra_dimension_fails(tmp_path: Path) -> None:
    data = tmp_path / "data"
    cube = _make_cube("tas", plev=[0.0, 1.0])  # extra dim with wrong meaning
    extra = cube.coord("air_pressure")
    extra.standard_name = None
    extra.var_name = "member"
    extra.units = "1"
    _save(cube, data)
    suite = _suite(tmp_path, [_TAS])
    report = check_model_data(data, suites=[suite], timerange=TIMERANGE)
    coordinates = _checks_by_name(report, "tas")["coordinates"]
    assert coordinates.status == "fail"


def test_timerange_outside_data_span(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _save(_make_cube("tas"), data)
    suite = _suite(tmp_path, [_TAS])
    report = check_model_data(data, suites=[suite], timerange="19790101/20141231")
    timerange = _checks_by_name(report, "tas")["timerange"]
    assert timerange.status == "fail"
    assert "2000-2001" in timerange.message


def test_level_outside_range_warns(tmp_path: Path) -> None:
    data = tmp_path / "data"
    # 1000-500 hPa only; the 200 hPa extraction will yield masked values
    _save(_make_cube("ta", plev=[100000.0, 50000.0]), data)
    suite = _suite(tmp_path, [_TA_LEVEL_200])
    report = check_model_data(data, suites=[suite], timerange=TIMERANGE)
    levels = _checks_by_name(report, "ta")["levels"]
    assert levels.status == "warn"
    assert report.passed  # warnings do not fail the check
    assert report.variables[0].status == "warn"


def test_derived_variable_sources_present(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _save(_make_cube("rlut", units="W m-2"), data)
    _save(_make_cube("rlutcs", units="W m-2"), data)
    suite = _suite(tmp_path, [_LWCRE])
    report = check_model_data(data, suites=[suite], timerange=TIMERANGE)
    assert report.passed
    sources = _checks_by_name(report, "lwcre")["sources"]
    assert sources.status == "pass"
    assert "rlut" in sources.message
    assert report.extra_var_names == ()


def test_derived_variable_source_missing(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _save(_make_cube("rlut", units="W m-2"), data)
    suite = _suite(tmp_path, [_LWCRE])
    report = check_model_data(data, suites=[suite], timerange=TIMERANGE)
    sources = _checks_by_name(report, "lwcre")["sources"]
    assert sources.status == "fail"
    assert "rlutcs" in sources.message


def test_duplicate_cubes_fail(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    cube_a = _make_cube("tas", year=2000, n_years=1)
    cube_b = _make_cube("tas", year=2001, n_years=1)
    cube_b.coord("latitude").points = [0.0, 20.0]  # prevents concatenation
    iris.save(cube_a, str(data / "tas_Amon_M_hist_r1i1p1f1_gn_200001-200012.nc"))
    iris.save(cube_b, str(data / "tas_Amon_M_hist_r1i1p1f1_gn_200101-200112.nc"))
    suite = _suite(tmp_path, [_TAS])
    report = check_model_data(data, suites=[suite], timerange=TIMERANGE)
    found = _checks_by_name(report, "tas")["found"]
    assert found.status == "fail"
    assert "concatenate" in found.message


def test_derived_variable_without_derivation_script(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _save(_make_cube("tas"), data)
    suite = _suite(
        tmp_path,
        [{"id": "phcint", "var_name": "phcint", "frequency": "mon"}],
    )
    report = check_model_data(data, suites=[suite], timerange=TIMERANGE)
    sources = _checks_by_name(report, "phcint")["sources"]
    assert sources.status == "fail"
    assert "pre-computed 'phcint'" in sources.message


def test_to_dict_is_json_serializable(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _save(_make_cube("tas"), data)
    suite = _suite(tmp_path, [_TAS])
    report = check_model_data(data, suites=[suite], timerange=TIMERANGE)
    parsed = json.loads(json.dumps(report.to_dict()))
    assert parsed["passed"] is True
    assert parsed["variables"][0]["status"] == "pass"


def test_requirements_dedupe_levels(tmp_path: Path) -> None:
    suite = _suite(tmp_path, [_TA_LEVEL_200, _TA_LEVEL_800, _TAS])
    requirements = suite_variable_requirements([suite])
    assert sorted(requirements) == ["ta", "tas"]
    assert len(requirements["ta"].levels) == 2


def test_requirements_skip_non_input_variables(tmp_path: Path) -> None:
    requirements = suite_variable_requirements(["Tier3_ecs"])
    assert requirements == {}
