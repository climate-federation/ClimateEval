"""Test per-reference timeranges and clipping of references (CE-2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest
from iris.cube import Cube, CubeList

from climateeval import Variable
from climateeval._utils import get_prepared_cube
from climateeval.data import DataSource, DataSourceInformation
from climateeval.diags.simple import (
    AnnualMeanTimeSeries,
    Map,
    SimpleDiagnosticInputData,
)
from climateeval.exceptions import MissingDataError
from tests.sample_data.create_sample_data_sources import (
    get_lat_coord,
    get_lon_coord,
)

if TYPE_CHECKING:
    from pathlib import Path

    from iris.coords import DimCoord


def _monthly_time_coord(first_year: int, last_year: int) -> DimCoord:
    from iris.coords import DimCoord  # noqa: PLC0415

    n_months = (last_year - first_year + 1) * 12
    # 30-day months starting on the first of January of first_year
    start = (first_year - 1970) * 360
    lower = start + 30.0 * np.arange(n_months)
    return DimCoord(
        lower + 15.0,
        bounds=np.stack([lower, lower + 30.0], axis=1),
        var_name="time",
        standard_name="time",
        long_name="Time Intervals",
        units="days since 1970-01-01 00:00:00",
    )


def make_cube(first_year: int, last_year: int, offset: float = 0.0) -> Cube:
    """Monthly 360-day-calendar ``tas`` cube covering whole years."""
    time = _monthly_time_coord(first_year, last_year)
    time.units = time.units.__class__(time.units.origin, calendar="360_day")
    data = offset + np.arange(time.shape[0], dtype=np.float32)[:, None, None]
    return Cube(
        np.broadcast_to(data, (time.shape[0], 3, 2)).copy(),
        var_name="tas",
        standard_name="air_temperature",
        long_name="Near-Surface Air Temperature",
        units="K",
        dim_coords_and_dims=[(time, 0), (get_lat_coord(), 1), (get_lon_coord(), 2)],
    )


def _years(cube: Cube) -> tuple[int, int]:
    time = cube.coord("time")
    dates = time.units.num2date(time.points)
    return dates[0].year, dates[-1].year


class StubReference(DataSource):
    """Reference whose record is 2000-2010 (like CERES-EBAF starting late)."""

    _information: ClassVar[DataSourceInformation] = DataSourceInformation(
        name="Stub",
        category="observation",
    )
    first_year: ClassVar[int] = 2000
    last_year: ClassVar[int] = 2010

    def download(self, variable: Variable, data_root_dir: Path | str) -> list[Path]:
        raise NotImplementedError

    def get_cube(
        self,
        data_root_dir: Path | str,  # noqa: ARG002
        variable: Variable,
        *,
        download_missing_data: bool = True,  # noqa: ARG002
    ) -> Cube:
        cube = make_cube(self.first_year, self.last_year, offset=100.0)
        return get_prepared_cube(CubeList([cube]), variable)


class NoOverlapReference(StubReference):
    first_year = 2030
    last_year = 2040


@pytest.fixture
def variable() -> Variable:
    return Variable("tas", "tas", "mon", timerange="19900101/20051231")


def _diag(
    cls: type = Map,
    variable: Variable | None = None,
    reference: type[DataSource] = StubReference,
    reference_timerange: str | None = None,
    **kwargs: object,
) -> Map:
    variable = variable or Variable("tas", "tas", "mon", timerange="19900101/20051231")
    return cls(
        "diag",
        {
            variable: SimpleDiagnosticInputData(
                reference=reference,
                reference_timerange=reference_timerange,
            ),
        },
        **kwargs,
    )


def test_partial_reference_is_dropped_by_default(variable: Variable) -> None:
    assert _diag(variable=variable)._get_reference_cubes() == {}


def test_partial_reference_fails_if_missing_data_fails(variable: Variable) -> None:
    diag = _diag(variable=variable, fail_on_missing_data=True)
    with pytest.raises(MissingDataError, match="data starts in year 2000"):
        diag._get_reference_cubes()


def test_clip_reference_timerange(variable: Variable) -> None:
    diag = _diag(variable=variable, clip_reference_timerange=True)
    cubes = diag._get_reference_cubes()
    (cube,) = cubes.values()
    assert cube.coords("latitude")  # Map: time-mean, no time coordinate
    raw = diag._get_raw_reference_cube(diag._reference_data[variable], variable)
    assert _years(raw) == (2000, 2005)


def test_clip_reference_timerange_logs_window(
    variable: Variable,
    capfd: pytest.CaptureFixture[str],
) -> None:
    from loguru import logger  # noqa: PLC0415

    messages: list[str] = []
    handler = logger.add(lambda m: messages.append(str(m)), level="INFO")
    try:
        _diag(variable=variable, clip_reference_timerange=True)._get_reference_cubes()
    finally:
        logger.remove(handler)
    assert any("clipped to 20000116/20051216" in m for m in messages)


def test_no_overlap_is_still_missing(variable: Variable) -> None:
    diag = _diag(
        variable=variable,
        reference=NoOverlapReference,
        clip_reference_timerange=True,
    )
    assert diag._get_reference_cubes() == {}
    diag = _diag(
        variable=variable,
        reference=NoOverlapReference,
        clip_reference_timerange=True,
        fail_on_missing_data=True,
    )
    with pytest.raises(MissingDataError, match="does not overlap"):
        diag._get_reference_cubes()


def test_reference_timerange_only_affects_reference() -> None:
    variable = Variable("tas", "tas", "mon", timerange="19900101/20051231")
    diag = _diag(variable=variable, reference_timerange="20000101/20041231")
    raw = diag._get_raw_reference_cube(diag._reference_data[variable], variable)
    assert _years(raw) == (2000, 2004)
    assert variable.timerange == "19900101/20051231"
    assert diag._get_reference_cubes()  # not dropped, nor clipped


def test_reference_timerange_without_model_timerange() -> None:
    variable = Variable("tas", "tas", "mon")  # "*" for the model
    diag = _diag(variable=variable, reference_timerange="20020101/20031231")
    raw = diag._get_raw_reference_cube(diag._reference_data[variable], variable)
    assert _years(raw) == (2002, 2003)


def test_reference_timerange_beyond_record_needs_clip() -> None:
    variable = Variable("tas", "tas", "mon")
    diag = _diag(variable=variable, reference_timerange="19900101/20051231")
    assert diag._get_reference_cubes() == {}
    diag = _diag(
        variable=variable,
        reference_timerange="19900101/20051231",
        clip_reference_timerange=True,
    )
    assert len(diag._get_reference_cubes()) == 1


@pytest.mark.parametrize("bad", ["2000/", "2000-01-01/2001-01-01", "foo", "20000101"])
def test_invalid_reference_timerange(bad: str) -> None:
    with pytest.raises(ValueError, match="reference_timerange must have the form"):
        SimpleDiagnosticInputData(reference=StubReference, reference_timerange=bad)


def test_from_suite_dict_reference_timerange() -> None:
    suite = {
        "variables": [
            {
                "id": "tas",
                "var_name": "tas",
                "frequency": "mon",
                "reference_data": f"{__name__}.StubReference",
                "reference_timerange": "20000101/20041231",
            },
        ],
    }
    diag = Map.from_suite_dict(
        "map",
        suite,
        {"timerange": "19900101/20051231"},
        clip_reference_timerange=True,
    )
    (variable,) = diag._variables
    assert variable.timerange == "19900101/20051231"
    assert not hasattr(variable, "reference_timerange")
    assert diag._reference_timeranges == {variable: "20000101/20041231"}
    assert diag._clip_reference_timerange is True

    # A run-time override (variable_kwargs) wins over the YAML
    diag = Map.from_suite_dict(
        "map",
        suite,
        {"reference_timerange": "20010101/20011231"},
    )
    assert next(iter(diag._reference_timeranges.values())) == "20010101/20011231"


def test_time_mean_metrics_with_different_windows() -> None:
    variable = Variable("tas", "tas", "mon", timerange="19900101/20051231")
    diag = _diag(variable=variable, reference_timerange="20000101/20041231")
    output = diag.get_output(
        CubeList([make_cube(1990, 2005)]),
        DataSourceInformation(name="model", category="CMIP6"),
    )
    assert output.metrics is not None
    assert output.metrics.count().execute() == 1
    variables = output.variables.to_pandas()
    assert list(variables["reference_timerange"]) == ["20000101/20041231"]
    assert list(variables["timerange"]) == ["19900101/20051231"]


def test_no_reference_timerange_column_by_default() -> None:
    diag = _diag()
    output = diag.get_output(
        CubeList([make_cube(2000, 2005)]),
        DataSourceInformation(name="model", category="CMIP6"),
    )
    assert "reference_timerange" not in output.variables.columns


def test_time_series_metrics_use_common_period() -> None:
    variable = Variable("tas", "tas", "mon", timerange="19900101/20051231")
    diag = _diag(
        AnnualMeanTimeSeries,
        variable=variable,
        clip_reference_timerange=True,
    )
    output = diag.get_output(
        CubeList([make_cube(1990, 2005)]),
        DataSourceInformation(name="model", category="CMIP6"),
    )
    assert output.metrics is not None
    metrics = output.metrics.to_pandas()
    assert len(metrics) == 1
    assert np.isfinite(metrics["weighted_rmse"]).all()
    raw = output.raw_output.to_pandas()
    assert raw.groupby("data_type")["time"].count().to_dict() == {
        "reference": 6,
        "to_benchmark": 16,
    }


def test_time_series_without_common_period_is_a_metric_error() -> None:
    variable = Variable("tas", "tas", "mon")
    diag = _diag(
        AnnualMeanTimeSeries,
        variable=variable,
        reference_timerange="20000101/20051231",
    )
    with pytest.raises(Exception, match="no common period"):
        diag.get_output(
            CubeList([make_cube(1950, 1960)]),
            DataSourceInformation(name="model", category="CMIP6"),
        )
