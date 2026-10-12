from __future__ import annotations

import datetime
from typing import ClassVar

import cf_units
import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube

from climateeval import Variable
from climateeval.diags.simple import Nino34, SimpleDiagnosticInputData

_N_MONTHS = 24

# Spatially uniform Nino 3.4 anomalies of the test cube. The second year is the
# negative of the first, so that the monthly climatology is exactly zero and
# the deseasonalised index is this series itself.
_FIRST_YEAR_ANOMALIES = [
    0.5,
    -1.5,
    2.0,
    0.25,
    -0.75,
    1.0,
    -2.0,
    0.0,
    1.25,
    -0.5,
    1.5,
    -1.0,
]
_ANOMALIES = np.array([*_FIRST_YEAR_ANOMALIES, *(-a for a in _FIRST_YEAR_ANOMALIES)])


class UnsmoothedNino34(Nino34):
    """Nino 3.4 index without the running mean."""

    _rolling_window_length: ClassVar[int] = 1


class InvalidNino34(Nino34):
    """Nino 3.4 index with an invalid window length."""

    _rolling_window_length: ClassVar[int] = 0


def _get_monthly_time_coord(n_months: int) -> DimCoord:
    units = cf_units.Unit("days since 1850-01-01 00:00:00", calendar="standard")
    month_starts = [
        units.date2num(datetime.datetime(1850 + month // 12, month % 12 + 1, 1))  # noqa: DTZ001
        for month in range(n_months + 1)
    ]
    bounds = np.array(
        [[month_starts[i], month_starts[i + 1]] for i in range(n_months)],
    )
    return DimCoord(
        bounds.mean(axis=1),
        bounds=bounds,
        var_name="time",
        standard_name="time",
        long_name="Time Intervals",
        units=units,
    )


@pytest.fixture
def variable_tos() -> Variable:
    return Variable("tos", "tos", "mon")


@pytest.fixture
def cube_tos() -> Cube:
    """Get a monthly cube that covers the Nino 3.4 region.

    The sea surface temperature is uniform in space, so the area mean of the
    index is exactly the prescribed time series.
    """
    lat = DimCoord(
        [-9.0, -3.0, 3.0, 9.0],
        var_name="lat",
        standard_name="latitude",
        long_name="Latitude",
        units="degrees_north",
    )
    lon = DimCoord(
        [180.0, 200.0, 220.0, 240.0, 250.0],
        var_name="lon",
        standard_name="longitude",
        long_name="Longitude",
        units="degrees_east",
    )
    time = _get_monthly_time_coord(_N_MONTHS)
    data = np.broadcast_to(
        (300.0 + _ANOMALIES)[:, np.newaxis, np.newaxis],
        (_N_MONTHS, len(lat.points), len(lon.points)),
    )
    return Cube(
        data.astype(np.float32),
        var_name="tos",
        standard_name="sea_surface_temperature",
        long_name="Sea Surface Temperature",
        units="K",
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
    )


@pytest.fixture
def variables_and_data(
    variable_tos: Variable,
) -> dict[Variable, SimpleDiagnosticInputData]:
    return {variable_tos: SimpleDiagnosticInputData(reference=None)}


def test_default_rolling_window_length() -> None:
    assert Nino34._rolling_window_length == 3


def test_smoothed_index(
    variables_and_data: dict[Variable, SimpleDiagnosticInputData],
    cube_tos: Cube,
    variable_tos: Variable,
) -> None:
    diagnostic = Nino34("diagnostic", variables_and_data)

    cube = diagnostic._safely_preprocess(cube_tos, variable_tos)

    # The running mean shortens the series by `window_length - 1` points
    expected = np.convolve(_ANOMALIES, np.ones(3) / 3.0, mode="valid")
    assert cube.shape == (_N_MONTHS - 2,)
    np.testing.assert_allclose(cube.data, expected, atol=1e-5)


def test_unsmoothed_index(
    variables_and_data: dict[Variable, SimpleDiagnosticInputData],
    cube_tos: Cube,
    variable_tos: Variable,
) -> None:
    diagnostic = UnsmoothedNino34("diagnostic", variables_and_data)

    cube = diagnostic._safely_preprocess(cube_tos, variable_tos)

    # A window length of 1 skips the smoothing: the full monthly index remains
    assert cube.shape == (_N_MONTHS,)
    np.testing.assert_allclose(cube.data, _ANOMALIES, atol=1e-5)


def test_unsmoothed_index_has_larger_std(
    variables_and_data: dict[Variable, SimpleDiagnosticInputData],
    cube_tos: Cube,
    variable_tos: Variable,
) -> None:
    smoothed = Nino34("diagnostic", variables_and_data)._safely_preprocess(
        cube_tos,
        variable_tos,
    )
    unsmoothed = UnsmoothedNino34(
        "diagnostic",
        variables_and_data,
    )._safely_preprocess(cube_tos, variable_tos)

    assert unsmoothed.data.std() > smoothed.data.std()


def test_invalid_rolling_window_length(
    variables_and_data: dict[Variable, SimpleDiagnosticInputData],
    cube_tos: Cube,
    variable_tos: Variable,
) -> None:
    diagnostic = InvalidNino34("diagnostic", variables_and_data)

    with pytest.raises(
        ValueError,
        match="'_rolling_window_length' must be at least 1, got 0",
    ):
        diagnostic._safely_preprocess(cube_tos, variable_tos)
