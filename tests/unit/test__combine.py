"""Tests for :mod:`climateeval._combine`.

Each test reproduces one way a real CMIP6 archive disagrees with itself about
coordinate metadata, and asserts that the record comes back as **one** cube —
because the failure being tested for is silent: without normalisation
``concatenate()`` returns several cubes for one variable and every diagnostic
downstream sees a truncated record.
"""

from __future__ import annotations

import numpy as np
from cf_units import Unit
from iris.coords import DimCoord
from iris.cube import Cube, CubeList
from loguru import logger

from climateeval._combine import (
    concatenate_cubes,
    normalise_coordinates,
    warn_on_split_variables,
)


def _cube(  # noqa: PLR0913
    year: int,
    *,
    var_name: str = "tas",
    calendar: str = "standard",
    lat_long_name: str | None = "latitude",
    time_attributes: dict | None = None,
    lat_dtype: type = np.float64,
    standard_name: str | None = "air_temperature",
) -> Cube:
    """One year of monthly data, with the metadata a test wants to vary."""
    time = DimCoord(
        np.arange(12) * 30.0 + 15.0 + (year - 2000) * 360.0,
        standard_name="time",
        var_name="time",
        units=Unit("days since 2000-01-01", calendar=calendar),
        attributes=dict(time_attributes or {}),
    )
    lat = DimCoord(
        np.array([-45.0, 45.0], dtype=lat_dtype),
        standard_name="latitude",
        long_name=lat_long_name,
        var_name="lat",
        units="degrees",
        bounds=np.array([[-90.0, 0.0], [0.0, 90.0]], dtype=lat_dtype),
    )
    lon = DimCoord(
        [0.0, 180.0],
        standard_name="longitude",
        var_name="lon",
        units="degrees",
    )
    return Cube(
        np.zeros((12, 2, 2), dtype=np.float32),
        standard_name=standard_name,
        var_name=var_name,
        units="K",
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
    )


def _n_concatenated(cubes: list[Cube]) -> int:
    return len(CubeList(cubes).concatenate())


# ---------------------------------------------------------------------------
# The real archive defects
# ---------------------------------------------------------------------------


def test_per_file_time_attribute_no_longer_splits_the_record() -> None:
    """IPSL-CM6A-LR writes a different `time:time_origin` in every file."""
    cubes = [
        _cube(2000, time_attributes={"time_origin": "2000-01-01 00:00:00"}),
        _cube(2001, time_attributes={"time_origin": "2001-01-01 00:00:00"}),
    ]
    assert _n_concatenated([c.copy() for c in cubes]) == 2

    result = concatenate_cubes(CubeList(cubes))
    assert len(result) == 1
    assert result[0].shape == (24, 2, 2)
    assert "time_origin" not in result[0].coord("time").attributes


def test_a_shared_coordinate_attribute_is_kept() -> None:
    """Only attributes that *differ* are dropped."""
    cubes = CubeList(
        [
            _cube(2000, time_attributes={"time_origin": "x", "keep": "same"}),
            _cube(2001, time_attributes={"time_origin": "y", "keep": "same"}),
        ],
    )
    result = concatenate_cubes(cubes)
    assert len(result) == 1
    assert result[0].coord("time").attributes == {"keep": "same"}


def test_capitalised_long_name_no_longer_splits_the_record() -> None:
    """AWI-CM-1-1-MR: `lat:long_name` is 'latitude' then 'Latitude'."""
    cubes = [
        _cube(2000, lat_long_name="latitude"),
        _cube(2001, lat_long_name="Latitude"),
    ]
    assert _n_concatenated([c.copy() for c in cubes]) == 2

    result = concatenate_cubes(CubeList(cubes))
    assert len(result) == 1
    assert result[0].coord("latitude").long_name == "latitude"


def test_long_name_is_not_unified_across_different_standard_names() -> None:
    """A matching `var_name` is not licence to merge two quantities."""
    first = _cube(2000)
    second = _cube(2001)
    second.coord("latitude").standard_name = "grid_latitude"
    result = concatenate_cubes(CubeList([first, second]))
    assert len(result) == 2


def test_float32_and_float64_grids_are_reconciled() -> None:
    """MIROC6 publishes the same ocean grid at two precisions."""
    cubes = [
        _cube(2000, lat_dtype=np.float32),
        _cube(2001, lat_dtype=np.float64),
    ]
    # The float32 round trip moves the points, so they do not match exactly.
    cubes[0].coord("latitude").points = np.array(
        [-45.000004, 45.000004],
        dtype=np.float64,
    )
    assert _n_concatenated([c.copy() for c in cubes]) == 2

    result = concatenate_cubes(CubeList(cubes))
    assert len(result) == 1
    # The float64 source wins, so no precision is invented.
    assert result[0].coord("latitude").points.tolist() == [-45.0, 45.0]


def test_a_genuinely_different_grid_is_never_merged() -> None:
    """KIOST-ESM publishes `tos` on two grids; NorESM2-LM writes a 1e30 fill."""
    cubes = CubeList([_cube(2000), _cube(2001)])
    cubes[1].coord("latitude").points = np.array([-45.0, 1e30])
    result = concatenate_cubes(cubes)
    assert len(result) == 2


def test_gregorian_and_proleptic_gregorian_are_unified_after_1582() -> None:
    """EC-Earth3-Veg alternates the two spellings between files."""
    cubes = [
        _cube(2000, calendar="gregorian"),
        _cube(2001, calendar="proleptic_gregorian"),
    ]
    assert _n_concatenated([c.copy() for c in cubes]) == 2

    result = concatenate_cubes(CubeList(cubes))
    assert len(result) == 1
    assert result[0].coord("time").units.calendar == "standard"


def test_a_real_calendar_difference_is_never_papered_over() -> None:
    """`standard` vs `noleap` (NESM3) is a different calendar, not a spelling."""
    cubes = CubeList([_cube(2000, calendar="standard"), _cube(2001, calendar="noleap")])
    result = concatenate_cubes(cubes)
    assert len(result) == 2


def test_gregorian_calendars_are_not_unified_before_1582() -> None:
    """Before the transition the two calendars are genuinely different."""
    first = _cube(2000, calendar="gregorian")
    second = _cube(2001, calendar="proleptic_gregorian")
    for cube in (first, second):
        coord = cube.coord("time")
        coord.units = Unit("days since 1500-01-01", calendar=coord.units.calendar)
    result = concatenate_cubes(CubeList([first, second]))
    assert len(result) == 2


def test_disagreeing_cube_standard_names_are_unified_by_majority() -> None:
    """IPSL-CM6A-LR labels `prw` two ways inside one run."""
    cubes = CubeList(
        [
            _cube(2000, var_name="prw", standard_name=None),
            _cube(2001, var_name="prw", standard_name=None),
            _cube(2002, var_name="prw", standard_name=None),
        ],
    )
    cubes[0].long_name = "Water Vapor Path"
    cubes[1].long_name = "Water Vapor Path"
    cubes[2].long_name = "Atmosphere Water Vapor Content"
    result = concatenate_cubes(cubes)
    assert len(result) == 1
    assert result[0].long_name == "Water Vapor Path"


# ---------------------------------------------------------------------------
# The loud failure
# ---------------------------------------------------------------------------


def test_a_variable_that_stays_split_is_reported() -> None:
    """The point of the module: never return a truncated record in silence."""
    messages: list[str] = []
    handle = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        cubes = CubeList(
            [_cube(2000, calendar="standard"), _cube(2001, calendar="noleap")],
        )
        result = concatenate_cubes(cubes, source="some/leaf")
    finally:
        logger.remove(handle)

    assert len(result) == 2
    assert any("did not concatenate into a single cube" in m for m in messages)
    assert any("'tas'" in m and "some/leaf" in m for m in messages)


def test_two_timeless_fields_of_one_name_are_not_a_split_record() -> None:
    """A CMORized observation file carries `area` on two grids; not a warning."""
    messages: list[str] = []

    handle = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        first = _cube(2000)
        second = _cube(2000)
        for cube in (first, second):
            cube.remove_coord("time")
        second.coord("latitude").points = np.array([-30.0, 30.0])
        warn_on_split_variables(CubeList([first[0], second[0]]))
    finally:
        logger.remove(handle)
    assert messages == []


def test_warn_on_split_variables_is_quiet_when_nothing_is_split() -> None:
    messages: list[str] = []
    handle = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    try:
        warn_on_split_variables(CubeList([_cube(2000)]))
    finally:
        logger.remove(handle)
    assert messages == []


def test_normalise_coordinates_returns_the_same_list() -> None:
    cubes = CubeList([_cube(2000)])
    assert normalise_coordinates(cubes) is cubes
