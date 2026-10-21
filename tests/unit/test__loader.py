"""Tests for :mod:`climateeval._loader`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import iris
import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube

from climateeval._loader import _overlaps_years, _timerange_years, load_cmor_dir

if TYPE_CHECKING:
    from pathlib import Path


def _make_cube(var_name: str, year: int, value: float = 1.0) -> Cube:
    """Create a small monthly cube covering one year."""
    time = DimCoord(
        np.arange(12) * 30.0 + 15.0,
        standard_name="time",
        units=f"days since {year}-01-01",
    )
    lat = DimCoord([0.0, 10.0], standard_name="latitude", units="degrees")
    lon = DimCoord([0.0, 10.0], standard_name="longitude", units="degrees")
    return Cube(
        np.full((12, 2, 2), value, dtype=np.float32),
        var_name=var_name,
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
    )


def _save(cube: Cube, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iris.save(cube, str(path))


def _cmor_name(var_name: str, year: int) -> str:
    return f"{var_name}_Amon_M_hist_r1i1p1f1_gn_{year}01-{year}12.nc"


def test_flat_dir_two_variables(tmp_path: Path) -> None:
    _save(_make_cube("tas", 2000), tmp_path / _cmor_name("tas", 2000))
    _save(_make_cube("pr", 2000), tmp_path / _cmor_name("pr", 2000))
    cubes = load_cmor_dir(tmp_path)
    assert sorted(c.var_name for c in cubes) == ["pr", "tas"]


def test_single_file(tmp_path: Path) -> None:
    path = tmp_path / _cmor_name("tas", 2000)
    _save(_make_cube("tas", 2000), path)
    cubes = load_cmor_dir(path)
    assert len(cubes) == 1


def test_year_split_files_are_concatenated(tmp_path: Path) -> None:
    _save(_make_cube("tas", 2000), tmp_path / _cmor_name("tas", 2000))
    _save(_make_cube("tas", 2001), tmp_path / _cmor_name("tas", 2001))
    cubes = load_cmor_dir(tmp_path)
    assert len(cubes) == 1
    assert cubes[0].coord("time").shape == (24,)


def test_timerange_filters_files(tmp_path: Path) -> None:
    _save(_make_cube("tas", 2000), tmp_path / _cmor_name("tas", 2000))
    _save(_make_cube("tas", 2010), tmp_path / _cmor_name("tas", 2010))
    cubes = load_cmor_dir(tmp_path, timerange="20000101/20051231")
    assert len(cubes) == 1
    assert cubes[0].coord("time").shape == (12,)


def test_drs_tree_prefers_native_grid_and_latest_version(tmp_path: Path) -> None:
    root = tmp_path / "Amon"
    _save(
        _make_cube("tas", 2000, value=1.0),
        root / "tas" / "gn" / "v20200101" / _cmor_name("tas", 2000),
    )
    _save(
        _make_cube("tas", 2000, value=2.0),
        root / "tas" / "gn" / "v20240101" / _cmor_name("tas", 2000),
    )
    _save(
        _make_cube("tas", 2000, value=3.0),
        root / "tas" / "gr" / "v20240101" / _cmor_name("tas", 2000),
    )
    _save(
        _make_cube("pr", 2000, value=4.0),
        root / "pr" / "gr" / "v20240101" / _cmor_name("pr", 2000),
    )
    cubes = load_cmor_dir(root)
    by_name = {c.var_name: c for c in cubes}
    assert sorted(by_name) == ["pr", "tas"]
    # native grid + latest version for tas; only grid available for pr
    assert float(by_name["tas"].data[0, 0, 0]) == pytest.approx(2.0)
    assert float(by_name["pr"].data[0, 0, 0]) == pytest.approx(4.0)


def test_empty_dir_returns_empty(tmp_path: Path) -> None:
    assert len(load_cmor_dir(tmp_path)) == 0


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        # Every CMOR precision, inside the window
        ("tas_Ayr_M_hist_r1_gn_2015-2025.nc", True),
        ("tos_Omon_M_hist_r1_gn_201501-201512.nc", True),
        ("tas_day_M_hist_r1_gn_20150101-20251231.nc", True),
        ("tas_6hrLev_M_hist_r1_gn_201501010600-202512311800.nc", True),
        ("tas_Amon_M_hist_r1_gn_201501-202512-clim.nc", True),
        ("CMIP6_X_hist_r1_20150116000000-20251216000000.nc", True),
        # Outside the window
        ("tas_Amon_M_hist_r1_gn_200001-201412.nc", False),
        ("tas_Amon_M_hist_r1_gn_202601-203012.nc", False),
        # Straddling either end, or touching the boundary year, is kept
        ("tas_Amon_M_hist_r1_gn_201001-201812.nc", True),
        ("tas_Amon_M_hist_r1_gn_202001-203012.nc", True),
        ("tas_Amon_M_hist_r1_gn_201001-201501.nc", True),
        ("tas_Amon_M_hist_r1_gn_202512-203012.nc", True),
        # Not a date range: never dropped
        ("areacella_fx_M_hist_r1_gn.nc", True),
        ("ERA5_tas_v5-1.nc", True),
        ("OBS_HadISST_2000-01.nc", True),
        ("tas_1-2.nc", True),
        ("some-name.nc", True),
    ],
)
def test_overlaps_years(filename: str, expected: bool) -> None:
    assert _overlaps_years(filename, (2015, 2025)) is expected


@pytest.mark.parametrize(
    ("timerange", "expected"),
    [
        (None, None),
        ("*", None),
        ("2015/2025", (2015, 2025)),
        ("201501/202512", (2015, 2025)),
        ("20150101/20251231", (2015, 2025)),
        ("20150101T0000/20251231T2359", (2015, 2025)),
        # Wildcards and durations are not narrowed: everything is loaded
        ("2015/*", None),
        ("*/2025", None),
        ("20150101/P10Y", None),
        ("P10Y/20251231", None),
    ],
)
def test_timerange_years(
    timerange: str | None,
    expected: tuple[int, int] | None,
) -> None:
    assert _timerange_years(timerange) == expected
