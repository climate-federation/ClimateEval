from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import iris
import numpy as np
import pytest
from cf_units import Unit
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube

import climateeval.data._oras5
from climateeval import Variable
from climateeval.data import ORAS5, DataSourceInformation
from climateeval.exceptions import DownloadError, MissingDataError

if TYPE_CHECKING:
    from pathlib import Path

    from iris.cube import CubeList

# Raw ORCA-like grid including the halo cells (one extra row at the top and
# one extra column on each side), and the grid without halo.
NY, NX = 4, 6
NY_CROP, NX_CROP = NY - 1, NX - 2
TIME_UNITS = Unit("days since 2000-01-01", calendar="standard")


@pytest.fixture
def raw_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "raw"
    path.mkdir()
    monkeypatch.setenv(ORAS5._RAW_DIR_ENV, str(path))
    return path


@pytest.fixture
def grid_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "grids"
    path.mkdir()
    monkeypatch.setenv(ORAS5._GRID_DIR_ENV, str(path))
    for grid in ("T", "u", "v"):
        iris.save(_grid_cube(), path / f"oras5_mesh_{grid}.nc")
    return path


def _lat_lon_points(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    lon, lat = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    return lat * 10.0 - 10.0, lon * 10.0 + 5.0


def _grid_cube() -> Cube:
    """Grid (mesh) file cube: stored as (x, y) with (x, y, 4) bounds."""
    # Same cell centres as the raw data without halo.
    lat, lon = (points[:-1, 1:-1] for points in _lat_lon_points(NY, NX))
    lat_t, lon_t = lat.T, lon.T  # (x, y)
    offsets = np.array([-5.0, -5.0, 5.0, 5.0])
    lat_bnds = lat_t[..., np.newaxis] + offsets
    lon_bnds = lon_t[..., np.newaxis] + np.roll(offsets, 1)
    cube = Cube(
        np.ones((NX_CROP, NY_CROP)),
        var_name="cell_area",
        units="m2",
    )
    cube.add_aux_coord(
        AuxCoord(lat_t, standard_name="latitude", units="degrees", bounds=lat_bnds),
        (0, 1),
    )
    cube.add_aux_coord(
        AuxCoord(lon_t, standard_name="longitude", units="degrees", bounds=lon_bnds),
        (0, 1),
    )
    return cube


def _raw_cube(raw_name: str, day: float, *, depth: str | None = None) -> Cube:
    """Raw NEMO-like cube with halo cells: (time, [depth,] y, x)."""
    shape: tuple[int, ...] = (1, NY, NX) if depth is None else (1, 2, NY, NX)
    data = np.arange(np.prod(shape), dtype=float).reshape(shape)
    cube = Cube(data, var_name=raw_name, units="1")
    cube.add_dim_coord(
        DimCoord(
            [day],
            standard_name="time",
            var_name="time_counter",
            units=TIME_UNITS,
        ),
        0,
    )
    if depth is not None:
        cube.add_dim_coord(DimCoord([0.5, 10.0], var_name=depth, units="m"), 1)
    lat, lon = _lat_lon_points(NY, NX)
    ndim = cube.ndim
    cube.add_aux_coord(
        AuxCoord(lat, standard_name="latitude", var_name="nav_lat", units="degrees"),
        (ndim - 2, ndim - 1),
    )
    cube.add_aux_coord(
        AuxCoord(lon, standard_name="longitude", var_name="nav_lon", units="degrees"),
        (ndim - 2, ndim - 1),
    )
    return cube


def _raw_filename(raw_name: str, year_month: str, dim: str = "2D") -> str:
    return f"{raw_name}_control_monthly_highres_{dim}_{year_month}_CONS_v0.1.nc"


def _write_raw(
    raw_dir: Path,
    raw_name: str,
    *,
    depth: str | None = None,
) -> list[Path]:
    paths = []
    for i, year_month in enumerate(("200001", "200002")):
        dim = "2D" if depth is None else "3D"
        path = raw_dir / _raw_filename(raw_name, year_month, dim)
        iris.save(_raw_cube(raw_name, 15.0 + 31.0 * i, depth=depth), path)
        paths.append(path)
    return paths


# ----------------------------------------------------------------------------
# Metadata / provenance
# ----------------------------------------------------------------------------


def test_information() -> None:
    data_source = ORAS5()

    assert data_source.name == "ORAS5"
    assert data_source.id == "reanalysis_ORAS5"
    assert data_source.information == DataSourceInformation(
        name="ORAS5",
        category="reanalysis",
        institute="ECMWF",
        references=("https://doi.org/10.5194/os-15-779-2019",),
    )


def test_download_raises(tmp_path: Path) -> None:
    with pytest.raises(DownloadError, match="cannot be downloaded automatically"):
        ORAS5().download(Variable("tos", "tos", "mon"), tmp_path)


def _stub_variable(var_name: str) -> Variable:
    """Stand-in for variables missing from ``variables.yml``."""
    return cast("Variable", SimpleNamespace(var_name=var_name, timerange="*"))


@pytest.mark.parametrize("var_name", sorted(ORAS5._RAW_VARIABLE_INFO))
def test_supported_variables(var_name: str) -> None:
    ORAS5()._check_supported_variables(_stub_variable(var_name))


def test_unsupported_variable() -> None:
    with pytest.raises(NotImplementedError, match="does not provide variable 'tas'"):
        ORAS5()._check_supported_variables(Variable("tas", "tas", "mon"))


def test_get_cube_unsupported_variable(tmp_path: Path) -> None:
    with pytest.raises(NotImplementedError, match="supported: "):
        ORAS5().get_cube(tmp_path, Variable("tas", "tas", "mon"))


# ----------------------------------------------------------------------------
# Paths and environment variables
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("var_name", "grid"),
    [("tos", "T"), ("mlotst", "T"), ("uo", "u"), ("tauuo", "u"), ("vo", "v")],
)
def test_grid_path_from_env(
    var_name: str,
    grid: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ORAS5._GRID_DIR_ENV, str(tmp_path))
    path = ORAS5()._get_grid_path(_stub_variable(var_name))
    assert path == tmp_path / f"oras5_mesh_{grid}.nc"


def test_grid_path_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ORAS5._GRID_DIR_ENV, raising=False)
    path = ORAS5()._get_grid_path(Variable("tos", "tos", "mon"))
    assert str(path) == f"{ORAS5._DEFAULT_GRID_DIR}/oras5_mesh_T.nc"


def test_raw_files_default_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ORAS5._RAW_DIR_ENV, raising=False)
    globbed: list[str] = []

    def fake_glob(self: Path, pattern: str) -> list[Path]:
        globbed.append(f"{self}/{pattern}")
        return []

    monkeypatch.setattr(climateeval.data._oras5.Path, "glob", fake_glob)
    assert ORAS5()._get_raw_files(Variable("tos", "tos", "mon")) == []
    assert globbed == [f"{ORAS5._DEFAULT_RAW_DIR}/sosstsst_*.nc"]


def test_raw_files_matched_by_raw_name_and_sorted(raw_dir: Path) -> None:
    names = [
        _raw_filename("sosstsst", "200002"),
        _raw_filename("sosstsst", "200001"),
        _raw_filename("sosaline", "200001"),
        "unrelated.nc",
    ]
    for name in names:
        (raw_dir / name).touch()

    files = ORAS5()._get_raw_files(Variable("tos", "tos", "mon"))

    assert [f.name for f in files] == sorted(names[:2])


def test_raw_files_restricted_to_timerange(raw_dir: Path) -> None:
    for year_month in ("199912", "200001", "200006", "200101"):
        (raw_dir / _raw_filename("sosstsst", year_month)).touch()
    variable = Variable("tos", "tos", "mon", timerange="2000/2000")

    files = ORAS5()._get_raw_files(variable)

    assert [ORAS5._file_year_month(f) for f in files] == [200001, 200006]


# ----------------------------------------------------------------------------
# Time selection helpers
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("somxl010_control_monthly_highres_2D_200001_CONS_v0.1.nc", 200001),
        ("vozocrte_control_monthly_highres_3D_198512_OPER_v0.1.nc", 198512),
        ("somxl010_no_stamp.nc", None),
        ("somxl010_2000.nc", None),
        ("somxl010_20000a.nc", None),
    ],
)
def test_file_year_month(name: str, expected: int | None, tmp_path: Path) -> None:
    assert ORAS5._file_year_month(tmp_path / name) == expected


@pytest.mark.parametrize(
    ("part", "is_start", "expected"),
    [
        ("2000", True, 200001),
        ("2000", False, 200012),
        ("200003", True, 200003),
        ("20000315", False, 200003),
        ("2000-03-15T00:00:00", True, 200003),
        ("*", True, None),
        ("", False, None),
        ("P5Y", True, None),
    ],
)
def test_parse_timerange_bound(
    part: str,
    *,
    is_start: bool,
    expected: int | None,
) -> None:
    assert ORAS5._parse_timerange_bound(part, is_start=is_start) == expected


@pytest.mark.parametrize("timerange", ["", "*"])
def test_filter_files_no_timerange(timerange: str, tmp_path: Path) -> None:
    files = [tmp_path / _raw_filename("x", "200001"), tmp_path / "x_nostamp.nc"]
    assert ORAS5()._filter_files_by_timerange(files, timerange) == files


@pytest.mark.parametrize(
    ("timerange", "expected"),
    [
        ("200002/200011", ["200006", "nostamp"]),
        ("2000/2001", ["200001", "200006", "200101", "nostamp"]),
        ("*/2000", ["199912", "200001", "200006", "nostamp"]),
        ("200006/*", ["200006", "200101", "nostamp"]),
        ("2001", ["200101", "nostamp"]),
    ],
)
def test_filter_files_by_timerange(
    timerange: str,
    expected: list[str],
    tmp_path: Path,
) -> None:
    stamps = ["199912", "200001", "200006", "200101", "nostamp"]
    files = [tmp_path / f"x_{stamp}.nc" for stamp in stamps]

    selected = ORAS5()._filter_files_by_timerange(files, timerange)

    assert [f.stem.split("_")[-1] for f in selected] == expected


# ----------------------------------------------------------------------------
# Loading and CMORization
# ----------------------------------------------------------------------------


def test_get_cube_no_files(raw_dir: Path, tmp_path: Path) -> None:
    (raw_dir / _raw_filename("sosstsst", "199001")).touch()
    variable = Variable("tos", "tos", "mon", timerange="2000/2001")

    with pytest.raises(MissingDataError, match="No raw files found"):
        ORAS5().get_cube(tmp_path, variable)


def test_load_native_cube_2d(raw_dir: Path, grid_dir: Path) -> None:
    files = _write_raw(raw_dir, "somxl010")
    _write_raw(raw_dir, "sosstsst")  # other variable in the same files dir

    cube = ORAS5()._load_native_cube(files, Variable("mlotst", "mlotst", "mon"))

    assert cube.var_name == "mlotst"
    assert cube.standard_name == "ocean_mixed_layer_thickness_defined_by_sigma_t"
    assert cube.long_name == "Ocean Mixed Layer Thickness Defined by Sigma T"
    assert cube.units == "m"
    # Halo removed: last row, first and last column.
    assert cube.shape == (2, NY_CROP, NX_CROP)
    expected = _raw_cube("somxl010", 15.0).data[0, :-1, 1:-1]
    np.testing.assert_array_equal(cube.data[0], expected)
    # Horizontal bounds come from the (transposed) grid file.
    grid = _grid_cube()
    for name in ("latitude", "longitude"):
        coord = cube.coord(name)
        assert coord.bounds.shape == (NY_CROP, NX_CROP, 4)
        np.testing.assert_allclose(
            coord.bounds,
            np.transpose(grid.coord(name).bounds, (1, 0, 2)),
        )
        np.testing.assert_allclose(coord.points, grid.coord(name).points.T)
    assert not cube.coords("depth")


def test_load_native_cube_3d(raw_dir: Path, grid_dir: Path) -> None:
    files = _write_raw(raw_dir, "vozocrte", depth="depthu")

    cube = ORAS5()._load_native_cube(files, Variable("uo", "uo", "mon"))

    assert cube.var_name == "uo"
    assert cube.standard_name == "sea_water_x_velocity"
    assert cube.units == "m s-1"
    assert cube.shape == (2, 2, NY_CROP, NX_CROP)
    depth = cube.coord("depth")
    assert depth.var_name == "lev"
    assert depth.long_name == "ocean depth coordinate"
    assert depth.units == "m"
    np.testing.assert_allclose(depth.points, [0.5, 10.0])
    assert cube.coord_dims(depth) == (1,)
    assert cube.coord("latitude").bounds.shape == (NY_CROP, NX_CROP, 4)


@pytest.mark.parametrize("raw_depth", ORAS5._RAW_DEPTH_NAMES)
def test_fix_depth_coordinate_converts_units(raw_depth: str) -> None:
    cube = Cube(np.zeros(2))
    cube.add_dim_coord(DimCoord([50.0, 100.0], var_name=raw_depth, units="cm"), 0)

    ORAS5()._fix_depth_coordinate(cube)

    coord = cube.coord("depth")
    assert coord.var_name == "lev"
    assert coord.units == "m"
    np.testing.assert_allclose(coord.points, [0.5, 1.0])


def test_fix_depth_coordinate_noop_without_depth() -> None:
    cube = Cube(np.zeros(2))
    cube.add_dim_coord(DimCoord([1.0, 2.0], var_name="other", units="m"), 0)

    ORAS5()._fix_depth_coordinate(cube)

    assert not cube.coords("depth")
    assert cube.coord("other").standard_name is None


def test_add_grid_bounds_missing_grid_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ORAS5._GRID_DIR_ENV, str(tmp_path / "does_not_exist"))
    cube = _raw_cube("sosstsst", 15.0)[..., :-1, 1:-1]

    with pytest.raises(OSError, match="oras5_mesh_T"):
        ORAS5()._add_grid_bounds(cube, Variable("tos", "tos", "mon"))


def test_get_cube(
    raw_dir: Path,
    grid_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for year_month in ("199912", "200001"):
        iris.save(
            _raw_cube("sosstsst", 15.0 if year_month == "200001" else -16.0),
            raw_dir / _raw_filename("sosstsst", year_month),
        )
    calls: list[tuple[CubeList, Variable]] = []

    def fake_get_prepared_cube(cubes: CubeList, variable: Variable) -> Cube:
        calls.append((cubes, variable))
        return cubes[0]

    monkeypatch.setattr(
        climateeval.data._oras5,
        "get_prepared_cube",
        fake_get_prepared_cube,
    )
    variable = Variable("tos", "tos", "mon", timerange="2000/2000")

    cube = ORAS5().get_cube(tmp_path, variable, download_missing_data=False)

    assert len(calls) == 1
    assert calls[0][1] is variable
    assert len(calls[0][0]) == 1
    # Only the file within the timerange was loaded.
    assert cube.shape == (1, NY_CROP, NX_CROP)
    assert cube.var_name == "tos"
    assert cube.units == "degC"
    assert cube.coord("latitude").has_bounds()
