from __future__ import annotations

from typing import TYPE_CHECKING

import iris
import numpy as np
import pytest
from iris.coords import DimCoord
from iris.cube import Cube

from climateeval import Coordinate, Variable
from climateeval._utils import str_to_object
from climateeval.data import DataSourceInformation, LocalCMORDataSource
from climateeval.diags.simple import SimpleDiagnostic
from climateeval.exceptions import DownloadError, MissingDataError

if TYPE_CHECKING:
    from pathlib import Path

REFERENCE_INFORMATION = DataSourceInformation(
    name="MyObs",
    category="observation",
)


def _make_cube(var_name: str, year: int, value: float) -> Cube:
    """Create a small monthly CMOR-compliant cube covering one year."""
    time = DimCoord(
        np.arange(12) * 30.0 + 15.0,
        var_name="time",
        standard_name="time",
        units=f"days since {year}-01-01",
    )
    lat = DimCoord(
        [-30.0, 30.0],
        var_name="lat",
        standard_name="latitude",
        units="degrees_north",
    )
    lon = DimCoord(
        [0.0, 90.0],
        var_name="lon",
        standard_name="longitude",
        units="degrees_east",
    )
    return Cube(
        np.full((12, 2, 2), value, dtype=np.float32),
        var_name=var_name,
        standard_name="air_temperature",
        units="K",
        dim_coords_and_dims=[(time, 0), (lat, 1), (lon, 2)],
    )


def _save(cube: Cube, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    iris.save(cube, str(path))


def _cmor_name(var_name: str, year: int, grid: str = "gn") -> str:
    return f"{var_name}_Amon_HeldOut_historical_r1i1p1f1_{grid}_{year}01-{year}12.nc"


@pytest.fixture
def cmor_dir(tmp_path: Path) -> Path:
    """Create a flat directory with two years of CMOR-compliant output."""
    cmor_dir = tmp_path / "cmor"
    for year in (2000, 2001):
        _save(
            _make_cube("tas", year, 280.0 + year - 2000),
            cmor_dir / _cmor_name("tas", year),
        )
    return cmor_dir


@pytest.fixture(autouse=True)
def _no_path_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure a user's environment does not leak into the tests."""
    monkeypatch.delenv(LocalCMORDataSource._PATH_ENV, raising=False)


def test_default_information() -> None:
    data_source = LocalCMORDataSource()

    assert data_source.name == "LocalCMOR"
    assert data_source.id == "model_LocalCMOR"
    assert data_source.preferred_grid == "gn"


def test_for_directory(cmor_dir: Path) -> None:
    data_source_cls = LocalCMORDataSource.for_directory(
        cmor_dir,
        name="HeldOutESM",
        category="model",
        institute="MPI-M",
        exp="historical",
        variant="r1i1p1f1",
        references=("https://example.org",),
    )

    # Suites instantiate data sources without arguments
    data_source = data_source_cls()

    assert issubclass(data_source_cls, LocalCMORDataSource)
    assert data_source.id == "model_HeldOutESM_historical_r1i1p1f1"
    assert data_source.information == DataSourceInformation(
        name="HeldOutESM",
        category="model",
        institute="MPI-M",
        exp="historical",
        variant="r1i1p1f1",
        references=("https://example.org",),
    )
    assert data_source.path == cmor_dir


def test_for_directory_class_is_importable(
    cmor_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generated class can be referenced by class path from a suite YAML."""
    data_source_cls = LocalCMORDataSource.for_directory(cmor_dir, name="HeldOutESM")
    monkeypatch.setattr(
        "climateeval.data._local_cmor.HeldOutESM",
        data_source_cls,
        raising=False,
    )

    loaded_cls = str_to_object("climateeval.data._local_cmor.HeldOutESM")

    assert loaded_cls is data_source_cls
    assert loaded_cls().id == "model_HeldOutESM"


def test_reference_data_of_simple_diagnostic(
    cmor_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A held-out model can act as the reference of a suite (perfect model)."""

    class MockDiagnostic(SimpleDiagnostic):
        _final_coordinates = (Coordinate("lat"),)

        def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
            return cube

    data_source_cls = LocalCMORDataSource.for_directory(
        cmor_dir,
        name="HeldOutESM",
        exp="historical",
    )
    monkeypatch.setattr(
        "climateeval.data._local_cmor.HeldOutESM",
        data_source_cls,
        raising=False,
    )

    diagnostic = MockDiagnostic.from_suite_dict(
        "mock_diagnostic",
        {
            "variables": [
                {
                    "id": "tas",
                    "var_name": "tas",
                    "frequency": "mon",
                    "reference_data": "climateeval.data._local_cmor.HeldOutESM",
                },
            ],
        },
        {},
    )

    reference_data = diagnostic._reference_data[Variable("tas", "tas", "mon")]
    assert isinstance(reference_data, LocalCMORDataSource)
    assert reference_data.id == "model_HeldOutESM_historical"


def test_get_cube_from_flat_dir(cmor_dir: Path) -> None:
    data_source = LocalCMORDataSource(cmor_dir, information=REFERENCE_INFORMATION)

    cube = data_source.get_cube("/not/used", Variable("tas", "tas", "mon"))

    assert cube.var_name == "tas"
    assert cube.shape == (24, 2, 2)


def test_get_cube_from_single_file(cmor_dir: Path) -> None:
    path = cmor_dir / _cmor_name("tas", 2000)
    data_source = LocalCMORDataSource(path, information=REFERENCE_INFORMATION)

    cube = data_source.get_cube("/not/used", Variable("tas", "tas", "mon"))

    assert cube.shape == (12, 2, 2)


def test_get_cube_from_drs_tree(tmp_path: Path) -> None:
    root = tmp_path / "Amon"
    for grid, value in (("gn", 300.0), ("gr", 400.0)):
        _save(
            _make_cube("tas", 2000, value),
            root / "tas" / grid / "v20200101" / _cmor_name("tas", 2000, grid),
        )
    variable = Variable("tas", "tas", "mon")

    native_grid = LocalCMORDataSource(root, information=REFERENCE_INFORMATION)
    regular_grid = LocalCMORDataSource(
        root,
        information=REFERENCE_INFORMATION,
        preferred_grid="gr",
    )

    assert native_grid.get_cube("/not/used", variable).data.mean() == 300.0
    assert regular_grid.get_cube("/not/used", variable).data.mean() == 400.0


def test_get_cube_honours_timerange(cmor_dir: Path) -> None:
    data_source = LocalCMORDataSource(cmor_dir, information=REFERENCE_INFORMATION)
    variable = Variable("tas", "tas", "mon", timerange="20010101/20011231")

    cube = data_source.get_cube("/not/used", variable)

    assert cube.shape == (12, 2, 2)
    assert cube.data.mean() == 281.0


def test_get_cube_from_env_variable(
    cmor_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shipped class can be pointed at data without writing any Python."""
    monkeypatch.setenv(LocalCMORDataSource._PATH_ENV, str(cmor_dir))

    cube = LocalCMORDataSource().get_cube("/not/used", Variable("tas", "tas", "mon"))

    assert cube.shape == (24, 2, 2)


def test_constructor_wins_over_env_variable(
    cmor_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(LocalCMORDataSource._PATH_ENV, "/not/used")

    data_source = LocalCMORDataSource(cmor_dir, information=REFERENCE_INFORMATION)

    assert data_source.path == cmor_dir


def test_missing_path(tmp_path: Path) -> None:
    data_source = LocalCMORDataSource(
        tmp_path / "does_not_exist",
        information=REFERENCE_INFORMATION,
    )

    with pytest.raises(MissingDataError, match="not found for"):
        data_source.get_cube("/not/used", Variable("tas", "tas", "mon"))


def test_no_path_given() -> None:
    with pytest.raises(MissingDataError, match="No path given for"):
        _ = LocalCMORDataSource().path


def test_no_data_for_variable(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    data_source = LocalCMORDataSource(empty_dir, information=REFERENCE_INFORMATION)

    with pytest.raises(MissingDataError, match="No data found in"):
        data_source.get_cube("/not/used", Variable("tas", "tas", "mon"))


def test_download_fails(cmor_dir: Path) -> None:
    data_source = LocalCMORDataSource(cmor_dir, information=REFERENCE_INFORMATION)

    with pytest.raises(DownloadError, match="local data that cannot be downloaded"):
        data_source.download(Variable("tas", "tas", "mon"), "/not/used")
