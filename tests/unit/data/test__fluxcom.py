from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import ClassVar

import iris
import numpy as np
import pytest
from esmvaltool.cmorizers.data.downloaders.ftp import FTPDownloader
from iris.coords import DimCoord
from iris.cube import Cube

import climateeval.data._fluxcom
from climateeval import Variable
from climateeval.data import FLUXCOM, DataSourceInformation

#: Raw FLUXCOM grid of the tests: latitudes descending, longitudes in
#: [-180, 180] (as in the original file).
RAW_LATS = [67.5, 22.5, -22.5, -67.5]
RAW_LONS = [-150.0, -90.0, -30.0, 30.0, 90.0, 150.0]

#: Raw land fraction (%), one distinct value per cell.
RAW_LAND_FRACTION = np.arange(24, dtype=np.float32).reshape(4, 6) * 4.0


def _raw_land_fraction_cube() -> Cube:
    return Cube(
        RAW_LAND_FRACTION,
        long_name="landfraction",
        var_name="landfraction",
        units="%",
        dim_coords_and_dims=[
            (DimCoord(RAW_LATS, var_name="latitude", units="degree"), 0),
            (DimCoord(RAW_LONS, var_name="longitude", units="degree"), 1),
        ],
    )


def _gpp_cube(lats: list[float], lons: list[float]) -> Cube:
    time = DimCoord(
        [15.0, 45.0],
        bounds=[[0.0, 31.0], [31.0, 59.0]],
        standard_name="time",
        var_name="time",
        units="days since 2000-01-01",
    )
    return Cube(
        np.ones((2, len(lats), len(lons)), dtype=np.float32),
        standard_name="gross_primary_productivity_of_biomass_expressed_as_carbon",
        var_name="gpp",
        units="kg m-2 s-1",
        dim_coords_and_dims=[
            (time, 0),
            (DimCoord(lats, standard_name="latitude", units="degrees"), 1),
            (DimCoord(lons, standard_name="longitude", units="degrees"), 2),
        ],
    )


class _FakeFTPDownloader(FTPDownloader):
    """Record requests and write a raw land fraction instead of downloading."""

    requests: ClassVar[list[tuple[str, ...]]] = []

    def connect(self) -> None:
        self.requests.append(("connect", self.server))

    def download_folder(
        self,
        server_path: str,
        sub_folder: str = "",  # noqa: ARG002
        filter_files: str | None = None,
    ) -> None:
        self.requests.append(("folder", server_path, filter_files or ""))

    def download_file(self, server_path: str, sub_folder: str = "") -> None:  # noqa: ARG002
        self.requests.append(("file", server_path))
        Path(self.local_folder).mkdir(parents=True, exist_ok=True)
        iris.save(
            _raw_land_fraction_cube(),
            Path(self.local_folder) / Path(server_path).name,
        )


@pytest.fixture
def fake_ftp(monkeypatch: pytest.MonkeyPatch) -> type[_FakeFTPDownloader]:
    monkeypatch.setattr(_FakeFTPDownloader, "requests", [])
    monkeypatch.setattr(climateeval.data._fluxcom, "FTPDownloader", _FakeFTPDownloader)
    return _FakeFTPDownloader


@pytest.fixture
def fake_gpp_format(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """Replace the ESMValTool GPP formatter by writing one CMOR-like file."""
    download_dirs: list[Path] = []

    def _format(
        self: FLUXCOM,
        data_root_dir: Path,
        variable: Variable,
        *,
        download_dir: Path,
    ) -> list[Path]:
        download_dirs.append(download_dir)
        assert (download_dir / "Tier3" / "FLUXCOM" / "landfraction.720.360.nc").exists()
        data_dir = self._get_data_dir(data_root_dir, variable)
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "OBS_FLUXCOM_reanaly_ANN-v1_Lmon_gpp_200001-200002.nc"
        iris.save(
            _gpp_cube(sorted(RAW_LATS), [30.0, 90.0, 150.0, 210.0, 270.0, 330.0]),
            path,
        )
        return [path]

    monkeypatch.setattr(FLUXCOM, "_esmvaltool_cmorizer_format", _format)
    return download_dirs


@pytest.fixture
def private_tempdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "tmp"
    path.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(path))
    return path


def test_data_source() -> None:
    data_source = FLUXCOM()
    assert data_source.name == "FLUXCOM"
    assert data_source.information == DataSourceInformation(
        name="FLUXCOM",
        category="observation",
        institute="MPI for Biogeochemistry",
        references=(
            "https://doi.org/10.5194/bg-17-1343-2020",
            "https://doi.org/10.1038/nature20780",
            "https://doi.org/10.5194/bg-13-4291-2016",
        ),
    )


def test_gpp_file_pattern() -> None:
    pattern = re.compile(climateeval.data._fluxcom.GPP_FILE_PATTERN)
    assert pattern.match("GPP.ANN.CRUNCEPv6.monthly.1980.nc")
    assert not pattern.match("GPP.MARS.CRUNCEPv6.monthly.1980.nc")
    assert not pattern.match("TER.ANN.CRUNCEPv6.monthly.1980.nc")


def test_download_unsupported_variable(
    tmp_path: Path,
    fake_ftp: type[_FakeFTPDownloader],
) -> None:
    with pytest.raises(NotImplementedError, match="does not provide variable 'tas'"):
        FLUXCOM().download(Variable("tas", "tas", "mon"), tmp_path)
    assert fake_ftp.requests == []


@pytest.mark.usefixtures("private_tempdir", "fake_ftp")
@pytest.mark.parametrize("var_name", ["gpp", "sftlf"])
def test_download(
    tmp_path: Path,
    fake_gpp_format: list[Path],
    var_name: str,
) -> None:
    data_root_dir = tmp_path / "data"
    frequency = "fx" if var_name == "sftlf" else "mon"

    paths = FLUXCOM().download(Variable(var_name, var_name, frequency), data_root_dir)

    # Both variables are formatted in one go, whichever was requested
    base = data_root_dir / "observation_FLUXCOM"
    assert len(list((base / "mon" / "gpp").glob("*.nc"))) == 1
    assert len(list((base / "fx" / "sftlf").glob("*.nc"))) == 1
    assert [p.parent.name for p in paths] == [var_name]
    # The raw data is removed after formatting
    assert not fake_gpp_format[0].exists()


@pytest.mark.usefixtures("private_tempdir", "fake_gpp_format")
def test_land_fraction_formatted_like_gpp(
    tmp_path: Path,
    fake_ftp: type[_FakeFTPDownloader],
) -> None:
    (path,) = FLUXCOM().download(Variable("sftlf", "sftlf", "fx"), tmp_path)

    cube = iris.load_cube(path)
    assert path.name == "OBS_FLUXCOM_reanaly_ANN-v1_fx_sftlf.nc"
    assert cube.var_name == "sftlf"
    assert cube.standard_name == "land_area_fraction"
    assert cube.units == "%"
    # Ascending latitudes and longitudes in [0, 360] (as the formatted GPP)
    np.testing.assert_allclose(cube.coord("latitude").points, sorted(RAW_LATS))
    np.testing.assert_allclose(
        cube.coord("longitude").points,
        [30.0, 90.0, 150.0, 210.0, 270.0, 330.0],
    )
    assert cube.coord("latitude").has_bounds()
    assert cube.coord("longitude").has_bounds()
    # The cell at 67.5N, 150W (raw row 0, column 0) moved with its coordinates
    assert cube.data[3, 3] == RAW_LAND_FRACTION[0, 0]
    assert cube.data[0, 0] == RAW_LAND_FRACTION[3, 3]


@pytest.mark.usefixtures("private_tempdir", "fake_gpp_format", "fake_ftp")
def test_get_cube_attaches_land_fraction(tmp_path: Path) -> None:
    cube = FLUXCOM().get_cube(tmp_path, Variable("gpp", "gpp", "mon"))

    land_fraction = cube.ancillary_variable("land_area_fraction")
    assert land_fraction.units == "%"
    assert cube.ancillary_variable_dims(land_fraction) == (1, 2)
    assert land_fraction.data[3, 3] == RAW_LAND_FRACTION[0, 0]


@pytest.mark.usefixtures("private_tempdir", "fake_gpp_format", "fake_ftp")
def test_get_cube_land_fraction_without_ancillary(tmp_path: Path) -> None:
    cube = FLUXCOM().get_cube(tmp_path, Variable("sftlf", "sftlf", "fx"))
    assert cube.var_name == "sftlf"
    assert not cube.ancillary_variables()


def test_get_cube_mismatched_land_fraction(tmp_path: Path) -> None:
    base = tmp_path / "observation_FLUXCOM"
    (base / "mon" / "gpp").mkdir(parents=True)
    (base / "fx" / "sftlf").mkdir(parents=True)
    iris.save(_gpp_cube([-45.0, 45.0], [90.0, 270.0]), base / "mon" / "gpp" / "gpp.nc")
    land_fraction = _gpp_cube([-60.0, 0.0, 60.0], [90.0, 270.0])[0]
    land_fraction.remove_coord("time")
    land_fraction.var_name = "sftlf"
    land_fraction.standard_name = "land_area_fraction"
    land_fraction.units = "%"
    iris.save(land_fraction, base / "fx" / "sftlf" / "sftlf.nc")

    with pytest.raises(ValueError, match="does not match the grid"):
        FLUXCOM().get_cube(
            tmp_path,
            Variable("gpp", "gpp", "mon"),
            download_missing_data=False,
        )
