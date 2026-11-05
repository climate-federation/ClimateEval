from __future__ import annotations

import tempfile
from pathlib import Path

import iris
import numpy as np
import pytest
from esmvaltool.cmorizers.data.downloaders.ftp import FTPDownloader
from iris.coords import DimCoord
from iris.cube import Cube

import climateeval.data._fluxcom
from climateeval import Variable
from climateeval.data import FLUXCOM

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


class _FakeFTPDownloader(FTPDownloader):
    """Write a raw land fraction instead of downloading."""

    def connect(self) -> None:
        pass

    def download_folder(
        self,
        server_path: str,
        sub_folder: str = "",
        filter_files: str | None = None,
    ) -> None:
        pass

    def download_file(self, server_path: str, sub_folder: str = "") -> None:  # noqa: ARG002
        Path(self.local_folder).mkdir(parents=True, exist_ok=True)
        iris.save(
            _raw_land_fraction_cube(),
            Path(self.local_folder) / Path(server_path).name,
        )


@pytest.fixture(autouse=True)
def _offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fake the FTP server and skip the ESMValTool GPP formatter."""
    monkeypatch.setattr(climateeval.data._fluxcom, "FTPDownloader", _FakeFTPDownloader)
    monkeypatch.setattr(
        FLUXCOM,
        "_esmvaltool_cmorizer_format",
        lambda *_args, **_kwargs: [],
    )
    private_tempdir = tmp_path / "tmp"
    private_tempdir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(private_tempdir))


def test_land_fraction_formatted_like_gpp(tmp_path: Path) -> None:
    (path,) = FLUXCOM().download(Variable("sftlf", "sftlf", "fx"), tmp_path / "data")

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
