"""Provide FLUXCOM gross primary production data."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import iris
from esmvaltool.cmorizers.data import utilities as cmorizer_utils
from esmvaltool.cmorizers.data.downloaders.ftp import FTPDownloader
from iris.coords import AncillaryVariable

from climateeval import Variable
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation

if TYPE_CHECKING:
    from iris.cube import Cube


#: Anonymous FTP server of the MPI for Biogeochemistry.
FTP_SERVER = "ftp.bgc-jena.mpg.de"

#: Folder of the FLUXCOM RS+METEO carbon fluxes (Jung et al., 2017 subset).
FTP_DIR = "/pub/outgoing/FluxCom/CarbonFluxes_v1_2017/RS+METEO"

#: Monthly GPP files (ANN method, CRUNCEPv6 forcing, one file per year).
GPP_FILE_PATTERN = r"GPP\.ANN\.CRUNCEPv6\.monthly\.\d{4}\.nc$"

#: Land fraction (in %) of the 0.5 degree FLUXCOM grid.
LAND_FRACTION_FILE = "landfraction.720.360.nc"


class FLUXCOM(ESMValToolCMORizerDataSource):
    """Manage FLUXCOM data.

    FLUXCOM RS+METEO (ANN method, CRUNCEPv6 forcing) upscales eddy-covariance
    measurements to monthly global gross primary production (``gpp``) on a
    0.5 degree grid for 1980-2013. The GPP is given per unit land area, so the
    land fraction of the grid (``sftlf``) that comes with the product is
    provided as well and attached to every ``gpp`` cube as an ancillary
    variable.

    ESMValTool's CMORizer for FLUXCOM has no downloader (its notes point to a
    registration portal). The files are, however, freely available (CC BY 4.0)
    on the anonymous FTP server of the MPI for Biogeochemistry, so they are
    downloaded from there and formatted with the ESMValTool CMORizer.

    """

    _information = DataSourceInformation(
        name="FLUXCOM",
        category="observation",
        institute="MPI for Biogeochemistry",
        references=(
            "https://doi.org/10.5194/bg-17-1343-2020",
            "https://doi.org/10.1038/nature20780",
            "https://doi.org/10.5194/bg-13-4291-2016",
        ),
    )

    _supported_var_names = ("gpp", "sftlf")
    _tier = 3

    def get_cube(
        self,
        data_root_dir: Path | str,
        variable: Variable,
        *,
        download_missing_data: bool = True,
    ) -> Cube:
        """Get data cube (with the land fraction attached to ``gpp``)."""
        cube = super().get_cube(
            data_root_dir,
            variable,
            download_missing_data=download_missing_data,
        )
        if variable.var_name != "gpp":
            return cube

        # Prepare the land fraction with the same regional settings as the
        # flux so that both end up on exactly the same grid
        land_fraction_variable = Variable(
            "sftlf",
            "sftlf",
            "fx",
            min_lat=variable.min_lat,
            max_lat=variable.max_lat,
            min_lon=variable.min_lon,
            max_lon=variable.max_lon,
        )
        land_fraction = super().get_cube(
            data_root_dir,
            land_fraction_variable,
            download_missing_data=download_missing_data,
        )
        dims = cube.coord_dims("latitude") + cube.coord_dims("longitude")
        if land_fraction.shape != tuple(cube.shape[d] for d in dims):
            msg = (
                f"Land fraction of {self.id} (shape {land_fraction.shape}) does "
                f"not match the grid of {cube.summary(shorten=True)}"
            )
            raise ValueError(msg)
        cube.add_ancillary_variable(
            AncillaryVariable(
                land_fraction.core_data(),
                standard_name=land_fraction.standard_name,
                long_name=land_fraction.long_name,
                var_name=land_fraction.var_name,
                units=land_fraction.units,
            ),
            dims,
        )
        return cube

    def download(self, variable: Variable, data_root_dir: Path | str) -> list[Path]:
        """Download data.

        Both variables are downloaded and formatted together, so a later
        request for the other one is served from disk.

        """
        data_root_dir = Path(data_root_dir)

        self._check_supported_variables(variable)

        dataset_info = {
            "tier": self._tier,
            "source": f"ftp://{FTP_SERVER}{FTP_DIR}",
            "last_access": datetime.now(UTC),
            "info": "",
        }

        # Raw data (about 170 MB) only lives in a temporary directory; our
        # "raw" data is the formatted data
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp_dir = Path(tmp_name)
            with self._set_esmvaltool_output_dir(tmp_dir):
                downloader = FTPDownloader(
                    original_data_dir=tmp_dir,
                    server=FTP_SERVER,
                    dataset=self.name,
                    dataset_info=dataset_info,
                    overwrite=False,
                )
                downloader.connect()
                downloader.download_folder(
                    f"{FTP_DIR}/CRUNCEPv6/raw/monthly/",
                    filter_files=GPP_FILE_PATTERN,
                )
                downloader.download_file(
                    f"{FTP_DIR}/landfraction/{LAND_FRACTION_FILE}",
                )

            land_fraction_path = self._format_land_fraction(
                Path(downloader.local_folder) / LAND_FRACTION_FILE,
                data_root_dir,
            )
            gpp_variable = (
                variable
                if variable.var_name == "gpp"
                else Variable("gpp", "gpp", "mon")
            )
            gpp_paths = self._esmvaltool_cmorizer_format(
                data_root_dir,
                gpp_variable,
                download_dir=tmp_dir,
            )

        if variable.var_name == "sftlf":
            return [land_fraction_path]
        return gpp_paths

    def _format_land_fraction(self, raw_path: Path, data_root_dir: Path) -> Path:
        """Format the FLUXCOM land fraction as CMOR ``sftlf``.

        The coordinates are fixed with the same ESMValTool utilities as the
        GPP, so both are on the same grid (ascending latitudes, longitudes in
        [0, 360]).

        """
        cmor_config: dict[str, Any] = cmorizer_utils.read_cmor_config(self.name)
        attributes = dict(cmor_config["attributes"])
        attributes["mip"] = "fx"
        var_info = cmor_config["cmor_table"].get_variable("fx", "sftlf")

        with setup_esmvaltool_config_and_logging():
            cube = iris.load_cube(raw_path, "landfraction")
            for coord_name, units in (
                ("latitude", "degrees_north"),
                ("longitude", "degrees_east"),
            ):
                coord = cube.coord(coord_name)
                coord.standard_name = coord_name
                coord.units = units
            cmorizer_utils.fix_var_metadata(cube, var_info)
            cube = cmorizer_utils.fix_coords(cube)
            cmorizer_utils.set_global_atts(cube, attributes)

            data_dir = self._get_data_dir(
                data_root_dir,
                Variable("sftlf", "sftlf", "fx"),
            )
            data_dir.mkdir(parents=True, exist_ok=True)
            cmorizer_utils.save_variable(cube, "sftlf", str(data_dir), attributes)

        return next(data_dir.glob("*_fx_sftlf.nc"))
