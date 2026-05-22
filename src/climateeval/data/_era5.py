"""Provide ERA5 data."""

from __future__ import annotations

import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cdsapi
import iris
from esmvalcore.cmor.fix import fix_data, fix_metadata

from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._utils import load_cubes
from climateeval.data._base import DataSource
from climateeval.data._utils import DataSourceInformation

if TYPE_CHECKING:
    from climateeval import Variable

ALL_DAYS = [f"{d:02d}" for d in range(1, 32)]
ALL_MONTHS = [f"{m:02d}" for m in range(1, 13)]
ALL_TIMES = [f"{h:02d}:00" for h in range(24)]
PRESSURE_LEVELS = [
    "1",
    "2",
    "3",
    "5",
    "7",
    "10",
    "20",
    "30",
    "50",
    "70",
    "100",
    "125",
    "150",
    "175",
    "200",
    "225",
    "250",
    "300",
    "350",
    "400",
    "450",
    "500",
    "550",
    "600",
    "650",
    "700",
    "750",
    "775",
    "800",
    "825",
    "850",
    "875",
    "900",
    "925",
    "950",
    "975",
    "1000",
]

VARIABLE_MAPPING: dict[str, str] = {
    "clivi": "total_column_cloud_ice_water",
    "clt": "total_cloud_cover",
    "hus": "specific_humidity",
    "lwp": "total_column_cloud_liquid_water",
    "pr": "total_precipitation",
    "rlut": "top_net_thermal_radiation",
    "rlutcs": "top_net_thermal_radiation_clear_sky",
    "ta": "temperature",
    "tas": "2m_temperature",
    "ua": "u_component_of_wind",
    "va": "v_component_of_wind",
}


class ERA5(DataSource):
    """Manage ERA5 data."""

    _INFORMATION = DataSourceInformation(
        name="ERA5",
        category="reanalysis",
        institute="ECMWF",
        references=(
            "https://doi.org/10.24381/cds.f17050d7",
            "https://doi.org/10.24381/cds.6860a573",
            "https://doi.org/10.24381/cds.adbb2d47",
        ),
    )

    def download(self, variable: Variable, data_root_dir: Path) -> list[Path]:
        """Download data.

        Note
        ----
        This requires a valid API key in ``~/.cdsapirc``. See section "Setup
        the CDS Api key" at https://cds.climate.copernicus.eu/how-to-api for
        details.

        """
        if variable.var_name not in VARIABLE_MAPPING:
            msg = f"Variable '{variable.var_name}' is not supported for ERA5"
            raise NotImplementedError(msg)

        if variable.ndim not in (3, 4):
            msg = (
                f"{variable.ndim}D data (variable '{variable.var_name}') is not "
                f"supported for ERA5"
            )
            raise ValueError(msg)

        # Raw downloaded data will be stored in temporary directory. Our "raw"
        # data will be the formatted data by ESMValTool.
        tmp_dir = Path(tempfile.TemporaryDirectory().name)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_target = tmp_dir / f"raw_{self.id}_{variable.var_name}.nc"

        # Download data
        dataset, request = self._get_cds_client_retrieve_kwargs(variable)
        request["variable"] = [VARIABLE_MAPPING[variable.var_name]]
        request["data_format"] = "netcdf"
        request["download_format"] = "unarchived"
        cdsapi.Client().retrieve(dataset, request).download(tmp_target)

        # Reformat data with ESMValTool and save
        target = self.format(
            tmp_target,
            variable,
            data_root_dir=data_root_dir,
        )

        return [target]

    def format(
        self,
        input_file: Path,
        variable: Variable,
        *,
        data_root_dir: Path,
    ) -> Path:
        era5_facets = self._get_esmvaltool_facets(variable)
        cubes = load_cubes(input_file)
        with setup_esmvaltool_config_and_logging():
            cube = fix_metadata(cubes, **era5_facets)[0]
            cube = fix_data(cube, **era5_facets)
        if cube.coords("expver"):
            cube.remove_coord("expver")
        data_dir = self._get_data_dir(data_root_dir, variable)
        data_dir.mkdir(parents=True, exist_ok=True)
        target = data_dir / self._get_output_filename(cube, **era5_facets)
        iris.save(cube, target)
        return target

    @abstractmethod
    def _get_cds_client_retrieve_kwargs(
        self,
        variable: Variable,
    ) -> tuple[str, dict[str, Any]]:
        """Get arguments for :meth:`cds.Client.retrieve`."""

    @abstractmethod
    def _get_esmvaltool_facets(self, variable: Variable) -> dict[str, Any]:
        """Get ESMValTool facets."""


class ERA5Monthly(ERA5):
    """Manage monthly ERA5 data."""

    def _get_cds_client_retrieve_kwargs(
        self,
        variable: Variable,
    ) -> tuple[str, dict[str, Any]]:
        """Get arguments for :meth:`cds.Client.retrieve`."""
        request: dict[str, Any] = {
            "product_type": "monthly_averaged_reanalysis",
            "year": [str(y) for y in range(1979, 2027)],
            "month": ALL_MONTHS,
            "time": ["00:00"],
        }

        if variable.ndim == 3:  # noqa: PLR2004
            dataset = "reanalysis-era5-single-levels-monthly-means"
        else:
            dataset = "reanalysis-era5-pressure-levels-monthly-means"
            request["pressure_level"] = PRESSURE_LEVELS

        return (dataset, request)

    def _get_esmvaltool_facets(self, variable: Variable) -> dict[str, Any]:
        """Get ESMValTool facets."""
        return {
            "short_name": variable.var_name,
            "project": "native6",
            "dataset": "ERA5",
            "mip": "Amon",  # so far, all supported variables are atmospheric
            "frequency": "mon",
        }


class ERA5Hourly(ERA5):
    """Manage hourly ERA5 data."""

    def _get_cds_client_retrieve_kwargs(
        self,
        variable: Variable,
    ) -> tuple[str, dict[str, Any]]:
        """Get arguments for :meth:`cds.Client.retrieve`."""
        request: dict[str, Any] = {
            "product_type": "reanalysis",
            "year": [str(y) for y in range(2000, 2001)],
            "month": ALL_MONTHS,
            "day": ALL_DAYS,
            "time": ALL_TIMES,
        }

        if variable.ndim == 3:  # noqa: PLR2004
            dataset = "reanalysis-era5-single-levels"
        else:
            dataset = "reanalysis-era5-pressure-levels"
            request["pressure_level"] = PRESSURE_LEVELS

        return (dataset, request)

    def _get_esmvaltool_facets(self, variable: Variable) -> dict[str, Any]:
        """Get ESMValTool facets."""
        return {
            "short_name": variable.var_name,
            "project": "native6",
            "dataset": "ERA5",
            "mip": "3hr",  # so far, all supported variables are atmospheric
            "frequency": "1hr",
        }
