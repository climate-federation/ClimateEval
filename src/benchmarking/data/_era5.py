"""Provide ERA5 data."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cdsapi
import iris
from esmvalcore.cmor.fix import fix_data, fix_metadata

from benchmarking.data._base import DataSource, DataSourceInformation

if TYPE_CHECKING:
    from benchmarking import Variable

MONTHS = [f"{m:02d}" for m in range(1, 13)]
YEARS = [str(y) for y in range(1979, 2027)]
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

BASE: dict[str, Any] = {
    "product_type": ["monthly_averaged_reanalysis"],
    "month": MONTHS,
    "time": ["00:00"],
    "year": YEARS,
}
REQUEST_SL: dict[str, Any] = {
    **BASE,
    "dataset": "reanalysis-era5-single-levels-monthly-means",
}
REQUEST_PL: dict[str, Any] = {
    **BASE,
    "dataset": "reanalysis-era5-pressure-levels-monthly-means",
    "pressure_level": PRESSURE_LEVELS,
}
REQUEST_MAP: dict[str, Any] = {
    "clivi": {**REQUEST_SL, "variable": ["total_column_cloud_ice_water"]},
    "clt": {**REQUEST_SL, "variable": ["total_cloud_cover"]},
    "cl": {**REQUEST_PL, "variable": ["fraction_of_cloud_cover"]},
    "hus": {**REQUEST_PL, "variable": ["specific_humidity"]},
    "lwp": {**REQUEST_SL, "variable": ["total_column_cloud_liquid_water"]},
    "pr": {**REQUEST_SL, "variable": ["total_precipitation"]},
    "rlut": {**REQUEST_SL, "variable": ["top_net_thermal_radiation"]},
    "rlutcs": {**REQUEST_SL, "variable": ["top_net_thermal_radiation_clear_sky"]},
    "ta": {**REQUEST_PL, "variable": ["temperature"]},
    "tas": {**REQUEST_SL, "variable": ["2m_temperature"]},
    "ua": {**REQUEST_PL, "variable": ["u_component_of_wind"]},
    "va": {**REQUEST_PL, "variable": ["v_component_of_wind"]},
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
        ),
    )

    def download(self, data_root_dir: Path, variable: Variable) -> list[Path]:
        """Download data.

        Note
        ----
        This requires a valid API key in ``~/.cdsapirc``. See section "Setup
        the CDS Api key" at https://cds.climate.copernicus.eu/how-to-api for
        details.

        """
        if variable.var_name not in REQUEST_MAP:
            msg = f"Variable '{variable.var_name}' not supported for ERA5"
            raise NotImplementedError(msg)

        request = dict(REQUEST_MAP[variable.var_name])
        request["data_format"] = "netcdf"
        request["download_format"] = "unarchived"
        dataset = request.pop("dataset")

        # Raw downloaded data will be stored in temporary directory. Our "raw"
        # data will be the formatted data by ESMValTool.
        tmp_dir = Path(tempfile.TemporaryDirectory().name)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_target = tmp_dir / f"raw_{self.id}_{variable.var_name}.nc"

        # Download data
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
        era5_facets = {
            "short_name": variable.var_name,
            "project": "native6",
            "dataset": self.name,
            "mip": variable.cmip6_table_id,
            "frequency": variable.frequency,
        }
        cubes = iris.load(input_file)
        cube = fix_metadata(cubes, **era5_facets)[0]
        cube = fix_data(cube, **era5_facets)
        if cube.coords("expver"):
            cube.remove_coord("expver")
        data_dir = self._get_data_dir(data_root_dir, variable.var_name)
        data_dir.mkdir(parents=True, exist_ok=True)
        target = data_dir / self._get_output_filename(cube, **era5_facets)
        iris.save(cube, target)
        return target
