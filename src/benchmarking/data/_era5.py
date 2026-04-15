"""Provide ERA5 data."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cdsapi
import iris
from esmvalcore.cmor.fix import fix_data, fix_metadata

from benchmarking.data._base import DataSource

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

REQUEST_BASE: dict[str, Any] = {
    "product_type": ["monthly_averaged_reanalysis"],
    "month": MONTHS,
    "time": ["00:00"],
    "year": YEARS,
    "cmip6_table_id": "Amon",  # for ESMValTool
}
REQUEST_SINGLE_LEVELS: dict[str, Any] = {
    **dict(REQUEST_BASE),
    "dataset": "reanalysis-era5-single-levels-monthly-means",
}
REQUEST_PRESSURE_LEVELS: dict[str, Any] = {
    **dict(REQUEST_BASE),
    "dataset": "reanalysis-era5-pressure-levels-monthly-means",
    "pressure_level": PRESSURE_LEVELS,
}
REQUEST_MAP: dict[str, Any] = {
    "hus": {**dict(REQUEST_PRESSURE_LEVELS), "variable": ["specific_humidity"]},
    "pr": {**dict(REQUEST_SINGLE_LEVELS), "variable": ["total_precipitation"]},
    "ta": {**dict(REQUEST_PRESSURE_LEVELS), "variable": ["temperature"]},
    "tas": {**dict(REQUEST_SINGLE_LEVELS), "variable": ["2m_temperature"]},
    "ua": {**dict(REQUEST_PRESSURE_LEVELS), "variable": ["u_component_of_wind"]},
    "va": {**dict(REQUEST_PRESSURE_LEVELS), "variable": ["v_component_of_wind"]},
}


class ERA5(DataSource):
    """Manage ERA5 data."""

    def _download(self, variable: Variable) -> list[Path]:
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
        cmip6_table_id = request.pop("cmip6_table_id")

        # Raw downloaded data will be stored in temporary directory. Our "raw"
        # data will be the formatted data by ESMValTool.
        tmp_dir = Path(tempfile.TemporaryDirectory().name)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_target = tmp_dir / f"raw_{self.name}_{variable.var_name}.nc"

        # Download data
        cdsapi.Client().retrieve(dataset, request).download(tmp_target)

        # Reformat data with ESMValTool and save
        era5_facets = {
            "short_name": variable.var_name,
            "project": "native6",
            "dataset": self.name,
            "mip": cmip6_table_id,
            "frequency": variable.frequency,
        }
        cubes = iris.load(tmp_target)
        cube = fix_metadata(cubes, **era5_facets)[0]
        cube = fix_data(cube, **era5_facets)
        if cube.coords("expver"):
            cube.remove_coord("expver")
        data_dir = self._get_data_dir(variable)
        data_dir.mkdir(parents=True, exist_ok=True)
        target = data_dir / self._get_output_filename(cube, **era5_facets)
        iris.save(cube, target)

        return [target]
