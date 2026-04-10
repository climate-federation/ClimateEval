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
YEARS = [str(y) for y in range(1979, 2025)]
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

REQUEST_MAP: dict[str, Any] = {
    "pr": {
        "dataset": "reanalysis-era5-single-levels-monthly-means",
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["total_precipitation"],
        "month": MONTHS,
        "time": ["00:00"],
        "year": YEARS,
        "cmip6_table_id": "Amon",  # for ESMValTool
    },
    "ta": {
        "dataset": "reanalysis-era5-pressure-levels-monthly-means",
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["temperature"],
        "month": MONTHS,
        "time": ["00:00"],
        "year": YEARS,
        "pressure_level": PRESSURE_LEVELS,
        "cmip6_table_id": "Amon",  # for ESMValTool
    },
    "tas": {
        "dataset": "reanalysis-era5-single-levels-monthly-means",
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["2m_temperature"],
        "month": MONTHS,
        "time": ["00:00"],
        "year": YEARS,
        "cmip6_table_id": "Amon",  # for ESMValTool
    },
}


class ERA5(DataSource):
    """Manage ERA5 data."""

    def _download(self, variable: Variable) -> list[Path]:
        """Download data."""
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
        download_dir = tmp_dir / "download"
        download_dir.mkdir(parents=True, exist_ok=True)
        tmp_target = download_dir / f"raw_{self.name}_{variable.var_name}.nc"

        # Download data
        cdsapi.Client().retrieve(dataset, request).download(tmp_target)

        # Reformat data with ESMValTool and save
        cubes = iris.load(tmp_target)
        cube = fix_metadata(
            cubes,
            short_name=variable.var_name,
            project="native6",
            dataset=self.name,
            mip=cmip6_table_id,
            frequency=variable.frequency,
        )[0]
        cube = fix_data(
            cube,
            short_name=variable.var_name,
            project="native6",
            dataset=self.name,
            mip=cmip6_table_id,
            frequency=variable.frequency,
        )
        data_dir = self._get_data_dir(variable)
        data_dir.mkdir(parents=True, exist_ok=True)
        target = data_dir / f"{self.name}_{variable.var_name}.nc"
        iris.save(cube, target)

        return [target]
