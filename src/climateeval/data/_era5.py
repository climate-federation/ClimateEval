"""Provide ERA5 data."""

from __future__ import annotations

import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cdsapi
import iris
from esmvalcore.cmor.fix import fix_data, fix_metadata
from esmvalcore.preprocessor import clip_timerange, daily_statistics
from iris.coords import CellMethod

from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._utils import load_cubes
from climateeval.data._base import DataSource
from climateeval.data._utils import DataSourceInformation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from iris.cube import Cube

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
    "evspsbl": "evaporation",
    "hus": "specific_humidity",
    "lwp": "total_column_cloud_liquid_water",
    "pr": "total_precipitation",
    "prw": "total_column_water_vapour",
    "ps": "surface_pressure",
    "rlut": "top_net_thermal_radiation",
    "rlutcs": "top_net_thermal_radiation_clear_sky",
    "ta": "temperature",
    "tas": "2m_temperature",
    "ua": "u_component_of_wind",
    "va": "v_component_of_wind",
}

DAILY_SOURCE_VARIABLES: dict[str, str] = {
    "tasmax": "tas",
    "tasmin": "tas",
}
"""Map daily variables to the hourly ERA5 variable they are aggregated from.

Daily variables that are not listed here are aggregated from the hourly ERA5
field of the same name (see :data:`VARIABLE_MAPPING`).

"""

DAILY_OPERATORS: dict[str, str] = {
    "tasmax": "max",
    "tasmin": "min",
}
"""Map daily variables to the operator used to aggregate the hourly data.

Daily variables that are not listed here are aggregated with the daily
``mean``. Note that the ESMValTool ``native6`` fixes convert accumulated ERA5
fields (e.g. total precipitation) to fluxes *before* the aggregation, so their
daily mean is the daily accumulation divided by the length of the day, i.e. the
daily precipitation rate in kg m-2 s-1.

"""

_CELL_METHODS: dict[str, str] = {
    "mean": "mean",
    "max": "maximum",
    "min": "minimum",
}
"""Map the daily operators to the CF cell method they produce."""


class ERA5(DataSource):
    """Manage ERA5 data."""

    _information = DataSourceInformation(
        name="ERA5",
        category="reanalysis",
        institute="ECMWF",
        references=(
            "https://doi.org/10.24381/cds.f17050d7",
            "https://doi.org/10.24381/cds.6860a573",
            "https://doi.org/10.24381/cds.adbb2d47",
        ),
    )

    def download(self, variable: Variable, data_root_dir: Path | str) -> list[Path]:
        """Download data.

        Note
        ----
        This requires a valid API key in ``~/.cdsapirc``. See section "Setup
        the CDS Api key" at https://cds.climate.copernicus.eu/how-to-api for
        details.

        """
        data_root_dir = Path(data_root_dir)

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
        input_file: Path | str,
        variable: Variable,
        *,
        data_root_dir: Path | str,
    ) -> Path:
        """Format ERA5 data."""
        data_root_dir = Path(data_root_dir)

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


class ERA5Daily(ERA5):
    """Manage daily ERA5 data.

    Daily data are aggregated from the hourly ERA5 reanalysis: the hourly
    fields are downloaded one year at a time over the variable's
    :attr:`~climateeval.Variable.timerange`, CMORized with the ESMValTool
    ``native6`` hourly fixes, and then reduced to daily statistics with
    ESMValCore's :func:`~esmvalcore.preprocessor.daily_statistics` (see
    :data:`DAILY_OPERATORS`). This provides the daily temperature extremes
    ``tasmax``/``tasmin``, which the CDS does not serve as part of the hourly
    or monthly reanalysis.

    Note
    ----
    Hourly ERA5 is large, so - unlike for :class:`ERA5Monthly` and
    :class:`ERA5Hourly` - the variable's ``timerange`` must be set explicitly
    (e.g. ``timerange="1990/2014"``); it defines the years that are requested
    from the CDS.

    The ESMValTool ``native6`` fixes shift the time stamps of accumulated ERA5
    fields (e.g. ``pr``) back by 30 minutes so that they label the middle of
    the accumulation interval. Their last day therefore misses the step that
    the following year contributes and is aggregated from 23 instead of 24
    hourly steps; request one additional year if that endpoint matters.

    """

    def download(self, variable: Variable, data_root_dir: Path | str) -> list[Path]:
        """Download data.

        One request is sent to the CDS per year of the variable's timerange;
        the downloaded hourly data are aggregated to daily data by
        :meth:`format`.

        Note
        ----
        This requires a valid API key in ``~/.cdsapirc``. See section "Setup
        the CDS Api key" at https://cds.climate.copernicus.eu/how-to-api for
        details.

        """
        data_root_dir = Path(data_root_dir)

        source_var_name = self._get_source_var_name(variable)
        if source_var_name not in VARIABLE_MAPPING:
            msg = f"Variable '{variable.var_name}' is not supported for daily ERA5"
            raise NotImplementedError(msg)

        # Raw downloaded data will be stored in temporary directory. Our "raw"
        # data will be the formatted data by ESMValTool.
        tmp_dir = Path(tempfile.TemporaryDirectory().name)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Download data (one request per year)
        (dataset, request) = self._get_cds_client_retrieve_kwargs(variable)
        years = request.pop("year")
        request["variable"] = [VARIABLE_MAPPING[source_var_name]]
        request["data_format"] = "netcdf"
        request["download_format"] = "unarchived"

        client = cdsapi.Client()
        tmp_targets: list[Path] = []
        for year in years:
            tmp_target = tmp_dir / f"raw_{self.id}_{variable.var_name}_{year}.nc"
            client.retrieve(dataset, {**request, "year": [year]}).download(tmp_target)
            tmp_targets.append(tmp_target)

        # Aggregate to daily data, reformat with ESMValTool and save
        target = self.format(tmp_targets, variable, data_root_dir=data_root_dir)

        return [target]

    def format(
        self,
        input_file: Path | str | Sequence[Path | str],
        variable: Variable,
        *,
        data_root_dir: Path | str,
    ) -> Path:
        """Aggregate hourly ERA5 data to daily data and format it.

        Parameters
        ----------
        input_file:
            Raw hourly ERA5 file, or the raw hourly files of all requested
            years (they are concatenated before the aggregation so that no
            day is split across two files).
        variable:
            Daily variable to produce.
        data_root_dir:
            Root directory the formatted file is written to.

        Returns
        -------
        :
            Path of the formatted daily file.

        """
        data_root_dir = Path(data_root_dir)

        # The hourly data is CMORized with the hourly ESMValTool fixes; these
        # cannot be applied to the aggregated data, because ESMValCore does not
        # implement the unit fix of accumulated ERA5 fields for daily data (it
        # has already been applied to the hourly data here).
        hourly_facets = self._get_hourly_esmvaltool_facets(variable)
        cubes = load_cubes(self._as_uris(input_file))
        with setup_esmvaltool_config_and_logging():
            cube = fix_metadata(cubes, **hourly_facets)[0]
            cube = fix_data(cube, **hourly_facets)
            if cube.coords("expver"):
                cube.remove_coord("expver")
            operator = DAILY_OPERATORS.get(variable.var_name, "mean")
            cube = daily_statistics(cube, operator)

            # Drop the partial day the 30 minute shift of accumulated fields
            # adds in front of the requested timerange
            cube = clip_timerange(cube, variable.timerange)

        self._set_daily_metadata(cube, variable, operator)

        era5_facets = self._get_esmvaltool_facets(variable)
        data_dir = self._get_data_dir(data_root_dir, variable)
        data_dir.mkdir(parents=True, exist_ok=True)
        target = data_dir / self._get_output_filename(cube, **era5_facets)
        iris.save(cube, target)

        return target

    def _get_cds_client_retrieve_kwargs(
        self,
        variable: Variable,
    ) -> tuple[str, dict[str, Any]]:
        """Get arguments for :meth:`cds.Client.retrieve`.

        The returned ``year`` entry lists all years of the variable's
        timerange; :meth:`download` sends one request per year.

        """
        if variable.ndim != 3:  # noqa: PLR2004
            msg = (
                f"{variable.ndim}D data (variable '{variable.var_name}') is not "
                f"supported for daily ERA5"
            )
            raise ValueError(msg)

        request: dict[str, Any] = {
            "product_type": "reanalysis",
            "year": [str(y) for y in self._get_years(variable)],
            "month": ALL_MONTHS,
            "day": ALL_DAYS,
            "time": ALL_TIMES,
        }

        return ("reanalysis-era5-single-levels", request)

    def _get_esmvaltool_facets(self, variable: Variable) -> dict[str, Any]:
        """Get ESMValTool facets of the aggregated daily data."""
        return {
            "short_name": variable.var_name,
            "project": "native6",
            "dataset": "ERA5",
            "mip": "day",  # so far, all supported variables are atmospheric
            "frequency": "day",
        }

    def _get_hourly_esmvaltool_facets(self, variable: Variable) -> dict[str, Any]:
        """Get ESMValTool facets of the raw hourly data."""
        return {
            "short_name": self._get_source_var_name(variable),
            "project": "native6",
            "dataset": "ERA5",
            "mip": "3hr",  # so far, all supported variables are atmospheric
            "frequency": "1hr",
        }

    @staticmethod
    def _as_uris(input_file: Path | str | Sequence[Path | str]) -> list[str]:
        """Get a list of URIs from one or more input files."""
        if isinstance(input_file, (Path, str)):
            input_file = [input_file]
        return [str(f) for f in input_file]

    @staticmethod
    def _get_source_var_name(variable: Variable) -> str:
        """Get the hourly ERA5 variable a daily variable is aggregated from."""
        return DAILY_SOURCE_VARIABLES.get(variable.var_name, variable.var_name)

    @staticmethod
    def _get_years(variable: Variable) -> list[int]:
        """Get the years of a variable's timerange."""
        start_end = variable.timerange.split("/")
        expected_n_bounds = 2
        if len(start_end) != expected_n_bounds or not all(
            s[:4].isdigit() for s in start_end
        ):
            msg = (
                f"Daily ERA5 data is aggregated from the hourly reanalysis, which "
                f"is downloaded one year at a time, so {variable} needs an "
                f"explicit timerange of the form '<start>/<end>' (e.g. "
                f"'1990/2014' or '19900101/20141231'), got '{variable.timerange}'"
            )
            raise ValueError(msg)

        (start_year, end_year) = (int(s[:4]) for s in start_end)
        if end_year < start_year:
            msg = (
                f"Start year must not be later than end year, got timerange "
                f"'{variable.timerange}' for {variable}"
            )
            raise ValueError(msg)

        return list(range(start_year, end_year + 1))

    @staticmethod
    def _set_daily_metadata(cube: Cube, variable: Variable, operator: str) -> None:
        """Set the metadata of the aggregated daily variable.

        Aggregating does not change the units, which have already been
        CMORized by the hourly ESMValTool fixes, so only the variable names and
        the cell method need to be updated here.
        """
        cube.standard_name = variable.standard_name or None
        cube.long_name = variable.long_name
        cube.var_name = variable.var_name
        cube.cell_methods = (CellMethod(_CELL_METHODS[operator], coords="time"),)
