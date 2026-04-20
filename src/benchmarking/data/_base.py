"""Provide base class for data sources."""

from __future__ import annotations

import shutil
import tempfile
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any, ClassVar

import iris
from esmvalcore.config import CFG
from esmvalcore.dataset import Dataset
from esmvaltool.cmorizers.data.cmorizer import DataCommand
from loguru import logger

from benchmarking._config import setup_esmvaltool_intake_data_source
from benchmarking._utils import get_prepared_cube

if TYPE_CHECKING:
    from collections.abc import Iterable

    from iris.cube import Cube, CubeList

    from benchmarking import Variable
    from benchmarking.data._utils import DataSourceInformation


logger = logger.opt(colors=True)


class DataSource(ABC):
    """Manage data source."""

    _INFORMATION: ClassVar[DataSourceInformation]

    @property
    def id(self) -> str:
        """Get data source ID."""
        return self.information.id

    @property
    def information(self) -> DataSourceInformation:
        """Get data source information."""
        return self._INFORMATION

    @property
    def name(self) -> str:
        """Get data source name."""
        return self.information.name

    @abstractmethod
    def download(self, data_root_dir: Path, variable: Variable) -> list[Path]:
        """Download data."""

    def get_cube(self, data_root_dir: Path, variable: Variable) -> CubeList:
        """Get data cube."""
        logger.debug(f"Loading {variable} of {self.id}")
        paths: list[Path] = list(
            self._get_data_dir(data_root_dir, variable.var_name).glob("*.nc"),
        )
        if paths:
            return get_prepared_cube(iris.load(paths), variable)

        logger.debug(f"No data found for {variable} of {self.id}")

        # Check if variable can be derived from existing data
        if variable.derived:
            paths = self._get_paths_of_required_variables(data_root_dir, variable)
            if paths:
                logger.debug(
                    f"Found required variables to derive {variable} of {self.id}",
                )
                return get_prepared_cube(iris.load(paths), variable)

        # Try downloading the variable
        logger.debug(f"Trying downloading {variable} of {self.id}")
        error_msg = ""
        try:
            paths = self.download(data_root_dir, variable)
        except Exception as exc:
            if not variable.derived:
                msg = f"Downloading {variable} of {self.id} failed: {exc}"
                raise ValueError(msg) from exc
            error_msg = f": {exc}"
        if paths:
            logger.info(f"Downloaded {self.id}:\n{pformat(paths)}")
            return get_prepared_cube(iris.load(paths), variable)

        logger.debug(
            f"Failed to download derived variable {variable} of {self.id}{error_msg}",
        )

        # Try downloading required variables
        get_required_variables = variable.get_required_variables()
        logger.debug(f"Trying required variables {get_required_variables}")
        paths = self._get_paths_of_required_variables(
            data_root_dir,
            variable,
            download=True,
        )
        cubes = iris.load(paths)
        return get_prepared_cube(cubes, variable)

    def _check_supported_variables(
        self,
        variable: Variable,
        supported_var_names: Iterable[str] | None,
    ) -> None:
        """Raise an error if a variable is not supported."""
        if (
            supported_var_names is not None
            and variable.var_name not in supported_var_names
        ):
            msg = f"{self.id} does not provide variable '{variable.var_name}'"
            raise ValueError(msg)

    def _esmvaltool_cmorizer_format(
        self,
        data_root_dir: Path,
        variable: Variable,
        *,
        download_dir: Path,
        file_prefix_pattern: str = "*",
        tier: int = 2,
    ) -> list[Path]:
        """Use ESMValTool CMORizer to format data.

        See
        https://docs.esmvaltool.org/en/latest/input.html#using-a-cmorizer-script.

        """
        logger.debug(f"Formatting {self.id} data (this may take while...)")

        # Store all ESMValTool output in temporary directory and only save the
        # relevant files.
        tmp_dir = Path(tempfile.TemporaryDirectory().name)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with CFG.context(output_dir=tmp_dir):
            DataCommand().format(self.name, original_data_dir=download_dir)

        latest_esmvaltool_output_dir = (
            sorted(tmp_dir.glob("data_formatting_*"))[-1] / f"Tier{tier}" / self.name
        )
        raw_files = latest_esmvaltool_output_dir.glob(
            f"{file_prefix_pattern}.nc",
        )

        # Save all downloaded variables (there might be more than the one
        # requested; this will potentially save lots of time)
        grouped_files = defaultdict(list)
        formatted_files: list[Path] = []
        for raw_file in raw_files:
            var_name = raw_file.stem.split("_")[-2]
            grouped_files[var_name].append(raw_file)
        for var_name, var_files in grouped_files.items():
            data_dir = self._get_data_dir(data_root_dir, var_name)

            # Do not overwrite existing other variables
            if var_name != variable.var_name and list(data_dir.glob("*.nc")):
                continue
            logger.debug(f"Downloaded and formatted '{var_name}' of {self.id}")
            data_dir.mkdir(parents=True, exist_ok=True)

            for var_file in var_files:
                formatted_file = data_dir / var_file.name
                shutil.move(var_file, formatted_file)
                formatted_files.append(formatted_file)

        return formatted_files

    def _esmvaltool_cmorizer_prepare(
        self,
        data_root_dir: Path,
        variable: Variable,
        *,
        supported_var_names: Iterable[str] | None = None,
        file_prefix_pattern: str = "*_",
        tier: int = 2,
    ) -> list[Path]:
        """Use ESMValTool CMORizer to prepare (download and format) data.

        See
        https://docs.esmvaltool.org/en/latest/input.html#using-a-cmorizer-script.

        """
        self._check_supported_variables(variable, supported_var_names)

        # Raw downloaded data by ESMValTool will be stored in temporary
        # directory. Our "raw" data will be the formatted data by ESMValTool.
        tmp_dir = Path(tempfile.TemporaryDirectory().name)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Download data
        with CFG.context(output_dir=tmp_dir):
            DataCommand().download(self.name, original_data_dir=tmp_dir)

        # Format data
        return self._esmvaltool_cmorizer_format(
            data_root_dir,
            variable,
            download_dir=tmp_dir,
            file_prefix_pattern=file_prefix_pattern,
            tier=tier,
        )

    def _esmvaltool_intake_download(
        self,
        data_root_dir: Path,
        variable: Variable,
        **facets: Any,
    ) -> list[Path]:
        """Use ESMValTool's intake-esgf data source to download data.

        See
        https://docs.esmvaltool.org/projects/ESMValCore/en/latest/quickstart/configure.html#data-sources

        """
        # Store downloaded data in temporary directory
        tmp_dir = Path(tempfile.TemporaryDirectory().name)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with setup_esmvaltool_intake_data_source(tmp_dir):
            cube = Dataset(**facets).load()

        # Save file into data directory
        data_dir = self._get_data_dir(data_root_dir, variable.var_name)
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / self._get_output_filename(cube, **facets)
        iris.save(cube, output_path)
        return [output_path]

    def _get_data_dir(self, data_root_dir: Path, var_name: str) -> Path:
        """Get data- and variable-specific subdirectory."""
        return data_root_dir / self.id / var_name

    def _get_output_filename(self, cube: Cube, **facets: Any) -> str:
        """Get output file name."""
        suffix = ".nc"
        if cube.coords("time"):
            time_coord = cube.coord("time")
            start_date = time_coord.units.num2date(time_coord.points[0]).strftime(
                "%Y%m%d%H%M%S",
            )
            end_date = time_coord.units.num2date(time_coord.points[-1]).strftime(
                "%Y%m%d%H%M%S",
            )
            suffix = f"_{start_date}-{end_date}{suffix}"
        prefix = "_".join(str(facet_value) for facet_value in facets.values())
        return f"{prefix}{suffix}"

    def _get_paths_of_required_variables(
        self,
        data_root_dir: Path,
        variable: Variable,
        *,
        download: bool = False,
    ) -> list[Path]:
        """Get paths of required variables."""
        paths: list[Path] = []
        for required_variable in variable.get_required_variables():
            paths_for_required_variable = list(
                self._get_data_dir(data_root_dir, required_variable.var_name).glob(
                    "*.nc",
                ),
            )
            if not paths_for_required_variable:
                # Do not return any paths if not all required variables are
                # available
                if not download:
                    return []

                # Download
                paths_for_required_variable = self.download(
                    data_root_dir,
                    required_variable,
                )
                logger.info(
                    f"Downloaded {self.id}:\n{pformat(paths_for_required_variable)}",
                )

            paths.extend(paths_for_required_variable)
        return paths
