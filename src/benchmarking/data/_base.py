"""Provide base class for data sources."""

from __future__ import annotations

import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

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


logger = logger.opt(colors=True)


@dataclass(frozen=True, kw_only=True)
class DiagnosticData:
    reference: type[DataSource]
    other: tuple[type[DataSource], ...] = ()


class DataSource(ABC):
    """Manage data source."""

    def __init__(self, data_dir: Path) -> None:
        """Initialize class instance."""
        self._data_dir = data_dir
        self._name = self.__class__.__name__

    @property
    def data_dir(self) -> Path:
        """Get path to data directory."""
        return self._data_dir

    @property
    def name(self) -> str:
        """Get dataset name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set dataset name."""
        self._name = value

    def get_cube(self, variable: Variable) -> CubeList:
        """Get data cube."""
        logger.debug(f"Loading {variable} of {self.name}")
        paths: list[Path] = list(self._get_data_dir(variable).glob("*.nc"))
        if not paths:
            logger.debug(f"No data found for {variable}, trying required variables")
            for required_variable in variable.get_required_variables():
                paths_for_required_variable = list(
                    self._get_data_dir(required_variable).glob("*.nc"),
                )
                if not paths_for_required_variable:
                    logger.debug(f"No data found for {required_variable}")
                    logger.debug(
                        f"Trying downloading {required_variable} for {self.name}",
                    )
                    paths_for_required_variable = self._download(required_variable)
                    logger.info(
                        f"Downloaded {required_variable} for {self.name}:\n"
                        f"{pformat(paths_for_required_variable)}",
                    )
                paths.extend(paths_for_required_variable)
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
            msg = f"{self.name} does not provide variable '{variable.var_name}'"
            raise ValueError(msg)

    @abstractmethod
    def _download(self, variable: Variable) -> list[Path]:
        """Download data."""

    def _esmvaltool_cmorizer_format(
        self,
        variable: Variable,
        dataset_name: str,
        *,
        download_dir: Path,
        file_prefix_pattern: str = "*_",
        tier: int = 2,
    ) -> list[Path]:
        """Use ESMValTool CMORizer to format data.

        See
        https://docs.esmvaltool.org/en/latest/input.html#using-a-cmorizer-script.

        """
        # Store all ESMValTool output in temporary directory and only save the
        # relevant files.
        tmp_dir = Path(tempfile.TemporaryDirectory().name)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        with CFG.context(output_dir=tmp_dir):
            DataCommand().format(dataset_name, original_data_dir=download_dir)

        latest_esmvaltool_output_dir = (
            sorted(tmp_dir.glob("data_formatting_*"))[-1] / f"Tier{tier}" / dataset_name
        )
        raw_files = latest_esmvaltool_output_dir.glob(
            f"{file_prefix_pattern}{variable.var_name}_*.nc",
        )
        data_dir = self._get_data_dir(variable)
        data_dir.mkdir(parents=True, exist_ok=True)
        for file in raw_files:
            shutil.move(file, data_dir / file.name)
        return list(data_dir.glob("*.nc"))

    def _esmvaltool_cmorizer_prepare(
        self,
        variable: Variable,
        dataset_name: str,
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

        # Download and format data
        with CFG.context(output_dir=tmp_dir):
            DataCommand().download(dataset_name, original_data_dir=tmp_dir)
        return self._esmvaltool_cmorizer_format(
            variable,
            dataset_name,
            download_dir=tmp_dir,
            file_prefix_pattern=file_prefix_pattern,
            tier=tier,
        )

    def _esmvaltool_intake_download(
        self,
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
        data_dir = self._get_data_dir(variable)
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / self._get_output_filename(cube, **facets)
        iris.save(cube, output_path)
        logger.debug(f"Saved {output_path}")
        return [output_path]

    def _get_data_dir(self, variable: Variable) -> Path:
        """Get data directory."""
        return self.data_dir / self.name / variable.var_name

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
