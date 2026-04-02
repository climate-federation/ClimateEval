"""Manage datasets."""

from __future__ import annotations

import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import iris
from esmvalcore.config import CFG
from esmvaltool.cmorizers.data.cmorizer import DataCommand

from benchmarking import get_prepared_cube
from benchmarking._config import configure_esmvaltool
from benchmarking._variable import Variable

if TYPE_CHECKING:
    from iris.cube import Cube

    from benchmarking._config import Config
    from benchmarking._diagnostics import Diagnostic


class Data(ABC):
    """Manage data."""

    def __init__(self, variable: str | Variable) -> None:
        """Initialize class instance."""
        if isinstance(variable, str):
            self._variable = Variable(variable)
        else:
            self._variable = variable
        self._name = self.__class__.__name__

    @property
    def name(self) -> str:
        """Get dataset name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set dataset name."""
        self._name = value

    @property
    def variable(self) -> Variable:
        """Get variable."""
        return self._variable

    def get_cube(self, diagnostic: Diagnostic, config: Config) -> Cube:
        """Get data as single cube."""
        paths = list(self._get_data_dir(config).glob("*.nc"))
        if not paths:
            self._download(config)
        cubes = iris.load(paths)
        cube = get_prepared_cube(cubes, self.variable, config)
        return diagnostic.preprocess(cube)

    @abstractmethod
    def _download(self, config: Config) -> None:
        """Download data."""

    def _esmvaltool_download(
        self,
        dataset_name: str,
        config: Config,
        file_prefix_pattern: str = "*_",
        tier: int = 2,
    ) -> None:
        """Use ESMValTool to download data."""
        configure_esmvaltool(config)

        # Raw downloaded data by ESMValTool will be stored in temporary
        # directory. Our "raw" data will be the formatted data by ESMValTool.
        # Store all output ESMValTool in temporary directory and only save the
        # relevant files.
        tmp_dir = Path(tempfile.TemporaryDirectory().name)
        download_dir = tmp_dir / "download"
        esmvaltool_dir = tmp_dir / "esmvaltool"
        download_dir.mkdir(parents=True, exist_ok=True)
        esmvaltool_dir.mkdir(parents=True, exist_ok=True)

        # Download and format data
        CFG["output_dir"] = esmvaltool_dir
        DataCommand().prepare(dataset_name, original_data_dir=download_dir)

        # Move formatted data into raw directory
        latest_esmvaltool_output_dir = (
            sorted(esmvaltool_dir.glob("data_preparation_*"))[-1]
            / f"Tier{tier}"
            / dataset_name
        )
        raw_files = latest_esmvaltool_output_dir.glob(
            f"{file_prefix_pattern}{self.variable.var_name}_*.nc",
        )
        data_dir = self._get_data_dir(config)
        data_dir.mkdir(parents=True, exist_ok=True)
        for file in raw_files:
            shutil.move(file, data_dir / file.name)

    def _get_data_dir(self, config: Config) -> Path:
        """Get data directory."""
        return config.data_dir / self.name / self.variable.var_name
