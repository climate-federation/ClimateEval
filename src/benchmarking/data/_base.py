"""Manage datasets."""

from __future__ import annotations

import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import intake_esgf
import iris
from esmvalcore.config import CFG
from esmvaltool.cmorizers.data.cmorizer import DataCommand

from benchmarking import Variable
from benchmarking._config import reset_esmvaltool_config

if TYPE_CHECKING:
    from iris.cube import Cube

    from benchmarking._config import Config


class DataSource(ABC):
    """Manage data source."""

    def __init__(self, variable: str | Variable, config: Config) -> None:
        """Initialize class instance."""
        if isinstance(variable, str):
            self._variable = Variable(variable)
        else:
            self._variable = variable
        self._config = config
        self._name = self.__class__.__name__

    @property
    def config(self) -> Config:
        """Get configuration."""
        return self._config

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

    def get_cubes(self) -> Cube:
        """Get raw data as cubes."""
        paths = list(self._get_data_dir().glob("*.nc"))
        if not paths:
            self._download()
        return iris.load(paths)

    @abstractmethod
    def _download(self) -> None:
        """Download data."""

    def _esmvaltool_download(
        self,
        dataset_name: str,
        file_prefix_pattern: str = "*_",
        tier: int = 2,
    ) -> None:
        """Use ESMValTool to download data."""
        reset_esmvaltool_config()

        # Raw downloaded data by ESMValTool will be stored in temporary
        # directory. Our "raw" data will be the formatted data by ESMValTool.
        # Store all ESMValTool output in temporary directory and only save the
        # relevant files.
        tmp_dir = Path(tempfile.TemporaryDirectory().name)
        download_dir = tmp_dir / "download"
        esmvaltool_dir = tmp_dir / "esmvaltool"
        download_dir.mkdir(parents=True, exist_ok=True)
        esmvaltool_dir.mkdir(parents=True, exist_ok=True)

        # Download and format data
        with CFG.context(output_dir=esmvaltool_dir):
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
        data_dir = self._get_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        for file in raw_files:
            shutil.move(file, data_dir / file.name)

    def _intake_esgf_download(self, **facets: Any) -> None:
        """Use intake-esgf to download data."""
        cat = intake_esgf.ESGFCatalog()
        with intake_esgf.conf.set(local_cache=self.config.data_dir):
            cat.search(**facets)
            paths = cat.to_path_dict()
        print(paths)
        assert 0

    def _get_data_dir(self) -> Path:
        """Get data directory."""
        return self.config.data_dir / self.name / self.variable.var_name
