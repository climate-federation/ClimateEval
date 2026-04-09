"""Manage data sources."""

from __future__ import annotations

import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any

import intake_esgf
import iris
from esmvalcore.config import CFG
from esmvaltool.cmorizers.data.cmorizer import DataCommand
from loguru import logger

from benchmarking import _get_prepared_cube

if TYPE_CHECKING:
    from iris.cube import CubeList

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
        paths: list[Path] = list(self._get_data_dir(variable).glob("*.nc"))
        if not paths:
            logger.info(f"Downloading {variable} for {self.name}")
            paths = self._download(variable)
            logger.info(f"Downloaded {variable} for {self.name}:\n{pformat(paths)}")
        cubes = iris.load(paths)
        return _get_prepared_cube(cubes, variable)

    @abstractmethod
    def _download(self, variable: Variable) -> list[Path]:
        """Download data."""

    def _esmvaltool_download(
        self,
        variable: Variable,
        dataset_name: str,
        file_prefix_pattern: str = "*_",
        tier: int = 2,
    ) -> list[Path]:
        """Use ESMValTool to download data."""
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
            f"{file_prefix_pattern}{variable.var_name}_*.nc",
        )
        data_dir = self._get_data_dir(variable)
        data_dir.mkdir(parents=True, exist_ok=True)
        for file in raw_files:
            shutil.move(file, data_dir / file.name)
        return list(data_dir.glob("*.nc"))

    def _intake_esgf_download(self, project: str, **facets: Any) -> list[Path]:
        """Use intake-esgf to download data."""
        with intake_esgf.conf.set(
            all_indices=True,
            esg_dataroot=[],
            local_cache=self.data_dir,
        ):
            cat = intake_esgf.ESGFCatalog()
            cat.project = intake_esgf.projects.projects[project.lower()]
            cat.search(**facets)
            paths = cat.to_path_dict()
        return list(paths.values())

    def _get_data_dir(self, variable: Variable) -> Path:
        """Get data directory."""
        return self.data_dir / self.name / variable.var_name
