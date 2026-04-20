"""Provide basic features of data sources."""

from __future__ import annotations

import contextlib
import tempfile
from copy import deepcopy
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import intake_esgf
import iris
import yaml
from esmvalcore.config import CFG
from esmvalcore.dataset import Dataset
from loguru import logger

from benchmarking import setup_esmvaltool_config_and_logging
from benchmarking.data._base import DataSource

if TYPE_CHECKING:
    from collections.abc import Generator

    from benchmarking import Variable


logger = logger.opt(colors=True)


ESMVALTOOL_INTAKE_DATA_SOURCES = (
    Path(str(resources.files("benchmarking.data")))
    / "config"
    / "esmvaltool_intake_data_sources.yml"
)


def _get_esmvaltool_intake_data_sources() -> dict[str, Any]:
    """Get ESMValTool intake-esgf data sources."""
    with ESMVALTOOL_INTAKE_DATA_SOURCES.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@contextlib.contextmanager
def _setup_esmvaltool_intake_data_source(cache_dir: Path) -> Generator[None]:
    """Set up ESMValTool intake-esgf data sources."""
    intake_esgf_options = {
        "all_indices": True,
        "esg_dataroot": [],
        "local_cache": cache_dir,
    }

    with contextlib.ExitStack() as stack:
        stack.enter_context(intake_esgf.conf.set(**intake_esgf_options))
        stack.enter_context(setup_esmvaltool_config_and_logging())
        esmvaltool_config = deepcopy(CFG)
        for project, data_config in _get_esmvaltool_intake_data_sources().items():
            esmvaltool_config["projects"][project]["data"] = data_config["data"]
        stack.enter_context(CFG.context(esmvaltool_config))
        yield


class ESMValToolIntakeDataSource(DataSource):
    """Manage ESMValTool Intake-ESGF data source."""

    _ESMVALTOOL_DATASET_FACETS: ClassVar[dict[str, Any]] = {}
    """Facets used to download data."""

    def download(self, data_root_dir: Path, variable: Variable) -> list[Path]:
        """Download data.

        See
        https://docs.esmvaltool.org/projects/ESMValCore/en/latest/quickstart/configure.html#data-sources.

        """
        facets = {
            "dataset": self.name,
            "mip": variable.cmip6_table_id,
            "short_name": variable.var_name,
            "frequency": variable.frequency,
        }
        facets.update(self._ESMVALTOOL_DATASET_FACETS)

        # Store downloaded data in temporary directory
        tmp_dir = Path(tempfile.TemporaryDirectory().name)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with _setup_esmvaltool_intake_data_source(tmp_dir):
            cube = Dataset(**facets).load()

        # Save file into data directory
        data_dir = self._get_data_dir(data_root_dir, variable.var_name)
        data_dir.mkdir(parents=True, exist_ok=True)
        output_path = data_dir / self._get_output_filename(cube, **facets)
        iris.save(cube, output_path)

        return [output_path]
