"""Provide data sources loaded via ESMValCore's native6 on-the-fly CMORizer.

Unlike :class:`~climateeval.data.ESMValToolCMORizerDataSource` (which writes
CMORized files to disk with an ESMValTool CMORizer script), these data sources
hand the raw files to ESMValCore's :class:`esmvalcore.dataset.Dataset`, which
applies the ``native6`` on-the-fly fixes (variable renaming, unit conversion,
coordinate/grid fixes) while loading. The result is a CMOR-compliant cube that
is then passed through ClimateEval's usual preprocessing.

"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from esmvalcore.dataset import Dataset
from esmvalcore.local import LocalFile
from iris.cube import CubeList
from loguru import logger

from climateeval._config import (
    setup_esmvaltool_config_and_logging,
    setup_iris_combine_policy,
)
from climateeval._utils import get_prepared_cube
from climateeval.data._base import DataSource
from climateeval.exceptions import DownloadError, MissingDataError

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable

logger = logger.opt(colors=True)


class ESMValCoreNative6DataSource(DataSource):
    """Manage data loaded via ESMValCore's ``native6`` on-the-fly CMORizer."""

    _supported_var_names: ClassVar[tuple[str, ...] | None] = None
    """Supported variable names."""

    _dataset_facets: ClassVar[dict[str, Any]] = {}
    """Extra facets passed to :class:`esmvalcore.dataset.Dataset` (e.g. version)."""

    def get_cube(
        self,
        data_root_dir: Path | str,
        variable: Variable,
        *,
        download_missing_data: bool = True,  # noqa: ARG002
    ) -> Cube:
        """Get data cube via ESMValCore's native6 CMORizer."""
        logger.debug(f"Loading {variable} of {self.id}")
        self._check_supported_variables(variable)

        with setup_esmvaltool_config_and_logging(), setup_iris_combine_policy():
            dataset = self._build_dataset(variable, Path(data_root_dir))
            cube = dataset.load()

        return get_prepared_cube(CubeList([cube]), variable)

    def download(self, variable: Variable, data_root_dir: Path | str) -> list[Path]:  # noqa: ARG002
        """Download data.

        Native6 data sources cannot be downloaded automatically; the raw files
        have to be staged manually (see the concrete data source for details).
        """
        msg = (
            f"{self.id} cannot be downloaded automatically; stage the raw files "
            f"manually (see the {type(self).__name__} documentation)"
        )
        raise DownloadError(msg)

    def _build_dataset(self, variable: Variable, data_root_dir: Path) -> Dataset:
        """Build an ESMValCore dataset for the requested variable."""
        facets: dict[str, Any] = {
            "project": "native6",
            "dataset": self.name,
            "mip": variable.cmip6_table_id or "Omon",
            "short_name": variable.var_name,
            "frequency": variable.frequency,
            **self._dataset_facets,
        }
        grid_path = self._get_grid_path(variable)
        if grid_path is not None:
            facets["horizontal_grid"] = str(grid_path)

        dataset = Dataset(**facets)
        # Resolve raw variable name, units, tier, ... from ESMValCore's
        # native6 extra facets.
        dataset.augment_facets()

        raw_name = dataset.facets.get("raw_name", variable.var_name)
        files = self._get_raw_files(variable, data_root_dir, str(raw_name))
        if not files:
            msg = f"No raw files found for {variable} of {self.id}"
            raise MissingDataError(msg)
        dataset.files = [LocalFile(file) for file in files]

        return dataset

    def _check_supported_variables(self, variable: Variable) -> None:
        """Raise an error if a variable is not supported."""
        if (
            self._supported_var_names is not None
            and variable.var_name not in self._supported_var_names
        ):
            msg = f"{self.id} does not provide variable '{variable.var_name}'"
            raise NotImplementedError(msg)

    def _get_grid_path(self, variable: Variable) -> Path | None:  # noqa: ARG002
        """Get path to the horizontal grid file (if the data source needs one)."""
        return None

    def _get_raw_files(
        self,
        variable: Variable,
        data_root_dir: Path,
        raw_name: str,
    ) -> list[Path]:
        """Get the raw (non-CMORized) input files for a variable."""
        raise NotImplementedError
