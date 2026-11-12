"""Provide observations that are staged locally as CMOR-like NetCDF files.

Some observational products have no ESMValTool CMORizer and no download path
that ClimateEval could maintain. They are prepared elsewhere (for AtmosphereBench:
by the scripts of its observation bundle) into CMOR-like NetCDF files under::

    <data_root>/<source id>/<frequency>/<var_name>/*.nc

where ``<source id>`` is the :attr:`~climateeval.data.DataSourceInformation.id`
of the data source (e.g. ``observation_GPCP-1DD``). :class:`LocalObservationDataSource`
reads exactly that layout: it globs the directory, keeps the files whose CMOR
date token (the trailing ``<start>-<end>`` of the file name) overlaps the
variable's ``timerange``, concatenates them and hands the result to
:func:`~climateeval._utils.get_prepared_cube`. Nothing is downloaded: if the
directory has no matching file, :class:`~climateeval.exceptions.MissingDataError`
is raised.

Subclasses only need to declare their provenance (``_information``), the
variables and frequencies they serve, and optionally the prep script that
provides the files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from loguru import logger

from climateeval._utils import get_prepared_cube, load_cubes
from climateeval.data._base import DataSource, _paths_within_timerange
from climateeval.exceptions import DownloadError, MissingDataError

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable

logger = logger.opt(colors=True)


class LocalObservationDataSource(DataSource):
    """Observations staged under ``<data_root>/<id>/<frequency>/<var_name>/``.

    Child classes define ``_information`` and :attr:`_supported_var_names` and
    :attr:`_supported_frequencies`; :attr:`_prep_script` names the script that
    produces the files, for the error message of missing data.
    """

    _supported_var_names: ClassVar[tuple[str, ...]]
    """Variables this product provides."""

    _supported_frequencies: ClassVar[tuple[str, ...]]
    """Frequencies this product provides."""

    _prep_script: ClassVar[str] = ""
    """Script of the AtmosphereBench observation bundle that prepares the data."""

    def get_cube(
        self,
        data_root_dir: Path | str,
        variable: Variable,
        *,
        download_missing_data: bool = True,  # noqa: ARG002
    ) -> Cube:
        """Get the staged data of a variable.

        Raises
        ------
        NotImplementedError
            The product does not provide the variable (or frequency).
        MissingDataError
            No staged file overlaps the variable's timerange.

        Note
        ----
        `download_missing_data` is ignored: nothing is ever downloaded.

        """
        self._check_supported_variables(variable)
        logger.debug(f"Loading {variable} of {self.id}")

        data_dir = self._get_data_dir(Path(data_root_dir), variable)
        paths = _paths_within_timerange(
            sorted(data_dir.glob("*.nc")),
            variable.timerange,
        )
        if not paths:
            raise MissingDataError(self._missing_data_message(variable, data_dir))
        return get_prepared_cube(load_cubes(paths), variable)

    def download(self, variable: Variable, data_root_dir: Path | str) -> list[Path]:
        """Never: the data is prepared by the AtmosphereBench observation bundle."""
        self._check_supported_variables(variable)
        msg = self._missing_data_message(
            variable,
            self._get_data_dir(Path(data_root_dir), variable),
        )
        raise DownloadError(msg)

    def _missing_data_message(self, variable: Variable, data_dir: Path) -> str:
        """Get the message for data that has not been staged."""
        script = (
            f"the prep script {self._prep_script}"
            if self._prep_script
            else "its prep script under obs/prep/"
        )
        return (
            f"No staged data of {variable} of {self.id} found in {data_dir}. "
            f"{self.id} is not downloaded by ClimateEval; it is provided by the "
            f"AtmosphereBench observation bundle (prepared by {script}). Stage "
            f"the bundle under the data root directory as "
            f"<data_root>/{self.id}/<frequency>/<var_name>/*.nc"
        )

    def _check_supported_variables(self, variable: Variable) -> None:
        """Raise if the variable, or its frequency, is not served."""
        if variable.var_name not in self._supported_var_names:
            msg = f"{self.id} does not provide variable '{variable.var_name}'"
            raise NotImplementedError(msg)
        if variable.frequency not in self._supported_frequencies:
            msg = (
                f"{self.id} does not provide variable '{variable.var_name}' with "
                f"frequency '{variable.frequency}' (supported frequencies: "
                f"{', '.join(self._supported_frequencies)})"
            )
            raise NotImplementedError(msg)
