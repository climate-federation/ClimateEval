"""Provide RAPID AMOC data.

The RAPID-MOCHA-WBTS array measures the Atlantic Meridional Overturning
Circulation at 26.5N. ESMValTool does not ship a CMORizer for it, but a
CMORized ``amoc`` file is distributed as an OBS6 Tier 3 dataset; locally
CMORized files of the overturning streamfunction (``msftmz``) and of the
northward ocean heat transport (``hfbasin``) at the array are available on
Levante as well. This module loads those files directly with Iris (the
variables are provided directly, so they are not derived).

Staging the data
----------------
Point ClimateEval at the CMORized RAPID file of each variable with the
corresponding environment variable:

- ``CLIMATEEVAL_RAPID_FILE``: the ``amoc`` file,
- ``CLIMATEEVAL_RAPID_MSFTMZ_FILE``: the ``msftmz`` file,
- ``CLIMATEEVAL_RAPID_HFBASIN_FILE``: the ``hfbasin`` file.

They default to the files staged on Levante.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from iris import NameConstraint
from iris.cube import CubeList
from loguru import logger

from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._utils import get_prepared_cube, load_cubes
from climateeval.data._base import DataSource
from climateeval.data._utils import DataSourceInformation
from climateeval.exceptions import DownloadError, MissingDataError

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable

logger = logger.opt(colors=True)


class RAPID(DataSource):
    """Manage RAPID AMOC data."""

    _information = DataSourceInformation(
        name="RAPID",
        category="observation",
        institute="RAPID-MOCHA-WBTS",
        exp="exp",
        variant="2024.1a",
        references=("https://doi.org/10.5285/223b34a3-2dc5-3fdc-e063-6c86abc0f730",),
    )

    #: Environment variable pointing at the CMORized RAPID file per variable.
    _FILE_ENVS: ClassVar[dict[str, str]] = {
        "amoc": "CLIMATEEVAL_RAPID_FILE",
        "msftmz": "CLIMATEEVAL_RAPID_MSFTMZ_FILE",
        "hfbasin": "CLIMATEEVAL_RAPID_HFBASIN_FILE",
    }
    #: Default (Levante) location of the CMORized RAPID file per variable.
    _DEFAULT_FILES: ClassVar[dict[str, str]] = {
        "amoc": (
            "/work/bd0854/DATA/ESMValTool2/OBS/Tier3/RAPID/"
            "OBS6_RAPID_exp_2024.1a_Omon_amoc_2004_2024.nc"
        ),
        "msftmz": "/work/bd1083/b382555/OBS_RAPID_Omon_msftmz_2004_2024.nc",
        "hfbasin": "/work/bd1083/b382555/OBS_RAPID_Omon_hfbasin_2004_2024.nc",
    }

    #: Variables provided directly (not derived) by the CMORized RAPID files.
    _supported_var_names: ClassVar[tuple[str, ...]] = tuple(_DEFAULT_FILES)

    def get_cube(
        self,
        data_root_dir: Path | str,  # noqa: ARG002
        variable: Variable,
        *,
        download_missing_data: bool = True,  # noqa: ARG002
    ) -> Cube:
        """Get data cube from the CMORized RAPID file."""
        logger.debug(f"Loading {variable} of {self.id}")
        self._check_supported_variables(variable)

        file_env = self._FILE_ENVS[variable.var_name]
        file = Path(os.environ.get(file_env, self._DEFAULT_FILES[variable.var_name]))
        if not file.is_file():
            msg = f"RAPID file {file} not found for {variable} of {self.id}"
            raise MissingDataError(msg)

        with setup_esmvaltool_config_and_logging():
            # The RAPID files already contain the CMORized variables, so
            # get_prepared_cube extracts them directly rather than deriving them.
            cube = load_cubes([str(file)]).extract_cube(
                NameConstraint(var_name=variable.var_name),
            )
            return get_prepared_cube(CubeList([cube]), variable)

    def download(self, variable: Variable, data_root_dir: Path | str) -> list[Path]:  # noqa: ARG002
        """Download data.

        RAPID cannot be downloaded automatically; stage the CMORized files
        manually and point ClimateEval at them (see the module docs).
        """
        file_env = self._FILE_ENVS.get(variable.var_name, "CLIMATEEVAL_RAPID_*_FILE")
        msg = (
            f"{self.id} cannot be downloaded automatically; stage the CMORized "
            f"RAPID {variable.var_name} file and set {file_env} (see "
            f"climateeval.data._rapid)"
        )
        raise DownloadError(msg)

    def _check_supported_variables(self, variable: Variable) -> None:
        """Raise an error if a variable is not supported."""
        if variable.var_name not in self._supported_var_names:
            msg = (
                f"{self.id} does not provide variable '{variable.var_name}' "
                f"(supported: {sorted(self._supported_var_names)})"
            )
            raise NotImplementedError(msg)
