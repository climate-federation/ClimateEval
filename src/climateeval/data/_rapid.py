"""Provide RAPID AMOC data.

The RAPID-MOCHA-WBTS array measures the Atlantic Meridional Overturning
Circulation at 26.5N. ESMValTool does not ship a CMORizer for it, but a
CMORized ``amoc`` file is distributed as an OBS6 Tier 3 dataset. This module
loads that file directly with Iris (the ``amoc`` variable is provided directly,
so it is not derived).

Staging the data
----------------
Point ClimateEval at the CMORized RAPID ``amoc`` file with the
``CLIMATEEVAL_RAPID_FILE`` environment variable. It defaults to the file staged
on Levante at the ESMValTool OBS6 Tier 3 location.
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

    #: Variables provided directly (not derived) by the CMORized RAPID file.
<<<<<<< HEAD
    _supported_var_names: ClassVar[tuple[str, ...]] = ("amoc", "msftmz",)

    _FILE_ENV: ClassVar[str] = "CLIMATEEVAL_RAPID_FILE"
    _DEFAULT_FILE: ClassVar[dict[str, str]] = {
        "amoc": (
            "/work/bd0854/DATA/ESMValTool2/OBS/Tier3/RAPID/"
            "OBS6_RAPID_exp_2024.1a_Omon_amoc_2004_2024.nc"
        ),
        "msftmz": ("/work/bd1083/b382555/OBS_RAPID_Omon_msftmz_2004_2024.nc"),
    }
=======
    _supported_var_names: ClassVar[tuple[str, ...]] = ("amoc",)

    _FILE_ENV: ClassVar[str] = "CLIMATEEVAL_RAPID_FILE"
    _DEFAULT_FILE: ClassVar[str] = (
        "/work/bd0854/DATA/ESMValTool2/OBS/Tier3/RAPID/"
        "OBS6_RAPID_exp_2024.1a_Omon_amoc_2004_2024.nc"
    )
>>>>>>> origin

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

<<<<<<< HEAD
        default_path = self._DEFAULT_FILE.get(variable.var_name)
        file = Path(os.environ.get(self._FILE_ENV, default_path))
=======
        file = Path(os.environ.get(self._FILE_ENV, self._DEFAULT_FILE))
>>>>>>> origin
        if not file.is_file():
            msg = f"RAPID file {file} not found for {variable} of {self.id}"
            raise MissingDataError(msg)

        with setup_esmvaltool_config_and_logging():
            # The RAPID file already contains the CMORized `amoc` variable, so
            # get_prepared_cube extracts it directly rather than deriving it.
            cube = load_cubes([str(file)]).extract_cube(
                NameConstraint(var_name=variable.var_name),
            )
            return get_prepared_cube(CubeList([cube]), variable)

    def download(self, variable: Variable, data_root_dir: Path | str) -> list[Path]:  # noqa: ARG002
        """Download data.

        RAPID cannot be downloaded automatically; stage the CMORized ``amoc``
        file manually and point ClimateEval at it (see the module docs).
        """
        msg = (
            f"{self.id} cannot be downloaded automatically; stage the CMORized "
            f"RAPID amoc file and set {self._FILE_ENV} (see climateeval.data._rapid)"
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
