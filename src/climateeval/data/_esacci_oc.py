"""Provide ESACCI-OC data (local override for Levante)."""

from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation

# Path to the pre-CMORized ESA-CCI Chl file on Levante.
# Use fv3.1 because it is 3-D (time, lat, lon) and does not require
# the lev=1 squeeze that the 4-D fv5.0 file would.
_LEVANTE_ESACCI_OC_FILE = Path(
    "/work/bd0854/DATA/ESMValTool2/OBS/Tier2/ESACCI-OC/"
    "OBS_ESACCI-OC_sat_fv3.1_Omon_chl_199709-201806.nc",
)


class ESACCIOC(ESMValToolCMORizerDataSource):
    """Manage ESACCI-OC chlorophyll-a data.

    Overrides the default ESMValTool cmorizer ``download`` method
    because the upstream CEDA FTP URL is dead. The pre-CMORized file
    is read directly from the DKRZ shared archive instead.
    """

    _INFORMATION = DataSourceInformation(
        name="ESACCI-OC",
        category="observation",
        institute="ESA",
        references=("https://climate.esa.int/en/projects/ocean-colour/",),
    )

    SUPPORTED_VAR_NAMES = ("chl",)
    _FILE_PREFIX_PATTERN = "OBS_ESACCI-OC_sat_fv3.1_Omon_chl_*"

    def download(self, variable, data_root_dir: Path) -> list[Path]:
        """Copy the pre-CMORized file from Levante instead of downloading."""
        self._check_supported_variables(variable)

        if not _LEVANTE_ESACCI_OC_FILE.is_file():
            msg = (
                f"Expected pre-CMORized ESACCI-OC file at "
                f"{_LEVANTE_ESACCI_OC_FILE}, but it is missing."
            )
            raise FileNotFoundError(msg)

        data_dir = self._get_data_dir(data_root_dir, variable.var_name)
        data_dir.mkdir(parents=True, exist_ok=True)
        target = data_dir / _LEVANTE_ESACCI_OC_FILE.name

        if not target.is_file():
            logger.debug(
                f"Copying pre-CMORized {self.id} from "
                f"{_LEVANTE_ESACCI_OC_FILE} to {target}",
            )
            shutil.copyfile(_LEVANTE_ESACCI_OC_FILE, target)
        else:
            logger.debug(f"Reusing existing {self.id} at {target}")

        return [target]
