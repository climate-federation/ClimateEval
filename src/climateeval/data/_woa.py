"""Provide WOA data."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation

if TYPE_CHECKING:
    from pathlib import Path

    from iris.cube import Cube

    from climateeval import Variable


class WOA(ESMValToolCMORizerDataSource):
    """Manage World Ocean Atlas (WOA) data."""

    _information = DataSourceInformation(
        name="WOA",
        category="observation",
        institute="NOAA NCEI",
        references=("https://doi.org/10.7289/V5/ATLAS-NESDIS-87",),
    )

    _supported_var_names = ("so", "thetao", "no3", "o2", "po4", "si")

    def get_cube(
        self,
        data_root_dir: Path | str,
        variable: Variable,
        *,
        download_missing_data: bool = True,
    ) -> Cube:
        """Get data cube.

        WOA is a long-term climatology (a single time field), so the model's
        time range must not be applied to it.
        """
        climatology_variable = replace(variable, timerange="*")
        return super().get_cube(
            data_root_dir,
            climatology_variable,
            download_missing_data=download_missing_data,
        )
