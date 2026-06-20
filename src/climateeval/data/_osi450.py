"""Provide OSI SAF OSI-450 sea-ice concentration data.

OSI-450 is the EUMETSAT OSI SAF global sea-ice concentration climate data
record, provided separately for the Northern (``OSI-450-nh``) and Southern
(``OSI-450-sh``) hemispheres. Both are CMORized via ESMValTool.

The CMORized OSI-450 files use the legacy variable name ``sic`` (rather than the
CMIP6 name ``siconc``), so the loaded cube is renamed to ``siconc``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from climateeval._utils import get_prepared_cube, load_cubes
from climateeval._variable import VARIABLES
from climateeval.data._esmvaltool_cmorizer import ESMValToolCMORizerDataSource
from climateeval.data._utils import DataSourceInformation

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable


class _OSI450(ESMValToolCMORizerDataSource):
    """Manage OSI-450 sea-ice concentration data (one hemisphere)."""

    _supported_var_names = ("siconc",)

    #: Legacy variable name used in the CMORized OSI-450 files.
    _raw_var_name: ClassVar[str] = "sic"

    def get_cube(
        self,
        data_root_dir: Path | str,
        variable: Variable,
        *,
        download_missing_data: bool = True,
    ) -> Cube:
        """Get data cube, renaming the legacy ``sic`` variable to ``siconc``."""
        data_root_dir = Path(data_root_dir)
        paths = list(self._get_data_dir(data_root_dir, variable).glob("*.nc"))
        if not paths:
            return super().get_cube(
                data_root_dir,
                variable,
                download_missing_data=download_missing_data,
            )
        cubes = load_cubes(paths)
        for cube in cubes:
            if cube.var_name == self._raw_var_name:
                cube.var_name = variable.var_name
                cube.standard_name = (
                    VARIABLES[variable.var_name]["standard_name"] or None
                )
        return get_prepared_cube(cubes, variable)


class OSI450NH(_OSI450):
    """Manage OSI-450 Northern Hemisphere sea-ice concentration data."""

    _information = DataSourceInformation(
        name="OSI-450-nh",
        category="reanalysis",
        institute="EUMETSAT OSI SAF",
        references=("http://osisaf.met.no/p/ice/",),
    )


class OSI450SH(_OSI450):
    """Manage OSI-450 Southern Hemisphere sea-ice concentration data."""

    _information = DataSourceInformation(
        name="OSI-450-sh",
        category="reanalysis",
        institute="EUMETSAT OSI SAF",
        references=("http://osisaf.met.no/p/ice/",),
    )
