"""Provide CERES-EBAF data."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger

from climateeval._utils import get_prepared_cube, load_cubes
from climateeval.data._base import _paths_within_timerange
from climateeval.data._esmvaltool_intake import ESMValToolIntakeDataSource
from climateeval.data._utils import DataSourceInformation

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable

PUBLISHED_CRE: dict[str, str] = {"lwcre": "rltcre", "swcre": "rstcre"}
"""Map the cloud radiative effect to the variable EBAF publishes it under.

obs4MIPs ``CERES-EBAF-4-2-1`` carries EBAF's own cloud radiative effect as
``rltcre`` (``toa_cre_lw_mon``) and ``rstcre`` (``toa_cre_sw_mon``), with the
sign of ``lwcre`` / ``swcre``: the effect of clouds on the net downward flux,
longwave positive and shortwave negative.

"""


class CERESEBAF(ESMValToolIntakeDataSource):
    """Manage CERES-EBAF data.

    The cloud radiative effect (``lwcre``, ``swcre``) is EBAF's own where its
    files are staged (:data:`PUBLISHED_CRE`), not the difference of the staged
    clear-sky and all-sky fluxes. The two are different quantities. The
    ``rlutcs`` / ``rsutcs`` of the obs4MIPs dataset are EBAF's *clear-scene*
    fluxes (the clear-sky flux of the cloud-free part of each region), whereas
    a model's clear-sky flux is the whole column with clouds removed, which is
    what EBAF's own cloud radiative effect is built on (EBAF Ed4.2 Data Quality
    Summary; Loeb et al. 2020, doi:10.1175/JCLI-D-19-0381.1). The derived
    longwave effect is about 2 W m-2 larger in the global mean. Without the
    published files the derived effect is used, with a log message.
    """

    _information = DataSourceInformation(
        name="CERES-EBAF",
        category="observation",
        institute="NASA-LaRC",
        references=("https://doi.org/10.5067/TERRA-AQUA-NOAA20/CERES/EBAF_L3B004.2.1",),
    )

    esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
        "project": "obs4MIPs",
        "dataset": "CERES-EBAF-4-2-1",
    }

    def get_cube(
        self,
        data_root_dir: Path | str,
        variable: Variable,
        *,
        download_missing_data: bool = True,
    ) -> Cube:
        """Get data cube; the cloud radiative effect is EBAF's own if staged."""
        published = PUBLISHED_CRE.get(variable.var_name)
        if published is not None:
            directory = Path(data_root_dir) / self.id / variable.frequency / published
            paths = _paths_within_timerange(
                list(directory.glob("*.nc")),
                variable.timerange,
            )
            if paths:
                logger.debug(
                    f"Loading {variable} of {self.id} from the published '{published}'",
                )
                cubes = load_cubes(paths)
                for cube in cubes:
                    if cube.var_name == published:
                        cube.var_name = variable.var_name
                        cube.attributes["cloud_radiative_effect"] = (
                            f"published by CERES-EBAF as {published}"
                        )
                return get_prepared_cube(cubes, variable)
            logger.info(
                f"No published '{published}' staged for {self.id} "
                f"({directory}): '{variable.var_name}' is derived from the "
                f"clear-scene and all-sky fluxes instead",
            )
        return super().get_cube(
            data_root_dir,
            variable,
            download_missing_data=download_missing_data,
        )
