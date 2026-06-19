"""Provide ORAS5 data.

ORAS5 is provided on the curvilinear ORCA025 grid by raw NEMO output files plus
separate grid (mesh) files. This module loads those files directly with Iris
(replicating what ESMValCore's ``native6`` ORAS5 fix does for the irregular
grid: drop the redundant halo cells, rename the variable, and attach the
horizontal cell bounds from the grid file). This keeps ORAS5 support independent
of the (currently unreleased) ESMValCore ORAS5 fix.

Staging the data
----------------
ORAS5 raw data cannot be downloaded automatically; download it from the
Copernicus Climate Data Store and the grid files from the link in the ESMValCore
documentation, then point ClimateEval at them with two environment variables:

- ``CLIMATEEVAL_ORAS5_RAW_DIR``: directory with the raw ``*.nc`` files.
- ``CLIMATEEVAL_ORAS5_GRID_DIR``: directory with ``oras5_mesh_{T,u,v}.nc``.

Both default to the data staged on Levante. Variables on the tracer grid use
``oras5_mesh_T.nc``; ``uo``/``tauuo`` use the ``u`` grid and ``vo`` the ``v``
grid.

Note
----
ORAS5 fields are on an irregular grid, so diagnostics must regrid with an
area-weighted scheme. In suites this is done by adding an ``area_weighted``
regrid (followed by ``mask_landsea``) to the variable's
``additional_preprocessors``; the simple diagnostics otherwise regrid with the
``linear`` scheme, which does not support irregular grids.

This loader currently handles 2-D (surface) variables only. Loading 3-D
variables additionally requires fixing the vertical coordinate.

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import iris
import numpy as np
from iris import NameConstraint
from iris.cube import CubeList
from loguru import logger

from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._utils import get_prepared_cube, load_cubes
from climateeval._variable import VARIABLES
from climateeval.data._base import DataSource
from climateeval.data._utils import DataSourceInformation
from climateeval.exceptions import DownloadError, MissingDataError

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable

logger = logger.opt(colors=True)


class ORAS5(DataSource):
    """Manage ORAS5 data."""

    _information = DataSourceInformation(
        name="ORAS5",
        category="reanalysis",
        institute="ECMWF",
        references=("https://doi.org/10.5194/os-15-779-2019",),
    )

    #: Map of CMIP variable name to (raw NEMO variable name, raw units). Only
    #: 2-D (surface) variables are supported by this loader.
    _RAW_VARIABLE_INFO: ClassVar[dict[str, tuple[str, str]]] = {
        "mlotst": ("somxl010", "m"),
        "tos": ("sosstsst", "degC"),
        "sos": ("sosaline", "0.001"),
        "zos": ("sossheig", "m"),
        "hfds": ("sohefldo", "W m-2"),
    }

    _GRID_BY_VAR: ClassVar[dict[str, str]] = {"uo": "u", "tauuo": "u", "vo": "v"}

    _RAW_DIR_ENV: ClassVar[str] = "CLIMATEEVAL_ORAS5_RAW_DIR"
    _GRID_DIR_ENV: ClassVar[str] = "CLIMATEEVAL_ORAS5_GRID_DIR"
    _DEFAULT_RAW_DIR: ClassVar[str] = (
        "/work/bd1083/b382555/extraobsraw/Tier2/ORAS5/data"
    )
    _DEFAULT_GRID_DIR: ClassVar[str] = (
        "/work/bd1083/b382555/extraobsraw/Tier2/ORAS5/grids"
    )

    def get_cube(
        self,
        data_root_dir: Path | str,  # noqa: ARG002
        variable: Variable,
        *,
        download_missing_data: bool = True,  # noqa: ARG002
    ) -> Cube:
        """Get data cube from the raw ORAS5 files."""
        logger.debug(f"Loading {variable} of {self.id}")
        self._check_supported_variables(variable)

        files = self._get_raw_files(variable)
        if not files:
            msg = f"No raw files found for {variable} of {self.id}"
            raise MissingDataError(msg)

        with setup_esmvaltool_config_and_logging():
            cube = self._load_native_cube(files, variable)
            return get_prepared_cube(CubeList([cube]), variable)

    def download(self, variable: Variable, data_root_dir: Path | str) -> list[Path]:  # noqa: ARG002
        """Download data.

        ORAS5 cannot be downloaded automatically; stage the raw data and grid
        files manually (see the module documentation).
        """
        msg = (
            f"{self.id} cannot be downloaded automatically; stage the raw data "
            f"and grid files manually (see the climateeval.data._oras5 docs)"
        )
        raise DownloadError(msg)

    def _check_supported_variables(self, variable: Variable) -> None:
        """Raise an error if a variable is not supported."""
        if variable.var_name not in self._RAW_VARIABLE_INFO:
            msg = (
                f"{self.id} does not provide variable '{variable.var_name}' "
                f"(supported: {sorted(self._RAW_VARIABLE_INFO)})"
            )
            raise NotImplementedError(msg)

    def _load_native_cube(self, files: list[Path], variable: Variable) -> Cube:
        """Load and CMORize raw ORAS5 files for a 2-D variable."""
        raw_name, raw_units = self._RAW_VARIABLE_INFO[variable.var_name]
        cube = load_cubes([str(file) for file in files]).extract_cube(
            NameConstraint(var_name=raw_name),
        )

        # Drop the redundant halo cells of the ORCA025 grid so that the data
        # aligns with the grid (mesh) file.
        cube = cube[..., :-1, 1:-1]

        # Set variable metadata.
        var_info = VARIABLES[variable.var_name]
        cube.var_name = variable.var_name
        cube.long_name = var_info["long_name"]
        cube.units = raw_units
        cube.standard_name = var_info["standard_name"] or None

        self._add_grid_bounds(cube, variable)

        return cube

    def _add_grid_bounds(self, cube: Cube, variable: Variable) -> None:
        """Attach horizontal cell bounds from the ORAS5 grid (mesh) file."""
        grid = iris.load_cube(
            str(self._get_grid_path(variable)),
            NameConstraint(var_name="cell_area"),
        )
        for coord_name in ("latitude", "longitude"):
            # Grid bounds are stored as (x, y, 4); the data is (..., y, x), so
            # transpose the first two axes to match.
            bounds = np.transpose(grid.coord(coord_name).core_bounds(), (1, 0, 2))
            cube.coord(coord_name).bounds = bounds

    def _get_grid_path(self, variable: Variable) -> Path:
        """Get path to the ORAS5 horizontal grid (mesh) file for a variable."""
        grid_dir = Path(os.environ.get(self._GRID_DIR_ENV, self._DEFAULT_GRID_DIR))
        grid = self._GRID_BY_VAR.get(variable.var_name, "T")
        return grid_dir / f"oras5_mesh_{grid}.nc"

    def _get_raw_files(self, variable: Variable) -> list[Path]:
        """Get the raw ORAS5 input files for a variable (matched by raw name)."""
        raw_name = self._RAW_VARIABLE_INFO[variable.var_name][0]
        raw_dir = Path(os.environ.get(self._RAW_DIR_ENV, self._DEFAULT_RAW_DIR))
        return sorted(raw_dir.glob(f"{raw_name}_*.nc"))
