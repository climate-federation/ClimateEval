"""Provide ORAS5 data.

ORAS5 is loaded through ESMValCore's ``native6`` on-the-fly CMORizer (see
:class:`~climateeval.data.ESMValCoreNative6DataSource`), which requires the
ORAS5 fix added in https://github.com/ESMValGroup/ESMValCore/pull/2422 (not yet
released as of ESMValCore 2.14). ORAS5 is on the curvilinear ORCA025 grid and
needs a horizontal grid (mesh) file to be read correctly.

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
regrid to the variable's ``additional_preprocessors`` (the simple diagnostics
otherwise regrid with the ``linear`` scheme, which does not support irregular
grids).

"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from climateeval.data._esmvalcore_native6 import ESMValCoreNative6DataSource
from climateeval.data._utils import DataSourceInformation

if TYPE_CHECKING:
    from climateeval import Variable


class ORAS5(ESMValCoreNative6DataSource):
    """Manage ORAS5 data."""

    _information = DataSourceInformation(
        name="ORAS5",
        category="reanalysis",
        institute="ECMWF",
        references=("https://doi.org/10.5194/os-15-779-2019",),
    )

    _supported_var_names = ("mlotst", "tos", "sos", "zos", "thetao", "so", "hfds")
    _dataset_facets: ClassVar[dict[str, str]] = {
        "type": "reanaly",
        "version": "CONS_v0.1",
    }

    _RAW_DIR_ENV: ClassVar[str] = "CLIMATEEVAL_ORAS5_RAW_DIR"
    _GRID_DIR_ENV: ClassVar[str] = "CLIMATEEVAL_ORAS5_GRID_DIR"
    _DEFAULT_RAW_DIR: ClassVar[str] = (
        "/work/bd1083/b382555/extraobsraw/Tier2/ORAS5/data"
    )
    _DEFAULT_GRID_DIR: ClassVar[str] = (
        "/work/bd1083/b382555/extraobsraw/Tier2/ORAS5/grids"
    )
    _GRID_BY_VAR: ClassVar[dict[str, str]] = {"uo": "u", "tauuo": "u", "vo": "v"}

    def _get_grid_path(self, variable: Variable) -> Path:
        """Get path to the ORAS5 horizontal grid (mesh) file for a variable."""
        grid_dir = Path(os.environ.get(self._GRID_DIR_ENV, self._DEFAULT_GRID_DIR))
        grid = self._GRID_BY_VAR.get(variable.var_name, "T")
        return grid_dir / f"oras5_mesh_{grid}.nc"

    def _get_raw_files(
        self,
        variable: Variable,  # noqa: ARG002
        data_root_dir: Path,  # noqa: ARG002
        raw_name: str,
    ) -> list[Path]:
        """Get the raw ORAS5 input files for a variable (matched by raw name)."""
        raw_dir = Path(os.environ.get(self._RAW_DIR_ENV, self._DEFAULT_RAW_DIR))
        return sorted(raw_dir.glob(f"{raw_name}_*.nc"))
