"""Provide locally staged CMOR-compliant data.

Unlike the other data sources, :class:`LocalCMORDataSource` does not know how to
download anything: it serves a directory of CMOR-compliant NetCDF files that the
user has already staged, through the same loader that reads the model being
benchmarked (:func:`climateeval._loader.load_cmor_dir`). It therefore accepts
the same layouts: a single file, a flat directory, or a CMOR DRS tree.

This covers two cases the shipped data sources do not:

- **Perfect-model tests.** A held-out ESM run acts as the *reference* of a
  suite, so a model can be scored against another model instead of against
  observations.
- **Locally staged observations.** A reference that ClimateEval cannot download
  (or that must be pinned to a specific local version) can be pointed at
  directly.

Staging the data
----------------
The directory, its provenance (:class:`~climateeval.data.DataSourceInformation`)
and the preferred grid can be given in three ways:

1. :meth:`LocalCMORDataSource.for_directory` builds a ready-to-use data source
   *class* (data sources are instantiated without arguments by the suites, so
   the settings have to live on the class). Assign it to a module-level name to
   reference it from a suite YAML::

       # mypackage/references.py
       from climateeval.data import LocalCMORDataSource

       HeldOutESM = LocalCMORDataSource.for_directory(
           "/data/held_out_esm/cmor/day",
           name="HeldOutESM",
           category="model",
           exp="historical",
           variant="r1i1p1f1",
       )

   .. code-block:: yaml

       variables:
         - id: tas
           var_name: tas
           frequency: day
           reference_data: mypackage.references.HeldOutESM

2. Constructor arguments, for the programmatic route (a :class:`DataSource`
   *instance* is what the diagnostics work with)::

       LocalCMORDataSource("/data/obs/tas", information=DataSourceInformation(...))

3. The ``CLIMATEEVAL_LOCAL_CMOR_DIR`` environment variable, which lets the
   shipped ``climateeval.data.LocalCMORDataSource`` be referenced from a suite
   YAML without writing any Python (the default provenance is then used).

"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from loguru import logger

from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval._loader import load_cmor_dir
from climateeval._utils import get_prepared_cube
from climateeval.data._base import DataSource
from climateeval.data._utils import DataSourceInformation
from climateeval.exceptions import DownloadError, MissingDataError

if TYPE_CHECKING:
    from typing import Self

    from iris.cube import Cube

    from climateeval import Variable

logger = logger.opt(colors=True)


class LocalCMORDataSource(DataSource):
    """Manage CMOR-compliant data staged in a local directory.

    Parameters
    ----------
    path:
        NetCDF file, flat directory of NetCDF files, or root of a CMOR DRS
        tree. Defaults to the class attribute set by
        :meth:`for_directory`, or to the ``CLIMATEEVAL_LOCAL_CMOR_DIR``
        environment variable.
    information:
        Provenance of the data. Defaults to the class attribute set by
        :meth:`for_directory`.
    preferred_grid:
        Grid label preferred when a DRS variable is available on several
        grids. Defaults to the class attribute set by :meth:`for_directory`
        (the native grid ``gn``).

    """

    _information: ClassVar[DataSourceInformation] = DataSourceInformation(
        name="LocalCMOR",
        category="model",
    )

    _path: ClassVar[Path | str | None] = None
    """Default path of the local data (see :meth:`for_directory`)."""

    _preferred_grid: ClassVar[str] = "gn"
    """Default grid label to prefer in DRS trees (see :meth:`for_directory`)."""

    _PATH_ENV: ClassVar[str] = "CLIMATEEVAL_LOCAL_CMOR_DIR"
    """Environment variable that provides the path if nothing else does."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        information: DataSourceInformation | None = None,
        preferred_grid: str | None = None,
    ) -> None:
        """Initialize class instance."""
        self._given_path = path
        self._given_information = information
        self._given_preferred_grid = preferred_grid

    @classmethod
    def for_directory(  # noqa: PLR0913
        cls,
        path: Path | str,
        *,
        name: str,
        category: str = "model",
        institute: str = "",
        exp: str = "",
        variant: str = "",
        references: tuple[str, ...] = (),
        preferred_grid: str = "gn",
    ) -> type[Self]:
        """Create a data source class for a local CMOR directory.

        Suites instantiate data sources without arguments, so the settings of a
        local data source have to live on the class. This builds such a class;
        assign it to a module-level name to be able to reference it from a
        suite YAML (see the module documentation).

        Parameters
        ----------
        path:
            NetCDF file, flat directory of NetCDF files, or root of a CMOR DRS
            tree.
        name:
            Name of the dataset (e.g. the model or observational product).
        category:
            Category of the dataset (e.g. ``model``, ``observation``).
        institute:
            Institute that provides the dataset.
        exp:
            Experiment of the dataset (e.g. ``historical``).
        variant:
            Variant of the dataset (e.g. ``r1i1p1f1``).
        references:
            References of the dataset.
        preferred_grid:
            Grid label preferred when a DRS variable is available on several
            grids.

        Returns
        -------
        :
            Subclass of this class that serves `path` and can be instantiated
            without arguments.

        """
        information = DataSourceInformation(
            name=name,
            category=category,
            institute=institute,
            exp=exp,
            variant=variant,
            references=references,
        )
        class_name = re.sub(r"\W", "_", f"{cls.__name__}_{information.id}")
        return type(
            class_name,
            (cls,),
            {
                "__doc__": f"Manage local CMOR data of {information.id}.",
                "_information": information,
                "_path": Path(path),
                "_preferred_grid": preferred_grid,
            },
        )

    @property
    def information(self) -> DataSourceInformation:
        """Get data source information."""
        if self._given_information is not None:
            return self._given_information
        return self._information

    @property
    def path(self) -> Path:
        """Get path of the local data."""
        path = self._path if self._given_path is None else self._given_path
        if path is None:
            path = os.environ.get(self._PATH_ENV)
        if path is None:
            msg = (
                f"No path given for {self.id}; pass one to "
                f"{type(self).__name__}.for_directory() or its constructor, or "
                f"set the environment variable {self._PATH_ENV}"
            )
            raise MissingDataError(msg)
        return Path(path).expanduser()

    @property
    def preferred_grid(self) -> str:
        """Get grid label preferred in DRS trees."""
        if self._given_preferred_grid is None:
            return self._preferred_grid
        return self._given_preferred_grid

    def get_cube(
        self,
        data_root_dir: Path | str,  # noqa: ARG002
        variable: Variable,
        *,
        download_missing_data: bool = True,  # noqa: ARG002
    ) -> Cube:
        """Get data cube from the local CMOR data.

        Note
        ----
        `data_root_dir` and `download_missing_data` are ignored: the data is
        read from :attr:`path`, and nothing is ever downloaded.

        """
        logger.debug(f"Loading {variable} of {self.id}")

        path = self.path
        if not path.exists():
            msg = f"Local CMOR path {path} not found for {variable} of {self.id}"
            raise MissingDataError(msg)

        # `load_cmor_dir` filters the files by their CMOR date token; the exact
        # time range is extracted by `get_prepared_cube`.
        timerange = None if variable.timerange == "*" else variable.timerange
        cubes = load_cmor_dir(
            path,
            timerange=timerange,
            preferred_grid=self.preferred_grid,
        )
        if not cubes:
            msg = f"No data found in {path} for {variable} of {self.id}"
            raise MissingDataError(msg)

        with setup_esmvaltool_config_and_logging():
            return get_prepared_cube(cubes, variable)

    def download(self, variable: Variable, data_root_dir: Path | str) -> list[Path]:  # noqa: ARG002
        """Download data.

        Local data cannot be downloaded; stage it and point ClimateEval at it
        (see the module docs).
        """
        msg = (
            f"{self.id} is local data that cannot be downloaded; stage the "
            f"CMOR-compliant files and point ClimateEval at them (see "
            f"climateeval.data._local_cmor)"
        )
        raise DownloadError(msg)
