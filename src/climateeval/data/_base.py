"""Provide base classes for data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger

from climateeval._utils import get_prepared_cube, load_cubes
from climateeval.data._utils import facet_to_str
from climateeval.exceptions import DownloadError

if TYPE_CHECKING:
    from collections.abc import Generator

    from iris.cube import Cube, CubeList
    from xarray import Dataset

    from climateeval import Variable
    from climateeval.data._utils import DataSourceInformation


logger = logger.opt(colors=True)


class DataSource(ABC):
    """Manage data source."""

    _information: ClassVar[DataSourceInformation]

    def __getattr__(self, name: str) -> Any:
        """Access attributes."""
        if name == "_information" and name not in self.__dict__:
            msg = (
                "Child classes of 'DataSource' should define the class variable "
                "'_information'"
            )
            raise TypeError(msg)
        return super().__getattribute__(name)

    @property
    def id(self) -> str:
        """Get data source ID."""
        return self.information.id

    @property
    def information(self) -> DataSourceInformation:
        """Get data source information."""
        return self._information

    @property
    def name(self) -> str:
        """Get data source name."""
        return self.information.name

    @abstractmethod
    def download(self, variable: Variable, data_root_dir: Path | str) -> list[Path]:
        """Download data."""

    def get_cube(
        self,
        data_root_dir: Path | str,
        variable: Variable,
        *,
        download_missing_data: bool = True,
    ) -> Cube:
        """Get data cube."""
        logger.debug(f"Loading {variable} of {self.id}")

        data_root_dir = Path(data_root_dir)
        paths: list[Path] = list(
            self._get_data_dir(data_root_dir, variable).glob("*.nc"),
        )
        if paths:
            return get_prepared_cube(load_cubes(paths), variable)

        logger.debug(f"No data found for {variable} of {self.id}")

        # Check if variable can be derived from existing data
        if variable.derived:
            paths = self._get_paths_of_required_variables(variable, data_root_dir)
            if paths:
                logger.debug(
                    f"Found required variables to derive {variable} of {self.id}",
                )
                return get_prepared_cube(load_cubes(paths), variable)

        if not download_missing_data:
            msg = (
                f"{variable} of {self.id} would need to be downloaded, but "
                f"downloads are disabled"
            )
            raise DownloadError(msg)

        # Try downloading the variable
        logger.debug(f"Trying downloading {variable} of {self.id}")
        error_msg = ""
        try:
            paths = self.download(variable, data_root_dir)
        except Exception as exc:
            if not variable.derived:
                msg = f"Downloading {variable} of {self.id} failed: {exc}"
                raise DownloadError(msg) from exc
            error_msg = f": {exc}"
        if paths:
            logger.info(f"Downloaded {self.id}:\n{pformat(paths)}")
            return get_prepared_cube(load_cubes(paths), variable)

        logger.debug(
            f"Failed to download derived variable {variable} of {self.id}{error_msg}",
        )

        # Try downloading required variables
        get_required_variables = variable.get_required_variables()
        logger.debug(f"Trying required variables {get_required_variables}")
        paths = self._get_paths_of_required_variables(
            variable,
            data_root_dir,
            download=True,
        )
        cubes = load_cubes(paths)
        return get_prepared_cube(cubes, variable)

    def _get_data_dir(self, data_root_dir: Path, variable: Variable) -> Path:
        """Get data- and variable-specific subdirectory."""
        return data_root_dir / self.id / variable.frequency / variable.var_name

    def _get_output_filename(self, cube: Cube, **facets: Any) -> str:
        """Get output file name."""
        suffix = ".nc"
        if cube.coords("time"):
            time_coord = cube.coord("time")
            start_date = time_coord.units.num2date(time_coord.points[0]).strftime(
                "%Y%m%d%H%M%S",
            )
            end_date = time_coord.units.num2date(time_coord.points[-1]).strftime(
                "%Y%m%d%H%M%S",
            )
            suffix = f"_{start_date}-{end_date}{suffix}"

        normalized_facets: dict[str, str] = {}
        for facet_key, facet_value in facets.items():
            if facet_key == "timerange":
                continue
            # Facet values may be lists (e.g. `exp = ["historical", "ssp245"]`
            # for experiments that are concatenated into a single time series)
            normalized_facets[facet_key] = facet_to_str(facet_value).replace(" ", "-")
        prefix = "_".join(normalized_facets.values())

        return f"{prefix}{suffix}"

    def _get_paths_of_required_variables(
        self,
        variable: Variable,
        data_root_dir: Path,
        *,
        download: bool = False,
    ) -> list[Path]:
        """Get paths of required variables."""
        paths: list[Path] = []
        for required_variable in variable.get_required_variables():
            paths_for_required_variable = list(
                self._get_data_dir(data_root_dir, required_variable).glob(
                    "*.nc",
                ),
            )
            if not paths_for_required_variable:
                # Do not return any paths if not all required variables are
                # available
                if not download:
                    return []

                # Download
                paths_for_required_variable = self.download(
                    required_variable,
                    data_root_dir,
                )
                logger.info(
                    f"Downloaded {self.id}:\n{pformat(paths_for_required_variable)}",
                )

            paths.extend(paths_for_required_variable)
        return paths


@dataclass(frozen=True, kw_only=True)
class ComplexDataSource:
    """Manage complex data sources (intended for usage within complex diagnostics)."""

    data: dict[str, CubeList | Dataset] = field(repr=False)
    information: DataSourceInformation = field(repr=False)
    id: str = field(init=False, compare=False)
    name: str = field(init=False, repr=False, compare=False)
    data_keys: tuple[str, ...] = field(init=False, compare=False)

    def __post_init__(self) -> None:
        """Initialize class instance."""
        # See https://docs.python.org/3/library/dataclasses.html#frozen-instances
        object.__setattr__(self, "id", self.information.id)
        object.__setattr__(self, "name", self.information.name)
        object.__setattr__(self, "data_keys", tuple(self.data.keys()))


class DataSourceGenerator(ABC):
    """Generate data sources."""

    @abstractmethod
    def generate(self, variable: Variable) -> Generator[DataSource]:
        """Generate data sources."""
