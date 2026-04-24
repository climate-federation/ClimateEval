"""Provide base classes for data sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pprint import pformat
from typing import TYPE_CHECKING, ClassVar

from loguru import logger

from benchmarking._utils import get_prepared_cube, load_cubes

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from iris.cube import Cube, CubeList
    from xarray import Dataset

    from benchmarking import Variable
    from benchmarking.data._utils import DataSourceInformation


logger = logger.opt(colors=True)


class DataSource(ABC):
    """Manage data source."""

    _INFORMATION: ClassVar[DataSourceInformation]

    @property
    def id(self) -> str:
        """Get data source ID."""
        return self.information.id

    @property
    def information(self) -> DataSourceInformation:
        """Get data source information."""
        return self._INFORMATION

    @property
    def name(self) -> str:
        """Get data source name."""
        return self.information.name

    @abstractmethod
    def download(self, data_root_dir: Path, variable: Variable) -> list[Path]:
        """Download data."""

    def get_cube(self, data_root_dir: Path, variable: Variable) -> Cube:
        """Get data cube."""
        logger.debug(f"Loading {variable} of {self.id}")
        paths: list[Path] = list(
            self._get_data_dir(data_root_dir, variable.var_name).glob("*.nc"),
        )
        if paths:
            return get_prepared_cube(load_cubes(paths), variable)

        logger.debug(f"No data found for {variable} of {self.id}")

        # Check if variable can be derived from existing data
        if variable.derived:
            paths = self._get_paths_of_required_variables(data_root_dir, variable)
            if paths:
                logger.debug(
                    f"Found required variables to derive {variable} of {self.id}",
                )
                return get_prepared_cube(load_cubes(paths), variable)

        # Try downloading the variable
        logger.debug(f"Trying downloading {variable} of {self.id}")
        error_msg = ""
        try:
            paths = self.download(data_root_dir, variable)
        except Exception as exc:
            if not variable.derived:
                msg = f"Downloading {variable} of {self.id} failed: {exc}"
                raise ValueError(msg) from exc
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
            data_root_dir,
            variable,
            download=True,
        )
        cubes = load_cubes(paths)
        return get_prepared_cube(cubes, variable)

    def _get_data_dir(self, data_root_dir: Path, var_name: str) -> Path:
        """Get data- and variable-specific subdirectory."""
        return data_root_dir / self.id / var_name

    def _get_output_filename(self, cube: Cube, **facets: str) -> str:
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
            normalized_facets[facet_key] = facet_value.replace(" ", "-")
        prefix = "_".join(
            str(facet_value) for facet_value in normalized_facets.values()
        )

        return f"{prefix}{suffix}"

    def _get_paths_of_required_variables(
        self,
        data_root_dir: Path,
        variable: Variable,
        *,
        download: bool = False,
    ) -> list[Path]:
        """Get paths of required variables."""
        paths: list[Path] = []
        for required_variable in variable.get_required_variables():
            paths_for_required_variable = list(
                self._get_data_dir(data_root_dir, required_variable.var_name).glob(
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
                    data_root_dir,
                    required_variable,
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
