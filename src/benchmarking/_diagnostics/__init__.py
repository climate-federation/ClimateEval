"""Benchmarking Diagnostics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
from iris.pandas import as_data_frame

from benchmarking import get_prepared_cube
from benchmarking._variable import Variable

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Mapping

    from iris.cube import Cube, CubeList
    from plotly.graph_objects import Figure
    from xarray import Dataset

    from benchmarking._config import Config
    from benchmarking._data import Data


class Diagnostic(ABC):
    """Manage diagnostics."""

    DATASET_STR: ClassVar[str] = "Dataset"

    def __init__(
        self,
        *,
        variables: Iterable[str],
        reference_data: Mapping[str, type[Data]],
        other_data: Mapping[str, Iterable[type[Data]]],
        config: Config,
    ) -> None:
        """Initialize class instance."""
        self._variables = [Variable(v) for v in variables]

        for var in self.variables:
            if var.var_name not in reference_data:
                msg = f"No reference data for variable '{var.var_name}' given"
                raise ValueError(msg)
        self._reference_data = {
            v: reference_data[v.var_name](v) for v in self.variables
        }

        _other_data: dict[Variable, tuple[Data, ...]] = {}
        for var in self.variables:
            if var.var_name in other_data:
                val = tuple(o(var) for o in other_data[var.var_name])
            else:
                val = ()
            _other_data[var] = val
        self._other_data = _other_data

        self._config = config
        self._tags = self._get_tags()

    @property
    def config(self) -> Config:
        """Get configuration."""
        return self._config

    @property
    def name(self) -> str:
        """Get diagnostic name."""
        return self.__class__.__name__

    @property
    def reference_data(self) -> dict[Variable, Data]:
        """Get reference data."""
        return self._reference_data

    @property
    def other_data(self) -> dict[Variable, tuple[Data, ...]]:
        """Get other data."""
        return self._other_data

    @property
    def tags(self) -> tuple[str, ...]:
        """Get tags."""
        return self._tags

    @property
    def variables(self) -> list[Variable]:
        """Get variables."""
        return self._variables

    @abstractmethod
    def create_plots(
        self,
        data: CubeList | Dataset,
        data_name: str,
        **kwargs: Any,
    ) -> Generator[Figure]:
        """Create plots."""

    @abstractmethod
    def get_metrics(self, data: CubeList | Dataset) -> dict[str, float]:
        """Get metrics."""

    @abstractmethod
    def preprocess(self, cube: Cube) -> Cube:
        """Preprocess cube."""

    def _get_cube(self, data: CubeList | Dataset, variable: Variable) -> Cube:
        """Get data cube."""
        cube = get_prepared_cube(data, variable, self.config)
        return self.preprocess(cube)

    def _get_df(
        self,
        data: CubeList | Dataset,
        data_name: str,
        variable: Variable,
    ) -> pd.DataFrame:
        """Get data frame containing all data."""
        ref_data = self.reference_data[variable]
        other_data = self.other_data[variable]

        cube = self._get_cube(data, variable)
        ref_cube = ref_data.get_cube(self, self.config)

        return pd.concat(
            [
                self._get_single_df(cube, data_name, variable),
                self._get_single_df(ref_cube, ref_data.name, variable),
                *(
                    self._get_single_df(o.get_cube(self, self.config), o.name, variable)
                    for o in other_data
                ),
            ],
            axis=0,
            ignore_index=True,
        )

    def _get_single_df(
        self,
        cube: Cube,
        name: str,
        variable: Variable,
    ) -> pd.DataFrame:
        """Get data frame for a single cube."""
        df = as_data_frame(
            cube,
            add_aux_coords=True,
            add_cell_measures=True,
            add_ancillary_variables=True,
        )
        df = df.rename(
            {cube.name(): variable.var_name},
            axis="columns",
        )
        df[self.DATASET_STR] = name
        return df

    def _get_tags(self) -> tuple[str, ...]:
        """Get diagnostic tags."""
        tags: list[str] = []
        tags.append(self.name.lower())
        tags.extend(v.var_name for v in self.variables)
        tags.extend(v.realm for v in self.variables)
        return tuple(sorted(set(tags)))
