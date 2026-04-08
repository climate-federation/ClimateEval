"""Benchmarking Diagnostics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

import ibis
import ibis.selectors
import pandas as pd
from esmvalcore.preprocessor import clip_timerange
from iris import Constraint, NameConstraint
from iris.coords import CoordExtent
from iris.pandas import as_data_frame
from ncdata.iris_xarray import cubes_from_xarray
from xarray import Dataset

from benchmarking import Variable

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from iris.cube import Cube, CubeList

    from benchmarking import Config
    from benchmarking.data import DataSource


class Diagnostic(ABC):
    """Manage diagnostics."""

    DATASET_STR: ClassVar[str] = "Dataset"

    def __init__(
        self,
        *,
        reference_data: Mapping[str, type[DataSource]],
        other_data: Mapping[str, Iterable[type[DataSource]]],
        config: Config,
    ) -> None:
        """Initialize class instance."""
        self._config = config

        if not reference_data:
            msg = "At least a single variable and reference dataset must be given."
            raise ValueError(msg)

        self._variables = [Variable(v) for v in reference_data]
        self._reference_data = {
            Variable(v): r(Variable(v), self.config) for v, r in reference_data.items()
        }

        _other_data: dict[Variable, tuple[DataSource, ...]] = {}
        for var in self.variables:
            if var.var_name in other_data:
                val = tuple(o(var, self.config) for o in other_data[var.var_name])
            else:
                val = ()
            _other_data[var] = val
        self._other_data = _other_data

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
    def reference_data(self) -> dict[Variable, DataSource]:
        """Get reference data."""
        return self._reference_data

    @property
    def other_data(self) -> dict[Variable, tuple[DataSource, ...]]:
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

    def get_table(
        self,
        data: CubeList | Dataset,
        data_name: str,
    ) -> ibis.Table:
        """Get table containing all data.

        Parameters
        ----------
        data:
            Data to benchmark.
        data_name:
            Name of data to benchmark.

        Returns
        -------
        :
            Data table (long form). Columns include variables, coordinates, and
            data names.

        """
        table = self._get_table_of_data(data, data_name)

        # Concatenate data for different data sources (i.e., add new rows)
        for ref_data in self.reference_data.values():
            ref_cubes = ref_data.get_cubes()
            new_table = self._get_table_of_data(ref_cubes, ref_data.name)
            table = ibis.union(table, new_table)

        print(table.to_pandas())
        assert 0

        for variable, ref_data in self.reference_data.items():
            df_var = self._get_table_of_data(data, data_name, variable)
            if table.empty:
                table = df_var

            # Merge
        return pd.concat(
            [
                self._get_table_of_cube(cube, data_name, variable),
                self._get_table_of_cube(ref_cube, ref_data.name, variable),
                *(
                    self._get_table_of_cube(
                        self._get_cube(o.get_cubes(self.config)),
                        o.name,
                        variable,
                    )
                    for o in other_data
                ),
            ],
            axis=0,
            ignore_index=True,
        )

    @abstractmethod
    def get_metrics(self, data: CubeList | Dataset) -> dict[str, float]:
        """Get metrics."""

    @abstractmethod
    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess cube."""

    def _get_cube(self, data: CubeList | Dataset, variable: Variable) -> Cube:
        """Get data cube."""
        cube = self._get_prepared_cube(data, variable, self.config)
        return self._preprocess(cube)

    def _get_prepared_cube(
        self,
        data: CubeList | Dataset,
        variable: Variable,
        config: Config,
    ) -> Cube:
        """Get prepared Iris cube."""
        if isinstance(data, Dataset):
            data = cubes_from_xarray(data)
        cube = data.extract_cube(NameConstraint(var_name=variable.var_name))

        # Make sure units are correct (this will potentiall raise an error)
        cube.convert_units(variable.units)

        # Make sure longitude is in [0, 360]
        if cube.coords("longitude"):
            cube = self._set_lon_to_0_360(cube)

        # Extract desired time range and bounding box
        if cube.coords("time") and config.timerange != "*":
            cube = clip_timerange(cube, config.timerange)
        if cube.coords("latitude"):
            cube = cube.extract(
                Constraint(
                    latitude=lambda cell: config.min_lat <= cell <= config.max_lat,
                ),
            )
        if cube.coords("longitude"):
            cube = cube.extract(
                Constraint(
                    longitude=lambda cell: config.min_lon <= cell <= config.max_lon,
                ),
            )

        return cube

    def _get_table_of_data(
        self,
        data: CubeList | Dataset,
        data_name: str,
    ) -> ibis.Table:
        """Get table of data to benchmark."""
        variable = self.variables[0]  # self.variables will never be empty
        cube = self._get_cube(data, variable)
        table = self._get_table_of_cube(cube, data_name, variable)
        for variable in self.variables[1:]:
            cube = self._get_cube(data, variable)
            new_table = self._get_table_of_cube(cube, data_name, variable)
            table = table.join(new_table, how="outer", rname="{name}_DUPLICATE")

            # Assume that aux_coords, cell_measures, and ancillary_variables
            # are equal across cubes within the group. Only add them when they
            # first appear.
            table = table.select(ibis.selectors.matches(r"^(?!.*_DUPLICATE)"))

        return table

    def _get_table_of_cube(
        self,
        cube: Cube,
        name: str,
        variable: Variable,
    ) -> ibis.Table:
        """Get table of a single cube."""
        df = as_data_frame(
            cube,
            add_aux_coords=True,
            add_cell_measures=True,
            add_ancillary_variables=True,
        ).reset_index()
        if "time" in df.columns:
            df["time"] = df["time"].astype(str)
        table = ibis.memtable(df)
        table = table.rename(**{variable.var_name: cube.name()})
        table = table.mutate(**{self.DATASET_STR: ibis.literal(name)})
        if "time" in table.columns:
            table = table.cast(time="timestamp")
        return table

    def _get_tags(self) -> tuple[str, ...]:
        """Get diagnostic tags."""
        tags: list[str] = []
        tags.append(self.name.lower())
        tags.extend(v.var_name for v in self.variables)
        tags.extend(v.realm for v in self.variables)
        return tuple(sorted(set(tags)))

    def _set_lon_to_0_360(self, cube: Cube) -> Cube:
        """Put longitude into [0, 360]."""
        cube = cube.copy()
        coord = cube.coord("longitude")

        # cube.intersection only works for cells with 0 or 2 bounds
        # Note: nbounds==0 means there are no bounds given, nbounds==2 implies a
        # regular grid with bounds in the grid direction, nbounds>2 implies an
        # irregular grid with bounds given as vertices of the cell polygon.
        if coord.ndim == 1 and coord.nbounds in (0, 2):
            lon_extent = CoordExtent(
                coord,
                0.0,
                360.0,
                min_inclusive=True,
                max_inclusive=False,
            )
            cube = cube.intersection(lon_extent)
        else:
            new_points = (coord.core_points().copy() + 360.0) % 360.0
            if coord.has_bounds():
                new_bounds = (coord.core_bounds().copy() + 360.0) % 360.0
            else:
                new_bounds = None
            new_coord = coord.copy(points=new_points, bounds=new_bounds)
            coord_dims = cube.coord_dims(coord)
            cube.remove_coord(coord)
            cube.add_aux_coord(new_coord, coord_dims)

        return cube
