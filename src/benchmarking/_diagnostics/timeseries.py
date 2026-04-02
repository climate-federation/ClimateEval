"""Benchmarking Diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import plotly.express as px
from esmvalcore.preprocessor import annual_statistics, area_statistics, regrid_time

from benchmarking._diagnostics import Diagnostic
from benchmarking._variable import COORDINATES

if TYPE_CHECKING:
    from collections.abc import Generator

    from iris.cube import Cube, CubeList
    from plotly.graph_objects import Figure
    from xarray import Dataset


class TimeSeries(Diagnostic):
    """Manage atmosphere timeseries diagnostics."""

    def create_plots(
        self,
        data: CubeList | Dataset,
        data_name: str,
        **kwargs: Any,
    ) -> Generator[Figure]:
        """Create plots."""
        for variable in self.variables:
            df = self._get_df(data, data_name, variable)
            yield px.line(
                data_frame=df,
                x="year",
                y=variable.var_name,
                color=self.DATASET_STR,
                **kwargs,
            )

    def get_metrics(self, _data: CubeList | Dataset) -> dict[str, float]:
        """Get metrics."""
        return {"rmse": 1.0}

    def preprocess(self, cube: Cube) -> Cube:
        """Preprocess data."""
        cube = area_statistics(cube, "mean")
        cube = annual_statistics(cube, "mean", keep_group_coordinates=True)
        cube = regrid_time(
            cube,
            frequency="yr",
            calendar="standard",
            units=COORDINATES["time"]["units"],
        )
        return cube  # noqa: RET504
