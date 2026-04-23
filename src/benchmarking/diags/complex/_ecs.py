"""Provide Effective Climate Sensitivity (ECS) diagnostic."""

from __future__ import annotations

from typing import TYPE_CHECKING

from esmvalcore.preprocessor import (
    annual_statistics,
    area_statistics,
    climate_statistics,
    regrid,
)
from iris import NameConstraint
from iris.cube import Cube, CubeList
from scipy import stats

from benchmarking import Variable, setup_esmvaltool_config_and_logging
from benchmarking._utils import get_prepared_cube
from benchmarking.diags._utils import DEFAULT_GRID
from benchmarking.diags.complex._base import ComplexDiagnostic

if TYPE_CHECKING:
    from xarray import Dataset

    from benchmarking.data import ComplexDataSource


class ECS(ComplexDiagnostic):
    """Manage ECS diagnostics.

    Additional keyword arguments
    ----------------------------
    year_slice: slice (default: slice(None, 150))
        Desired years expressed as :obj:`slice`. By default, use the first 150
        years.

    """

    REQUIRED_DATA_KEYS = ("4xco2", "picontrol")

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        data_dict = complex_data_source.data
        tas = Variable("tas", "tas", "mon")
        rtnt = Variable("rtnt", "rtnt", "mon")

        cubes_4xco2 = CubeList([])
        cubes_picontrol = CubeList([])
        for variable in (tas, rtnt):
            cubes_4xco2.append(self._get_cube(data_dict["4xco2"], variable))
            cubes_picontrol.append(self._get_cube(data_dict["picontrol"], variable))

        # Calculate change in abrupt4xCO2 run by subtracting PI-control
        # climatology from it. This is not the most optimal way of doing it
        # (better: subtract linear fit of PI-control run from the "same"
        # simulation years determined by branch_time; see
        # https://doi.org/10.5194/esd-11-1233-2020), but does not depend on the
        # presence and/or correctness of the branch_time attribute.
        tas_cube = cubes_4xco2.extract_cube(
            NameConstraint(var_name=tas.var_name),
        ) - climate_statistics(
            cubes_picontrol.extract_cube(NameConstraint(var_name=tas.var_name)),
            "mean",
            "full",
        )
        rtnt_cube = cubes_4xco2.extract_cube(
            NameConstraint(var_name=rtnt.var_name),
        ) - climate_statistics(
            cubes_picontrol.extract_cube(NameConstraint(var_name=rtnt.var_name)),
            "mean",
            "full",
        )

        # Gregory regression (rtnt = F_4x + lambda * tas)
        reg = stats.linregress(tas_cube.data, rtnt_cube.data)

        return {
            Variable("ecs", "ecs", "fx"): Cube(
                -reg.intercept / 2.0 / reg.slope,
                var_name="ecs",
            ),
            Variable("f2x", "f2x", "fx"): Cube(reg.intercept / 2.0, var_name="f2x"),
            Variable("f4x", "f4x", "fx"): Cube(reg.intercept, var_name="f2x"),
            Variable("lambda", "lambda", "fx"): Cube(reg.slope, var_name="lambda"),
            Variable("lambda_stderr", "lambda_stderr", "fx"): Cube(
                reg.stderr,
                var_name="lambda_stderr",
            ),
            Variable("p_value", "p_value", "fx"): Cube(reg.pvalue, var_name="p_value"),
            Variable("r2", "r2", "fx"): Cube(reg.rvalue**2, var_name="r2"),
        }

    def _get_cube(self, data: CubeList | Dataset, variable: Variable) -> Cube:
        """Get data cube."""
        cube = get_prepared_cube(data, variable)
        return self._preprocess(cube)

    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess data."""
        with setup_esmvaltool_config_and_logging():
            cube = regrid(
                cube,
                DEFAULT_GRID,
                "linear",
                cache_weights=True,
            )  # to ensure we are on a regular grid
            cube = area_statistics(cube, "mean")
            cube = annual_statistics(cube, "mean")
        slices: list[slice] = [slice(None)] * cube.ndim
        slices[cube.coord_dims("time")[0]] = self._kwargs.get(
            "year_slice",
            slice(None, 150),
        )
        cube = cube[tuple(slices)]
        return cube  # noqa: RET504
