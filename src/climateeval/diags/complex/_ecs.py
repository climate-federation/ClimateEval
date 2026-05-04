"""Provide Effective Climate Sensitivity (ECS) diagnostic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from climateeval import (
    Variable,
    setup_esmvaltool_config_and_logging,
    setup_iris_combine_policy,
)
from climateeval._utils import get_prepared_cube
from climateeval.data import (
    CMIP6Abrupt4xCO2R1I1P1F1,
    CMIP6PiControlR1I1P1F1,
    ComplexDataSource,
    DataSourceInformation,
)
from climateeval.data._esmvaltool_intake import filter_intake_data_sources
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.complex._base import ComplexDiagnostic
from esmvalcore.preprocessor import (
    annual_statistics,
    area_statistics,
    climate_statistics,
    regrid,
)
from iris import NameConstraint
from iris.cube import Cube, CubeList
from loguru import logger
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from xarray import Dataset


logger = logger.opt(colors=True)


class ECS(ComplexDiagnostic):
    """Manage ECS diagnostics.

    Additional keyword arguments
    ----------------------------
    year_slice: slice (default: slice(None, 150))
        Desired years expressed as :obj:`slice`. By default, use the first 150
        years.

    """

    REQUIRED_DATA_KEYS = ("4xco2", "picontrol")
    _RLUT: ClassVar[Variable] = Variable("rlut", "rlut", "mon")
    _RSDT: ClassVar[Variable] = Variable("rsdt", "rsdt", "mon")
    _RSUT: ClassVar[Variable] = Variable("rsut", "rsut", "mon")
    _RTNT: ClassVar[Variable] = Variable("rtnt", "rtnt", "mon")
    _TAS: ClassVar[Variable] = Variable("tas", "tas", "mon")

    def __init__(
        self,
        *,
        data_root_dir: Path | None = None,
        fail_on_missing_data: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize class instance."""
        super().__init__(
            data_root_dir=data_root_dir,
            fail_on_missing_data=fail_on_missing_data,
            **kwargs,
        )
        self._years_slice: slice = self._kwargs.get("year_slice", slice(None, 150))
        logger.debug(f"Using year slice '{self._years_slice}' for Gregory regression")

    def _calculate_raw_output(
        self,
        complex_data_source: ComplexDataSource,
    ) -> dict[Variable, Cube]:
        """Calculate diagnostic raw output."""
        data_dict = complex_data_source.data
        cubes_4xco2 = CubeList([])
        cubes_picontrol = CubeList([])
        for variable in (self._TAS, self._RTNT):
            cubes_4xco2.append(self._get_cube(data_dict["4xco2"], variable))
            cubes_picontrol.append(self._get_cube(data_dict["picontrol"], variable))

        # Calculate change in abrupt4xCO2 run by subtracting PI-control
        # climatology from it. This is not the most optimal way of doing it
        # (better: subtract linear fit of PI-control run from the "same"
        # simulation years determined by branch_time; see
        # https://doi.org/10.5194/esd-11-1233-2020), but does not depend on the
        # presence and/or correctness of the branch_time attribute.
        tas_picontrol = climate_statistics(
            cubes_picontrol.extract_cube(NameConstraint(var_name=self._TAS.var_name)),
            "mean",
            "full",
        )
        tas_cube = (
            cubes_4xco2.extract_cube(NameConstraint(var_name=self._TAS.var_name))
            - tas_picontrol
        )
        rtnt_picontrol = climate_statistics(
            cubes_picontrol.extract_cube(NameConstraint(var_name=self._RTNT.var_name)),
            "mean",
            "full",
        )
        rtnt_cube = (
            cubes_4xco2.extract_cube(NameConstraint(var_name=self._RTNT.var_name))
            - rtnt_picontrol
        )

        # Select desired years
        tas_cube = tas_cube[self._years_slice]
        rtnt_cube = rtnt_cube[self._years_slice]

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

    def _get_other_data(self) -> Generator[ComplexDataSource]:
        """Get other data."""
        logger.debug(f"Retrieving other data for compley diagnostic {self.name}")
        variables = (self._RLUT, self._RSDT, self._RSUT, self._TAS)
        generators = [CMIP6PiControlR1I1P1F1().generate(var) for var in variables] + [
            CMIP6Abrupt4xCO2R1I1P1F1().generate(var) for var in variables
        ]

        grouped_data_sources = filter_intake_data_sources(
            *generators,
            group_by="dataset",
            required_facets=(
                {"exp": "piControl", "short_name": "rlut"},
                {"exp": "piControl", "short_name": "rsdt"},
                {"exp": "piControl", "short_name": "rsut"},
                {"exp": "piControl", "short_name": "tas"},
                {"exp": "abrupt-4xCO2", "short_name": "rlut"},
                {"exp": "abrupt-4xCO2", "short_name": "rsdt"},
                {"exp": "abrupt-4xCO2", "short_name": "rsut"},
                {"exp": "abrupt-4xCO2", "short_name": "tas"},
            ),
        )
        for dataset, data_sources in grouped_data_sources.items():
            logger.debug(f"Found other data of CMIP6 model {dataset}")

            # We only need one data source per experiment (one data source
            # can handle all variables)
            data_source_4xco2 = next(
                d
                for d in data_sources
                if d.__class__.__name__ == "CMIP6Abrupt4xCO2R1I1P1F1DataSource"
            )
            data_source_picontrol = next(
                d
                for d in data_sources
                if d.__class__.__name__ == "CMIP6PiControlR1I1P1F1DataSource"
            )

            data_information = DataSourceInformation(
                name=dataset,
                category="CMIP6",
                institute=data_source_4xco2.information.category,
                exp="piControl abrupt-4xCO2",
                variant="r1i1p1f1",
                references=data_source_4xco2.information.references,
            )

            with setup_iris_combine_policy():
                try:
                    data = {
                        "4xco2": CubeList(
                            data_source_4xco2.get_cube(self.data_root_dir, var)
                            for var in variables
                        ),
                        "picontrol": CubeList(
                            data_source_picontrol.get_cube(self.data_root_dir, var)
                            for var in variables
                        ),
                    }
                except Exception as exc:  # noqa: BLE001
                    self._handle_missing_data(data_source_4xco2.id, variables, exc)
                    continue
            yield ComplexDataSource(data=data, information=data_information)

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
        return cube  # noqa: RET504
