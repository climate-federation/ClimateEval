"""Provide Effective Climate Sensitivity (ECS) diagnostic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

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

from climateeval import Variable
from climateeval._config import (
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
from climateeval.diags._base import DiagnosticOutput
from climateeval.diags._utils import DEFAULT_GRID
from climateeval.diags.complex._base import ComplexDiagnostic

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

    _required_data_keys = ("4xco2", "picontrol")

    # Input variables
    _rlut: ClassVar[Variable] = Variable("rlut", "rlut", "mon")
    _rsdt: ClassVar[Variable] = Variable("rsdt", "rsdt", "mon")
    _rsut: ClassVar[Variable] = Variable("rsut", "rsut", "mon")
    _rtnt: ClassVar[Variable] = Variable("rtnt", "rtnt", "mon")
    _tas: ClassVar[Variable] = Variable("tas", "tas", "mon")

    # Output variables
    _ecs: ClassVar[Variable] = Variable("ecs", "ecs", "fx")
    _f2x: ClassVar[Variable] = Variable("f2x", "f2x", "fx")
    _f4x: ClassVar[Variable] = Variable("f4x", "f4x", "fx")
    _lambda: ClassVar[Variable] = Variable("lambda", "lambda", "fx")
    _lambda_stderr: ClassVar[Variable] = Variable(
        "lambda_stderr",
        "lambda_stderr",
        "fx",
    )
    _p_value: ClassVar[Variable] = Variable("p_value", "p_value", "fx")
    _r2: ClassVar[Variable] = Variable("r2", "r2", "fx")

    def __init__(
        self,
        name: str,
        *,
        data_root_dir: Path | str | None = None,
        fail_on_missing_data: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize class instance."""
        super().__init__(
            name,
            data_root_dir=data_root_dir,
            fail_on_missing_data=fail_on_missing_data,
            **kwargs,
        )
        if "year_slice" in self._kwargs and not isinstance(
            self._kwargs["year_slice"],
            slice,
        ):
            self._kwargs["year_slice"] = slice(*self._kwargs["year_slice"])
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
        for variable in (self._tas, self._rtnt):
            cubes_4xco2.append(self._get_cube(data_dict["4xco2"], variable))
            cubes_picontrol.append(self._get_cube(data_dict["picontrol"], variable))

        # Calculate change in abrupt4xCO2 run by subtracting PI-control
        # climatology from it. This is not the most optimal way of doing it
        # (better: subtract linear fit of PI-control run from the "same"
        # simulation years determined by branch_time; see
        # https://doi.org/10.5194/esd-11-1233-2020), but does not depend on the
        # presence and/or correctness of the branch_time attribute.
        tas_picontrol = climate_statistics(
            cubes_picontrol.extract_cube(NameConstraint(var_name=self._tas.var_name)),
            "mean",
            "full",
        )
        tas_cube = (
            cubes_4xco2.extract_cube(NameConstraint(var_name=self._tas.var_name))
            - tas_picontrol
        )
        rtnt_picontrol = climate_statistics(
            cubes_picontrol.extract_cube(NameConstraint(var_name=self._rtnt.var_name)),
            "mean",
            "full",
        )
        rtnt_cube = (
            cubes_4xco2.extract_cube(NameConstraint(var_name=self._rtnt.var_name))
            - rtnt_picontrol
        )

        # Select desired years
        tas_cube = tas_cube[self._years_slice]
        rtnt_cube = rtnt_cube[self._years_slice]

        # Gregory regression (rtnt = F_4x + lambda * tas)
        reg = stats.linregress(tas_cube.data, rtnt_cube.data)

        return {
            self._ecs: Cube(-reg.intercept / 2.0 / reg.slope, var_name="ecs"),
            self._f2x: Cube(reg.intercept / 2.0, var_name="f2x"),
            self._f4x: Cube(reg.intercept, var_name="f2x"),
            self._lambda: Cube(reg.slope, var_name="lambda"),
            self._lambda_stderr: Cube(reg.stderr, var_name="lambda_stderr"),
            self._p_value: Cube(reg.pvalue, var_name="p_value"),
            self._r2: Cube(reg.rvalue**2, var_name="r2"),
        }

    def _get_cube(self, data: CubeList | Dataset, variable: Variable) -> Cube:
        """Get data cube."""
        cube = get_prepared_cube(data, variable)
        return self._preprocess(cube)

    def _get_other_data(self) -> Generator[ComplexDataSource]:
        """Get other data."""
        logger.debug(f"Retrieving other data for compley diagnostic '{self.name}'")
        variables = (self._rlut, self._rsdt, self._rsut, self._tas)
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
                category=data_source_4xco2.information.category,
                institute=data_source_4xco2.information.institute,
                exp="piControl abrupt-4xCO2",
                variant=data_source_4xco2.information.variant,
                references=data_source_4xco2.information.references,
            )

            with setup_iris_combine_policy():
                try:
                    data = {
                        "4xco2": CubeList(
                            data_source_4xco2.get_cube(
                                self.data_root_dir,
                                var,
                                download_missing_data=self._download_missing_data,
                            )
                            for var in variables
                        ),
                        "picontrol": CubeList(
                            data_source_picontrol.get_cube(
                                self.data_root_dir,
                                var,
                                download_missing_data=self._download_missing_data,
                            )
                            for var in variables
                        ),
                    }
                except Exception as exc:  # noqa: BLE001
                    self._handle_missing_data(data_source_4xco2.id, variables, exc)
                    continue
            yield ComplexDataSource(data=data, information=data_information)

    def _postprocess_output(self, output: DiagnosticOutput) -> DiagnosticOutput:
        """Postprocess diagnostic output."""
        if output.raw_output is None:
            return output

        # Drop unused column
        raw_output = output.raw_output.drop("level_0")

        # Remove superfluous rows
        unique_columns = (self._data_id_column, self._data_type_column)
        variables = (
            self._ecs.id,
            self._f2x.id,
            self._f4x.id,
            self._lambda.id,
            self._lambda_stderr.id,
            self._p_value.id,
            self._r2.id,
        )
        raw_output = self._remove_superfluous_rows(
            raw_output,
            unique_columns=unique_columns,
            variable_columns=variables,
        )

        # Sort columns
        raw_output = raw_output.select((*unique_columns, *variables))

        return DiagnosticOutput(
            raw_output=raw_output,
            metrics=output.metrics,
            variables=output.variables,
            data_sources=output.data_sources,
        )

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
