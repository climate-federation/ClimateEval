from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from climateeval.data import (
    GPCP,
    MERRA2,
    DataSourceGenerator,
    ERA5Monthly,
    HadCRUT5,
)
from climateeval.diags.simple import SimpleDiagnosticInputData

if TYPE_CHECKING:
    from collections.abc import Generator

    from climateeval import Variable
    from climateeval.data import DataSource


@pytest.fixture
def variables_and_data_2d(
    variable_tas: Variable,
    variable_pr: Variable,
) -> dict[Variable, SimpleDiagnosticInputData]:
    return {
        variable_tas: SimpleDiagnosticInputData(reference=HadCRUT5),
        variable_pr: SimpleDiagnosticInputData(reference=GPCP),
    }


@pytest.fixture
def variables_and_data_2d_with_other(
    variable_tas: Variable,
) -> dict[Variable, SimpleDiagnosticInputData]:
    class MockDataSourceGenerator(DataSourceGenerator):
        def generate(self, _variable: Variable) -> Generator[DataSource]:
            """Generate data sources."""
            yield from (ERA5Monthly(),)

    return {
        variable_tas: SimpleDiagnosticInputData(
            reference=HadCRUT5,
            other=(MockDataSourceGenerator, MERRA2),
        ),
    }


@pytest.fixture
def variables_and_data_3d(
    variable_ta: Variable,
) -> dict[Variable, SimpleDiagnosticInputData]:
    return {
        variable_ta: SimpleDiagnosticInputData(reference=ERA5Monthly),
    }
