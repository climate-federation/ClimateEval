from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from climateeval.diags.simple import AnnualCycleSpatialDistribution

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable
    from climateeval.diags.simple import SimpleDiagnosticInputData


@pytest.fixture
def diagnostic(
    variables_and_data_2d: dict[Variable, SimpleDiagnosticInputData],
) -> AnnualCycleSpatialDistribution:
    return AnnualCycleSpatialDistribution(variables_and_data_2d)


def test__safely_preprocess(
    diagnostic: AnnualCycleSpatialDistribution,
    cube_2d: Cube,
) -> None:
    diagnostic._safely_preprocess(cube_2d)
