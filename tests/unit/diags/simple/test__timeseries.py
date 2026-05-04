from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from climateeval.diags.simple import TimeSeries

if TYPE_CHECKING:
    from climateeval import Variable
    from climateeval.diags.simple import SimpleDiagnosticInputData
    from iris.cube import Cube


@pytest.fixture
def diagnostic(
    variables_and_data_2d: dict[Variable, SimpleDiagnosticInputData],
) -> TimeSeries:
    return TimeSeries(variables_and_data_2d)


def test__safely_preprocess(diagnostic: TimeSeries, cube_2d: Cube) -> None:
    diagnostic._safely_preprocess(cube_2d)
