from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from climateeval.diags.simple import TimeSeriesNoPreprocessing

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable
    from climateeval.diags.simple import SimpleDiagnosticInputData


@pytest.fixture
def diagnostic(
    variables_and_data_2d: dict[Variable, SimpleDiagnosticInputData],
) -> TimeSeriesNoPreprocessing:
    return TimeSeriesNoPreprocessing("diagnostic", variables_and_data_2d)


def test_name(diagnostic: TimeSeriesNoPreprocessing) -> None:
    assert diagnostic.name == "diagnostic"


def test__safely_preprocess(
    diagnostic: TimeSeriesNoPreprocessing,
    cube_2d: Cube,
    variable_tas: Variable,
) -> None:
    cube_2d = cube_2d[:, 0, 0]
    diagnostic._safely_preprocess(cube_2d, variable_tas)
