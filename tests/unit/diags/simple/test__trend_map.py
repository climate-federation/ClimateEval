from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from climateeval.diags.simple import TrendMap

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable
    from climateeval.diags.simple import SimpleDiagnosticInputData


@pytest.fixture
def diagnostic(
    variables_and_data_2d: dict[Variable, SimpleDiagnosticInputData],
) -> TrendMap:
    return TrendMap("diagnostic", variables_and_data_2d)


def test_name(diagnostic: TrendMap) -> None:
    assert diagnostic.name == "diagnostic"


def test__safely_preprocess(
    diagnostic: TrendMap,
    cube_2d: Cube,
    variable_tas: Variable,
) -> None:
    cube = diagnostic._safely_preprocess(cube_2d, variable_tas)
    assert cube.units == "K yr-1"
    assert [c.name() for c in cube.dim_coords] == ["latitude", "longitude"]
