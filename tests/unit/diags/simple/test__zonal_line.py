from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from climateeval.diags.simple import ZonalLine

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable
    from climateeval.diags.simple import SimpleDiagnosticInputData


@pytest.fixture
def diagnostic(
    variables_and_data_2d: dict[Variable, SimpleDiagnosticInputData],
) -> ZonalLine:
    return ZonalLine("diagnostic", variables_and_data_2d)


def test_name(diagnostic: ZonalLine) -> None:
    assert diagnostic.name == "diagnostic"


def test__safely_preprocess(
    diagnostic: ZonalLine,
    cube_2d: Cube,
    variable_tas: Variable,
) -> None:
    diagnostic._safely_preprocess(cube_2d, variable_tas)
