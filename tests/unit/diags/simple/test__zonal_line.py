from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from benchmarking.diags.simple import ZonalLine

if TYPE_CHECKING:
    from iris.cube import Cube

    from benchmarking import Variable
    from benchmarking.diags.simple import SimpleDiagnosticInputData


@pytest.fixture
def diagnostic(
    variables_and_data_2d: dict[Variable, SimpleDiagnosticInputData],
) -> ZonalLine:
    return ZonalLine(variables_and_data_2d)


def test__safely_preprocess(diagnostic: ZonalLine, cube_2d: Cube) -> None:
    diagnostic._safely_preprocess(cube_2d)
