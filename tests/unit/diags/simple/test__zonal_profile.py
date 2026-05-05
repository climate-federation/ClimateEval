from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from climateeval.diags.simple import ZonalProfileAirPressure

if TYPE_CHECKING:
    from iris.cube import Cube

    from climateeval import Variable
    from climateeval.diags.simple import SimpleDiagnosticInputData


@pytest.fixture
def diagnostic(
    variables_and_data_3d: dict[Variable, SimpleDiagnosticInputData],
) -> ZonalProfileAirPressure:
    return ZonalProfileAirPressure(variables_and_data_3d)


def test__safely_preprocess(diagnostic: ZonalProfileAirPressure, cube_3d: Cube) -> None:
    diagnostic._safely_preprocess(cube_3d)
