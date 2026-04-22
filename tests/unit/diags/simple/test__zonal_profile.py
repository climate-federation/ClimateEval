from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from benchmarking.diags.simple import ZonalProfileAirPressure

if TYPE_CHECKING:
    from pathlib import Path

    from iris.cube import Cube

    from benchmarking import Variable
    from benchmarking.diags.simple import SimpleDiagnosticInputData


@pytest.fixture
def diagnostic(
    variables_and_data_3d: dict[Variable, SimpleDiagnosticInputData],
    data_root_dir: Path,
) -> ZonalProfileAirPressure:
    return ZonalProfileAirPressure(variables_and_data_3d, data_root_dir=data_root_dir)


def test__safely_preprocess(diagnostic: ZonalProfileAirPressure, cube_3d: Cube) -> None:
    diagnostic._safely_preprocess(cube_3d)
