from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from benchmarking.diags import ZonalProfileAirPressure
from tests.unit.diags import assert_preprocessing_works

if TYPE_CHECKING:
    from pathlib import Path

    from iris.cube import Cube

    from benchmarking import Variable
    from benchmarking.data import DiagnosticData


@pytest.fixture
def diagnostic(
    variables_and_data_3d: dict[Variable, DiagnosticData],
    data_dir: Path,
) -> ZonalProfileAirPressure:
    return ZonalProfileAirPressure(variables_and_data_3d, data_dir=data_dir)


def test__preprocess(diagnostic: ZonalProfileAirPressure, cube_3d: Cube) -> None:
    assert_preprocessing_works(diagnostic, cube_3d)
