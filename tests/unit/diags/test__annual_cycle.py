from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from benchmarking.diags import AnnualCycle
from tests.unit.diags import assert_preprocessing_works

if TYPE_CHECKING:
    from pathlib import Path

    from iris.cube import Cube

    from benchmarking import Variable
    from benchmarking.data import DiagnosticData


@pytest.fixture
def diagnostic(
    variables_and_data_2d: dict[Variable, DiagnosticData],
    data_dir: Path,
) -> AnnualCycle:
    return AnnualCycle(variables_and_data_2d, data_dir=data_dir)


def test__preprocess(diagnostic: AnnualCycle, cube_2d: Cube) -> None:
    assert_preprocessing_works(diagnostic, cube_2d)
