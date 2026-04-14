from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from benchmarking.diags import ZonalLine

if TYPE_CHECKING:
    from pathlib import Path

    from iris.cube import Cube

    from benchmarking import Variable
    from benchmarking.data import DiagnosticData


@pytest.fixture
def diagnostic(
    variables_and_data_2d: dict[Variable, DiagnosticData],
    data_dir: Path,
) -> ZonalLine:
    return ZonalLine(variables_and_data_2d, data_dir=data_dir)


def test__safely_preprocess(diagnostic: ZonalLine, cube_2d: Cube) -> None:
    diagnostic._safely_preprocess(cube_2d)
