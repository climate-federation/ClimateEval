from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from benchmarking.diags import DiurnalCycle

if TYPE_CHECKING:
    from pathlib import Path

    from iris.cube import Cube

    from benchmarking import Variable
    from benchmarking.data import DiagnosticData


@pytest.fixture
def diagnostic(
    variables_and_data_2d: dict[Variable, DiagnosticData],
    data_root_dir: Path,
) -> DiurnalCycle:
    return DiurnalCycle(variables_and_data_2d, data_root_dir=data_root_dir)


def test__safely_preprocess(diagnostic: DiurnalCycle, cube_2d: Cube) -> None:
    diagnostic._safely_preprocess(cube_2d)
