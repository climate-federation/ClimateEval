from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from benchmarking import Variable
from benchmarking.data import DiagnosticData, HadCRUT5
from benchmarking.diags import TimeSeries

if TYPE_CHECKING:
    from pathlib import Path

    from iris.cube import Cube


@pytest.fixture
def diagnostic(
    variables_and_data_2d: dict[Variable, DiagnosticData],
    data_dir: Path,
) -> TimeSeries:
    return TimeSeries(variables_and_data_2d, data_dir=data_dir)


def test__safely_preprocess(diagnostic: TimeSeries, cube_2d: Cube) -> None:
    diagnostic._safely_preprocess(cube_2d)


def test_non_unique_variables_fail(data_dir: Path) -> None:
    var_1 = Variable("tas", "tas", "mon")
    var_2 = Variable("tas", "tas", "day")
    data = DiagnosticData(reference=HadCRUT5)
    variables_and_data = {var_1: data, var_2: data}
    msg = r"Variable IDs must be unique for diagnostic TimeSeries, got ['tas', 'tas']"
    with pytest.raises(ValueError, match=re.escape(msg)):
        TimeSeries(variables_and_data, data_dir=data_dir)
