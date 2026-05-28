from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from climateeval import Coordinate, Variable
from climateeval.data import MERRA2, ERA5Monthly, HadCRUT5
from climateeval.diags.simple import (
    SimpleDiagnostic,
    SimpleDiagnosticInputData,
)

if TYPE_CHECKING:
    from pathlib import Path

    from iris.cube import Cube


class MockDiagnostic(SimpleDiagnostic):
    _FINAL_COORDINATES = (Coordinate("lat"),)

    def _preprocess(self, cube: Cube, _variable: Variable) -> Cube:
        """Preprocess cube."""
        return cube


@pytest.fixture
def diagnostic(
    variables_and_data_2d_with_other: dict[Variable, SimpleDiagnosticInputData],
) -> MockDiagnostic:
    return MockDiagnostic("mock_diagnostic", variables_and_data_2d_with_other)


def test___init__(diagnostic: MockDiagnostic, temporary_data_root_dir: Path) -> None:
    assert diagnostic.data_root_dir == temporary_data_root_dir
    assert diagnostic._final_coordinate_names == ("latitude",)
    assert diagnostic.name == "mock_diagnostic"

    expected_variable = Variable("my_tas", "tas", "mon")
    assert diagnostic._variables == (expected_variable,)
    assert len(diagnostic._reference_data) == 1
    assert expected_variable in diagnostic._reference_data
    assert isinstance(diagnostic._reference_data[expected_variable], HadCRUT5)
    assert len(diagnostic._other_data) == 1
    assert expected_variable in diagnostic._other_data
    assert len(diagnostic._other_data[expected_variable]) == 2
    assert isinstance(diagnostic._other_data[expected_variable][0], ERA5Monthly)
    assert isinstance(diagnostic._other_data[expected_variable][1], MERRA2)


def test_non_unique_variables_fail() -> None:
    var_1 = Variable("tas", "tas", "mon")
    var_2 = Variable("tas", "tas", "day")
    data = SimpleDiagnosticInputData(reference=HadCRUT5)
    variables_and_data = {var_1: data, var_2: data}
    msg = (
        r"Variable IDs must be unique for diagnostic 'mock_diagnostic', got "
        r"['tas', 'tas']"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        MockDiagnostic("mock_diagnostic", variables_and_data)
