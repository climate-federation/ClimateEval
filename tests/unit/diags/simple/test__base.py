from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from climateeval import Coordinate, Variable
from climateeval.data import ERA5, MERRA2, HadCRUT5
from climateeval.diags.simple import (
    SimpleDiagnostic,
    SimpleDiagnosticInputData,
)

if TYPE_CHECKING:
    from pathlib import Path

    from iris.cube import Cube


class MockDiagnostic(SimpleDiagnostic):
    FINAL_COORDINATES = (Coordinate("lat"),)

    def _preprocess(self, cube: Cube) -> Cube:
        """Preprocess cube."""
        return cube


@pytest.fixture
def diagnostic(
    variables_and_data_2d_with_other: dict[Variable, SimpleDiagnosticInputData],
) -> MockDiagnostic:
    return MockDiagnostic(variables_and_data_2d_with_other)


def test___init__(diagnostic: MockDiagnostic, temporary_data_root_dir: Path) -> None:
    assert diagnostic.data_root_dir == temporary_data_root_dir
    assert diagnostic.final_coordinate_names == ("latitude",)
    assert diagnostic.name == "MockDiagnostic"

    expected_variable = Variable("my_tas", "tas", "mon")
    assert diagnostic.variables == (expected_variable,)
    assert len(diagnostic.reference_data) == 1
    assert expected_variable in diagnostic.reference_data
    assert isinstance(diagnostic.reference_data[expected_variable], HadCRUT5)
    assert len(diagnostic.other_data) == 1
    assert expected_variable in diagnostic.other_data
    assert len(diagnostic.other_data[expected_variable]) == 2
    assert isinstance(diagnostic.other_data[expected_variable][0], ERA5)
    assert isinstance(diagnostic.other_data[expected_variable][1], MERRA2)


def test_non_unique_variables_fail() -> None:
    var_1 = Variable("tas", "tas", "mon")
    var_2 = Variable("tas", "tas", "day")
    data = SimpleDiagnosticInputData(reference=HadCRUT5)
    variables_and_data = {var_1: data, var_2: data}
    msg = (
        r"Variable IDs must be unique for diagnostic MockDiagnostic, got ['tas', 'tas']"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        MockDiagnostic(variables_and_data)
