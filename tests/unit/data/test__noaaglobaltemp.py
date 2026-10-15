from __future__ import annotations

import pytest

from climateeval import Variable
from climateeval.data import DataSourceInformation, NOAAGlobalTemp


@pytest.fixture
def data_source() -> NOAAGlobalTemp:
    return NOAAGlobalTemp()


def test_data_source(data_source: NOAAGlobalTemp) -> None:
    assert data_source.name == "NOAAGlobalTemp"
    assert data_source.information == DataSourceInformation(
        name="NOAAGlobalTemp",
        category="observation",
        institute="NOAA-NCEI",
        references=("https://doi.org/10.25921/9qth-2p70",),
    )


def test_supported_variables(data_source: NOAAGlobalTemp) -> None:
    data_source._check_supported_variables(Variable("tasa", "tasa", "mon"))

    with pytest.raises(NotImplementedError, match="does not provide variable 'tas'"):
        data_source._check_supported_variables(Variable("tas", "tas", "mon"))
