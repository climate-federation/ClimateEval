from __future__ import annotations

import pytest

from climateeval import Variable
from climateeval.data import GISTEMP, DataSourceInformation


@pytest.fixture
def data_source() -> GISTEMP:
    return GISTEMP()


def test_data_source(data_source: GISTEMP) -> None:
    assert data_source.name == "GISTEMP"
    assert data_source.information == DataSourceInformation(
        name="GISTEMP",
        category="observation",
        institute="NASA-GISS",
        references=("https://doi.org/10.1029/2018JD029522",),
    )


def test_supported_variables(data_source: GISTEMP) -> None:
    data_source._check_supported_variables(Variable("tasa", "tasa", "mon"))

    with pytest.raises(NotImplementedError, match="does not provide variable 'tas'"):
        data_source._check_supported_variables(Variable("tas", "tas", "mon"))
