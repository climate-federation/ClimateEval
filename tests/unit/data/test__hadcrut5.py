from __future__ import annotations

import pytest

from climateeval.data import DataSourceInformation, HadCRUT5


@pytest.fixture
def data_source() -> HadCRUT5:
    return HadCRUT5()


def test_data_source(data_source: HadCRUT5) -> None:
    assert data_source.name == "HadCRUT5"
    assert data_source.information == DataSourceInformation(
        name="HadCRUT5",
        category="observation",
        institute="Met Office Hadley Centre",
        references=("https://doi.org/10.1029/2019JD032361",),
    )
