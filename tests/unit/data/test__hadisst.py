from __future__ import annotations

import pytest

from climateeval.data import DataSourceInformation, HadISST


@pytest.fixture
def data_source() -> HadISST:
    return HadISST()


def test_data_source(data_source: HadISST) -> None:
    assert data_source.name == "HadISST"
    assert data_source.information == DataSourceInformation(
        name="HadISST",
        category="reanalysis",
        institute="Met Office Hadley Centre",
        references=("https://doi.org/10.1029/2002JD002670",),
    )
