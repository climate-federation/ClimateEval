from __future__ import annotations

import pytest

from climateeval.data import EN4, DataSourceInformation


@pytest.fixture
def data_source() -> EN4:
    return EN4()


def test_data_source(data_source: EN4) -> None:
    assert data_source.name == "EN4"
    assert data_source.information == DataSourceInformation(
        name="EN4",
        category="reanalysis",
        institute="Met Office Hadley Centre",
        references=("https://doi.org/10.1002/2013JC009067",),
    )
