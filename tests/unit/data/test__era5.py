from __future__ import annotations

import pytest

from climateeval.data import ERA5, DataSourceInformation


@pytest.fixture
def data_source() -> ERA5:
    return ERA5()


def test_data_source(data_source: ERA5) -> None:
    assert data_source.name == "ERA5"
    assert data_source.information == DataSourceInformation(
        name="ERA5",
        category="reanalysis",
        institute="ECMWF",
        references=(
            "https://doi.org/10.24381/cds.f17050d7",
            "https://doi.org/10.24381/cds.6860a573",
        ),
    )
