from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from climateeval.data import DataSourceInformation, ERA5Daily, ERA5Hourly, ERA5Monthly

if TYPE_CHECKING:
    from climateeval.data._era5 import ERA5


@pytest.fixture(params=["monthly", "daily", "hourly"])
def data_source(request: pytest.FixtureRequest) -> ERA5:
    if request.param == "monthly":
        return ERA5Monthly()
    if request.param == "daily":
        return ERA5Daily()
    return ERA5Hourly()


def test_data_source(data_source: ERA5) -> None:
    assert data_source.name == "ERA5"
    assert data_source.information == DataSourceInformation(
        name="ERA5",
        category="reanalysis",
        institute="ECMWF",
        references=(
            "https://doi.org/10.24381/cds.f17050d7",
            "https://doi.org/10.24381/cds.6860a573",
            "https://doi.org/10.24381/cds.adbb2d47",
        ),
    )
