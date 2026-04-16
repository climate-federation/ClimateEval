from __future__ import annotations

import pytest

from benchmarking.data import DataSourceInformation, ESACCICloud


@pytest.fixture
def data_source() -> ESACCICloud:
    return ESACCICloud()


def test_data_source(data_source: ESACCICloud) -> None:
    assert data_source.name == "ESACCI-CLOUD"
    assert data_source.information == DataSourceInformation(
        name="ESACCI-CLOUD",
        type="observation",
        institute="DWD",
        exp="",
        variant="",
        references=("https://doi.org/10.5676/DWD/ESA_Cloud_cci/AVHRR-PM/V003",),
    )
