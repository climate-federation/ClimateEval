from __future__ import annotations

import pytest

from benchmarking.data import CERESEBAF, DataSourceInformation


@pytest.fixture
def data_source() -> CERESEBAF:
    return CERESEBAF()


def test_data_source(data_source: CERESEBAF) -> None:
    assert data_source.name == "CERES-EBAF"
    assert data_source.information == DataSourceInformation(
        type="observation",
        institute="NASA-LaRC",
        name="CERES-EBAF",
        exp="",
        variant="",
        references=("https://doi.org/10.5067/TERRA-AQUA-NOAA20/CERES/EBAF_L3B004.2.1",),
    )
