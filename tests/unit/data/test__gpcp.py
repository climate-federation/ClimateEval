from __future__ import annotations

import pytest

from benchmarking.data import GPCP, DataSourceInformation


@pytest.fixture
def data_source() -> GPCP:
    return GPCP()


def test_data_source(data_source: GPCP) -> None:
    assert data_source.name == "GPCP"
    assert data_source.information == DataSourceInformation(
        type="observation",
        institute="NASA-GSFC",
        name="GPCP",
        exp="",
        variant="",
        references=(
            "https://doi.org/10.5067/MEASURES/GPCP/DATA306",
            "https://doi.org/10.5067/MEASURES/GPCP/DATA307",
        ),
    )
