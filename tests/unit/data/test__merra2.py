from __future__ import annotations

import pytest

from benchmarking.data import MERRA2, DataSourceInformation


@pytest.fixture
def data_source() -> MERRA2:
    return MERRA2()


def test_data_source(data_source: MERRA2) -> None:
    assert data_source.name == "MERRA2"
    assert data_source.information == DataSourceInformation(
        name="MERRA2",
        category="reanalysis",
        institute="NASA GMAO",
        references=(
            "https://doi.org/10.5067/0JRLVL8YV2Y4",
            "https://doi.org/10.5067/AP1B0BA5PD2K",
        ),
    )
