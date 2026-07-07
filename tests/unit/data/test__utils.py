from __future__ import annotations

import pytest

from climateeval.data import DataSourceInformation


def test_data_source_information_defaults() -> None:
    data_source_info = DataSourceInformation(name="NAME", category="CATEGORY")

    assert data_source_info.name == "NAME"
    assert data_source_info.category == "CATEGORY"
    assert data_source_info.institute == ""
    assert data_source_info.exp == ""
    assert data_source_info.variant == ""
    assert data_source_info.references == ()


@pytest.mark.parametrize(
    ("data_source_information", "id_"),
    [
        (DataSourceInformation(name="N A M E", category="C"), "C_N-A-M-E"),
        (DataSourceInformation(name="N", category="C", exp="E X P"), "C_N_E-X-P"),
        (
            DataSourceInformation(
                name="N",
                category="C",
                institute="I",
                exp="E",
                variant="V",
                references=("REFERENCE1", "REFERENCE2"),
            ),
            "C_N_E_V",
        ),
    ],
)
def test_data_source_information_id(
    data_source_information: DataSourceInformation,
    id_: str,
) -> None:
    assert data_source_information.id == id_
