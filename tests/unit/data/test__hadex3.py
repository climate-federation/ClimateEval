from __future__ import annotations

import pytest

from climateeval import Variable
from climateeval.data import DataSourceInformation, HadEX3


@pytest.fixture
def data_source() -> HadEX3:
    return HadEX3()


def test_data_source(data_source: HadEX3) -> None:
    assert data_source.name == "HadEX3"
    assert data_source.information == DataSourceInformation(
        name="HadEX3",
        category="observation",
        institute="Met Office Hadley Centre",
        references=("https://doi.org/10.1029/2019JD032263",),
    )


@pytest.mark.parametrize("var_name", ["rx1day", "rx5day", "tnn", "txx"])
def test_supported_variables(data_source: HadEX3, var_name: str) -> None:
    data_source._check_supported_variables(Variable(var_name, var_name, "yr"))


def test_unsupported_variable(data_source: HadEX3) -> None:
    with pytest.raises(NotImplementedError, match="does not provide variable 'tas'"):
        data_source._check_supported_variables(Variable("tas", "tas", "yr"))


def test_unsupported_frequency(data_source: HadEX3) -> None:
    """The monthly indices the CMORizer also writes are not used."""
    msg = "does not provide variable 'txx' with frequency 'mon'"
    with pytest.raises(NotImplementedError, match=msg):
        data_source._check_supported_variables(Variable("txx", "txx", "mon"))


def test_file_prefix_pattern_excludes_monthly_indices(data_source: HadEX3) -> None:
    """The CMORizer writes each index as ``yr`` *and* ``Amon`` field."""
    assert data_source._file_prefix_pattern == "*_yr_*"
