from __future__ import annotations

import pytest

from climateeval import Variable
from climateeval.data import CRU, DataSourceInformation


@pytest.fixture
def data_source() -> CRU:
    return CRU()


def test_data_source(data_source: CRU) -> None:
    assert data_source.name == "CRU"
    assert data_source.information == DataSourceInformation(
        name="CRU",
        category="observation",
        institute="University of East Anglia CRU",
        references=("https://doi.org/10.1038/s41597-020-0453-3",),
    )


@pytest.mark.parametrize("var_name", ["clt", "pr", "tas", "tasmax", "tasmin"])
def test_supported_variables(data_source: CRU, var_name: str) -> None:
    data_source._check_supported_variables(Variable(var_name, var_name, "mon"))


def test_unsupported_variable(data_source: CRU) -> None:
    with pytest.raises(NotImplementedError, match="does not provide variable 'ts'"):
        data_source._check_supported_variables(Variable("ts", "ts", "mon"))


def test_file_prefix_pattern_excludes_emon_variables(data_source: CRU) -> None:
    """The CMORizer also writes ``evspsblpot`` (``Emon``), which must be skipped."""
    assert data_source._file_prefix_pattern == "*_Amon_*"
