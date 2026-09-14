from __future__ import annotations

import pytest

from climateeval import Variable
from climateeval.data import BerkeleyEarth, DataSourceInformation


@pytest.fixture
def data_source() -> BerkeleyEarth:
    return BerkeleyEarth()


def test_data_source(data_source: BerkeleyEarth) -> None:
    assert data_source.name == "BerkeleyEarth"
    assert data_source.information == DataSourceInformation(
        name="BerkeleyEarth",
        category="observation",
        institute="Berkeley Earth",
        references=("https://doi.org/10.4172/2327-4581.1000101",),
    )


@pytest.mark.parametrize("var_name", ["tas", "tasa"])
def test_supported_variables(data_source: BerkeleyEarth, var_name: str) -> None:
    data_source._check_supported_variables(Variable(var_name, var_name, "mon"))


def test_unsupported_variable(data_source: BerkeleyEarth) -> None:
    with pytest.raises(NotImplementedError, match="does not provide variable 'pr'"):
        data_source._check_supported_variables(Variable("pr", "pr", "mon"))


def test_file_prefix_pattern_excludes_land_sea_mask(
    data_source: BerkeleyEarth,
) -> None:
    """The CMORizer also writes an ``fx`` land-sea mask, which must be skipped."""
    assert data_source._file_prefix_pattern == "*_Amon_*"
