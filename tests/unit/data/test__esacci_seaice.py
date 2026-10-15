from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from climateeval import Variable
from climateeval.data import (
    DataSourceInformation,
    ESACCISeaIceNH,
    ESACCISeaIceSH,
)

if TYPE_CHECKING:
    from pathlib import Path

    from climateeval.data._esacci_seaice import _ESACCISeaIce

REFERENCES = ("https://doi.org/10.5285/eade27004395466aaa006135e1b2ad1a",)


@pytest.fixture(params=[ESACCISeaIceNH, ESACCISeaIceSH])
def data_source(request: pytest.FixtureRequest) -> _ESACCISeaIce:
    return request.param()


def test_northern_hemisphere_data_source() -> None:
    data_source = ESACCISeaIceNH()

    assert data_source.name == "ESACCI-SEAICE"
    assert data_source.id == "observation_ESACCI-SEAICE_nh"
    assert data_source.information == DataSourceInformation(
        name="ESACCI-SEAICE",
        category="observation",
        institute="ESA",
        variant="nh",
        references=REFERENCES,
    )
    assert data_source._file_prefix_pattern == "*-NH_SImon_*"


def test_southern_hemisphere_data_source() -> None:
    data_source = ESACCISeaIceSH()

    assert data_source.name == "ESACCI-SEAICE"
    assert data_source.id == "observation_ESACCI-SEAICE_sh"
    assert data_source.information == DataSourceInformation(
        name="ESACCI-SEAICE",
        category="observation",
        institute="ESA",
        variant="sh",
        references=REFERENCES,
    )
    assert data_source._file_prefix_pattern == "*-SH_SImon_*"


def test_hemispheres_use_separate_data_dirs(tmp_path: Path) -> None:
    """Both hemispheres share a CMORizer, so their data must not be mixed."""
    variable = Variable("siconc", "siconc", "mon")

    nh_dir = ESACCISeaIceNH()._get_data_dir(tmp_path, variable)
    sh_dir = ESACCISeaIceSH()._get_data_dir(tmp_path, variable)

    assert nh_dir != sh_dir


def test_supported_variable(data_source: _ESACCISeaIce) -> None:
    data_source._check_supported_variables(Variable("siconc", "siconc", "mon"))


def test_unsupported_variable(data_source: _ESACCISeaIce) -> None:
    with pytest.raises(NotImplementedError, match="does not provide variable 'tos'"):
        data_source._check_supported_variables(Variable("tos", "tos", "mon"))


def test_unsupported_frequency(data_source: _ESACCISeaIce) -> None:
    """The daily concentrations the CMORizer also writes are not used."""
    msg = "does not provide variable 'siconc' with frequency 'day'"
    with pytest.raises(NotImplementedError, match=msg):
        data_source._check_supported_variables(Variable("siconc", "siconc", "day"))
