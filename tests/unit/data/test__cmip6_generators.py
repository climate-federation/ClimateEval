from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from climateeval import Variable
from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval.data import (
    CMIP6Abrupt4xCO2R1I1P1F1,
    CMIP6DataSourceGenerator,
    CMIP6HistoricalAllMembers,
    CMIP6HistoricalR1I1P1F1,
    CMIP6HistoricalSSP245,
    CMIP6PiControlR1I1P1F1,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class MockDataset:
    """Minimal stand-in for `esmvalcore.dataset.Dataset`."""

    #: Facets of the datasets that `from_files` pretends to have found.
    discovered_facets: list[dict[str, Any]] = []  # noqa: RUF012

    #: Facets the (mocked) dataset has been created with.
    requested_facets: dict[str, Any] = {}  # noqa: RUF012

    def __init__(self, **facets: Any) -> None:
        type(self).requested_facets = dict(facets)
        self.facets = facets

    def from_files(self) -> Iterator[MockDataset]:
        for facets in self.discovered_facets:
            dataset = object.__new__(type(self))
            dataset.facets = dict(facets)
            yield dataset


@pytest.fixture
def variable_tos() -> Variable:
    return Variable("tos", "tos", "mon")


@pytest.fixture
def mock_dataset(monkeypatch: pytest.MonkeyPatch) -> type[MockDataset]:
    monkeypatch.setattr(
        "climateeval.data._esmvaltool_intake.Dataset",
        MockDataset,
    )
    return MockDataset


def _discovered(exp: str | list[str], ensemble: str) -> dict[str, Any]:
    return {
        "project": "CMIP6",
        "exp": exp,
        "dataset": "MPI-ESM1-2-LR",
        "institute": "MPI-M",
        "ensemble": ensemble,
        "grid": "gn",
        "mip": "Omon",
        "short_name": "tos",
        "frequency": "mon",
    }


@pytest.mark.parametrize(
    ("generator_cls", "exp", "ensemble"),
    [
        (CMIP6HistoricalR1I1P1F1, "historical", "r1i1p1f1"),
        (CMIP6PiControlR1I1P1F1, "piControl", "r1i1p1f1"),
        (CMIP6Abrupt4xCO2R1I1P1F1, "abrupt-4xCO2", "r1i1p1f1"),
        (CMIP6HistoricalAllMembers, "historical", "r*i1p1f1"),
        (CMIP6HistoricalSSP245, ["historical", "ssp245"], "r*i1p1f1"),
    ],
)
def test_facets(
    generator_cls: type[CMIP6DataSourceGenerator],
    exp: str | list[str],
    ensemble: str,
) -> None:
    facets = generator_cls.esmvaltool_dataset_facets

    assert facets["project"] == "CMIP6"
    assert facets["exp"] == exp
    assert facets["ensemble"] == ensemble
    assert facets["dataset"] == "*"


@pytest.mark.parametrize(
    ("exp", "expected_references"),
    [
        (
            "historical",
            (
                "https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.MPI-M."
                "MPI-ESM1-2-LR",
            ),
        ),
        (
            ["historical", "ssp245"],
            (
                "https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.MPI-M."
                "MPI-ESM1-2-LR",
                "https://www.wdc-climate.de/ui/cmip6?input=CMIP6.ScenarioMIP."
                "MPI-M.MPI-ESM1-2-LR",
            ),
        ),
        (
            # Identical activities are only reported once
            ["historical", "amip"],
            (
                "https://www.wdc-climate.de/ui/cmip6?input=CMIP6.CMIP.MPI-M."
                "MPI-ESM1-2-LR",
            ),
        ),
    ],
)
def test_get_references(
    exp: str | list[str],
    expected_references: tuple[str, ...],
) -> None:
    generator = CMIP6HistoricalSSP245()

    with setup_esmvaltool_config_and_logging():
        references = generator._get_references(**_discovered(exp, "r1i1p1f1"))

    assert references == expected_references


def test_generate_one_data_source_per_member(
    mock_dataset: type[MockDataset],
    variable_tos: Variable,
) -> None:
    mock_dataset.discovered_facets = [
        _discovered("historical", "r1i1p1f1"),
        _discovered("historical", "r2i1p1f1"),
    ]
    generator = CMIP6HistoricalAllMembers()

    with setup_esmvaltool_config_and_logging():
        data_sources = list(generator.generate(variable_tos))

    assert mock_dataset.requested_facets["ensemble"] == "r*i1p1f1"
    assert mock_dataset.requested_facets["short_name"] == "tos"
    assert [d.id for d in data_sources] == [
        "CMIP6_MPI-ESM1-2-LR_historical_r1i1p1f1",
        "CMIP6_MPI-ESM1-2-LR_historical_r2i1p1f1",
    ]
    assert [d.information.variant for d in data_sources] == [
        "r1i1p1f1",
        "r2i1p1f1",
    ]
    assert all(d.information.exp == "historical" for d in data_sources)


def test_generate_concatenated_experiments(
    mock_dataset: type[MockDataset],
    variable_tos: Variable,
) -> None:
    mock_dataset.discovered_facets = [
        _discovered(["historical", "ssp245"], "r1i1p1f1"),
        _discovered(["historical", "ssp245"], "r2i1p1f1"),
    ]
    generator = CMIP6HistoricalSSP245()

    with setup_esmvaltool_config_and_logging():
        data_sources = list(generator.generate(variable_tos))

    assert mock_dataset.requested_facets["exp"] == ["historical", "ssp245"]
    assert mock_dataset.requested_facets["timerange"] == "19790101/*"
    assert [d.id for d in data_sources] == [
        "CMIP6_MPI-ESM1-2-LR_historical-ssp245_r1i1p1f1",
        "CMIP6_MPI-ESM1-2-LR_historical-ssp245_r2i1p1f1",
    ]
    for data_source in data_sources:
        assert data_source.information.exp == "historical-ssp245"
        assert data_source.esmvaltool_dataset_facets["exp"] == [
            "historical",
            "ssp245",
        ]
        assert len(data_source.information.references) == 2


def test_get_output_filename_with_concatenated_experiments(
    mock_dataset: type[MockDataset],
    variable_tos: Variable,
    cube_2d: Any,
) -> None:
    mock_dataset.discovered_facets = [
        _discovered(["historical", "ssp245"], "r1i1p1f1"),
    ]
    generator = CMIP6HistoricalSSP245()

    with setup_esmvaltool_config_and_logging():
        data_source = next(iter(generator.generate(variable_tos)))

    filename = data_source._get_output_filename(
        cube_2d,
        **data_source.esmvaltool_dataset_facets,
    )

    assert filename.startswith("CMIP6_historical-ssp245_MPI-ESM1-2-LR_MPI-M_")
    assert filename.endswith(".nc")
