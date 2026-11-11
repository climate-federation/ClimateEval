from __future__ import annotations

import contextlib
import re
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

import climateeval.data._esmvaltool_intake
from climateeval import Variable
from climateeval.data import (
    CMIP6DataSourceGenerator,
    DataSourceInformation,
    ESMValToolIntakeDataSource,
)
from climateeval.data._esmvaltool_intake import filter_intake_data_sources
from tests.sample_data.create_sample_data_sources import get_simple_2d_cube

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator
    from pathlib import Path

    from iris.cube import Cube


@pytest.mark.parametrize(
    ("attribute", "exception_cls", "error_msg"),
    [
        (
            "information",
            TypeError,
            r"Child classes of 'DataSource' should define the class variable "
            r"'_information'",
        ),
        (
            "esmvaltool_dataset_facets",
            TypeError,
            r"Child classes of 'ESMValToolIntakeDataSource' should define the class "
            r"variable 'esmvaltool_dataset_facets'",
        ),
        (
            "invalid_attribute",
            AttributeError,
            r"'ESMValToolIntakeDataSource' object has no attribute 'invalid_attribute'",
        ),
    ],
)
def test_data_source_attribute_access_fail(
    attribute: str,
    exception_cls: type[Exception],
    error_msg: str,
) -> None:
    data_source = ESMValToolIntakeDataSource()
    with pytest.raises(exception_cls, match=re.escape(error_msg)):
        getattr(data_source, attribute)


@pytest.mark.parametrize(
    ("attribute", "exception_cls", "error_msg"),
    [
        (
            "esmvaltool_dataset_facets",
            TypeError,
            r"Child classes of 'ESMValToolIntakeDataSourceGenerator' should define "
            r"the class variable 'esmvaltool_dataset_facets'",
        ),
        (
            "invalid_attribute",
            AttributeError,
            r"'CMIP6DataSourceGenerator' object has no attribute "
            r"'invalid_attribute'",
        ),
    ],
)
def test_data_source_generator_attribute_access_fail(
    attribute: str,
    exception_cls: type[Exception],
    error_msg: str,
) -> None:
    data_source = CMIP6DataSourceGenerator()
    with pytest.raises(exception_cls, match=re.escape(error_msg)):
        getattr(data_source, attribute)


@pytest.fixture
def generators() -> tuple[Generator[ESMValToolIntakeDataSource], ...]:
    data_source_facets_1 = [
        {"dataset": "A", "exp": "pi", "short_name": "tas"},
        {"dataset": "A", "exp": "hist", "short_name": "pr"},
        {"dataset": "B", "exp": "hist", "short_name": "pr", "mip": "Amon"},
        {"dataset": "C", "exp": "hist", "short_name": "pr", "mip": "Amon"},
    ]
    data_source_facets_2 = [
        {"dataset": "B", "exp": "pi", "short_name": "tas"},
        {"dataset": "D", "exp": "hist", "mip": "Amon"},
        {"dataset": "D", "exp": "pi", "short_name": "pr", "mip": "Amon"},
        {"dataset": "D", "exp": "hist", "short_name": "tas", "mip": "Amon"},
        {"dataset": "E", "mip": "Lmon"},
        {"dataset": "F"},
    ]
    generator_1 = (
        type("CLS", (ESMValToolIntakeDataSource,), {"esmvaltool_dataset_facets": f})()
        for f in data_source_facets_1
    )
    generator_2 = (
        type("CLS", (ESMValToolIntakeDataSource,), {"esmvaltool_dataset_facets": f})()
        for f in data_source_facets_2
    )
    return (generator_1, generator_2)


@pytest.mark.parametrize(
    ("group_by", "required_facets", "filtered_facets"),
    [
        (
            "mip",
            (),
            {
                "Amon": [
                    {"dataset": "B", "exp": "hist", "short_name": "pr", "mip": "Amon"},
                    {"dataset": "C", "exp": "hist", "short_name": "pr", "mip": "Amon"},
                    {"dataset": "D", "exp": "hist", "mip": "Amon"},
                    {"dataset": "D", "exp": "pi", "short_name": "pr", "mip": "Amon"},
                    {"dataset": "D", "exp": "hist", "short_name": "tas", "mip": "Amon"},
                ],
                "Lmon": [
                    {"dataset": "E", "mip": "Lmon"},
                ],
            },
        ),
        (
            "dataset",
            (
                {"exp": "pi", "short_name": "tas"},
                {"exp": "hist", "short_name": "pr"},
            ),
            {
                "A": [
                    {"dataset": "A", "exp": "pi", "short_name": "tas"},
                    {"dataset": "A", "exp": "hist", "short_name": "pr"},
                ],
                "B": [
                    {"dataset": "B", "exp": "pi", "short_name": "tas"},
                    {"dataset": "B", "exp": "hist", "short_name": "pr", "mip": "Amon"},
                ],
            },
        ),
        (
            "dataset",
            (
                {"short_name": "pr"},
                {"exp": "hist"},
            ),
            {
                "A": [
                    {"dataset": "A", "exp": "pi", "short_name": "tas"},
                    {"dataset": "A", "exp": "hist", "short_name": "pr"},
                ],
                "B": [
                    {"dataset": "B", "exp": "pi", "short_name": "tas"},
                    {"dataset": "B", "exp": "hist", "short_name": "pr", "mip": "Amon"},
                ],
                "C": [
                    {"dataset": "C", "exp": "hist", "short_name": "pr", "mip": "Amon"},
                ],
                "D": [
                    {"dataset": "D", "exp": "hist", "mip": "Amon"},
                    {"dataset": "D", "exp": "pi", "short_name": "pr", "mip": "Amon"},
                    {"dataset": "D", "exp": "hist", "short_name": "tas", "mip": "Amon"},
                ],
            },
        ),
        (
            "short_name",
            ({"exp": "hist"},),
            {
                "tas": [
                    {"dataset": "A", "exp": "pi", "short_name": "tas"},
                    {"dataset": "B", "exp": "pi", "short_name": "tas"},
                    {"dataset": "D", "exp": "hist", "short_name": "tas", "mip": "Amon"},
                ],
                "pr": [
                    {"dataset": "A", "exp": "hist", "short_name": "pr"},
                    {"dataset": "B", "exp": "hist", "short_name": "pr", "mip": "Amon"},
                    {"dataset": "C", "exp": "hist", "short_name": "pr", "mip": "Amon"},
                    {"dataset": "D", "exp": "pi", "short_name": "pr", "mip": "Amon"},
                ],
            },
        ),
        (
            "exp",
            (
                {"dataset": "A", "short_name": "pr"},
                {"dataset": "B", "short_name": "pr", "mip": "Amon"},
                {"dataset": "C", "short_name": "pr", "mip": "Amon"},
            ),
            {
                "hist": [
                    {"dataset": "A", "exp": "hist", "short_name": "pr"},
                    {"dataset": "B", "exp": "hist", "short_name": "pr", "mip": "Amon"},
                    {"dataset": "C", "exp": "hist", "short_name": "pr", "mip": "Amon"},
                    {"dataset": "D", "exp": "hist", "mip": "Amon"},
                    {"dataset": "D", "exp": "hist", "short_name": "tas", "mip": "Amon"},
                ],
            },
        ),
    ],
)
def test_filter_intake_data_sources(
    group_by: str,
    required_facets: tuple[dict[str, str]],
    filtered_facets: dict[str, list[dict[str, str]]],
    generators: tuple[Generator[ESMValToolIntakeDataSource], ...],
) -> None:
    filtered_data_sources = filter_intake_data_sources(
        *generators,
        group_by=group_by,
        required_facets=required_facets,
    )

    assert len(filtered_data_sources) == len(filtered_facets)
    for group, data_sources in filtered_data_sources.items():
        assert group in filtered_facets
        actual_facets = {
            frozenset(f.esmvaltool_dataset_facets.items()) for f in data_sources
        }
        expected_facets = {frozenset(f.items()) for f in filtered_facets[group]}
        assert actual_facets == expected_facets


class _FakeDataset:
    """Record supplementaries instead of searching and loading data."""

    instances: ClassVar[list[_FakeDataset]] = []

    def __init__(self, **facets: Any) -> None:
        self.facets = facets
        self.supplementaries: list[dict[str, Any]] = []
        self.instances.append(self)

    def add_supplementary(self, **facets: Any) -> None:
        self.supplementaries.append(facets)

    def from_files(self) -> Iterator[_FakeDataset]:
        yield self

    def load(self) -> Cube:
        return get_simple_2d_cube(var_name=self.facets["short_name"], units="1")


class _CESM2(ESMValToolIntakeDataSource):
    _information = DataSourceInformation(
        name="CESM2",
        category="CMIP6",
        institute="NCAR",
    )
    esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "historical",
        "ensemble": "r1i1p1f1",
        "grid": "gn",
    }


@pytest.fixture
def fake_dataset(monkeypatch: pytest.MonkeyPatch) -> type[_FakeDataset]:
    monkeypatch.setattr(_FakeDataset, "instances", [])
    monkeypatch.setattr(climateeval.data._esmvaltool_intake, "Dataset", _FakeDataset)
    monkeypatch.setattr(
        climateeval.data._esmvaltool_intake,
        "_setup_esmvaltool_intake_data_source",
        lambda _cache_dir: contextlib.nullcontext(),
    )
    return _FakeDataset


@pytest.mark.parametrize(
    ("var_name", "supplementaries"),
    [("gpp", ["sftlf", "areacella"]), ("tas", [])],
)
def test_download_land_supplementaries(
    tmp_path: Path,
    fake_dataset: type[_FakeDataset],
    var_name: str,
    supplementaries: list[str],
) -> None:
    (path,) = _CESM2().download(Variable(var_name, var_name, "mon"), tmp_path)

    (dataset,) = fake_dataset.instances
    assert [s["short_name"] for s in dataset.supplementaries] == supplementaries
    for supplementary in dataset.supplementaries:
        assert supplementary["mip"] == "fx"
        assert supplementary["exp"] == supplementary["ensemble"] == "*"
    assert path.exists()
