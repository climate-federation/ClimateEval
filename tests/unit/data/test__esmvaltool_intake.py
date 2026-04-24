from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from benchmarking.data import (
    ESMValToolIntakeDataSource,
)
from benchmarking.data._esmvaltool_intake import filter_intake_data_sources

if TYPE_CHECKING:
    from collections.abc import Generator


@dataclass(frozen=True)
class MockESMValToolIntakeDataSource(ESMValToolIntakeDataSource):
    ESMVALTOOL_DATASET_FACETS: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def generators() -> tuple[Generator[MockESMValToolIntakeDataSource], ...]:
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
        MockESMValToolIntakeDataSource(ESMVALTOOL_DATASET_FACETS=f)
        for f in data_source_facets_1
    )
    generator_2 = (
        MockESMValToolIntakeDataSource(ESMVALTOOL_DATASET_FACETS=f)
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
    generators: tuple[Generator[MockESMValToolIntakeDataSource], ...],
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
            frozenset(f.ESMVALTOOL_DATASET_FACETS.items()) for f in data_sources
        }
        expected_facets = {frozenset(f.items()) for f in filtered_facets[group]}
        assert actual_facets == expected_facets
