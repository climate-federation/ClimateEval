from __future__ import annotations

import pytest
import yaml
from esmvaltool.cmorizers.data.cmorizer import DATASETS_FILE, _Formatter

import climateeval.data
from climateeval.data import ESMValToolCMORizerDataSource

ESMVALTOOL_DATASETS: dict = yaml.safe_load(DATASETS_FILE.read_text())["datasets"]

DATA_SOURCES: list[type[ESMValToolCMORizerDataSource]] = sorted(
    (
        obj
        for obj in vars(climateeval.data).values()
        if isinstance(obj, type)
        and issubclass(obj, ESMValToolCMORizerDataSource)
        and obj is not ESMValToolCMORizerDataSource
    ),
    key=lambda cls: cls.__name__,
)

# ESMValTool ships a CMORizer for these datasets, but no downloader: the raw
# data has to be staged by hand before ``download()`` can format it.
NO_AUTOMATIC_DOWNLOAD = frozenset({"EN4"})

parametrize_data_sources = pytest.mark.parametrize(
    "data_source_cls",
    DATA_SOURCES,
    ids=[cls.__name__ for cls in DATA_SOURCES],
)


@parametrize_data_sources
def test_data_source_is_known_to_esmvaltool(
    data_source_cls: type[ESMValToolCMORizerDataSource],
) -> None:
    """``name`` must match an ESMValTool CMORizer dataset."""
    assert data_source_cls().name in ESMVALTOOL_DATASETS


@parametrize_data_sources
def test_data_source_tier_matches_esmvaltool(
    data_source_cls: type[ESMValToolCMORizerDataSource],
) -> None:
    """``_tier`` selects the output directory, so it must match ESMValTool."""
    name = data_source_cls().name
    assert data_source_cls._tier == ESMVALTOOL_DATASETS[name]["tier"]


@parametrize_data_sources
def test_data_source_can_be_downloaded_automatically(
    data_source_cls: type[ESMValToolCMORizerDataSource],
) -> None:
    """``download()`` relies on an ESMValTool downloader for the dataset."""
    name = data_source_cls().name
    if name in NO_AUTOMATIC_DOWNLOAD:
        pytest.skip(f"ESMValTool has no downloader for {name}")
    assert _Formatter.has_downloader(name)
