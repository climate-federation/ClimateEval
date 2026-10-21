from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from climateeval import Variable
from climateeval.data import DataSource, _base
from climateeval.data._base import _paths_within_timerange
from climateeval.data._utils import DataSourceInformation

if TYPE_CHECKING:
    from pathlib import Path


class MockDataSource(DataSource):
    def download(self, _variable: Variable, _data_root_dir: Path | str) -> list[Path]:
        return []


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
            "invalid_attribute",
            AttributeError,
            r"'MockDataSource' object has no attribute 'invalid_attribute'",
        ),
    ],
)
def test_data_source_attribute_access_fail(
    attribute: str,
    exception_cls: type[Exception],
    error_msg: str,
) -> None:
    data_source = MockDataSource()
    with pytest.raises(exception_cls, match=re.escape(error_msg)):
        getattr(data_source, attribute)


class StagedDataSource(DataSource):
    """A data source backed by a staged, time-split variable directory."""

    _information = DataSourceInformation(
        name="EC-Earth3",
        category="CMIP6",
        exp="historical-ssp245",
        variant="r1i1p1f1",
    )

    def download(self, _variable: Variable, _data_root_dir: Path | str) -> list[Path]:
        return []


#: One CMOR file per year, the layout the EC-Earth3 family stages `tos` in.
def _yearly_files(directory: Path, years: range) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for year in years:
        name = f"tos_Omon_EC-Earth3_historical_r1i1p1f1_gn_{year}01-{year}12.nc"
        (directory / name).touch()


@pytest.mark.parametrize(
    ("timerange", "expected"),
    [
        ("20150101/20251231", [f"{y}01-{y}12" for y in range(2015, 2026)]),
        # A window no file covers is not narrowed at all: the caller must
        # still see the files it would have seen, so the "data does not cover
        # the requested range" error downstream is unchanged.
        ("21000101/21101231", [f"{y}01-{y}12" for y in range(2010, 2031)]),
        # No window at all
        ("*", [f"{y}01-{y}12" for y in range(2010, 2031)]),
    ],
)
def test_paths_within_timerange(
    tmp_path: Path,
    timerange: str,
    expected: list[str],
) -> None:
    _yearly_files(tmp_path, range(2010, 2031))
    paths = sorted(tmp_path.glob("*.nc"))
    kept = _paths_within_timerange(paths, timerange)
    assert sorted(p.stem.rsplit("_", 1)[-1] for p in kept) == sorted(expected)


def test_paths_within_timerange_keeps_a_file_without_a_date_token(
    tmp_path: Path,
) -> None:
    """An observational product need not follow the CMOR date convention."""
    _yearly_files(tmp_path, range(2010, 2031))
    (tmp_path / "HadISST_sst.nc").touch()
    kept = _paths_within_timerange(sorted(tmp_path.glob("*.nc")), "20150101/20151231")
    assert [p.name for p in kept] == [
        "HadISST_sst.nc",
        "tos_Omon_EC-Earth3_historical_r1i1p1f1_gn_201501-201512.nc",
    ]


def test_get_cube_only_loads_files_inside_the_timerange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`get_cube` must not load 251 yearly files for an 11-year window."""
    data_source = StagedDataSource()
    _yearly_files(tmp_path / data_source.id / "mon" / "tos", range(1850, 2101))

    loaded: list[list[Path]] = []

    def record(paths: list[Path]) -> list[Path]:
        loaded.append(paths)
        return paths

    monkeypatch.setattr(_base, "load_cubes", record)
    monkeypatch.setattr(_base, "get_prepared_cube", lambda cubes, _variable: cubes)

    variable = Variable("tos", "tos", "mon", timerange="20150101/20251231")
    data_source.get_cube(tmp_path, variable, download_missing_data=False)

    (paths,) = loaded
    assert len(paths) == 11
    assert sorted(p.stem.rsplit("_", 1)[-1] for p in paths) == [
        f"{y}01-{y}12" for y in range(2015, 2026)
    ]
