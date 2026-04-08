from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.diags import TimeSeries

if TYPE_CHECKING:
    from benchmarking import Config
    from benchmarking.data import DataSource


def test_tags(reference_data: dict[str, type[DataSource]], config: Config) -> None:
    diag = TimeSeries(
        reference_data=reference_data,
        other_data={},
        config=config,
    )
    assert diag.tags == ("atmosphere", "pr", "tas", "timeseries")
