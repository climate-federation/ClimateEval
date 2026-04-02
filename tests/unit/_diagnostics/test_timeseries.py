from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking._diagnostics.timeseries import TimeSeries

if TYPE_CHECKING:
    from benchmarking._config import Config
    from benchmarking._data import Data


def test_tags(reference_data: dict[str, type[Data]], config: Config) -> None:
    diag = TimeSeries(
        variables=["tas", "pr"],
        reference_data=reference_data,
        other_data={},
        config=config,
    )
    assert diag.tags == ("atmosphere", "pr", "tas", "timeseries")
