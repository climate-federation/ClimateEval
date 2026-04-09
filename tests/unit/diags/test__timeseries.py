from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.diags import TimeSeries

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking import Variable
    from benchmarking.data import DiagnosticData


def test_tags(
    diagnostic_data_2d: dict[Variable, DiagnosticData],
    data_dir: Path,
) -> None:
    diag = TimeSeries(diagnostic_data_2d, data_dir=data_dir)
    assert diag.tags == ("atmosphere", "pr", "tas", "timeseries")
