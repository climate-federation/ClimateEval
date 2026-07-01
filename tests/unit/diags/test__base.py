from __future__ import annotations

from typing import TYPE_CHECKING

import ibis
import pandas as pd

from climateeval.diags._base import Diagnostic
from tests._utils import assert_table_equal

if TYPE_CHECKING:
    from climateeval.data import DataSourceInformation
    from climateeval.diags._base import DiagnosticOutput


class _DummyDiagnostic(Diagnostic):
    """Minimal concrete diagnostic to exercise base-class helpers."""

    @classmethod
    def from_suite_dict(cls, *_args: object, **_kwargs: object) -> _DummyDiagnostic:
        raise NotImplementedError

    def get_output(
        self,
        _data: object,
        _data_information: DataSourceInformation,
    ) -> DiagnosticOutput:
        raise NotImplementedError


def _remove(
    table: ibis.Table,
    unique_columns: list[str],
    variable_columns: list[str],
) -> ibis.Table:
    diagnostic = _DummyDiagnostic.__new__(_DummyDiagnostic)
    return diagnostic._remove_superfluous_rows(
        table,
        unique_columns=unique_columns,
        variable_columns=variable_columns,
    )


def test_remove_superfluous_rows_keeps_data() -> None:
    table = ibis.memtable(
        pd.DataFrame(
            {
                "data_id": pd.Series(["m", "m"], dtype="str"),
                "data_type": pd.Series(
                    ["to_benchmark", "to_benchmark"],
                    dtype="str",
                ),
                "latitude": pd.Series([0.0, 1.0], dtype="float64"),
                "tos": pd.Series([10.0, 11.0], dtype="float64"),
            },
        ),
    )

    result = _remove(table, ["data_id", "data_type", "latitude"], ["tos"])

    assert_table_equal(result, table)


def test_remove_superfluous_rows_backfills_missing_variable() -> None:
    """A variable with no data at all must yield an all-NaN column, not crash.

    This happens when the benchmarked dataset lacks a suite variable and its
    reference is unavailable, so the variable never makes it into the combined
    table.
    """
    table = ibis.memtable(
        pd.DataFrame(
            {
                "data_id": pd.Series(["m", "m"], dtype="str"),
                "data_type": pd.Series(
                    ["to_benchmark", "to_benchmark"],
                    dtype="str",
                ),
                "latitude": pd.Series([0.0, 1.0], dtype="float64"),
                "tos": pd.Series([10.0, 11.0], dtype="float64"),
            },
        ),
    )

    # 'chl' is requested but entirely absent from the table.
    result = _remove(table, ["data_id", "data_type", "latitude"], ["chl", "tos"])
    result_df = result.to_pandas().sort_values("latitude").reset_index(drop=True)

    assert "chl" in result_df.columns
    assert result_df["chl"].isna().all()
    assert list(result_df["tos"]) == [10.0, 11.0]
