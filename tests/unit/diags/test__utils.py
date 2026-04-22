from __future__ import annotations

import ibis
import numpy as np
import pandas as pd
import pytest

from benchmarking.diags._utils import union


@pytest.fixture
def table_tas() -> ibis.Table:
    return ibis.memtable(
        pd.DataFrame(
            {
                "data_id": pd.Series(["A", "B"], dtype="str"),
                "data_type": pd.Series(["to_benchmark", "reference"], dtype="str"),
                "latitude": pd.Series([0.0, 1.0], dtype="float32"),
                "tas": pd.Series([10.0, 20.0], dtype="float64"),
            },
        ),
    )


@pytest.fixture
def table_pr() -> ibis.Table:
    return ibis.memtable(
        pd.DataFrame(
            {
                "data_id": pd.Series(["B", "C"], dtype="str"),
                "data_type": pd.Series(["reference", "other"], dtype="str"),
                "latitude": pd.Series([1, 2], dtype="int64"),
                "longitude": pd.Series([100.0, 200.0], dtype="float64"),
                "pr": pd.Series([30.0, 40.0], dtype="float32"),
            },
        ),
    )


def test_union(table_tas: ibis.Table, table_pr: ibis.Table) -> None:
    table = union(table_tas, table_pr)

    expected_schema = ibis.schema(
        {
            "data_id": "str",
            "data_type": "str",
            "latitude": "float32",
            "tas": "float64",
            "longitude": "float64",
            "pr": "float32",
        },
    )

    assert table.schema() == expected_schema

    expected_table = ibis.memtable(
        pd.DataFrame(
            {
                "data_id": pd.Series(["A", "B", "B", "C"], dtype="str"),
                "data_type": pd.Series(
                    ["to_benchmark", "reference", "reference", "other"],
                    dtype="str",
                ),
                "latitude": pd.Series([0.0, 1.0, 1.0, 2.0], dtype="float32"),
                "tas": pd.Series([10.0, 20.0, np.nan, np.nan], dtype="float64"),
                "longitude": pd.Series([np.nan, np.nan, 100, 200], dtype="float64"),
                "pr": pd.Series([np.nan, np.nan, 30.0, 40.0], dtype="float32"),
            },
        ),
    )
    assert table.to_pandas().equals(expected_table.to_pandas())
