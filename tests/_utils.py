"""Provide utility functions for tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from climateeval.suites import Suite
from tests import SAMPLE_DATA_ROOT_DIR, SUITES_YML_DIR

if TYPE_CHECKING:
    import ibis
    from pytest_regressions import DataFrameRegressionFixture


def assert_masked_array_allclose(
    array_1: np.ma.MaskedArray,
    array_2: np.ma.MaskedArray,
) -> None:
    np.testing.assert_allclose(array_1, array_2)
    np.testing.assert_equal(array_1.mask, array_2.mask)


def assert_table_equal(
    left_table: ibis.Table,
    right_table: ibis.Table,
    **kwargs: Any,
) -> None:
    print(left_table.to_pandas())  # noqa: T201
    print(right_table.to_pandas())  # noqa: T201

    assert left_table.schema() == right_table.schema()

    # Use pandas to compare the actual data in the tables
    # Note: since Ibis makes no guarantees about row order, we sort the
    # corresponding data frames first and reset the index before comparing them
    left_df = (
        left_table.to_pandas()
        .sort_values(by=list(left_table.columns))
        .reset_index(drop=True)
    )
    right_df = (
        right_table.to_pandas()
        .sort_values(by=list(right_table.columns))
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(left_df, right_df, **kwargs)


def check_table(
    table: ibis.Table,
    dataframe_regression: DataFrameRegressionFixture,
) -> None:
    # Use pandas data frames to compare the actual data in the tables
    # Note: since Ibis makes no guarantees about row order, we sort the
    # corresponding data frame first and reset the index before performing
    # checks
    df = table.to_pandas().sort_values(by=list(table.columns)).reset_index(drop=True)
    dataframe_regression.check(df)


def get_all_table_names(connection: ibis.BaseBackend) -> tuple[tuple[str, str], ...]:
    return tuple(
        (d, t)
        for c in connection.list_catalogs()
        for d in connection.list_databases(catalog=c)
        for t in connection.list_tables(database=f"{c}.{d}")
    )


def get_test_suite(suite_name: str) -> Suite:
    return Suite(
        SUITES_YML_DIR / suite_name,
        diagnostic_kwargs={
            "data_root_dir": SAMPLE_DATA_ROOT_DIR,
            "fail_on_missing_data": True,
            "download_missing_data": False,
        },
    )
