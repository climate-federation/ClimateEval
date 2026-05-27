"""Provide utility functions for tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import ibis


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


def get_all_table_names(connection: ibis.BaseBackend) -> tuple[tuple[str, str], ...]:
    return tuple(
        (d, t)
        for c in connection.list_catalogs()
        for d in connection.list_databases(catalog=c)
        for t in connection.list_tables(database=f"{c}.{d}")
    )
