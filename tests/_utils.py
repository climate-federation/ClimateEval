"""Provide utility functions for tests."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    import ibis


def assert_table_equal(
    left_table: ibis.Table,
    right_table: ibis.Table,
    **kwargs: Any,
) -> None:
    assert left_table.schema() == right_table.schema()
    kwargs.setdefault("check_like", True)
    pd.testing.assert_frame_equal(
        left_table.to_pandas(), right_table.to_pandas(), **kwargs
    )
