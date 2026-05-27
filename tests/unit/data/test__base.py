from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from climateeval.data import DataSource

if TYPE_CHECKING:
    from pathlib import Path

    from climateeval import Variable


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
            r"'_INFORMATION'",
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
