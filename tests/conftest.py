from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from benchmarking import Config
from benchmarking.data import GPCPSG, HadCRUT5

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking.data import DataSource


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data")


@pytest.fixture
def reference_data() -> dict[str, type[DataSource]]:
    return {
        "tas": HadCRUT5,
        "pr": GPCPSG,
    }
