from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from benchmarking._config import Config
from benchmarking._data.gpcp_sg import GPCPSG
from benchmarking._data.hadcrut5 import HadCRUT5

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking._data import Data


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(data_dir=tmp_path / "data")


@pytest.fixture
def reference_data() -> dict[str, type[Data]]:
    return {
        "tas": HadCRUT5,
        "pr": GPCPSG,
    }
