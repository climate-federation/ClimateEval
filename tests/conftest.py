from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from benchmarking import Variable
from benchmarking.data import GPCP, DiagnosticData, HadCRUT5

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def diagnostic_data_2d() -> dict[Variable, DiagnosticData]:
    return {
        Variable("my_tas", "tas", "mon"): DiagnosticData(reference=HadCRUT5),
        Variable("my_pr", "pr", "mon"): DiagnosticData(reference=GPCP),
    }
