from __future__ import annotations

from importlib import resources
import re
from typing import TYPE_CHECKING

import pytest
import yaml

from benchmarking import Variable
from benchmarking.data import DiagnosticData, HadCRUT5
from benchmarking.diags import TimeSeries
from benchmarking.suites import Suite

if TYPE_CHECKING:
    from pathlib import Path

    from iris.cube import Cube


@pytest.fixture
def suite(data_root_dir: Path) -> Suite:
    return Suite("Tier2", diagnostic_kwargs={"data_root_dir": data_root_dir})


def test_suite(suite: Suite, data_root_dir: Path) -> None:
    assert suite.name == "Tier2"
    assert (
        suite._suite_definition_file
        == Path(str(resources.files("benchmarking.suites"))) / "Tier2.yml"
    )
    with suite._suite_definition_file.open("r", encoding="utf-8") as file:
        assert suite._suite_definition == yaml.safe_load(file)
    assert suite._diagnostic_kwargs == {"data_root_dir": data_root_dir}
    assert suite._variable_kwargs == {}
