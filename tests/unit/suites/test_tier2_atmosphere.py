from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest
import yaml

from benchmarking.suites import Suite


@pytest.fixture
def suite(data_root_dir: Path) -> Suite:
    return Suite("Tier2_atmosphere", diagnostic_kwargs={"data_root_dir": data_root_dir})


def test_suite(suite: Suite, data_root_dir: Path) -> None:
    assert suite.name == "Tier2_atmosphere"
    assert (
        suite._suite_definition_file
        == Path(str(resources.files("benchmarking.suites"))) / "Tier2_atmosphere.yml"
    )
    with suite._suite_definition_file.open("r", encoding="utf-8") as file:
        assert suite._suite_definition == yaml.safe_load(file)
    assert suite._diagnostic_kwargs == {"data_root_dir": data_root_dir}
    assert suite._variable_kwargs == {}
