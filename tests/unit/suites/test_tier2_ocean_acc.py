from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest
import yaml

from climateeval.suites import Suite


@pytest.fixture
def suite() -> Suite:
    return Suite("Tier2_ocean_acc")


def test_suite(suite: Suite) -> None:
    assert suite.name == "Tier2_ocean_acc"
    assert (
        suite._suite_definition_file
        == Path(str(resources.files("climateeval.suites"))) / "Tier2_ocean_acc.yml"
    )
    with suite._suite_definition_file.open("r", encoding="utf-8") as file:
        assert suite._suite_definition == yaml.safe_load(file)
    assert suite._diagnostic_kwargs == {}
    assert suite._variable_kwargs == {}
