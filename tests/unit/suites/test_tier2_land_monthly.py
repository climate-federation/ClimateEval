from __future__ import annotations

from importlib import resources
from pathlib import Path

import pytest
import yaml

from climateeval.suites import Suite


@pytest.fixture
def suite() -> Suite:
    return Suite("Tier2_land_monthly")


def test_suite(suite: Suite) -> None:
    assert suite.name == "Tier2_land_monthly"
    assert (
        suite._suite_definition_file
        == Path(str(resources.files("climateeval.suites"))) / "Tier2_land_monthly.yml"
    )
    with suite._suite_definition_file.open("r", encoding="utf-8") as file:
        assert suite._suite_definition == yaml.safe_load(file)
    assert suite._diagnostic_kwargs == {}
    assert suite._variable_kwargs == {}


def test_suite_diagnostics(suite: Suite) -> None:
    diagnostics = {d["name"]: d["diagnostic"] for d in suite._suite_definition}
    assert diagnostics == {
        "annual_cycle": "climateeval.diags.simple.LandTotalAnnualCycle",
        "performance_metrics": (
            "climateeval.diags.simple.LandAnnualCycleSpatialDistribution"
        ),
        "annual_mean_timeseries": "climateeval.diags.simple.LandTotalTimeSeries",
        "map": "climateeval.diags.simple.LandMap",
    }


def test_suite_variables(suite: Suite) -> None:
    for diagnostic in suite._suite_definition:
        assert diagnostic["variables"] == [
            {
                "id": "gpp",
                "var_name": "gpp",
                "frequency": "mon",
                "units": "g m-2 day-1",
                "reference_data": "climateeval.data.FLUXCOM",
                "other_data": ["climateeval.data.CMIP6HistoricalR1I1P1F1"],
            },
        ]
