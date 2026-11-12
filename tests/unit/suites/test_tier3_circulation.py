from __future__ import annotations

import pytest

from climateeval.diags.simple import (
    EddyDrivenJetLatitude,
    HadleyCellEdge,
    WalkerCirculationStrength,
)
from climateeval.suites import Suite


@pytest.fixture
def suite() -> Suite:
    return Suite("Tier3_circulation")


def test_diagnostics_load(suite: Suite) -> None:
    diagnostics = suite._get_diagnostics()

    assert isinstance(diagnostics["HadleyCellEdge"], HadleyCellEdge)
    assert isinstance(diagnostics["EddyDrivenJetLatitude"], EddyDrivenJetLatitude)
    assert isinstance(
        diagnostics["WalkerCirculationStrength"],
        WalkerCirculationStrength,
    )


def test_hemisphere_options_reach_the_variables(suite: Suite) -> None:
    diagnostic = suite._get_diagnostics()["HadleyCellEdge"]

    options = {v.id: v.diagnostic_kwargs for v in diagnostic._variables}

    assert options == {
        "hadley_edge_nh": {"hemisphere": "north"},
        "hadley_edge_sh": {"hemisphere": "south"},
    }
