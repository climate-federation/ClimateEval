from __future__ import annotations

from climateeval.diags.simple import PrecipitationStatistics
from climateeval.suites import Suite


def test_diagnostic_loads() -> None:
    diagnostics = Suite("Tier3_precip_statistics")._get_diagnostics()

    diagnostic = diagnostics["precipitation_statistics"]
    assert isinstance(diagnostic, PrecipitationStatistics)
    ids = {v.id for v in diagnostic._variables}
    assert {"pr_tropics", "pr_tropical_ocean", "pr_tropical_land"} <= ids
