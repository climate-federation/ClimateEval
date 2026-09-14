from __future__ import annotations

from climateeval.suites import Suite


def test_forced_response_suite_builds() -> None:
    suite = Suite("Tier3_forced_response")
    assert suite.name == "Tier3_forced_response"
    assert list(suite._get_diagnostics()) == [
        "land_ocean_warming_ratio",
        "arctic_amplification",
    ]


def test_transport_suite_builds() -> None:
    # Regression guard: MeridionalHeatTransport needs only piControl, so it
    # must live in its own suite (a suite passes one data dict to every
    # diagnostic; mixing it with the 4xco2+picontrol diagnostics would raise).
    suite = Suite("Tier3_transport")
    assert suite.name == "Tier3_transport"
    assert list(suite._get_diagnostics()) == ["meridional_heat_transport"]
