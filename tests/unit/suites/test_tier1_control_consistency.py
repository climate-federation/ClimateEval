from __future__ import annotations

from climateeval.suites import Suite


def test_control_consistency_suite_builds() -> None:
    suite = Suite("Tier1_control_consistency")
    assert suite.name == "Tier1_control_consistency"
    assert list(suite._get_diagnostics()) == [
        "energy_balance",
        "budget_closure",
        "clausius_clapeyron_scaling",
    ]


def test_covariances_suite_builds() -> None:
    # Regression guard: the covariance diagnostics need a historical
    # experiment, so they must live in their own suite (a suite passes one
    # data dict to every diagnostic; mixing them with the piControl
    # diagnostics would raise).
    suite = Suite("Tier3_covariances")
    assert suite.name == "Tier3_covariances"
    assert list(suite._get_diagnostics()) == [
        "clear_sky_lw_feedback",
        "precipitation_buoyancy",
    ]
