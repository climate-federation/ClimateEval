from __future__ import annotations

from climateeval.suites import Suite


def test_daily_dynamics_suite_builds() -> None:
    suite = Suite("Tier3_daily_dynamics")
    assert suite.name == "Tier3_daily_dynamics"
    assert list(suite._get_diagnostics()) == [
        "geostrophic_balance",
        "mjo_east_west_power_ratio",
    ]


def test_coupled_variability_suite_builds() -> None:
    suite = Suite("Tier3_coupled_variability")
    assert suite.name == "Tier3_coupled_variability"
    assert list(suite._get_diagnostics()) == [
        "bjerknes_compensation",
        "enso_teleconnection_patterns",
    ]


def test_itcz_efe_suite_builds() -> None:
    # Regression guard: the three suites are split by experiment (`day`,
    # `picontrol`, `historical`) because a suite passes one data dict to every
    # diagnostic it contains.
    suite = Suite("Tier3_itcz_efe")
    assert suite.name == "Tier3_itcz_efe"
    assert list(suite._get_diagnostics()) == ["itcz_energy_flux_equator"]
