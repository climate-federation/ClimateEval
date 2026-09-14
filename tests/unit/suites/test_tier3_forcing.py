from __future__ import annotations

from climateeval.suites import Suite


def test_amip_erf_suite_builds() -> None:
    suite = Suite("Tier3_amip_erf")
    assert suite.name == "Tier3_amip_erf"
    assert list(suite._get_diagnostics()) == ["amip_4xco2_erf"]


def test_patch_feedback_suite_builds() -> None:
    suite = Suite("Tier3_patch_feedback")
    assert suite.name == "Tier3_patch_feedback"
    assert list(suite._get_diagnostics()) == ["gfmip_patch_feedback"]


def test_aerosol_erf_suite_builds() -> None:
    # Regression guard: the three forcing diagnostics need three different
    # sets of experiments, and a suite passes one data dict to every
    # diagnostic it contains, so each needs its own suite.
    suite = Suite("Tier3_aerosol_erf")
    assert suite.name == "Tier3_aerosol_erf"
    assert list(suite._get_diagnostics()) == ["aerosol_erf"]
