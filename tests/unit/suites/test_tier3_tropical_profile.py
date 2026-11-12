from __future__ import annotations

from climateeval.diags.complex import InversionStrength, MSUTemperature
from climateeval.suites import Suite


def test_diagnostics_load() -> None:
    diagnostics = Suite("Tier3_tropical_profile")._get_diagnostics()

    assert isinstance(diagnostics["msu_temperature"], MSUTemperature)
    assert isinstance(diagnostics["inversion_strength"], InversionStrength)
    assert diagnostics["msu_temperature"]._required_data_keys == ("amip",)
    assert diagnostics["inversion_strength"]._landsea_mask == "sea_only"
