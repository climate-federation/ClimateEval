from __future__ import annotations

import pytest

from climateeval._config import setup_esmvaltool_config_and_logging
from climateeval.data import (
    CMIP6Amip4xCO2R1I1P1F1,
    CMIP6AmipR1I1P1F1,
    CMIP6DataSourceGenerator,
    CMIP6HistAerR1I1P1F1,
    CMIP6LgmR1I1P1F1,
    CMIP6Lig127kR1I1P1F1,
    CMIP6MidHoloceneR1I1P1F1,
)

# Generator, experiment and the activity the CMIP6 controlled vocabulary
# assigns to that experiment.
_GENERATORS = [
    (CMIP6HistAerR1I1P1F1, "hist-aer", "DAMIP"),
    (CMIP6AmipR1I1P1F1, "amip", "CMIP"),
    (CMIP6Amip4xCO2R1I1P1F1, "amip-4xCO2", "CFMIP"),
    (CMIP6LgmR1I1P1F1, "lgm", "PMIP"),
    (CMIP6MidHoloceneR1I1P1F1, "midHolocene", "PMIP"),
    (CMIP6Lig127kR1I1P1F1, "lig127k", "PMIP"),
]


@pytest.mark.parametrize(("generator_cls", "exp", "activity"), _GENERATORS)
def test_facets(
    generator_cls: type[CMIP6DataSourceGenerator],
    exp: str,
    activity: str,
) -> None:
    facets = generator_cls.esmvaltool_dataset_facets

    assert facets["project"] == "CMIP6"
    assert facets["exp"] == exp
    assert facets["dataset"] == "*"
    assert facets["institute"] == "*"
    assert facets["ensemble"] == "r1i1p1f1"
    assert facets["grid"] == "*"

    # These experiments have no common time axis (the paleoclimate slices) or
    # are used in full (hist-aer, amip), so no time range is preselected
    assert "timerange" not in facets


@pytest.mark.parametrize(("generator_cls", "exp", "activity"), _GENERATORS)
def test_get_references(
    generator_cls: type[CMIP6DataSourceGenerator],
    exp: str,
    activity: str,
) -> None:
    generator = generator_cls()

    with setup_esmvaltool_config_and_logging():
        references = generator._get_references(
            exp=exp,
            institute="MPI-M",
            dataset="MPI-ESM1-2-LR",
        )

    assert references == (
        f"https://www.wdc-climate.de/ui/cmip6?input=CMIP6.{activity}.MPI-M."
        f"MPI-ESM1-2-LR",
    )
