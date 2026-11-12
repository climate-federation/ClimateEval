from __future__ import annotations

import re
from importlib import resources
from pathlib import Path

import pytest

from climateeval.suites import Suite

SUITES_YML_DIR: Path = Path(str(resources.files("tests.unit.suites"))) / "suites_yml"


def test_empty_suite() -> None:
    suite = Suite(SUITES_YML_DIR / "empty_suite.yml")
    assert suite.name == "empty_suite"
    assert suite._suite_definition_file == SUITES_YML_DIR / "empty_suite.yml"
    assert suite._suite_definition == []
    assert suite._diagnostic_kwargs == {}
    assert suite._variable_kwargs == {}
    assert suite._get_diagnostics() == {}


def test_missing_name_fail() -> None:
    suite = Suite(SUITES_YML_DIR / "missing_name.yml")
    assert suite.name == "missing_name"
    assert suite._suite_definition_file == SUITES_YML_DIR / "missing_name.yml"
    assert suite._diagnostic_kwargs == {}
    assert suite._variable_kwargs == {}
    msg = (
        r"Expected key 'name' in diagnostic located at index 0 in suite "
        r"'missing_name'"
    )
    with pytest.raises(KeyError, match=re.escape(msg)):
        suite._get_diagnostics()


def test_missing_diagnostic_fail() -> None:
    suite = Suite(SUITES_YML_DIR / "missing_diagnostic.yml")
    assert suite.name == "missing_diagnostic"
    assert suite._suite_definition_file == SUITES_YML_DIR / "missing_diagnostic.yml"
    assert suite._diagnostic_kwargs == {}
    assert suite._variable_kwargs == {}
    msg = (
        r"Expected key 'diagnostic' in diagnostic located at index 0 in suite "
        r"'missing_diagnostic'"
    )
    with pytest.raises(KeyError, match=re.escape(msg)):
        suite._get_diagnostics()


def test_missing_variables_fail() -> None:
    suite = Suite(SUITES_YML_DIR / "missing_variables.yml")
    assert suite.name == "missing_variables"
    assert suite._suite_definition_file == SUITES_YML_DIR / "missing_variables.yml"
    assert suite._diagnostic_kwargs == {}
    assert suite._variable_kwargs == {}
    msg = (
        r"Expected key 'variables' in suite definition of simple diagnostic "
        r"'annual_cycle'"
    )
    with pytest.raises(KeyError, match=re.escape(msg)):
        suite._get_diagnostics()


def test_missing_additional_diagnostic_kwargs_fail() -> None:
    suite = Suite(SUITES_YML_DIR / "missing_additional_diagnostic_kwargs.yml")
    assert suite.name == "missing_additional_diagnostic_kwargs"
    assert suite._suite_definition_file == (
        SUITES_YML_DIR / "missing_additional_diagnostic_kwargs.yml"
    )
    assert suite._diagnostic_kwargs == {}
    assert suite._variable_kwargs == {}
    msg = (
        r"Expected key 'additional_diagnostic_kwargs' in suite definition of "
        r"complex diagnostic 'ecs'"
    )
    with pytest.raises(KeyError, match=re.escape(msg)):
        suite._get_diagnostics()


def test_invalid_diagnostic_fail() -> None:
    suite = Suite(SUITES_YML_DIR / "invalid_diagnostic.yml")
    assert suite.name == "invalid_diagnostic"
    assert suite._suite_definition_file == SUITES_YML_DIR / "invalid_diagnostic.yml"
    assert suite._diagnostic_kwargs == {}
    assert suite._variable_kwargs == {}
    msg = (
        r"Diagnostic 'invalid_diagnostic' is not a valid diagnostic; should be "
        r"a subclass of climateeval.diags.simple.SimpleDiagnostic or "
        r"climateeval.diags.copmlex.ComplexDiagnostic"
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        suite._get_diagnostics()


def test_non_unique_diagnostic_names_fail() -> None:
    suite = Suite(SUITES_YML_DIR / "non_unique_diagnostic_names.yml")
    assert suite.name == "non_unique_diagnostic_names"
    assert suite._suite_definition_file == (
        SUITES_YML_DIR / "non_unique_diagnostic_names.yml"
    )
    assert suite._diagnostic_kwargs == {}
    assert suite._variable_kwargs == {}
    msg = (
        r"Diagnostic names must be unique for suite 'non_unique_diagnostic_names', "
        r"got multiple diagnostics named 'annual_cycle'"
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        suite._get_diagnostics()


def test_grid_in_suite_entries() -> None:
    diagnostics = Suite(SUITES_YML_DIR / "grid.yml")._get_diagnostics()
    assert diagnostics["map_default"]._target_grid == "2x2"
    assert diagnostics["map_fine"]._target_grid == "1x1"
    assert diagnostics["sea_ice"]._target_grid == "1x1"  # sea-ice default
    assert diagnostics["ecs"]._target_grid == "2.5x2.5"
    assert diagnostics["ecs_default"]._target_grid == "2x2"


def test_grid_through_suite_diagnostic_kwargs() -> None:
    suite = Suite(SUITES_YML_DIR / "grid.yml", diagnostic_kwargs={"grid": "1.5x1.5"})
    diagnostics = suite._get_diagnostics()
    # Suite-level kwargs override the entries, like all diagnostic kwargs
    assert {d._target_grid for d in diagnostics.values()} == {"1.5x1.5"}


def test_diagnostic_kwargs_grid_reaches_every_shipped_suite_diagnostic() -> None:
    from climateeval.suites import resolve_suite_definition_file  # noqa: PLC0415

    # No diagnostic of the shipped suites rejects a suite-wide 'grid'
    for name in ("Tier2_sea_ice_monthly", "Tier1_control_consistency"):
        suite = Suite(
            resolve_suite_definition_file(name),
            diagnostic_kwargs={"grid": "1x1"},
        )
        assert {d._target_grid for d in suite._get_diagnostics().values()} == {"1x1"}
