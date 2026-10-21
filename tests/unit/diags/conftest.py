"""Fixtures shared by the diagnostic unit tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from climateeval.data._esmvaltool_intake import ESMValToolIntakeDataSourceGenerator

if TYPE_CHECKING:
    from collections.abc import Generator

    from climateeval import Variable
    from climateeval.data._esmvaltool_intake import ESMValToolIntakeDataSource


@pytest.fixture(autouse=True)
def no_intake_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the diagnostics from discovering CMIP6 data on the test machine.

    The CMIP6 comparison ensembles of the complex diagnostics are discovered
    through the ambient ESMValTool configuration (``~/.config/esmvaltool``).
    On a machine with a CMIP6 archive, a unit test would therefore find real
    models and then fail to load them from its temporary data directory - or
    silently exercise a different code path than on a machine without one.
    Tests that want an ensemble substitute their own generator.
    """

    def generate(
        _self: ESMValToolIntakeDataSourceGenerator,
        _variable: Variable,
    ) -> Generator[ESMValToolIntakeDataSource]:
        yield from ()

    monkeypatch.setattr(ESMValToolIntakeDataSourceGenerator, "generate", generate)
