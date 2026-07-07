"""Provide GPCP data."""

from __future__ import annotations

from typing import Any, ClassVar

from climateeval.data._esmvaltool_intake import ESMValToolIntakeDataSource
from climateeval.data._utils import DataSourceInformation


class GPCP(ESMValToolIntakeDataSource):
    """Manage GPCP data."""

    _information = DataSourceInformation(
        name="GPCP",
        category="observation",
        institute="NASA-GSFC",
        references=(
            "https://doi.org/10.5067/MEASURES/GPCP/DATA306",
            "https://doi.org/10.5067/MEASURES/GPCP/DATA307",
        ),
    )

    esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
        "project": "obs4MIPs",
        "dataset": "GPCP-3-3",
    }
