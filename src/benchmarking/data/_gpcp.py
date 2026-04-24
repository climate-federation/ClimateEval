"""Provide GPCP data."""

from __future__ import annotations

from typing import Any, ClassVar

from benchmarking.data._esmvaltool_intake import ESMValToolIntakeDataSource
from benchmarking.data._utils import DataSourceInformation


class GPCP(ESMValToolIntakeDataSource):
    """Manage GPCP data."""

    _INFORMATION = DataSourceInformation(
        name="GPCP",
        category="observation",
        institute="NASA-GSFC",
        references=(
            "https://doi.org/10.5067/MEASURES/GPCP/DATA306",
            "https://doi.org/10.5067/MEASURES/GPCP/DATA307",
        ),
    )

    ESMVALTOOL_DATASET_FACETS: ClassVar[dict[str, Any]] = {
        "project": "obs4MIPs",
        "dataset": "GPCP-3-3",
    }
