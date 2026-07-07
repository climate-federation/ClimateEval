"""Provide CERES-EBAF data."""

from __future__ import annotations

from typing import Any, ClassVar

from climateeval.data._esmvaltool_intake import ESMValToolIntakeDataSource
from climateeval.data._utils import DataSourceInformation


class CERESEBAF(ESMValToolIntakeDataSource):
    """Manage CERES-EBAF data."""

    _information = DataSourceInformation(
        name="CERES-EBAF",
        category="observation",
        institute="NASA-LaRC",
        references=("https://doi.org/10.5067/TERRA-AQUA-NOAA20/CERES/EBAF_L3B004.2.1",),
    )

    esmvaltool_dataset_facets: ClassVar[dict[str, Any]] = {
        "project": "obs4MIPs",
        "dataset": "CERES-EBAF-4-2-1",
    }
