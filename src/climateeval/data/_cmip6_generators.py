"""Generate data sources from CMIP6 piControl simulations."""

from __future__ import annotations

from typing import Any, ClassVar

from climateeval.data._esmvaltool_intake import ESMValToolIntakeDataSourceGenerator
from esmvalcore.cmor.table import CMOR_TABLES


class CMIP6DataSourceGenerator(ESMValToolIntakeDataSourceGenerator):
    """Generate data sources from CMIP6 output."""

    def _get_references(self, **facets: Any) -> tuple[str, ...]:
        """Get data source references."""
        activity = CMOR_TABLES["CMIP6"].activities[facets["exp"]][0]
        return (
            f"https://www.wdc-climate.de/ui/cmip6?input=CMIP6.{activity}."
            f"{facets['institute']}.{facets['dataset']}",
        )


class CMIP6Abrupt4xCO2R1I1P1F1(CMIP6DataSourceGenerator):
    """Generate data sources from CMIP6 abrupt-4xCO2 simulations (r1i1p1f1)."""

    ESMVALTOOL_DATASET_FACETS: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "abrupt-4xCO2",
        "dataset": "*",
        "institute": "*",
        "ensemble": "r1i1p1f1",
        "grid": "*",
    }


class CMIP6PiControlR1I1P1F1(CMIP6DataSourceGenerator):
    """Generate data sources from CMIP6 piControl simulations (r1i1p1f1)."""

    ESMVALTOOL_DATASET_FACETS: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "piControl",
        "dataset": "*",
        "institute": "*",
        "ensemble": "r1i1p1f1",
        "grid": "*",
    }


class CMIP6HistoricalR1I1P1F1(CMIP6DataSourceGenerator):
    """Generate data sources from CMIP6 historical simulations (r1i1p1f1)."""

    ESMVALTOOL_DATASET_FACETS: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "historical",
        "dataset": "*",
        "institute": "*",
        "ensemble": "r1i1p1f1",
        "grid": "*",
        "timerange": "19790101/20141231",
    }
