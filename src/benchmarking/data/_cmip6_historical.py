"""Generate data sources from CMIP6 historical simulations."""

from __future__ import annotations

from typing import Any, ClassVar

from benchmarking.data._esmvaltool_intake import ESMValToolIntakeDataSourceGenerator


class CMIP6HistoricalR1I1P1F1(ESMValToolIntakeDataSourceGenerator):
    """Generate data sources from CMIP6 historical simulations (r1i1p1f1)."""

    _ESMVALTOOL_DATASET_FACETS: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "historical",
        "dataset": "*",
        "institute": "*",
        "ensemble": "r1i1p1f1",
        "grid": "*",
        "timerange": "19790101/20141231",
    }
