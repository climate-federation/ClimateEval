"""Generate data sources from CMIP6 piControl simulations."""

from __future__ import annotations

from typing import Any, ClassVar

from benchmarking.data._esmvaltool_intake import ESMValToolIntakeDataSourceGenerator


class CMIP6PiControlR1I1P1F1(ESMValToolIntakeDataSourceGenerator):
    """Generate data sources from CMIP6 piControl simulations (r1i1p1f1)."""

    ESMVALTOOL_DATASET_FACETS: ClassVar[dict[str, Any]] = {
        "project": "CMIP6",
        "exp": "piControl",
        "dataset": "*",
        "institute": "*",
        "ensemble": "r1i1p1f1",
        "grid": "*",
    }
