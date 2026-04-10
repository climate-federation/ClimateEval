"""Provide atmosphere evaluation suite."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmarking import LEVEL_200HPA, LEVEL_800HPA, Variable
from benchmarking.data import CERESEBAF, ERA5, GPCP, DiagnosticData, HadCRUT5
from benchmarking.diags import Map, TimeSeries
from benchmarking.suites._base import Suite

if TYPE_CHECKING:
    from benchmarking.diags import Diagnostic


class AtmosphereMonthly(Suite):
    """Manage atmosphere evaluation suite (monthly data)."""

    def _get_variables_and_data(self) -> dict[Variable, DiagnosticData]:
        """Get data to evaluate."""
        variable_kwargs: dict[str, Any] = {
            "frequency": "mon",
            "timerange": "20000101/20011231",
        }
        variable_kwargs.update(self._variable_kwargs)

        return {
            Variable("lwcre", "lwcre", **variable_kwargs): DiagnosticData(
                reference=CERESEBAF,
            ),
            Variable("pr", "pr", **variable_kwargs): DiagnosticData(
                reference=GPCP,
                other=(ERA5,),
            ),
            Variable("rsut", "rsut", **variable_kwargs): DiagnosticData(
                reference=CERESEBAF,
            ),
            Variable("tas", "tas", **variable_kwargs): DiagnosticData(
                reference=HadCRUT5,
                other=(ERA5,),
            ),
            Variable(
                "ta200",
                "ta",
                vertical_level=LEVEL_200HPA,
                **variable_kwargs,
            ): DiagnosticData(reference=ERA5),
            Variable(
                "ta800",
                "ta",
                vertical_level=LEVEL_800HPA,
                **variable_kwargs,
            ): DiagnosticData(reference=ERA5),
        }

    def _get_diagnostics(self) -> tuple[type[Diagnostic], ...]:
        """Get diagnostics to run."""
        return (Map, TimeSeries)
