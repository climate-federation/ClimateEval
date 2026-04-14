"""Provide Tier 2 evaluation suite."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from benchmarking import LEVEL_200HPA, LEVEL_800HPA, Variable
from benchmarking.data import CERESEBAF, ERA5, GPCP, DiagnosticData, HadCRUT5
from benchmarking.diags import Map, TimeSeries, ZonalLine, ZonalProfileAirPressure
from benchmarking.suites._base import Suite

if TYPE_CHECKING:
    from benchmarking.diags import Diagnostic


class Tier2Atmosphere(Suite):
    """Manage Tier 2 atmosphere evaluation suite."""

    def _get_diagnostics(self) -> tuple[Diagnostic, ...]:
        """Get diagnostics to run."""
        variable_kwargs: dict[str, Any] = {
            "frequency": "mon",
            "timerange": "20000101/20011231",
        }
        variable_kwargs.update(self._variable_kwargs)

        variables_and_data = {
            Variable("lwcre", "lwcre", **variable_kwargs): DiagnosticData(
                reference=CERESEBAF,
            ),
            Variable(
                "pr",
                "pr",
                "mon",
                timerange="20040101/20061231",
                units="mm day-1",
            ): DiagnosticData(
                reference=GPCP,
                other=(ERA5,),
            ),
            Variable("rsut", "rsut", **variable_kwargs): DiagnosticData(
                reference=CERESEBAF,
            ),
            Variable(
                "tas",
                "tas",
                "mon",
                timerange="20000101/20051231",
            ): DiagnosticData(
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
            Variable(
                "ta",
                "ta",
                **variable_kwargs,
            ): DiagnosticData(reference=ERA5),
        }

        return (
            TimeSeries(variables_and_data, **self._diagnostic_kwargs),
            Map(variables_and_data, **self._diagnostic_kwargs),
        )
