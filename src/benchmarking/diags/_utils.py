"""Provide diagnostic utilities."""

from __future__ import annotations

from dataclasses import dataclass

import ibis
from loguru import logger

logger = logger.opt(colors=True)

DEFAULT_GRID: str = "2x2"
PLEV19_LEVELS: list[float] = [
    100000.0,
    92500.0,
    85000.0,
    70000.0,
    60000.0,
    50000.0,
    40000.0,
    30000.0,
    25000.0,
    20000.0,
    15000.0,
    10000.0,
    7000.0,
    5000.0,
    3000.0,
    2000.0,
    1000.0,
    500.0,
    100.0,
]


@dataclass(frozen=True, kw_only=True)
class DiagnosticOutput:
    """Manage diagnostic output."""

    raw_output: ibis.Table | None
    metrics: ibis.Table | None
    variables: ibis.Table
    data_sources: ibis.Table

    @classmethod
    def union(cls, *diagnostic_output: DiagnosticOutput) -> DiagnosticOutput:
        """Create union of output."""
        # data
        all_raw_output = [o.raw_output for o in diagnostic_output]
        if None in all_raw_output and not all(d is None for d in all_raw_output):
            msg = "Table 'data' must be None for either all or no diagnostics"
            raise ValueError(msg)
        raw_output_union = (
            None if None in all_raw_output else ibis.union(*all_raw_output)
        )

        # metrics
        all_metrics = [o.metrics for o in diagnostic_output]
        if None in all_metrics and not all(d is None for d in all_metrics):
            msg = "Table 'metrics' must be None for either all or no diagnostics"
            raise ValueError(msg)
        metrics_union = None if None in all_metrics else ibis.union(*all_metrics)

        # variables
        variables_union = ibis.union(*(o.variables for o in diagnostic_output))

        # data_sources
        data_sources_union = ibis.union(*(o.data_sources for o in diagnostic_output))

        return cls(
            raw_output=raw_output_union,
            metrics=metrics_union,
            variables=variables_union,
            data_sources=data_sources_union,
        )
