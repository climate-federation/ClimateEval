"""Provide GPCP data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.data._base import DataSource, DataSourceInformation

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking import Variable


class GPCP(DataSource):
    """Manage GPCP data."""

    _INFORMATION = DataSourceInformation(
        name="GPCP",
        type="observation",
        institute="NASA-GSFC",
        exp="",
        variant="",
        references=(
            "https://doi.org/10.5067/MEASURES/GPCP/DATA306",
            "https://doi.org/10.5067/MEASURES/GPCP/DATA307",
        ),
    )

    def download(self, data_root_dir: Path, variable: Variable) -> list[Path]:
        """Download data."""
        return self._esmvaltool_intake_download(
            data_root_dir,
            variable,
            project="obs4MIPs",
            dataset="GPCP-3-3",
            mip=variable.cmip6_table_id,
            short_name=variable.var_name,
            frequency=variable.frequency,
        )
