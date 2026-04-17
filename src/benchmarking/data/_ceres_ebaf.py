"""Provide CERES-EBAF data."""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarking.data._base import DataSource, DataSourceInformation

if TYPE_CHECKING:
    from pathlib import Path

    from benchmarking import Variable


class CERESEBAF(DataSource):
    """Manage CERES-EBAF data."""

    _INFORMATION = DataSourceInformation(
        name="CERES-EBAF",
        category="observation",
        institute="NASA-LaRC",
        references=("https://doi.org/10.5067/TERRA-AQUA-NOAA20/CERES/EBAF_L3B004.2.1",),
    )

    def download(self, data_root_dir: Path, variable: Variable) -> list[Path]:
        """Download data."""
        return self._esmvaltool_intake_download(
            data_root_dir,
            variable,
            project="obs4MIPs",
            dataset="CERES-EBAF-4-2-1",
            mip=variable.cmip6_table_id,
            short_name=variable.var_name,
            frequency=variable.frequency,
        )
